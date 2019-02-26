#include "serialization.h"
#include "pool.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/metainfo.pb.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/quantized_pool/detail.h>
#include <catboost/libs/quantization_schema/detail.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <util/digest/numeric.h>
#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/array_size.h>
#include <util/generic/deque.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/length.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>

using NCB::NIdl::TPoolMetainfo;
using NCB::NIdl::TPoolQuantizationSchema;
using NCB::NQuantizationDetail::GetFakeDocIdColumnIndex;
using NCB::NQuantizationDetail::GetFakeGroupIdColumnIndex;
using NCB::NQuantizationDetail::GetFakeSubgroupIdColumnIndex;

static const char Magic[] = "CatboostQuantizedPool";
static const size_t MagicSize = Y_ARRAY_SIZE(Magic);  // yes, with terminating zero
static const char MagicEnd[] = "CatboostQuantizedPoolEnd";
static const size_t MagicEndSize = Y_ARRAY_SIZE(MagicEnd);  // yes, with terminating zero
static const ui32 Version = 1;
static const ui32 VersionHash = IntHash(Version);

template <typename T>
static TDeque<ui32> CollectAndSortKeys(const T& m) {
    TDeque<ui32> res;
    for (const auto kv : m) {
        Y_ASSERT(kv.first <= static_cast<size_t>(Max<ui32>()));
        res.push_back(kv.first);
    }
    Sort(res);
    return res;
}

template <typename T>
static void WriteLittleEndian(const T value, IOutputStream* const output) {
    const auto le = HostToLittle(value);
    output->Write(&le, sizeof(le));
}

template <typename T>
static void ReadLittleEndian(T* const value, IInputStream* const input) {
    T le;
    const auto bytesRead = input->Load(&le, sizeof(le));
    CB_ENSURE(bytesRead == sizeof(le));
    *value = LittleToHost(le);
}

template <typename T>
static void ReadLittleEndian(T* const value, ui8 const** const input) {
    *value = LittleToHost(ReadUnaligned<T>(*input));
    *input += sizeof(T);
}

static void AddPadding(const ui64 alignment, TCountingOutput* const output) {
    if (output->Counter() % alignment == 0) {
        return;
    }

    const auto bytesToWrite = alignment - output->Counter() % alignment;
    for (ui64 i = 0; i < bytesToWrite; ++i) {
        output->Write('\0');
    }
}

static void SkipPadding(const ui64 alignment, TCountingInput* const input) {
    if (input->Counter() % alignment == 0) {
        return;
    }

    const auto bytesToSkip = alignment - input->Counter() % alignment;
    const auto bytesSkipped = input->Skip(bytesToSkip);
    CB_ENSURE(bytesToSkip == bytesSkipped);
}

namespace {
    struct TChunkInfo {
        ui32 Size = 0;
        ui64 Offset = 0;
        ui32 DocumentOffset = 0;
        ui32 DocumentsInChunkCount = 0;

        TChunkInfo() = default;
        TChunkInfo(ui32 size, ui64 offset, ui32 documentOffset, ui32 documentsInChunkCount)
            : Size(size)
            , Offset(offset)
            , DocumentOffset(documentOffset)
            , DocumentsInChunkCount(documentsInChunkCount) {
        }
    };
}

static void WriteChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    TCountingOutput* const output,
    TDeque<TChunkInfo>* const chunkInfos,
    flatbuffers::FlatBufferBuilder* const builder) {

    builder->Clear();

    const auto quantsOffset = builder->CreateVector(
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size());
    NCB::NIdl::TQuantizedFeatureChunkBuilder chunkBuilder(*builder);
    chunkBuilder.add_BitsPerDocument(chunk.Chunk->BitsPerDocument());
    chunkBuilder.add_Quants(quantsOffset);
    builder->Finish(chunkBuilder.Finish());

    AddPadding(16, output);

    const auto chunkOffset = output->Counter();
    output->Write(builder->GetBufferPointer(), builder->GetSize());

    chunkInfos->emplace_back(builder->GetSize(), chunkOffset, chunk.DocumentOffset, chunk.DocumentCount);
}

static void WriteHeader(TCountingOutput* const output) {
    output->Write(Magic, MagicSize);
    WriteLittleEndian(Version, output);
    WriteLittleEndian(VersionHash, output);

    const ui32 metainfoSize = 0;
    WriteLittleEndian(metainfoSize, output);

    AddPadding(16, output);

    // we may add some metainfo here
}

static TPoolMetainfo MakePoolMetainfo(
    const THashMap<size_t, size_t>& columnIndexToLocalIndex,
    const TConstArrayRef<EColumn> columnTypes,
    const TConstArrayRef<TString> columnNames,
    const size_t documentCount,
    const TConstArrayRef<size_t> ignoredColumnIndices) {

    Y_ASSERT(columnIndexToLocalIndex.size() == columnTypes.size());
    Y_ASSERT(columnNames.empty() || columnIndexToLocalIndex.size() == columnNames.size());
    const ui32 columnCount = columnTypes.size();

    TPoolMetainfo metainfo;
    metainfo.SetDocumentCount(documentCount);
    metainfo.SetStringDocIdFakeColumnIndex(GetFakeDocIdColumnIndex(columnCount));
    metainfo.SetStringGroupIdFakeColumnIndex(GetFakeGroupIdColumnIndex(columnCount));
    metainfo.SetStringSubgroupIdFakeColumnIndex(GetFakeSubgroupIdColumnIndex(columnCount));

    for (const auto [columnIndex, localIndex] : columnIndexToLocalIndex) {
        NCB::NIdl::EColumnType pbColumnType;
        switch (columnTypes[localIndex]) {
            case EColumn::Num:
                pbColumnType = NCB::NIdl::CT_NUMERIC;
                break;
            case EColumn::Categ:
                pbColumnType = NCB::NIdl::CT_CATEGORICAL;
                break;
            case EColumn::Label:
                pbColumnType = NCB::NIdl::CT_LABEL;
                break;
            case EColumn::Baseline:
                pbColumnType = NCB::NIdl::CT_BASELINE;
                break;
            case EColumn::Weight:
                pbColumnType = NCB::NIdl::CT_WEIGHT;
                break;
            case EColumn::SampleId:
                pbColumnType = NCB::NIdl::CT_DOCUMENT_ID;
                break;
            case EColumn::GroupId:
                pbColumnType = NCB::NIdl::CT_GROUP_ID;
                break;
            case EColumn::GroupWeight:
                pbColumnType = NCB::NIdl::CT_GROUP_WEIGHT;
                break;
            case EColumn::SubgroupId:
                pbColumnType = NCB::NIdl::CT_SUBGROUP_ID;
                break;
            case EColumn::Sparse:
                pbColumnType = NCB::NIdl::CT_SPARSE;
                break;
            case EColumn::Timestamp:
            case EColumn::Prediction:
            case EColumn::Auxiliary:
                ythrow TCatBoostException() << "unexpected column type in quantized pool";
        }

        metainfo.MutableColumnIndexToType()->insert({static_cast<ui32>(columnIndex), pbColumnType});
        if (columnNames) {
            metainfo.MutableColumnIndexToName()->insert({
                static_cast<ui32>(columnIndex),
                columnNames[localIndex]});
        }
    }

    if (ignoredColumnIndices) {
        metainfo.MutableIgnoredColumnIndices()->Reserve(ignoredColumnIndices.size());
        for (const auto index : ignoredColumnIndices) {
            metainfo.AddIgnoredColumnIndices(index);
        }
    }

    return metainfo;
}

static void WriteAsOneFile(const NCB::TQuantizedPool& pool, IOutputStream* slave) {
    TCountingOutput output(slave);

    WriteHeader(&output);

    const auto chunksOffset = output.Counter();

    const auto sortedTrueFeatureIndices = CollectAndSortKeys(pool.ColumnIndexToLocalIndex);
    TDeque<TDeque<TChunkInfo>> perFeatureChunkInfos;
    perFeatureChunkInfos.resize(pool.ColumnIndexToLocalIndex.size());
    {
        flatbuffers::FlatBufferBuilder builder;
        for (const auto trueFeatureIndex : sortedTrueFeatureIndices) {
            const auto localIndex = pool.ColumnIndexToLocalIndex.at(trueFeatureIndex);
            auto* const chunkInfos = &perFeatureChunkInfos[localIndex];
            for (const auto& chunk : pool.Chunks[localIndex]) {
                WriteChunk(chunk, &output, chunkInfos, &builder);
            }
        }
    }

    const ui64 poolMetainfoSizeOffset = output.Counter();
    {
        const auto poolMetainfo = MakePoolMetainfo(
            pool.ColumnIndexToLocalIndex,
            pool.ColumnTypes,
            pool.ColumnNames,
            pool.DocumentCount,
            pool.IgnoredColumnIndices);
        const ui32 poolMetainfoSize = poolMetainfo.ByteSizeLong();
        WriteLittleEndian(poolMetainfoSize, &output);
        poolMetainfo.SerializeToStream(&output);
    }

    const ui64 quantizationSchemaSizeOffset = output.Counter();
    const ui32 quantizationSchemaSize = pool.QuantizationSchema.ByteSizeLong();
    WriteLittleEndian(quantizationSchemaSize, &output);
    pool.QuantizationSchema.SerializeToStream(&output);

    const ui64 featureCountOffset = output.Counter();
    const ui32 featureCount = sortedTrueFeatureIndices.size();
    WriteLittleEndian(featureCount, &output);
    for (const ui32 trueFeatureIndex : sortedTrueFeatureIndices) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(trueFeatureIndex);
        const ui32 chunkCount = perFeatureChunkInfos[localIndex].size();

        WriteLittleEndian(trueFeatureIndex, &output);
        WriteLittleEndian(chunkCount, &output);
        for (const auto& chunkInfo : perFeatureChunkInfos[localIndex]) {
            WriteLittleEndian(chunkInfo.Size, &output);
            WriteLittleEndian(chunkInfo.Offset, &output);
            WriteLittleEndian(chunkInfo.DocumentOffset, &output);
            WriteLittleEndian(chunkInfo.DocumentsInChunkCount, &output);
        }
    }

    WriteLittleEndian(chunksOffset, &output);
    WriteLittleEndian(poolMetainfoSizeOffset, &output);
    WriteLittleEndian(quantizationSchemaSizeOffset, &output);
    WriteLittleEndian(featureCountOffset, &output);
    output.Write(MagicEnd, MagicEndSize);
}

void NCB::SaveQuantizedPool(const TQuantizedPool& pool, IOutputStream* const output) {
    WriteAsOneFile(pool, output);
}

static void ValidatePoolPart(const TConstArrayRef<char> blob) {
    // TODO(yazevnul)
    (void)blob;
}

static void ReadHeader(TCountingInput* const input) {
    char magic[MagicSize];
    const auto magicSize = input->Load(magic, MagicSize);
    CB_ENSURE(MagicSize == magicSize);
    CB_ENSURE(!std::memcmp(magic, Magic, MagicSize));

    ui32 version;
    ReadLittleEndian(&version, input);
    CB_ENSURE(Version == version);

    ui32 versionHash;
    ReadLittleEndian(&versionHash, input);
    CB_ENSURE(VersionHash == versionHash);

    ui32 metainfoSize;
    ReadLittleEndian(&metainfoSize, input);

    SkipPadding(16, input);

    const auto metainfoBytesSkipped = input->Skip(metainfoSize);
    CB_ENSURE(metainfoSize == metainfoBytesSkipped);
}

template <typename T>
static T RoundUpTo(const T value, const T multiple) {
    if (value % multiple == 0) {
        return value;
    }

    return value + (multiple - value % multiple);
}

static void AddPoolMetainfo(const TPoolMetainfo& metainfo, NCB::TQuantizedPool* const pool) {
    pool->DocumentCount = metainfo.GetDocumentCount();
    pool->IgnoredColumnIndices.assign(
        metainfo.GetIgnoredColumnIndices().begin(),
        metainfo.GetIgnoredColumnIndices().end());

    // size of mapping (column_index; column_type) can only be greater than size of mapping
    // (column_index; local_index) becase first must contain all columns, while last may not
    // contain columns for constant features or ignored columns
    CB_ENSURE(
        metainfo.GetColumnIndexToType().size() >= pool->ColumnIndexToLocalIndex.size(),
        LabeledOutput(metainfo.GetColumnIndexToType().size(), pool->ColumnIndexToLocalIndex.size()));

    if (metainfo.GetColumnIndexToType().size() != pool->ColumnIndexToLocalIndex.size()) {
        for (const auto [columnIndex, columnType] : metainfo.GetColumnIndexToType()) {
            const auto inserted  = pool->ColumnIndexToLocalIndex.emplace(
                columnIndex,
                pool->ColumnIndexToLocalIndex.size()).second;

            if (inserted) {
                // create empty chunks vector for this column
                pool->Chunks.push_back({});
            }
        }
    }

    pool->ColumnTypes.resize(pool->ColumnIndexToLocalIndex.size());
    pool->ColumnNames.resize(pool->ColumnIndexToLocalIndex.size());

    CB_ENSURE(pool->ColumnTypes.size() == pool->Chunks.size(),
        "ColumnTypes array should have the same size as Chunks array");

    CB_ENSURE(
        metainfo.GetColumnIndexToName().empty()
        || metainfo.GetColumnIndexToName().size() == metainfo.GetColumnIndexToType().size(),
        "column names must either not be present (in pools generated by old quantizer) or their "
        "number must be the same as number of column types"
        LabeledOutput(
            metainfo.GetColumnIndexToName().size(),
            metainfo.GetColumnIndexToType().size()));

    for (const auto [columnIndex, localIndex] : pool->ColumnIndexToLocalIndex) {
        const auto pbType = metainfo.GetColumnIndexToType().at(columnIndex);
        EColumn type;
        switch (pbType) {
            case NCB::NIdl::CT_UNKNOWN:
                ythrow TCatBoostException() << "unknown column type in quantized pool";
            case NCB::NIdl::CT_NUMERIC:
                type = EColumn::Num;
                break;
            case NCB::NIdl::CT_LABEL:
                type = EColumn::Label;
                break;
            case NCB::NIdl::CT_WEIGHT:
                type = EColumn::Weight;
                break;
            case NCB::NIdl::CT_GROUP_WEIGHT:
                type = EColumn::GroupWeight;
                break;
            case NCB::NIdl::CT_BASELINE:
                type = EColumn::Baseline;
                break;
            case NCB::NIdl::CT_SUBGROUP_ID:
                type = EColumn::SubgroupId;
                break;
            case NCB::NIdl::CT_DOCUMENT_ID:
                type = EColumn::SampleId;
                break;
            case NCB::NIdl::CT_GROUP_ID:
                type = EColumn::GroupId;
                break;
            case NCB::NIdl::CT_CATEGORICAL:
                type = EColumn::Categ;
                break;
            case NCB::NIdl::CT_SPARSE:
                type = EColumn::Sparse;
                break;
            case NCB::NIdl::CT_TIMESTAMP:
                type = EColumn::Timestamp;
                break;
            case NCB::NIdl::CT_PREDICTION:
                type = EColumn::Prediction;
                break;
            case NCB::NIdl::CT_AUXILIARY:
                type = EColumn::Auxiliary;
                break;
        }

        pool->ColumnTypes[localIndex] = type;

        const auto it = metainfo.GetColumnIndexToName().find(columnIndex);
        if (it != metainfo.GetColumnIndexToName().end()) {
            pool->ColumnNames[localIndex] = it->second;
        }
    }
}

namespace {
    struct TEpilogOffsets {
        ui64 ChunksOffset = 0;
        ui64 PoolMetainfoSizeOffset = 0;
        ui64 QuantizationSchemaSizeOffset = 0;
        ui64 FeatureCountOffset = 0;
    };
}

static TEpilogOffsets ReadEpilogOffsets(const TConstArrayRef<char> blob) {
    TEpilogOffsets offsets;

    CB_ENSURE(!std::memcmp(MagicEnd, blob.data() + blob.size() - MagicEndSize, MagicEndSize));

    offsets.ChunksOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 4));

    offsets.PoolMetainfoSizeOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 3));
    CB_ENSURE(offsets.PoolMetainfoSizeOffset < blob.size());
    CB_ENSURE(offsets.PoolMetainfoSizeOffset > offsets.ChunksOffset);

    offsets.QuantizationSchemaSizeOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 2));
    CB_ENSURE(offsets.QuantizationSchemaSizeOffset < blob.size());
    CB_ENSURE(offsets.QuantizationSchemaSizeOffset > offsets.PoolMetainfoSizeOffset);

    offsets.FeatureCountOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64)));
    CB_ENSURE(offsets.FeatureCountOffset < blob.size());
    CB_ENSURE(offsets.FeatureCountOffset > offsets.QuantizationSchemaSizeOffset);

    return offsets;
}

static void CollectChunks(const TConstArrayRef<char> blob, NCB::TQuantizedPool& pool) {
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    TPoolMetainfo poolMetainfo;
    const auto poolMetainfoSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset));
    const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset + sizeof(ui32),
        poolMetainfoSize);
    CB_ENSURE(poolMetainfoParsed);

    const auto quantizationSchemaSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset));
    const auto quantizationSchemaParsed = pool.QuantizationSchema.ParseFromArray(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset + sizeof(ui32),
        quantizationSchemaSize);
    CB_ENSURE(quantizationSchemaParsed);

    TMemoryInput epilog(
        blob.data() + epilogOffsets.FeatureCountOffset,
        blob.size() - epilogOffsets.FeatureCountOffset - MagicEndSize - sizeof(ui64) + 4);

    TVector<TVector<NCB::TQuantizedPool::TChunkDescription>> stringColumnChunks;
    THashMap<ui32, EColumn> stringColumnIndexToColumnType;

    ui32 featureCount;
    ReadLittleEndian(&featureCount, &epilog);
    for (ui32 i = 0; i < featureCount; ++i) {
        ui32 featureIndex;
        ReadLittleEndian(&featureIndex, &epilog);

        CB_ENSURE(!pool.ColumnIndexToLocalIndex.contains(featureIndex),
            "Quantized pool should have unique feature indices, but " <<
            LabeledOutput(featureIndex) << " is repeated.");

        ui32 localFeatureIndex;
        const bool isFakeColumn = NCB::NQuantizationSchemaDetail::IsFakeIndex(featureIndex, poolMetainfo);
        if (!isFakeColumn) {
            localFeatureIndex = pool.Chunks.size();
            pool.ColumnIndexToLocalIndex.emplace(featureIndex, localFeatureIndex);
            pool.Chunks.push_back({});
        } else {
            EColumn columnType;
            if (featureIndex == poolMetainfo.GetStringDocIdFakeColumnIndex()) {
                columnType = EColumn::SampleId;
            } else if (featureIndex == poolMetainfo.GetStringGroupIdFakeColumnIndex()) {
                columnType = EColumn::GroupId;
            } else if (featureIndex == poolMetainfo.GetStringSubgroupIdFakeColumnIndex()){
                columnType = EColumn::SubgroupId;
            } else {
                CB_ENSURE(false, "Bad column type. Should be one of: DocId, GroupId, SubgroupId.");
            }
            stringColumnIndexToColumnType[stringColumnChunks.size()] = columnType;
            stringColumnChunks.push_back({});
        }
        auto& chunks = isFakeColumn ? stringColumnChunks.back() : pool.Chunks[localFeatureIndex];

        ui32 chunkCount;
        ReadLittleEndian(&chunkCount, &epilog);
        ui32 chunkSize;
        ui64 chunkOffset;
        ui32 docOffset;
        ui32 docsInChunkCount;
        const size_t featureEpilogBytes = chunkCount * (sizeof(chunkSize) + sizeof(chunkOffset) + sizeof(docOffset) + sizeof(docsInChunkCount));
        TVector<ui8> featureEpilog(featureEpilogBytes);
        CB_ENSURE(featureEpilogBytes == epilog.Load(featureEpilog.data(), featureEpilogBytes));
        const auto* featureEpilogPtr = featureEpilog.data();
        for (ui32 chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            ReadLittleEndian(&chunkSize, &featureEpilogPtr);

            ReadLittleEndian(&chunkOffset, &featureEpilogPtr);
            CB_ENSURE(chunkOffset >= epilogOffsets.ChunksOffset);
            CB_ENSURE(chunkOffset < blob.size());

            ReadLittleEndian(&docOffset, &featureEpilogPtr);

            ReadLittleEndian(&docsInChunkCount, &featureEpilogPtr);

            const TConstArrayRef<char> chunkBlob{blob.data() + chunkOffset, chunkSize};
            // TODO(yazevnul): validate flatbuffer, including document count
            const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(chunkBlob.data());

            chunks.emplace_back(docOffset, docsInChunkCount, chunk);
        }
    }

    AddPoolMetainfo(poolMetainfo, &pool);

    // `pool.ColumnTypes` expected to have the same size as number of columns in pool,
    // but `pool.Chunks` may also contain chunks with fake columns (with DocId, GroupId and SubgroupId),
    // `AddPoolMetaInfo` works with assumption that `pool.Chunks` and `pool.ColumnTypes` have the same size,
    // so to keep this assumption true we add string chunks after `AddPoolMetainfo` invocation.
    if (pool.HasStringColumns = !stringColumnChunks.empty()) {
        for (ui32 stringColumnId = 0; stringColumnId < stringColumnChunks.size(); ++stringColumnId) {
            const EColumn columnType = stringColumnIndexToColumnType[stringColumnId];
            if (columnType == EColumn::SampleId) {
                pool.StringDocIdLocalIndex = pool.Chunks.size();
            } else if (columnType == EColumn::GroupId) {
                pool.StringGroupIdLocalIndex = pool.Chunks.size();
            } else if (columnType == EColumn::SubgroupId) {
                pool.StringSubgroupIdLocalIndex = pool.Chunks.size();
            } else {
                CB_ENSURE(false, "Bad column type. Should be one of: DocId, GroupId, SubgroupId.");
            }
            pool.Chunks.push_back(std::move(stringColumnChunks[stringColumnId]));
        }
    }
}

NCB::TQuantizedPool NCB::LoadQuantizedPool(
    const TStringBuf path,
    const TLoadQuantizedPoolParameters& params) {

    TQuantizedPool pool;
    pool.Blobs.push_back(params.LockMemory
        ? TBlob::LockedFromFile(TString(path))
        : TBlob::FromFile(TString(path)));

    // TODO(yazevnul): optionally precharge pool

    const TConstArrayRef<char> blobView{
        pool.Blobs.back().AsCharPtr(),
        pool.Blobs.back().Size()};

    ValidatePoolPart(blobView);
    CollectChunks(blobView, pool);

    return pool;
}

static NCB::TQuantizedPoolDigest GetQuantizedPoolDigest(
    const TPoolMetainfo& poolMetainfo,
    const TPoolQuantizationSchema& quantizationSchema) {

    NCB::TQuantizedPoolDigest digest;
    const auto columnIndices = CollectAndSortKeys(poolMetainfo.GetColumnIndexToType());
    size_t featureIndex = std::numeric_limits<size_t>::max();
    for (const auto& columnIndex : columnIndices) {
        const auto columnType = poolMetainfo.GetColumnIndexToType().at(columnIndex);
        featureIndex += columnType == NCB::NIdl::CT_NUMERIC || columnType == NCB::NIdl::CT_CATEGORICAL;
        switch (columnType) {
            case NCB::NIdl::CT_UNKNOWN:
                ythrow TCatBoostException() << "unknown column type in quantized pool";
            case NCB::NIdl::CT_NUMERIC: {
                if (quantizationSchema.GetFeatureIndexToSchema().count(featureIndex) == 0) {
                    continue; //TODO(kirillovs): is this proper way to skip ignored features?
                }
                const auto& borders = quantizationSchema
                    .GetFeatureIndexToSchema()
                    .at(featureIndex)
                    .GetBorders();
                if (borders.empty()) {
                    // const feature, do nothing
                } else if (borders.size() < 1 << 1) {
                    ++digest.NumericFeature1BitCount;
                } else if (borders.size() < 1 << 4) {
                    ++digest.NumericFeature4BitCount;
                } else if (borders.size() < 1 << 8) {
                    ++digest.NumericFeature8BitCount;
                } else {
                    ythrow TCatBoostException() << "unsupported quantized feature bitness";
                }
                break;
            }
            case NCB::NIdl::CT_LABEL:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_WEIGHT:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_GROUP_WEIGHT:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_BASELINE:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_SUBGROUP_ID:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_DOCUMENT_ID:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_GROUP_ID:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_CATEGORICAL:
                // TODO(yazevnul): account them too when categorical features will be quantized
                break;
            case NCB::NIdl::CT_SPARSE:
                // not implemented in CatBoost yet
                break;
            case NCB::NIdl::CT_TIMESTAMP:
                // not implemented for quantization yet;
            case NCB::NIdl::CT_PREDICTION:
            case NCB::NIdl::CT_AUXILIARY:
                // these are always ignored
                break;
        }
    }

    digest.NumericFeatureCount =
        digest.NumericFeature1BitCount +
        digest.NumericFeature4BitCount +
        digest.NumericFeature8BitCount;

    return digest;
}

NCB::TQuantizedPoolDigest NCB::CalculateQuantizedPoolDigest(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<char> blob(file.AsCharPtr(), file.Size());
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    const auto columnsInfoSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset));
    TPoolMetainfo poolMetainfo;
    const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset + sizeof(ui32),
        columnsInfoSize);
    CB_ENSURE(poolMetainfoParsed);

    const auto quantizationSchemaSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset));
    NCB::NIdl::TPoolQuantizationSchema quantizationSchema;
    quantizationSchema.ParseFromArray(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset + sizeof(ui32),
        quantizationSchemaSize);

    return ::GetQuantizedPoolDigest(poolMetainfo, quantizationSchema);
}

NCB::NIdl::TPoolQuantizationSchema NCB::LoadQuantizationSchemaFromPool(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<char> blob(file.AsCharPtr(), file.Size());
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    NCB::NIdl::TPoolQuantizationSchema quantizationSchema;
    const auto quantizationSchemaSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset));
    const auto quantizationSchemaParsed = quantizationSchema.ParseFromArray(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset + sizeof(ui32),
        quantizationSchemaSize);
    CB_ENSURE(quantizationSchemaParsed);

    return quantizationSchema;
}

NCB::NIdl::TPoolMetainfo NCB::LoadPoolMetainfo(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<char> blob(file.AsCharPtr(), file.Size());
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    NCB::NIdl::TPoolMetainfo poolMetainfo;
    const auto poolMetainfoSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset));
    const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset + sizeof(ui32),
        poolMetainfoSize);
    CB_ENSURE(poolMetainfoParsed);

    return poolMetainfo;
}
