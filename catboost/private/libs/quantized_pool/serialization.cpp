#include "serialization.h"
#include "detail.h"
#include "pool.h"
#include "loader.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/metainfo.pb.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/quantized_pool/detail.h>
#include <catboost/private/libs/quantization_schema/detail.h>
#include <catboost/private/libs/quantization_schema/serialization.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>

#include <util/digest/numeric.h>
#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/array_size.h>
#include <util/generic/cast.h>
#include <util/generic/deque.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/length.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>
#include <util/system/info.h>


using NCB::NIdl::TPoolMetainfo;
using NCB::NIdl::TPoolQuantizationSchema;
using NCB::NQuantizationDetail::GetFakeDocIdColumnIndex;
using NCB::NQuantizationDetail::GetFakeGroupIdColumnIndex;
using NCB::NQuantizationDetail::GetFakeSubgroupIdColumnIndex;
using NCB::NQuantizationDetail::IdlColumnTypeToEColumn;

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
                pbColumnType = NCB::NIdl::CT_TIMESTAMP;
                break;
            case EColumn::HashedCateg:
            case EColumn::Prediction:
            case EColumn::Auxiliary:
            case EColumn::Text:
            case EColumn::NumVector:
            case EColumn::Features:
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
        poolMetainfo.SerializeToArcadiaStream(&output);
    }

    const ui64 quantizationSchemaSizeOffset = output.Counter();
    const ui32 quantizationSchemaSize = pool.QuantizationSchema.ByteSizeLong();
    WriteLittleEndian(quantizationSchemaSize, &output);
    pool.QuantizationSchema.SerializeToArcadiaStream(&output);

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

static void ValidatePoolPart(const TConstArrayRef<ui8> blob) {
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

void NCB::AddPoolMetainfo(const TPoolMetainfo& metainfo, NCB::TQuantizedPool* const pool) {
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
        for (const auto& [columnIndex, columnType] : metainfo.GetColumnIndexToType()) {
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
        pool->ColumnTypes[localIndex] = NCB::NQuantizationDetail::IdlColumnTypeToEColumn(pbType);

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

static TEpilogOffsets ReadEpilogOffsets(const TConstArrayRef<ui8> blob) {
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

static void ParseQuantizedPool(
    const TMaybe<std::function<void(TConstArrayRef<ui8>)>>& onMetainfo,
    const TMaybe<std::function<void(TConstArrayRef<ui8>)>>& onBorders,
    const TMaybe<std::function<bool(ui32)>>& onColumn, // ignore column chunks if returns false
    const TMaybe<std::function<void(TConstArrayRef<ui8>, ui32, ui32)>>& onChunk,
    TConstArrayRef<ui8> blob
) {
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    const auto poolMetainfoSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.PoolMetainfoSizeOffset));
    if (onMetainfo) {
        (*onMetainfo)(MakeArrayRef(blob.data() + epilogOffsets.PoolMetainfoSizeOffset + sizeof(ui32), poolMetainfoSize));
    }

    const auto quantizationSchemaSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset));
    if (onBorders) {
        (*onBorders)(MakeArrayRef(blob.data() + epilogOffsets.QuantizationSchemaSizeOffset + sizeof(ui32), quantizationSchemaSize));
    }

    TMemoryInput epilog(
        blob.data() + epilogOffsets.FeatureCountOffset,
        blob.size() - epilogOffsets.FeatureCountOffset - MagicEndSize - sizeof(ui64) + 4);

    ui32 featureCount;
    ReadLittleEndian(&featureCount, &epilog);
    for (ui32 i = 0; i < featureCount; ++i) {
        ui32 featureIndex;
        ReadLittleEndian(&featureIndex, &epilog);

        bool ignoreColumnChucks = false;
        if (onColumn) {
            ignoreColumnChucks = !(*onColumn)(featureIndex);
        }

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
        if (ignoreColumnChucks || !onChunk) {
            continue;
        }
        for (ui32 chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            ReadLittleEndian(&chunkSize, &featureEpilogPtr);

            ReadLittleEndian(&chunkOffset, &featureEpilogPtr);
            CB_ENSURE(chunkOffset >= epilogOffsets.ChunksOffset);
            CB_ENSURE(chunkOffset < blob.size());

            ReadLittleEndian(&docOffset, &featureEpilogPtr);

            ReadLittleEndian(&docsInChunkCount, &featureEpilogPtr);

            (*onChunk)(MakeArrayRef(blob.data() + chunkOffset, chunkSize), docOffset, docsInChunkCount);
        }
    }
}

namespace {
    class TFileQuantizedPoolLoader : public NCB::IQuantizedPoolLoader {
    public:
        explicit TFileQuantizedPoolLoader(const NCB::TPathWithScheme& pathWithScheme)
            : PathWithScheme(pathWithScheme)
        {}
        void LoadQuantizedPool(NCB::TLoadQuantizedPoolParameters params) override;
        NCB::TQuantizedPool ExtractQuantizedPool() override;
        TVector<ui8> LoadQuantizedColumn(ui32 columnIdx) override;
        TVector<ui8> LoadQuantizedColumn(ui32 columnIdx, ui64 offset, ui64 count) override;
        NCB::TPathWithScheme GetPoolPathWithScheme() const override;
    private:
        NCB::TPathWithScheme PathWithScheme;
        NCB::TQuantizedPool Pool;
    };
}

void TFileQuantizedPoolLoader::LoadQuantizedPool(NCB::TLoadQuantizedPoolParameters params) {
    CB_ENSURE_INTERNAL(
        params.DatasetSubset.Range == NCB::TDatasetSubset().Range,
        "Scheme quantized supports only default load subset range"
    );

    Pool.Blobs.push_back(params.LockMemory
        ? TBlob::LockedFromFile(TString(PathWithScheme.Path))
        : TBlob::FromFile(TString(PathWithScheme.Path)));

    // TODO(yazevnul): optionally precharge pool

    const TConstArrayRef<ui8> blob{
        Pool.Blobs.back().AsUnsignedCharPtr(),
        Pool.Blobs.back().Size()};

    ValidatePoolPart(blob);

    TPoolMetainfo poolMetainfo;
    auto parseMetainfo = [&] (TConstArrayRef<ui8> bytes) {
        const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(poolMetainfoParsed);
    };
    auto parseSchema = [&] (TConstArrayRef<ui8> bytes) {
        const auto quantizationSchemaParsed = Pool.QuantizationSchema.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(quantizationSchemaParsed);
    };

    TVector<TVector<NCB::TQuantizedPool::TChunkDescription>> stringColumnChunks;
    THashMap<ui32, EColumn> stringColumnIndexToColumnType;

    TVector<NCB::TQuantizedPool::TChunkDescription>* currentChunksPointer = nullptr;
    auto parseColumn = [&] (ui32 columnIndex) -> bool {
        CB_ENSURE(!Pool.ColumnIndexToLocalIndex.contains(columnIndex),
            "Quantized pool should have unique column indices, but " <<
            LabeledOutput(columnIndex) << " is repeated.");

        const bool isFakeColumn = NCB::NQuantizationSchemaDetail::IsFakeIndex(columnIndex, poolMetainfo);
        if (!isFakeColumn) {
            if (!params.DatasetSubset.HasFeatures) {
                auto pbColumnType = poolMetainfo.columnindextotype().at(columnIndex);
                if (IsFactorColumn(IdlColumnTypeToEColumn(pbColumnType))) {
                    return false;
                }
            }

            const auto localFeatureIndex = Pool.Chunks.size();
            Pool.ColumnIndexToLocalIndex.emplace(columnIndex, localFeatureIndex);
            Pool.Chunks.push_back({});
            currentChunksPointer = &Pool.Chunks.back();
        } else {
            EColumn columnType;
            if (columnIndex == poolMetainfo.GetStringDocIdFakeColumnIndex()) {
                columnType = EColumn::SampleId;
            } else if (columnIndex == poolMetainfo.GetStringGroupIdFakeColumnIndex()) {
                columnType = EColumn::GroupId;
            } else if (columnIndex == poolMetainfo.GetStringSubgroupIdFakeColumnIndex()){
                columnType = EColumn::SubgroupId;
            } else {
                CB_ENSURE(false, "Bad column type. Should be one of: DocId, GroupId, SubgroupId.");
            }
            stringColumnIndexToColumnType[stringColumnChunks.size()] = columnType;
            stringColumnChunks.push_back({});
            currentChunksPointer = &stringColumnChunks.back();
        }
        return true;
    };
    auto parseChunk = [&] (TConstArrayRef<ui8> bytes, ui32 docOffset, ui32 docsInChunkCount) {
        const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(bytes.data());
        currentChunksPointer->emplace_back(docOffset, docsInChunkCount, chunk);
    };
    ParseQuantizedPool(
        parseMetainfo,
        parseSchema,
        parseColumn,
        parseChunk,
        blob);

    AddPoolMetainfo(poolMetainfo, &Pool);

    // `Pool.ColumnTypes` expected to have the same size as number of columns in pool,
    // but `Pool.Chunks` may also contain chunks with fake columns (with DocId, GroupId and SubgroupId),
    // `AddPoolMetaInfo` works with assumption that `Pool.Chunks` and `Pool.ColumnTypes` have the same size,
    // so to keep this assumption true we add string chunks after `AddPoolMetainfo` invocation.
    if (Pool.HasStringColumns = !stringColumnChunks.empty()) {
        for (ui32 stringColumnId = 0; stringColumnId < stringColumnChunks.size(); ++stringColumnId) {
            const EColumn columnType = stringColumnIndexToColumnType[stringColumnId];
            if (columnType == EColumn::SampleId) {
                Pool.StringDocIdLocalIndex = Pool.Chunks.size();
            } else if (columnType == EColumn::GroupId) {
                Pool.StringGroupIdLocalIndex = Pool.Chunks.size();
            } else if (columnType == EColumn::SubgroupId) {
                Pool.StringSubgroupIdLocalIndex = Pool.Chunks.size();
            } else {
                CB_ENSURE(false, "Bad column type. Should be one of: DocId, GroupId, SubgroupId.");
            }
            Pool.Chunks.push_back(std::move(stringColumnChunks[stringColumnId]));
        }
    }
}

NCB::TQuantizedPool TFileQuantizedPoolLoader::ExtractQuantizedPool() {
    return std::move(Pool);
}

TVector<ui8> TFileQuantizedPoolLoader::LoadQuantizedColumn(ui32 /*columnIdx*/) {
    CB_ENSURE_INTERNAL(false, "Schema quantized does not support columnwise loading");
}

TVector<ui8> TFileQuantizedPoolLoader::LoadQuantizedColumn(ui32 /*columnIdx*/, ui64 /*offset*/, ui64 /*count*/) {
    CB_ENSURE_INTERNAL(false, "Schema quantized does not support columnwise loading");
}

NCB::TPathWithScheme TFileQuantizedPoolLoader::GetPoolPathWithScheme() const {
    return PathWithScheme;
}

NCB::TQuantizedPoolLoaderFactory::TRegistrator<TFileQuantizedPoolLoader> FileQuantizedPoolLoaderReg("quantized");

NCB::TQuantizedPool NCB::LoadQuantizedPool(
    const NCB::TPathWithScheme& pathWithScheme,
    const TLoadQuantizedPoolParameters& params
) {
    const auto poolLoader = GetProcessor<IQuantizedPoolLoader, const TPathWithScheme&>(pathWithScheme, pathWithScheme);
    poolLoader->LoadQuantizedPool(params);
    return poolLoader->ExtractQuantizedPool();
}

NCB::TQuantizedPoolDigest NCB::GetQuantizedPoolDigest(
    const NCB::NIdl::TPoolMetainfo& poolMetainfo,
    const NCB::NIdl::TPoolQuantizationSchema& quantizationSchema) {

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
                } else if (borders.size() < 1 << 16) {
                    ++digest.NumericFeature16BitCount;
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
            case NCB::NIdl::CT_TIMESTAMP:
                ++digest.NonFeatureColumnCount;
                break;
            case NCB::NIdl::CT_CATEGORICAL:
                ++digest.CategoricFeatureCount;
                break;
            case NCB::NIdl::CT_SPARSE:
                // not implemented in CatBoost yet
                break;
            case NCB::NIdl::CT_PREDICTION:
            case NCB::NIdl::CT_AUXILIARY:
                // these are always ignored
                break;
        }
    }

    digest.NumericFeatureCount =
        digest.NumericFeature1BitCount +
        digest.NumericFeature4BitCount +
        digest.NumericFeature8BitCount +
        digest.NumericFeature16BitCount;

    digest.ClassesCount = quantizationSchema.classnames_size();

    return digest;
}

NCB::TQuantizedPoolDigest NCB::CalculateQuantizedPoolDigest(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<ui8> blob(file.AsUnsignedCharPtr(), file.Size());
    TPoolMetainfo poolMetainfo;
    auto parseMetainfo = [&] (TConstArrayRef<ui8> bytes) {
        const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(poolMetainfoParsed);
    };
    NCB::NIdl::TPoolQuantizationSchema quantizationSchema;
    auto parseSchema = [&] (TConstArrayRef<ui8> bytes) {
        const auto quantizationSchemaParsed = quantizationSchema.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(quantizationSchemaParsed);
    };
    ParseQuantizedPool(
        parseMetainfo,
        parseSchema,
        /*parseColumn*/ Nothing(),
        /*parseChunk*/ Nothing(),
        blob);

    return GetQuantizedPoolDigest(poolMetainfo, quantizationSchema);
}

NCB::NIdl::TPoolQuantizationSchema NCB::LoadQuantizationSchemaFromPool(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<ui8> blob(file.AsUnsignedCharPtr(), file.Size());
    NCB::NIdl::TPoolQuantizationSchema quantizationSchema;
    auto parseSchema = [&] (TConstArrayRef<ui8> bytes) {
        const auto quantizationSchemaParsed = quantizationSchema.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(quantizationSchemaParsed);
    };
    ParseQuantizedPool(
        /*parseMetainfo*/ Nothing(),
        parseSchema,
        /*parseColumn*/ Nothing(),
        /*parseChunk*/ Nothing(),
        blob);

    return quantizationSchema;
}

size_t NCB::EstimateIdsLength(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<ui8> blob(file.AsUnsignedCharPtr(), file.Size());

    TPoolMetainfo poolMetainfo;
    auto parseMetainfo = [&] (TConstArrayRef<ui8> bytes) {
        const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(poolMetainfoParsed);
    };
    TVector<TVector<NCB::TQuantizedPool::TChunkDescription>> stringColumnChunks;
    bool isFakeColumn = false;
    auto parseColumn = [&] (ui32 columnIndex) -> bool {
        isFakeColumn = NCB::NQuantizationSchemaDetail::IsFakeIndex(columnIndex, poolMetainfo);
        return true;
    };
    size_t estimatedIdsLength = 0;
    auto parseChunk = [&] (TConstArrayRef<ui8> bytes, ui32 docOffset, ui32 docsInChunkCount) {
        const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(bytes.data());
        if (isFakeColumn && docOffset == 0) {
            estimatedIdsLength += 1 + chunk->Quants()->size() / (docsInChunkCount + 1);
        }
    };
    ParseQuantizedPool(
        parseMetainfo,
        /*parseSchema*/ Nothing(),
        parseColumn,
        parseChunk,
        blob);
    return estimatedIdsLength;
}

void NCB::EstimateGroupSize(
    const TStringBuf path,
    double* groupSize,
    double* sqrGroupSize,
    size_t* maxGroupSize
) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<ui8> blob(file.AsUnsignedCharPtr(), file.Size());

    TPoolMetainfo poolMetainfo;
    auto parseMetainfo = [&] (TConstArrayRef<ui8> bytes) {
        const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(poolMetainfoParsed);
    };
    auto onColumn = [&] (ui32 columnIndex) -> bool {
        const auto hasType = poolMetainfo.GetColumnIndexToType().count(columnIndex);
        return hasType && poolMetainfo.GetColumnIndexToType().at(columnIndex) == NCB::NIdl::CT_GROUP_ID;
    };
    ui64 groupCount = 0;
    ui64 sumSqrGroupSize = 0;
    size_t thisGroupSize = 1;
    ui64 docCount = 0;
    *maxGroupSize = 1;
    auto onChunk = [&] (TConstArrayRef<ui8> bytes, ui32 docOffset, ui32 docsInChunkCount) {
        if (docOffset > 0) {
            return;
        }
        const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(bytes.data());
        CB_ENSURE(chunk->BitsPerDocument() == sizeof(ui64) * 8, "Group ids should be 64-bits");
        const ui64* groupIds = reinterpret_cast<const ui64*>(chunk->Quants()->data());
        for (auto idx : xrange((ui32)1, docsInChunkCount)) {
            thisGroupSize += 1;
            if (groupIds[idx - 1] != groupIds[idx]) {
                groupCount += 1;
                sumSqrGroupSize += Sqr(thisGroupSize);
                *maxGroupSize = Max(*maxGroupSize, thisGroupSize);
                thisGroupSize = 1;
            }
        }
        docCount += docsInChunkCount;
    };
    ParseQuantizedPool(
        parseMetainfo,
        /*parseSchema*/ Nothing(),
        onColumn,
        onChunk,
        blob);
    *groupSize = groupCount ? static_cast<double>(docCount) / groupCount : 1;
    *sqrGroupSize = groupCount ? static_cast<double>(sumSqrGroupSize) / groupCount : 1;
}

NCB::NIdl::TPoolMetainfo NCB::LoadPoolMetainfo(const TStringBuf path) {
    const auto file = TBlob::FromFile(TString(path));
    const TConstArrayRef<ui8> blob(file.AsUnsignedCharPtr(), file.Size());
    TPoolMetainfo poolMetainfo;
    auto parseMetainfo = [&] (TConstArrayRef<ui8> bytes) {
        const auto poolMetainfoParsed = poolMetainfo.ParseFromArray(bytes.data(), bytes.size());
        CB_ENSURE(poolMetainfoParsed);
    };
    ParseQuantizedPool(
        parseMetainfo,
        /*parseSchema*/ Nothing(),
        /*parseColumn*/ Nothing(),
        /*parseChunk*/ Nothing(),
        blob);
    return poolMetainfo;
}


namespace NCB {

    template <class T>
    static void AddToPool(const TSrcColumn<T>& srcColumn, TQuantizedPool* quantizedPool) {
        quantizedPool->ColumnTypes.push_back(srcColumn.Type);

        size_t documentOffset = 0;
        TVector<TQuantizedPool::TChunkDescription> chunks;
        for (const auto& dataPart : srcColumn.Data) {
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(
                NIdl::CreateTQuantizedFeatureChunk(
                    builder,
                    static_cast<NIdl::EBitsPerDocumentFeature>(sizeof(T)*8),
                    builder.CreateVector(
                        reinterpret_cast<const ui8*>(dataPart.data()),
                        sizeof(T)*dataPart.size()
                    )
                )
            );
            quantizedPool->Blobs.push_back(TBlob::Copy(builder.GetBufferPointer(), builder.GetSize()));

            chunks.emplace_back(
                documentOffset,
                documentOffset + dataPart.size(),
                flatbuffers::GetRoot<NIdl::TQuantizedFeatureChunk>(
                    quantizedPool->Blobs.back().AsCharPtr()
                )
            );
            documentOffset += dataPart.size();
        }
        quantizedPool->Chunks.push_back(std::move(chunks));
    }


    template <class T>
    static void AddToPool(const TMaybe<TSrcColumn<T>>& srcColumn, TQuantizedPool* quantizedPool) {
        if (srcColumn) {
            AddToPool(*srcColumn, quantizedPool);
        }
    }

    static void AddFeatureDataToPool(
        const THolder<TSrcColumnBase>& srcColumn,
        EColumn columnType,
        TQuantizedPool* quantizedPool
    ) {
        if (srcColumn) {
            if (auto* column = dynamic_cast<TSrcColumn<ui8>*>(srcColumn.Get())) {
                AddToPool(*column, quantizedPool);
            } else if (auto* column = dynamic_cast<TSrcColumn<ui16>*>(srcColumn.Get())) {
                AddToPool(*column, quantizedPool);
            } else if (auto* column = dynamic_cast<TSrcColumn<ui32>*>(srcColumn.Get())) {
                AddToPool(*column, quantizedPool);
            } else {
                CB_ENSURE(false, "Unexpected srcColumn type for feature data");
            }
        } else {
            AddToPool(TSrcColumn<ui8>(columnType), quantizedPool);
        }
    }


    void SaveQuantizedPool(
        const TSrcData& srcData,
        TString fileName
    ) {
        TQuantizedPool pool;
        pool.DocumentCount = srcData.DocumentCount;
        for (auto localIndex : xrange(srcData.LocalIndexToColumnIndex.size())) {
            pool.ColumnIndexToLocalIndex.emplace(srcData.LocalIndexToColumnIndex[localIndex], localIndex);
        }
        pool.QuantizationSchema = QuantizationSchemaToProto(srcData.PoolQuantizationSchema);
        pool.ColumnNames = srcData.ColumnNames;
        pool.IgnoredColumnIndices = srcData.IgnoredColumnIndices;

        for (const auto& floatFeature : srcData.FloatFeatures) {
            AddFeatureDataToPool(floatFeature, EColumn::Num, &pool);
        }
        for (const auto& catFeature : srcData.CatFeatures) {
            AddFeatureDataToPool(catFeature, EColumn::Categ, &pool);
        }

        AddToPool(srcData.GroupIds, &pool);
        AddToPool(srcData.SubgroupIds, &pool);

        AddToPool(srcData.Target, &pool);

        for (const auto& oneBaseline : srcData.Baseline) {
            AddToPool(oneBaseline, &pool);
        }

        AddToPool(srcData.Weights, &pool);
        AddToPool(srcData.GroupWeights, &pool);


        TFileOutput output(fileName);
        SaveQuantizedPool(pool, &output);
    }


    template <class TDst, class T, EFeatureValuesType ValuesType>
    THolder<TSrcColumnBase> GenerateSrcColumn(
        const IQuantizedFeatureValuesHolder<T, ValuesType>& featureColumn
    ) {
        EColumn columnType;
        switch (featureColumn.GetFeatureType()) {
            case EFeatureType::Float:
                columnType = EColumn::Num;
                break;
            case EFeatureType::Categorical:
                columnType = EColumn::Categ;
                break;
            default:
                CB_ENSURE_INTERNAL(false, "Unsupported feature type" << featureColumn.GetFeatureType());
        }
        THolder<TSrcColumn<TDst>> dst(new TSrcColumn<TDst>(columnType));

        featureColumn.ForEachBlock(
            [&dst] (auto blockStartIdx, auto block) {
                Y_UNUSED(blockStartIdx);
                dst->Data.push_back(TVector<TDst>(block.begin(), block.end()));
            },
            QUANTIZED_POOL_COLUMN_DEFAULT_SLICE_COUNT
        );

        return dst;
    }


    static TSrcData BuildSrcDataFromDataProvider(
        TDataProviderPtr dataProvider,
        NPar::ILocalExecutor* localExecutor
    ) {
        TSrcData srcData;

        const auto* const quantizedObjectsData =
            dynamic_cast<const TQuantizedObjectsDataProvider*>(dataProvider->ObjectsData.Get());
        CB_ENSURE(quantizedObjectsData, "Pool is not quantized");

        const auto& quantizedFeaturesInfo = quantizedObjectsData->GetQuantizedFeaturesInfo();
        const auto& featuresLayout = quantizedFeaturesInfo->GetFeaturesLayout();

        CB_ENSURE(
            !featuresLayout->GetEmbeddingFeatureCount(),
            "Quantized pool file format does not support embedding features yet");

        CB_ENSURE(
            !featuresLayout->GetTextFeatureCount(),
            "Quantized pool file format does not support text features yet");


        srcData.DocumentCount = dataProvider->GetObjectCount();
        srcData.ObjectsOrder = quantizedObjectsData->GetOrder();


        TVector<TString> columnNames;

        srcData.PoolQuantizationSchema.ClassLabels = dataProvider->MetaInfo.ClassLabels;

        //features
        TVector<TString> floatFeatureNames;
        TVector<TString> catFeatureNames;

        for (auto externalFeatureIdx : xrange(featuresLayout->GetExternalFeatureCount())) {
            const auto featureMetaInfo = featuresLayout->GetExternalFeatureMetaInfo(externalFeatureIdx);

            if (featureMetaInfo.Type == EFeatureType::Float) {
                const auto floatFeatureIdx =
                    featuresLayout->GetInternalFeatureIdx<EFeatureType::Float>(externalFeatureIdx);

                //for quantizationSchema
                const auto featureBorders = quantizedFeaturesInfo->HasBorders(floatFeatureIdx)
                    ? quantizedFeaturesInfo->GetBorders(floatFeatureIdx)
                    : TVector<float>();
                const auto featureNanMode = quantizedFeaturesInfo->HasNanMode(floatFeatureIdx)
                    ? quantizedFeaturesInfo->GetNanMode(floatFeatureIdx)
                    : ENanMode::Forbidden;

                //for floatFeatures
                TMaybeData<const IQuantizedFloatValuesHolder*> feature =
                    quantizedObjectsData->GetFloatFeature(*floatFeatureIdx);

                THolder<TSrcColumnBase> maybeFeatureColumn;
                if (feature) {
                    CB_ENSURE(quantizedFeaturesInfo->HasBorders(floatFeatureIdx) &&
                        quantizedFeaturesInfo->HasNanMode(floatFeatureIdx),
                        "Borders and NaN processing mode are required for numerical features");

                    CB_ENSURE_INTERNAL(feature.GetRef(),
                        "GetFloatFeature returned nullptr for feature " << externalFeatureIdx << " which is not ignored");

                    //init feature
                    if (featureBorders.size() > Max<ui8>()) {
                        maybeFeatureColumn = GenerateSrcColumn<ui16>(**feature);
                    } else {
                        maybeFeatureColumn = GenerateSrcColumn<ui8>(**feature);
                    }
                } else {
                    /* TODO(akhropov): This should be a valid check but we currently require possibility so save
                     * data without features data but with IsAvailable == true.
                     * Uncomment after fixing MLTOOLS-3604
                     */
                    //CB_ENSURE_INTERNAL(!featureMetaInfo.IsAvailable, "If GetFloatFeature returns Nothing(), feature must be unavailable");
                }

                srcData.PoolQuantizationSchema.Borders.push_back(std::move(featureBorders));
                srcData.PoolQuantizationSchema.NanModes.push_back(featureNanMode);
                srcData.PoolQuantizationSchema.FloatFeatureIndices.push_back(externalFeatureIdx);
                srcData.FloatFeatures.push_back(std::move(maybeFeatureColumn));
                floatFeatureNames.push_back(featureMetaInfo.Name);
            } else {
                Y_ASSERT(featureMetaInfo.Type == EFeatureType::Categorical);

                const auto catFeatureIdx =
                    featuresLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(externalFeatureIdx);

                TMaybeData<const IQuantizedCatValuesHolder*> feature =
                    quantizedObjectsData->GetCatFeature(*catFeatureIdx);

                TMap<ui32, TValueWithCount> dstCatFeaturePerfectHash
                    = quantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx).ToMap();

                THolder<TSrcColumnBase> maybeFeatureColumn;
                if (feature) {
                    if (dstCatFeaturePerfectHash.size() > ((size_t)Max<ui16>() + 1)) {
                        maybeFeatureColumn = GenerateSrcColumn<ui32>(**feature);
                    } else if (dstCatFeaturePerfectHash.size() > ((size_t)Max<ui8>() + 1)) {
                        maybeFeatureColumn = GenerateSrcColumn<ui16>(**feature);
                    } else {
                        maybeFeatureColumn = GenerateSrcColumn<ui8>(**feature);
                    }
                } else {
                    /* TODO(akhropov): This should be a valid check but we currently require possibility so save
                     * data without features data but with IsAvailable == true.
                     * Uncomment after fixing MLTOOLS-3604
                     */
                    //CB_ENSURE_INTERNAL(!featureMetaInfo.IsAvailable, "If GetCatFeature returns Nothing(), feature must be unavailable");
                }

                srcData.PoolQuantizationSchema.CatFeatureIndices.push_back(externalFeatureIdx);
                srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                    std::move(dstCatFeaturePerfectHash)
                );
                srcData.CatFeatures.push_back(std::move(maybeFeatureColumn));
                catFeatureNames.push_back(featureMetaInfo.Name);
            }
        }

        // TODO(akhropov): keep column indices from src raw pool columns metadata
        TVector<size_t> localIndexToColumnIndex;
        localIndexToColumnIndex.insert(
            localIndexToColumnIndex.end(),
            srcData.PoolQuantizationSchema.FloatFeatureIndices.begin(),
            srcData.PoolQuantizationSchema.FloatFeatureIndices.end()
        );
        localIndexToColumnIndex.insert(
            localIndexToColumnIndex.end(),
            srcData.PoolQuantizationSchema.CatFeatureIndices.begin(),
            srcData.PoolQuantizationSchema.CatFeatureIndices.end()
        );

        columnNames.insert(columnNames.end(), floatFeatureNames.begin(), floatFeatureNames.end());
        columnNames.insert(columnNames.end(), catFeatureNames.begin(), catFeatureNames.end());

        //groupIds
        const auto& groupIds = quantizedObjectsData->GetGroupIds();
        if (groupIds) {
            srcData.GroupIds = GenerateSrcColumn<TGroupId>(groupIds.GetRef(), EColumn::GroupId);
            columnNames.push_back("GroupId");
        }

        //subGroupIds
        const auto& subGroupIds = quantizedObjectsData->GetSubgroupIds();
        if (subGroupIds) {
            srcData.SubgroupIds = GenerateSrcColumn<TSubgroupId>(subGroupIds.GetRef(), EColumn::SubgroupId);
            columnNames.push_back("SubgroupId");
        }

        //target
        const ERawTargetType rawTargetType = dataProvider->RawTargetData.GetTargetType();
        switch (rawTargetType) {
            case ERawTargetType::Boolean:
            case ERawTargetType::Integer:
            case ERawTargetType::Float:
                {
                    CB_ENSURE(
                        dataProvider->RawTargetData.GetTargetDimension() == 1,
                        "Multidimensional targets are not currently supported"
                    );
                    TVector<float> targetNumeric;
                    targetNumeric.yresize(dataProvider->GetObjectCount());
                    TArrayRef<float> targetNumericRef = targetNumeric;
                    dataProvider->RawTargetData.GetNumericTarget(
                        TArrayRef<TArrayRef<float>>(&targetNumericRef, 1)
                    );

                    srcData.Target = GenerateSrcColumn<float>(
                        TConstArrayRef<float>(targetNumeric),
                        EColumn::Label
                    );
                    columnNames.push_back("Target");
                }
                break;
            case ERawTargetType::String:
                /* TODO(akhropov): Properly support string targets: MLTOOLS-2393.
                 * This is temporary solution for compatibility for saving pools loaded from files.
                 */
                {
                    CB_ENSURE(
                        dataProvider->RawTargetData.GetTargetDimension() == 1,
                        "Multidimensional targets are not currently supported"
                    );

                    TVector<TConstArrayRef<TString>> targetAsStrings;
                    dataProvider->RawTargetData.GetStringTargetRef(&targetAsStrings);

                    TVector<float> targetFloat;
                    targetFloat.yresize(dataProvider->GetObjectCount());
                    TArrayRef<float> targetFloatRef = targetFloat;
                    TConstArrayRef<TString> targetAsStringsRef = targetAsStrings[0];

                    localExecutor->ExecRangeBlockedWithThrow(
                        [targetFloatRef, targetAsStringsRef] (int i) {
                            CB_ENSURE(
                                TryFromString(targetAsStringsRef[i], targetFloatRef[i]),
                                "String target type is not currently supported"
                            );
                        },
                        0,
                        SafeIntegerCast<int>(dataProvider->GetObjectCount()),
                        /*batchSizeOrZeroForAutoBatchSize*/ 0,
                        NPar::TLocalExecutor::WAIT_COMPLETE
                    );

                    srcData.Target = GenerateSrcColumn<float>(
                        TConstArrayRef<float>(targetFloat),
                        EColumn::Label
                    );
                    columnNames.push_back("Target");
                }
                break;
            case ERawTargetType::None:
                break;
        }
        //baseline
        const auto& baseline = dataProvider->RawTargetData.GetBaseline();
        if (baseline) {
            for (size_t baselineIdx : xrange(baseline.GetRef().size())) {
                TSrcColumn<float> currentBaseline = GenerateSrcColumn<float>(baseline.GetRef()[baselineIdx], EColumn::Baseline);
                srcData.Baseline.emplace_back(currentBaseline);
                columnNames.push_back("Baseline " + ToString(baselineIdx));
            }
        }

        //weights
        const auto& weights = dataProvider->RawTargetData.GetWeights();
        if (!weights.IsTrivial()) {
            srcData.Weights = GenerateSrcColumn<float>(weights.GetNonTrivialData(), EColumn::Weight);
            columnNames.push_back("Weight");
        }

        //groupWeights
        const auto& groupWeights = dataProvider->RawTargetData.GetGroupWeights();
        if (!groupWeights.IsTrivial()) {
            srcData.GroupWeights = GenerateSrcColumn<float>(groupWeights.GetNonTrivialData(), EColumn::GroupWeight);
            columnNames.push_back("GroupWeight");
        }

        // Sequential order after features
        // TODO(akhropov): keep column indices from src raw pool columns metadata
        localIndexToColumnIndex.resize(columnNames.size());
        std::iota(
            localIndexToColumnIndex.begin() + featuresLayout->GetExternalFeatureCount(),
            localIndexToColumnIndex.end(),
            featuresLayout->GetExternalFeatureCount());

        //fill other attributes
        srcData.ColumnNames = columnNames;
        srcData.LocalIndexToColumnIndex = localIndexToColumnIndex;

        return srcData;
    }


    void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName) {
        const auto threadCount = NSystemInfo::CachedNumberOfCpus();
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount);

        TSrcData srcData = BuildSrcDataFromDataProvider(dataProvider, &localExecutor);

        SaveQuantizedPool(srcData, fileName);
    }
}
