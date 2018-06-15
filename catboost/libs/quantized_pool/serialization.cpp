#include "serialization.h"
#include "pool.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/column_types.pb.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/helpers/exception.h>

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

using NCB::NIdl::TColumnsInfo;
using NCB::NIdl::TPoolQuantizationSchema;

static const char Magic[] = "CatboostQuantizedPool";
static const size_t MagicSize = Y_ARRAY_SIZE(Magic);  // yes, with terminating zero
static const char MagicEnd[] = "CatboostQuantizedPoolEnd";
static const size_t MagicEndSize = Y_ARRAY_SIZE(MagicEnd);  // yes, with terminating zero
static const ui32 Version = 1;
static const ui32 VersionHash = IntHash(Version);

static TDeque<ui32> CollectAndSortKeys(const THashMap<size_t, size_t>& m) {
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
        ui32 Size{0};
        ui64 Offset{0};
        ui32 DocumentOffset{0};
        ui32 DocumentsInChunkCount{0};

        TChunkInfo() = default;
        TChunkInfo(ui32 size, ui64 offset, ui32 documentOffset, ui32 documentsInChunkCount)
            : Size{size}
            , Offset{offset}
            , DocumentOffset{documentOffset}
            , DocumentsInChunkCount{documentsInChunkCount} {
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

    const ui32 metaInfoSize = 0;
    WriteLittleEndian(metaInfoSize, output);

    AddPadding(16, output);

    // we may add some metainfo here
}

static TColumnsInfo MakeColumnsInfo(
    const THashMap<size_t, size_t>& trueFeatureIndexToLocalIndex,
    const TConstArrayRef<EColumn> columnTypes) {

    Y_ASSERT(trueFeatureIndexToLocalIndex.size() == columnTypes.size());

    TColumnsInfo columnsInfo;
    for (const auto& kv : trueFeatureIndexToLocalIndex) {
        NCB::NIdl::EColumnType pbColumnType;
        switch (columnTypes[kv.second]) {
            case EColumn::Num:
                pbColumnType = NCB::NIdl::CT_NUMERIC;
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
            case EColumn::DocId:
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
            case EColumn::Timestamp:
            case EColumn::Sparse:
            case EColumn::Prediction:
            case EColumn::Categ:
            case EColumn::Auxiliary:
                ythrow TCatboostException() << "unexpected column type in quantized pool";
        }

        columnsInfo.MutableColumnIndexToType()->insert({static_cast<ui32>(kv.first), pbColumnType});
    }
    return columnsInfo;
}

static void WriteAsOneFile(const NCB::TQuantizedPool& pool, IOutputStream* slave) {
    TCountingOutput output{slave};

    WriteHeader(&output);

    const auto chunksOffset = output.Counter();

    const auto sortedTrueFeatureIndices = CollectAndSortKeys(pool.TrueFeatureIndexToLocalIndex);
    TDeque<TDeque<TChunkInfo>> perFeatureChunkInfos;
    perFeatureChunkInfos.resize(pool.TrueFeatureIndexToLocalIndex.size());
    {
        flatbuffers::FlatBufferBuilder builder;
        for (const auto trueFeatureIndex : sortedTrueFeatureIndices) {
            const auto localIndex = pool.TrueFeatureIndexToLocalIndex.at(trueFeatureIndex);
            auto* const chunkInfos = &perFeatureChunkInfos[localIndex];
            for (const auto& chunk : pool.Chunks[localIndex]) {
                WriteChunk(chunk, &output, chunkInfos, &builder);
            }
        }
    }

    const ui64 columnsInfoSizeOffset = output.Counter();
    {
        const auto columnsInfo = MakeColumnsInfo(
            pool.TrueFeatureIndexToLocalIndex,
            pool.ColumnTypes);
        const ui32 columnsInfoSize = columnsInfo.ByteSizeLong();
        WriteLittleEndian(columnsInfoSize, &output);
        columnsInfo.SerializeToStream(&output);
    }

    const ui64 quantizationSchemaSizeOffset = output.Counter();
    const ui32 quantizationSchemaSize = pool.QuantizationSchema.ByteSizeLong();
    WriteLittleEndian(quantizationSchemaSize, &output);
    pool.QuantizationSchema.SerializeToStream(&output);

    const ui64 featureCountOffset = output.Counter();
    const ui32 featureCount = sortedTrueFeatureIndices.size();
    WriteLittleEndian(featureCount, &output);
    for (const ui32 trueFeatureIndex : sortedTrueFeatureIndices) {
        const auto localIndex = pool.TrueFeatureIndexToLocalIndex.at(trueFeatureIndex);
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
    WriteLittleEndian(columnsInfoSizeOffset, &output);
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

    ui32 metaInfoSize;
    ReadLittleEndian(&metaInfoSize, input);

    SkipPadding(16, input);

    const auto metaInfoBytesSkipped = input->Skip(metaInfoSize);
    CB_ENSURE(metaInfoSize == metaInfoBytesSkipped);
}

template <typename T>
static T RoundUpTo(const T value, const T multiple) {
    if (value % multiple == 0) {
        return value;
    }

    return value + (multiple - value % multiple);
}

static void TransformToCppEnumArray(
    const THashMap<size_t, size_t>& trueFeatureIndexToLocalIndex,
    const TColumnsInfo& columnsInfo,
    TVector<EColumn>* const columnTypes) {

    CB_ENSURE(trueFeatureIndexToLocalIndex.size() == columnsInfo.GetColumnIndexToType().size());

    columnTypes->resize(trueFeatureIndexToLocalIndex.size());
    for (const auto kv : trueFeatureIndexToLocalIndex) {
        const auto pbType = columnsInfo.GetColumnIndexToType().at(kv.first);
        EColumn type;
        switch (pbType) {
            case NCB::NIdl::CT_UNKNOWN:
                ythrow TCatboostException() << "unknown column type in quantized pool";
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
                type = EColumn::DocId;
                break;
            case NCB::NIdl::CT_GROUP_ID:
                type = EColumn::GroupId;
                break;
        }

        columnTypes->operator[](kv.second) = type;
    }
}

namespace {
    struct TEpilogOffsets {
        ui64 ChunksOffset = 0;
        ui64 ColumnsInfoSizeOffset = 0;
        ui64 QuantizationSchemaSizeOffset = 0;
        ui64 FeatureCountOffset = 0;
    };
}

static TEpilogOffsets ReadEpilogOffsets(const TConstArrayRef<char> blob) {
    TEpilogOffsets offsets;

    CB_ENSURE(!std::memcmp(MagicEnd, blob.data() + blob.size() - MagicEndSize, MagicEndSize));

    offsets.ChunksOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 4));

    offsets.ColumnsInfoSizeOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 3));
    CB_ENSURE(offsets.ColumnsInfoSizeOffset < blob.size());
    CB_ENSURE(offsets.ColumnsInfoSizeOffset > offsets.ChunksOffset);

    offsets.QuantizationSchemaSizeOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - MagicEndSize - sizeof(ui64) * 2));
    CB_ENSURE(offsets.QuantizationSchemaSizeOffset < blob.size());
    CB_ENSURE(offsets.QuantizationSchemaSizeOffset > offsets.ColumnsInfoSizeOffset);

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

    const auto columnsInfo = [blob, epilogOffsets]{
        const auto size = LittleToHost(ReadUnaligned<ui32>(
            blob.data() + epilogOffsets.ColumnsInfoSizeOffset));
        TColumnsInfo columnsInfo;
        const auto columnsInfoParsed = columnsInfo.ParseFromArray(
            blob.data() + epilogOffsets.ColumnsInfoSizeOffset + sizeof(ui32),
            size);
        CB_ENSURE(columnsInfoParsed);
        return columnsInfo;
    }();

    const auto quantizationSchemaSize = LittleToHost(ReadUnaligned<ui32>(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset));
    const auto quantizationSchemaParsed = pool.QuantizationSchema.ParseFromArray(
        blob.data() + epilogOffsets.QuantizationSchemaSizeOffset + sizeof(ui32),
        quantizationSchemaSize);
    CB_ENSURE(quantizationSchemaParsed);

    TMemoryInput epilog(
        blob.data() + epilogOffsets.FeatureCountOffset,
        blob.size() - epilogOffsets.FeatureCountOffset - MagicEndSize - sizeof(ui64) + 4);

    ui32 featureCount;
    ReadLittleEndian(&featureCount, &epilog);
    for (ui32 i = 0; i < featureCount; ++i) {
        ui32 featureIndex;
        ReadLittleEndian(&featureIndex, &epilog);

        ui32 localFeatureIndex;
        if (const auto* const it = pool.TrueFeatureIndexToLocalIndex.FindPtr(featureIndex)) {
            localFeatureIndex = *it;
        } else {
            localFeatureIndex = pool.Chunks.size();
            pool.TrueFeatureIndexToLocalIndex.emplace(featureIndex, localFeatureIndex);
            pool.Chunks.push_back({});
        }

        ui32 chunkCount;
        ReadLittleEndian(&chunkCount, &epilog);
        for (ui32 chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            ui32 chunkSize;
            ReadLittleEndian(&chunkSize, &epilog);

            ui64 chunkOffset;
            ReadLittleEndian(&chunkOffset, &epilog);
            CB_ENSURE(chunkOffset >= epilogOffsets.ChunksOffset);
            CB_ENSURE(chunkOffset < blob.size());

            ui32 docOffset;
            ReadLittleEndian(&docOffset, &epilog);

            ui32 docsInChunkCount;
            ReadLittleEndian(&docsInChunkCount, &epilog);

            const TConstArrayRef<char> chunkBlob{blob.data() + chunkOffset, chunkSize};
            // TODO(yazevnul): validate flatbuffer, including document count
            const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(chunkBlob.data());

            pool.Chunks[localFeatureIndex].emplace_back(docOffset, docsInChunkCount, chunk);
        }
    }

    TransformToCppEnumArray(
        pool.TrueFeatureIndexToLocalIndex,
        columnsInfo,
        &pool.ColumnTypes);
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

static void CollectDigest(
    const TConstArrayRef<char> blob,
    NCB::TQuantizedPoolDigest* const digest) {

    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave(blob.data(), blob.size());
        TCountingInput input(&slave);
        ReadHeader(&input);
        return input.Counter();
    }();
    const auto epilogOffsets = ReadEpilogOffsets(blob);
    CB_ENSURE(chunksOffsetByReading == epilogOffsets.ChunksOffset);

    TMemoryInput epilog(
        blob.data() + epilogOffsets.FeatureCountOffset,
        blob.size() - epilogOffsets.FeatureCountOffset - MagicEndSize - sizeof(ui64) * 4);

    ui32 featureCount;
    ReadLittleEndian(&featureCount, &epilog);
    for (ui32 i = 0; i < featureCount; ++i) {
        ui32 featureIndex;
        ReadLittleEndian(&featureIndex, &epilog);

        ui32 localFeatureIndex;
        if (const auto* const it = digest->TrueFeatureIndexToLocalIndex.FindPtr(featureIndex)) {
            localFeatureIndex = *it;
        } else {
            localFeatureIndex = digest->Chunks.size();
            digest->TrueFeatureIndexToLocalIndex.emplace(featureIndex, localFeatureIndex);
            digest->Chunks.push_back({});
        }

        ui32 chunkCount;
        ReadLittleEndian(&chunkCount, &epilog);
        for (ui32 chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            ui32 chunkSize;
            ReadLittleEndian(&chunkSize, &epilog);

            ui64 chunkOffset;
            ReadLittleEndian(&chunkOffset, &epilog);
            CB_ENSURE(chunkOffset >= epilogOffsets.ChunksOffset);
            CB_ENSURE(chunkOffset < blob.size());

            ui32 docOffset;
            ReadLittleEndian(&docOffset, &epilog);

            ui32 docsInChunkCount;
            ReadLittleEndian(&docsInChunkCount, &epilog);

            digest->Chunks[localFeatureIndex].emplace_back(
                docOffset,
                docsInChunkCount,
                chunkSize);
        }
    }
}

NCB::TQuantizedPoolDigest NCB::ReadQuantizedPoolDigest(const TStringBuf path) {
    TQuantizedPoolDigest digest;
    const auto file = TBlob::FromFile(TString(path));
    CollectDigest({file.AsCharPtr(), file.Size()}, &digest);

    digest.DocumentCount.resize(digest.TrueFeatureIndexToLocalIndex.size());
    digest.ChunkSizeInBytesSums.resize(digest.TrueFeatureIndexToLocalIndex.size());

    for (const auto kv : digest.TrueFeatureIndexToLocalIndex) {
        for (const auto& chunk : digest.Chunks[kv.second]) {
            digest.DocumentCount[kv.second] += chunk.DocumentCount;
            digest.ChunkSizeInBytesSums[kv.second] += chunk.SizeInBytes;
        }
    }

    return digest;
}

NCB::NIdl::TPoolQuantizationSchema NCB::LoadQuantizationSchema(const TStringBuf path) {
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
