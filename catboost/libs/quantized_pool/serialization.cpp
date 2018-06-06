#include "serialization.h"
#include "pool.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
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

static const char Magic[] = "CatboostQuantizedPoolPart";
static const size_t MagicSize = Y_ARRAY_SIZE(Magic);  // yes, with terminating zero
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
    const auto bytesRead = input->Read(&le, sizeof(le));
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
    Y_ENSURE(bytesToSkip == bytesSkipped);
}

namespace {
    struct TChunkInfo {
        ui64 Offset{0};
        ui32 DocumentOffset{0};
        ui32 DocumentsInChunkCount{0};

        TChunkInfo() = default;
        TChunkInfo(ui64 offset, ui32 documentOffset, ui32 documentsInChunkCount)
            : Offset{offset}
            , DocumentOffset{documentOffset}
            , DocumentsInChunkCount{documentsInChunkCount} {
        }
    };
}

static void WriteChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    TCountingOutput* output,
    TDeque<TChunkInfo>* chunkInfos,
    flatbuffers::FlatBufferBuilder* builder) {

    chunkInfos->emplace_back(output->Counter(), chunk.DocumentOffset, chunk.DocumentCount);
    builder->Clear();

    const auto quantsOffset = builder->CreateVector(
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size());
    NCB::NIdl::TQuantizedFeatureChunkBuilder chunkBuilder(*builder);
    chunkBuilder.add_BitsPerDocument(chunk.Chunk->BitsPerDocument());
    chunkBuilder.add_Quants(quantsOffset);
    builder->Finish(chunkBuilder.Finish());

    AddPadding(16, output);
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

static void WriteAsOneFile(const NCB::TQuantizedPool& pool, IOutputStream* slave) {
    TCountingOutput output{slave};

    WriteHeader(&output);

    const auto chunksOffset = output.Counter();

    const auto sortedTrueFeatureIndices = CollectAndSortKeys(pool.TrueFeatureIndexToLocalIndex);
    TDeque<TDeque<TChunkInfo>> perFeatureChunkInfos;
    perFeatureChunkInfos.resize(pool.TrueFeatureIndexToLocalIndex.size());
    {
        // NOTE: if we want it to be supported by filesystems with filesize limit of 2^32 bytes
        // (e.g. FAT) or operating systems with 2^32 limit of address space (any 32-bit OS) then we
        // must start checking if file size is reaching its limit, stop writing chunks and start
        // write epilog with chunk offsets.

        flatbuffers::FlatBufferBuilder builder;
        for (const auto trueFeatureIndex : sortedTrueFeatureIndices) {
            const auto localIndex = pool.TrueFeatureIndexToLocalIndex.at(trueFeatureIndex);
            auto* const chunkInfos = &perFeatureChunkInfos[localIndex];
            for (const auto& chunk : pool.Chunks[localIndex]) {
                WriteChunk(chunk, &output, chunkInfos, &builder);
            }
        }
    }

    const ui64 epilogOffset = output.Counter();

    const ui32 featureCountInFile = sortedTrueFeatureIndices.size();
    WriteLittleEndian(featureCountInFile, &output);
    for (const ui32 trueFeatureIndex : sortedTrueFeatureIndices) {
        const auto localIndex = pool.TrueFeatureIndexToLocalIndex.at(trueFeatureIndex);
        const ui32 chunkCount = perFeatureChunkInfos[localIndex].size();

        WriteLittleEndian(trueFeatureIndex, &output);
        WriteLittleEndian(chunkCount, &output);
        for (const auto& chunkInfo : perFeatureChunkInfos[localIndex]) {
            WriteLittleEndian(chunkInfo.Offset, &output);
            WriteLittleEndian(chunkInfo.DocumentOffset, &output);
            WriteLittleEndian(chunkInfo.DocumentsInChunkCount, &output);
        }
    }

    WriteLittleEndian(chunksOffset, &output);
    WriteLittleEndian(epilogOffset, &output);
}

TVector<TString> NCB::SaveQuantizedPool(
    const TQuantizedPool& schema,
    TStringBuf directory,
    TStringBuf basename,
    TStringBuf extension) {

    // NOTE: if we want to support 32-bit OSes we must write into multiple files each not greater
    // then 2^32 bytes.

    const auto path = TFsPath{directory} / TString::Join(basename, ".", extension);

    // TODO(yazevnul): make it atomic (first write to tmp file, then move it)

    TFileOutput output{path};
    WriteAsOneFile(schema, &output);

    return {ToString(path)};
}

static void ValidatePoolPart(const TConstArrayRef<char> blob) {
    // TODO(yazevnul)
    (void)blob;
}

static void ReadHeader(TCountingInput* const input) {
    char magic[MagicSize];
    const auto magicSize = input->Read(magic, MagicSize);
    CB_ENSURE(magicSize == magicSize);
    CB_ENSURE(!std::memcmp(magic, Magic, MagicSize));

    ui32 version;
    ReadLittleEndian(&version, input);
    CB_ENSURE(version == Version);

    ui32 versionHash;
    ReadLittleEndian(&versionHash, input);

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

static void CollectChunks(const TConstArrayRef<char> blob, NCB::TQuantizedPool& pool) {
    const auto chunksOffsetByReading = [blob] {
        TMemoryInput slave{blob.data(), blob.size()};
        TCountingInput input{&slave};

        ReadHeader(&input);

        return input.Counter();
    }();

    const auto chunksOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - sizeof(ui64) - sizeof(ui64)));
    CB_ENSURE(chunksOffset == chunksOffsetByReading);

    const auto epilogOffset = LittleToHost(ReadUnaligned<ui64>(
        blob.data() + blob.size() - sizeof(ui64)));
    CB_ENSURE(epilogOffset <= blob.size() - sizeof(ui64));

    TMemoryInput epilog{blob.data() + epilogOffset, blob.size() - epilogOffset - 2 * sizeof(ui64)};

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
            ui64 chunkSizeOffset;
            ReadLittleEndian(&chunkSizeOffset, &epilog);

            ui32 docOffset;
            ReadLittleEndian(&docOffset, &epilog);

            ui32 docsInChunkCount;
            ReadLittleEndian(&docsInChunkCount, &epilog);

            const auto chunkSize = LittleToHost(ReadUnaligned<ui32>(blob.data() + chunkSizeOffset));
            const auto chunkOffset = RoundUpTo<ui64>(chunkSizeOffset + sizeof(ui64), 16);

            const TConstArrayRef<char> chunkBlob{blob.data() + chunkOffset, chunkSize};
            // TODO(yazevnul): validate flatbuffer, including document count
            const auto* const chunk = flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(chunkBlob.data());

            pool.Chunks[localFeatureIndex].emplace_back(docOffset, docsInChunkCount, chunk);
        }
    }
}

NCB::TQuantizedPool NCB::LoadQuantizedPool(
    const TConstArrayRef<TString> files,
    const TLoadQuantizedPoolParameters& params) {

    TQuantizedPool pool;
    for (const auto& path : files) {
        pool.Blobs.push_back(params.LockMemory
            ? TBlob::LockedFromFile(path)
            : TBlob::FromFile(path));

        // TODO(yazevnul): optionally precharge pool

        const TConstArrayRef<char> blobView{
            pool.Blobs.back().AsCharPtr(),
            pool.Blobs.back().Size()};

        ValidatePoolPart(blobView);
        CollectChunks(blobView, pool);
    }

    return pool;
}
