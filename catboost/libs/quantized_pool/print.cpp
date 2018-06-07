#include "pool.h"
#include "print.h"

#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/system/unaligned_mem.h>

static TDeque<size_t> CollectAndSortKeys(const THashMap<size_t, size_t>& map) {
    TDeque<size_t> keys;
    for (const auto& kv : map) {
        keys.push_back(kv.first);
    }

    Sort(keys);

    return keys;
}

static TDeque<NCB::TQuantizedPool::TChunkDescription> GetChunksSortedByOffset(
    const TConstArrayRef<NCB::TQuantizedPool::TChunkDescription> chunks) {

    TDeque<NCB::TQuantizedPool::TChunkDescription> res(chunks.begin(), chunks.end());
    Sort(res, [](const auto& lhs, const auto& rhs) {
        return lhs.DocumentOffset < rhs.DocumentOffset;
    });
    return res;
}

static size_t GetMaxFeatureCountInChunk(const NCB::NIdl::TQuantizedFeatureChunk& chunk) {
    return size_t{8}
        * sizeof(*chunk.Quants()->begin())
        * chunk.Quants()->size()
        / static_cast<size_t>(chunk.BitsPerDocument());
}

static void PrintHumanReadableChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    const NCB::NIdl::TFeatureQuantizationSchema* const schema,
    IOutputStream* const output) {

    // TODO(yazevnul): support rest of bitness options
    // TODO(yazevnul): generated enum members are ugly, maybe rename them?
    CB_ENSURE(chunk.Chunk->BitsPerDocument() == NCB::NIdl::EBitsPerDocumentFeature_BPDF_8);

    CB_ENSURE(chunk.DocumentCount <= GetMaxFeatureCountInChunk(*chunk.Chunk));

    TUnalignedMemoryIterator<ui8> borderIndexIt{
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size()};
    for (size_t i = 0; i < chunk.DocumentCount; ++i, (void)borderIndexIt.Next()) {
        if (i > 0) {
            (*output) << ' ';
        }

        const size_t borderIndex = borderIndexIt.Cur();
        if (schema) {
            const auto value = schema->GetBorders(borderIndex);
            (*output) << value;
        } else {
            (*output) << borderIndex;
        }
    }

    (*output) << '\n';
}

static void PrintHumanReadable(
    const NCB::TQuantizedPool& pool,
    const NCB::TPrintQuantizedPoolParameters&,
    IOutputStream* const output,
    const NCB::NIdl::TPoolQuantizationSchema* const schema) {

    (*output) << pool.TrueFeatureIndexToLocalIndex.size() << '\n';

    const auto featureIndices = CollectAndSortKeys(pool.TrueFeatureIndexToLocalIndex);
    for (const auto featureIndex : featureIndices) {
        const auto localFeatureIndex = pool.TrueFeatureIndexToLocalIndex.at(featureIndex);
        const auto chunks = GetChunksSortedByOffset(pool.Chunks[localFeatureIndex]);
        const auto* const featureQuantizationSchema = schema
            ? &schema->GetFeatureIndexToSchema().at(featureIndex)
            : nullptr;

        (*output) << featureIndex << ' ' << chunks.size() << '\n';
        for (const auto& chunk : chunks) {
            (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
            PrintHumanReadableChunk(chunk, featureQuantizationSchema, output);
        }
    }
}

void NCB::PrintQuantizedPool(
    const TQuantizedPool& pool,
    const TPrintQuantizedPoolParameters& params,
    IOutputStream* const output,
    const NIdl::TPoolQuantizationSchema* const schema) {

    switch (params.Format) {
        case NCB::EQuantizedPoolPrintFormat::HumanReadable:
            PrintHumanReadable(pool, params, output, schema);
            return;
        case NCB::EQuantizedPoolPrintFormat::Unknown:
            break;
    }

    ythrow TCatboostException{} << "unknown format";
}
