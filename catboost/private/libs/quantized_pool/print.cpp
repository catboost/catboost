#include "detail.h"
#include "pool.h"
#include "print.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/stream/labeled.h>
#include <util/stream/output.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>

using NCB::NQuantizationDetail::IsDoubleColumn;
using NCB::NQuantizationDetail::IsFloatColumn;
using NCB::NQuantizationDetail::IsRequiredColumn;
using NCB::NQuantizationDetail::IsUi32Column;
using NCB::NQuantizationDetail::IsUi64Column;
using NCB::NQuantizationDetail::IsStringColumn;

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
    return size_t(8)
        * sizeof(decltype(*chunk.Quants()->begin()))
        * chunk.Quants()->size()
        / static_cast<size_t>(chunk.BitsPerDocument());
}

template <typename T>
static void PrintHumanReadableNumericChunkImpl(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    const NCB::NIdl::TFeatureQuantizationSchema* const schema,
    IOutputStream* const output) {

    TUnalignedMemoryIterator<T> borderIndexIt{
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size()};
    for (size_t i = 0; i < chunk.DocumentCount; ++i, (void)borderIndexIt.Next()) {
        if (i > 0) {
            (*output) << ' ';
        }

        if (schema) {
            const auto index = static_cast<size_t>(borderIndexIt.Cur());
            if (index >= static_cast<size_t>(schema->GetBorders().size())) {
                (*output) << '>' << *schema->GetBorders().rbegin();
            } else {
                (*output) << '<' << schema->GetBorders(borderIndexIt.Cur());
            }
        } else {
            // `operator <<` treats `ui8` as character, so we force it to treat it like a number
            (*output) << static_cast<ui64>(borderIndexIt.Cur());
        }
    }
}

static void PrintHumanReadableNumericChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    const NCB::NIdl::TFeatureQuantizationSchema* const schema,
    IOutputStream* const output) {

    CB_ENSURE(
        chunk.DocumentCount <= GetMaxFeatureCountInChunk(*chunk.Chunk),
        LabeledOutput(chunk.DocumentCount, GetMaxFeatureCountInChunk(*chunk.Chunk)));

    // TODO(yazevnul): support rest of bitness options
    switch (const auto bitsPerDocument = chunk.Chunk->BitsPerDocument()) {
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_8:
            PrintHumanReadableNumericChunkImpl<ui8>(chunk, schema, output);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_16:
            PrintHumanReadableNumericChunkImpl<ui16>(chunk, schema, output);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_32:
            PrintHumanReadableNumericChunkImpl<ui32>(chunk, schema, output);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_64:
            PrintHumanReadableNumericChunkImpl<ui64>(chunk, schema, output);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_1:
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_2:
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_4:
            ythrow TCatBoostException() << LabeledOutput((int)bitsPerDocument) << " is not supported";
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_UKNOWN:
            ythrow TCatBoostException() << "invalid value";
    }
}

static void PrintHumanReadableNumericChunks(
    const TDeque<NCB::TQuantizedPool::TChunkDescription>& chunks,
    const NCB::NIdl::TFeatureQuantizationSchema* const schema,
    const bool chunkWise,
    IOutputStream* const output) {

    if (chunkWise) {
        for (const auto& chunk : chunks) {
            (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
            PrintHumanReadableNumericChunk(chunk, schema, output);
            (*output) << '\n';
        }
        return;
    }

    for (size_t i = 0, iEnd = chunks.size(); i < iEnd; ++i) {
        if (i > 0) {
            (*output) << ' ';
        }
        PrintHumanReadableNumericChunk(chunks[i], schema, output);
    }

    (*output) << '\n';
}

template <typename T>
static void PrintHumanReadableChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    IOutputStream* const output) {

    CB_ENSURE(static_cast<size_t>(chunk.Chunk->BitsPerDocument()) == sizeof(T) * 8);
    CB_ENSURE(
        chunk.DocumentCount <= GetMaxFeatureCountInChunk(*chunk.Chunk),
        LabeledOutput(chunk.DocumentCount, GetMaxFeatureCountInChunk(*chunk.Chunk)));

    TUnalignedMemoryIterator<T> it(
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size());
    for (size_t i = 0; i < chunk.DocumentCount; ++i, (void)it.Next()) {
        if (i > 0) {
            (*output) << ' ';
        }

        (*output) << it.Cur();
    }

}

template <typename T>
static void PrintHumanReadableChunks(
    const TDeque<NCB::TQuantizedPool::TChunkDescription>& chunks,
    const bool chunkWise,
    IOutputStream* const output) {

    if (chunkWise) {
        for (const auto& chunk : chunks) {
            (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
            PrintHumanReadableChunk<T>(chunk, output);
            (*output) << '\n';
        }
        return;
    }

    for (size_t i = 0, iEnd = chunks.size(); i < iEnd; ++i) {
        if (i > 0) {
            (*output) << ' ';
        }

        PrintHumanReadableChunk<T>(chunks[i], output);
    }

    (*output) << '\n';
}


static void PrintHumanReadableStringChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    IOutputStream* const output) {

    const ui8* data = chunk.Chunk->Quants()->data();
    size_t dataSizeLeft = chunk.Chunk->Quants()->size();
    for (size_t i = 0; i < chunk.DocumentCount; ++i) {
        if (i > 0) {
            (*output) << ' ';
        }

        CB_ENSURE(dataSizeLeft >= sizeof(ui32), LabeledOutput(i, dataSizeLeft));
        const ui32 tokenSize = LittleToHost(ReadUnaligned<ui32>(data));
        data += sizeof(ui32);
        dataSizeLeft -= sizeof(ui32);

        CB_ENSURE(dataSizeLeft >= tokenSize, LabeledOutput(i, dataSizeLeft, tokenSize));
        (*output) << TStringBuf(reinterpret_cast<const char*>(data), tokenSize);
        data += tokenSize;
        dataSizeLeft -= tokenSize;
    }
}

static void PrintHumanReadableStringChunks(
    const NCB::TQuantizedPool& pool,
    const EColumn columnType,
    const bool chunkWise,
    IOutputStream* const output) {

    ui32 localIndex = 0;
    if (columnType == EColumn::SampleId) {
        localIndex = pool.StringDocIdLocalIndex;
    } else if (columnType == EColumn::GroupId) {
        localIndex = pool.StringGroupIdLocalIndex;
    } else if (columnType == EColumn::SubgroupId) {
        localIndex = pool.StringSubgroupIdLocalIndex;
    } else {
        CB_ENSURE(false, LabeledOutput(columnType) << "; Should be one of: " << EColumn::SampleId << ", " << EColumn::GroupId << ", " << EColumn::SubgroupId);
    }

    const auto chunks = GetChunksSortedByOffset(pool.Chunks[localIndex]);

    if (chunkWise) {
        for (const auto& chunk : chunks) {
            (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
            PrintHumanReadableStringChunk(chunk, output);
            (*output) << '\n';
        }
        return;
    }

    for (size_t i = 0, iEnd = chunks.size(); i < iEnd; ++i) {
        if (i > 0) {
            (*output) << ' ';
        }

        PrintHumanReadableStringChunk(chunks[i], output);
    }

    (*output) << '\n';
}

static void PrintHumanReadable(
    const NCB::TQuantizedPool& pool,
    const NCB::TPrintQuantizedPoolParameters& params,
    IOutputStream* const output) {

    const auto resolveBorders = params.ResolveBorders;
    const auto chunkWise = params.Format == NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise;

    (*output) << pool.ColumnIndexToLocalIndex.size() << ' ' << pool.DocumentCount << '\n';

    const auto columnIndices = CollectAndSortKeys(pool.ColumnIndexToLocalIndex);
    size_t featureIndex = std::numeric_limits<size_t>::max();
    for (const auto columnIndex : columnIndices) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(columnIndex);
        const auto columnType = pool.ColumnTypes.at(localIndex);
        featureIndex += columnType == EColumn::Num || columnType == EColumn::Categ;
        const auto chunks = GetChunksSortedByOffset(pool.Chunks.at(localIndex));

        const auto& floatFeatureIndexToSchema = pool.QuantizationSchema.GetFeatureIndexToSchema();
        const auto& catFeatureIndexToSchema = pool.QuantizationSchema.GetCatFeatureIndexToSchema();
        if ((columnType == EColumn::Num && floatFeatureIndexToSchema.count(featureIndex) == 0) ||
            (columnType == EColumn::Categ && catFeatureIndexToSchema.count(featureIndex) == 0)) {
            // The feature schema is missing when the feature is ignored.
            continue;
        }

        (*output)
            << columnIndex << ' '
            << columnType << ' '
            << chunks.size();

        if (columnType == EColumn::Num || columnType == EColumn::Categ) {
            (*output) << ' ' << featureIndex;
        }

        (*output) << '\n';

        if (columnType == EColumn::Sparse || columnType == EColumn::Auxiliary) {
            CB_ENSURE(chunks.empty(), LabeledOutput(columnIndex));
            continue;
        } if (columnType == EColumn::Num) {
            const auto& quantizationSchema = floatFeatureIndexToSchema.at(featureIndex);
            PrintHumanReadableNumericChunks(
                chunks,
                resolveBorders ? &quantizationSchema : nullptr,
                chunkWise,
                output);
        } else if (columnType == EColumn::Categ) { //TODO(ivankozlov98) write print catfeature schema
            PrintHumanReadableNumericChunks(
                chunks,
                nullptr,
                chunkWise,
                output);
        } else if (IsRequiredColumn(columnType)) {
            if (IsFloatColumn(columnType)) {
                PrintHumanReadableChunks<float>(chunks, chunkWise, output);
            } else if (IsDoubleColumn(columnType)) {
                PrintHumanReadableChunks<double>(chunks, chunkWise, output);
            } else if (IsUi32Column(columnType)) {
                PrintHumanReadableChunks<ui32>(chunks, chunkWise, output);
            } else if (IsUi64Column(columnType)) {
                PrintHumanReadableChunks<ui64>(chunks, chunkWise, output);
            }

            if (pool.HasStringColumns && IsStringColumn(columnType)) {
                PrintHumanReadableStringChunks(pool, columnType, chunkWise, output);
            }
        } else {
            ythrow TCatBoostException() << "unexpected " << LabeledOutput(columnType, columnIndex);
        }
    }
}

void NCB::PrintQuantizedPool(
    const NCB::TQuantizedPool& pool,
    const NCB::TPrintQuantizedPoolParameters& params,
    IOutputStream* const output) {

    switch (params.Format) {
        case NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise:
        case NCB::EQuantizedPoolPrintFormat::HumanReadableColumnWise:
            ::PrintHumanReadable(pool, params, output);
            return;
        case NCB::EQuantizedPoolPrintFormat::Unknown:
            break;
    }

    ythrow TCatBoostException() << "unknown format";
}
