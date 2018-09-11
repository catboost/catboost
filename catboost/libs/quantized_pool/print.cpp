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
#include <util/stream/output.h>
#include <util/system/unaligned_mem.h>
#include <util/system/byteorder.h>

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
        * sizeof(*chunk.Quants()->begin())
        * chunk.Quants()->size()
        / static_cast<size_t>(chunk.BitsPerDocument());
}

static void PrintHumanReadableNumericChunk(
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

        if (schema) {
            const auto index = borderIndexIt.Cur();
            if (index >= schema->GetBorders().size()) {
                (*output) << '>' << *schema->GetBorders().rbegin();
            } else {
                (*output) << '<' << schema->GetBorders(borderIndexIt.Cur());
            }
        } else {
            // `operator <<` treats `ui8` as character, so we force it to treat it like a number
            (*output) << static_cast<ui64>(borderIndexIt.Cur());
        }
    }

    (*output) << '\n';
}

template <typename T>
static void PrintHumanReadableChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    IOutputStream* const output) {

    CB_ENSURE(static_cast<size_t>(chunk.Chunk->BitsPerDocument()) == sizeof(T) * 8);
    CB_ENSURE(chunk.DocumentCount <= GetMaxFeatureCountInChunk(*chunk.Chunk));

    TUnalignedMemoryIterator<T> it(
        chunk.Chunk->Quants()->data(),
        chunk.Chunk->Quants()->size());
    for (size_t i = 0; i < chunk.DocumentCount; ++i, (void)it.Next()) {
        if (i > 0) {
            (*output) << ' ';
        }

        (*output) << it.Cur();
    }

    (*output) << '\n';
}

static void PrintHumanReadableStringChunk(
    const NCB::TQuantizedPool::TChunkDescription& chunk,
    IOutputStream* const output) {

    CB_ENSURE(chunk.DocumentCount <= GetMaxFeatureCountInChunk(*chunk.Chunk));

    const ui8* data = chunk.Chunk->Quants()->data();
    size_t dataSizeLeft = chunk.Chunk->Quants()->size();
    for (size_t i = 0; i < chunk.DocumentCount; ++i) {
        if (i > 0) {
            (*output) << ' ';
        }

        CB_ENSURE(dataSizeLeft >= sizeof(ui32));
        const ui32 tokenSize = LittleToHost(ReadUnaligned<ui32>(data));
        data += sizeof(ui32);
        dataSizeLeft -= sizeof(ui32);

        CB_ENSURE(dataSizeLeft >= tokenSize);
        (*output) << TStringBuf(reinterpret_cast<const char*>(data), tokenSize);
        data += tokenSize;
        dataSizeLeft -= tokenSize;
    }

    (*output) << '\n';
}

static void PrintHumanReadableStringColumn(
    const NCB::TQuantizedPool& pool,
    const EColumn columnType,
    IOutputStream* const output) {

    ui32 localIndex = 0;
    if (columnType == EColumn::DocId) {
        localIndex = pool.StringDocIdLocalIndex;
    } else if (columnType == EColumn::GroupId) {
        localIndex = pool.StringGroupIdLocalIndex;
    } else if (columnType == EColumn::SubgroupId) {
        localIndex = pool.StringSubgroupIdLocalIndex;
    } else {
        CB_ENSURE(false, "Bad column type. Should be one of: DocId, GroupId, SubgroupId.");
    }

    for (const auto& chunk : GetChunksSortedByOffset(pool.Chunks[localIndex])) {
        PrintHumanReadableStringChunk(chunk, output);
    }
}

static void PrintHumanReadable(
    const NCB::TQuantizedPool& pool,
    const NCB::TPrintQuantizedPoolParameters&,
    const bool resolveBorders,
    IOutputStream* const output) {

    (*output) << pool.ColumnIndexToLocalIndex.size() << ' ' << pool.DocumentCount << '\n';

    const auto columnIndices = CollectAndSortKeys(pool.ColumnIndexToLocalIndex);
    size_t featureIndex = std::numeric_limits<size_t>::max();
    for (const auto columnIndex : columnIndices) {
        const auto localIndex = pool.ColumnIndexToLocalIndex.at(columnIndex);
        const auto columnType = pool.ColumnTypes.at(localIndex);
        featureIndex += columnType == EColumn::Num || columnType == EColumn::Categ;
        const auto chunks = GetChunksSortedByOffset(pool.Chunks.at(localIndex));

        const auto& featureIndexToSchema = pool.QuantizationSchema.GetFeatureIndexToSchema();
        if (columnType == EColumn::Num && featureIndexToSchema.count(featureIndex) == 0) {
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

        if (columnType == EColumn::Categ || columnType == EColumn::Sparse || columnType == EColumn::Auxiliary) {
            CB_ENSURE(chunks.empty());
            continue;
        } if (columnType == EColumn::Num) {
            const auto& quantizationSchema = featureIndexToSchema.at(featureIndex);
            for (const auto& chunk : chunks) {
                (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
                PrintHumanReadableNumericChunk(chunk, resolveBorders ? &quantizationSchema : nullptr, output);
            }
        } else if (IsRequiredColumn(columnType)) {
            for (const auto& chunk : chunks) {
                (*output) << chunk.DocumentOffset << ' ' << chunk.DocumentCount << '\n';
                if (IsFloatColumn(columnType)) {
                    PrintHumanReadableChunk<float>(chunk, output);
                } else if (IsDoubleColumn(columnType)) {
                    PrintHumanReadableChunk<double>(chunk, output);
                } else if (IsUi32Column(columnType)) {
                    PrintHumanReadableChunk<ui32>(chunk, output);
                } else if (IsUi64Column(columnType)) {
                    PrintHumanReadableChunk<ui64>(chunk, output);
                }
            }
            if (pool.HasStringColumns && IsStringColumn(columnType)) {
                PrintHumanReadableStringColumn(pool, columnType, output);
            }
        } else {
            ythrow TCatboostException() << "unexpected column type " << columnType;
        }
    }
}

void NCB::PrintQuantizedPool(
    const NCB::TQuantizedPool& pool,
    const NCB::TPrintQuantizedPoolParameters& params,
    IOutputStream* const output) {

    switch (params.Format) {
        case NCB::EQuantizedPoolPrintFormat::HumanReadable:
            ::PrintHumanReadable(pool, params, false, output);
            return;
        case NCB::EQuantizedPoolPrintFormat::HumanReadableResolveBorders:
            ::PrintHumanReadable(pool, params, true, output);
            return;
        case NCB::EQuantizedPoolPrintFormat::Unknown:
            break;
    }

    ythrow TCatboostException() << "unknown format";
}
