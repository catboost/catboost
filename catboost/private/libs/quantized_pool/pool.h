#pragma once

#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/column_description/column.h>

#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/memory/blob.h>
#include <util/system/types.h>

namespace NCB {
    namespace NIdl {
        struct TQuantizedFeatureChunk;
    }
}

namespace NCB {
    struct TQuantizedPool {
        struct TChunkDescription {
            size_t DocumentOffset = 0;
            size_t DocumentCount = 0;
            const NIdl::TQuantizedFeatureChunk* Chunk = nullptr;

            TChunkDescription() = default;
            TChunkDescription(
                size_t documentOffset,
                size_t documentCount,
                const NIdl::TQuantizedFeatureChunk* chunk)
                : DocumentOffset(documentOffset)
                , DocumentCount(documentCount)
                , Chunk(chunk) {
            }
        };

        size_t DocumentCount = 0;
        // Maps feature column index in original pool to indices used in this structure.
        //
        // Example: `TrueFeatureIndexToLocalIndex = {{1, 0}, {5, 1}}` -- then all info about feature
        // in column 5 will be present in `Chunks[1]`.
        //
        THashMap<size_t, size_t> ColumnIndexToLocalIndex;
        bool HasStringColumns = false;
        ui32 StringDocIdLocalIndex = -1;
        ui32 StringGroupIdLocalIndex = -1;
        ui32 StringSubgroupIdLocalIndex = -1;
        // TODO(yazevnul): replace with native C++ `TPoolQuantizationSchema`
        NIdl::TPoolQuantizationSchema QuantizationSchema;
        TVector<EColumn> ColumnTypes;
        TVector<TString> ColumnNames;
        TDeque<TVector<TChunkDescription>> Chunks;
        // TODO(yazevnul): add convenient interface:
        //     TChunkIterator GetChunksByColumnIndex(...);
        //     TChunkIterator GetChunksByFlatFeatureIndex(...);
        //     TChunkIterator GetChunksByNumericFeatureIndex(...);

        TVector<size_t> IgnoredColumnIndices;

        TVector<TBlob> Blobs;
        TVector<TVector<ui8>> ChunkStorage; // non-mapped storage
    };

    struct TQuantizedPoolDigest {
        size_t CategoricFeatureCount = 0;

        size_t NumericFeatureCount = 0;
        size_t NumericFeature1BitCount = 0;
        size_t NumericFeature4BitCount = 0;
        size_t NumericFeature8BitCount = 0;
        size_t NumericFeature16BitCount = 0;

        size_t NonFeatureColumnCount = 0;

        size_t ClassesCount = 0;
        size_t EstimatedIdsLength = 0;
        double EstimatedGroupSize = 0;
        double EstimatedSqrGroupSize = 0;
        size_t EstimatedMaxGroupSize = 0;
    };
}
