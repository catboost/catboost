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
            size_t DocumentOffset{0};
            size_t DocumentCount{0};
            const NIdl::TQuantizedFeatureChunk* Chunk{nullptr};

            TChunkDescription() = default;
            TChunkDescription(
                size_t documentOffset,
                size_t documentCount,
                const NIdl::TQuantizedFeatureChunk* chunk)
                : DocumentOffset{documentOffset}
                , DocumentCount{documentCount}
                , Chunk{chunk} {
            }
        };

        // Maps feature column index in original pool to indices used in this structure.
        //
        // Example: `TrueFeatureIndexToLocalIndex = {{1, 0}, {5, 1}}` -- then all info about feature
        // in column 5 will be present in `Chunks[1]`.
        //
        THashMap<size_t, size_t> TrueFeatureIndexToLocalIndex;
        NIdl::TPoolQuantizationSchema QuantizationSchema;
        TVector<EColumn> ColumnTypes;
        TDeque<TVector<TChunkDescription>> Chunks;
        TVector<TBlob> Blobs;
    };

    struct TQuantizedPoolDigest {
        struct TChunkDescription {
            size_t DocumentOffset{0};
            size_t DocumentCount{0};
            size_t SizeInBytes{0};

            TChunkDescription() = default;
            TChunkDescription(
                size_t documentOffset,
                size_t documentCount,
                size_t sizeInBytes)
                : DocumentOffset{documentOffset}
                , DocumentCount{documentCount}
                , SizeInBytes{sizeInBytes} {
            }
        };

        // Maps feature column index in original pool to indices used in this structure.
        //
        // Example: `TrueFeatureIndexToLocalIndex = {{1, 0}, {5, 1}}` -- then all info about feature
        // in column 5 will be present in `Chunks[1]`, `DocumentCount[1]` and `ChunkSizeInBytesSums[1]`
        //
        THashMap<size_t, size_t> TrueFeatureIndexToLocalIndex;
        TDeque<TVector<TChunkDescription>> Chunks;
        TVector<size_t> DocumentCount;
        TVector<size_t> ChunkSizeInBytesSums;
    };
}
