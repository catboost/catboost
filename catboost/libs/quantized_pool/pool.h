#pragma once

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

        THashMap<size_t, size_t> TrueFeatureIndexToLocalIndex;
        TDeque<TVector<TChunkDescription>> Chunks;
        TVector<TBlob> Blobs;
    };
}
