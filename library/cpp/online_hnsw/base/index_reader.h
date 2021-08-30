#pragma once

#include <util/memory/blob.h>
#include <util/generic/vector.h>

namespace NOnlineHnsw {
    class TOnlineHnswIndexReader {
    public:
        void ReadIndex(const TBlob& blob, TVector<ui32>* numNeighborsInLevels, TVector<const ui32*>* levels) const;
    };
} // namespace NOnlineHnsw
