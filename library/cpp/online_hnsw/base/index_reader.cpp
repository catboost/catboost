#include "index_reader.h"

#include <util/generic/algorithm.h>
#include <util/generic/yexception.h>


namespace NOnlineHnsw {
    void TOnlineHnswIndexReader::ReadIndex(const TBlob& blob,
                                           TVector<ui32>* numNeighborsInLevels,
                                           TVector<const ui32*>* levels) const {
        const ui32* data = reinterpret_cast<const ui32*>(blob.Begin());
        const ui32* const end = reinterpret_cast<const ui32*>(blob.End());
        ui32 maxNeighbors = *data++;
        Y_ENSURE(maxNeighbors > 0);
        ui32 numLevels = *data++;

        TVector<ui32> levelSizes(numLevels);
        numNeighborsInLevels->resize(numLevels);
        for (ui32 level = 0; level < numLevels; ++level) {
            levelSizes[level] = *data++;
            (*numNeighborsInLevels)[level] = Min(maxNeighbors, levelSizes[level] - 1);
        }

        levels->resize(numLevels);
        for (ui32 level = 0; level < numLevels; ++level) {
            (*levels)[level] = data;
            data += size_t((*numNeighborsInLevels)[level]) * levelSizes[level];
        }

        Y_ENSURE(data == end);
    }
} // namespace NOnlineHnsw
