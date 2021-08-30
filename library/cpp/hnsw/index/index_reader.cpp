#include "index_reader.h"

#include <util/generic/yexception.h>

namespace NHnsw {
   void THnswIndexReader::ReadIndex(const TBlob& blob,
                                    TVector<ui32>* numNeighborsInLevels,
                                    TVector<const ui32*>* levels) const {
        if (blob.Empty()) {
            return;
        }

        const ui32* data = reinterpret_cast<const ui32*>(blob.Begin());
        const ui32* const end = reinterpret_cast<const ui32*>(blob.End());
        ui32 numItems = *data++;
        const ui32 maxNeighbors = *data++;
        const ui32 levelSizeDecay = *data++;

        Y_ENSURE(levelSizeDecay > 1, "levelSizeDecay should be greater than 1");
        if (numItems == 1) {
            levels->push_back(data);
            numNeighborsInLevels->push_back(0);
        } else {
            for (; numItems > 1; numItems /= levelSizeDecay) {
                Y_ENSURE(data < end);
                levels->push_back(data);
                numNeighborsInLevels->push_back(Min(maxNeighbors, numItems - 1));
                data += size_t(numItems) * numNeighborsInLevels->back();
            }
        }

        Y_ENSURE(data == end);
    }
} // namespace Hnsw
