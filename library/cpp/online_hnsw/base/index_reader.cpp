#include "index_reader.h"

#include <library/cpp/hnsw/index/layout/index_layout.h>

#include <util/generic/algorithm.h>
#include <util/generic/yexception.h>

#include <climits>

namespace NOnlineHnsw {
    void TOnlineHnswIndexReader::ReadIndex(const TBlob& blob, NHnsw::THnswIndexLayout* layout) const {
        *layout = {};
        if (blob.Empty()) {
            return;
        }

        Y_ENSURE(blob.Size() % sizeof(ui32) == 0, "online hnsw index size is not ui32-aligned");
        const ui32* data = reinterpret_cast<const ui32*>(blob.Begin());
        const size_t numWords = blob.Size() / sizeof(ui32);
        size_t offset = 0;

        Y_ENSURE(numWords >= 2, "online hnsw index blob is too short to hold a header");
        ui32 maxNeighbors = data[offset++];
        Y_ENSURE(maxNeighbors > 0);
        ui32 numLevels = data[offset++];
        Y_ENSURE(numLevels <= numWords - offset, "online hnsw index blob is too short to hold level sizes");

        TVector<ui32> levelSizes(numLevels);
        for (ui32 level = 0; level < numLevels; ++level) {
            levelSizes[level] = data[offset++];
            Y_ENSURE(levelSizes[level] > 0, "online hnsw level must not be empty");
        }

        layout->Format = NHnsw::ENeighborIdFormat::Ui32;
        layout->BitsPerId = 32;
        layout->Payload = reinterpret_cast<const ui8*>(data + offset);
        layout->Levels.reserve(numLevels);

        ui64 bitOffset = 0;
        for (ui32 level = 0; level < numLevels; ++level) {
            const ui32 numNeighbors = Min(maxNeighbors, levelSizes[level] - 1);
            const size_t remainingWords = numWords - offset;
            Y_ENSURE(
                numNeighbors == 0 || levelSizes[level] <= remainingWords / numNeighbors,
                "online hnsw index blob is too short to hold level rows"
            );
            const size_t levelWords = size_t(numNeighbors) * levelSizes[level];
            layout->Levels.push_back({
                .NumNeighbors = numNeighbors,
                .BitOffset = bitOffset,
            });
            offset += levelWords;
            bitOffset += ui64(levelWords) * sizeof(ui32) * CHAR_BIT;
        }

        Y_ENSURE(offset == numWords, "online hnsw index size does not match its level geometry");
    }
} // namespace NOnlineHnsw
