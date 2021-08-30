#include "index_writer.h"
#include "index_data.h"

#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>


namespace NHnsw {
    void DebugIndexDump(const THnswIndexData& index, IOutputStream& out) {
        out << "Header:"
            << " NumItems=" << index.NumItems
            << " MaxNeighbors=" << index.MaxNeighbors
            << " LevelSizeDecay=" << index.LevelSizeDecay
            << "\n";

        out << "Items dump: \n\n";

        TVector<const ui32*> levels;
        TVector<ui32> numNeighborsInLevels;
        TVector<ui32> numItemsInLevels;
        {
            const ui32* data = index.FlatLevels.begin();
            for (i64 numItems = index.NumItems; numItems > 1; numItems /= index.LevelSizeDecay) {
                Y_ENSURE(data < index.FlatLevels.end());
                levels.push_back(data);
                numNeighborsInLevels.push_back(Min<i64>(index.MaxNeighbors, numItems - 1));
                numItemsInLevels.push_back(numItems);
                data += numItems * numNeighborsInLevels.back();
            }

            Y_ENSURE(data == index.FlatLevels.end());
        }

        for (auto levelNum : xrange<i64>(levels.size() - 1, -1, -1)) {
            for (auto itemId : xrange(numItemsInLevels[levelNum])) {
                out << "At level " << levelNum << " of id " << itemId << ":";
                for (auto neighborId : xrange(numNeighborsInLevels[levelNum])) {
                    out << " " << (levels[levelNum] + itemId * numNeighborsInLevels[levelNum])[neighborId];
                }
                out << "\n";
            }
        }
    }

    size_t ExpectedSize(const THnswIndexData& index) {
        return sizeof(index.NumItems)
            + sizeof(index.MaxNeighbors)
            + sizeof(index.LevelSizeDecay)
            + index.FlatLevels.size() * sizeof(index.FlatLevels[0]);
    }

    void WriteIndex(const THnswIndexData& index, IOutputStream& out) {
        out.Write(&index.NumItems, sizeof(index.NumItems));
        out.Write(&index.MaxNeighbors, sizeof(index.MaxNeighbors));
        out.Write(&index.LevelSizeDecay, sizeof(index.LevelSizeDecay));
        out.Write(index.FlatLevels.data(), index.FlatLevels.size() * sizeof(index.FlatLevels[0]));
    }

    void WriteIndex(const THnswIndexData& index, const TString& outputFilename) {
        TFixedBufferFileOutput out(outputFilename);
        WriteIndex(index, out);
    }

}
