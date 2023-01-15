#include "index_writer.h"

#include <util/stream/file.h>


namespace NOnlineHnsw {
    size_t ExpectedSize(const TOnlineHnswIndexData& index) {
        return sizeof(index.MaxNeighbors)
            + sizeof(ui32)
            + sizeof(decltype(index.LevelSizes)::value_type) * index.LevelSizes.size()
            + sizeof(decltype(index.FlatLevels)::value_type) * index.FlatLevels.size();
    }

    void WriteIndex(const TOnlineHnswIndexData& index, IOutputStream& out) {
        out.Write(&index.MaxNeighbors, sizeof(index.MaxNeighbors));
        ui32 numLevels = index.LevelSizes.size();
        out.Write(&numLevels, sizeof(ui32));
        out.Write(index.LevelSizes.data(), sizeof(decltype(index.LevelSizes)::value_type) * index.LevelSizes.size());
        out.Write(index.FlatLevels.data(), sizeof(decltype(index.FlatLevels)::value_type) * index.FlatLevels.size());
    }

    void WriteIndex(const TOnlineHnswIndexData& index, const TString& fileName) {
        TFileOutput out(fileName);
        WriteIndex(index, out);
    }
} // namespace NOnlineHnsw
