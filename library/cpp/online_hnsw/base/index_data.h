#pragma once

#include <util/generic/vector.h>


namespace NOnlineHnsw {
    struct TOnlineHnswIndexData {
        ui32 MaxNeighbors;
        TVector<ui32> LevelSizes;
        TVector<ui32> FlatLevels;
    };
} // namespace NOnlineHnsw
