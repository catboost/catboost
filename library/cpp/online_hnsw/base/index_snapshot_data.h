#pragma once

#include <library/cpp/hnsw/tools/build_dense_vector_index/distance.h>
#include <library/cpp/online_hnsw/base/build_options.h>

#include <util/generic/vector.h>
#include <util/ysaveload.h>

namespace NOnlineHnsw {
    template <class TDistanceResult>
    struct TOnlineHnswIndexSnapshot {
        struct TDynamicDenseGraphSnapshot {
            ui32 MaxNeighborCount;
            ui32 MaxSize;
            ui32 Size;
            ui32 NeighborCount;
            TVector<TDistanceResult> Distances;
            TVector<ui32> Ids;

            Y_SAVELOAD_DEFINE(
                MaxNeighborCount,
                MaxSize,
                Size,
                NeighborCount,
                Distances,
                Ids);
        };

        TOnlineHnswBuildOptions Options;
        TVector<TDynamicDenseGraphSnapshot> Levels;
        TVector<ui32> LevelSizes;
        TVector<ui32> DiverseNeighborsNums;

        Y_SAVELOAD_DEFINE(
            Options,
            Levels,
            LevelSizes,
            DiverseNeighborsNums);
    };
} // namespace NOnlineHnsw
