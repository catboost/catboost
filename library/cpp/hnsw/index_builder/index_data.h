#pragma once

#include "internal_build_options.h"

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NHnsw {
    struct THnswIndexData {
        ui32 NumItems;
        ui32 MaxNeighbors;
        ui32 LevelSizeDecay;
        TVector<ui32> FlatLevels;
    };

    template <class TLevels>
    THnswIndexData ConstructIndexData(const THnswInternalBuildOptions& opts, const TLevels& levels) {
        THnswIndexData index;
        index.NumItems = levels.empty() ? 0 : levels.front().GetSize();
        index.MaxNeighbors = opts.MaxNeighbors;
        index.LevelSizeDecay = opts.LevelSizeDecay;

        size_t elems = 0;
        for (const auto& level : levels) {
            elems += level.GetSize() * level.GetNeighborsCount();
        }
        index.FlatLevels.reserve(elems);

        for (const auto& level : levels) {
            index.FlatLevels.insert(index.FlatLevels.end(), level.GetIds().begin(), level.GetIds().end());
        }
        return index;
    }

}
