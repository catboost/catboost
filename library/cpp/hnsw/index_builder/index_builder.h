#pragma once

#include "build_routines.h"
#include "index_data.h"
#include "internal_build_options.h"

namespace NHnsw {
    struct THnswBuildOptions;

    /**
 * @brief Method for building HNSW indexes.
 *
 * The easiest way to use it, is to define a custom TDistance class,
 * that has TResult and TLess defined.
 * If you do so then building your index is as simple as:
 * @code
 *   THnswIndexData indexData = BuildIndex<TDistance>(opts, itemStorage);
 * @endcode
 *
 * TItemStorage must provide one typedef:
 * - TItem
 * and two methods:
 * - GetItem(size_t id) - typically returning TItem or const TItem&
 * - size_t GetNumItems()
 *
 * Please, refer to hnsw/ut/main.cpp for a comprehensive usage example.
 */
    template <class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess,
              class TItemStorage>
    THnswIndexData BuildIndex(const THnswBuildOptions& opts,
                              const TItemStorage& itemStorage,
                              const TDistance& distance = {},
                              const TDistanceLess& distanceLess = {}) {
        TDistanceTraits<TDistance, TDistanceResult, TDistanceLess> distanceTraits(distance, distanceLess);
        const THnswInternalBuildOptions internalOpts(opts);
        return BuildIndexWithTraits(internalOpts, distanceTraits, itemStorage);
    }

    template <class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess,
              class TItemStorage>
    THnswIndexData BuildForUpdatesIndex(const THnswBuildOptions& opts,
                                        const TItemStorage& itemStorage,
                                        const TDistance& distance = {},
                                        const TDistanceLess& distanceLess = {}) {
        Y_ENSURE(opts.SnapshotFile != "" || opts.SnapshotBlobPtr, "SnapshotFile is empty and SnapshotBlobPtr == nullptr");
        TDistanceTraits<TDistance, TDistanceResult, TDistanceLess> distanceTraits(distance, distanceLess);
        const THnswInternalBuildOptions internalOpts(opts);
        return BuildForUpdatesIndexWithTraits(internalOpts, distanceTraits, itemStorage);
    }
}
