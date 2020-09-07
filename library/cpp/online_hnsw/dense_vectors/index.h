#pragma once

#include "item_storage.h"

#include <library/cpp/online_hnsw/base/item_storage_index.h>
#include <library/cpp/hnsw/index_builder/dense_vector_distance.h>


namespace NOnlineHnsw {
    template<class TVectorComponent,
             class TDistance,
             class TDistanceResult = typename TDistance::TResult,
             class TDistanceLess = typename TDistance::TLess>
    class TOnlineHnswDenseVectorIndex: public TOnlineHnswItemStorageIndexBase<TOnlineHnswDenseVectorIndex<TVectorComponent,
                                                                                                          TDistance,
                                                                                                          TDistanceResult,
                                                                                                          TDistanceLess>,
                                                                              const TVectorComponent*,
                                                                              NHnsw::TDistanceWithDimension<TVectorComponent, TDistance>,
                                                                              TDistanceResult,
                                                                              TDistanceLess>
                                     , public TDenseVectorExtendableItemStorage<TVectorComponent> {
        using TIndexBase = TOnlineHnswItemStorageIndexBase<TOnlineHnswDenseVectorIndex<TVectorComponent,
                                                                                       TDistance,
                                                                                       TDistanceResult,
                                                                                       TDistanceLess>,
                                                           const TVectorComponent*,
                                                           NHnsw::TDistanceWithDimension<TVectorComponent, TDistance>,
                                                           TDistanceResult,
                                                           TDistanceLess>;
        using TItemStorageBase = TDenseVectorExtendableItemStorage<TVectorComponent>;
    public:
        using TItem = const TVectorComponent*;

        TOnlineHnswDenseVectorIndex(const NOnlineHnsw::TOnlineHnswBuildOptions& opts,
                                    size_t dimension,
                                    const TDistance& distance = {},
                                    const TDistanceLess& distanceLess = {},
                                    const size_t maxSize = 0)
            : TIndexBase(opts, NHnsw::TDistanceWithDimension<TVectorComponent, TDistance>(distance, dimension), distanceLess)
            , TItemStorageBase(dimension, maxSize)
        {
        }
    };
} // namespace NOnlineHnsw
