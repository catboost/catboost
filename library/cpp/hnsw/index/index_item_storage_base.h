#pragma once

#include "index_base.h"

namespace NHnsw {
    /**
     * @brief Base class for HnswIndex.
     * Inheritance is based on CRTP.
     * If item storage is created outside your class, see index_base.h.
     *
     * @code
     *    class THnswIndex: public THnswItemStorageIndexBase<THnswIndex> {
     *        using TBase = THnswIndexBase<THnswIndex>;
     *    public:
     *        THnswIndex(const TBlob& indexBlob, ...)
     *            : TBase(indexBlob)
     *            , ...
     *        {
     *        }
     *        // In order to perform searches THnswIndex must implement this single method.
     *        const TItem& GetItem(ui32 id) const {
     *            return ...;
     *        }
     *    };
     * @endcode
     *
     * Please, refer to hnsw/ut/main.cpp for a comprehensive usage example.
     */

    template <class TItemStorage>
    class THnswItemStorageIndexBase: protected  THnswIndexBase {
    public:
        using THnswIndexBase::THnswIndexBase;

        template <class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            size_t topSize,
            size_t searchNeighborhoodSize,
            size_t distanceCalcLimit,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {}) const
        {
            return THnswIndexBase::GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query,
                topSize,
                searchNeighborhoodSize,
                distanceCalcLimit,
                static_cast<const TItemStorage&>(*this),
                distance,
                distanceLess);
        }

        template <class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            size_t topSize,
            size_t searchNeighborhoodSize,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {}) const
        {
            return THnswIndexBase::GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query,
                topSize,
                searchNeighborhoodSize,
                Max<size_t>(),
                static_cast<const TItemStorage&>(*this),
                distance,
                distanceLess);
        }
    };

}
