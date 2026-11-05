#pragma once

#include "index_base.h"

#include <utility>

namespace NHnsw {
    /**
     * @brief Base class for HnswIndex.
     * Inheritance is based on CRTP.
     * If item storage is created outside your class, see index_base.h.
     *
     * @code
     *    using TGraphIndex = THnswIndexCompressed<TCustomGraphCodec>;
     *    class THnswIndex: public THnswItemStorageIndexBaseImpl<THnswIndex, TGraphIndex> {
     *        using TBase = THnswItemStorageIndexBaseImpl<THnswIndex, TGraphIndex>;
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

    template <class TItemStorage, class TIndexBase>
    class THnswItemStorageIndexBaseImpl: protected TIndexBase {
    public:
        using TIndexBase::TIndexBase;

        template <class TDistanceResult>
        using TNeighbor = typename TIndexBase::template TNeighbor<TDistanceResult>;

        template <class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem,
                  class TFilter = TDefaultFilter,
                  class TSearchContext = TDefaultSearchContext>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            const TSearchParameters& params,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {},
            TFilter&& filter = {},
            TSearchContext&& context = {}) const
        {
            return TIndexBase::template GetNearestNeighbors<
                TItemStorage,
                TDistance,
                TDistanceResult,
                TDistanceLess,
                TItem,
                TFilter,
                TSearchContext>(
                    query,
                    params,
                    static_cast<const TItemStorage&>(*this),
                    distance,
                    distanceLess,
                    std::forward<TFilter>(filter),
                    std::forward<TSearchContext>(context));
        }

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
            const TDistanceLess& distanceLess = {},
            const size_t stopSearchSize = 1,
            const EFilterMode filterMode = EFilterMode::NO_FILTER,
            const TFilterBase& filter = {},
            const size_t filterCheckLimit = Max<size_t>()) const
        {
            return TIndexBase::template GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query,
                topSize,
                searchNeighborhoodSize,
                distanceCalcLimit,
                static_cast<const TItemStorage&>(*this),
                distance,
                distanceLess,
                stopSearchSize,
                filterMode,
                filter,
                filterCheckLimit);
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
            const TDistanceLess& distanceLess = {},
            const size_t stopSearchSize = 1,
            const EFilterMode filterMode = EFilterMode::NO_FILTER,
            const TFilterBase& filter = {},
            const size_t filterCheckLimit = Max<size_t>()) const
        {
            return TIndexBase::template GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query,
                topSize,
                searchNeighborhoodSize,
                Max<size_t>(),
                static_cast<const TItemStorage&>(*this),
                distance,
                distanceLess,
                stopSearchSize,
                filterMode,
                filter,
                filterCheckLimit);
        }
    };

}
