#pragma once

#include "index_reader.h"
#include "filter_base.h"
#include "neighbors_getter.h"

#include <library/cpp/hnsw/helpers/distance.h>
#include <library/cpp/hnsw/helpers/is_item_marked_deleted.h>
#include <library/cpp/hnsw/helpers/search_context.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/queue.h>
#include <util/memory/blob.h>

namespace NHnsw {
    /**
     * @brief Parameters for HNSW search configuration.
     *
     * @param TopSize                 The search will return at most this much nearest items.
     * @param SearchNeighborhoodSize  Size of the dynamic candidate list (ef).
     *                                Increasing this makes search slower but more accurate.
     * @param DistanceCalcLimit       Limit on the number of distance computations.
     * @param StopSearchSize          Minimum number of nearest neighbors found
     *                                before termination conditions are evaluated.
     * @param FilterMode              Filtering mode in HNSW, no filtration by default.
     * @param FilterCheckLimit        Limit on the number of times the filter Check() method is called.
     */
    struct TSearchParameters {
        size_t TopSize;
        size_t SearchNeighborhoodSize;
        size_t DistanceCalcLimit = Max<size_t>();
        size_t StopSearchSize = 1;
        EFilterMode FilterMode = EFilterMode::NO_FILTER;
        size_t FilterCheckLimit = Max<size_t>();
    };

    /**
     * This class uses ItemStorage created outside successor of this class.
     * If you don't need separate ItemStorage see index_item_storage_base.h.
     */

    class THnswIndexBase {
    public:
        template <class TDistanceResult>
        struct TNeighbor {
            TDistanceResult Dist;
            ui32 Id;
        };

    public:
        template <class TIndexReader = THnswIndexReader>
        explicit THnswIndexBase(const TString& filename, const TIndexReader& indexReader = TIndexReader())
            : THnswIndexBase(TBlob::PrechargedFromFile(filename), indexReader)
        {
        }
        template <class TIndexReader = THnswIndexReader>
        explicit THnswIndexBase(const TBlob& blob, const TIndexReader& indexReader = TIndexReader()) {
            Reset<TIndexReader>(blob, indexReader);
        }

        /**
         * @brief Method for searching HNSW in index.
         *
         * @param query             Query item for which nearest neighbors are retrieved.
         * @param params            Structure containing search constraints and performance tuning.
         * @param itemStorage       Storage providing GetItem(ui32 id) for the index.
         * @param distance          Distance metric implementation
         * @param distanceLess      Comparator for distance results
         * @param filter            Optional class for item-level filtering.
         * @param externalContext   Optional pointer to a reusable TSearchContext.
         *                          Allows reusing internal buffers across queries to eliminate allocation overhead
         */
        template <class TItemStorage,
                  class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem,
                  class TSearchContext = TDefaultSearchContext>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            const TSearchParameters& params,
            const TItemStorage& itemStorage,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {},
            const TFilterBase& filter = {},
            TSearchContext* externalContext = nullptr) const
        {
            if (Levels.empty() || params.SearchNeighborhoodSize == 0 || params.StopSearchSize == 0) {
                return {};
            }

            TMaybe<TSearchContext> localContextStorage;
            TSearchContext& context = externalContext ? *externalContext : localContextStorage.ConstructInPlace();

            ui32 entryId = 0;
            NPrivate::TDistanceAdapter<TDistance, TDistanceResult, TItemStorage, TItem> distanceAdapter(
                itemStorage, distance, params.DistanceCalcLimit);
            auto entryDist = distanceAdapter.Calc(query, entryId);
            for (ui32 level = GetNumLevels(); level-- > 1 && !distanceAdapter.IsLimitReached(); ) {
                for (bool entryChanged = true; entryChanged && !distanceAdapter.IsLimitReached(); ) {
                    entryChanged = false;
                    const TNeighborsView neighbors(GetNeighbors(level, entryId), GetNumNeighbors(level));
                    distanceAdapter.Prefetch(neighbors);
                    for (ui32 id: neighbors) {
                        if (distanceAdapter.IsLimitReached()) {
                            break;
                        }
                        const auto distToQuery = distanceAdapter.Calc(query, id);
                        if (distanceLess(distToQuery, entryDist)) {
                            entryDist = distToQuery;
                            entryId = id;
                            entryChanged = true;
                        }
                    }
                }
            }

            using TResultItem = TNeighbor<TDistanceResult>;
            auto neighborLess = [&distanceLess](const TResultItem& a, const TResultItem& b) {
                return distanceLess(a.Dist, b.Dist);
            };
            auto neighborGreater = [&neighborLess](const TResultItem& a, const TResultItem& b) {
                return neighborLess(b, a);
            };

            TPriorityQueue<TResultItem, TVector<TResultItem>, decltype(neighborLess)> nearest(neighborLess);
            nearest.Container().reserve(params.SearchNeighborhoodSize + 1);
            TPriorityQueue<TResultItem, TVector<TResultItem>, decltype(neighborGreater)> candidates(neighborGreater);

            TFilterWithLimit filterWithLimit(filter, params.FilterCheckLimit);
            auto neighborsGetter = CreateNeighborsGetter(params.FilterMode, filterWithLimit, context);

            if (!NPrivate::IsItemMarkedDeleted(itemStorage, entryId) && (params.FilterMode == EFilterMode::NO_FILTER || filterWithLimit.Check(entryId))) {
                nearest.push({entryDist, entryId});
            }

            candidates.push({entryDist, entryId});
            context.TryMarkVisited(entryId);

            while (!candidates.empty() && !distanceAdapter.IsLimitReached() && (params.FilterMode == EFilterMode::NO_FILTER || !filterWithLimit.IsLimitReached())) {
                auto cur = candidates.top();
                candidates.pop();
                if (nearest.size() >= params.StopSearchSize && distanceLess(nearest.top().Dist, cur.Dist)) {
                    break;
                }
                const auto neighbors = neighborsGetter->GetLayerNeighbors(cur.Id);
                distanceAdapter.Prefetch(neighbors);
                for (ui32 id: neighbors) {
                    if (distanceAdapter.IsLimitReached()) {
                        break;
                    }
                    const auto distToQuery = distanceAdapter.Calc(query, id);
                    if (nearest.size() < params.SearchNeighborhoodSize || distanceLess(distToQuery, nearest.top().Dist)) {
                        candidates.push({distToQuery, id});

                        if (NPrivate::IsItemMarkedDeleted(itemStorage, id)) {
                            continue;
                        }

                        if (params.FilterMode == EFilterMode::FILTER_NEAREST) {
                            if (filterWithLimit.IsLimitReached()) {
                                break;
                            }
                            if (!filterWithLimit.Check(id)) {
                                continue;
                            }
                        }

                        nearest.push({distToQuery, id});
                        if (nearest.size() > params.SearchNeighborhoodSize) {
                            nearest.pop();
                        }
                    }
                }
            }

            while (nearest.size() > params.TopSize) {
                nearest.pop();
            }
            TVector<TResultItem> result;
            result.reserve(nearest.size());
            for (; !nearest.empty(); nearest.pop()) {
                result.push_back(nearest.top());
            }
            std::reverse(result.begin(), result.end());
            return result;
        }

        /**
         * @brief Method for searching HNSW in index.
         * See FindApproximateNeighbors from `../index_builder/build_routines.h` for algo details.
         * The easiest way to use it, is to define a custom TDistance class,
         * that has TResult and TLess defined.
         * If you do so then searching is as simple as:
         * @code
         *   auto results = index.GetNearestNeighbors<TDistance>(item, topSize, searchNeighborhoodSize, maxCandidatesToCheck);
         * @endcode
         *
         * @param query                     Nearest neighbors for this item will be retrieved.
         * @param topSize                   The search will return at most this much nearest items.
         * @param searchNeighborhoodSize    Increasing this value makes the search slower but more accurate.
         *                                  Typically, search time depends linearly on this param.
         *                                  If the value is too low search could return less than topSize results.
         * @param distanceCalcLimit         Limit of distance calculations.
         * @param itemStorage               Storage with method GetItem(ui32 id) which provides item with given id.
         * @param stopSearchSize            Minimum number of nearest neighbors at which to stop search if
         *                                  the best from candidates is worse than the worst of nearest neighbors
         * @param filterMode                Filtering mode in HNSW, no filtration by default
         * @param filter                    Class with Check(id) method that returns true if an item passes the filter
         * @param filterCheckLimit          Limit of the number of items for which filters are checked
         */
        template <class TItemStorage,
                  class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            size_t topSize,
            size_t searchNeighborhoodSize,
            size_t distanceCalcLimit,
            const TItemStorage& itemStorage,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {},
            const size_t stopSearchSize = 1,
            const EFilterMode filterMode = EFilterMode::NO_FILTER,
            const TFilterBase& filter = {},
            const size_t filterCheckLimit = Max<size_t>()) const
        {
            const TSearchParameters params = {
                .TopSize = topSize,
                .SearchNeighborhoodSize = searchNeighborhoodSize,
                .DistanceCalcLimit = distanceCalcLimit,
                .StopSearchSize = stopSearchSize,
                .FilterMode = filterMode,
                .FilterCheckLimit = filterCheckLimit,
            };
            return GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query, params, itemStorage, distance, distanceLess, filter);
        }

        /**
         * @brief Method for searching HNSW in index.
         * The easiest way to use it, is to define a custom TDistance class,
         * that has TResult and TLess defined.
         * If you do so then searching is as simple as:
         * @code
         *   auto results = index.GetNearestNeighbors<TDistance>(item, topSize, searchNeighborhoodSize);
         * @endcode
         *
         * @param query                     Nearest neighbors for this item will be retrieved.
         * @param topSize                   The search will return at most this much nearest items.
         * @param searchNeighborhoodSize    Increasing this value makes the search slower but more accurate.
         *                                  Typically, search time depends linearly on this param.
         *                                  If the value is too low search could return less than topSize results.
         * @param itemStorage               Storage with method GetItem(ui32 id) which provides item with given id.
         * @param stopSearchSize            Minimum number of nearest neighbors at which to stop search if
         *                                  the best from candidates is worse than the worst of nearest neighbors
         * @param filterMode                Filtering mode in HNSW, no filtration by default
         * @param filter                    Class with Check(id) method that returns true if an item passes the filter
         * @param filterCheckLimit          Limit of the number of items for which filters are checked
         */
        template <class TItemStorage,
                  class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            size_t topSize,
            size_t searchNeighborhoodSize,
            const TItemStorage& itemStorage,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {},
            const size_t stopSearchSize = 1,
            const EFilterMode filterMode = EFilterMode::NO_FILTER,
            const TFilterBase& filter = {},
            const size_t filterCheckLimit = Max<size_t>()) const
        {
            return GetNearestNeighbors(query, topSize, searchNeighborhoodSize, Max<size_t>(), itemStorage, distance, distanceLess, stopSearchSize, filterMode, filter, filterCheckLimit);
        }

    protected:
        template <class TIndexReader>
        void Reset(const TBlob& blob, const TIndexReader& indexReader) {
            Data = blob;
            indexReader.ReadIndex(Data, &NumNeighborsInLevels, &Levels);
        }

        const ui32* GetNeighbors(ui32 level, ui32 id) const {
            return Levels[level] + id * NumNeighborsInLevels[level];
        }
        size_t GetNumLevels() const {
            return Levels.size();
        }
        size_t GetNumNeighbors(ui32 level) const {
            return NumNeighborsInLevels[level];
        }

        template <typename TSearchContext>
        THolder<INeighborsGetter> CreateNeighborsGetter(
            const EFilterMode filterMode, const TFilterWithLimit& filter, TSearchContext& context) const
        {
            switch (filterMode) {
                case EFilterMode::ACORN:
                    return MakeHolder<TAcornNeighborsGetter<TSearchContext>>(
                        Levels[0], GetNumNeighbors(0), context, filter);
                default:
                    return MakeHolder<TNeighborsGetterBase<TSearchContext>>(Levels[0], GetNumNeighbors(0), context);
            }
        }

    private:
        TBlob Data;
        TVector<ui32> NumNeighborsInLevels;
        TVector<const ui32*> Levels;
    };

} // namespace NHnsw
