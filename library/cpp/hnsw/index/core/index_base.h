#pragma once

#include "filter.h"
#include "graph_accessor.h"
#include "neighbors_getter.h"

#include <library/cpp/hnsw/helpers/distance.h>
#include <library/cpp/hnsw/helpers/is_item_marked_deleted.h>
#include <library/cpp/hnsw/helpers/search_context.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/queue.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/system/compiler.h>

#include <type_traits>
#include <utility>

namespace NHnsw {
    /**
     * @brief Parameters for HNSW search configuration.
     *
     * @param TopSize                   The search will return at most this much nearest items.
     * @param SearchNeighborhoodSize    Size of the dynamic candidate list (ef).
     *                                  Increasing this makes search slower but more accurate.
     * @param DistanceCalcLimit         Limit on the number of distance computations.
     * @param StopSearchSize            Minimum number of nearest neighbors found
     *                                  before termination conditions are evaluated.
     * @param UseBaseLevelForEntryInit  If true, entry-point refinement also traverses the base
     *                                  level (level 0); by default only upper levels are used.
     */
    struct TSearchParameters {
        size_t TopSize;
        size_t SearchNeighborhoodSize;
        size_t DistanceCalcLimit = Max<size_t>();
        size_t StopSearchSize = 1;
        bool UseBaseLevelForEntryInit = false;
    };

    /**
     * This class uses ItemStorage created outside successor of this class.
     * If you don't need separate ItemStorage see index_item_storage_base.h.
     */
    template <class TGraphAccess>
    class THnswIndexBaseImpl {
    public:
        template <class TDistanceResult>
        struct TNeighbor {
            TDistanceResult Dist;
            ui32 Id;
        };

    public:
        explicit THnswIndexBaseImpl(const TString& filename)
            requires std::is_constructible_v<TGraphAccess, TBlob>
            : THnswIndexBaseImpl(TBlob::PrechargedFromFile(filename))
        {
        }

        explicit THnswIndexBaseImpl(TBlob blob)
            requires std::is_constructible_v<TGraphAccess, TBlob>
            : GraphAccess(std::move(blob))
        {
        }

        template <class TLoader>
            requires std::is_constructible_v<TGraphAccess, TBlob, const TLoader&>
        THnswIndexBaseImpl(const TString& filename, const TLoader& loader)
            : THnswIndexBaseImpl(TBlob::PrechargedFromFile(filename), loader)
        {
        }

        template <class TLoader>
            requires std::is_constructible_v<TGraphAccess, TBlob, const TLoader&>
        THnswIndexBaseImpl(TBlob blob, const TLoader& loader)
            : GraphAccess(std::move(blob), loader)
        {
        }

        /**
         * @brief Method for searching HNSW in index.
         *
         * @param query             Query item for which nearest neighbors are retrieved.
         * @param params            Structure containing search constraints and performance tuning.
         * @param itemStorage       Storage providing GetItem(ui32 id) for the index.
         * @param distance          Distance metric implementation
         * @param distanceLess      Comparator for distance results
         * @param filter            Optional filter for controlling item acceptance and graph exploration
         * @param context           Optional search context for reusing internal buffers across queries
         */
        template <class TItemStorage,
                  class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess,
                  class TItem,
                  class TFilter = TDefaultFilter,
                  class TSearchContext = TDefaultSearchContext>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(
            const TItem& query,
            const TSearchParameters& params,
            const TItemStorage& itemStorage,
            const TDistance& distance = {},
            const TDistanceLess& distanceLess = {},
            TFilter&& filter = {},
            TSearchContext&& context = {}) const
        {
            if (GetNumLevels() == 0 || params.SearchNeighborhoodSize == 0 || params.StopSearchSize == 0) {
                return {};
            }

            ui32 entryId = 0;
            NPrivate::TDistanceAdapter<TDistance, TDistanceResult, TItemStorage, TItem> distanceAdapter(
                itemStorage, distance, params.DistanceCalcLimit);
            auto entryDist = distanceAdapter.Calc(query, entryId);
            const ui32 minLevel = params.UseBaseLevelForEntryInit ? 0 : 1;
            TVector<ui32> entryRowScratch = MakeRowScratch();
            for (ui32 level = GetNumLevels(); level-- > minLevel && !distanceAdapter.IsLimitReached(); ) {
                auto reader = GraphAccess.CreateLevelReader(level);
                for (bool entryChanged = true; entryChanged && !distanceAdapter.IsLimitReached(); ) {
                    entryChanged = false;
                    const TNeighborsView neighbors = reader.GetRow(entryId, entryRowScratch);
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
            TPriorityQueue<TResultItem, TVector<TResultItem>, decltype(neighborLess)> nearest(neighborLess);
            nearest.Container().reserve(params.SearchNeighborhoodSize + 1);

            using TFilterState = typename std::decay_t<TFilter>::TState;
            struct TCandidate {
                TResultItem ResultItem;
                Y_NO_UNIQUE_ADDRESS TFilterState FilterState;
            };
            auto candidateGreater = [&neighborLess](const TCandidate& a, const TCandidate& b) {
                return neighborLess(b.ResultItem, a.ResultItem);
            };
            TPriorityQueue<TCandidate, TVector<TCandidate>, decltype(candidateGreater)> candidates(candidateGreater);

            using TFilterResult = typename std::decay_t<TFilter>::TResult;
            auto addResultItem = [&](const TResultItem& resultItem, const TFilterResult& filterResult) {
                switch (filterResult.Verdict) {
                    case EFilterVerdict::Accept:
                        if (!NPrivate::IsItemMarkedDeleted(itemStorage, resultItem.Id)) {
                            nearest.push(resultItem);
                        }
                        [[fallthrough]];
                    case EFilterVerdict::Explore:
                        candidates.push({resultItem, filterResult.State});
                        break;
                    case EFilterVerdict::Reject:
                        break;
                };
            };

            context.TryMarkVisited(entryId);
            addResultItem({entryDist, entryId}, filter.Check(entryId));

            GraphAccess.VisitNeighborsGetter(filter, context, [&]<CNeighborsGetter TGetter>(TGetter& neighborsGetter) {
                while (!candidates.empty() && !distanceAdapter.IsLimitReached() && !filter.IsLimitReached()) {
                    auto cur = candidates.top();
                    candidates.pop();
                    if (nearest.size() >= params.StopSearchSize && distanceLess(nearest.top().Dist, cur.ResultItem.Dist)) {
                        break;
                    }
                    const auto neighbors = neighborsGetter.GetLayerNeighbors(cur.ResultItem.Id);
                    distanceAdapter.Prefetch(neighbors);
                    for (ui32 id: neighbors) {
                        if (distanceAdapter.IsLimitReached()) {
                            break;
                        }
                        if constexpr (!TGetter::Prefiltered) {
                            if (filter.IsLimitReached()) {
                                break;
                            }
                        }
                        const auto distToQuery = distanceAdapter.Calc(query, id);
                        if (nearest.size() < params.SearchNeighborhoodSize || distanceLess(distToQuery, nearest.top().Dist)) {
                            const auto filterResult = [&]() {
                                if constexpr (TGetter::Prefiltered) {
                                    return TFilterResult{};
                                } else {
                                    return filter.Check(id, cur.FilterState);
                                }
                            }();
                            addResultItem({distToQuery, id}, filterResult);
                            if (nearest.size() > params.SearchNeighborhoodSize) {
                                nearest.pop();
                            }
                        }
                    }
                }
            });

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
            };
            TFilterAdapter filterAdapter(filter, filterMode, filterCheckLimit);
            return GetNearestNeighbors<TItemStorage, TDistance, TDistanceResult, TDistanceLess, TItem>(
                query, params, itemStorage, distance, distanceLess, filterAdapter);
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
        const ui32* GetNeighbors(ui32 level, ui32 id) const {
            return GraphAccess.GetNeighbors(level, id);
        }

        /**
         * @brief Neighbor row of item @p id at @p level, in whichever id format the index holds.
         *
         * @p scratch is only touched for the packed formats, and the returned view stays valid
         * only until the next call made with the same buffer. See TLevelRows::GetRow.
         */
        TNeighborsView GetNeighborsView(ui32 level, ui32 id, TVector<ui32>& scratch) const {
            auto reader = GraphAccess.CreateLevelReader(level);
            return reader.GetRow(id, scratch);
        }

        size_t GetNumLevels() const {
            return GraphAccess.GetNumLevels();
        }

        size_t GetNumNeighbors(ui32 level) const {
            return GraphAccess.GetNumNeighbors(level);
        }

    private:
        TVector<ui32> MakeRowScratch() const {
            TVector<ui32> scratch;
            scratch.reserve(GraphAccess.GetRowScratchSize());
            return scratch;
        }

        TGraphAccess GraphAccess;
    };

    /** @brief HNSW index whose graph representation is provided by a concrete codec. */
    template <CGraphCodec TCodec>
    using THnswIndexCompressed = THnswIndexBaseImpl<TCodecGraphAccessor<TCodec>>;

} // namespace NHnsw
