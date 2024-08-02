#pragma once

#include "index_reader.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <library/cpp/hnsw/helpers/is_item_marked_deleted.h>

#include <util/generic/vector.h>
#include <util/generic/queue.h>
#include <util/memory/blob.h>

namespace NHnsw {
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
            const TDistanceLess& distanceLess = {}) const
        {
            if (Levels.empty() || searchNeighborhoodSize == 0) {
                return {};
            }
            ui32 entryId = 0;
            auto entryDist = distance(query, itemStorage.GetItem(entryId));
            bool distanceCalcLimitReached = --distanceCalcLimit == 0;
            for (ui32 level = GetNumLevels(); level-- > 1 && !distanceCalcLimitReached;) {
                for (bool entryChanged = true; entryChanged && !distanceCalcLimitReached;) {
                    entryChanged = false;
                    const ui32* neighbors = GetNeighbors(level, entryId);
                    size_t numNeighbors = GetNumNeighbors(level);
                    PrefetchNeighbors(itemStorage, neighbors, numNeighbors, distanceCalcLimit, nullptr);
                    for (size_t i = 0; i < numNeighbors && !distanceCalcLimitReached; ++i) {
                        ui32 id = neighbors[i];
                        auto distToQuery = distance(query, itemStorage.GetItem(id));
                        distanceCalcLimitReached = --distanceCalcLimit == 0;
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
            nearest.Container().reserve(searchNeighborhoodSize + 1);
            TPriorityQueue<TResultItem, TVector<TResultItem>, decltype(neighborGreater)> candidates(neighborGreater);
            TDenseHashSet<ui32> visited(/*emptyKey*/ Max<ui32>());

            if (!NPrivate::IsItemMarkedDeleted(itemStorage, entryId)) {
                nearest.push({entryDist, entryId});
            }

            candidates.push({entryDist, entryId});
            visited.Insert(entryId);

            while (!candidates.empty() && !distanceCalcLimitReached) {
                auto cur = candidates.top();
                candidates.pop();
                if (!nearest.empty() && distanceLess(nearest.top().Dist, cur.Dist)) {
                    break;
                }
                const ui32* neighbors = GetNeighbors(/*level*/ 0, cur.Id);
                size_t numNeighbors = GetNumNeighbors(/*level*/ 0);
                PrefetchNeighbors(itemStorage, neighbors, numNeighbors, distanceCalcLimit, &visited);
                for (size_t i = 0; i < numNeighbors && !distanceCalcLimitReached; ++i) {
                    ui32 id = neighbors[i];
                    if (visited.Has(id)) {
                        continue;
                    }
                    auto distToQuery = distance(query, itemStorage.GetItem(id));
                    distanceCalcLimitReached = --distanceCalcLimit == 0;
                    if (nearest.size() < searchNeighborhoodSize || distanceLess(distToQuery, nearest.top().Dist)) {
                        if (!NPrivate::IsItemMarkedDeleted(itemStorage, id)) {
                            nearest.push({distToQuery, id});
                        }
                        candidates.push({distToQuery, id});
                        visited.Insert(id);
                        if (nearest.size() > searchNeighborhoodSize) {
                            nearest.pop();
                        }
                    }
                }
            }

            while (nearest.size() > topSize) {
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
            const TDistanceLess& distanceLess = {}) const
        {
            return GetNearestNeighbors(query, topSize, searchNeighborhoodSize, Max<size_t>(), itemStorage, distance, distanceLess);
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

        template<typename TItemStorage>
        void PrefetchNeighbors(const TItemStorage&, const ui32*, size_t, size_t, const TDenseHashSet<ui32>*) const {
            // Do nothing for TItemStorage without PrefetchItem
        }

        template<typename TItemStorage>
        requires requires(TItemStorage itemStorage) {
            itemStorage.PrefetchItem(0);
        }
        void PrefetchNeighbors(const TItemStorage& itemStorage, const ui32* neighbors, size_t numNeighbors, size_t distanceCalcLimit, const TDenseHashSet<ui32>* visited) const {
            for (size_t i = 0; i < numNeighbors; ++i) {
                ui32 id = neighbors[i];
                if (visited != nullptr && visited->Has(id)) {
                    continue;
                }
                itemStorage.PrefetchItem(id);
                if (--distanceCalcLimit == 0) {
                    break;
                }
            }
        }

    private:
        TBlob Data;
        TVector<ui32> NumNeighborsInLevels;
        TVector<const ui32*> Levels;
    };

}
