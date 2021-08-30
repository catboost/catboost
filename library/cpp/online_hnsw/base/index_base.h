#pragma once

#include "build_options.h"
#include "dynamic_dense_graph.h"
#include "index_data.h"
#include "index_snapshot_data.h"

#include <library/cpp/hnsw/index_builder/build_routines.h>
#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <library/cpp/threading/local_executor/local_executor.h>

namespace NOnlineHnsw {
    /**
     * This class uses ItemStorage created outside.
     * If you don't need separate ItemStorage see item_storage_index.h
     */
    template<class TDistance,
             class TDistanceResult = typename TDistance::TResult,
             class TDistanceLess = typename TDistance::TLess>
    class TOnlineHnswIndexBase {
        using TLevel = TDynamicDenseGraph<TDistance, TDistanceResult, TDistanceLess>;
        using TDistanceTraits = NHnsw::TDistanceTraits<TDistance, TDistanceResult, TDistanceLess>;
        using TNeighbors = typename TDistanceTraits::TNeighbors;
        using TNeighbor = typename TDistanceTraits::TNeighbor;
        using TNeighborMaxQueue = typename TDistanceTraits::TNeighborMaxQueue;
        using TNeighborMinQueue = typename TDistanceTraits::TNeighborMinQueue;
        using TIndexSnapshot = typename NOnlineHnsw::TOnlineHnswIndexSnapshot<TDistanceResult>;

    public:
        explicit TOnlineHnswIndexBase(const NOnlineHnsw::TOnlineHnswBuildOptions& opts,
                                      const TDistance& distance = {},
                                      const TDistanceLess& distanceLess = {})
            : DistanceTraits(distance, distanceLess)
            , Opts(opts)
        {
            Opts.CheckOptions();
            if (Opts.LevelSizeDecay == TOnlineHnswBuildOptions::AUTO_SELECT) {
                Opts.LevelSizeDecay = Max<size_t>(2, Opts.MaxNeighbors / 2);
            }

            if (Opts.NumVertices == 0 || Opts.NumVertices == TOnlineHnswBuildOptions::AUTO_SELECT) {
                LevelSizes = {Opts.LevelSizeDecay};
            } else {
                DiverseNeighborsNums.reserve(Opts.NumVertices);
                auto levelSizesList = NHnsw::GetLevelSizes(Opts.NumVertices, Opts.LevelSizeDecay);
                LevelSizes = TDeque<size_t>(levelSizesList.begin(), levelSizesList.end());
            }

            Levels.emplace_front(Min(LevelSizes.back() - 1, Opts.MaxNeighbors), LevelSizes.back());
        }

        /**
         * @brief Method to construct index data for later static index creation.
         * In order to create static index do as follows:
         * @code
         *   auto indexData = onlineIndex.ConstructIndexData();
         *   TBufferOutput bufferOutput;
         *   NOnlineHnsw::WriteIndex(indexData, bufferOutput); // see index_writer.h
         *   TBlob indexBlob(TBlob::FromBuffer(bufferOutput.Buffer()));
         *   THnswIndexBase staticIndex = THnswIndexBase(indexBlob, TOnlineHnswIndexReader()); // see index_reader.h
         * @endcode
         */
        NOnlineHnsw::TOnlineHnswIndexData ConstructIndexData() const {
            TOnlineHnswIndexData index;
            index.MaxNeighbors = Opts.MaxNeighbors;

            size_t numElements = 0;
            for (const auto& level : Levels) {
                if (level.GetSize() == 0) {
                    continue;
                }
                numElements += level.GetSize() * level.GetNeighborCount();
                index.LevelSizes.push_back(level.GetSize());
            }
            index.FlatLevels.reserve(numElements);

            for (const auto& level : Levels) {
                if (level.GetSize() == 0) {
                    continue;
                }
                if (level.GetMaxNeighborCount() == level.GetNeighborCount()) {
                    index.FlatLevels.insert(index.FlatLevels.end(), level.GetIds().begin(), level.GetIds().end());
                } else {
                    auto shrinkedLevel = level.GetShrinkedIds();
                    index.FlatLevels.insert(index.FlatLevels.end(), shrinkedLevel.begin(), shrinkedLevel.end());
                }
            }

            return index;
        }

        /**
         * @brief Method for restoring state of online index from snapshot previously stored using #ConstructSnapshot().
         * In order to restore index from snapshot do as follows:
         * @code
         *  IInputStream* in = ...;
         *  ui32 distanceTypeOrdinal = -1;
         *  ::Load(in, distanceTypeOrdinal);
         *  TOnlineHnswIndexSnapshot<TDistanceResult> indexSnapshot;
         *  ::Load(in, indexSnapshot);
         *  EDistance distanceType = static_cast<EDistance>(distanceTypeOrdinal);
         *  switch (distanceType) {
         *    case EDistance::DotProduct: {
         *      using TIndex = TOnlineHnswIndexBase<TDotProduct<TDistanceResult>>;
         *      TIndex index = TIndex::RestoreFromSnapshot(indexSnapshot, distance, distanceLess);
         *      // do smth with index
         *    }
         *    case EDistance::L1Distance: {
         *      using TIndex = TOnlineHnswIndexBase<TL1Distance<TDistanceResult>>;
         *      TIndex index = TIndex::RestoreFromSnapshot(indexSnapshot, distance, distanceLess);
         *      // do smth with index
         *    }
         *  }
         *  return RestoreIndex(indexSnapshot, distanceType, dimension)
         * @endcode
         *
         * @param snapshot   Snapshot from which the state will be restored.
         *
         * @result           Restored Online HNSW index.
         */
        static TOnlineHnswIndexBase RestoreFromSnapshot(const TIndexSnapshot& snapshot,
                                                        const TDistance& distance = {},
                                                        const TDistanceLess& distanceLess = {})
        {
            TOnlineHnswIndexBase restoredIndex(snapshot.Options, distance, distanceLess);
            restoredIndex.Levels.clear();
            for (const auto& levelSnapshot : snapshot.Levels) {
                restoredIndex.Levels.push_back(TLevel::RestoreFromSnapshot(levelSnapshot));
            }
            restoredIndex.LevelSizes = TDeque<size_t>(snapshot.LevelSizes.begin(),
                                                      snapshot.LevelSizes.end());
            restoredIndex.DiverseNeighborsNums = TVector<size_t>(snapshot.DiverseNeighborsNums.begin(),
                                                                 snapshot.DiverseNeighborsNums.end());
            return restoredIndex;
        }


        /**
         * @brief Method for constructing plain object for current index state to write it to file and restore later.
         * In order to save index snapshot do as follows:
         * @code
         *   IOutputStream* out = ...;
         *   EDistance distanceType = GetDistanceTypeOfIndex(onlineIndex); // determine type as you wish
         *   ::Save(out, static_cast<ui32>(distanceType));
         *   auto indexSnapshot = onlineIndex.ConstructSnapshot();
         *   ::Save(out, indexSnapshot);
         * @endcode
         *
         * @result Snapshot of Online HNSW Index which does is not templated by TDistance.
         */
        TIndexSnapshot ConstructSnapshot() const {
            TIndexSnapshot snapshot;

            snapshot.Options = Opts;

            snapshot.Levels.reserve(Levels.size());
            for (const auto& level : Levels) {
                snapshot.Levels.push_back(level.ConstructSnapshot());
            }

            snapshot.LevelSizes = TVector<ui32>(LevelSizes.begin(), LevelSizes.end());
            snapshot.DiverseNeighborsNums = TVector<ui32>(DiverseNeighborsNums.begin(), DiverseNeighborsNums.end());

            return snapshot;
        }

        /**
         * @brief Method for searching in HNSW index.
         *
         * @param item          Nearest neighbors for this item will be retrieved.
         * @param itemStorage   ItemStorage containing all items index was given in GetNearestNeighborsAndAddItem.
         * @param topSize       No more than topSize neighbors will be retrieved.
         *
         * @result              Nearest neighbors sorted by distance.
         *                      Access item id and distance by neighbor.Id and neighbor.Dist respectively.
         */
        template <class TItem, class TItemStorage>
        TNeighbors GetNearestNeighbors(const TItem& item, const TItemStorage& itemStorage, const size_t topSize = Max<size_t>()) const {
            if (Opts.MaxNeighbors + 1 >= itemStorage.GetNumItems()) {
                return GetNearestNeighborsNaive(item, topSize, itemStorage);
            }

            TNeighbors result;
            NHnsw::NRoutines::FindApproximateNeighbors(DistanceTraits, itemStorage, Levels, Opts.SearchNeighborhoodSize, item, &result, topSize);
            Reverse(result.begin(), result.end());
            return result;
        }

        /**
         * @brief Method for adding items in HNSW index.
         *
         * @param item          New item to add.
         * @param itemStorage   ItemStorage containing all items index was given earlier. It will be extended by new item.
         *
         * @result              Nearest neighbors to item sorted by distance in format of GetNearestNeighbors result.
         *                      New item is not considered as neighbor.
         */
        template <class TItem, class TItemStorage>
        TNeighbors GetNearestNeighborsAndAddItem(const TItem& item, TItemStorage* itemStorage) {
            auto nearestNeighbors = GetNearestNeighbors(item, *itemStorage);

            itemStorage->AddItem(item);
            AddNewLevelIfLastIsFull();
            ExtendLastLevel<TItem, TItemStorage>(nearestNeighbors, *itemStorage);

            return nearestNeighbors;
        }

    private:

        template <class TItem, class TItemStorage>
        TNeighbors GetNearestNeighborsNaive(const TItem& item, const size_t topSize, const TItemStorage& itemStorage) const {
            TNeighborMaxQueue nearest(DistanceTraits.NeighborLess);
            for (size_t neighborId = 0; neighborId < itemStorage.GetNumItems(); ++neighborId) {
                const auto& neighbor = itemStorage.GetItem(neighborId);
                const TDistanceResult neighborDistance = DistanceTraits.Distance(item, neighbor);
                const bool isFull = nearest.size() == topSize;
                if (!isFull || DistanceTraits.DistanceLess(neighborDistance, nearest.top().Dist)) {
                    nearest.push({neighborDistance, neighborId});
                    if (isFull) {
                        nearest.pop();
                    }
                }
            }

            TNeighbors result(nearest.size());
            for (size_t orderPosition = result.size(); orderPosition-- > 0;) {
                result[orderPosition] = nearest.top();
                nearest.pop();
            }

            return result;
        }

        void AddNewLevelIfLastIsFull() {
            const size_t lastLevelMaxSize = *(LevelSizes.rbegin() + (Levels.size() - 1));
            if (Levels.front().GetSize() == lastLevelMaxSize) {
                if (LevelSizes.size() == Levels.size()) {
                    LevelSizes.emplace_front(LevelSizes.front() * Opts.LevelSizeDecay);
                }
                const size_t newLevelSize = *(LevelSizes.rbegin() + Levels.size());
                Levels.emplace_front(Min(Opts.MaxNeighbors, newLevelSize - 1), newLevelSize, Levels.front());
            }
        }

        template <class TItem, class TItemStorage>
        void ExtendLastLevel(const TNeighbors& neighbors, const TItemStorage& itemStorage) {
            DiverseNeighborsNums.push_back(0);
            TNeighbors trimmedNeighbors;
            TrimSortedNeighbors<TItem, TItemStorage>(neighbors, itemStorage, &trimmedNeighbors, &DiverseNeighborsNums.back());

            auto& level = Levels.front();
            const size_t newItemId = level.GetSize();

            for (const auto& edge : trimmedNeighbors) {
                TryAddInverseEdge<TItem, TItemStorage>(edge, newItemId, itemStorage);
            }

            level.Append(trimmedNeighbors);
        }

        template <class TItem, class TItemStorage>
        void TryAddInverseEdge(const TNeighbor& edge, const size_t sourceId, const TItemStorage& itemStorage) {
            const auto& level = Levels.front();
            const auto& newDistance = edge.Dist;
            const size_t destinationId = edge.Id;

            const size_t numDiverseNeighbors = DiverseNeighborsNums[destinationId];
            const size_t newNumNeighbors = Min(level.GetNeighborCount() + 1, Opts.MaxNeighbors);

            const TItem& sourceItem = itemStorage.GetItem(sourceId);
            const auto& neighborIds = level.NeighborIds(destinationId);
            const auto& neighborDistances = level.NeighborDistances(destinationId);

            bool isDiverse = true;
            size_t neighborPos;
            for (neighborPos = 0; neighborPos < numDiverseNeighbors; ++neighborPos) {
                if (DistanceTraits.DistanceLess(newDistance, neighborDistances[neighborPos])) {
                    break;
                }
                size_t diverseNeighborId = neighborIds[neighborPos];
                const TItem& diverseNeighborItem = itemStorage.GetItem(diverseNeighborId);

                auto distToDiverseNeighbor = DistanceTraits.Distance(diverseNeighborItem, sourceItem);
                if (NeighborBreaksDiversity(distToDiverseNeighbor, newDistance)) {
                    isDiverse = false;
                    break;
                }
            }

            const bool needRetrim = level.GetNeighborCount() > 0 && isDiverse && neighborPos < numDiverseNeighbors;
            if (needRetrim) {
                RetrimAndAddInverseEdge<TItem, TItemStorage>(edge, sourceId, itemStorage);
                return;
            }

            size_t newNeighborPos = numDiverseNeighbors;
            if (!isDiverse) {
                while (newNeighborPos < level.GetNeighborCount() &&
                       DistanceTraits.DistanceLess(neighborDistances[newNeighborPos], newDistance)) {
                    ++newNeighborPos;
                }
            }
            if (newNeighborPos >= newNumNeighbors) {
                return;
            }
            DiverseNeighborsNums[destinationId] += isDiverse;
            AddEdgeOnPosition(newNeighborPos, newNumNeighbors, destinationId, sourceId, newDistance);
        }

        void AddEdgeOnPosition(const size_t newNeighborPos,
                               const size_t newNumNeighbors,
                               const size_t edgeStartId,
                               const size_t edgeEndId,
                               const TDistanceResult& edgeDistance) {
            auto& level = Levels.front();
            const auto& neighborIds = level.NeighborIds(edgeStartId);
            const auto& neighborDistances = level.NeighborDistances(edgeStartId);

            TNeighbors neighbors;
            neighbors.reserve(newNumNeighbors);

            for (size_t oldNeighborPos = 0; oldNeighborPos < newNeighborPos; ++oldNeighborPos) {
                neighbors.emplace_back(TNeighbor{neighborDistances[oldNeighborPos], neighborIds[oldNeighborPos]});
            }
            neighbors.emplace_back(TNeighbor{edgeDistance, edgeEndId});
            for (size_t oldNeighborPos = newNeighborPos; neighbors.size() < newNumNeighbors; ++oldNeighborPos) {
                neighbors.emplace_back(TNeighbor{neighborDistances[oldNeighborPos], neighborIds[oldNeighborPos]});
            }

            level.ReplaceNeighbors(edgeStartId, neighbors);
        }

        template <class TItem, class TItemStorage>
        void RetrimAndAddInverseEdge(const TNeighbor& edge, size_t sourceId, const TItemStorage& itemStorage) {
            auto& level = Levels.front();
            const auto& newDistance = edge.Dist;
            size_t destinationId = edge.Id;

            size_t numDiverseNeighbors = DiverseNeighborsNums[destinationId];

            TNeighbors neighbors;
            neighbors.reserve(level.GetNeighborCount() + 1);

            const auto& neighborIds = level.NeighborIds(destinationId);
            const auto& neighborDistances = level.NeighborDistances(destinationId);

            bool newEdgeAvailable = true;
            for (size_t diverseNeighborPos = 0, clusteringNeighborPos = numDiverseNeighbors;
                 neighbors.size() < level.GetNeighborCount() + 1;)
            {
                bool diverseAvailable = diverseNeighborPos !=numDiverseNeighbors;
                bool clusteringAvailable = clusteringNeighborPos != level.GetNeighborCount();
                TNeighbor nextNeighbor;

                if (clusteringAvailable && (!diverseAvailable || DistanceTraits.DistanceLess(
                        neighborDistances[clusteringNeighborPos],
                        neighborDistances[diverseNeighborPos])))
                {
                    nextNeighbor = TNeighbor{neighborDistances[clusteringNeighborPos],
                                             neighborIds[clusteringNeighborPos]};
                    ++clusteringNeighborPos;
                } else if (diverseAvailable) {
                    nextNeighbor = TNeighbor{neighborDistances[diverseNeighborPos], neighborIds[diverseNeighborPos]};
                    ++diverseNeighborPos;
                }

                bool addOldEdge = diverseAvailable || clusteringAvailable;

                if (newEdgeAvailable) {
                    if (!addOldEdge || DistanceTraits.DistanceLess(newDistance, nextNeighbor.Dist)) {
                        neighbors.emplace_back(TNeighbor{newDistance, sourceId});
                        newEdgeAvailable = false;
                    }
                }
                if (addOldEdge && neighbors.size() < level.GetNeighborCount() + 1) {
                    neighbors.emplace_back(nextNeighbor);
                }
            }


            TNeighbors trimmedNeighbors;
            TrimSortedNeighbors<TItem, TItemStorage>(neighbors, itemStorage, &trimmedNeighbors, &DiverseNeighborsNums[destinationId]);

            level.ReplaceNeighbors(destinationId, trimmedNeighbors);
        }

        template <class TItem, class TItemStorage>
        void TrimSortedNeighbors(const TNeighbors& neighbors, const TItemStorage& itemStorage, TNeighbors* result, size_t* numDiverseNeighbors) {
            if (neighbors.size() == 0) {
                *numDiverseNeighbors = 0;
                return;
            }

            size_t maxNeighbors = Min(Opts.MaxNeighbors, neighbors.size());

            result->reserve(maxNeighbors);
            result->emplace_back(neighbors[0]);

            TNeighbors clusteringNeighbors;
            for (size_t neighborPos = 1; neighborPos < neighbors.size() && result->size() < maxNeighbors; ++neighborPos) {
                auto& current = neighbors[neighborPos];
                const TItem& currentItem = itemStorage.GetItem(current.Id);

                bool take = true;
                for (const auto& takenNeighbor : *result) {
                    const TItem& neighborItem = itemStorage.GetItem(takenNeighbor.Id);
                    const TDistanceResult distToNeighbor = DistanceTraits.Distance(currentItem, neighborItem);
                    if (NeighborBreaksDiversity(distToNeighbor, current.Dist)) {
                        take = false;
                        break;
                    }
                }

                if (take) {
                    result->push_back(current);
                } else if (result->size() + clusteringNeighbors.size() < maxNeighbors) {
                    clusteringNeighbors.push_back(current);
                }
            }
            *numDiverseNeighbors = result->size();

            for (size_t clusteringNeighborPos = 0; result->size() < maxNeighbors; ++clusteringNeighborPos) {
                result->push_back(clusteringNeighbors[clusteringNeighborPos]);
            }
        }

        inline bool NeighborBreaksDiversity(const TDistanceResult& distToDiverseNeighbor,
                                            const TDistanceResult& distToCenter) const {
            return DistanceTraits.DistanceLess(distToDiverseNeighbor, distToCenter);
        }

    private:
        TDistanceTraits DistanceTraits;
        TOnlineHnswBuildOptions Opts;
        TDeque<TLevel> Levels;
        TDeque<size_t> LevelSizes;
        TVector<size_t> DiverseNeighborsNums;
    };
} // namespace NOnlineHnsw
