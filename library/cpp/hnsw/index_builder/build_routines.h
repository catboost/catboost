#pragma once

#include "internal_build_options.h"
#include "index_data.h"

#include <library/cpp/hnsw/helpers/interrupt.h>
#include <library/cpp/hnsw/logging/logging.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/guid.h>
#include <util/generic/maybe.h>
#include <util/generic/queue.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/generic/ylimits.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/stream/format.h>
#include <util/stream/labeled.h>
#include <util/system/fs.h>
#include <util/system/hp_timer.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>

#include <utility>

#include <stddef.h>


namespace NHnsw {
    namespace NPrivate {
        template <class T>
        class TVectorPool {
        public:
            TVector<T>* Allocate() {
                if (StorageUsed == Storage.size()) {
                    Storage.emplace_back();
                } else {
                    Storage[StorageUsed].clear();
                }

                return &Storage[StorageUsed++];
            }

            void FreeAll() {
                StorageUsed = 0;
            }

        private:
            TDeque<TVector<T>> Storage;
            // Don't even delete vectors inside deque. Just clear them, so the allocated memory stays with us.
            size_t StorageUsed = 0;
        };
    }

    constexpr double REPORT_PROGRESS_INTERVAL = 1.0; // seconds

    TVector<size_t> GetLevelSizes(size_t numVectors, size_t levelSizeDecay);

    template <class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess>
    struct TDistanceTraits {
        struct TNeighbor {
            TDistanceResult Dist;
            size_t Id;

            Y_SAVELOAD_DEFINE(Dist, Id);
        };

        using TNeighbors = TVector<TNeighbor>;

        // We store already computed neighbors in dense format to improve memory locality.
        // Also, we use 'columnar' storage because most of the time we need only neighbor ids.
        // We switch between dense storage (TDenseGraph) and regular one (TGraph) as we need.
        class TDenseGraph {
        public:
            TDenseGraph() = default;

            TDenseGraph(size_t neighborsCount, size_t maxSize)
                : NeighborsCount(neighborsCount)
                , MaxSize(maxSize)
            {
                Distances.reserve(maxSize * NeighborsCount);
                Ids.reserve(maxSize * NeighborsCount);
            }

            size_t GetSize() const { return Size; }

            size_t GetNeighborsCount() const { return NeighborsCount; }

            const TVector<size_t>& GetIds() const { return Ids; }

            TArrayRef<const size_t> NeighborIds(size_t index) const {
                return {Ids.data() + index * NeighborsCount, NeighborsCount};
            }

            void AppendBatch(const TVector<TNeighbors>& batch) {
                // Make sure we don't reallocate memory here.
                // This is very important for performance.
                Y_ASSERT(Distances.size() + batch.size() * NeighborsCount <= Distances.capacity());
                Y_ASSERT(Ids.size() + batch.size() * NeighborsCount <= Ids.capacity());

                for (const auto& neighbors: batch) {
                    Y_ABORT_UNLESS(neighbors.size() == NeighborsCount);
                    for (const auto& neighbor: neighbors) {
                        Distances.push_back(neighbor.Dist);
                        Ids.push_back(neighbor.Id);
                    }
                }

                Size += batch.size();
            }

            void CopyNeighborsFrom(const TDenseGraph& other) {
                Distances.insert(Distances.end(), other.Distances.begin(), other.Distances.end());
                Ids.insert(Ids.end(), other.Ids.begin(), other.Ids.end());
                Size = other.Size;
            }

            void AppendNeighborsTo(size_t index, TNeighbors* result) const {
                result->reserve(result->size() + NeighborsCount);

                for (const auto i: xrange(index * NeighborsCount, (index + 1) * NeighborsCount)) {
                    result->push_back({Distances[i], Ids[i]});
                }
            }

            void ReplaceNeighbors(size_t index, const TNeighbors& neighbors) {
                const size_t offset = index * NeighborsCount;

                for (const size_t i: xrange(neighbors.size())) {
                    Distances[i + offset] = neighbors[i].Dist;
                    Ids[i + offset] = neighbors[i].Id;
                }
            }

            void Reserve(const size_t maxSize) {
                MaxSize = Max(MaxSize, maxSize);
                Distances.reserve(MaxSize * NeighborsCount);
                Ids.reserve(MaxSize * NeighborsCount);
            }

            void Save(IOutputStream* out) const {
                SaveMany(out, NeighborsCount, MaxSize, Distances, Ids, Size);
            }

            void Load(IInputStream* in) {
                LoadMany(in, NeighborsCount, MaxSize);
                Distances.reserve(MaxSize * NeighborsCount);
                Ids.reserve(MaxSize * NeighborsCount);
                LoadMany(in, Distances, Ids, Size);
            }

        private:
            size_t NeighborsCount;
            size_t MaxSize;
            TVector<TDistanceResult> Distances;
            TVector<size_t> Ids;
            size_t Size = 0;
        };

        using TGraph = TVector<TNeighbors>;
        using TLevels = TDeque<TGraph>;
        using TDenseLevels = TDeque<TDenseGraph>;

        struct TNeighborLess : TDistanceLess {
            TNeighborLess(const TDistanceLess& base)
                : TDistanceLess(base)
            {
            }
            bool operator()(const TNeighbor& a, const TNeighbor& b) const {
                return TDistanceLess::operator()(a.Dist, b.Dist);
            }
        };
        struct TNeighborGreater : TNeighborLess {
            TNeighborGreater(const TNeighborLess& base)
                : TNeighborLess(base)
            {
            }
            bool operator()(const TNeighbor& a, const TNeighbor& b) const {
                return TNeighborLess::operator()(b, a);
            }
        };
        using TNeighborMinQueue = TPriorityQueue<TNeighbor, TVector<TNeighbor>, TNeighborGreater>;
        using TNeighborMaxQueue = TPriorityQueue<TNeighbor, TVector<TNeighbor>, TNeighborLess>;

        const TDistance Distance;
        const TDistanceLess DistanceLess;
        const TNeighborLess NeighborLess;
        const TNeighborGreater NeighborGreater;

        TDistanceTraits(const TDistance& distance = {},
                        const TDistanceLess& distanceLess = {})
            : Distance(distance)
            , DistanceLess(distanceLess)
            , NeighborLess(distanceLess)
            , NeighborGreater(NeighborLess)
        {
        }
    };


    namespace NRoutines {
        template <class TDistanceTraits, class TLevels, class TItemStorage>
        void FindApproximateNeighbors(const TDistanceTraits& distanceTraits,
                                      const TItemStorage& itemStorage,
                                      const TLevels& levels,
                                      const size_t searchNeighborhoodSize,
                                      const typename TItemStorage::TItem& query,
                                      typename TDistanceTraits::TNeighbors* result,
                                      const size_t topSize = Max<size_t>()) {

            using TNeighborMaxQueue = typename TDistanceTraits::TNeighborMaxQueue;
            using TNeighborMinQueue = typename TDistanceTraits::TNeighborMinQueue;

            // find local minimum v (vertex which is closer to query than any of its neighbors) at level n ...
            // ... and continue searching from v at level (n - 1)
            // (algorithm 2 from paper by Malkov & Yashunin with ef=1)
            size_t entryId = 0;
            auto entryDist = distanceTraits.Distance(query, itemStorage.GetItem(entryId));
            for (size_t level = levels.size(); level-- > 1; ) {
                for (bool entryChanged = true; entryChanged; ) {
                    entryChanged = false;
                    for (const size_t neighborId: levels[level].NeighborIds(entryId)) {
                        auto distToQuery = distanceTraits.Distance(query, itemStorage.GetItem(neighborId));
                        if (distanceTraits.DistanceLess(distToQuery, entryDist)) {
                            entryDist = distToQuery;
                            entryId = neighborId;
                            entryChanged = true;
                        }
                    }
                }
            }

            // algorithm 2 from paper by Malkov & Yashunin with ef=searchNeighborhoodSize
            TNeighborMaxQueue nearest(distanceTraits.NeighborLess);
            TNeighborMinQueue candidates(distanceTraits.NeighborGreater);
            TDenseHashSet<size_t> visited(/*emptyMarker*/Max<size_t>());
            nearest.push({entryDist, entryId});
            candidates.push({entryDist, entryId});
            visited.Insert(entryId);

            const auto& thisLevel = levels[0];

            while (!candidates.empty()) {
                auto cur = candidates.top();
                candidates.pop();
                if (distanceTraits.DistanceLess(nearest.top().Dist, cur.Dist)) {
                    break;
                }
                for (const size_t neighborId: thisLevel.NeighborIds(cur.Id)) {
                    if (visited.Has(neighborId)) {
                        continue;
                    }
                    auto distToQuery = distanceTraits.Distance(query, itemStorage.GetItem(neighborId));
                    if (nearest.size() < searchNeighborhoodSize || distanceTraits.DistanceLess(distToQuery, nearest.top().Dist)) {
                        nearest.push({distToQuery, neighborId});
                        candidates.push({distToQuery, neighborId});
                        visited.Insert(neighborId);
                        if (nearest.size() > searchNeighborhoodSize) {
                            nearest.pop();
                        }
                    }
                }
            }

            while (nearest.size() > topSize) {
                nearest.pop();
            }
            result->reserve(nearest.size());
            for (; !nearest.empty(); nearest.pop()) {
                result->push_back(nearest.top());
            }
        }
    } // NRoutines


    template <class TDistanceTraits, class TItemStorage>
    class TIndexBuilder {
        using TDenseGraph = typename TDistanceTraits::TDenseGraph;
        using TDenseLevels = typename TDistanceTraits::TDenseLevels;
        using TGraph = typename TDistanceTraits::TGraph;
        using TItem = typename TItemStorage::TItem;
        using TNeighbor = typename TDistanceTraits::TNeighbor;
        using TNeighborMaxQueue = typename TDistanceTraits::TNeighborMaxQueue;
        using TNeighborMinQueue = typename TDistanceTraits::TNeighborMinQueue;
        using TNeighbors = typename TDistanceTraits::TNeighbors;

    public:
        TIndexBuilder(const THnswInternalBuildOptions& opts,
                      const TDistanceTraits& distanceTraits,
                      const TItemStorage& itemStorage)
            : Opts(opts)
            , DistanceTraits(distanceTraits)
            , ItemStorage(itemStorage)
        {
        }

        void SaveSnapshotToStream(size_t buildEnd, IOutputStream& out) {
            size_t numItems = ItemStorage.GetNumItems();
            size_t maxNeighbors = Opts.MaxNeighbors;
            size_t levelSizeDecay = Opts.LevelSizeDecay;

            ::SaveMany(&out, numItems, maxNeighbors, levelSizeDecay, buildEnd, Levels);
            out.Finish();

            HNSW_LOG << "\nSaved " << buildEnd << " items to snapshot" << Endl;
        }

        void MaybeSaveSnapshot(size_t buildEnd, const bool isModifiable = false) {
            if (isModifiable && (Levels.front().GetNeighborsCount() != Opts.MaxNeighbors || buildEnd == 0)) {
                return;
            }

            if (!Opts.SnapshotFile.empty()) {
                TString tempName = Opts.SnapshotFile + "_" + CreateGuidAsString() + ".tmp";
                try {
                    HNSW_LOG << "\nSaving to snapshot file: " << Opts.SnapshotFile << Endl;
                    TOFStream out(tempName);
                    SaveSnapshotToStream(buildEnd, out);
                    NFs::Rename(tempName, Opts.SnapshotFile);
                } catch (...) {
                    HNSW_LOG << "\nCan't save snapshot. Exception: " << CurrentExceptionMessage() << Endl;
                    if (NFs::Exists(tempName)) {
                        NFs::Remove(tempName);
                    }
                }
            }

            if (Opts.SnapshotBlobPtr) {
                try {
                    HNSW_LOG << "\nSaving to snapshot blob" << Endl;
                    TBufferOutput out;
                    SaveSnapshotToStream(buildEnd, out);
                    *Opts.SnapshotBlobPtr = TBlob::FromBuffer(out.Buffer());
                } catch (...) {
                    HNSW_LOG << "\nCan't save snapshot. Exception: " << CurrentExceptionMessage() << Endl;
                }
            }
        }

        void TryRestoreFromSnapshotFromStream(size_t* builtSize, IInputStream& in, const bool isModifiable = false) {
            try {
                size_t restoredNumItems;
                size_t restoredMaxNeighbors;
                size_t restoredLevelSizeDecay;

                ::LoadMany(&in, restoredNumItems, restoredMaxNeighbors, restoredLevelSizeDecay, *builtSize, Levels);

                if (isModifiable) {
                    Y_ENSURE(restoredNumItems <= ItemStorage.GetNumItems(), LabeledOutput(restoredNumItems, ItemStorage.GetNumItems()));
                } else {
                    Y_ENSURE(restoredNumItems == ItemStorage.GetNumItems(), LabeledOutput(restoredNumItems, ItemStorage.GetNumItems()));
                }
                Y_ENSURE(restoredMaxNeighbors == Opts.MaxNeighbors, "Different MaxNeighbors in snapshot");
                Y_ENSURE(restoredLevelSizeDecay == Opts.LevelSizeDecay, "Different LevelSizeDecay in snapshot");

                HNSW_LOG << "Restored " << *builtSize << " items" << Endl;
            } catch (...) {
                HNSW_LOG << "Can't restore from snapshot. Exception: " << CurrentExceptionMessage() << Endl;
                throw;
            }
        }

        void TryRestoreFromSnapshot(size_t* builtSize, const bool isModifiable = false) {
            if (!Opts.SnapshotFile.empty() && NFs::Exists(Opts.SnapshotFile)) {
                HNSW_LOG << "\nTrying to restore from snapshot file: " << Opts.SnapshotFile << Endl;
                TIFStream in(Opts.SnapshotFile);
                TryRestoreFromSnapshotFromStream(builtSize, in, isModifiable);
            }

            if (!*builtSize && Opts.SnapshotBlobPtr && !Opts.SnapshotBlobPtr->Empty()) {
                // Restore from blob only if nothing was restored from file.
                HNSW_LOG << "\nTrying to restore from snapshot blob!" << Endl;
                TMemoryInput in(Opts.SnapshotBlobPtr->AsStringBuf());
                TryRestoreFromSnapshotFromStream(builtSize, in, isModifiable);
            }
        }

        THnswIndexData BuildForUpdates() {
            Y_ENSURE(Opts.MaxNeighbors < Opts.BatchSize, LabeledOutput(Opts.MaxNeighbors, Opts.BatchSize));
            Y_ENSURE(Opts.LevelSizeDecay > ItemStorage.GetNumItems(), LabeledOutput(Opts.LevelSizeDecay, ItemStorage.GetNumItems()));
            Y_ENSURE(Opts.SnapshotFile != "" || Opts.SnapshotBlobPtr);
            return BuildImpl(true);
        }

        THnswIndexData Build() {
            return BuildImpl(false);
        }

        /* apply heuristic: draw edge from q to candidate vertex v iff there is no candidate vertex u: v is closer to u then to q;
         * quite similar to algorithm 4 from paper by Malkov & Yashunin...
         * ... with keepPrunedConnections=true and extendCandidates=false.
         */
        void TrimNeighbors(TNeighbors* neighbors) {
            TNeighborMinQueue candidates(neighbors->begin(), neighbors->end(), DistanceTraits.NeighborGreater);
            TNeighbors nearestNotAdded;
            neighbors->clear();
            while (!candidates.empty() && neighbors->size() < Opts.MaxNeighbors) {
                auto cur = candidates.top();
                candidates.pop();
                const auto& curVector = ItemStorage.GetItem(cur.Id);
                bool add = true;
                for (const auto& n : *neighbors) {
                    auto distToNeighbor = DistanceTraits.Distance(curVector, ItemStorage.GetItem(n.Id));
                    if (DistanceTraits.DistanceLess(distToNeighbor, cur.Dist)) {
                        add = false;
                        break;
                    }
                }
                if (add) {
                    neighbors->push_back(cur);
                } else if (nearestNotAdded.size() + neighbors->size() < Opts.MaxNeighbors) {
                    nearestNotAdded.push_back(cur);
                }
            }
            for (size_t i = 0; i < nearestNotAdded.size() && neighbors->size() < Opts.MaxNeighbors; ++i) {
                neighbors->push_back(nearestNotAdded[i]);
            }
        }

        void FindApproximateNeighbors(const TItem& query, TNeighbors* result) const {
            NRoutines::FindApproximateNeighbors(DistanceTraits, ItemStorage, Levels, Opts.SearchNeighborhoodSize, query, result);
        }

        void BuildApproximateNeighbors(size_t batchBegin, size_t batchEnd, TGraph* level) {
            auto task = [&](int id) {
                auto& neighbors = (*level)[id - batchBegin];
                // find (no more than) Opts.SearchNeighborhoodSize nearest neighbors by HNSW...
                FindApproximateNeighbors(ItemStorage.GetItem(id), &neighbors);
                // ...and trim them to (no more than) Opts.MaxNeighbors ones by heuristic.
                TrimNeighbors(&neighbors);
            };
            LocalExecutor.ExecRange(task, batchBegin, batchEnd, NPar::TLocalExecutor::WAIT_COMPLETE);
        }

        // find Opts.NumExactCandidates exact candidates in the given batch
        void FindExactNeighborsInBatch(size_t batchBegin, size_t batchEnd, size_t queryId, const TItem& query, TNeighbors* result) {
            TNeighborMaxQueue nearest(DistanceTraits.NeighborLess);
            for (size_t id = batchBegin; id < batchEnd; ++id) {
                if (id == queryId) {
                    continue;
                }
                auto dist = DistanceTraits.Distance(query, ItemStorage.GetItem(id));
                if (nearest.size() < Opts.NumExactCandidates || DistanceTraits.DistanceLess(dist, nearest.top().Dist)) {
                    nearest.push({dist, id});
                    if (nearest.size() > Opts.NumExactCandidates) {
                        nearest.pop();
                    }
                }
            }

            for (; !nearest.empty(); nearest.pop()) {
                result->push_back(nearest.top());
            }
        }

        void AddExactNeighborsInBatch(size_t batchBegin, size_t batchEnd, TGraph* level) {
            auto task = [&](int id) {
                auto& neighbors = (*level)[id - batchBegin];
                // find (no more then) Opts.NumExactCandidates exact neighbors...
                FindExactNeighborsInBatch(batchBegin, batchEnd, id, ItemStorage.GetItem(id), &neighbors);
                // ...and trim them to (no more then) Opts.MaxNeighbors ones by heuristic.
                TrimNeighbors(&neighbors);
            };
            LocalExecutor.ExecRange(task, batchBegin, batchEnd, NPar::TLocalExecutor::WAIT_COMPLETE);
        }

        void UpdatePreviousNeighbors(size_t batchBegin, size_t batchEnd, TDenseGraph* denseLevel, TGraph* level) {
            // For each new edge u -> v create edge v -> u (especially important for interbatch edges)...

            // This is going to be tricky because it needs to be fast: the method is executed
            // on single thread between batches.

            NeighborsPool.FreeAll();

            // Figure out which neighbors need to be added to vertices in previous batches.
            TDenseHash<size_t, TNeighbors*> additionalNeighbors(
                    /* emptyMarker = */ Max<size_t>(),
                    /* initSize = */ (batchEnd - batchBegin) * denseLevel->GetNeighborsCount());

            for (const size_t i: xrange(level->size())) {
                auto& neighbors = (*level)[i];
                for (const auto& n : neighbors) {
                    // Where the new neighbor will go.
                    TNeighbors* toPush;
                    if (n.Id >= batchBegin) { // Neighbor is in the current batch.
                        Y_ASSERT(n.Id < batchEnd);
                        toPush = level->data() + (n.Id - batchBegin);
                    } else { // Neighbor is in one the previous batches.
                        if (auto toPushPtr = additionalNeighbors.FindPtr(n.Id)) {
                            toPush = *toPushPtr;
                        } else {
                            toPush = additionalNeighbors.emplace(n.Id, NeighborsPool.Allocate()).first->second;
                        }
                    }
                    toPush->push_back({n.Dist, i + batchBegin});
                }
            }

            TVector<std::pair<TMaybe<size_t>, TNeighbors*>> toTrim;
            toTrim.reserve(additionalNeighbors.Size() + level->size());
            for (auto& neighbors: *level) {
                // We will need to trim only vertices that obtained new neighbors.
                if (neighbors.size() > denseLevel->GetNeighborsCount()) {
                    toTrim.emplace_back(Nothing(), &neighbors);
                }
            }
            for (auto neighborData : additionalNeighbors) {
                toTrim.emplace_back(neighborData.first, neighborData.second);
            }

            // ... and then trim neighbors
            auto task = [&](int id) {
                auto [maybeId, neighbors] = toTrim[id];
                if (maybeId) {
                    // Add all neighbors that the vertex already has.
                    denseLevel->AppendNeighborsTo(*maybeId, neighbors);
                }
                SortUniqueBy(*neighbors, [](const auto& n) { return n.Id; });
                TrimNeighbors(neighbors);
                if (maybeId) {
                    denseLevel->ReplaceNeighbors(*maybeId, *neighbors);
                }
            };
            LocalExecutor.ExecRange(task, 0, toTrim.size(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }

        void ProcessBatch(size_t batchBegin, size_t batchEnd, TDenseGraph* level) {
            Y_ENSURE(level != nullptr);

            THPTimer watch;
            TVector<TNeighbors> batchNeighbors(batchEnd - batchBegin);
            if (batchBegin > 0) {
                BuildApproximateNeighbors(batchBegin, batchEnd, &batchNeighbors);
                if (Opts.Verbose) {
                    HNSW_LOG << "\tbuild ann " << watch.PassedReset() / (batchEnd - batchBegin) << Endl;
                }
                CheckInterrupted(); // check after long-lasting operation
            }

            AddExactNeighborsInBatch(batchBegin, batchEnd, &batchNeighbors);
            if (Opts.Verbose) {
                HNSW_LOG << "\tbuild exact " << watch.PassedReset() / (batchEnd - batchBegin) << Endl;
            }
            CheckInterrupted(); // check after long-lasting operation

            UpdatePreviousNeighbors(batchBegin, batchEnd, level, &batchNeighbors);
            level->AppendBatch(batchNeighbors);
            if (Opts.Verbose) {
                HNSW_LOG << "\tbuild prev " << watch.PassedReset() / (batchEnd - batchBegin) << Endl;
            }
            CheckInterrupted(); // check after long-lasting operation
        }

        void BuildLevel(size_t levelSize, size_t builtLevelSize, size_t batchSize, bool isModifiable = false) {
            auto& level = Levels.front();
            if (Levels.size() > 1 && builtLevelSize == 0) {
                // copy previous level to new empty one
                const auto& previousLevel = Levels[1];
                if (previousLevel.GetSize() >= batchSize) {
                    level.CopyNeighborsFrom(previousLevel);
                    builtLevelSize = previousLevel.GetSize();
                }
            }

            THPTimer localWatch;
            double lastReportProgressTime = GlobalWatch.Passed();
            double lastSaveSnapshotTime = GlobalWatch.Passed();
            for (size_t batchBegin = builtLevelSize; batchBegin < levelSize;) {
                const size_t curBatchSize = Min(levelSize - batchBegin, batchSize);
                if (isModifiable && curBatchSize != batchSize) {
                    // Save the last complete batch. If all batches are complete save after the loop.
                    MaybeSaveSnapshot(batchBegin, isModifiable);
                }
                const size_t batchEnd = batchBegin + curBatchSize;
                ProcessBatch(batchBegin, batchEnd, &level);
                batchBegin = batchEnd;

                if (Opts.ReportProgress) {
                    const double passedTime = GlobalWatch.Passed();
                    if (passedTime - lastReportProgressTime > REPORT_PROGRESS_INTERVAL) {
                        const double progress = (double)batchEnd / ItemStorage.GetNumItems();
                        HNSW_LOG << "\rProgress: " << Prec(progress * 100, PREC_POINT_DIGITS, 3) << "%\t";
                        HNSW_LOG << "Time passed: " << HumanReadable(TDuration::Seconds(passedTime));

                        lastReportProgressTime = passedTime;
                    }
                }
                if (Opts.Verbose) {
                    size_t numProcessed = batchEnd - builtLevelSize;
                    HNSW_LOG << Endl << batchEnd << '\t' << localWatch.Passed() / numProcessed << '\t' << numProcessed / localWatch.Passed() << Endl;
                }
                if (GlobalWatch.Passed() - lastSaveSnapshotTime > Opts.SnapshotInterval) {
                    MaybeSaveSnapshot(batchEnd, isModifiable);
                    lastSaveSnapshotTime = GlobalWatch.Passed();
                }
            }
            if (!isModifiable || levelSize % batchSize == 0) {
                MaybeSaveSnapshot(levelSize, isModifiable);
            }
        }

    private:
        THnswIndexData BuildImpl(const bool isModifiable = false) {
            LocalExecutor.RunAdditionalThreads(Opts.NumThreads - 1);

            const size_t numItems = ItemStorage.GetNumItems();
            auto levelSizes = GetLevelSizes(numItems, Opts.LevelSizeDecay);

            if (isModifiable) {
                Y_ENSURE(levelSizes.size() <= 1);
            }

            size_t alreadyBuilt = 0;
            TryRestoreFromSnapshot(&alreadyBuilt, isModifiable);
            for (size_t level = levelSizes.size(); level-- > 0;) {
                if (alreadyBuilt >= levelSizes[level]) {
                    continue;
                }
                if (Opts.ReportProgress) {
                    HNSW_LOG << Endl << "Building level " << level << " size " << levelSizes[level] << Endl;
                }
                size_t batchSize = level == 0 ? Opts.BatchSize : Opts.UpperLevelBatchSize;
                size_t buildStart = alreadyBuilt;
                if (Levels.size() < levelSizes.size() - level) {
                    Levels.emplace_front(Min(Opts.MaxNeighbors, levelSizes[level] - 1), levelSizes[level]);
                    buildStart = 0;
                }
                if (isModifiable) {
                    Levels.front().Reserve(numItems);
                }
                BuildLevel(levelSizes[level], buildStart, batchSize, isModifiable);
            }

            if (Opts.ReportProgress) {
                HNSW_LOG << Endl << "Done in " << HumanReadable(TDuration::Seconds(GlobalWatch.Passed())) << Endl;
            }
            return ConstructIndexData(Opts, Levels);
        }

        const THnswInternalBuildOptions& Opts;
        const TDistanceTraits& DistanceTraits;
        const TItemStorage& ItemStorage;
        NPar::TLocalExecutor LocalExecutor;
        TDenseLevels Levels;
        THPTimer GlobalWatch;
        NPrivate::TVectorPool<TNeighbor> NeighborsPool;
    };

    template <class TDistanceTraits, class TItemStorage>
    void TrimNeighbors(const THnswInternalBuildOptions& opts,
                       const TDistanceTraits& distanceTraits,
                       const TItemStorage& itemStorage,
                       typename TDistanceTraits::TNeighbors* neighbors) {
        TIndexBuilder<TDistanceTraits, TItemStorage>(opts, distanceTraits, itemStorage).TrimNeighbors(neighbors);
    }

    template <class TDistanceTraits, class TItemStorage>
    void FindExactNeighborsInBatch(const THnswInternalBuildOptions& opts,
                                   const TDistanceTraits& distanceTraits,
                                   const TItemStorage& itemStorage,
                                   size_t batchBegin,
                                   size_t batchEnd,
                                   size_t queryId,
                                   const typename TItemStorage::TItem& query,
                                   typename TDistanceTraits::TNeighbors* result) {
        TIndexBuilder<TDistanceTraits, TItemStorage>(opts, distanceTraits, itemStorage)
            .FindExactNeighborsInBatch(batchBegin, batchEnd, queryId, query, result);
    }

    template <class TDistanceTraits, class TItemStorage>
    THnswIndexData BuildIndexWithTraits(const THnswInternalBuildOptions& opts,
                                        const TDistanceTraits& distanceTraits,
                                        const TItemStorage& itemStorage) {
        return TIndexBuilder<TDistanceTraits, TItemStorage>(opts, distanceTraits, itemStorage).Build();
    }

    template <class TDistanceTraits, class TItemStorage>
    THnswIndexData BuildForUpdatesIndexWithTraits(const THnswInternalBuildOptions& opts,
                                                  const TDistanceTraits& distanceTraits,
                                                  const TItemStorage& itemStorage) {
        return TIndexBuilder<TDistanceTraits, TItemStorage>(opts, distanceTraits, itemStorage).BuildForUpdates();
    }
}
