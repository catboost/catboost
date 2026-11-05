#pragma once

#include <util/generic/deque.h>
#include <util/generic/queue.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/stream/fwd.h>
#include <util/system/types.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>

#include <stddef.h>

namespace NHnsw {
    template <class TDistance,
            class TDistanceResult = typename TDistance::TResult,
            class TDistanceLess = typename TDistance::TLess,
            class TId = size_t>
    struct TDistanceTraits {
        struct TNeighbor {
            TDistanceResult Dist;
            TId Id;

            TNeighbor() = default;
            TNeighbor(TDistanceResult dist, size_t id)
                : Dist(dist)
                , Id(static_cast<TId>(id))
            {
            }

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

            const TVector<TId>& GetIds() const { return Ids; }

            TArrayRef<const TId> NeighborIds(size_t index) const {
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

            void ClearDistances() {
                Distances = TVector<TDistanceResult>();
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
            TVector<TId> Ids;
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

    template <class TDistance,
            class TDistanceResult = typename TDistance::TResult,
            class TDistanceLess = typename TDistance::TLess>
    using TDistanceTraitsUi32 = TDistanceTraits<TDistance, TDistanceResult, TDistanceLess, ui32>;
}
