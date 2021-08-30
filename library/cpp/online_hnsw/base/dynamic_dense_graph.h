#pragma once

#include "index_snapshot_data.h"

#include <library/cpp/hnsw/index_builder/build_routines.h>
#include <util/generic/array_ref.h>

namespace NOnlineHnsw {
    // Can maintain graph at the beginning, when the neighborhoods are small.
    template<class TDistance,
             class TDistanceResult = typename TDistance::TResult,
             class TDistanceLess = typename TDistance::TLess>
    class TDynamicDenseGraph {
        using TDistanceTraits = NHnsw::TDistanceTraits<TDistance, TDistanceResult, TDistanceLess>;
        using TNeighbors = typename TDistanceTraits::TNeighbors;
        using TGraphSnapshot = typename NOnlineHnsw::TOnlineHnswIndexSnapshot<TDistanceResult>::TDynamicDenseGraphSnapshot;

    public:
        TDynamicDenseGraph(size_t maxNeighborCount, size_t maxSize)
            : MaxNeighborCount(maxNeighborCount)
            , MaxSize(maxSize)
            , Size(0)
            , NeighborCount(0)
        {
            // Not actually dynamic, just pretends to be.
            Distances.reserve(MaxSize * MaxNeighborCount);
            Ids.reserve(MaxSize * MaxNeighborCount);
        }

        TDynamicDenseGraph(size_t maxNeighborCount, size_t maxSize, const TDynamicDenseGraph& other)
            : TDynamicDenseGraph(maxNeighborCount, maxSize) {
            Y_ASSERT(other.Size <= MaxSize);
            Y_ASSERT(other.MaxNeighborCount <= MaxNeighborCount);

            Size = other.Size;
            NeighborCount = other.NeighborCount;

            if (MaxNeighborCount == other.MaxNeighborCount) {
                Distances.insert(Distances.end(), other.Distances.begin(), other.Distances.end());
                Ids.insert(Ids.end(), other.Ids.begin(), other.Ids.end());
                return;
            }

            Distances.reserve(other.GetSize() * MaxNeighborCount);
            Ids.reserve(other.GetSize() * MaxNeighborCount);
            for (size_t vertexId = 0; vertexId < other.GetSize(); ++vertexId) {
                ExtendWithPadding(Distances, other.NeighborDistances(vertexId));
                ExtendWithPadding(Ids, other.NeighborIds(vertexId));
            }
        }

        static TDynamicDenseGraph RestoreFromSnapshot(const TGraphSnapshot& snapshot) {
            return TDynamicDenseGraph(snapshot.MaxNeighborCount,
                                      snapshot.MaxSize,
                                      snapshot.Size,
                                      snapshot.NeighborCount,
                                      snapshot.Distances,
                                      TVector<size_t>(snapshot.Ids.begin(), snapshot.Ids.end()));
        }

        TGraphSnapshot ConstructSnapshot() const {
            TGraphSnapshot snapshot;
            snapshot.MaxNeighborCount = static_cast<ui32>(MaxNeighborCount);
            snapshot.MaxSize = static_cast<ui32>(MaxSize);
            snapshot.Size = static_cast<ui32>(Size);
            snapshot.NeighborCount = static_cast<ui32>(NeighborCount);
            snapshot.Distances = Distances;
            snapshot.Ids = TVector<ui32>(Ids.begin(), Ids.end());
            return snapshot;
        }

        size_t GetSize() const {
            return Size;
        }

        size_t GetMaxNeighborCount() const {
            return MaxNeighborCount;
        }

        size_t GetNeighborCount() const {
            return NeighborCount;
        }

        TArrayRef<const size_t> NeighborIds(size_t index) const {
            return {Ids.data() + index * MaxNeighborCount, NeighborCount};
        }

        TArrayRef<const TDistanceResult> NeighborDistances(size_t index) const {
            return {Distances.data() + index * MaxNeighborCount, NeighborCount};
        }

        void Append(const TNeighbors& neighbors) {
            Y_ASSERT(Size < MaxSize);
            Y_ASSERT(neighbors.size() <= MaxNeighborCount);

            for (const auto& neighbor : neighbors) {
                Distances.emplace_back(neighbor.Dist);
                Ids.emplace_back(neighbor.Id);
            }

            Distances.resize(Distances.size() + MaxNeighborCount - neighbors.size());
            Ids.resize(Ids.size() + MaxNeighborCount - neighbors.size());

            ++Size;

            if (NeighborCount < MaxNeighborCount) {
                NeighborCount = Size - 1;
            }
        }

        void ReplaceNeighbors(size_t index, const TNeighbors& neighbors) {
            Y_ASSERT(index < Size);
            const size_t offset = index * MaxNeighborCount;

            for (size_t position = 0; position < neighbors.size(); ++position) {
                Distances[offset + position] = neighbors[position].Dist;
                Ids[offset + position] = neighbors[position].Id;
            }
        }

        const TVector<size_t>& GetIds() const {
            return Ids;
        }

        TVector<size_t> GetShrinkedIds() const {
            TVector<size_t> shrinkedIds;
            shrinkedIds.reserve(Size * NeighborCount);
            for (size_t vertexId = 0; vertexId < Size; ++vertexId) {
                const size_t offset = vertexId * MaxNeighborCount;
                shrinkedIds.insert(shrinkedIds.end(), Ids.begin() + offset, Ids.begin() + offset + NeighborCount);
            }
            return shrinkedIds;
        }

    private:
        TDynamicDenseGraph(
            size_t maxNeighborCount,
            size_t maxSize,
            size_t size,
            size_t neighborCount,
            TVector<TDistanceResult> distances,
            TVector<size_t> ids)
            : MaxNeighborCount(maxNeighborCount)
            , MaxSize(maxSize)
            , Size(size)
            , NeighborCount(neighborCount)
            , Distances(std::move(distances))
            , Ids(std::move(ids))
        {
        }

        template <class T>
        void ExtendWithPadding(TVector<T>& data, const TArrayRef<const T>& newItems) {
            data.insert(data.end(), newItems.begin(), newItems.begin() + NeighborCount);
            data.resize(data.size() + MaxNeighborCount - NeighborCount);
        }

    private:
        size_t MaxNeighborCount;
        size_t MaxSize;
        size_t Size;
        size_t NeighborCount;

        TVector<TDistanceResult> Distances;
        TVector<size_t> Ids;
    };
} // namespace NOnlineHnsw
