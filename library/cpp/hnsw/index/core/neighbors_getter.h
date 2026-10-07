#pragma once

#include "filter_base.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace NHnsw {
    using TNeighborsView = TConstArrayRef<ui32>;

    template <class TFunc>
    concept CNeighborVisitor = std::is_invocable_v<TFunc, ui32>;

    /**
     * @brief Static interface used by the common neighbors getter.
     *
     * A row access implementation controls how a graph row is materialized and how it is scanned.
     * The latter is separate so that a compressed representation can decode directly into the
     * callback instead of allocating an intermediate row.
     */
    template <class TRowAccessor>
    concept CNeighborRowAccessor = requires(TRowAccessor& rows,
                                            ui32 id,
                                            void (*consumer)(ui32)) {
        requires std::is_convertible_v<decltype(rows.GetNumNeighbors()), size_t>;
        requires std::is_same_v<decltype(rows.GetRow(id)), TNeighborsView>;
        rows.ForEachInRow(id, consumer);
    };

    /** @brief Static interface consumed by the base-level HNSW search loop.
     */
    template <class TGetter>
    concept CNeighborsGetter = requires(TGetter& getter, ui32 id) {
        typename std::bool_constant<TGetter::Prefiltered>;
        requires std::is_same_v<decltype(getter.GetLayerNeighbors(id)), TNeighborsView>;
    };

    /**
     * @brief Fallback row access for graph codecs without a specialized neighbors getter.
     *
     * GetRow() may either return a zero-copy view into graph-owned memory or decode the
     * row into the provided scratch buffer. The returned view backed by the scratch
     * buffer remains valid only until the next row is read.
     *
     * A graph codec can avoid row materialization by providing VisitNeighborsGetter(),
     * which lets the search iterate over encoded neighbors directly.
     */
    template <class TLevelReader>
    class TReaderNeighborRowAccessor {
    public:
        TReaderNeighborRowAccessor(TLevelReader reader, ui32 numNeighbors)
            : Reader(std::move(reader))
            , NumNeighbors(numNeighbors)
        {
            Scratch.reserve(numNeighbors);
        }

        ui32 GetNumNeighbors() const {
            return NumNeighbors;
        }

        template <CNeighborVisitor TFunc>
        void ForEachInRow(ui32 id, TFunc&& func) {
            for (ui32 neighbor : GetRow(id)) {
                func(neighbor);
            }
        }

        TNeighborsView GetRow(ui32 id) {
            return Reader.GetRow(id, Scratch);
        }

    private:
        TLevelReader Reader;
        ui32 NumNeighbors;
        TVector<ui32> Scratch;
    };

    /**
     * @brief Common unfiltered neighbors getter.
     *
     * TRowAccessor is a static customization point. Built-in packed formats use a specialized
     * implementation, while arbitrary codecs can use TReaderNeighborRowAccessor.
     */
    template <CNeighborRowAccessor TRowAccessor, class TSearchContext>
    class TNeighborsGetter {
    public:
        static constexpr bool Prefiltered = false;

        TNeighborsGetter(TRowAccessor rows, TSearchContext& context)
            : Rows(std::move(rows))
            , Context(context)
        {
            NeighborsBuffer.reserve(Rows.GetNumNeighbors());
        }

        TNeighborsView GetLayerNeighbors(ui32 id) {
            NeighborsBuffer.clear();
            Rows.ForEachInRow(id, [this](ui32 neighbor) {
                if (Context.TryMarkVisited(neighbor)) {
                    NeighborsBuffer.push_back(neighbor);
                }
            });
            return NeighborsBuffer;
        }

    protected:
        TNeighborsView GetRow(ui32 id) {
            return Rows.GetRow(id);
        }

        size_t GetNumNeighbors() const {
            return Rows.GetNumNeighbors();
        }

        TNeighborsView PrefilterVisited(TNeighborsView neighbors) {
            NeighborsBuffer.clear();
            for (ui32 id: neighbors) {
                if (Context.TryMarkVisited(id)) {
                    NeighborsBuffer.push_back(id);
                }
            }
            return NeighborsBuffer;
        }

    private:
        TRowAccessor Rows;
        TSearchContext& Context;
        TVector<ui32> NeighborsBuffer;
    };

    template <CNeighborRowAccessor TRowAccessor, class TSearchContext, class TFilter>
    class TAcornNeighborsGetterImpl: public TNeighborsGetter<TRowAccessor, TSearchContext> {
        using TBase = TNeighborsGetter<TRowAccessor, TSearchContext>;

    public:
        static constexpr bool Prefiltered = true;

        TAcornNeighborsGetterImpl(TRowAccessor rows, TSearchContext& context, TFilter& filter)
            : TBase(std::move(rows), context)
            , Filter(filter)
        {
            const size_t numNeighbors = this->GetNumNeighbors();
            AcornNeighbors.resize(numNeighbors * numNeighbors, 0);
            SecondHopStorage.resize(numNeighbors, 0);
        }

        TNeighborsView GetLayerNeighbors(const ui32 id) {
            ui32 acornCount = 0;
            ScanNeighbors(id, acornCount, /*isFirstHop*/ true);

            const size_t numSecondHops = this->GetNumNeighbors() - acornCount;

            for (size_t i = 0; i < numSecondHops; ++i) {
                ScanNeighbors(SecondHopStorage[i], acornCount, /*isFirstHop*/ false);
            }

            return this->PrefilterVisited(TNeighborsView{AcornNeighbors.data(), acornCount});
        }

    private:
        void ScanNeighbors(const ui32 id, ui32& acornCount, bool isFirstHop) {
            const TNeighborsView neighbors = this->GetRow(id);
            for (size_t i = 0; i < neighbors.size() && !Filter.IsLimitReached(); ++i) {
                ui32 neighbor = neighbors[i];

                if (isFirstHop && !SeenInFirstHop.Insert(neighbor)) {
                    continue;
                }

                if (!isFirstHop && SeenInFirstHop.Has(neighbor)) {
                    continue;
                }

                bool passesFilter = true;
                if (const auto* filterOk = FilterResult.FindPtr(neighbor)) {
                    passesFilter = *filterOk;
                    if (passesFilter || !isFirstHop) {
                        continue;
                    }
                } else {
                    passesFilter = Filter.Check(neighbor).Verdict == EFilterVerdict::Accept;
                    FilterResult[neighbor] = passesFilter;
                }

                if (passesFilter) {
                    AcornNeighbors[acornCount++] = neighbor;
                } else if (isFirstHop) {
                    SecondHopStorage[i - acornCount] = neighbor;
                }
            }
        }

    private:
        TFilter& Filter;
        TVector<ui32> AcornNeighbors;

        TDenseHash<ui32, bool> FilterResult;
        TDenseHashSet<ui32> SeenInFirstHop;

        TVector<ui32> SecondHopStorage;
    };
} // namespace NHnsw
