#pragma once

#include "filter_base.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/types.h>

namespace NHnsw {
    using TNeighborsView = TArrayRef<const ui32>;

    class INeighborsGetter {
    public:
        virtual ~INeighborsGetter() = default;

        virtual TNeighborsView GetLayerNeighbors(const ui32 id) = 0;
    };

    template <typename TSearchContext>
    class TNeighborsGetterBase: public INeighborsGetter {
    public:
        TNeighborsGetterBase(const ui32* level, const ui32 numNeighbors, TSearchContext& context)
            : Level(level)
            , NumNeighbors(numNeighbors)
            , Context(context)
        {
            NeighborsBuffer.reserve(NumNeighbors);
        }

        TNeighborsView GetLayerNeighbors(const ui32 id) override {
            return PrefilterVisited(TNeighborsView{GetNeighbors(id), NumNeighbors});
        }

    protected:
        const ui32* GetNeighbors(const ui32 id) {
            return Level + id * NumNeighbors;
        }

        size_t GetNumNeighbors() {
            return NumNeighbors;
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
        const ui32* Level;
        const size_t NumNeighbors;
        TSearchContext& Context;
        TVector<ui32> NeighborsBuffer;
    };

    template <typename TSearchContext>
    class TAcornNeighborsGetter: public TNeighborsGetterBase<TSearchContext> {
    public:
        TAcornNeighborsGetter(const ui32* level, const ui32 numNeighbors, TSearchContext& context, const TFilterWithLimit& filter)
            : TNeighborsGetterBase<TSearchContext>(level, numNeighbors, context)
            , Filter(filter)
        {
            AcornNeighbors.resize(numNeighbors * numNeighbors, 0);
            SecondHopStorage.resize(numNeighbors, 0);
        }

        TNeighborsView GetLayerNeighbors(const ui32 id) override {
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
            auto neighbors = this->GetNeighbors(id);
            for (size_t i = 0; i < this->GetNumNeighbors() && !Filter.IsLimitReached(); ++i) {
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
                    passesFilter = Filter.Check(neighbor);
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
        const TFilterWithLimit& Filter;
        TVector<ui32> AcornNeighbors;

        TDenseHash<ui32, bool> FilterResult;
        TDenseHashSet<ui32> SeenInFirstHop;

        TVector<ui32> SecondHopStorage;
    };

} // namespace NHnsw
