#pragma once

#include "filter_base.h"

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

#include <utility>

namespace NHnsw {

    class INeighborsGetter {
    public:
        virtual ~INeighborsGetter() = default;

        virtual std::pair<const ui32*, ui32> GetLayerNeighbors(const ui32 id) = 0;
    };

    class TNeighborsGetterBase: public INeighborsGetter {
    public:
        TNeighborsGetterBase(const ui32* level, const ui32 numNeighbors)
            : Level(level)
            , NumNeighbors(numNeighbors)
        {
        }

        std::pair<const ui32*, ui32> GetLayerNeighbors(const ui32 id) override {
            return std::make_pair(GetNeighbors(id), GetNumNeighbors());
        }

    protected:
        const ui32* GetNeighbors(const ui32 id) {
            return Level + id * NumNeighbors;
        }

        size_t GetNumNeighbors() {
            return NumNeighbors;
        }

    private:
        const ui32* Level;
        const size_t NumNeighbors;
    };

    class TAcornNeighborsGetter: public TNeighborsGetterBase {
    public:
        TAcornNeighborsGetter(const ui32* level, const ui32 numNeighbors, const TFilterWithLimit& filter)
            : TNeighborsGetterBase(level, numNeighbors)
            , Filter(filter)
        {
            AcornNeighbors.resize(numNeighbors * numNeighbors, 0);
            SecondHopStorage.resize(numNeighbors, 0);
        }

        std::pair<const ui32*, ui32> GetLayerNeighbors(const ui32 id) override {
            ui32 acornCount = 0;
            ScanNeighbors(id, acornCount, /*isFirstHop*/ true);

            const size_t numSecondHops = GetNumNeighbors() - acornCount;

            for (size_t i = 0; i < numSecondHops; ++i) {
                ScanNeighbors(SecondHopStorage[i], acornCount, /*isFirstHop*/ false);
            }

            return std::make_pair(AcornNeighbors.data(), acornCount);
        }

    private:
        void ScanNeighbors(const ui32 id, ui32& acornCount, bool isFirstHop) {
            auto neighbors = GetNeighbors(id);
            for (size_t i = 0; i < GetNumNeighbors() && !Filter.IsLimitReached(); ++i) {
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
