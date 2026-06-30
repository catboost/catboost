#pragma once

#include <util/generic/array_ref.h>
#include <util/system/types.h>

#include <cstddef>

namespace NHnsw::NPrivate {
    template <class TDistance,
              class TDistanceResult,
              class TItemStorage,
              class TItem>
    class TDistanceAdapter {
    public:
        TDistanceAdapter(const TItemStorage& itemStorage, const TDistance& distance, size_t distanceCalcLimit)
            : ItemStorage_(itemStorage)
            , Distance_(distance)
            , CalcLimit_(distanceCalcLimit)
        {}

        bool IsLimitReached() const {
            return CalcLimit_ && CalcLimit_ <= CalcCount_;
        }

        TDistanceResult Calc(const TItem& query, const ui32 id) {
            ++CalcCount_;
            const auto& storedItem = ItemStorage_.GetItem(id);
            if constexpr (requires { Distance_(query, storedItem, id); }) {
                return Distance_(query, storedItem, id);
            } else {
                return Distance_(query, storedItem);
            }
        }

        void Prefetch(TArrayRef<const ui32> neighbors) {
            if constexpr (requires { ItemStorage_.PrefetchItem(0); }) {
                CalcCount_ += neighbors.size();
                if (IsLimitReached()) {
                    return;
                }
                for (ui32 id: neighbors) {
                    ItemStorage_.PrefetchItem(id);
                }
            }
        }

    private:
        const TItemStorage& ItemStorage_;
        const TDistance& Distance_;
        const size_t CalcLimit_;
        size_t CalcCount_ = 0;
    };
} // namespace NHnsw::NPrivate
