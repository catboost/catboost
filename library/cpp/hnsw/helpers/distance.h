#pragma once

#include <util/system/types.h>

#include <type_traits>

namespace NHnsw::NPrivate {

    template <class TDistance,
              class TDistanceResult,
              class TItemStorage,
              class TItem>
    TDistanceResult CalcDistance(const TItemStorage& itemStorage, const TDistance& distance, const TItem& query, const ui32 id) {
        if constexpr (std::is_invocable_r_v<TDistanceResult, decltype(distance), decltype(query), decltype(itemStorage.GetItem(id)), decltype(id)>) {
            return distance(query, itemStorage.GetItem(id), id);
        } else {
            return distance(query, itemStorage.GetItem(id));
        }
    }

} // namespace NHnsw::NPrivate
