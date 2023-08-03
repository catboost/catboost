#pragma once

#include <util/generic/typetraits.h>

namespace NHnsw::NPrivate {
    Y_HAS_MEMBER(IsItemMarkedDeleted, IsItemMarkedDeleted);

    template <typename TItemStorage>
    bool IsItemMarkedDeleted(const TItemStorage& itemStorage, const ui32 id) {
        if constexpr (THasIsItemMarkedDeleted<TItemStorage>::value) {
            return itemStorage.IsItemMarkedDeleted(id);
        }
        return false;
    }

} // namespace NHnsw::NPrivate
