#pragma once

#include <util/system/types.h>

namespace NCB {

    struct TTokenId {
        ui32 Id;

        TTokenId()
            : Id(static_cast<ui32>(-1)) {

        }

        TTokenId(ui32 id)
            : Id(id) {

        }


        operator ui32() const {
            return Id;
        }

        bool operator==(const TTokenId& rhs) const {
            return Id == rhs.Id;
        }
        bool operator!=(const TTokenId& rhs) const {
            return !(rhs == *this);
        }

        bool operator<(const TTokenId& rhs) const {
            return Id < rhs.Id;
        }
        bool operator>(const TTokenId& rhs) const {
            return rhs < *this;
        }
        bool operator<=(const TTokenId& rhs) const {
            return !(rhs < *this);
        }
        bool operator>=(const TTokenId& rhs) const {
            return !(*this < rhs);
        }
    };

    using TText = TDenseHash<TTokenId, ui32>;
}




template <>
struct THash<NCB::TTokenId> {
    inline size_t operator()(NCB::TTokenId id) const {
        return THash<ui32>()(id.Id);
    }
};

