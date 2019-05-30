#pragma once

#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <library/containers/dense_hash/dense_hash.h>


namespace NCB {

    struct TTokenId {
        ui32 Id;
        static constexpr ui32 ILLEGAL_TOKEN_ID = Max<ui32>();

        TTokenId()
            : Id(ILLEGAL_TOKEN_ID) {}

        TTokenId(ui32 id)
            : Id(id) {}

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

    class TText : public TDenseHash<TTokenId, ui32> {
        using TBase = TDenseHash<TTokenId, ui32>;
    public:
        TText() : TBase() {}

        bool operator!=(const TText& rhs) const {
            return !(*this == rhs);
        }
    };

    using TTextColumn = TMaybeOwningConstArrayHolder<TText>;

}
