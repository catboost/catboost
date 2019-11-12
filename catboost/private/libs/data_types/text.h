#pragma once

#include <library/containers/dense_hash/dense_hash.h>
#include <util/stream/output.h>

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

        bool operator==(ui32 rhs) const {
            return Id == rhs;
        }

        bool operator!=(ui32 rhs) const {
            return !(rhs == *this);
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

        Y_SAVELOAD_DEFINE(Id);
    };

    class TText : public TDenseHash<TTokenId, ui32> {
        using TBase = TDenseHash<TTokenId, ui32>;
    public:
        TText() : TBase() {}

        TText(std::initializer_list<std::pair<TTokenId, ui32>> initializerList)
            : TBase()
        {
            for (const auto& value: initializerList) {
                insert(value);
            }
        }

        bool operator!=(const TText& rhs) const {
            return !(*this == rhs);
        }
    };
}

template <>
inline void Out<NCB::TText>(IOutputStream& stream, const NCB::TText& text) {
    for (const auto& [tokenId, count] : text) {
        stream << "TokenId=" << static_cast<ui32>(tokenId) << ", Count=" << count << Endl;
    }
}
