#pragma once

#include <util/digest/city.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/guid.h>
#include <util/string/hex.h>
#include <util/str_stl.h>

#include <contrib/libs/cxxsupp/libcxx/include/array>

namespace NCB {

    struct TGuid {
        TGuid() = default;

        explicit TGuid(TGUID guid) {
            CopyN(reinterpret_cast<char*>(guid.dw), GUID_SIZE, Value.data());
        }

        explicit TGuid(const char* string) {
            CopyN(string, GUID_SIZE, Value.data());
        }

        void Swap(TGuid& other) {
            if (this != &other) {
                Value.swap(other.Value);
            }
        }

        TGuid& operator=(TGuid guid) {
            DoSwap(*this, guid);
            return *this;
        }

        bool operator==(const TGuid& rhs) const;
        bool operator!=(const TGuid& rhs) const;

    public:
        static constexpr ui32 GUID_SIZE = 16;
        std::array<char, GUID_SIZE> Value = {"___ILLEGAL_GUID"};
        TArrayRef<ui32> dw = MakeArrayRef(
            reinterpret_cast<ui32*>(Value.data()),
            reinterpret_cast<ui32*>(Value.data() + GUID_SIZE)
        );
    };

    TGuid CreateGuid();

} // ncb

template <>
struct THash<NCB::TGuid> {
    size_t operator()(const NCB::TGuid& guid) const noexcept {
        return CityHash64(guid.Value.data(), NCB::TGuid::GUID_SIZE);
    }
};

inline IOutputStream& operator<<(IOutputStream& out, const NCB::TGuid& guid) {
    out << HexEncode(guid.Value.data(), NCB::TGuid::GUID_SIZE);
    return out;
}
