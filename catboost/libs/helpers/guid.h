#pragma once

#include <catboost/libs/helpers/flatbuffers/guid.fbs.h>

#include <util/digest/city.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/guid.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/string/hex.h>
#include <util/str_stl.h>

#include <contrib/libs/cxxsupp/libcxx/include/array>

namespace NCB {

    struct TGuid {
        TGuid() = default;

        TGuid(TGuid&& rhs) {
            Value = std::move(rhs.Value);
            dw = MakeArrayRef(
                reinterpret_cast<ui32*>(Value.data()),
                reinterpret_cast<ui32*>(Value.data() + GUID_SIZE)
            );
        }

        TGuid(const TGuid& rhs) {
            CopyN(rhs.Value.data(), GUID_SIZE, Value.data());
        }

        explicit TGuid(TGUID guid) {
            CopyN(reinterpret_cast<char*>(guid.dw), GUID_SIZE, Value.data());
        }

        explicit TGuid(const char* string) {
            CopyN(string, GUID_SIZE, Value.data());
        }

        void Swap(TGuid& other) {
            if (*this != other) {
                std::swap(dw[0], other.dw[0]);
                std::swap(dw[1], other.dw[1]);
                std::swap(dw[2], other.dw[2]);
                std::swap(dw[3], other.dw[3]);
            }
        }

        bool IsInitialized() const {
            return std::equal(Value.begin(), Value.end(), DefaultValue.begin());
        }

        TGuid& operator=(TGuid guid) {
            DoSwap(*this, guid);
            return *this;
        }

        void Save(IOutputStream* stream) const {
            stream->Write(Value.data(), GUID_SIZE);
        }

        void Load(IInputStream* stream) {
            ui32 readBytes = stream->Load(Value.data(), GUID_SIZE);
            Y_ASSERT(readBytes == GUID_SIZE);
        }

        bool operator<(const TGuid& rhs) const;
        bool operator>(const TGuid& rhs) const;
        bool operator<=(const TGuid& rhs) const;
        bool operator>=(const TGuid& rhs) const;
        bool operator==(const TGuid& rhs) const;
        bool operator!=(const TGuid& rhs) const;

    public:
        static constexpr ui32 GUID_SIZE = 16;
        const std::array<char, GUID_SIZE> DefaultValue = {"___ILLEGAL_GUID"};
        std::array<char, GUID_SIZE> Value = DefaultValue;
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


inline NCB::TGuid GuidFromFbs(const NCatBoostFbs::TGuid* fbsGuid) {
    NCB::TGuid guid;
    guid.dw[0] = fbsGuid->dw0();
    guid.dw[1] = fbsGuid->dw1();
    guid.dw[2] = fbsGuid->dw2();
    guid.dw[3] = fbsGuid->dw3();
    return guid;
}

inline NCatBoostFbs::TGuid CreateFbsGuid(const NCB::TGuid& guid) {
    return NCatBoostFbs::TGuid(guid.dw[0], guid.dw[1], guid.dw[2], guid.dw[3]);
}

