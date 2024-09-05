#pragma once

#include <util/generic/utility.h>
#include <util/generic/strbuf.h>

#include <utility>

// NOTE: These functions provide CityHash 1.0 implementation whose results are *different* from
// the mainline version of CityHash.

using uint128 = std::pair<ui64, ui64>;

constexpr ui64 Uint128Low64(const uint128& x) {
    return x.first;
}

constexpr ui64 Uint128High64(const uint128& x) {
    return x.second;
}

// Hash functions for a byte array.
// http://en.wikipedia.org/wiki/CityHash

Y_PURE_FUNCTION ui64 CityHash64(const char* buf, size_t len) noexcept;

Y_PURE_FUNCTION ui64 CityHash64WithSeed(const char* buf, size_t len, ui64 seed) noexcept;

Y_PURE_FUNCTION ui64 CityHash64WithSeeds(const char* buf, size_t len, ui64 seed0, ui64 seed1) noexcept;

Y_PURE_FUNCTION uint128 CityHash128(const char* s, size_t len) noexcept;

Y_PURE_FUNCTION uint128 CityHash128WithSeed(const char* s, size_t len, uint128 seed) noexcept;

// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
inline ui64 Hash128to64(const uint128& x) {
    // Murmur-inspired hashing.
    const ui64 kMul = 0x9ddfea08eb382d69ULL;
    ui64 a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
    a ^= (a >> 47);
    ui64 b = (Uint128High64(x) ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}

namespace NPrivateCityHash {
    template <class TStringType>
    inline TStringBuf GetBufFromStr(const TStringType& str) {
        static_assert(std::is_integral<std::remove_reference_t<decltype(*str.data())>>::value, "invalid type passed to hash function");
        return TStringBuf(reinterpret_cast<const char*>(str.data()), (str.size()) * sizeof(*str.data()));
    }
} // namespace NPrivateCityHash

template <class TStringType>
inline ui64 CityHash64(const TStringType& str) {
    TStringBuf buf = NPrivateCityHash::GetBufFromStr(str);
    return CityHash64(buf.data(), buf.size());
}

template <class TStringType>
inline ui64 CityHash64WithSeeds(const TStringType& str, ui64 seed0, ui64 seed1) {
    TStringBuf buf = NPrivateCityHash::GetBufFromStr(str);
    return CityHash64WithSeeds(buf.data(), buf.size(), seed0, seed1);
}

template <class TStringType>
inline ui64 CityHash64WithSeed(const TStringType& str, ui64 seed) {
    TStringBuf buf = NPrivateCityHash::GetBufFromStr(str);
    return CityHash64WithSeed(buf.data(), buf.size(), seed);
}

template <class TStringType>
inline uint128 CityHash128(const TStringType& str) {
    TStringBuf buf = NPrivateCityHash::GetBufFromStr(str);
    return CityHash128(buf.data(), buf.size());
}

template <class TStringType>
inline uint128 CityHash128WithSeed(const TStringType& str, uint128 seed) {
    TStringBuf buf = NPrivateCityHash::GetBufFromStr(str);
    return CityHash128WithSeed(buf.data(), buf.size(), seed);
}
