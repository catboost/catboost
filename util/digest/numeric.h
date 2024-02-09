#pragma once

#include <util/generic/typelist.h>
#include <util/system/defaults.h>

/*
 * original url (now dead): http://www.cris.com/~Ttwang/tech/inthash.htm
 * copy: https://gist.github.com/badboy/6267743
 */

static constexpr ui8 IntHashImpl(ui8 key8) noexcept {
    size_t key = key8;

    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);

    return static_cast<ui8>(key);
}

static constexpr ui16 IntHashImpl(ui16 key16) noexcept {
    size_t key = key16;

    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);

    return static_cast<ui16>(key);
}

static constexpr ui32 IntHashImpl(ui32 key) noexcept {
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);

    return key;
}

static constexpr ui64 IntHashImpl(ui64 key) noexcept {
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);

    return key;
}

template <class T>
static constexpr T IntHash(T t) noexcept {
    using TCvt = TFixedWidthUnsignedInt<T>;

    return IntHashImpl((TCvt)(t));
}

/*
 * can handle floats && pointers
 */
template <class T>
static constexpr size_t NumericHash(T t) noexcept {
    using TCvt = TFixedWidthUnsignedInt<T>;

    union Y_HIDDEN {
        T t;
        TCvt cvt;
    } u{t};

    return (size_t)IntHash(u.cvt);
}

template <class T>
[[nodiscard]] static constexpr T CombineHashes(T l, T r) noexcept {
    return IntHash(l) ^ r;
}
