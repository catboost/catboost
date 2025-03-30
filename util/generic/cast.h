#pragma once

#include "typetraits.h"
#include "yexception.h"

#include <util/system/compat.h>
#include <util/system/type_name.h>
#include <util/system/unaligned_mem.h>
#include <util/system/yassert.h>

#include <cstdlib>

template <class T, class F>
static inline T VerifyDynamicCast(F f) {
    if (!f) {
        return nullptr;
    }

    T ret = dynamic_cast<T>(f);

    Y_ABORT_UNLESS(ret, "verify cast failed");

    return ret;
}

#if !defined(NDEBUG)
    #define USE_DEBUG_CHECKED_CAST
#endif

namespace NPrivate {
    template <typename T, typename F>
    static T DynamicCast(F f) {
        return dynamic_cast<T>(f);
    }
} // namespace NPrivate

/*
 * replacement for dynamic_cast(dynamic_cast in debug mode, else static_cast)
 */
template <class T, class F>
static inline T CheckedCast(F f) {
#if defined(USE_DEBUG_CHECKED_CAST)
    return VerifyDynamicCast<T>(f);
#else
    /* Make sure F is polymorphic.
     * Without this cast, CheckedCast with non-polymorphic F
     * incorrectly compiled without error in release mode.
     */
    {
        auto&& x = &::NPrivate::DynamicCast<T, F>;

        (void)x;
    }

    return static_cast<T>(f);
#endif // USE_DEBUG_CHECKED_CAST
}

/*
 * be polite
 */
#undef USE_DEBUG_CHECKED_CAST

template <bool isUnsigned>
class TInteger;

template <>
class TInteger<true> {
public:
    template <class TUnsigned>
    static constexpr bool IsNegative(TUnsigned) noexcept {
        return false;
    }
};

template <>
class TInteger<false> {
public:
    template <class TSigned>
    static constexpr bool IsNegative(const TSigned value) noexcept {
        return value < 0;
    }
};

template <class TType>
constexpr bool IsNegative(const TType value) noexcept {
    return TInteger<std::is_unsigned<TType>::value>::IsNegative(value);
}

namespace NPrivate {
    template <class T>
    using TUnderlyingTypeOrSelf = typename std::conditional<
        std::is_enum<T>::value,
        std::underlying_type<T>, // Lazy evaluatuion: do not call ::type here, because underlying_type<T> is undefined if T is not an enum.
        std::enable_if<true, T>  // Wrapping T in a class, that has member ::type typedef.
        >::type::type;           // Left ::type is for std::conditional, right ::type is for underlying_type/enable_if

    template <class TSmall, class TLarge>
    struct TSafelyConvertible {
        using TSmallInt = TUnderlyingTypeOrSelf<TSmall>;
        using TLargeInt = TUnderlyingTypeOrSelf<TLarge>;

        static constexpr bool Result = std::is_integral<TSmallInt>::value && std::is_integral<TLargeInt>::value &&
                                       ((std::is_signed<TSmallInt>::value == std::is_signed<TLargeInt>::value && sizeof(TSmallInt) >= sizeof(TLargeInt)) ||
                                        (std::is_signed<TSmallInt>::value && sizeof(TSmallInt) > sizeof(TLargeInt)));
    };

    template <class TLargeInt>
    [[noreturn]] void ThrowBadIntegerCast(const TLargeInt largeInt, const std::type_info& smallIntType, const TStringBuf reason) {
        ythrow TBadCastException() << "Conversion '" << TypeName<TLargeInt>() << '{' << largeInt << "}' to '"
                                   << TypeName(smallIntType)
                                   << "', " << reason;
    }

} // namespace NPrivate

template <class TSmallInt, class TLargeInt>
constexpr std::enable_if_t<::NPrivate::TSafelyConvertible<TSmallInt, TLargeInt>::Result, TSmallInt> SafeIntegerCast(TLargeInt largeInt) noexcept {
    return static_cast<TSmallInt>(largeInt);
}

template <class TSmall, class TLarge>
inline std::enable_if_t<!::NPrivate::TSafelyConvertible<TSmall, TLarge>::Result, TSmall> SafeIntegerCast(TLarge largeInt) {
    using TSmallInt = ::NPrivate::TUnderlyingTypeOrSelf<TSmall>;
    using TLargeInt = ::NPrivate::TUnderlyingTypeOrSelf<TLarge>;

    if constexpr (std::is_unsigned<TSmallInt>::value && std::is_signed<TLargeInt>::value) {
        if (IsNegative(largeInt)) {
            ::NPrivate::ThrowBadIntegerCast(TLargeInt(largeInt), typeid(TSmallInt), "negative value converted to unsigned"sv);
        }
    }

    TSmallInt smallInt = TSmallInt(largeInt);

    if constexpr (std::is_signed<TSmallInt>::value && std::is_unsigned<TLargeInt>::value) {
        if (IsNegative(smallInt)) {
            ::NPrivate::ThrowBadIntegerCast(TLargeInt(largeInt), typeid(TSmallInt), "positive value converted to negative"sv);
        }
    }

    if (TLargeInt(smallInt) != largeInt) {
        ::NPrivate::ThrowBadIntegerCast(TLargeInt(largeInt), typeid(TSmallInt), "loss of data"sv);
    }

    return static_cast<TSmall>(smallInt);
}

template <class TSmallInt, class TLargeInt>
inline TSmallInt IntegerCast(TLargeInt largeInt) noexcept {
    try {
        return SafeIntegerCast<TSmallInt>(largeInt);
    } catch (const yexception& exc) {
        Y_ABORT("IntegerCast: %s", exc.what());
    }
}

/* Convert given enum value to its underlying type. This is just a shortcut for
 * `static_cast<std::underlying_type_t<EEnum>>(enum_)`.
 */
template <typename T>
constexpr std::underlying_type_t<T> ToUnderlying(const T enum_) noexcept {
    return static_cast<std::underlying_type_t<T>>(enum_);
}

// std::bit_cast from c++20
template <class TTarget, class TSource>
TTarget BitCast(const TSource& source) {
    static_assert(sizeof(TSource) == sizeof(TTarget), "Size mismatch");
    static_assert(std::is_trivially_copyable<TSource>::value, "TSource is not trivially copyable");
    static_assert(std::is_trivial<TTarget>::value, "TTarget is not trivial");

    // Support volatile qualifiers.
    // ReadUnaligned does not work with volatile pointers, so cast away
    // volatileness beforehand.
    using TNonvolatileSource = std::remove_volatile_t<TSource>;
    using TNonvolatileTarget = std::remove_volatile_t<TTarget>;

    return ReadUnaligned<TNonvolatileTarget>(&const_cast<const TNonvolatileSource&>(source));
}
