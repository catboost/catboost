#pragma once

#include "int128_util.h"

#include <util/generic/bitops.h>
#include <util/system/compiler.h>
#include <util/system/defaults.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/string/builder.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

#if !defined(_little_endian_) && !defined(_big_endian_)
    static_assert(false, "Platform endianness is not supported");
#endif

template <bool IsSigned>
class TInteger128 {
public:
    TInteger128() noexcept = default;

#if defined(_little_endian_)
    constexpr TInteger128(const ui64 high, const ui64 low) noexcept
        : Low_(low)
        , High_(high)
    {
    }
#elif defined(_big_endian_)
    constexpr TInteger128(const ui64 high, const ui64 low) noexcept
        : High_(high)
        , Low_(low)
    {
    }
#endif

    constexpr TInteger128(const TInteger128<!IsSigned> other) noexcept
        : TInteger128{GetHigh(other), GetLow(other)}
    {
    }

#if defined(_little_endian_)
    constexpr TInteger128(const char other) noexcept
        : Low_{static_cast<ui64>(other)}
        , High_{0}
    {
    }

    constexpr TInteger128(const ui8 other) noexcept
        : Low_{other}
        , High_{0}
    {
    }

    constexpr TInteger128(const ui16 other) noexcept
        : Low_{other}
        , High_{0}
    {
    }

    constexpr TInteger128(const ui32 other) noexcept
        : Low_{other}
        , High_{0}
    {
    }

    constexpr TInteger128(const ui64 other) noexcept
        : Low_{other}
        , High_{0}
    {
    }

#if defined(Y_HAVE_INT128)
    constexpr TInteger128(const unsigned __int128 other) noexcept
        : Low_{static_cast<ui64>(other & ~ui64{0})}
        , High_{static_cast<ui64>(other >> 64)}
    {
    }
#endif

    constexpr TInteger128(const i8 other) noexcept
        : Low_{static_cast<ui64>(other)}
        , High_{other < 0 ? std::numeric_limits<ui64>::max() : 0}
    {
    }

    constexpr TInteger128(const i16 other) noexcept
        : Low_{static_cast<ui64>(other)}
        , High_{other < 0 ? std::numeric_limits<ui64>::max() : 0}
    {
    }

    constexpr TInteger128(const i32 other) noexcept
        : Low_(static_cast<ui64>(other))
        , High_{other < 0 ? std::numeric_limits<ui64>::max() : 0}
    {
    }

    constexpr TInteger128(const i64 other) noexcept
        : Low_(static_cast<ui64>(other))
        , High_{other < 0 ? std::numeric_limits<ui64>::max() : 0}
    {
    }

#if defined(Y_HAVE_INT128)
    template <bool IsSigned2 = IsSigned, std::enable_if_t<!IsSigned2, bool> = false>
    constexpr TInteger128(const signed __int128 other) noexcept
        : Low_{static_cast<ui64>(other & ~ui64{0})}
        , High_{static_cast<ui64>(static_cast<unsigned __int128>(other) >> 64)}
    {
    }

    template <bool IsSigned2 = IsSigned, typename std::enable_if_t<IsSigned2, bool> = false>
    constexpr TInteger128(const signed __int128 other) noexcept
        : Low_{static_cast<ui64>(other & ~ui64(0))}
        , High_{static_cast<ui64>(other >> 64)}
    {
    }
#endif

#elif defined(_big_endian_)
    static_assert(false, "Big-endian will be later");
#endif // _little_endian_ or _big_endian_

    constexpr TInteger128& operator=(const char other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const ui8 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const ui16 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const ui32 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const ui64 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

#if defined(Y_HAVE_INT128)
    constexpr TInteger128& operator=(const unsigned __int128 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }
#endif

    constexpr TInteger128& operator=(const i8 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const i16 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const i32 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

    constexpr TInteger128& operator=(const i64 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }

#if defined(Y_HAVE_INT128)
    constexpr TInteger128& operator=(const signed __int128 other) noexcept {
        *this = TInteger128{other};
        return *this;
    }
#endif // Y_HAVE_INT128

    constexpr TInteger128& operator+=(const TInteger128 other) noexcept {
        return *this = *this + other;
    }

    constexpr TInteger128& operator-=(const TInteger128 other) noexcept {
        return *this = *this - other;
    }

    constexpr TInteger128& operator*=(const TInteger128 other) noexcept {
        return *this = *this * other;
    }

    constexpr TInteger128& operator&=(const TInteger128 other) noexcept {
        return *this = *this & other;
    }

    constexpr TInteger128& operator^=(const TInteger128 other) noexcept {
        return *this = *this ^ other;
    }

    constexpr TInteger128& operator|=(const TInteger128 other) noexcept {
        return *this = *this | other;
    }

    constexpr TInteger128& operator<<=(int n) noexcept {
        *this = *this << n;
        return *this;
    }

    constexpr TInteger128& operator>>=(int n) noexcept {
        *this = *this >> n;
        return *this;
    }

    constexpr TInteger128& operator++() noexcept {
        *this += 1;
        return *this;
    }

    constexpr TInteger128 operator++(int) noexcept {
        const TInteger128 ret{*this};
        this->operator++();
        return ret;
    }

    constexpr TInteger128& operator--() noexcept {
        *this -= 1;
        return *this;
    }

    constexpr TInteger128 operator--(int) noexcept {
        const TInteger128 ret{*this};
        this->operator--();
        return ret;
    }

    explicit constexpr operator bool() const noexcept {
        return Low_ || High_;
    }

    explicit constexpr operator char() const noexcept {
        return static_cast<char>(Low_);
    }

    explicit constexpr operator ui8() const noexcept {
        return static_cast<ui8>(Low_);
    }

    explicit constexpr operator i8() const noexcept {
        return static_cast<i8>(Low_);
    }

    explicit constexpr operator ui16() const noexcept {
        return static_cast<ui16>(Low_);
    }

    explicit constexpr operator i16() const noexcept {
        return static_cast<i16>(Low_);
    }

    explicit constexpr operator ui32() const noexcept {
        return static_cast<ui32>(Low_);
    }

    explicit constexpr operator i32() const noexcept {
        return static_cast<i32>(Low_);
    }

    explicit constexpr operator ui64() const noexcept {
        return static_cast<ui64>(Low_);
    }

    explicit constexpr operator i64() const noexcept {
        return static_cast<i64>(Low_);
    }

#if defined(Y_HAVE_INT128)
    explicit constexpr operator unsigned __int128() const noexcept {
        return (static_cast<unsigned __int128>(High_) << 64) + Low_;
    }

    explicit constexpr operator signed __int128() const noexcept {
        return (static_cast<__int128>(High_) << 64) + Low_;
    }
#endif

private:
#if defined(_little_endian_)
    ui64 Low_;
    ui64 High_;
#elif defined(_big_endian_)
    ui64 High_;
    ui64 Low_;
#endif
    template <bool IsSigned2>
    friend constexpr ui64 GetHigh(TInteger128<IsSigned2> value) noexcept;

    template <bool IsSigned2>
    friend constexpr ui64 GetLow(TInteger128<IsSigned2> value) noexcept;

    friend constexpr bool signbit(TInteger128 arg) noexcept {
        if constexpr (IsSigned) {
            return GetHigh(arg) & 0x8000000000000000;
        } else {
            Y_UNUSED(arg);
            return false;
        }
    }

    friend constexpr TInteger128 abs(TInteger128 arg) noexcept {
        if constexpr (IsSigned) {
            return signbit(arg) ? (-arg) : arg;
        } else {
            return arg;
        }
    }

    friend IOutputStream& operator<<(IOutputStream& out, const TInteger128& other);
}; // class TInteger128

using ui128 = TInteger128<false>;
using i128 = TInteger128<true>;

constexpr ui128 operator+(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator+( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator-(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator-( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator-(ui128 num) noexcept;
constexpr  i128 operator-( i128 num) noexcept;
constexpr ui128 operator*(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator*( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator/(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator/( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator%(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator%( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator|(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator|( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator&(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator&( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator^(ui128 lhs, ui128 rhs) noexcept;
constexpr  i128 operator^( i128 lhs,  i128 rhs) noexcept;
constexpr ui128 operator<<(ui128 lhs, int n) noexcept;
constexpr  i128 operator<<( i128 lhs, int n) noexcept;
constexpr ui128 operator>>(ui128 lhs, int n) noexcept;
constexpr  i128 operator>>( i128 lhs, int n) noexcept;

template <bool IsSigned>
size_t MostSignificantBit(const TInteger128<IsSigned> v);

namespace std {
    //// type traits
    template <bool IsSigned>
    struct is_integral<TInteger128<IsSigned>> : public std::true_type{};

    template <bool IsSigned>
    struct is_class<TInteger128<IsSigned>> : public std::false_type{};

    template <>
    struct is_signed<ui128> : public std::false_type{};

    template <>
    struct is_signed<i128> : public std::true_type{};
}

template <bool IsSigned>
constexpr ui64 GetHigh(const TInteger128<IsSigned> value) noexcept {
    return value.High_;
}

template <bool IsSigned>
constexpr ui64 GetLow(const TInteger128<IsSigned> value) noexcept {
    return value.Low_;
}

template <class T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, i128>>* = nullptr>
constexpr ui128 operator-(const ui128 lhs, const T rhs) noexcept {
    return lhs - static_cast<ui128>(rhs);
}

template <class T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, ui128>>* = nullptr>
constexpr ui128 operator-(const i128 lhs, const T rhs) noexcept {
    return static_cast<ui128>(lhs) - rhs;
}

// specialize std templates
namespace std {
    // numeric limits
    // see full list at https://en.cppreference.com/w/cpp/types/numeric_limits
    template <bool IsSigned>
    struct numeric_limits<TInteger128<IsSigned>> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = IsSigned;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NAN = false;
        static constexpr bool has_signaling_NAN = false;
        static constexpr float_denorm_style has_denorm = std::denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr float_round_style round_style = std::round_toward_zero;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = true;
        static constexpr int digits = CHAR_BIT * sizeof(ui128) - (IsSigned ? 1 : 0);
        static constexpr int digits10 = 38; // std::numeric_limits<ui128>::digits * std::log10(2);
        static constexpr int max_digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = 0;
        static constexpr int min_exponent10 = 0;
        static constexpr int max_exponent = 0;
        static constexpr int max_exponent10 = 0;
        static constexpr bool traps = std::numeric_limits<ui64>::traps; // same as of any other ui*
        static constexpr bool tinyness_before = false;

        static constexpr TInteger128<IsSigned> min() noexcept {
            if constexpr (IsSigned) {
                return TInteger128<IsSigned>{
                    static_cast<ui64>(std::numeric_limits<i64>::min()),
                    0
                };
            }
            else {
                return 0;
            }
        }

        static constexpr TInteger128<IsSigned> lowest() noexcept {
            return min();
        }

        static constexpr TInteger128<IsSigned> max() noexcept {
            if constexpr (IsSigned) {
                return TInteger128<IsSigned>{
                    static_cast<ui64>(std::numeric_limits<i64>::max()),
                    std::numeric_limits<ui64>::max()
                };
            }
            else {
                return TInteger128<IsSigned>{
                    std::numeric_limits<ui64>::max(),
                    std::numeric_limits<ui64>::max()
                };
            }
        }

        static constexpr TInteger128<IsSigned> epsilon() noexcept {
            return 0;
        }

        static constexpr TInteger128<IsSigned> round_error() noexcept {
            return 0;
        }

        static constexpr TInteger128<IsSigned> infinity() noexcept {
            return 0;
        }

        static constexpr TInteger128<IsSigned> quiet_NAN() noexcept {
            return 0;
        }

        static constexpr TInteger128<IsSigned> signaling_NAN() noexcept {
            return 0;
        }

        static constexpr TInteger128<IsSigned> denorm_min() noexcept {
            return 0;
        }
    };
}

constexpr bool operator==(const ui128 lhs, const ui128 rhs) noexcept {
    return GetLow(lhs) == GetLow(rhs) && GetHigh(lhs) == GetHigh(rhs);
}

constexpr bool operator==(const i128 lhs, const i128 rhs) noexcept {
    return GetLow(lhs) == GetLow(rhs) && GetHigh(lhs) == GetHigh(rhs);
}

constexpr bool operator!=(const ui128 lhs, const ui128 rhs) noexcept {
    return !(lhs == rhs);
}

constexpr bool operator!=(const i128 lhs, const i128 rhs) noexcept {
    return !(lhs == rhs);
}

constexpr bool operator<(const ui128 lhs, const ui128 rhs) noexcept {
    if (GetHigh(lhs) != GetHigh(rhs)) {
        return GetHigh(lhs) < GetHigh(rhs);
    }

    return GetLow(lhs) < GetLow(rhs);
}

constexpr bool operator<(const i128 lhs, const i128 rhs) noexcept {
    if (lhs == 0 && rhs == 0) {
        return false;
    }

    const bool lhsIsNegative = signbit(lhs);
    const bool rhsIsNegative = signbit(rhs);

    if (lhsIsNegative && !rhsIsNegative) {
        return true;
    }

    if (!lhsIsNegative && rhsIsNegative) {
        return false;
    }

    // both are negative or both are positive
    if (GetHigh(lhs) != GetHigh(rhs)) {
        return GetHigh(lhs) < GetHigh(rhs);
    }

    return GetLow(lhs) < GetLow(rhs);
}

constexpr bool operator>(const ui128 lhs, const ui128 rhs) noexcept {
    return rhs < lhs;
}

constexpr bool operator>(const i128 lhs, const i128 rhs) noexcept {
    return rhs < lhs;
}

constexpr bool operator<=(const ui128 lhs, const ui128 rhs) noexcept {
    return !(rhs < lhs);
}

constexpr bool operator<=(const i128 lhs, const i128 rhs) noexcept {
    return !(rhs < lhs);
}

constexpr bool operator>=(const ui128 lhs, const ui128 rhs) noexcept {
    return !(lhs < rhs);
}

constexpr bool operator>=(const i128 lhs, const i128 rhs) noexcept {
    return !(lhs < rhs);
}

constexpr ui128 operator+(const ui128 lhs, const ui128 rhs) noexcept {
    const ui128 result{GetHigh(lhs) + GetHigh(rhs), GetLow(lhs) + GetLow(rhs)};
    if (GetLow(result) < GetLow(lhs)) {
        return ui128{GetHigh(result) + 1, GetLow(result)};
    }
    return result;
}

constexpr i128 operator+(const i128 lhs, const i128 rhs) noexcept {
    const i128 result{GetHigh(lhs) + GetHigh(rhs), GetLow(lhs) + GetLow(rhs)};
    if (GetLow(result) < GetLow(lhs)) {
        return i128{GetHigh(result) + 1, GetLow(result)};
    }
    return result;
}

constexpr ui128 operator-(const ui128 lhs, const ui128 rhs) noexcept {
    const ui128 result{GetHigh(lhs) - GetHigh(rhs), GetLow(lhs) - GetLow(rhs)};
    if (GetLow(result) > GetLow(lhs)) { // underflow
        return ui128{GetHigh(result) - 1, GetLow(result)};
    }
    return result;
}

constexpr i128 operator-(const i128 lhs, const i128 rhs) noexcept {
    const i128 result{GetHigh(lhs) - GetHigh(rhs), GetLow(lhs) - GetLow(rhs)};
    if (GetLow(result) > GetLow(lhs)) { // underflow
        return i128{GetHigh(result) - 1, GetLow(result)};
    }
    return result;
}

constexpr ui128 operator-(const ui128 num) noexcept {
    const ui128 result{~GetHigh(num), ~GetLow(num) + 1};
    if (GetLow(result) == 0) {
        return ui128{GetHigh(result) + 1, GetLow(result)};
    }
    return result;
}

constexpr i128 operator-(const i128 num) noexcept {
    const i128 result{~GetHigh(num), ~GetLow(num) + 1};
    if (GetLow(result) == 0) {
        return i128{GetHigh(result) + 1, GetLow(result)};
    }
    return result;
}

constexpr ui128 operator*(const ui128 lhs, const ui128 rhs) noexcept {
    if (rhs == 0) {
        return 0;
    }
    if (rhs == 1) {
        return lhs;
    }

    ui128 result{};
    ui128 t = rhs;

    for (size_t i = 0; i < 128; ++i) {
        if ((t & 1) != 0) {
            result += (lhs << i);
        }

        t = t >> 1;
    }

    return result;
}

constexpr i128 operator*(const i128 lhs, const i128 rhs) noexcept {
    if (rhs == 0) {
        return 0;
    }
    if (rhs == 1) {
        return lhs;
    }

    i128 result{};
    i128 t = rhs;

    for (size_t i = 0; i < 128; ++i) {
        if ((t & 1) != 0) {
            result += (lhs << i);
        }

        t = t >> 1;
    }

    return result;
}

namespace NPrivateInt128 {
    // NOTE: division by zero is UB and can be changed in future
    constexpr void DivMod128(const ui128 lhs, const ui128 rhs, ui128* const quo, ui128* const rem) {
        if (!quo && !rem) {
            return;
        }

        constexpr size_t n_udword_bits = sizeof(ui64) * CHAR_BIT;
        constexpr size_t n_utword_bits = sizeof(ui128) * CHAR_BIT;

        ui128 q{};
        ui128 r{};

        unsigned sr{};

        /* special cases, X is unknown, K != 0 */
        if (GetHigh(lhs) == 0)
        {
            if (GetHigh(rhs) == 0)
            {
                /* 0 X
                 * ---
                 * 0 X
                 */
                if (rem) {
                    *rem = GetLow(lhs) % GetLow(rhs);
                }
                if (quo) {
                    *quo = GetLow(lhs) / GetLow(rhs);
                }
                return;
            }
            /* 0 X
             * ---
             * K X
             */
            if (rem) {
                *rem = GetLow(lhs);
            }
            if (quo) {
                *quo = 0;
            }
            return;
        }
        /* n.s.high != 0 */
        if (GetLow(rhs) == 0)
        {
            if (GetHigh(rhs) == 0)
            {
                /* K X
                 * ---
                 * 0 0
                 */
                if (rem) {
                    *rem = GetHigh(lhs) % GetLow(rhs);
                }
                if (quo) {
                    *quo = GetHigh(lhs) / GetLow(rhs);
                }
                return;
            }
            /* d.s.high != 0 */
            if (GetLow(lhs) == 0)
            {
                /* K 0
                 * ---
                 * K 0
                 */
                if (rem) {
                    *rem = ui128{GetHigh(lhs) % GetHigh(rhs), 0};
                }
                if (quo) {
                    *quo = GetHigh(lhs) / GetHigh(rhs);
                }
                return;
            }
            /* K K
             * ---
             * K 0
             */
            if ((GetHigh(rhs) & (GetHigh(rhs) - 1)) == 0)     /* if d is a power of 2 */
            {
                if (rem) {
                    *rem = ui128{GetHigh(lhs) & (GetHigh(rhs) - 1), GetLow(lhs)};
                }
                if (quo) {
                    *quo = GetHigh(lhs) >> CountLeadingZeroBits(GetHigh(rhs));
                }
                return;
            }
            /* K K
             * ---
             * K 0
             */
            sr = CountLeadingZeroBits(GetHigh(rhs)) - CountLeadingZeroBits(GetHigh(lhs));
            /* 0 <= sr <= n_udword_bits - 2 or sr large */
            if (sr > n_udword_bits - 2)
            {
                if (rem) {
                    *rem = lhs;
                }
                if (quo) {
                    *quo = 0;
                }
                return;
            }
            ++sr;
            /* 1 <= sr <= n_udword_bits - 1 */
            /* q.all = n.all << (n_utword_bits - sr); */
            q = ui128{
                GetLow(lhs) << (n_udword_bits - sr),
                0
            };
            r = ui128{
                GetHigh(lhs) >> sr,
                (GetHigh(lhs) << (n_udword_bits - sr)) | (GetLow(lhs) >> sr)
            };
        }
        else  /* d.s.low != 0 */
        {
            if (GetHigh(rhs) == 0)
            {
                /* K X
                * ---
                * 0 K
                */
                if ((GetLow(rhs) & (GetLow(rhs) - 1)) == 0)     /* if d is a power of 2 */
                {
                    if (rem) {
                        *rem = ui128{0, GetLow(lhs) & (GetLow(rhs) - 1)};
                    }
                    if (GetLow(rhs) == 1) {
                        if (quo) {
                            *quo = lhs;
                        }
                        return;
                    }
                    sr = CountTrailingZeroBits(GetLow(rhs));
                    if (quo) {
                        *quo = ui128{
                            GetHigh(lhs) >> sr,
                            (GetHigh(lhs) << (n_udword_bits - sr)) | (GetLow(lhs) >> sr)
                        };
                        return;
                    }
                }
                /* K X
                * ---
                * 0 K
                */
                sr = 1 + n_udword_bits + CountLeadingZeroBits(GetLow(rhs))
                                    - CountLeadingZeroBits(GetHigh(lhs));
                /* 2 <= sr <= n_utword_bits - 1
                * q.all = n.all << (n_utword_bits - sr);
                * r.all = n.all >> sr;
                */
                if (sr == n_udword_bits)
                {
                    q = ui128{GetLow(lhs), 0};
                    r = ui128{0, GetHigh(lhs)};
                }
                else if (sr < n_udword_bits)  // 2 <= sr <= n_udword_bits - 1
                {
                    q = ui128{
                        GetLow(lhs) << (n_udword_bits - sr),
                        0
                    };
                    r = ui128{
                        GetHigh(lhs) >> sr,
                        (GetHigh(lhs) << (n_udword_bits - sr)) | (GetLow(lhs) >> sr)
                    };
                }
                else              // n_udword_bits + 1 <= sr <= n_utword_bits - 1
                {
                    q = ui128{
                        (GetHigh(lhs) << (n_utword_bits - sr)) | (GetLow(lhs) >> (sr - n_udword_bits)),
                        GetLow(lhs) << (n_utword_bits - sr)
                    };
                    r = ui128{
                        0,
                        GetHigh(lhs) >> (sr - n_udword_bits)
                    };
                }
            }
            else
            {
                /* K X
                * ---
                * K K
                */
                sr = CountLeadingZeroBits(GetHigh(rhs)) - CountLeadingZeroBits(GetHigh(lhs));
                /*0 <= sr <= n_udword_bits - 1 or sr large */
                if (sr > n_udword_bits - 1)
                {
                    if (rem) {
                        *rem = lhs;
                    }
                    if (quo) {
                        *quo = 0;
                    }
                    return;
                }
                ++sr;
                /* 1 <= sr <= n_udword_bits
                * q.all = n.all << (n_utword_bits - sr);
                * r.all = n.all >> sr;
                */
                if (sr == n_udword_bits)
                {
                    q = ui128{
                        GetLow(lhs),
                        0
                    };
                    r = ui128{
                        0,
                        GetHigh(lhs)
                    };
                }
                else
                {
                    r = ui128{
                        GetHigh(lhs) >> sr,
                        (GetHigh(lhs) << (n_udword_bits - sr)) | (GetLow(lhs) >> sr)
                    };
                    q = ui128{
                        GetLow(lhs) << (n_udword_bits - sr),
                        0
                    };
                }
            }
        }
        /* Not a special case
         * q and r are initialized with:
         * q = n << (128 - sr);
         * r = n >> sr;
         * 1 <= sr <= 128 - 1
         */
        ui32 carry = 0;
        for (; sr > 0; --sr)
        {
            /* r:q = ((r:q)  << 1) | carry */
            r = ui128{
                (GetHigh(r) << 1) | (GetLow(r)  >> (n_udword_bits - 1)),
                (GetLow(r)  << 1) | (GetHigh(q) >> (n_udword_bits - 1))
            };
            q = ui128{
                (GetHigh(q) << 1) | (GetLow(q)  >> (n_udword_bits - 1)),
                (GetLow(q)  << 1) | carry
            };
            carry = 0;
            if (r >= rhs) {
                r -= rhs;
                carry = 1;
            }
        }
        q = (q << 1) | carry;
        if (rem) {
            *rem = r;
        }
        if (quo) {
            *quo = q;
        }
    }

    struct TSignedDivisionResult {
        i128 Quotient;
        i128 Remainder;
    };

    constexpr TSignedDivisionResult Divide(i128 lhs, i128 rhs) noexcept;
}

constexpr ui128 operator/(const ui128 lhs, const ui128 rhs) noexcept {
    ui128 quotient{};
    NPrivateInt128::DivMod128(lhs, rhs, &quotient, nullptr);
    return quotient;
}

constexpr i128 operator/(const i128 lhs, const i128 rhs) noexcept {
    i128 a = abs(lhs);
    i128 b = abs(rhs);

    ui128 quotient{};
    NPrivateInt128::DivMod128(a, b, &quotient, nullptr);
    if (signbit(lhs) ^ signbit(rhs)) {
        quotient = -quotient;
    }
    return quotient;
}

constexpr ui128 operator%(const ui128 lhs, const ui128 rhs) noexcept {
    ui128 remainder{};
    NPrivateInt128::DivMod128(lhs, rhs, nullptr, &remainder);
    return remainder;
}

constexpr i128 operator%(const i128 lhs, const i128 rhs) noexcept {
    i128 a = abs(lhs);
    i128 b = abs(rhs);
    ui128 remainder{};
    NPrivateInt128::DivMod128(a, b, nullptr, &remainder);
    if (signbit(lhs)) {
        remainder = -remainder;
    }
    return remainder;
}

constexpr ui128 operator<<(const ui128 lhs, int n) noexcept {
    if (n < 64) {
        if (n != 0) {
            return
                ui128{
                    (GetHigh(lhs) << n) | (GetLow(lhs) >> (64 - n)),
                    GetLow(lhs) << n
                };
        }
        return lhs;
    }
    return ui128{GetLow(lhs) << (n - 64), 0};
}

constexpr ui128 operator>>(const ui128 lhs, int n) noexcept {
    if (n < 64) {
        if (n != 0) {
            return
                ui128{
                    GetHigh(lhs) >> n,
                    (GetLow(lhs) >> n) | (GetHigh(lhs) << (64 - n))
                };
        }
        return lhs;
    }
    return ui128{0, GetHigh(lhs) >> (n - 64)};
}


constexpr bool operator!(const ui128 num) noexcept {
    return !GetHigh(num) && !GetLow(num);
}

constexpr ui128 operator~(const ui128 num) noexcept {
    return ui128{~GetHigh(num), ~GetLow(num)};
}

constexpr ui128 operator|(const ui128 lhs, const ui128 rhs) noexcept {
    return ui128{GetHigh(lhs) | GetHigh(rhs), GetLow(lhs) | GetLow(rhs)};
}

constexpr ui128 operator&(const ui128 lhs, const ui128 rhs) noexcept {
    return ui128{GetHigh(lhs) & GetHigh(rhs), GetLow(lhs) & GetLow(rhs)};
}

constexpr  ui128 operator^(const ui128 lhs, const ui128 rhs) noexcept {
    return ui128{GetHigh(lhs) ^ GetHigh(rhs), GetLow(lhs) ^ GetLow(rhs)};
}


IOutputStream& operator<<(IOutputStream& out, const ui128& other);

// For THashMap
template <>
struct THash<ui128> {
    inline size_t operator()(const ui128& num) const {
        return THash<ui64>()(GetHigh(num)) + THash<ui64>()(GetLow(num));
    }
};

template <>
class TSerializer<ui128> {
public:
    static void Save(IOutputStream* out, const ui128& Number);
    static void Load(IInputStream* in, ui128& Number);
};

template <>
inline TString ToString<ui128>(const ui128& number) {
    return TStringBuilder{} << number;
}

template <>
inline ui128 FromStringImpl<ui128>(const char* data, size_t length) {
    if (length < 20) {
        return ui128{ FromString<ui64>(data, length) };
    } else {
        ui128 result = 0;
        const TStringBuf string(data, length);
        for (auto&& c : string) {
            if (!std::isdigit(c)) {
                ythrow TFromStringException() << "Unexpected symbol \""sv << c << "\""sv;
            }

            ui128 x1 = result;
            ui128 x2 = x1 + x1;
            ui128 x4 = x2 + x2;
            ui128 x8 = x4 + x4;
            ui128 x10 = x8 + x2;
            ui128 s = c - '0';
            result = x10 + s;

            if (GetHigh(result) < GetHigh(x1)) {
                ythrow TFromStringException() << TStringBuf("Integer overflow");
            }
        }

        return result;
    }
}

#if defined(Y_HAVE_INT128)
template <>
inline TString ToString<unsigned __int128>(const unsigned __int128& number) {
    return ToString(ui128{number});
}

template <>
inline unsigned __int128 FromStringImpl<unsigned __int128>(const char* data, size_t length) {
    return static_cast<unsigned __int128>(FromString<ui128>(data, length));
}
#endif

// operators


namespace NPrivateInt128 {
    // very naive algorithm of division
    // no contract for divide by zero (i.e. it is UB) (may be changed in future)
    constexpr TSignedDivisionResult Divide(i128 lhs, i128 rhs) noexcept {
        TSignedDivisionResult result {};

        // check trivial cases
        // X/0 = +/- inf, X%0 = X
        if (rhs == 0) {
            // UB, let's return: `X / 0 = +inf`, and `X % 0 = X`
            result.Quotient = signbit(lhs) ? std::numeric_limits<i128>::min() : std::numeric_limits<i128>::max();
            result.Remainder = lhs;
        }

        // 0/Y = 0, 0%Y = 0
        else if (lhs == 0) {
            result.Quotient = 0;
            result.Remainder = 0;
        }

        // X/1 = X, X%1 = 0
        else if (rhs == 1) {
            result.Quotient = lhs;
            result.Remainder = 0;
        }

        // X/-1 = -X, X%(-1) = 0
        else if (rhs == -1) {
            result.Quotient = -lhs;
            result.Remainder = 0;
        }

        // abs(X)<abs(Y), X/Y = 0, X%Y = X
        else if (abs(lhs) < abs(rhs)) {
            result.Quotient = 0;
            result.Remainder = lhs;
        }

        else if (lhs == rhs) {
            result.Quotient = 1;
            result.Remainder = 0;
        }

        else if (lhs == -rhs) {
            result.Quotient = -1;
            result.Remainder = 0;
        }

        else if (abs(lhs) > abs(rhs)) {
            const bool quotientMustBeNegative = signbit(lhs) ^ signbit(rhs);
            const bool remainderMustBeNegative = signbit(lhs);

            lhs = abs(lhs);
            rhs = abs(rhs);

            // result is division of two ui64
            if (GetHigh(lhs) == 0 && GetHigh(rhs) == 0) {
                result.Quotient = GetLow(lhs) / GetLow(rhs);
                result.Remainder = GetLow(lhs) % GetLow(rhs);
            }

            // naive shift-and-subtract
            // https://stackoverflow.com/questions/5386377/division-without-using
            i128 denominator = rhs;
            result.Quotient = 0;
            result.Remainder = lhs;

            const size_t shift = MostSignificantBit(lhs) - MostSignificantBit(denominator);
            denominator <<= shift;

            for (size_t i = 0; i <= shift; ++i) {
                result.Quotient <<= 1;
                if (result.Remainder >= denominator) {
                    result.Remainder -= denominator;
                    result.Quotient |= 1;
                }
                denominator >>= 1;
            }

            if (quotientMustBeNegative) {
                result.Quotient = -result.Quotient;
            }

            if (remainderMustBeNegative) {
                result.Remainder = -result.Remainder;
            }
        }

        return result;
    }
} // namespace NPrivateInt128

constexpr i128 operator<<(const i128 lhs, int n) noexcept {
    if (n < 64) {
        if (n != 0) {
            return
                i128{
                    (GetHigh(lhs) << n) | (GetLow(lhs) >> (64 - n)),
                    GetLow(lhs) << n
                };
        }
        return lhs;
    }
    return i128{GetLow(lhs) << (n - 64), 0};
}

constexpr i128 operator>>(const i128 lhs, int n) noexcept {
    if (n < 64) {
        if (n != 0) {
            return
                i128{
                    GetHigh(lhs) >> n,
                    (GetLow(lhs) >> n) | (GetHigh(lhs) << (64 - n))
                };
        }
        return lhs;
    }
    return i128{0, GetHigh(lhs) >> (n - 64)};
}

constexpr bool operator!(const i128 num) noexcept {
    return !GetHigh(num) && !GetLow(num);
}

constexpr i128 operator~(const i128 num) noexcept {
    return i128{~GetHigh(num), ~GetLow(num)};
}

constexpr i128 operator|(const i128 lhs, const i128 rhs) noexcept {
    return i128{GetHigh(lhs) | GetHigh(rhs), GetLow(lhs) | GetLow(rhs)};
}

constexpr i128 operator&(const i128 lhs, const i128 rhs) noexcept {
    return i128{GetHigh(lhs) & GetHigh(rhs), GetLow(lhs) & GetLow(rhs)};
}

constexpr i128 operator^(const i128 lhs, const i128 rhs) noexcept {
    return i128{GetHigh(lhs) ^ GetHigh(rhs), GetLow(lhs) ^ GetLow(rhs)};
}


IOutputStream& operator<<(IOutputStream& out, const i128& other);

// For THashMap
template <>
struct THash<i128> {
    inline size_t operator()(const i128& num) const {
        return THash<ui64>()(GetHigh(num)) + THash<ui64>()(GetLow(num));
    }
};

template <>
class TSerializer<i128> {
public:
    static void Save(IOutputStream* out, const i128& Number);
    static void Load(IInputStream* in, i128& Number);
};

template <>
inline TString ToString<i128>(const i128& number) {
    return TStringBuilder{} << number;
}

template <>
inline i128 FromStringImpl<i128>(const char* data, size_t length) {
    if (length < 20) {
        return i128{ FromString<ui64>(data, length) };
    } else {
        i128 result = 0;
        const TStringBuf string(data, length);
        for (auto&& c : string) {
            if (!std::isdigit(c)) {
                ythrow TFromStringException() << "Unexpected symbol \""sv << c << "\""sv;
            }

            i128 x1 = result;
            i128 x2 = x1 + x1;
            i128 x4 = x2 + x2;
            i128 x8 = x4 + x4;
            i128 x10 = x8 + x2;
            i128 s = c - '0';
            result = x10 + s;

            if (GetHigh(result) < GetHigh(x1)) {
                ythrow TFromStringException() << TStringBuf("Integer overflow");
            }
        }

        return result;
    }
}

#if defined(Y_HAVE_INT128)
template <>
inline TString ToString<signed __int128>(const signed __int128& number) {
    return ToString(i128{number});
}

template <>
inline signed __int128 FromStringImpl<signed __int128>(const char* data, size_t length) {
    return static_cast<signed __int128>(FromString<i128>(data, length));
}
#endif

template <bool IsSigned>
Y_FORCE_INLINE size_t MostSignificantBit(const TInteger128<IsSigned> v) {
    if (ui64 hi = GetHigh(v)) {
        return MostSignificantBit(hi) + 64;
    }
    return MostSignificantBit(GetLow(v));
}
