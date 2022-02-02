#pragma once

#include "ylimits.h"
#include "typelist.h"

#include <util/system/compiler.h>
#include <util/system/yassert.h>

#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace NBitOps {
    namespace NPrivate {
        template <unsigned N, typename T>
        struct TClp2Helper {
            static Y_FORCE_INLINE T Calc(T t) noexcept {
                const T prev = TClp2Helper<N / 2, T>::Calc(t);

                return prev | (prev >> N);
            }
        };

        template <typename T>
        struct TClp2Helper<0u, T> {
            static Y_FORCE_INLINE T Calc(T t) noexcept {
                return t - 1;
            }
        };

        extern const ui64 WORD_MASK[];
        extern const ui64 INVERSE_WORD_MASK[];

        // see http://www-graphics.stanford.edu/~seander/bithacks.html#ReverseParallel

        Y_FORCE_INLINE ui64 SwapOddEvenBits(ui64 v) {
            return ((v >> 1ULL) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1ULL);
        }

        Y_FORCE_INLINE ui64 SwapBitPairs(ui64 v) {
            return ((v >> 2ULL) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2ULL);
        }

        Y_FORCE_INLINE ui64 SwapNibbles(ui64 v) {
            return ((v >> 4ULL) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4ULL);
        }

        Y_FORCE_INLINE ui64 SwapOddEvenBytes(ui64 v) {
            return ((v >> 8ULL) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8ULL);
        }

        Y_FORCE_INLINE ui64 SwapBytePairs(ui64 v) {
            return ((v >> 16ULL) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16ULL);
        }

        Y_FORCE_INLINE ui64 SwapByteQuads(ui64 v) {
            return (v >> 32ULL) | (v << 32ULL);
        }

#if defined(__GNUC__)
        inline unsigned GetValueBitCountImpl(unsigned int value) noexcept {
            Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned int>::digits - __builtin_clz(value);
        }

        inline unsigned GetValueBitCountImpl(unsigned long value) noexcept {
            Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned long>::digits - __builtin_clzl(value);
        }

        inline unsigned GetValueBitCountImpl(unsigned long long value) noexcept {
            Y_ASSERT(value); // because __builtin_clz* have undefined result for zero.
            return std::numeric_limits<unsigned long long>::digits - __builtin_clzll(value);
        }
#else
        /// Stupid realization for non-GCC. Can use BSR from x86 instructions set.
        template <typename T>
        inline unsigned GetValueBitCountImpl(T value) noexcept {
            Y_ASSERT(value);     // because __builtin_clz* have undefined result for zero.
            unsigned result = 1; // result == 0 - impossible value, see Y_ASSERT().
            value >>= 1;
            while (value) {
                value >>= 1;
                ++result;
            }

            return result;
        }
#endif

#if defined(__GNUC__)
        inline unsigned CountTrailingZeroBitsImpl(unsigned int value) noexcept {
            Y_ASSERT(value); // because __builtin_ctz* have undefined result for zero.
            return __builtin_ctz(value);
        }

        inline unsigned CountTrailingZeroBitsImpl(unsigned long value) noexcept {
            Y_ASSERT(value); // because __builtin_ctz* have undefined result for zero.
            return __builtin_ctzl(value);
        }

        inline unsigned CountTrailingZeroBitsImpl(unsigned long long value) noexcept {
            Y_ASSERT(value); // because __builtin_ctz* have undefined result for zero.
            return __builtin_ctzll(value);
        }
#else
        /// Stupid realization for non-GCC. Can use BSF from x86 instructions set.
        template <typename T>
        inline unsigned CountTrailingZeroBitsImpl(T value) noexcept {
            Y_ASSERT(value); // because __builtin_ctz* have undefined result for zero.
            unsigned result = 0;
            while (!(value & 1)) {
                value >>= 1;
                ++result;
            }

            return result;
        }
#endif

        template <typename T>
        Y_FORCE_INLINE T RotateBitsLeftImpl(T value, const ui8 shift) noexcept {
            constexpr ui8 bits = sizeof(T) * 8;
            constexpr ui8 mask = bits - 1;
            Y_ASSERT(shift <= mask);

            // do trick with mask to avoid undefined behaviour
            return (value << shift) | (value >> ((-shift) & mask));
        }

        template <typename T>
        Y_FORCE_INLINE T RotateBitsRightImpl(T value, const ui8 shift) noexcept {
            constexpr ui8 bits = sizeof(T) * 8;
            constexpr ui8 mask = bits - 1;
            Y_ASSERT(shift <= mask);

            // do trick with mask to avoid undefined behaviour
            return (value >> shift) | (value << ((-shift) & mask));
        }

#if defined(_x86_) && defined(__GNUC__)
        Y_FORCE_INLINE ui8 RotateBitsRightImpl(ui8 value, ui8 shift) noexcept {
            __asm__("rorb   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui16 RotateBitsRightImpl(ui16 value, ui8 shift) noexcept {
            __asm__("rorw   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui32 RotateBitsRightImpl(ui32 value, ui8 shift) noexcept {
            __asm__("rorl   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui8 RotateBitsLeftImpl(ui8 value, ui8 shift) noexcept {
            __asm__("rolb   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui16 RotateBitsLeftImpl(ui16 value, ui8 shift) noexcept {
            __asm__("rolw   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui32 RotateBitsLeftImpl(ui32 value, ui8 shift) noexcept {
            __asm__("roll   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

    #if defined(_x86_64_)
        Y_FORCE_INLINE ui64 RotateBitsRightImpl(ui64 value, ui8 shift) noexcept {
            __asm__("rorq   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }

        Y_FORCE_INLINE ui64 RotateBitsLeftImpl(ui64 value, ui8 shift) noexcept {
            __asm__("rolq   %%cl, %0"
                    : "=r"(value)
                    : "0"(value), "c"(shift));
            return value;
        }
    #endif
#endif
    }
}

/**
 * Computes the next power of 2 higher or equal to the integer parameter `t`.
 * If `t` is a power of 2 will return `t`.
 * Result is undefined for `t == 0`.
 */
template <typename T>
static inline T FastClp2(T t) noexcept {
    Y_ASSERT(t > 0);
    using TCvt = typename ::TUnsignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;
    return 1 + ::NBitOps::NPrivate::TClp2Helper<sizeof(TCvt) * 4, T>::Calc(static_cast<TCvt>(t));
}

/**
 * Check if integer is a power of 2.
 */
template <typename T>
Y_CONST_FUNCTION constexpr bool IsPowerOf2(T v) noexcept {
    return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Returns the number of leading 0-bits in `value`, starting at the most significant bit position.
 */
template <typename T>
static inline unsigned GetValueBitCount(T value) noexcept {
    Y_ASSERT(value > 0);
    using TCvt = typename ::TUnsignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;
    return ::NBitOps::NPrivate::GetValueBitCountImpl(static_cast<TCvt>(value));
}

/**
 * Returns the number of trailing 0-bits in `value`, starting at the least significant bit position
 */
template <typename T>
static inline unsigned CountTrailingZeroBits(T value) noexcept {
    Y_ASSERT(value > 0);
    using TCvt = typename ::TUnsignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::type;
    return ::NBitOps::NPrivate::CountTrailingZeroBitsImpl(static_cast<TCvt>(value));
}

/*
 * Returns 64-bit mask with `bits` lower bits set.
 */
Y_FORCE_INLINE ui64 MaskLowerBits(ui64 bits) {
    return ::NBitOps::NPrivate::WORD_MASK[bits];
}

/*
 * Return 64-bit mask with `bits` set starting from `skipbits`.
 */
Y_FORCE_INLINE ui64 MaskLowerBits(ui64 bits, ui64 skipbits) {
    return MaskLowerBits(bits) << skipbits;
}

/*
 * Return 64-bit mask with all bits set except for `bits` lower bits.
 */
Y_FORCE_INLINE ui64 InverseMaskLowerBits(ui64 bits) {
    return ::NBitOps::NPrivate::INVERSE_WORD_MASK[bits];
}

/*
 * Return 64-bit mask with all bits set except for `bits` bitst starting from `skipbits`.
 */
Y_FORCE_INLINE ui64 InverseMaskLowerBits(ui64 bits, ui64 skipbits) {
    return ~MaskLowerBits(bits, skipbits);
}

/*
 * Returns 0-based position of the most significant bit that is set. 0 for 0.
 */
Y_FORCE_INLINE ui64 MostSignificantBit(ui64 v) {
#ifdef __GNUC__
    ui64 res = v ? (63 - __builtin_clzll(v)) : 0;
#elif defined(_MSC_VER) && defined(_64_)
    unsigned long res = 0;
    if (v)
        _BitScanReverse64(&res, v);
#else
    ui64 res = 0;
    if (v)
        while (v >>= 1)
            ++res;
#endif
    return res;
}

/**
 * Returns 0-based position of the least significant bit that is set. 0 for 0.
 */
Y_FORCE_INLINE ui64 LeastSignificantBit(ui64 v) {
#ifdef __GNUC__
    ui64 res = v ? __builtin_ffsll(v) - 1 : 0;
#elif defined(_MSC_VER) && defined(_64_)
    unsigned long res = 0;
    if (v)
        _BitScanForward64(&res, v);
#else
    ui64 res = 0;
    if (v) {
        while (!(v & 1)) {
            ++res;
            v >>= 1;
        }
    }
#endif
    return res;
}

/*
 * Returns 0 - based position of the most significant bit (compile time)
 * 0 for 0.
 */
constexpr ui64 MostSignificantBitCT(ui64 x) {
    return x > 1 ? 1 + MostSignificantBitCT(x >> 1) : 0;
}

/*
 * Return rounded up binary logarithm of `x`.
 */
Y_FORCE_INLINE ui8 CeilLog2(ui64 x) {
    return static_cast<ui8>(MostSignificantBit(x - 1)) + 1;
}

Y_FORCE_INLINE ui8 ReverseBytes(ui8 t) {
    return t;
}

Y_FORCE_INLINE ui16 ReverseBytes(ui16 t) {
    return static_cast<ui16>(::NBitOps::NPrivate::SwapOddEvenBytes(t));
}

Y_FORCE_INLINE ui32 ReverseBytes(ui32 t) {
    return static_cast<ui32>(::NBitOps::NPrivate::SwapBytePairs(
        ::NBitOps::NPrivate::SwapOddEvenBytes(t)));
}

Y_FORCE_INLINE ui64 ReverseBytes(ui64 t) {
    return ::NBitOps::NPrivate::SwapByteQuads((::NBitOps::NPrivate::SwapOddEvenBytes(t)));
}

Y_FORCE_INLINE ui8 ReverseBits(ui8 t) {
    return static_cast<ui8>(::NBitOps::NPrivate::SwapNibbles(
        ::NBitOps::NPrivate::SwapBitPairs(
            ::NBitOps::NPrivate::SwapOddEvenBits(t))));
}

Y_FORCE_INLINE ui16 ReverseBits(ui16 t) {
    return static_cast<ui16>(::NBitOps::NPrivate::SwapOddEvenBytes(
        ::NBitOps::NPrivate::SwapNibbles(
            ::NBitOps::NPrivate::SwapBitPairs(
                ::NBitOps::NPrivate::SwapOddEvenBits(t)))));
}

Y_FORCE_INLINE ui32 ReverseBits(ui32 t) {
    return static_cast<ui32>(::NBitOps::NPrivate::SwapBytePairs(
        ::NBitOps::NPrivate::SwapOddEvenBytes(
            ::NBitOps::NPrivate::SwapNibbles(
                ::NBitOps::NPrivate::SwapBitPairs(
                    ::NBitOps::NPrivate::SwapOddEvenBits(t))))));
}

Y_FORCE_INLINE ui64 ReverseBits(ui64 t) {
    return ::NBitOps::NPrivate::SwapByteQuads(
        ::NBitOps::NPrivate::SwapBytePairs(
            ::NBitOps::NPrivate::SwapOddEvenBytes(
                ::NBitOps::NPrivate::SwapNibbles(
                    ::NBitOps::NPrivate::SwapBitPairs(
                        ::NBitOps::NPrivate::SwapOddEvenBits(t))))));
}

/*
 * Reverse first `bits` bits
 * 1000111000111000 , bits = 6 => 1000111000000111
 */
template <typename T>
Y_FORCE_INLINE T ReverseBits(T v, ui64 bits) {
    return bits ? (T(v & ::InverseMaskLowerBits(bits)) | T(ReverseBits(T(v & ::MaskLowerBits(bits)))) >> ((ui64{sizeof(T)} << ui64{3}) - bits)) : v;
}

/*
 * Referse first `bits` bits starting from `skipbits` bits
 * 1000111000111000 , bits = 4, skipbits = 2 => 1000111000011100
 */
template <typename T>
Y_FORCE_INLINE T ReverseBits(T v, ui64 bits, ui64 skipbits) {
    return (T(ReverseBits((v >> skipbits), bits)) << skipbits) | T(v & MaskLowerBits(skipbits));
}

/* Rotate bits left. Also known as left circular shift.
 */
template <typename T>
Y_FORCE_INLINE T RotateBitsLeft(T value, const ui8 shift) noexcept {
    static_assert(std::is_unsigned<T>::value, "must be unsigned arithmetic type");
    return ::NBitOps::NPrivate::RotateBitsLeftImpl((TFixedWidthUnsignedInt<T>)value, shift);
}

/* Rotate bits right. Also known as right circular shift.
 */
template <typename T>
Y_FORCE_INLINE T RotateBitsRight(T value, const ui8 shift) noexcept {
    static_assert(std::is_unsigned<T>::value, "must be unsigned arithmetic type");
    return ::NBitOps::NPrivate::RotateBitsRightImpl((TFixedWidthUnsignedInt<T>)value, shift);
}

/* Rotate bits left. Also known as left circular shift.
 */
template <typename T>
constexpr T RotateBitsLeftCT(T value, const ui8 shift) noexcept {
    static_assert(std::is_unsigned<T>::value, "must be unsigned arithmetic type");

    // do trick with mask to avoid undefined behaviour
    return (value << shift) | (value >> ((-shift) & (sizeof(T) * 8 - 1)));
}

/* Rotate bits right. Also known as right circular shift.
 */
template <typename T>
constexpr T RotateBitsRightCT(T value, const ui8 shift) noexcept {
    static_assert(std::is_unsigned<T>::value, "must be unsigned arithmetic type");

    // do trick with mask to avoid undefined behaviour
    return (value >> shift) | (value << ((-shift) & (sizeof(T) * 8 - 1)));
}

/* Remain `size` bits to current `offset` of `value`
   size, offset are less than number of bits in size_type
 */
template <size_t Offset, size_t Size, class T>
Y_FORCE_INLINE T SelectBits(T value) {
    static_assert(Size < sizeof(T) * 8, "violated: Size < sizeof(T) * 8");
    static_assert(Offset < sizeof(T) * 8, "violated: Offset < sizeof(T) * 8");
    T id = 1;
    return (value >> Offset) & ((id << Size) - id);
}

/* Set `size` bits of `bits` to current offset of `value`. Requires that bits <= (1 << size) - 1
   size, offset are less than number of bits in size_type
 */
template <size_t Offset, size_t Size, class T>
void SetBits(T& value, T bits) {
    static_assert(Size < sizeof(T) * 8, "violated: Size < sizeof(T) * 8");
    static_assert(Offset < sizeof(T) * 8, "violated: Offset < sizeof(T) * 8");
    T id = 1;
    T maxValue = ((id << Size) - id);
    Y_ASSERT(bits <= maxValue);
    value &= ~(maxValue << Offset);
    value |= bits << Offset;
}

inline constexpr ui64 NthBit64(int bit) {
    return ui64(1) << bit;
}

inline constexpr ui64 Mask64(int bits) {
    return NthBit64(bits) - 1;
}
