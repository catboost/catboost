#pragma once

#include <util/system/compiler.h>
#include <util/system/types.h>

#if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
#endif

/**
 * Calculates the number of buckets for the hash table that will hold the given
 * number of elements.
 *
 * @param elementCount                  Number of elements that the hash table will hold.
 * @returns                             Number of buckets, a prime number that is
 *                                      greater or equal to `elementCount`.
 */
Y_CONST_FUNCTION
unsigned long HashBucketCount(unsigned long elementCount);

namespace NPrivate {

    /// Implementation of algorithm 4.1 from: Torbjörn Granlund and Peter L. Montgomery. 1994. Division by invariant integers using multiplication.
    /// @see https://gmplib.org/~tege/divcnst-pldi94.pdf
    template <typename TDivisor, typename TDividend, typename MulUnsignedUpper>
    class TReciprocalDivisor {
        static_assert(sizeof(TDivisor) <= sizeof(TDividend), "TDivisor and TDividend should have the same size");

    public:
        constexpr TReciprocalDivisor() noexcept = default;

        constexpr TReciprocalDivisor(TDividend reciprocal, ui8 reciprocalShift, i8 hint, TDivisor divisor) noexcept
            : Reciprocal(reciprocal)
            , Divisor(divisor)
            , ReciprocalShift(reciprocalShift)
            , Hint(hint)
        {
        }

        /// Return (dividend % Divisor)
        Y_FORCE_INLINE TDividend Remainder(TDividend dividend) const noexcept {
            if (Y_UNLIKELY(Divisor == 1)) {
                return 0;
            }
            TDividend r = dividend - Quotent(dividend) * Divisor;
            return r;
        }

        Y_FORCE_INLINE TDivisor operator()() const noexcept {
            return Divisor;
        }

        Y_FORCE_INLINE static constexpr TReciprocalDivisor One() noexcept {
            return {1u, 0u, -1, 1u};
        }

    private:
        /// returns (dividend / Divisor)
        Y_FORCE_INLINE TDividend Quotent(TDividend dividend) const noexcept {
            const TDividend t = MulUnsignedUpper{}(dividend, Reciprocal);
            const TDividend q = (t + ((dividend - t) >> 1)) >> ReciprocalShift;
            return q;
        }

    public:
        TDividend Reciprocal = 0;
        TDivisor Divisor = 0;
        ui8 ReciprocalShift = 0;
        i8 Hint = 0; ///< Additional data: needless for division, but useful for the adjacent divisors search
    };

    template <typename T, typename TExtended, size_t shift>
    struct TMulUnsignedUpper {
        /// Return the high bits of the product of two unsigned integers.
        Y_FORCE_INLINE T operator()(T a, T b) const noexcept {
            return (static_cast<TExtended>(a) * static_cast<TExtended>(b)) >> shift;
        }
    };

#if defined(_32_)
    using THashDivisor = ::NPrivate::TReciprocalDivisor<ui32, ui32, TMulUnsignedUpper<ui32, ui64, 32>>;
#else
    #if defined(Y_HAVE_INT128)
    using THashDivisor = ::NPrivate::TReciprocalDivisor<ui32, ui64, TMulUnsignedUpper<ui64, unsigned __int128, 64>>;
    #elif defined(_MSC_VER) && defined(_M_X64)
    struct TMulUnsignedUpperVCIntrin {
        /// Return the high 64 bits of the product of two 64-bit unsigned integers.
        Y_FORCE_INLINE ui64 operator()(ui64 a, ui64 b) const noexcept {
            return __umulh(a, b);
        }
    };
    using THashDivisor = ::NPrivate::TReciprocalDivisor<ui32, ui64, TMulUnsignedUpperVCIntrin>;
    #else
    template <typename TDivisor, typename TDividend>
    class TNaiveDivisor {
    public:
        constexpr TNaiveDivisor() noexcept = default;

        constexpr TNaiveDivisor(TDivisor divisor) noexcept
            : Divisor(divisor)
        {
        }

        constexpr TNaiveDivisor(TDividend reciprocal, ui8 reciprocalShift, i8 hint, TDivisor divisor) noexcept
            : TNaiveDivisor(divisor)
        {
            Y_UNUSED(reciprocal);
            Y_UNUSED(reciprocalShift);
            Y_UNUSED(hint);
        }

        Y_FORCE_INLINE TDividend Remainder(TDividend dividend) const noexcept {
            return dividend % Divisor;
        }

        Y_FORCE_INLINE TDivisor operator()() const noexcept {
            return Divisor;
        }

        Y_FORCE_INLINE static constexpr TNaiveDivisor One() noexcept {
            return {1u};
        }

    public:
        TDivisor Divisor = 0;
        static constexpr i8 Hint = -1;
    };

    using THashDivisor = ::NPrivate::TNaiveDivisor<ui32, ui64>;
    #endif
#endif
} // namespace NPrivate

Y_CONST_FUNCTION
::NPrivate::THashDivisor HashBucketCountExt(unsigned long elementCount);

Y_CONST_FUNCTION
::NPrivate::THashDivisor HashBucketCountExt(unsigned long elementCount, int hint);
