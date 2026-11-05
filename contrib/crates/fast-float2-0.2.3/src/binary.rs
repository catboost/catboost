use crate::common::AdjustedMantissa;
use crate::float::Float;
use crate::table::{LARGEST_POWER_OF_FIVE, POWER_OF_FIVE_128, SMALLEST_POWER_OF_FIVE};

#[inline]
pub fn compute_float<F: Float>(q: i64, mut w: u64) -> AdjustedMantissa {
    let am_zero = AdjustedMantissa::zero_pow2(0);
    let am_inf = AdjustedMantissa::zero_pow2(F::INFINITE_POWER);
    let am_error = AdjustedMantissa::zero_pow2(-1);

    if w == 0 || q < F::SMALLEST_POWER_OF_TEN as i64 {
        return am_zero;
    } else if q > F::LARGEST_POWER_OF_TEN as i64 {
        return am_inf;
    }
    let lz = w.leading_zeros();
    w <<= lz;
    let (lo, hi) = compute_product_approx(q, w, F::MANTISSA_EXPLICIT_BITS + 3);
    if lo == 0xFFFF_FFFF_FFFF_FFFF {
        let inside_safe_exponent = (-27..=55).contains(&q);
        if !inside_safe_exponent {
            return am_error;
        }
    }
    let upperbit = (hi >> 63) as i32;
    let mut mantissa = hi >> (upperbit + 64 - F::MANTISSA_EXPLICIT_BITS as i32 - 3);
    let mut power2 = power(q as i32) + upperbit - lz as i32 - F::MINIMUM_EXPONENT;
    if power2 <= 0 {
        if -power2 + 1 >= 64 {
            return am_zero;
        }
        mantissa >>= -power2 + 1;
        mantissa += mantissa & 1;
        mantissa >>= 1;
        power2 = (mantissa >= (1_u64 << F::MANTISSA_EXPLICIT_BITS)) as i32;
        return AdjustedMantissa {
            mantissa,
            power2,
        };
    }
    if lo <= 1
        && q >= F::MIN_EXPONENT_ROUND_TO_EVEN as i64
        && q <= F::MAX_EXPONENT_ROUND_TO_EVEN as i64
        && mantissa & 3 == 1
        && (mantissa << (upperbit + 64 - F::MANTISSA_EXPLICIT_BITS as i32 - 3)) == hi
    {
        mantissa &= !1_u64;
    }
    mantissa += mantissa & 1;
    mantissa >>= 1;
    if mantissa >= (2_u64 << F::MANTISSA_EXPLICIT_BITS) {
        mantissa = 1_u64 << F::MANTISSA_EXPLICIT_BITS;
        power2 += 1;
    }
    mantissa &= !(1_u64 << F::MANTISSA_EXPLICIT_BITS);
    if power2 >= F::INFINITE_POWER {
        return am_inf;
    }
    AdjustedMantissa {
        mantissa,
        power2,
    }
}

#[inline]
fn power(q: i32) -> i32 {
    (q.wrapping_mul(152_170 + 65536) >> 16) + 63
}

#[inline]
fn full_multiplication(a: u64, b: u64) -> (u64, u64) {
    let r = (a as u128) * (b as u128);
    (r as u64, (r >> 64) as u64)
}

// This will compute or rather approximate w * 5**q and return a pair of 64-bit
// words approximating the result, with the "high" part corresponding to the
// most significant bits and the low part corresponding to the least significant
// bits.
#[inline]
fn compute_product_approx(q: i64, w: u64, precision: usize) -> (u64, u64) {
    debug_assert!(q >= SMALLEST_POWER_OF_FIVE as i64);
    debug_assert!(q <= LARGEST_POWER_OF_FIVE as i64);
    debug_assert!(precision <= 64);

    let mask = if precision < 64 {
        0xFFFF_FFFF_FFFF_FFFF_u64 >> precision
    } else {
        0xFFFF_FFFF_FFFF_FFFF_u64
    };
    let index = (q - SMALLEST_POWER_OF_FIVE as i64) as usize;
    // NOTE: this cannot be ellided by the compiler, but the proof the index
    // must be within the bounds is non-trivial, especially because this
    // comes from a parsed result. Since this is unlikely to have any major
    // performance implications, as is determined empirically, we keep the
    // bounds check despite the performance hit.
    let (lo5, hi5) = POWER_OF_FIVE_128[index];
    let (mut first_lo, mut first_hi) = full_multiplication(w, lo5);
    if first_hi & mask == mask {
        let (_, second_hi) = full_multiplication(w, hi5);
        first_lo = first_lo.wrapping_add(second_hi);
        if second_hi > first_lo {
            first_hi += 1;
        }
    }
    (first_lo, first_hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_multiplication() {
        fn check(a: u64, b: u64, lo: u64, hi: u64) {
            assert_eq!(full_multiplication(a, b), (lo, hi));
            assert_eq!(full_multiplication(b, a), (lo, hi));
        }
        check(1 << 0, 1 << 0, 1, 0);
        check(1 << 0, 1 << 63, 1 << 63, 0);
        check(1 << 1, 1 << 63, 0, 1);
        check(1 << 63, 1 << 0, 1 << 63, 0);
        check(1 << 63, 1 << 1, 0, 1);
        check(1 << 63, 1 << 2, 0, 2);
        check(1 << 63, 1 << 63, 0, 1 << 62);
    }
}
