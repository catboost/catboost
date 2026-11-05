//! This module contains a Rust port of the `__multi3` compiler builtin that is
//! typically used for implementing 64-bit multiplication on 32-bit platforms.
//!
//! This port is adapted to use 128-bit high and low words and return carry
//! information in order to implement 256-bit overflowing multiplication.
//!
//! This source is ported from LLVM project from C:
//! <https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/builtins/multi3.c>

use crate::{int::I256, uint::U256};
use core::mem::MaybeUninit;

#[inline]
pub fn umulddi3(a: &u128, b: &u128) -> U256 {
    const BITS_IN_DWORD_2: u32 = 64;
    const LOWER_MASK: u128 = u128::MAX >> BITS_IN_DWORD_2;

    let mut low = (a & LOWER_MASK) * (b & LOWER_MASK);
    let mut t = low >> BITS_IN_DWORD_2;
    low &= LOWER_MASK;
    t += (a >> BITS_IN_DWORD_2) * (b & LOWER_MASK);
    low += (t & LOWER_MASK) << BITS_IN_DWORD_2;
    let mut high = t >> BITS_IN_DWORD_2;
    t = low >> BITS_IN_DWORD_2;
    low &= LOWER_MASK;
    t += (b >> BITS_IN_DWORD_2) * (a & LOWER_MASK);
    low += (t & LOWER_MASK) << BITS_IN_DWORD_2;
    high += t >> BITS_IN_DWORD_2;
    high += (a >> BITS_IN_DWORD_2) * (b >> BITS_IN_DWORD_2);

    U256::from_words(high, low)
}

#[inline]
pub fn mul2(r: &mut U256, a: &U256) {
    let (a, b) = (*r, a);
    // SAFETY: `multi3` does not write `MaybeUninit::uninit()` to `res` and
    // `U256` does not implement `Drop`.
    let res = unsafe { &mut *(r as *mut U256).cast() };
    mul3(res, &a, b);
}

#[inline]
pub fn mul3(res: &mut MaybeUninit<U256>, a: &U256, b: &U256) {
    let mut r = umulddi3(a.low(), b.low());

    let hi_lo = a.high().wrapping_mul(*b.low());
    let lo_hi = a.low().wrapping_mul(*b.high());
    *r.high_mut() = r.high().wrapping_add(hi_lo.wrapping_add(lo_hi));

    res.write(r);
}

#[inline]
pub fn umulc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool {
    let mut res = umulddi3(a.low(), b.low());

    let (hi_lo, overflow_hi_lo) = a.high().overflowing_mul(*b.low());
    let (lo_hi, overflow_lo_hi) = a.low().overflowing_mul(*b.high());
    let (hi, overflow_hi) = hi_lo.overflowing_add(lo_hi);
    let (high, overflow_high) = res.high().overflowing_add(hi);
    *res.high_mut() = high;

    let overflow_hi_hi = (*a.high() != 0) & (*b.high() != 0);

    r.write(res);
    overflow_hi_lo | overflow_lo_hi | overflow_hi | overflow_high | overflow_hi_hi
}

#[inline]
pub fn imulc(res: &mut MaybeUninit<I256>, a: &I256, b: &I256) -> bool {
    mul3(cast!(uninit: res), cast!(ref: a), cast!(ref: b));
    if *a == I256::MIN {
        return *b != 0 && *b != 1;
    }
    if *b == I256::MIN {
        return *a != 0 && *a != 1;
    }
    let sa = a >> (I256::BITS - 1);
    let abs_a = (a ^ sa).wrapping_sub(sa);
    let sb = b >> (I256::BITS - 1);
    let abs_b = (b ^ sb).wrapping_sub(sb);
    if abs_a < 2 || abs_b < 2 {
        return false;
    }
    if sa == sb {
        abs_a > I256::MAX / abs_b
    } else {
        abs_a > I256::MIN / -abs_b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsU256;

    fn umul(a: impl AsU256, b: impl AsU256) -> (U256, bool) {
        let mut r = MaybeUninit::uninit();
        let overflow = umulc(&mut r, &a.as_u256(), &b.as_u256());
        (unsafe { r.assume_init() }, overflow)
    }

    #[test]
    fn multiplication() {
        assert_eq!(umul(6, 7), (42.as_u256(), false));

        assert_eq!(umul(U256::MAX, 1), (U256::MAX, false));
        assert_eq!(umul(1, U256::MAX), (U256::MAX, false));
        assert_eq!(umul(U256::MAX, 0), (U256::ZERO, false));
        assert_eq!(umul(0, U256::MAX), (U256::ZERO, false));

        assert_eq!(umul(U256::MAX, 5), (U256::MAX ^ 4, true));
        assert_eq!(
            umul(u128::MAX, u128::MAX),
            (U256::from_words(!0 << 1, 1), false),
        );
    }
}
