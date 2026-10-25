//! Module implementing subtraction intrinsics.

use crate::{int::I256, uint::U256};
use core::mem::MaybeUninit;

#[inline]
pub fn sub2(r: &mut U256, a: &U256) {
    let (lo, carry) = r.low().overflowing_sub(*a.low());
    *r.low_mut() = lo;
    *r.high_mut() = r.high().wrapping_sub(carry as _).wrapping_sub(*a.high());
}

#[inline]
pub fn sub3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) {
    let (lo, carry) = a.low().overflowing_sub(*b.low());
    let hi = a.high().wrapping_sub(carry as _).wrapping_sub(*b.high());

    r.write(U256::from_words(hi, lo));
}

#[inline]
pub fn usubc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool {
    let (lo, carry_lo) = a.low().overflowing_sub(*b.low());
    let (hi, carry_c) = a.high().overflowing_sub(carry_lo as _);
    let (hi, carry_hi) = hi.overflowing_sub(*b.high());

    r.write(U256::from_words(hi, lo));
    carry_c || carry_hi
}

#[inline]
pub fn isubc(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) -> bool {
    sub3(cast!(uninit: r), cast!(ref: a), cast!(ref: b));
    let s = unsafe { r.assume_init_ref() };
    (*b >= 0 && s > a) || (*b < 0 && s <= a)
}
