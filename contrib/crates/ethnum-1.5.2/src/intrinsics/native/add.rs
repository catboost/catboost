//! Module implementing addition intrinsics.

use crate::{int::I256, uint::U256};
use core::mem::MaybeUninit;

#[inline]
pub fn add2(r: &mut U256, a: &U256) {
    let (lo, carry) = r.low().overflowing_add(*a.low());
    *r.low_mut() = lo;
    *r.high_mut() = r.high().wrapping_add(carry as _).wrapping_add(*a.high());
}

#[inline]
pub fn add3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) {
    let (lo, carry) = a.low().overflowing_add(*b.low());
    let hi = a.high().wrapping_add(carry as _).wrapping_add(*b.high());

    r.write(U256::from_words(hi, lo));
}

#[inline]
pub fn uaddc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool {
    let (lo, carry_lo) = a.low().overflowing_add(*b.low());
    let (hi, carry_c) = a.high().overflowing_add(carry_lo as _);
    let (hi, carry_hi) = hi.overflowing_add(*b.high());

    r.write(U256::from_words(hi, lo));
    carry_c || carry_hi
}

#[inline]
pub fn iaddc(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) -> bool {
    add3(cast!(uninit: r), cast!(ref: a), cast!(ref: b));
    let s = unsafe { r.assume_init_ref() };
    (*b >= 0 && s < a) || (*b < 0 && s >= a)
}
