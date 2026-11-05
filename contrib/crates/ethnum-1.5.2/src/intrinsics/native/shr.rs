//! Module containing logical right shift intrinsic.

use crate::{int::I256, uint::U256};
use core::mem::MaybeUninit;

#[inline]
pub fn sar2(r: &mut I256, a: u32) {
    debug_assert!(a < 256, "shr intrinsic called with overflowing shift");

    let (hi, lo) = if a == 0 {
        return;
    } else if a < 128 {
        (
            r.high() >> a,
            ((*r.low() as u128) >> a | ((*r.high() as u128) << (128 - a))) as i128,
        )
    } else {
        (r.high() >> 127, (r.high() >> (a & 0x7f)))
    };

    *r = I256::from_words(hi, lo);
}

#[inline]
pub fn sar3(r: &mut MaybeUninit<I256>, a: &I256, b: u32) {
    debug_assert!(b < 256, "shr intrinsic called with overflowing shift");

    let (hi, lo) = if b == 0 {
        (*a.high(), *a.low())
    } else if b < 128 {
        (
            a.high() >> b,
            ((*a.low() as u128) >> b | ((*a.high() as u128) << (128 - b))) as i128,
        )
    } else {
        (a.high() >> 127, a.high() >> (b & 0x7f))
    };

    r.write(I256::from_words(hi, lo));
}

#[inline]
pub fn shr2(r: &mut U256, a: u32) {
    debug_assert!(a < 256, "shr intrinsic called with overflowing shift");

    let (hi, lo) = if a == 0 {
        return;
    } else if a < 128 {
        (r.high() >> a, r.low() >> a | (r.high() << (128 - a)))
    } else {
        (0, r.high() >> (a & 0x7f))
    };

    *r = U256::from_words(hi, lo);
}

#[inline]
pub fn shr3(r: &mut MaybeUninit<U256>, a: &U256, b: u32) {
    debug_assert!(b < 256, "shr intrinsic called with overflowing shift");

    let (hi, lo) = if b == 0 {
        (*a.high(), *a.low())
    } else if b < 128 {
        (a.high() >> b, a.low() >> b | (a.high() << (128 - b)))
    } else {
        (0, a.high() >> (b & 0x7f))
    };

    r.write(U256::from_words(hi, lo));
}
