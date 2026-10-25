//! Module containing arithmetic left shift intrinsic.

use crate::uint::U256;
use core::mem::MaybeUninit;

#[inline]
pub fn shl2(r: &mut U256, a: u32) {
    debug_assert!(a < 256, "shl intrinsic called with overflowing shift");

    let (hi, lo) = if a == 0 {
        return;
    } else if a < 128 {
        ((r.high() << a) | (r.low() >> (128 - a)), r.low() << a)
    } else {
        (r.low() << (a & 0x7f), 0)
    };

    *r = U256::from_words(hi, lo);
}

#[inline]
pub fn shl3(r: &mut MaybeUninit<U256>, a: &U256, b: u32) {
    debug_assert!(b < 256, "shl intrinsic called with overflowing shift");

    let (hi, lo) = if b == 0 {
        (*a.high(), *a.low())
    } else if b < 128 {
        ((a.high() << b) | (a.low() >> (128 - b)), a.low() << b)
    } else {
        (a.low() << (b & 0x7f), 0)
    };

    r.write(U256::from_words(hi, lo));
}
