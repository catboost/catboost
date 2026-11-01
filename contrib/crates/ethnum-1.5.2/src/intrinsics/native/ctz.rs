//! This module implements intrinsics for counting trailing and leading zeros
//! for 256-bit integers.

use crate::uint::U256;

#[inline]
pub fn ctlz(a: &U256) -> u32 {
    let f = -((*a.high() == 0) as i128) as u128;
    ((a.high() & !f) | (a.low() & f)).leading_zeros() + ((f as u32) & 128)
}

#[inline]
pub fn cttz(a: &U256) -> u32 {
    let f = -((*a.low() == 0) as i128) as u128;
    ((a.high() & f) | (a.low() & !f)).trailing_zeros() + ((f as u32) & 128)
}
