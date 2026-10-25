//! This module implements right and left rotation (**not** shifting) intrinsics
//! for 256-bit integers.

use crate::uint::U256;
use core::mem::MaybeUninit;

#[inline]
pub fn rol3(r: &mut MaybeUninit<U256>, a: &U256, b: u32) {
    r.write((a << (b & 0xff)) | (a >> ((256 - b) & 0xff)));
}

#[inline]
pub fn ror3(r: &mut MaybeUninit<U256>, a: &U256, b: u32) {
    r.write((a >> (b & 0xff)) | (a << ((256 - b) & 0xff)));
}
