//! Module containing signed wrapping functions around various intrinsics that
//! are agnostic to sign.
//!
//! This module can be helpful when using intrinsics directly.

pub use super::{
    add2 as uadd2, add3 as uadd3, ctlz as uctlz, cttz as ucttz, iaddc, idiv2, idiv3, imulc, irem2,
    irem3, isubc, mul2 as umul2, mul3 as umul3, rol3 as urol3, ror3 as uror3, sar2 as isar2,
    sar3 as isar3, shl2 as ushl2, shl3 as ushl3, shr2 as ushr2, shr3 as ushr3, sub2 as usub2,
    sub3 as usub3, uaddc, udiv2, udiv3, umulc, urem2, urem3, usubc,
};
use crate::int::I256;
use core::mem::MaybeUninit;

#[inline]
pub fn iadd2(r: &mut I256, a: &I256) {
    super::add2(cast!(mut: r), cast!(ref: a));
}

#[inline]
pub fn iadd3(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) {
    super::add3(cast!(uninit: r), cast!(ref: a), cast!(ref: b));
}

#[inline]
pub fn isub2(r: &mut I256, a: &I256) {
    super::sub2(cast!(mut: r), cast!(ref: a));
}

#[inline]
pub fn isub3(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) {
    super::sub3(cast!(uninit: r), cast!(ref: a), cast!(ref: b));
}

#[inline]
pub fn imul2(r: &mut I256, a: &I256) {
    super::mul2(cast!(mut: r), cast!(ref: a));
}

#[inline]
pub fn imul3(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) {
    super::mul3(cast!(uninit: r), cast!(ref: a), cast!(ref: b));
}

#[inline]
pub fn ishl2(r: &mut I256, a: u32) {
    super::shl2(cast!(mut: r), a);
}

#[inline]
pub fn ishl3(r: &mut MaybeUninit<I256>, a: &I256, b: u32) {
    super::shl3(cast!(uninit: r), cast!(ref: a), b);
}

#[inline]
pub fn irol3(r: &mut MaybeUninit<I256>, a: &I256, b: u32) {
    super::rol3(cast!(uninit: r), cast!(ref: a), b);
}

#[inline]
pub fn iror3(r: &mut MaybeUninit<I256>, a: &I256, b: u32) {
    super::ror3(cast!(uninit: r), cast!(ref: a), b);
}

#[inline]
pub fn ictlz(a: &I256) -> u32 {
    super::ctlz(cast!(ref: a))
}

#[inline]
pub fn icttz(a: &I256) -> u32 {
    super::cttz(cast!(ref: a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uint::U256;
    use core::alloc::Layout;

    #[test]
    fn layout() {
        assert_eq!(Layout::new::<I256>(), Layout::new::<U256>());
    }
}
