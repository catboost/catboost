//! This module contains definitions for LLVM IR generated intrinsics.

// NOTE: LLVM IR generated intrinsics for `{i,u}div i256`, `{i,u}rem i256`, and
// `imul i256` produce an error when compiling. Use the native implementations
// even when generated intrinsics are enabled.
#[path = "native/divmod.rs"]
mod divmod;
#[path = "native/mul.rs"]
#[allow(dead_code)]
mod mul;

pub use self::{divmod::*, mul::imulc};
use crate::{int::I256, uint::U256};
use core::mem::{self, MaybeUninit};

macro_rules! def {
    ($(
        $(#[$a:meta])*
        pub fn $name:ident(
            $($p:ident : $t:ty),*
        ) $(-> $ret:ty)?;
    )*) => {$(
        $(#[$a])*
        pub fn $name(
            $($p: $t,)*
        ) $(-> $ret)? {
            unsafe {
                ethnum_intrinsics::$name($(
                    #[allow(
                        clippy::missing_transmute_annotations,
                        clippy::transmute_ptr_to_ptr,
                        clippy::useless_transmute,
                    )]
                    mem::transmute::<_, _>($p)
                ),*)
            }
        }
    )*};
}

def! {
    pub fn add2(r: &mut U256, a: &U256);
    pub fn add3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256);
    pub fn uaddc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool;
    pub fn iaddc(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) -> bool;

    pub fn sub2(r: &mut U256, a: &U256);
    pub fn sub3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256);
    pub fn usubc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool;
    pub fn isubc(r: &mut MaybeUninit<I256>, a: &I256, b: &I256) -> bool;

    pub fn mul2(r: &mut U256, a: &U256);
    pub fn mul3(r: &mut MaybeUninit<U256>, a: &U256, b: &U256);
    pub fn umulc(r: &mut MaybeUninit<U256>, a: &U256, b: &U256) -> bool;

    pub fn shl2(r: &mut U256, a: u32);
    pub fn shl3(r: &mut MaybeUninit<U256>, a: &U256, b: u32);

    pub fn sar2(r: &mut I256, a: u32);
    pub fn sar3(r: &mut MaybeUninit<I256>, a: &I256, b: u32);
    pub fn shr2(r: &mut U256, a: u32);
    pub fn shr3(r: &mut MaybeUninit<U256>, a: &U256, b: u32);

    pub fn rol3(r: &mut MaybeUninit<U256>, a: &U256, b: u32);
    pub fn ror3(r: &mut MaybeUninit<U256>, a: &U256, b: u32);

    pub fn ctlz(a: &U256) -> u32;
    pub fn cttz(a: &U256) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{alloc::Layout, mem};

    #[test]
    fn layout() {
        // Small note on alignment: Since we pass in pointers to our wide
        // integer types we need only to make sure that the alignment of
        // `ethnum::{I256, U256}` types are larger than the FFI-safe type (i.e.
        // the alignment is compatible with the FFI type).

        assert_eq!(
            Layout::new::<I256>(),
            Layout::new::<ethnum_intrinsics::I256>()
                .align_to(mem::align_of::<I256>())
                .unwrap(),
        );
        assert_eq!(
            Layout::new::<U256>(),
            Layout::new::<ethnum_intrinsics::I256>()
                .align_to(mem::align_of::<U256>())
                .unwrap(),
        );
    }
}
