//! Module `core::ops` trait implementations.
//!
//! Trait implementations for `i128` are also provided to allow notation such
//! as:
//!
//! ```
//! # use ethnum::I256;
//!
//! let a = 1 + I256::ONE;
//! let b = I256::ONE + 1;
//! dbg!(a, b);
//! ```

use super::I256;
use crate::intrinsics::signed::*;

impl_ops! {
    for I256 | i128 {
        add => iadd2, iadd3, iaddc;
        mul => imul2, imul3, imulc;
        sub => isub2, isub3, isubc;

        div => idiv2, idiv3;
        rem => irem2, irem3;

        shl => ishl2, ishl3;
        shr => isar2, isar3;
    }
}

impl_ops_neg! {
    for I256 {
        add => iadd2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uint::U256;
    use core::ops::*;

    #[test]
    fn trait_implementations() {
        trait Implements {}
        impl Implements for I256 {}
        impl Implements for &'_ I256 {}

        fn assert_ops<T>()
        where
            for<'a> T: Implements
                + Add<&'a i128>
                + Add<&'a I256>
                + Add<i128>
                + Add<I256>
                + AddAssign<&'a i128>
                + AddAssign<&'a I256>
                + AddAssign<i128>
                + AddAssign<I256>
                + BitAnd<&'a i128>
                + BitAnd<&'a I256>
                + BitAnd<i128>
                + BitAnd<I256>
                + BitAndAssign<&'a i128>
                + BitAndAssign<&'a I256>
                + BitAndAssign<i128>
                + BitAndAssign<I256>
                + BitOr<&'a i128>
                + BitOr<&'a I256>
                + BitOr<i128>
                + BitOr<I256>
                + BitOrAssign<&'a i128>
                + BitOrAssign<&'a I256>
                + BitOrAssign<i128>
                + BitOrAssign<I256>
                + BitXor<&'a i128>
                + BitXor<&'a I256>
                + BitXor<i128>
                + BitXor<I256>
                + BitXorAssign<&'a i128>
                + BitXorAssign<&'a I256>
                + BitXorAssign<i128>
                + BitXorAssign<I256>
                + Div<&'a i128>
                + Div<&'a I256>
                + Div<i128>
                + Div<I256>
                + DivAssign<&'a i128>
                + DivAssign<&'a I256>
                + DivAssign<i128>
                + DivAssign<I256>
                + Mul<&'a i128>
                + Mul<&'a I256>
                + Mul<i128>
                + Mul<I256>
                + MulAssign<&'a i128>
                + MulAssign<&'a I256>
                + MulAssign<i128>
                + MulAssign<I256>
                + Neg
                + Not
                + Rem<&'a i128>
                + Rem<&'a I256>
                + Rem<i128>
                + Rem<I256>
                + RemAssign<&'a i128>
                + RemAssign<&'a I256>
                + RemAssign<i128>
                + RemAssign<I256>
                + Shl<&'a i128>
                + Shl<&'a i16>
                + Shl<&'a I256>
                + Shl<&'a i32>
                + Shl<&'a i64>
                + Shl<&'a i8>
                + Shl<&'a isize>
                + Shl<&'a u128>
                + Shl<&'a u16>
                + Shl<&'a U256>
                + Shl<&'a u32>
                + Shl<&'a u64>
                + Shl<&'a u8>
                + Shl<&'a usize>
                + Shl<i128>
                + Shl<i16>
                + Shl<I256>
                + Shl<i32>
                + Shl<i64>
                + Shl<i8>
                + Shl<isize>
                + Shl<u128>
                + Shl<u16>
                + Shl<U256>
                + Shl<u32>
                + Shl<u64>
                + Shl<u8>
                + Shl<usize>
                + ShlAssign<&'a i128>
                + ShlAssign<&'a i16>
                + ShlAssign<&'a I256>
                + ShlAssign<&'a i32>
                + ShlAssign<&'a i64>
                + ShlAssign<&'a i8>
                + ShlAssign<&'a isize>
                + ShlAssign<&'a u128>
                + ShlAssign<&'a u16>
                + ShlAssign<&'a U256>
                + ShlAssign<&'a u32>
                + ShlAssign<&'a u64>
                + ShlAssign<&'a u8>
                + ShlAssign<&'a usize>
                + ShlAssign<i128>
                + ShlAssign<i16>
                + ShlAssign<I256>
                + ShlAssign<i32>
                + ShlAssign<i64>
                + ShlAssign<i8>
                + ShlAssign<isize>
                + ShlAssign<u128>
                + ShlAssign<u16>
                + ShlAssign<U256>
                + ShlAssign<u32>
                + ShlAssign<u64>
                + ShlAssign<u8>
                + ShlAssign<usize>
                + Shr<&'a i128>
                + Shr<&'a i16>
                + Shr<&'a I256>
                + Shr<&'a i32>
                + Shr<&'a i64>
                + Shr<&'a i8>
                + Shr<&'a isize>
                + Shr<&'a u128>
                + Shr<&'a u16>
                + Shr<&'a U256>
                + Shr<&'a u32>
                + Shr<&'a u64>
                + Shr<&'a u8>
                + Shr<&'a usize>
                + Shr<i128>
                + Shr<i16>
                + Shr<I256>
                + Shr<i32>
                + Shr<i64>
                + Shr<i8>
                + Shr<isize>
                + Shr<u128>
                + Shr<u16>
                + Shr<U256>
                + Shr<u32>
                + Shr<u64>
                + Shr<u8>
                + Shr<usize>
                + ShrAssign<&'a i128>
                + ShrAssign<&'a i16>
                + ShrAssign<&'a I256>
                + ShrAssign<&'a i32>
                + ShrAssign<&'a i64>
                + ShrAssign<&'a i8>
                + ShrAssign<&'a isize>
                + ShrAssign<&'a u128>
                + ShrAssign<&'a u16>
                + ShrAssign<&'a U256>
                + ShrAssign<&'a u32>
                + ShrAssign<&'a u64>
                + ShrAssign<&'a u8>
                + ShrAssign<&'a usize>
                + ShrAssign<i128>
                + ShrAssign<i16>
                + ShrAssign<I256>
                + ShrAssign<i32>
                + ShrAssign<i64>
                + ShrAssign<i8>
                + ShrAssign<isize>
                + ShrAssign<u128>
                + ShrAssign<u16>
                + ShrAssign<U256>
                + ShrAssign<u32>
                + ShrAssign<u64>
                + ShrAssign<u8>
                + ShrAssign<usize>
                + Sub<&'a i128>
                + Sub<&'a I256>
                + Sub<i128>
                + Sub<I256>
                + SubAssign<&'a i128>
                + SubAssign<&'a I256>
                + SubAssign<i128>
                + SubAssign<I256>,
            for<'a> &'a T: Implements
                + Add<&'a i128>
                + Add<&'a I256>
                + Add<i128>
                + Add<I256>
                + BitAnd<&'a i128>
                + BitAnd<&'a I256>
                + BitAnd<i128>
                + BitAnd<I256>
                + BitOr<&'a i128>
                + BitOr<&'a I256>
                + BitOr<i128>
                + BitOr<I256>
                + BitXor<&'a i128>
                + BitXor<&'a I256>
                + BitXor<i128>
                + BitXor<I256>
                + Div<&'a i128>
                + Div<&'a I256>
                + Div<i128>
                + Div<I256>
                + Mul<&'a i128>
                + Mul<&'a I256>
                + Mul<i128>
                + Mul<I256>
                + Neg
                + Not
                + Rem<&'a i128>
                + Rem<&'a I256>
                + Rem<i128>
                + Rem<I256>
                + Shl<&'a i128>
                + Shl<&'a i16>
                + Shl<&'a I256>
                + Shl<&'a i32>
                + Shl<&'a i64>
                + Shl<&'a i8>
                + Shl<&'a isize>
                + Shl<&'a u128>
                + Shl<&'a u16>
                + Shl<&'a U256>
                + Shl<&'a u32>
                + Shl<&'a u64>
                + Shl<&'a u8>
                + Shl<&'a usize>
                + Shl<i128>
                + Shl<i16>
                + Shl<I256>
                + Shl<i32>
                + Shl<i64>
                + Shl<i8>
                + Shl<isize>
                + Shl<u128>
                + Shl<u16>
                + Shl<U256>
                + Shl<u32>
                + Shl<u64>
                + Shl<u8>
                + Shl<usize>
                + Shr<&'a i128>
                + Shr<&'a i16>
                + Shr<&'a I256>
                + Shr<&'a i32>
                + Shr<&'a i64>
                + Shr<&'a i8>
                + Shr<&'a isize>
                + Shr<&'a u128>
                + Shr<&'a u16>
                + Shr<&'a U256>
                + Shr<&'a u32>
                + Shr<&'a u64>
                + Shr<&'a u8>
                + Shr<&'a usize>
                + Shr<i128>
                + Shr<i16>
                + Shr<I256>
                + Shr<i32>
                + Shr<i64>
                + Shr<i8>
                + Shr<isize>
                + Shr<u128>
                + Shr<u16>
                + Shr<U256>
                + Shr<u32>
                + Shr<u64>
                + Shr<u8>
                + Shr<usize>
                + Sub<&'a i128>
                + Sub<&'a I256>
                + Sub<i128>
                + Sub<I256>,
        {
        }

        assert_ops::<I256>();
    }
}
