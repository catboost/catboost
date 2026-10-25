use core::fmt::{Debug, Display};
use core::ops::{Add, Div, Mul, Neg};

mod private {
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait Float:
    Sized
    + private::Sealed
    + Div<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Add<Output = Self>
    + PartialEq
    + PartialOrd
    + Default
    + Clone
    + Copy
    + Debug
    + Display
{
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const NEG_NAN: Self;

    const MANTISSA_EXPLICIT_BITS: usize;
    const MIN_EXPONENT_ROUND_TO_EVEN: i32;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32;
    const MIN_EXPONENT_FAST_PATH: i64;
    const MAX_EXPONENT_FAST_PATH: i64;
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i64;
    const MINIMUM_EXPONENT: i32;
    const INFINITE_POWER: i32;
    const SIGN_INDEX: usize;
    const SMALLEST_POWER_OF_TEN: i32;
    const LARGEST_POWER_OF_TEN: i32;

    const MAX_MANTISSA_FAST_PATH: u64 = 2_u64 << Self::MANTISSA_EXPLICIT_BITS;

    fn from_u64(v: u64) -> Self;
    fn from_u64_bits(v: u64) -> Self;
    fn pow10_fast_path(exponent: usize) -> Self;
}

impl private::Sealed for f32 {
}

impl Float for f32 {
    const INFINITY: Self = core::f32::INFINITY;
    const NEG_INFINITY: Self = core::f32::NEG_INFINITY;
    const NAN: Self = core::f32::NAN;
    const NEG_NAN: Self = -core::f32::NAN;

    const MANTISSA_EXPLICIT_BITS: usize = 23;
    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -17;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 10;
    const MIN_EXPONENT_FAST_PATH: i64 = -10; // assuming FLT_EVAL_METHOD = 0
    const MAX_EXPONENT_FAST_PATH: i64 = 10;
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i64 = 17;
    const MINIMUM_EXPONENT: i32 = -127;
    const INFINITE_POWER: i32 = 0xFF;
    const SIGN_INDEX: usize = 31;
    const SMALLEST_POWER_OF_TEN: i32 = -65;
    const LARGEST_POWER_OF_TEN: i32 = 38;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v as f32
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        f32::from_bits((v & 0xFFFF_FFFF) as u32)
    }

    #[inline]
    fn pow10_fast_path(exponent: usize) -> Self {
        #[allow(clippy::use_self)]
        const TABLE: [f32; 16] =
            [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 0., 0., 0., 0., 0.];
        TABLE[exponent & 15]
    }
}

impl private::Sealed for f64 {
}

impl Float for f64 {
    const INFINITY: Self = core::f64::INFINITY;
    const NEG_INFINITY: Self = core::f64::NEG_INFINITY;
    const NAN: Self = core::f64::NAN;
    const NEG_NAN: Self = -core::f64::NAN;

    const MANTISSA_EXPLICIT_BITS: usize = 52;
    const MIN_EXPONENT_ROUND_TO_EVEN: i32 = -4;
    const MAX_EXPONENT_ROUND_TO_EVEN: i32 = 23;
    const MIN_EXPONENT_FAST_PATH: i64 = -22; // assuming FLT_EVAL_METHOD = 0
    const MAX_EXPONENT_FAST_PATH: i64 = 22;
    const MAX_EXPONENT_DISGUISED_FAST_PATH: i64 = 37;
    const MINIMUM_EXPONENT: i32 = -1023;
    const INFINITE_POWER: i32 = 0x7FF;
    const SIGN_INDEX: usize = 63;
    const SMALLEST_POWER_OF_TEN: i32 = -342;
    const LARGEST_POWER_OF_TEN: i32 = 308;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v as f64
    }

    #[inline]
    fn from_u64_bits(v: u64) -> Self {
        f64::from_bits(v)
    }

    #[inline]
    fn pow10_fast_path(exponent: usize) -> Self {
        #[allow(clippy::use_self)]
        const TABLE: [f64; 32] = [
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
            1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        TABLE[exponent & 31]
    }
}
