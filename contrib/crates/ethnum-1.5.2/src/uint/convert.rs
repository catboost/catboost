//! Module contains conversions for [`U256`] to and from primimitive types.

use super::U256;
use crate::{error::tfie, int::I256};
use core::{convert::TryFrom, num::TryFromIntError};

macro_rules! impl_from {
    ($($t:ty),* $(,)?) => {$(
        impl From<$t> for U256 {
            #[inline]
            fn from(value: $t) -> Self {
                U256::new(value.into())
            }
        }
    )*};
}

impl_from! {
    bool, u8, u16, u32, u64, u128,
}

macro_rules! impl_try_from {
    ($($t:ty),* $(,)?) => {$(
        impl TryFrom<$t> for U256 {
            type Error = TryFromIntError;

            #[inline]
            fn try_from(value: $t) -> Result<Self, Self::Error> {
                Ok(U256::new(u128::try_from(value)?))
            }
        }
    )*};
}

impl_try_from! {
    i8, i16, i32, i64, i128,
    isize, usize,
}

impl TryFrom<I256> for U256 {
    type Error = TryFromIntError;

    fn try_from(value: I256) -> Result<Self, Self::Error> {
        if value < 0 {
            return Err(tfie());
        }
        Ok(value.as_u256())
    }
}

/// This trait defines `as` conversions (casting) from primitive types to
/// [`U256`].
///
/// [`U256`]: struct.U256.html
///
/// # Examples
///
/// Note that in Rust casting from a negative signed integer sign to a larger
/// unsigned interger sign extends. Additionally casting a floating point value
/// to an integer is a saturating operation, with `NaN` converting to `0`. So:
///
/// ```
/// # use ethnum::{U256, AsU256};
/// assert_eq!((-1i32).as_u256(), U256::MAX);
/// assert_eq!(u32::MAX.as_u256(), 0xffffffff);
///
/// assert_eq!(f64::NEG_INFINITY.as_u256(), 0);
/// assert_eq!((-1.0f64).as_u256(), 0);
/// assert_eq!(f64::INFINITY.as_u256(), U256::MAX);
/// assert_eq!(2.0f64.powi(257).as_u256(), U256::MAX);
/// assert_eq!(f64::NAN.as_u256(), 0);
/// ```
pub trait AsU256 {
    /// Perform an `as` conversion to a [`U256`].
    ///
    /// [`U256`]: struct.U256.html
    #[allow(clippy::wrong_self_convention)]
    fn as_u256(self) -> U256;
}

impl AsU256 for U256 {
    #[inline]
    fn as_u256(self) -> U256 {
        self
    }
}

impl AsU256 for I256 {
    #[inline]
    fn as_u256(self) -> U256 {
        I256::as_u256(self)
    }
}

macro_rules! impl_as_u256 {
    ($($t:ty),* $(,)?) => {$(
        impl AsU256 for $t {
            #[inline]
            fn as_u256(self) -> U256 {
                #[allow(unused_comparisons)]
                let hi = if self >= 0 { 0 } else { !0 };
                U256::from_words(hi, self as _)
            }
        }
    )*};
}

impl_as_u256! {
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    isize, usize,
}

impl AsU256 for bool {
    #[inline]
    fn as_u256(self) -> U256 {
        U256::new(self as _)
    }
}

macro_rules! impl_as_u256_float {
    ($($t:ty [$b:ty]),* $(,)?) => {$(
        impl AsU256 for $t {
            #[inline]
            fn as_u256(self) -> U256 {
                // The conversion follows roughly the same rules as converting
                // `f64` to other primitive integer types:
                // - `NaN` => `0`
                // - `(-∞, 0]` => `0`
                // - `(0, U256::MAX]` => `value as U256`
                // - `(U256::MAX, +∞)` => `U256::MAX`

                const M: $b = (<$t>::MANTISSA_DIGITS - 1) as _;
                const MAN_MASK: $b = !(!0 << M);
                const MAN_ONE: $b = 1 << M;
                const EXP_MASK: $b = !0 >> <$t>::MANTISSA_DIGITS;
                const EXP_OFFSET: $b = EXP_MASK / 2;

                if self >= 1.0 {
                    let bits = self.to_bits();
                    let exponent = ((bits >> M) & EXP_MASK) - EXP_OFFSET;
                    let mantissa = (bits & MAN_MASK) | MAN_ONE;
                    if exponent <= M {
                        U256::from(mantissa >> (M - exponent))
                    } else if exponent < 256 {
                        U256::from(mantissa) << (exponent - M)
                    } else {
                        U256::MAX
                    }
                } else {
                    U256::ZERO
                }
            }
        }
    )*};
}

impl_as_u256_float! {
    f32[u32], f64[u64],
}

macro_rules! impl_try_into {
    ($($t:ty),* $(,)?) => {$(
        impl TryFrom<U256> for $t {
            type Error = TryFromIntError;

            #[inline]
            fn try_from(x: U256) -> Result<Self, Self::Error> {
                if x <= <$t>::MAX.as_u256() {
                    Ok(*x.low() as _)
                } else {
                    Err(tfie())
                }
            }
        }
    )*};
}

impl_try_into! {
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    isize, usize,
}

macro_rules! impl_into_float {
    ($($t:ty => $f:ident),* $(,)?) => {$(
        impl From<U256> for $t {
            #[inline]
            fn from(x: U256) -> $t {
                x.$f()
            }
        }
    )*};
}

impl_into_float! {
    f32 => as_f32, f64 => as_f64,
}
