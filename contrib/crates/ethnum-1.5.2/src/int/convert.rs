//! Module contains conversions for [`I256`] to and from primimitive types.

use super::I256;
use crate::{error::tfie, uint::U256};
use core::num::TryFromIntError;

macro_rules! impl_from {
    ($($t:ty),* $(,)?) => {$(
        impl From<$t> for I256 {
            #[inline]
            fn from(value: $t) -> Self {
                value.as_i256()
            }
        }
    )*};
}

impl_from! {
    bool,
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
}

impl TryFrom<U256> for I256 {
    type Error = TryFromIntError;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        if value > I256::MAX.as_u256() {
            return Err(tfie());
        }
        Ok(value.as_i256())
    }
}

/// This trait defines `as` conversions (casting) from primitive types to
/// [`I256`].
///
/// [`I256`]: struct.I256.html
///
/// # Examples
///
/// Casting a floating point value to an integer is a saturating operation,
/// with `NaN` converting to `0`. So:
///
/// ```
/// # use ethnum::{I256, AsI256};
/// assert_eq!((-1i32).as_i256(), -I256::ONE);
/// assert_eq!(u32::MAX.as_i256(), 0xffffffff);
///
/// assert_eq!(-13.37f64.as_i256(), -13);
/// assert_eq!(42.0f64.as_i256(), 42);
/// assert_eq!(
///     f32::MAX.as_i256(),
///     0xffffff00000000000000000000000000u128.as_i256(),
/// );
/// assert_eq!(
///     f32::MIN.as_i256(),
///     -0xffffff00000000000000000000000000u128.as_i256(),
/// );
///
/// assert_eq!(f64::NEG_INFINITY.as_i256(), I256::MIN);
/// assert_eq!((-2.0f64.powi(256)).as_i256(), I256::MIN);
/// assert_eq!(f64::INFINITY.as_i256(), I256::MAX);
/// assert_eq!(2.0f64.powi(256).as_i256(), I256::MAX);
/// assert_eq!(f64::NAN.as_i256(), 0);
/// ```
pub trait AsI256 {
    /// Perform an `as` conversion to a [`I256`].
    ///
    /// [`I256`]: struct.I256.html
    #[allow(clippy::wrong_self_convention)]
    fn as_i256(self) -> I256;
}

impl AsI256 for I256 {
    #[inline]
    fn as_i256(self) -> I256 {
        self
    }
}

impl AsI256 for U256 {
    #[inline]
    fn as_i256(self) -> I256 {
        U256::as_i256(self)
    }
}

macro_rules! impl_as_i256 {
    ($($t:ty),* $(,)?) => {$(
        impl AsI256 for $t {
            #[inline]
            fn as_i256(self) -> I256 {
                #[allow(unused_comparisons)]
                let hi = if self >= 0 { 0 } else { !0 };
                I256::from_words(hi, self as _)
            }
        }
    )*};
}

impl_as_i256! {
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    isize, usize,
}

impl AsI256 for bool {
    #[inline]
    fn as_i256(self) -> I256 {
        I256::new(self as _)
    }
}

macro_rules! impl_as_i256_float {
    ($($t:ty [$b:ty]),* $(,)?) => {$(
        impl AsI256 for $t {
            #[inline]
            fn as_i256(self) -> I256 {
                // The conversion follows roughly the same rules as converting
                // `f64` to other primitive integer types:
                // - `NaN` => `0`
                // - `(-∞, I256::MIN]` => `I256::MIN`
                // - `(I256::MIN, I256::MAX]` => `value as I256`
                // - `(I256::MAX, +∞)` => `I256::MAX`

                const M: $b = (<$t>::MANTISSA_DIGITS - 1) as _;
                const MAN_MASK: $b = !(!0 << M);
                const MAN_ONE: $b = 1 << M;
                const EXP_MASK: $b = !0 >> <$t>::MANTISSA_DIGITS;
                const EXP_OFFSET: $b = EXP_MASK / 2;
                const ABS_MASK: $b = !0 >> 1;
                const SIG_MASK: $b = !ABS_MASK;

                let abs = <$t>::from_bits(self.to_bits() & ABS_MASK);
                let sign = -(((self.to_bits() & SIG_MASK) >> (<$b>::BITS - 2)) as i128)
                    .wrapping_sub(1); // if self >= 0. { 1 } else { -1 }
                if abs >= 1.0 {
                    let bits = abs.to_bits();
                    let exponent = ((bits >> M) & EXP_MASK) - EXP_OFFSET;
                    let mantissa = (bits & MAN_MASK) | MAN_ONE;
                    if exponent <= M {
                        (I256::from(mantissa >> (M - exponent))) * sign
                    } else if exponent < 255 {
                        (I256::from(mantissa) << (exponent - M)) * sign
                    } else if sign > 0 {
                        I256::MAX
                    } else {
                        I256::MIN
                    }
                } else {
                    I256::ZERO
                }
            }
        }
    )*};
}

impl_as_i256_float! {
    f32[u32], f64[u64],
}

macro_rules! impl_try_into {
    ($($t:ty),* $(,)?) => {$(
        impl TryFrom<I256> for $t {
            type Error = TryFromIntError;

            #[inline]
            fn try_from(x: I256) -> Result<Self, Self::Error> {
                if x >= <$t>::MIN.as_i256() && x <= <$t>::MAX.as_i256() {
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
        impl From<I256> for $t {
            #[inline]
            fn from(x: I256) -> $t {
                x.$f()
            }
        }
    )*};
}

impl_into_float! {
    f32 => as_f32, f64 => as_f64,
}

#[cfg(test)]
mod tests {
    use crate::I256;
    use core::str::FromStr as _;

    #[test]
    fn checked_conversion() {
        assert_eq!(i32::try_from(I256::new(-10)).unwrap(), -10);
        assert_eq!(i32::try_from(I256::new(10)).unwrap(), 10);
        assert!(i32::try_from(I256::MIN).is_err());
        assert!(i32::try_from(I256::MAX).is_err());
    }

    #[test]
    fn github_issue_44() {
        // A big number.
        let lhs_i256 = I256::from(i128::MAX);

        // Adding 19 zeros.
        let scaled_lhs = lhs_i256 * I256::from_str("10000000000000000000").unwrap();

        // One and 18 zeros.
        let rhs_i256 = I256::from_str("-1000000000000000000").unwrap();

        // So result is basically -i128::MAX and one zero.
        let result = scaled_lhs.checked_div(rhs_i256).unwrap();

        assert!(i128::try_from(result).is_err());
    }
}
