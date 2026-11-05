//! Root module for 256-bit signed integer type.

mod api;
mod cmp;
mod convert;
mod fmt;
mod iter;
mod ops;
mod parse;

pub use self::convert::AsI256;
use crate::uint::U256;
use core::{mem::MaybeUninit, num::ParseIntError};

/// A 256-bit signed integer type.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct I256(pub [i128; 2]);

impl I256 {
    /// The additive identity for this integer type, i.e. `0`.
    pub const ZERO: Self = I256([0; 2]);

    /// The multiplicative identity for this integer type, i.e. `1`.
    pub const ONE: Self = I256::new(1);

    /// The multiplicative inverse for this integer type, i.e. `-1`.
    pub const MINUS_ONE: Self = I256::new(-1);

    /// Creates a new 256-bit integer value from a primitive `i128` integer.
    #[inline]
    pub const fn new(value: i128) -> Self {
        I256::from_words(value >> 127, value)
    }

    /// Creates a new 256-bit integer value from high and low words.
    #[inline]
    pub const fn from_words(hi: i128, lo: i128) -> Self {
        #[cfg(target_endian = "little")]
        {
            I256([lo, hi])
        }
        #[cfg(target_endian = "big")]
        {
            I256([hi, lo])
        }
    }

    /// Splits a 256-bit integer into high and low words.
    #[inline]
    pub const fn into_words(self) -> (i128, i128) {
        #[cfg(target_endian = "little")]
        {
            let I256([lo, hi]) = self;
            (hi, lo)
        }
        #[cfg(target_endian = "big")]
        {
            let I256([hi, lo]) = self;
            (hi, lo)
        }
    }

    /// Get the low 128-bit word for this signed integer.
    #[inline]
    pub fn low(&self) -> &i128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[1]
        }
    }

    /// Get the low 128-bit word for this signed integer as a mutable reference.
    #[inline]
    pub fn low_mut(&mut self) -> &mut i128 {
        #[cfg(target_endian = "little")]
        {
            &mut self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &mut self.0[1]
        }
    }

    /// Get the high 128-bit word for this signed integer.
    #[inline]
    pub fn high(&self) -> &i128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[1]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[0]
        }
    }

    /// Get the high 128-bit word for this signed integer as a mutable
    /// reference.
    #[inline]
    pub fn high_mut(&mut self) -> &mut i128 {
        #[cfg(target_endian = "little")]
        {
            &mut self.0[1]
        }
        #[cfg(target_endian = "big")]
        {
            &mut self.0[0]
        }
    }

    /// Converts a prefixed string slice in base 16 to an integer.
    ///
    /// The string is expected to be an optional `+` or `-` sign followed by
    /// the `0x` prefix and finally the digits. Leading and trailing whitespace
    /// represent an error.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::from_str_hex("0x2A"), Ok(I256::new(42)));
    /// assert_eq!(I256::from_str_hex("-0xa"), Ok(I256::new(-10)));
    /// ```
    pub fn from_str_hex(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_radix(src, 16, Some("0x"))
    }

    /// Converts a prefixed string slice in a base determined by the prefix to
    /// an integer.
    ///
    /// The string is expected to be an optional `+` or `-` sign followed by
    /// the one of the supported prefixes and finally the digits. Leading and
    /// trailing whitespace represent an error. The base is determined based
    /// on the prefix:
    ///
    /// * `0b`: base `2`
    /// * `0o`: base `8`
    /// * `0x`: base `16`
    /// * no prefix: base `10`
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::from_str_prefixed("-0b101"), Ok(I256::new(-0b101)));
    /// assert_eq!(I256::from_str_prefixed("0o17"), Ok(I256::new(0o17)));
    /// assert_eq!(I256::from_str_prefixed("-0xa"), Ok(I256::new(-0xa)));
    /// assert_eq!(I256::from_str_prefixed("42"), Ok(I256::new(42)));
    /// ```
    pub fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_prefixed(src)
    }

    /// Same as [`I256::from_str_prefixed`] but as a `const fn`. This method is
    /// not intended to be used directly but rather through the [`crate::int`]
    /// macro.
    #[doc(hidden)]
    pub const fn const_from_str_prefixed(src: &str) -> Self {
        parse::const_from_str_prefixed(src)
    }

    /// Cast to a primitive `i8`.
    #[inline]
    pub const fn as_i8(self) -> i8 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i16`.
    #[inline]
    pub const fn as_i16(self) -> i16 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i32`.
    #[inline]
    pub const fn as_i32(self) -> i32 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i64`.
    #[inline]
    pub const fn as_i64(self) -> i64 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i128`.
    #[inline]
    pub const fn as_i128(self) -> i128 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u8`.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u16`.
    #[inline]
    pub const fn as_u16(self) -> u16 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u32`.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u64`.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u128`.
    #[inline]
    pub const fn as_u128(self) -> u128 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a `U256`.
    #[inline]
    pub const fn as_u256(self) -> U256 {
        let Self([a, b]) = self;
        U256([a as _, b as _])
    }

    /// Cast to a primitive `isize`.
    #[inline]
    pub const fn as_isize(self) -> isize {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `usize`.
    #[inline]
    pub const fn as_usize(self) -> usize {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `f32`.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.as_f64() as _
    }

    /// Cast to a primitive `f64`.
    #[inline]
    pub fn as_f64(self) -> f64 {
        let sign = self.signum128() as f64;
        self.unsigned_abs().as_f64() * sign
    }

    /// Performs integer and division and returns the quotient and the remainder as a tuple. This is equivelent to `(self / rhs, self % rhs)`, but more effecient.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0, and will panic on overflow iff debug assertions are enabled.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(22).div_rem(I256::new(5)), (I256::new(4), I256::new(2)));
    /// assert_eq!(I256::new(-22).div_rem(I256::new(5)), (I256::new(-4), I256::new(-2)));
    /// assert_eq!(I256::new(22).div_rem(I256::new(-5)), (I256::new(-4), I256::new(2)));
    /// assert_eq!(I256::new(-22).div_rem(I256::new(-5)), (I256::new(4), I256::new(-2)));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        if self == Self::MIN && rhs == -1 {
            panic!("attempt to divide with overflow")
        }
        self.wrapping_div_rem(rhs)
    }

    /// Performs euclidean division and returns the quotient and the remainder as a tuple.
    ///
    /// This computes the integers `q` and `r` such that self = q * rhs + r, with `q = self.div_euclid` and `r = self.rem_euclid(rhs)`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0, and will panic on overflow iff debug assertions are enabled.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(22).div_rem_euclid(I256::new(5)), (I256::new(4), I256::new(2)));
    /// assert_eq!(I256::new(-22).div_rem_euclid(I256::new(5)), (I256::new(-5), I256::new(3)));
    /// assert_eq!(I256::new(22).div_rem_euclid(I256::new(-5)), (I256::new(-4), I256::new(2)));
    /// assert_eq!(I256::new(-22).div_rem_euclid(I256::new(-5)), (I256::new(5), I256::new(3)));
    ///
    /// # assert_eq!(I256::new(20).div_rem_euclid(I256::new(5)), (I256::new(4), I256::new(0)));
    /// # assert_eq!(I256::new(-20).div_rem_euclid(I256::new(5)), (I256::new(-4), I256::new(0)));
    /// # assert_eq!(I256::new(20).div_rem_euclid(I256::new(-5)), (I256::new(-4), I256::new(0)));
    /// # assert_eq!(I256::new(-20).div_rem_euclid(I256::new(-5)), (I256::new(4), I256::new(0)));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        if self == Self::MIN && rhs == -1 {
            panic!("attempt to divide with overflow")
        }
        self.wrapping_div_rem_euclid(rhs)
    }

    /// Checked division. Computes `self.div_rem(rhs)`,
    /// returning `None` if `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!((I256::MIN + 1).checked_div_rem(I256::new(-1)), Some((I256::MAX, I256::new(0))));
    /// assert_eq!(I256::MIN.checked_div_rem(I256::new(-1)), None);
    /// assert_eq!(I256::new(1).checked_div_rem(I256::new(0)), None);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn checked_div_rem(self, rhs: Self) -> Option<(Self, Self)> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            if rhs.cmp(&I256::ZERO) == core::cmp::Ordering::Equal {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }

            let mut res: MaybeUninit<Self> = MaybeUninit::uninit();
            let mut rem: MaybeUninit<Self> = MaybeUninit::uninit();
            crate::intrinsics::idivmod4(&mut res, &self, &rhs, Some(&mut rem));
            unsafe { Some(((res.assume_init()), (rem.assume_init()))) }
        }
    }

    /// Checked Euclidean division. Computes `self.div_rem_euclid(rhs)`,
    /// returning `None` if `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!((I256::MIN + 1).checked_div_rem_euclid(I256::new(-1)), Some((I256::MAX, I256::new(0))));
    /// assert_eq!(I256::MIN.checked_div_rem_euclid(I256::new(-1)), None);
    /// assert_eq!(I256::new(1).checked_div_rem_euclid(I256::new(0)), None);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn checked_div_rem_euclid(self, rhs: Self) -> Option<(Self, Self)> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            if rhs.cmp(&I256::ZERO) == core::cmp::Ordering::Equal {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }

            Some(self.wrapping_div_rem_euclid(rhs))
        }
    }

    /// Saturating integer division. Computes `self.div_rem(rhs)`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).saturating_div_rem(I256::new(2)), (I256::new(2), I256::new(1)));
    /// assert_eq!(I256::MAX.saturating_div_rem(I256::new(-1)), (I256::MIN + 1, I256::new(0)));
    /// assert_eq!(I256::MIN.saturating_div_rem(I256::new(-1)), (I256::MAX, I256::new(0)));
    /// ```
    /// ```should_panic (expected = "attempt to divide by zero")
    /// # use ethnum::I256;
    /// let _ = I256::new(1).saturating_div_rem(I256::ZERO);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn saturating_div_rem(self, rhs: Self) -> (Self, Self) {
        match self.overflowing_div_rem(rhs) {
            (q, r, false) => (q, r),
            (_q, r, true) => (Self::MAX, r), // MIN / -1 is the only possible saturating overflow
        }
    }

    /// Saturating integer division. Computes `self.div_rem_euclid(rhs)`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).saturating_div_rem_euclid(I256::new(2)), (I256::new(2), I256::new(1)));
    /// assert_eq!(I256::MAX.saturating_div_rem_euclid(I256::new(-1)), (I256::MIN + 1, I256::new(0)));
    /// assert_eq!(I256::MIN.saturating_div_rem_euclid(I256::new(-1)), (I256::MAX, I256::new(0)));
    /// ```
    /// ```should_panic (expected = "attempt to divide by zero")
    /// # use ethnum::I256;
    /// let _ = I256::new(1).saturating_div_rem_euclid(I256::ZERO);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn saturating_div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        match self.overflowing_div_rem_euclid(rhs) {
            (q, r, false) => (q, r),
            (_q, r, true) => (Self::MAX, r),
        }
    }

    /// Performs integer and division and returns the quotient and the remainder as a tuple. This is equivelent to `(self.wrapping_div(rhs), self.wrapping_rem(rhs))`, but more effecient.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(22).wrapping_div_rem(I256::new(5)), (I256::new(4), I256::new(2)));
    /// assert_eq!(I256::new(-22).wrapping_div_rem(I256::new(5)), (I256::new(-4), I256::new(-2)));
    /// assert_eq!(I256::new(22).wrapping_div_rem(I256::new(-5)), (I256::new(-4), I256::new(2)));
    /// assert_eq!(I256::new(-22).wrapping_div_rem(I256::new(-5)), (I256::new(4), I256::new(-2)));
    ///
    /// assert_eq!(I256::MIN.wrapping_div_rem(I256::MINUS_ONE), (I256::MIN, I256::ZERO));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn wrapping_div_rem(self, rhs: Self) -> (Self, Self) {
        if rhs == 0 {
            if rhs.cmp(&I256::ZERO) != core::cmp::Ordering::Equal {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }
            panic!("attempt to divide by zero");
        }
        if rhs.cmp(&I256::ZERO) == core::cmp::Ordering::Equal {
            // The optimizer understands inequalities better
            unsafe { core::hint::unreachable_unchecked() }
        }

        let mut res: MaybeUninit<Self> = MaybeUninit::uninit();
        let mut rem: MaybeUninit<Self> = MaybeUninit::uninit();
        crate::intrinsics::idivmod4(&mut res, &self, &rhs, Some(&mut rem));
        unsafe { ((res.assume_init()), (rem.assume_init())) }
    }

    /// Performs euclidean division and returns the quotient and the remainder as a tuple.
    ///
    /// This computes the integers `q` and `r` such that self = q * rhs + r, with `q = self.wrapping_div_euclid` and `r = self.wrapping_rem_euclid(rhs)`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(22).wrapping_div_rem_euclid(I256::new(5)), (I256::new(4), I256::new(2)));
    /// assert_eq!(I256::new(-22).wrapping_div_rem_euclid(I256::new(5)), (I256::new(-5), I256::new(3)));
    /// assert_eq!(I256::new(22).wrapping_div_rem_euclid(I256::new(-5)), (I256::new(-4), I256::new(2)));
    /// assert_eq!(I256::new(-22).wrapping_div_rem_euclid(I256::new(-5)), (I256::new(5), I256::new(3)));
    ///
    /// assert_eq!(I256::MIN.wrapping_div_rem(I256::MINUS_ONE), (I256::MIN, I256::ZERO));
    /// # assert_eq!(I256::new(20).wrapping_div_rem_euclid(I256::new(5)), (I256::new(4), I256::new(0)));
    /// # assert_eq!(I256::new(-20).wrapping_div_rem_euclid(I256::new(5)), (I256::new(-4), I256::new(0)));
    /// # assert_eq!(I256::new(20).wrapping_div_rem_euclid(I256::new(-5)), (I256::new(-4), I256::new(0)));
    /// # assert_eq!(I256::new(-20).wrapping_div_rem_euclid(I256::new(-5)), (I256::new(4), I256::new(0)));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn wrapping_div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        let dividend_sign = self.is_negative();
        let quotient_sign = dividend_sign ^ rhs.is_negative();
        let abs_dividend = self.unsigned_abs();
        let abs_divisor = rhs.unsigned_abs();

        let (q, r) = abs_dividend.div_rem(abs_divisor);
        let mut quotient = q.as_i256();
        let mut remainder = r.as_i256();

        let adjust_remainder = dividend_sign && remainder != 0;
        if adjust_remainder {
            remainder = abs_divisor.as_i256() - remainder;
            // cannot overflow
        }
        if remainder.is_negative() || remainder.as_u256() >= abs_divisor {
            debug_assert!(false);
            unsafe { core::hint::unreachable_unchecked() }
        }

        if adjust_remainder {
            if quotient_sign {
                quotient = !quotient;
                // quotient = -quotient - 1
            } else {
                quotient += 1;
                // cannot overflow
            }
        } else if quotient_sign {
            quotient = quotient.wrapping_neg();
        }

        (quotient, remainder)
    }

    /// Calculates the quotient and the remainder when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the quotient and the remainder along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would occur then
    /// `self` is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).overflowing_div_rem(I256::new(2)), (I256::new(2), I256::new(1), false));
    /// assert_eq!(I256::MIN.overflowing_div_rem(I256::new(-1)), (I256::MIN, I256::new(0), true));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn overflowing_div_rem(self, rhs: Self) -> (Self, Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (self, Self::ZERO, true)
        } else {
            let (q, r) = self.wrapping_div_rem(rhs);
            (q, r, false)
        }
    }

    /// Calculates the quotient and remainder of Euclidean division `self.div_rem_euclid(rhs)`.
    ///
    /// Returns a tuple of the quotient and the remainder along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would occur then
    /// `self` is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).overflowing_div_rem_euclid(I256::new(2)), (I256::new(2), I256::new(1), false));
    /// assert_eq!(I256::MIN.overflowing_div_rem_euclid(I256::new(-1)), (I256::MIN, I256::new(0), true));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn overflowing_div_rem_euclid(self, rhs: Self) -> (Self, Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (self, Self::ZERO, true)
        } else {
            let (q, r) = self.wrapping_div_rem_euclid(rhs);
            (q, r, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::I256;

    #[test]
    #[allow(clippy::float_cmp)]
    fn converts_to_f64() {
        assert_eq!((-I256::from_words(1, 0)).as_f64(), -(2.0f64.powi(128)))
    }
}
