//! Root module for 256-bit unsigned integer type.

mod api;
mod cmp;
mod convert;
mod fmt;
mod iter;
mod ops;
mod parse;

pub use self::convert::AsU256;
use crate::I256;
use core::{mem::MaybeUninit, num::ParseIntError};

/// A 256-bit unsigned integer type.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct U256(pub [u128; 2]);

impl U256 {
    /// The additive identity for this integer type, i.e. `0`.
    pub const ZERO: Self = U256([0; 2]);

    /// The multiplicative identity for this integer type, i.e. `1`.
    pub const ONE: Self = U256::new(1);

    /// Creates a new 256-bit integer value from a primitive `u128` integer.
    #[inline]
    pub const fn new(value: u128) -> Self {
        U256::from_words(0, value)
    }

    /// Creates a new 256-bit integer value from high and low words.
    #[inline]
    pub const fn from_words(hi: u128, lo: u128) -> Self {
        #[cfg(target_endian = "little")]
        {
            U256([lo, hi])
        }
        #[cfg(target_endian = "big")]
        {
            U256([hi, lo])
        }
    }

    /// Splits a 256-bit integer into high and low words.
    #[inline]
    pub const fn into_words(self) -> (u128, u128) {
        #[cfg(target_endian = "little")]
        {
            let U256([lo, hi]) = self;
            (hi, lo)
        }
        #[cfg(target_endian = "big")]
        {
            let U256([hi, lo]) = self;
            (hi, lo)
        }
    }

    /// Get the low 128-bit word for this unsigned integer.
    #[inline]
    pub fn low(&self) -> &u128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[1]
        }
    }

    /// Get the low 128-bit word for this unsigned integer as a mutable
    /// reference.
    #[inline]
    pub fn low_mut(&mut self) -> &mut u128 {
        #[cfg(target_endian = "little")]
        {
            &mut self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &mut self.0[1]
        }
    }

    /// Get the high 128-bit word for this unsigned integer.
    #[inline]
    pub fn high(&self) -> &u128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[1]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[0]
        }
    }

    /// Get the high 128-bit word for this unsigned integer as a mutable
    /// reference.
    #[inline]
    pub fn high_mut(&mut self) -> &mut u128 {
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
    /// The string is expected to be an optional `+` sign followed by the `0x`
    /// prefix and finally the digits. Leading and trailing whitespace represent
    /// an error.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::from_str_hex("0x2A"), Ok(U256::new(42)));
    /// ```
    pub fn from_str_hex(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_radix(src, 16, Some("0x"))
    }

    /// Converts a prefixed string slice in a base determined by the prefix to
    /// an integer.
    ///
    /// The string is expected to be an optional `+` sign followed by the one of
    /// the supported prefixes and finally the digits. Leading and trailing
    /// whitespace represent an error. The base is determined based on the
    /// prefix:
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
    /// # use ethnum::U256;
    /// assert_eq!(U256::from_str_prefixed("0b101"), Ok(U256::new(0b101)));
    /// assert_eq!(U256::from_str_prefixed("0o17"), Ok(U256::new(0o17)));
    /// assert_eq!(U256::from_str_prefixed("0xa"), Ok(U256::new(0xa)));
    /// assert_eq!(U256::from_str_prefixed("42"), Ok(U256::new(42)));
    /// ```
    pub fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_prefixed(src)
    }

    /// Same as [`U256::from_str_prefixed`] but as a `const fn`. This method is
    /// not intended to be used directly but rather through the [`crate::uint`]
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

    /// Cast to a `I256`.
    #[inline]
    pub const fn as_i256(self) -> I256 {
        let Self([a, b]) = self;
        I256([a as _, b as _])
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
        lo
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
        match self.into_words() {
            (0, lo) => lo as _,
            _ => f32::INFINITY,
        }
    }

    /// Cast to a primitive `f64`.
    #[inline]
    pub fn as_f64(self) -> f64 {
        // NOTE: Binary representation of 2**128. This is used because `powi` is
        // neither `const` nor `no_std`.
        const HI: u64 = 0x47f0000000000000;
        let (hi, lo) = self.into_words();
        (hi as f64) * f64::from_bits(HI) + (lo as f64)
    }

    /// Performs integer and division and returns the quotient and the remainder as a tuple. This is faster than computing the quotient and remainder separately.
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
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(7).div_rem(U256::new(4)), (U256::new(1), U256::new(3)));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn div_rem(self, rhs: Self) -> (Self, Self) {
        if rhs == 0 {
            if rhs > 0 {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }
            panic!("attempt to divide by zero");
        }
        if rhs <= 0 {
            // The optimizer understands inequalities better
            unsafe { core::hint::unreachable_unchecked() }
        }

        let mut res: MaybeUninit<Self> = MaybeUninit::uninit();
        let mut rem: MaybeUninit<Self> = MaybeUninit::uninit();
        crate::intrinsics::udivmod4(&mut res, &self, &rhs, Some(&mut rem));
        let ret = unsafe { ((res.assume_init()), (rem.assume_init())) };

        // This helps the optimizer figure out when it can use smaller
        // operands for later functions.
        // SAFETY: Relies on the fact that rhs is at least 1.
        if ret.1 >= rhs {
            unsafe { core::hint::unreachable_unchecked() }
        }
        if ret.1 > self {
            unsafe { core::hint::unreachable_unchecked() }
        }
        if ret.0 > self {
            unsafe { core::hint::unreachable_unchecked() }
        }

        ret
    }

    /// Performs Euclidean division.
    ///
    /// Since, for the positive integers, all common definitions of division are
    /// equal, this is exactly equal to `self.div_rem(rhs)`.
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
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(7).div_rem_euclid(U256::new(4)), (U256::new(1), U256::new(3)));
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        self.div_rem(rhs)
    }

    /// Checked integer division. Computes `self.div_rem(rhs)`, returning `None` if
    /// `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(128).checked_div_rem(U256::new(2)), Some((U256::new(64), U256::new(0))));
    /// assert_eq!(U256::new(1).checked_div_rem(U256::new(0)), None);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    pub fn checked_div_rem(self, rhs: Self) -> Option<(Self, Self)> {
        if rhs == Self::ZERO {
            if rhs > 0 {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }

            None
        } else {
            if rhs <= 0 {
                // The optimizer understands inequalities better
                unsafe { core::hint::unreachable_unchecked() }
            }

            Some(self.div_rem(rhs))
        }
    }

    /// Checked Euclidean division. Computes `self.div_rem_euclid(rhs)`, returning `None` if
    /// `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(128).checked_div_rem_euclid(U256::new(2)), Some((U256::new(64), U256::new(0))));
    /// assert_eq!(U256::new(1).checked_div_rem_euclid(U256::new(0)), None);
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    pub fn checked_div_rem_euclid(self, rhs: Self) -> Option<(Self, Self)> {
        self.checked_div_rem(rhs)
    }

    /// Saturating integer division. Computes `self.div_rem(rhs)`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).saturating_div_rem(U256::new(2)), (U256::new(2), U256::new(1)));
    /// ```
    ///
    /// ```should_panic (expected = "attempt to divide by zero")
    /// # use ethnum::U256;
    /// let _ = U256::new(1).saturating_div_rem(U256::ZERO);
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn saturating_div_rem(self, rhs: Self) -> (Self, Self) {
        self.div_rem(rhs)
    }

    /// Saturating integer division. Computes `self.div_rem_euclid(rhs)`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).saturating_div_rem_euclid(U256::new(2)), (U256::new(2), U256::new(1)));
    /// ```
    ///
    /// ```should_panic (expected = "attempt to divide by zero")
    /// # use ethnum::U256;
    /// let _ = U256::new(1).saturating_div_rem_euclid(U256::ZERO);
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn saturating_div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        self.div_rem(rhs)
    }

    /// Wrapping (modular) division. Computes `self.div_rem(rhs)`. Wrapped division on
    /// unsigned types is just normal division. There's no way wrapping could
    /// ever happen. This function exists, so that all operations are accounted
    /// for in the wrapping operations.
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
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_div_rem(U256::new(10)), (U256::new(10), U256::new(0)));
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn wrapping_div_rem(self, rhs: Self) -> (Self, Self) {
        self.div_rem(rhs)
    }

    /// Wrapping Euclidean division. Computes `self.div_rem_euclid(rhs)`. Wrapped division on
    /// unsigned types is just normal division. There's no way wrapping could
    /// ever happen. This function exists, so that all operations are accounted
    /// for in the wrapping operations. Since, for the positive integers, all common
    /// definitions of division are equal, this is exactly equal to `self.div_rem(rhs)`.
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
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_div_rem_euclid(U256::new(10)), (U256::new(10), U256::new(0)));
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn wrapping_div_rem_euclid(self, rhs: Self) -> (Self, Self) {
        self.div_rem(rhs)
    }

    /// Calculates the quotient and the remainder when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the divisor and the remainder along with a boolean indicating whether
    /// an arithmetic overflow would occur. Note that for unsigned integers
    /// overflow never occurs, so the second value is always `false`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).overflowing_div_rem(U256::new(2)), (U256::new(2), U256::new(1), false));
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn overflowing_div_rem(self, rhs: Self) -> (Self, Self, bool) {
        let (q, r) = self.div_rem(rhs);
        (q, r, false)
    }

    /// Calculates the quotient of Euclidean division `self.div_rem_euclid(rhs)`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
    /// an arithmetic overflow would occur. Note that for unsigned integers
    /// overflow never occurs, so the second value is always `false`.  Since,
    /// for the positive integers, all common definitions of division are equal,
    /// this is exactly equal to `self.div_rem(rhs)`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).overflowing_div_rem_euclid(U256::new(2)), (U256::new(2), U256::new(1), false));
    /// ```
    #[inline(always)]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[track_caller]
    pub fn overflowing_div_rem_euclid(self, rhs: Self) -> (Self, Self, bool) {
        let (q, r) = self.div_rem(rhs);
        (q, r, false)
    }
}

#[cfg(test)]
mod tests {
    use crate::uint::U256;

    #[test]
    #[allow(clippy::float_cmp)]
    fn converts_to_f64() {
        assert_eq!(U256::from_words(1, 0).as_f64(), 2.0f64.powi(128))
    }
}
