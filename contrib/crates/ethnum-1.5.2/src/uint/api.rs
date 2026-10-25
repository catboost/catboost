//! Module containing integer aritimetic methods closely following the Rust
//! standard library API for `uN` types.

use super::U256;
use crate::{intrinsics, I256};
use core::{
    mem::{self, MaybeUninit},
    num::ParseIntError,
};

impl U256 {
    /// The smallest value that can be represented by this integer type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::MIN, U256::new(0));
    /// ```
    pub const MIN: Self = Self([0; 2]);

    /// The largest value that can be represented by this integer type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(
    ///     U256::MAX.to_string(),
    ///     "115792089237316195423570985008687907853269984665640564039457584007913129639935",
    /// );
    /// ```
    pub const MAX: Self = Self([!0; 2]);

    /// The size of this integer type in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::BITS, 256);
    /// ```
    pub const BITS: u32 = 256;

    /// Converts a string slice in a given base to an integer.
    ///
    /// The string is expected to be an optional `+` sign followed by digits.
    /// Leading and trailing whitespace represent an error. Digits are a subset
    /// of these characters, depending on `radix`:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// # Panics
    ///
    /// This function panics if `radix` is not in the range from 2 to 36.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::from_str_radix("A", 16), Ok(U256::new(10)));
    /// ```
    #[inline]
    pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
        crate::parse::from_str_radix(src, radix, None)
    }

    /// Returns the number of ones in the binary representation of `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0b01001100);
    /// assert_eq!(n.count_ones(), 3);
    /// ```
    #[inline]
    pub const fn count_ones(self) -> u32 {
        let Self([a, b]) = self;
        a.count_ones() + b.count_ones()
    }

    /// Returns the number of zeros in the binary representation of `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::MIN.count_zeros(), 256);
    /// assert_eq!(U256::MAX.count_zeros(), 0);
    /// ```
    #[inline]
    pub const fn count_zeros(self) -> u32 {
        let Self([a, b]) = self;
        a.count_zeros() + b.count_zeros()
    }

    /// Returns the number of leading zeros in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::MAX >> 2u32;
    /// assert_eq!(n.leading_zeros(), 2);
    /// ```
    #[inline(always)]
    pub fn leading_zeros(self) -> u32 {
        intrinsics::signed::uctlz(&self)
    }

    /// Returns the number of trailing zeros in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0b0101000);
    /// assert_eq!(n.trailing_zeros(), 3);
    /// ```
    #[inline(always)]
    pub fn trailing_zeros(self) -> u32 {
        intrinsics::signed::ucttz(&self)
    }

    /// Returns the number of leading ones in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = !(U256::MAX >> 2u32);
    /// assert_eq!(n.leading_ones(), 2);
    /// ```
    #[inline]
    pub fn leading_ones(self) -> u32 {
        (!self).leading_zeros()
    }

    /// Returns the number of trailing ones in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0b1010111);
    /// assert_eq!(n.trailing_ones(), 3);
    /// ```
    #[inline]
    pub fn trailing_ones(self) -> u32 {
        (!self).trailing_zeros()
    }

    /// Shifts the bits to the left by a specified amount, `n`, wrapping the
    /// truncated bits to the end of the resulting integer.
    ///
    /// Please note this isn't the same operation as the `<<` shifting
    /// operator!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::from_words(
    ///     0x13f40000000000000000000000000000,
    ///     0x00000000000000000000000000004f76,
    /// );
    /// let m = U256::new(0x4f7613f4);
    /// assert_eq!(n.rotate_left(16), m);
    /// ```
    #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let mut r = MaybeUninit::uninit();
        intrinsics::signed::urol3(&mut r, &self, n);
        unsafe { r.assume_init() }
    }

    /// Shifts the bits to the right by a specified amount, `n`, wrapping the
    /// truncated bits to the beginning of the resulting integer.
    ///
    /// Please note this isn't the same operation as the `>>` shifting operator!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0x4f7613f4);
    /// let m = U256::from_words(
    ///     0x13f40000000000000000000000000000,
    ///     0x00000000000000000000000000004f76,
    /// );
    ///
    /// assert_eq!(n.rotate_right(16), m);
    /// ```
    #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
    #[inline(always)]
    pub fn rotate_right(self, n: u32) -> Self {
        let mut r = MaybeUninit::uninit();
        intrinsics::signed::uror3(&mut r, &self, n);
        unsafe { r.assume_init() }
    }

    /// Reverses the byte order of the integer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    /// assert_eq!(
    ///     n.swap_bytes(),
    ///     U256::from_words(
    ///         0x1f1e1d1c_1b1a1918_17161514_13121110,
    ///         0x0f0e0d0c_0b0a0908_07060504_03020100,
    ///     ),
    /// );
    /// ```
    #[inline]
    pub const fn swap_bytes(self) -> Self {
        let Self([a, b]) = self;
        Self([b.swap_bytes(), a.swap_bytes()])
    }

    /// Reverses the bit pattern of the integer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    /// assert_eq!(
    ///     n.reverse_bits(),
    ///     U256::from_words(
    ///         0xf878b838_d8589818_e868a828_c8488808,
    ///         0xf070b030_d0509010_e060a020_c0408000,
    ///     ),
    /// );
    /// ```
    #[inline]
    pub const fn reverse_bits(self) -> Self {
        let Self([a, b]) = self;
        Self([b.reverse_bits(), a.reverse_bits()])
    }

    /// Converts an integer from big endian to the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0x1A);
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(U256::from_be(n), n);
    /// } else {
    ///     assert_eq!(U256::from_be(n), n.swap_bytes());
    /// }
    /// ```
    #[inline(always)]
    #[allow(clippy::wrong_self_convention)]
    pub const fn from_be(x: Self) -> Self {
        #[cfg(target_endian = "big")]
        {
            x
        }
        #[cfg(not(target_endian = "big"))]
        {
            x.swap_bytes()
        }
    }

    /// Converts an integer from little endian to the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0x1A);
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(U256::from_le(n), n)
    /// } else {
    ///     assert_eq!(U256::from_le(n), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
    #[allow(clippy::wrong_self_convention)]
    pub const fn from_le(x: Self) -> Self {
        #[cfg(target_endian = "little")]
        {
            x
        }
        #[cfg(not(target_endian = "little"))]
        {
            x.swap_bytes()
        }
    }

    /// Converts `self` to big endian from the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0x1A);
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(n.to_be(), n)
    /// } else {
    ///     assert_eq!(n.to_be(), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
    pub const fn to_be(self) -> Self {
        #[cfg(target_endian = "big")]
        {
            self
        }
        #[cfg(not(target_endian = "big"))]
        {
            self.swap_bytes()
        }
    }

    /// Converts `self` to little endian from the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// let n = U256::new(0x1A);
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(n.to_le(), n)
    /// } else {
    ///     assert_eq!(n.to_le(), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
    pub const fn to_le(self) -> Self {
        #[cfg(target_endian = "little")]
        {
            self
        }
        #[cfg(not(target_endian = "little"))]
        {
            self.swap_bytes()
        }
    }

    /// Checked integer addition. Computes `self + rhs`, returning `None` if
    /// overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!((U256::MAX - 2).checked_add(U256::new(1)), Some(U256::MAX - 1));
    /// assert_eq!((U256::MAX - 2).checked_add(U256::new(3)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let (a, b) = self.overflowing_add(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked addition with a signed integer. Computes `self + rhs`,
    /// returning `None` if overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(U256::new(1).checked_add_signed(I256::new(2)), Some(U256::new(3)));
    /// assert_eq!(U256::new(1).checked_add_signed(I256::new(-2)), None);
    /// assert_eq!((U256::MAX - 2).checked_add_signed(I256::new(3)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn checked_add_signed(self, rhs: I256) -> Option<Self> {
        let (a, b) = self.overflowing_add_signed(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked integer subtraction. Computes `self - rhs`, returning `None` if
    /// overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(1).checked_sub(U256::new(1)), Some(U256::ZERO));
    /// assert_eq!(U256::new(0).checked_sub(U256::new(1)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (a, b) = self.overflowing_sub(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked integer multiplication. Computes `self * rhs`, returning `None`
    /// if overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).checked_mul(U256::new(1)), Some(U256::new(5)));
    /// assert_eq!(U256::MAX.checked_mul(U256::new(2)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let (a, b) = self.overflowing_mul(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked integer division. Computes `self / rhs`, returning `None` if
    /// `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(128).checked_div(U256::new(2)), Some(U256::new(64)));
    /// assert_eq!(U256::new(1).checked_div(U256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs == U256::ZERO {
            None
        } else {
            Some(self / rhs)
        }
    }

    /// Checked Euclidean division. Computes `self.div_euclid(rhs)`, returning
    /// `None` if `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(128).checked_div_euclid(U256::new(2)), Some(U256::new(64)));
    /// assert_eq!(U256::new(1).checked_div_euclid(U256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        if rhs == U256::ZERO {
            None
        } else {
            Some(self.div_euclid(rhs))
        }
    }

    /// Checked integer remainder. Computes `self % rhs`, returning `None` if
    /// `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).checked_rem(U256::new(2)), Some(U256::new(1)));
    /// assert_eq!(U256::new(5).checked_rem(U256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs == U256::ZERO {
            None
        } else {
            Some(self % rhs)
        }
    }

    /// Checked Euclidean modulo. Computes `self.rem_euclid(rhs)`, returning
    /// `None` if `rhs == 0`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).checked_rem_euclid(U256::new(2)), Some(U256::new(1)));
    /// assert_eq!(U256::new(5).checked_rem_euclid(U256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        if rhs == U256::ZERO {
            None
        } else {
            Some(self.rem_euclid(rhs))
        }
    }

    /// Checked negation. Computes `-self`, returning `None` unless `self == 0`.
    ///
    /// Note that negating any positive integer will overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::ZERO.checked_neg(), Some(U256::ZERO));
    /// assert_eq!(U256::new(1).checked_neg(), None);
    /// ```
    #[inline]
    pub fn checked_neg(self) -> Option<Self> {
        let (a, b) = self.overflowing_neg();
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked shift left. Computes `self << rhs`, returning `None` if `rhs` is
    /// larger than or equal to the number of bits in `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(0x1).checked_shl(4), Some(U256::new(0x10)));
    /// assert_eq!(U256::new(0x10).checked_shl(257), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_shl(self, rhs: u32) -> Option<Self> {
        let (a, b) = self.overflowing_shl(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked shift right. Computes `self >> rhs`, returning `None` if `rhs`
    /// is larger than or equal to the number of bits in `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(0x10).checked_shr(4), Some(U256::new(0x1)));
    /// assert_eq!(U256::new(0x10).checked_shr(257), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_shr(self, rhs: u32) -> Option<Self> {
        let (a, b) = self.overflowing_shr(rhs);
        if b {
            None
        } else {
            Some(a)
        }
    }

    /// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
    /// overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(2).checked_pow(5), Some(U256::new(32)));
    /// assert_eq!(U256::MAX.checked_pow(2), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_pow(self, mut exp: u32) -> Option<Self> {
        let mut base = self;
        let mut acc = U256::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.checked_mul(base)?;
            }
            exp /= 2;
            base = base.checked_mul(base)?;
        }

        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        if exp == 1 {
            acc = acc.checked_mul(base)?;
        }

        Some(acc)
    }

    /// Saturating integer addition. Computes `self + rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).saturating_add(U256::new(1)), U256::new(101));
    /// assert_eq!(U256::MAX.saturating_add(U256::new(127)), U256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        self.checked_add(rhs).unwrap_or(U256::MAX)
    }

    /// Saturating addition with a signed integer. Computes `self + rhs`,
    /// saturating at the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(U256::new(1).saturating_add_signed(I256::new(2)), U256::new(3));
    /// assert_eq!(U256::new(1).saturating_add_signed(I256::new(-2)), U256::new(0));
    /// assert_eq!((U256::MAX - 2).saturating_add_signed(I256::new(4)), U256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn saturating_add_signed(self, rhs: I256) -> Self {
        let (res, overflow) = self.overflowing_add(rhs.as_u256());
        if overflow == (rhs < 0) {
            res
        } else if overflow {
            Self::MAX
        } else {
            Self::ZERO
        }
    }

    /// Saturating integer subtraction. Computes `self - rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).saturating_sub(U256::new(27)), U256::new(73));
    /// assert_eq!(U256::new(13).saturating_sub(U256::new(127)), U256::new(0));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        self.checked_sub(rhs).unwrap_or(U256::MIN)
    }

    /// Saturating integer multiplication. Computes `self * rhs`, saturating at
    /// the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(2).saturating_mul(U256::new(10)), U256::new(20));
    /// assert_eq!((U256::MAX).saturating_mul(U256::new(10)), U256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        match self.checked_mul(rhs) {
            Some(x) => x,
            None => Self::MAX,
        }
    }

    /// Saturating integer division. Computes `self / rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).saturating_div(U256::new(2)), U256::new(2));
    /// ```
    ///
    /// ```should_panic
    /// # use ethnum::U256;
    /// let _ = U256::new(1).saturating_div(U256::ZERO);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn saturating_div(self, rhs: Self) -> Self {
        // on unsigned types, there is no overflow in integer division
        self.wrapping_div(rhs)
    }

    /// Saturating integer exponentiation. Computes `self.pow(exp)`, saturating
    /// at the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(4).saturating_pow(3), U256::new(64));
    /// assert_eq!(U256::MAX.saturating_pow(2), U256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_pow(self, exp: u32) -> Self {
        match self.checked_pow(exp) {
            Some(x) => x,
            None => Self::MAX,
        }
    }

    /// Wrapping (modular) addition. Computes `self + rhs`, wrapping around at
    /// the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(200).wrapping_add(U256::new(55)), U256::new(255));
    /// assert_eq!(U256::new(200).wrapping_add(U256::MAX), U256::new(199));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::uadd3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) addition with a signed integer. Computes
    /// `self + rhs`, wrapping around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(U256::new(1).wrapping_add_signed(I256::new(2)), U256::new(3));
    /// assert_eq!(U256::new(1).wrapping_add_signed(I256::new(-2)), U256::MAX);
    /// assert_eq!((U256::MAX - 2).wrapping_add_signed(I256::new(4)), U256::new(1));
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn wrapping_add_signed(self, rhs: I256) -> Self {
        self.wrapping_add(rhs.as_u256())
    }

    /// Wrapping (modular) subtraction. Computes `self - rhs`, wrapping around
    /// at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_sub(U256::new(100)), U256::new(0));
    /// assert_eq!(U256::new(100).wrapping_sub(U256::MAX), U256::new(101));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::usub3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) multiplication. Computes `self * rhs`, wrapping
    /// around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u8` is used here.
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(10).wrapping_mul(U256::new(12)), U256::new(120));
    /// assert_eq!(U256::MAX.wrapping_mul(U256::new(2)), U256::MAX - 1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::umul3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) division. Computes `self / rhs`. Wrapped division on
    /// unsigned types is just normal division. There's no way wrapping could
    /// ever happen. This function exists, so that all operations are accounted
    /// for in the wrapping operations.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_div(U256::new(10)), U256::new(10));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        self / rhs
    }

    /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`. Wrapped
    /// division on unsigned types is just normal division. There's no way
    /// wrapping could ever happen. This function exists, so that all operations
    /// are accounted for in the wrapping operations. Since, for the positive
    /// integers, all common definitions of division are equal, this is exactly
    /// equal to `self.wrapping_div(rhs)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_div_euclid(U256::new(10)), U256::new(10));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_div_euclid(self, rhs: Self) -> Self {
        self / rhs
    }

    /// Wrapping (modular) remainder. Computes `self % rhs`. Wrapped remainder
    /// calculation on unsigned types is just the regular remainder calculation.
    /// There's no way wrapping could ever happen. This function exists, so that
    /// all operations are accounted for in the wrapping operations.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_rem(U256::new(10)), U256::new(0));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_rem(self, rhs: Self) -> Self {
        self % rhs
    }

    /// Wrapping Euclidean modulo. Computes `self.rem_euclid(rhs)`. Wrapped
    /// modulo calculation on unsigned types is just the regular remainder
    /// calculation. There's no way wrapping could ever happen. This function
    /// exists, so that all operations are accounted for in the wrapping
    /// operations. Since, for the positive integers, all common definitions of
    /// division are equal, this is exactly equal to `self.wrapping_rem(rhs)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).wrapping_rem_euclid(U256::new(10)), U256::new(0));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_rem_euclid(self, rhs: Self) -> Self {
        self % rhs
    }

    /// Wrapping (modular) negation. Computes `-self`, wrapping around at the
    /// boundary of the type.
    ///
    /// Since unsigned types do not have negative equivalents all applications
    /// of this function will wrap (except for `-0`). For values smaller than
    /// the corresponding signed type's maximum the result is the same as
    /// casting the corresponding signed value. Any larger values are equivalent
    /// to `MAX + 1 - (val - MAX - 1)` where `MAX` is the corresponding signed
    /// type's maximum.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `i8` is used here.
    ///
    /// ```
    /// # use ethnum::{U256, AsU256};
    /// assert_eq!(U256::new(100).wrapping_neg(), (-100i128).as_u256());
    /// assert_eq!(
    ///     U256::from_words(i128::MIN as _, 0).wrapping_neg(),
    ///     U256::from_words(i128::MIN as _, 0),
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_neg(self) -> Self {
        self.overflowing_neg().0
    }

    /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping
    /// shift-left is restricted to the range of the type, rather than the bits
    /// shifted out of the LHS being returned to the other end. The primitive
    /// integer types all implement a `rotate_left` function, which maybe what
    /// you want instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(1).wrapping_shl(7), U256::new(128));
    /// assert_eq!(U256::new(1).wrapping_shl(128), U256::from_words(1, 0));
    /// assert_eq!(U256::new(1).wrapping_shl(256), U256::new(1));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_shl(self, rhs: u32) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::ushl3(&mut result, &self, rhs & 0xff);
        unsafe { result.assume_init() }
    }

    /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-right; the RHS of a
    /// wrapping shift-right is restricted to the range of the type, rather than
    /// the bits shifted out of the LHS being returned to the other end. The
    /// primitive integer types all implement a `rotate_right` function, which
    /// may be what you want instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(128).wrapping_shr(7), U256::new(1));
    /// assert_eq!(U256::from_words(128, 0).wrapping_shr(128), U256::new(128));
    /// assert_eq!(U256::new(128).wrapping_shr(256), U256::new(128));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_shr(self, rhs: u32) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::ushr3(&mut result, &self, rhs & 0xff);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`, wrapping
    /// around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(3).wrapping_pow(5), U256::new(243));
    /// assert_eq!(
    ///     U256::new(1337).wrapping_pow(42),
    ///     U256::from_words(
    ///         45367329835866155830012179193722278514,
    ///         159264946433345088039815329994094210673,
    ///     ),
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_pow(self, mut exp: u32) -> Self {
        let mut base = self;
        let mut acc = U256::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.wrapping_mul(base);
            }
            exp /= 2;
            base = base.wrapping_mul(base);
        }

        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        if exp == 1 {
            acc = acc.wrapping_mul(base);
        }

        acc
    }

    /// Calculates `self` + `rhs`
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).overflowing_add(U256::new(2)), (U256::new(7), false));
    /// assert_eq!(U256::MAX.overflowing_add(U256::new(1)), (U256::new(0), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::uaddc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Calculates `self` + `rhs` with a signed `rhs`
    ///
    /// Returns a tuple of the addition along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would
    /// have occurred then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(U256::new(1).overflowing_add_signed(I256::new(2)), (U256::new(3), false));
    /// assert_eq!(U256::new(1).overflowing_add_signed(I256::new(-2)), (U256::MAX, true));
    /// assert_eq!((U256::MAX - 2).overflowing_add_signed(I256::new(4)), (U256::new(1), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn overflowing_add_signed(self, rhs: I256) -> (Self, bool) {
        let (res, overflowed) = self.overflowing_add(rhs.as_u256());
        (res, overflowed ^ (rhs < 0))
    }

    /// Calculates `self` - `rhs`
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would have
    /// occurred then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).overflowing_sub(U256::new(2)), (U256::new(3), false));
    /// assert_eq!(U256::new(0).overflowing_sub(U256::new(1)), (U256::MAX, true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::usubc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Computes the absolute difference between `self` and `other`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(100).abs_diff(U256::new(80)), 20);
    /// assert_eq!(U256::new(100).abs_diff(U256::new(110)), 10);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn abs_diff(self, other: Self) -> Self {
        if self < other {
            other - self
        } else {
            self - other
        }
    }

    /// Calculates the multiplication of `self` and `rhs`.
    ///
    /// Returns a tuple of the multiplication along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would have
    /// occurred then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u32` is used here.
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(5).overflowing_mul(U256::new(2)), (U256::new(10), false));
    /// assert_eq!(
    ///     U256::MAX.overflowing_mul(U256::new(2)),
    ///     (U256::MAX - 1, true),
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::umulc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Calculates the divisor when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
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
    /// assert_eq!(U256::new(5).overflowing_div(U256::new(2)), (U256::new(2), false));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_div(self, rhs: Self) -> (Self, bool) {
        (self / rhs, false)
    }

    /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
    /// an arithmetic overflow would occur. Note that for unsigned integers
    /// overflow never occurs, so the second value is always `false`.  Since,
    /// for the positive integers, all common definitions of division are equal,
    /// this is exactly equal to `self.overflowing_div(rhs)`.
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
    /// assert_eq!(U256::new(5).overflowing_div_euclid(U256::new(2)), (U256::new(2), false));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
        (self / rhs, false)
    }

    /// Calculates the remainder when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the remainder after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. Note that for
    /// unsigned integers overflow never occurs, so the second value is always
    /// `false`.
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
    /// assert_eq!(U256::new(5).overflowing_rem(U256::new(2)), (U256::new(1), false));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
        (self % rhs, false)
    }

    /// Calculates the remainder `self.rem_euclid(rhs)` as if by Euclidean
    /// division.
    ///
    /// Returns a tuple of the modulo after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. Note that for
    /// unsigned integers overflow never occurs, so the second value is always
    /// `false`. Since, for the positive integers, all common definitions of
    /// division are equal, this operation is exactly equal to
    /// `self.overflowing_rem(rhs)`.
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
    /// assert_eq!(U256::new(5).overflowing_rem_euclid(U256::new(2)), (U256::new(1), false));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
        (self % rhs, false)
    }

    /// Negates self in an overflowing fashion.
    ///
    /// Returns `!self + 1` using wrapping operations to return the value that
    /// represents the negation of this unsigned value. Note that for positive
    /// unsigned values overflow always occurs, but negating 0 does not
    /// overflow.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::{U256, AsU256};
    /// assert_eq!(U256::new(0).overflowing_neg(), (U256::new(0), false));
    /// assert_eq!(U256::new(2).overflowing_neg(), ((-2i32).as_u256(), true));
    /// ```
    #[inline]
    pub fn overflowing_neg(self) -> (Self, bool) {
        ((!self).wrapping_add(U256::ONE), self != U256::ZERO)
    }

    /// Shifts self left by `rhs` bits.
    ///
    /// Returns a tuple of the shifted version of self along with a boolean
    /// indicating whether the shift value was larger than or equal to the
    /// number of bits. If the shift value is too large, then value is masked
    /// (N-1) where N is the number of bits, and this value is then used to
    /// perform the shift.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(0x1).overflowing_shl(4), (U256::new(0x10), false));
    /// assert_eq!(U256::new(0x1).overflowing_shl(260), (U256::new(0x10), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
        (self.wrapping_shl(rhs), rhs > 255)
    }

    /// Shifts self right by `rhs` bits.
    ///
    /// Returns a tuple of the shifted version of self along with a boolean
    /// indicating whether the shift value was larger than or equal to the
    /// number of bits. If the shift value is too large, then value is masked
    /// (N-1) where N is the number of bits, and this value is then used to
    /// perform the shift.
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(0x10).overflowing_shr(4), (U256::new(0x1), false));
    /// assert_eq!(U256::new(0x10).overflowing_shr(260), (U256::new(0x1), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
        (self.wrapping_shr(rhs), rhs > 255)
    }

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    ///
    /// Returns a tuple of the exponentiation along with a bool indicating
    /// whether an overflow happened.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(3).overflowing_pow(5), (U256::new(243), false));
    /// assert_eq!(
    ///     U256::new(1337).overflowing_pow(42),
    ///     (
    ///         U256::from_words(
    ///             45367329835866155830012179193722278514,
    ///             159264946433345088039815329994094210673,
    ///         ),
    ///         true,
    ///     )
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn overflowing_pow(self, mut exp: u32) -> (Self, bool) {
        let mut base = self;
        let mut acc = U256::ONE;
        let mut overflown = false;
        // Scratch space for storing results of overflowing_mul.
        let mut r;

        while exp > 1 {
            if (exp & 1) == 1 {
                r = acc.overflowing_mul(base);
                acc = r.0;
                overflown |= r.1;
            }
            exp /= 2;
            r = base.overflowing_mul(base);
            base = r.0;
            overflown |= r.1;
        }

        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        if exp == 1 {
            r = acc.overflowing_mul(base);
            acc = r.0;
            overflown |= r.1;
        }

        (acc, overflown)
    }

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(2).pow(5), U256::new(32));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn pow(self, mut exp: u32) -> Self {
        let mut base = self;
        let mut acc = U256::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            exp /= 2;
            base = base * base;
        }

        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        if exp == 1 {
            acc *= base;
        }

        acc
    }

    /// Performs Euclidean division.
    ///
    /// Since, for the positive integers, all common definitions of division are
    /// equal, this is exactly equal to `self / rhs`.
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
    /// assert_eq!(U256::new(7).div_euclid(U256::new(4)), U256::new(1));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn div_euclid(self, rhs: Self) -> Self {
        self / rhs
    }

    /// Calculates the least remainder of `self (mod rhs)`.
    ///
    /// Since, for the positive integers, all common definitions of division are
    /// equal, this is exactly equal to `self % rhs`.
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
    /// assert_eq!(U256::new(7).rem_euclid(U256::new(4)), U256::new(3));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        self % rhs
    }

    /// Returns `true` if and only if `self == 2^k` for some `k`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert!(U256::new(16).is_power_of_two());
    /// assert!(!U256::new(10).is_power_of_two());
    /// ```
    #[inline]
    pub fn is_power_of_two(self) -> bool {
        self.count_ones() == 1
    }

    /// Returns one less than next power of two. (For 8u8 next power of two is
    /// 8u8 and for 6u8 it is 8u8).
    ///
    /// 8u8.one_less_than_next_power_of_two() == 7
    /// 6u8.one_less_than_next_power_of_two() == 7
    ///
    /// This method cannot overflow, as in the `next_power_of_two` overflow
    /// cases it instead ends up returning the maximum value of the type, and
    /// can return 0 for 0.
    #[inline]
    fn one_less_than_next_power_of_two(self) -> Self {
        if self <= 1 {
            return U256::ZERO;
        }

        let p = self - 1;
        let z = p.leading_zeros();
        U256::MAX >> z
    }

    /// Returns the smallest power of two greater than or equal to `self`.
    ///
    /// When return value overflows (i.e., `self > (1 << (N-1))` for type `uN`),
    /// it panics in debug mode and return value is wrapped to 0 in release mode
    /// (the only situation in which method can return 0).
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(2).next_power_of_two(), U256::new(2));
    /// assert_eq!(U256::new(3).next_power_of_two(), U256::new(4));
    /// ```
    #[inline]
    pub fn next_power_of_two(self) -> Self {
        self.one_less_than_next_power_of_two() + 1
    }

    /// Returns the smallest power of two greater than or equal to `n`. If the
    /// next power of two is greater than the type's maximum value, `None` is
    /// returned, otherwise the power of two is wrapped in `Some`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::new(2).checked_next_power_of_two(), Some(U256::new(2)));
    /// assert_eq!(U256::new(3).checked_next_power_of_two(), Some(U256::new(4)));
    /// assert_eq!(U256::MAX.checked_next_power_of_two(), None);
    /// ```
    #[inline]
    pub fn checked_next_power_of_two(self) -> Option<Self> {
        self.one_less_than_next_power_of_two()
            .checked_add(U256::ONE)
    }

    /// Return the memory representation of this integer as a byte array in big
    /// endian (network) byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let bytes = U256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    /// assert_eq!(
    ///     bytes.to_be_bytes(),
    ///     [
    ///         0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ///         0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ///     ],
    /// );
    /// ```
    #[inline]
    pub fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()] {
        self.to_be().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in
    /// little endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let bytes = U256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    /// assert_eq!(
    ///     bytes.to_le_bytes(),
    ///     [
    ///         0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
    ///         0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    ///     ],
    /// );
    /// ```
    #[inline]
    pub fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()] {
        self.to_le().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code should
    /// use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: #method.to_be_bytes
    /// [`to_le_bytes`]: #method.to_le_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let bytes = U256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    /// assert_eq!(
    ///     bytes.to_ne_bytes(),
    ///     if cfg!(target_endian = "big") {
    ///         [
    ///             0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ///             0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ///         ]
    ///     } else {
    ///         [
    ///             0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
    ///             0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    ///         ]
    ///     }
    /// );
    /// ```
    #[inline]
    pub fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()] {
        unsafe { mem::transmute(self) }
    }

    /// Create an integer value from its representation as a byte array in big
    /// endian.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let value = U256::from_be_bytes([
    ///     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ///     0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    /// ]);
    /// assert_eq!(
    ///     value,
    ///     U256::from_words(
    ///         0x00010203_04050607_08090a0b_0c0d0e0f,
    ///         0x10111213_14151617_18191a1b_1c1d1e1f,
    ///     ),
    /// );
    /// ```
    ///
    /// When starting from a slice rather than an array, fallible conversion
    /// APIs can be used:
    ///
    /// ```
    /// # use ethnum::U256;
    /// use std::convert::TryInto;
    ///
    /// fn read_be_u256(input: &mut &[u8]) -> U256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<U256>());
    ///     *input = rest;
    ///     U256::from_be_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        Self::from_be(Self::from_ne_bytes(bytes))
    }

    /// Create an integer value from its representation as a byte array in
    /// little endian.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let value = U256::from_le_bytes([
    ///     0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
    ///     0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    /// ]);
    /// assert_eq!(
    ///     value,
    ///     U256::from_words(
    ///         0x00010203_04050607_08090a0b_0c0d0e0f,
    ///         0x10111213_14151617_18191a1b_1c1d1e1f,
    ///     ),
    /// );
    /// ```
    ///
    /// When starting from a slice rather than an array, fallible conversion
    /// APIs can be used:
    ///
    /// ```
    /// # use ethnum::U256;
    /// use std::convert::TryInto;
    ///
    /// fn read_be_u256(input: &mut &[u8]) -> U256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<U256>());
    ///     *input = rest;
    ///     U256::from_le_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        Self::from_le(Self::from_ne_bytes(bytes))
    }

    /// Create an integer value from its memory representation as a byte array
    /// in native endianness.
    ///
    /// As the target platform's native endianness is used, portable code likely
    /// wants to use [`from_be_bytes`] or [`from_le_bytes`], as appropriate
    /// instead.
    ///
    /// [`from_be_bytes`]: #method.from_be_bytes
    /// [`from_le_bytes`]: #method.from_le_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::U256;
    /// let value = U256::from_ne_bytes(if cfg!(target_endian = "big") {
    ///     [
    ///         0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ///         0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ///     ]
    /// } else {
    ///     [
    ///         0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
    ///         0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    ///     ]
    /// });
    /// assert_eq!(
    ///     value,
    ///     U256::from_words(
    ///         0x00010203_04050607_08090a0b_0c0d0e0f,
    ///         0x10111213_14151617_18191a1b_1c1d1e1f,
    ///     ),
    /// );
    /// ```
    ///
    /// When starting from a slice rather than an array, fallible conversion
    /// APIs can be used:
    ///
    /// ```
    /// # use ethnum::U256;
    /// use std::convert::TryInto;
    ///
    /// fn read_be_u256(input: &mut &[u8]) -> U256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<U256>());
    ///     *input = rest;
    ///     U256::from_ne_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        unsafe { mem::transmute(bytes) }
    }
}
