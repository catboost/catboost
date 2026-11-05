//! Module containing integer aritimetic methods closely following the Rust
//! standard library API for `iN` types.

use crate::{intrinsics, I256, U256};
use core::{
    mem::{self, MaybeUninit},
    num::ParseIntError,
};

impl I256 {
    /// The smallest value that can be represented by this integer type,
    /// -2<sup>255</sup>.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(
    ///     I256::MIN.to_string(),
    ///     "-57896044618658097711785492504343953926634992332820282019728792003956564819968",
    /// );
    /// ```
    pub const MIN: Self = Self::from_words(i128::MIN, 0);

    /// The largest value that can be represented by this integer type,
    /// 2<sup>255</sup> - 1.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(
    ///     I256::MAX.to_string(),
    ///     "57896044618658097711785492504343953926634992332820282019728792003956564819967",
    /// );
    /// ```
    pub const MAX: Self = Self::from_words(i128::MAX, -1);

    /// The size of this integer type in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::BITS, 256);
    /// ```
    pub const BITS: u32 = 256;

    /// Converts a string slice in a given base to an integer.
    ///
    /// The string is expected to be an optional `+` or `-` sign followed by
    /// digits. Leading and trailing whitespace represent an error. Digits are a
    /// subset of these characters, depending on `radix`:
    ///
    ///  * `0-9`
    ///  * `a-z`
    ///  * `A-Z`
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::from_str_radix("A", 16), Ok(I256::new(10)));
    /// ```
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
    /// # use ethnum::I256;
    /// let n = I256::new(0b100_0000);
    ///
    /// assert_eq!(n.count_ones(), 1);
    /// ```
    ///
    #[doc(alias = "popcount")]
    #[doc(alias = "popcnt")]
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::MAX.count_zeros(), 1);
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
    /// # use ethnum::I256;
    /// let n = I256::new(-1);
    ///
    /// assert_eq!(n.leading_zeros(), 0);
    /// ```
    #[inline(always)]
    pub fn leading_zeros(self) -> u32 {
        intrinsics::signed::ictlz(&self)
    }

    /// Returns the number of trailing zeros in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::new(-4);
    ///
    /// assert_eq!(n.trailing_zeros(), 2);
    /// ```
    #[inline(always)]
    pub fn trailing_zeros(self) -> u32 {
        intrinsics::signed::icttz(&self)
    }

    /// Returns the number of leading ones in the binary representation of
    /// `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::new(-1);
    ///
    /// assert_eq!(n.leading_ones(), 256);
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
    /// # use ethnum::I256;
    /// let n = I256::new(3);
    ///
    /// assert_eq!(n.trailing_ones(), 2);
    /// ```
    #[inline]
    pub fn trailing_ones(self) -> u32 {
        (!self).trailing_zeros()
    }

    /// Shifts the bits to the left by a specified amount, `n`,
    /// wrapping the truncated bits to the end of the resulting integer.
    ///
    /// Please note this isn't the same operation as the `<<` shifting operator!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::from_words(
    ///     0x13f40000000000000000000000000000,
    ///     0x00000000000000000000000000004f76,
    /// );
    /// let m = I256::new(0x4f7613f4);
    ///
    /// assert_eq!(n.rotate_left(16), m);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let mut r = MaybeUninit::uninit();
        intrinsics::signed::irol3(&mut r, &self, n);
        unsafe { r.assume_init() }
    }

    /// Shifts the bits to the right by a specified amount, `n`,
    /// wrapping the truncated bits to the beginning of the resulting
    /// integer.
    ///
    /// Please note this isn't the same operation as the `>>` shifting operator!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::new(0x4f7613f4);
    /// let m = I256::from_words(
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
        intrinsics::signed::iror3(&mut r, &self, n);
        unsafe { r.assume_init() }
    }

    /// Reverses the byte order of the integer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    ///
    /// assert_eq!(
    ///     n.swap_bytes(),
    ///     I256::from_words(
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

    /// Reverses the order of bits in the integer. The least significant bit
    /// becomes the most significant bit, second least-significant bit becomes
    /// second most-significant bit, etc.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let n = I256::from_words(
    ///     0x00010203_04050607_08090a0b_0c0d0e0f,
    ///     0x10111213_14151617_18191a1b_1c1d1e1f,
    /// );
    ///
    /// assert_eq!(
    ///     n.reverse_bits(),
    ///     I256::from_words(
    ///         0xf878b838_d8589818_e868a828_c8488808_u128 as _,
    ///         0xf070b030_d0509010_e060a020_c0408000_u128 as _,
    ///     ),
    /// );
    /// ```
    #[inline]
    #[must_use]
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
    /// # use ethnum::I256;
    /// let n = I256::new(0x1A);
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(I256::from_be(n), n)
    /// } else {
    ///     assert_eq!(I256::from_be(n), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
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
    /// # use ethnum::I256;
    /// let n = I256::new(0x1A);
    ///
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(I256::from_le(n), n)
    /// } else {
    ///     assert_eq!(I256::from_le(n), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
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
    /// # use ethnum::I256;
    /// let n = I256::new(0x1A);
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(n.to_be(), n)
    /// } else {
    ///     assert_eq!(n.to_be(), n.swap_bytes())
    /// }
    /// ```
    #[inline(always)]
    pub const fn to_be(self) -> Self {
        // or not to be?
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
    /// # use ethnum::I256;
    /// let n = I256::new(0x1A);
    ///
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

    /// Checked integer addition. Computes `self + rhs`, returning `None`
    /// if overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!((I256::MAX - 2).checked_add(I256::new(1)), Some(I256::MAX - 1));
    /// assert_eq!((I256::MAX - 2).checked_add(I256::new(3)), None);
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

    /// Checked addition with an unsigned integer. Computes `self + rhs`,
    /// returning `None` if overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(1).checked_add_unsigned(U256::new(2)), Some(I256::new(3)));
    /// assert_eq!((I256::MAX - 2).checked_add_unsigned(U256::new(3)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn checked_add_unsigned(self, rhs: U256) -> Option<Self> {
        let (a, b) = self.overflowing_add_unsigned(rhs);
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
    /// # use ethnum::I256;
    /// assert_eq!((I256::MIN + 2).checked_sub(I256::new(1)), Some(I256::MIN + 1));
    /// assert_eq!((I256::MIN + 2).checked_sub(I256::new(3)), None);
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

    /// Checked subtraction with an unsigned integer. Computes `self - rhs`,
    /// returning `None` if overflow occurred.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(1).checked_sub_unsigned(U256::new(2)), Some(I256::new(-1)));
    /// assert_eq!((I256::MIN + 2).checked_sub_unsigned(U256::new(3)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn checked_sub_unsigned(self, rhs: U256) -> Option<Self> {
        let (a, b) = self.overflowing_sub_unsigned(rhs);
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::MAX.checked_mul(I256::new(1)), Some(I256::MAX));
    /// assert_eq!(I256::MAX.checked_mul(I256::new(2)), None);
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
    /// `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!((I256::MIN + 1).checked_div(I256::new(-1)), Some(I256::MAX));
    /// assert_eq!(I256::MIN.checked_div(I256::new(-1)), None);
    /// assert_eq!(I256::new(1).checked_div(I256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            let mut result = MaybeUninit::uninit();
            intrinsics::signed::idiv3(&mut result, &self, &rhs);
            Some(unsafe { result.assume_init() })
        }
    }

    /// Checked Euclidean division. Computes `self.div_euclid(rhs)`,
    /// returning `None` if `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!((I256::MIN + 1).checked_div_euclid(I256::new(-1)), Some(I256::MAX));
    /// assert_eq!(I256::MIN.checked_div_euclid(I256::new(-1)), None);
    /// assert_eq!(I256::new(1).checked_div_euclid(I256::new(0)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            Some(self.div_euclid(rhs))
        }
    }

    /// Checked integer remainder. Computes `self % rhs`, returning `None` if
    /// `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).checked_rem(I256::new(2)), Some(I256::new(1)));
    /// assert_eq!(I256::new(5).checked_rem(I256::new(0)), None);
    /// assert_eq!(I256::MIN.checked_rem(I256::new(-1)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            let mut result = MaybeUninit::uninit();
            intrinsics::signed::irem3(&mut result, &self, &rhs);
            Some(unsafe { result.assume_init() })
        }
    }

    /// Checked Euclidean remainder. Computes `self.rem_euclid(rhs)`, returning
    /// `None` if `rhs == 0` or the division results in overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).checked_rem_euclid(I256::new(2)), Some(I256::new(1)));
    /// assert_eq!(I256::new(5).checked_rem_euclid(I256::new(0)), None);
    /// assert_eq!(I256::MIN.checked_rem_euclid(I256::new(-1)), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        if rhs == 0 || (self == Self::MIN && rhs == -1) {
            None
        } else {
            Some(self.rem_euclid(rhs))
        }
    }

    /// Checked negation. Computes `-self`, returning `None` if `self == MIN`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).checked_neg(), Some(I256::new(-5)));
    /// assert_eq!(I256::MIN.checked_neg(), None);
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

    /// Checked shift left. Computes `self << rhs`, returning `None` if `rhs`
    /// is larger than or equal to the number of bits in `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(0x1).checked_shl(4), Some(I256::new(0x10)));
    /// assert_eq!(I256::new(0x1).checked_shl(257), None);
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(0x10).checked_shr(4), Some(I256::new(0x1)));
    /// assert_eq!(I256::new(0x10).checked_shr(256), None);
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

    /// Checked absolute value. Computes `self.abs()`, returning `None` if
    /// `self == MIN`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(-5).checked_abs(), Some(I256::new(5)));
    /// assert_eq!(I256::MIN.checked_abs(), None);
    /// ```
    #[inline]
    pub fn checked_abs(self) -> Option<Self> {
        if self.is_negative() {
            self.checked_neg()
        } else {
            Some(self)
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(8).checked_pow(2), Some(I256::new(64)));
    /// assert_eq!(I256::MAX.checked_pow(2), None);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn checked_pow(self, mut exp: u32) -> Option<Self> {
        if exp == 0 {
            return Some(Self::ONE);
        }
        let mut base = self;
        let mut acc = Self::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.checked_mul(base)?;
            }
            exp /= 2;
            base = base.checked_mul(base)?;
        }
        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        acc.checked_mul(base)
    }

    /// Saturating integer addition. Computes `self + rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).saturating_add(I256::new(1)), 101);
    /// assert_eq!(I256::MAX.saturating_add(I256::new(100)), I256::MAX);
    /// assert_eq!(I256::MIN.saturating_add(I256::new(-1)), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        match self.checked_add(rhs) {
            Some(x) => x,
            None => {
                if rhs > 0 {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }
    }

    /// Saturating addition with an unsigned integer. Computes `self + rhs`,
    /// saturating at the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(1).saturating_add_unsigned(U256::new(2)), 3);
    /// assert_eq!(I256::MAX.saturating_add_unsigned(U256::new(100)), I256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn saturating_add_unsigned(self, rhs: U256) -> Self {
        // Overflow can only happen at the upper bound
        match self.checked_add_unsigned(rhs) {
            Some(x) => x,
            None => Self::MAX,
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).saturating_sub(I256::new(127)), -27);
    /// assert_eq!(I256::MIN.saturating_sub(I256::new(100)), I256::MIN);
    /// assert_eq!(I256::MAX.saturating_sub(I256::new(-1)), I256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        match self.checked_sub(rhs) {
            Some(x) => x,
            None => {
                if rhs > 0 {
                    Self::MIN
                } else {
                    Self::MAX
                }
            }
        }
    }

    /// Saturating subtraction with an unsigned integer. Computes `self - rhs`,
    /// saturating at the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(100).saturating_sub_unsigned(U256::new(127)), -27);
    /// assert_eq!(I256::MIN.saturating_sub_unsigned(U256::new(100)), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn saturating_sub_unsigned(self, rhs: U256) -> Self {
        // Overflow can only happen at the lower bound
        match self.checked_sub_unsigned(rhs) {
            Some(x) => x,
            None => Self::MIN,
        }
    }

    /// Saturating integer negation. Computes `-self`, returning `MAX` if
    /// `self == MIN` instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).saturating_neg(), -100);
    /// assert_eq!(I256::new(-100).saturating_neg(), 100);
    /// assert_eq!(I256::MIN.saturating_neg(), I256::MAX);
    /// assert_eq!(I256::MAX.saturating_neg(), I256::MIN + 1);
    /// ```
    #[inline(always)]
    pub fn saturating_neg(self) -> Self {
        I256::ZERO.saturating_sub(self)
    }

    /// Saturating absolute value. Computes `self.abs()`, returning `MAX` if
    /// `self == MIN` instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).saturating_abs(), 100);
    /// assert_eq!(I256::new(-100).saturating_abs(), 100);
    /// assert_eq!(I256::MIN.saturating_abs(), I256::MAX);
    /// assert_eq!((I256::MIN + 1).saturating_abs(), I256::MAX);
    /// ```
    #[inline]
    pub fn saturating_abs(self) -> Self {
        if self.is_negative() {
            self.saturating_neg()
        } else {
            self
        }
    }

    /// Saturating integer multiplication. Computes `self * rhs`, saturating at
    /// the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).saturating_mul(I256::new(12)), 120);
    /// assert_eq!(I256::MAX.saturating_mul(I256::new(10)), I256::MAX);
    /// assert_eq!(I256::MIN.saturating_mul(I256::new(10)), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        match self.checked_mul(rhs) {
            Some(x) => x,
            None => {
                if (self < 0) == (rhs < 0) {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).saturating_div(I256::new(2)), 2);
    /// assert_eq!(I256::MAX.saturating_div(I256::new(-1)), I256::MIN + 1);
    /// assert_eq!(I256::MIN.saturating_div(I256::new(-1)), I256::MAX);
    /// ```
    ///
    /// ```should_panic
    /// # use ethnum::I256;;
    /// let _ = I256::new(1).saturating_div(I256::new(0));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_div(self, rhs: Self) -> Self {
        match self.overflowing_div(rhs) {
            (result, false) => result,
            (_result, true) => Self::MAX, // MIN / -1 is the only possible saturating overflow
        }
    }

    /// Saturating integer exponentiation. Computes `self.pow(exp)`,
    /// saturating at the numeric bounds instead of overflowing.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(-4).saturating_pow(3), -64);
    /// assert_eq!(I256::MIN.saturating_pow(2), I256::MAX);
    /// assert_eq!(I256::MIN.saturating_pow(3), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn saturating_pow(self, exp: u32) -> Self {
        match self.checked_pow(exp) {
            Some(x) => x,
            None if self < 0 && exp % 2 == 1 => Self::MIN,
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).wrapping_add(I256::new(27)), 127);
    /// assert_eq!(I256::MAX.wrapping_add(I256::new(2)), I256::MIN + 1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::iadd3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) addition with an unsigned integer. Computes
    /// `self + rhs`, wrapping around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(100).wrapping_add_unsigned(U256::new(27)), 127);
    /// assert_eq!(I256::MAX.wrapping_add_unsigned(U256::new(2)), I256::MIN + 1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline(always)]
    pub fn wrapping_add_unsigned(self, rhs: U256) -> Self {
        self.wrapping_add(rhs.as_i256())
    }

    /// Wrapping (modular) subtraction. Computes `self - rhs`, wrapping around
    /// at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(0).wrapping_sub(I256::new(127)), -127);
    /// assert_eq!(I256::new(-2).wrapping_sub(I256::MAX), I256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::isub3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) subtraction with an unsigned integer. Computes
    /// `self - rhs`, wrapping around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(0).wrapping_sub_unsigned(U256::new(127)), -127);
    /// assert_eq!(I256::new(-2).wrapping_sub_unsigned(U256::MAX), -1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline(always)]
    pub fn wrapping_sub_unsigned(self, rhs: U256) -> Self {
        self.wrapping_sub(rhs.as_i256())
    }

    /// Wrapping (modular) multiplication. Computes `self * rhs`, wrapping
    /// around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).wrapping_mul(I256::new(12)), 120);
    /// assert_eq!(I256::MAX.wrapping_mul(I256::new(2)), -2);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::imul3(&mut result, &self, &rhs);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) division. Computes `self / rhs`, wrapping around at
    /// the boundary of the type.
    ///
    /// The only case where such wrapping can occur is when one divides
    /// `MIN / -1` on a signed type (where `MIN` is the negative minimal value
    /// for the type); this is equivalent to `-MIN`, a positive value that is
    /// too large to represent in the type. In such a case, this function
    /// returns `MIN` itself.
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
    /// assert_eq!(I256::new(100).wrapping_div(I256::new(10)), 10);
    /// assert_eq!(I256::MIN.wrapping_div(I256::new(-1)), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        self.overflowing_div(rhs).0
    }

    /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`,
    /// wrapping around at the boundary of the type.
    ///
    /// Wrapping will only occur in `MIN / -1` on a signed type (where `MIN` is
    /// the negative minimal value for the type). This is equivalent to `-MIN`,
    /// a positive value that is too large to represent in the type. In this
    /// case, this method returns `MIN` itself.
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
    /// assert_eq!(I256::new(100).wrapping_div_euclid(I256::new(10)), 10);
    /// assert_eq!(I256::MIN.wrapping_div_euclid(I256::new(-1)), I256::MIN);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_div_euclid(self, rhs: Self) -> Self {
        self.overflowing_div_euclid(rhs).0
    }

    /// Wrapping (modular) remainder. Computes `self % rhs`, wrapping around at
    /// the boundary of the type.
    ///
    /// Such wrap-around never actually occurs mathematically; implementation
    /// artifacts make `x % y` invalid for `MIN / -1` on a signed type (where
    /// MIN` is the negative minimal value). In such a case, this function
    /// returns `0`.
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
    /// assert_eq!(I256::new(100).wrapping_rem(I256::new(10)), 0);
    /// assert_eq!(I256::MIN.wrapping_rem(I256::new(-1)), 0);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_rem(self, rhs: Self) -> Self {
        self.overflowing_rem(rhs).0
    }

    /// Wrapping Euclidean remainder. Computes `self.rem_euclid(rhs)`, wrapping
    /// around at the boundary of the type.
    ///
    /// Wrapping will only occur in `MIN % -1` on a signed type (where `MIN` is
    /// the negative minimal value for the type). In this case, this method
    /// returns 0.
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
    /// assert_eq!(I256::new(100).wrapping_rem_euclid(I256::new(10)), 0);
    /// assert_eq!(I256::MIN.wrapping_rem_euclid(I256::new(-1)), 0);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_rem_euclid(self, rhs: Self) -> Self {
        self.overflowing_rem_euclid(rhs).0
    }

    /// Wrapping (modular) negation. Computes `-self`, wrapping around at the
    /// boundary of the type.
    ///
    /// The only case where such wrapping can occur is when one negates `MIN` on
    /// a signed type (where `MIN` is the negative minimal value for the type);
    /// this is a positive value that is too large to represent in the type. In
    /// such a case, this function returns `MIN` itself.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(100).wrapping_neg(), -100);
    /// assert_eq!(I256::MIN.wrapping_neg(), I256::MIN);
    /// ```
    #[inline(always)]
    pub fn wrapping_neg(self) -> Self {
        Self::ZERO.wrapping_sub(self)
    }

    /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping
    /// shift-left is restricted to the range of the type, rather than the bits
    /// shifted out of the LHS being returned to the other end. The primitive
    /// integer types all implement a [`rotate_left`](Self::rotate_left)
    /// function, which may be what you want instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(-1).wrapping_shl(7), -128);
    /// assert_eq!(I256::new(-1).wrapping_shl(128), I256::from_words(-1, 0));
    /// assert_eq!(I256::new(-1).wrapping_shl(256), -1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_shl(self, rhs: u32) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::ishl3(&mut result, &self, rhs & 0xff);
        unsafe { result.assume_init() }
    }

    /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-right; the RHS of a
    /// wrapping shift-right is restricted to the range of the type, rather than
    /// the bits shifted out of the LHS being returned to the other end. The
    /// primitive integer types all implement a
    /// [`rotate_right`](Self::rotate_right) function, which may be what you
    /// want instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(-128).wrapping_shr(7), -1);
    /// assert_eq!((-128i16).wrapping_shr(64), -128);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn wrapping_shr(self, rhs: u32) -> Self {
        let mut result = MaybeUninit::uninit();
        intrinsics::signed::isar3(&mut result, &self, rhs & 0xff);
        unsafe { result.assume_init() }
    }

    /// Wrapping (modular) absolute value. Computes `self.abs()`, wrapping
    /// around at the boundary of the type.
    ///
    /// The only case where such wrapping can occur is when one takes the
    /// absolute value of the negative minimal value for the type; this is a
    /// positive value that is too large to represent in the type. In such a
    /// case, this function returns `MIN` itself.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(100).wrapping_abs(), 100);
    /// assert_eq!(I256::new(-100).wrapping_abs(), 100);
    /// assert_eq!(I256::MIN.wrapping_abs(), I256::MIN);
    /// assert_eq!(
    ///     I256::MIN.wrapping_abs().as_u256(),
    ///     U256::from_words(
    ///         0x80000000000000000000000000000000,
    ///         0x00000000000000000000000000000000,
    ///     ),
    /// );
    /// ```
    #[allow(unused_attributes)]
    #[inline]
    pub fn wrapping_abs(self) -> Self {
        if self.is_negative() {
            self.wrapping_neg()
        } else {
            self
        }
    }

    /// Computes the absolute value of `self` without any wrapping
    /// or panicking.
    ///
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(100).unsigned_abs(), 100);
    /// assert_eq!(I256::new(-100).unsigned_abs(), 100);
    /// assert_eq!(
    ///     I256::MIN.unsigned_abs(),
    ///     U256::from_words(
    ///         0x80000000000000000000000000000000,
    ///         0x00000000000000000000000000000000,
    ///     ),
    /// );
    /// ```
    #[inline(always)]
    pub fn unsigned_abs(self) -> U256 {
        self.wrapping_abs().as_u256()
    }

    /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`,
    /// wrapping around at the boundary of the type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(3).wrapping_pow(4), 81);
    /// assert_eq!(3i8.wrapping_pow(5), -13);
    /// assert_eq!(3i8.wrapping_pow(6), -39);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn wrapping_pow(self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut base = self;
        let mut acc = Self::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.wrapping_mul(base);
            }
            exp /= 2;
            base = base.wrapping_mul(base);
        }

        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        acc.wrapping_mul(base)
    }

    /// Calculates `self` + `rhs`
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).overflowing_add(I256::new(2)), (I256::new(7), false));
    /// assert_eq!(I256::MAX.overflowing_add(I256::new(1)), (I256::MIN, true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::iaddc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Calculates `self` + `rhs` with an unsigned `rhs`
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
    /// assert_eq!(I256::new(1).overflowing_add_unsigned(U256::new(2)), (I256::new(3), false));
    /// assert_eq!((I256::MIN).overflowing_add_unsigned(U256::MAX), (I256::MAX, false));
    /// assert_eq!((I256::MAX - 2).overflowing_add_unsigned(U256::new(3)), (I256::MIN, true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn overflowing_add_unsigned(self, rhs: U256) -> (Self, bool) {
        let rhs = rhs.as_i256();
        let (res, overflowed) = self.overflowing_add(rhs);
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
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).overflowing_sub(I256::new(2)), (I256::new(3), false));
    /// assert_eq!(I256::MIN.overflowing_sub(I256::new(1)), (I256::MAX, true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::isubc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Calculates `self` - `rhs` with an unsigned `rhs`
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would
    /// have occurred then the wrapped value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(1).overflowing_sub_unsigned(U256::new(2)), (I256::new(-1), false));
    /// assert_eq!((I256::MAX).overflowing_sub_unsigned(U256::MAX), (I256::MIN, false));
    /// assert_eq!((I256::MIN + 2).overflowing_sub_unsigned(U256::new(3)), (I256::MAX, true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                    without modifying the original"]
    #[inline]
    pub fn overflowing_sub_unsigned(self, rhs: U256) -> (Self, bool) {
        let rhs = rhs.as_i256();
        let (res, overflowed) = self.overflowing_sub(rhs);
        (res, overflowed ^ (rhs < 0))
    }

    /// Computes the absolute difference between `self` and `other`.
    ///
    /// This function always returns the correct answer without overflow or
    /// panics by returning an unsigned integer.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::{I256, U256};
    /// assert_eq!(I256::new(100).abs_diff(I256::new(80)), 20);
    /// assert_eq!(I256::new(100).abs_diff(I256::new(110)), 10);
    /// assert_eq!(I256::new(-100).abs_diff(I256::new(80)), 180);
    /// assert_eq!(I256::new(-100).abs_diff(I256::new(-120)), 20);
    /// assert_eq!(I256::MIN.abs_diff(I256::MAX), U256::MAX);
    /// assert_eq!(I256::MAX.abs_diff(I256::MIN), U256::MAX);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn abs_diff(self, other: Self) -> U256 {
        if self < other {
            // Converting a non-negative x from signed to unsigned by using
            // `x as U` is left unchanged, but a negative x is converted
            // to value x + 2^N. Thus if `s` and `o` are binary variables
            // respectively indicating whether `self` and `other` are
            // negative, we are computing the mathematical value:
            //
            //    (other + o*2^N) - (self + s*2^N)    mod  2^N
            //    other - self + (o-s)*2^N            mod  2^N
            //    other - self                        mod  2^N
            //
            // Finally, taking the mod 2^N of the mathematical value of
            // `other - self` does not change it as it already is
            // in the range [0, 2^N).
            other.as_u256().wrapping_sub(self.as_u256())
        } else {
            self.as_u256().wrapping_sub(other.as_u256())
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
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(5).overflowing_mul(I256::new(2)), (I256::new(10), false));
    /// assert_eq!(I256::MAX.overflowing_mul(I256::new(2)), (I256::new(-2), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline(always)]
    pub fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let mut result = MaybeUninit::uninit();
        let overflow = intrinsics::signed::imulc(&mut result, &self, &rhs);
        (unsafe { result.assume_init() }, overflow)
    }

    /// Calculates the divisor when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would occur then self
    /// is returned.
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
    /// assert_eq!(I256::new(5).overflowing_div(I256::new(2)), (I256::new(2), false));
    /// assert_eq!(I256::MIN.overflowing_div(I256::new(-1)), (I256::MIN, true));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    pub fn overflowing_div(self, rhs: Self) -> (Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (self, true)
        } else {
            (self / rhs, false)
        }
    }

    /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
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
    /// assert_eq!(I256::new(5).overflowing_div_euclid(I256::new(2)), (I256::new(2), false));
    /// assert_eq!(I256::MIN.overflowing_div_euclid(I256::new(-1)), (I256::MIN, true));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    pub fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (self, true)
        } else {
            (self.div_euclid(rhs), false)
        }
    }

    /// Calculates the remainder when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the remainder after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. If an overflow
    /// would occur then 0 is returned.
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
    /// assert_eq!(I256::new(5).overflowing_rem(I256::new(2)), (I256::new(1), false));
    /// assert_eq!(I256::MIN.overflowing_rem(I256::new(-1)), (I256::new(0), true));
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    pub fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (Self::ZERO, true)
        } else {
            (self % rhs, false)
        }
    }

    /// Overflowing Euclidean remainder. Calculates `self.rem_euclid(rhs)`.
    ///
    /// Returns a tuple of the remainder after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. If an overflow
    /// would occur then 0 is returned.
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
    /// assert_eq!(I256::new(5).overflowing_rem_euclid(I256::new(2)), (I256::new(1), false));
    /// assert_eq!(I256::MIN.overflowing_rem_euclid(I256::new(-1)), (I256::new(0), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
        if self == Self::MIN && rhs == -1 {
            (Self::ZERO, true)
        } else {
            (self.rem_euclid(rhs), false)
        }
    }

    /// Negates self, overflowing if this is equal to the minimum value.
    ///
    /// Returns a tuple of the negated version of self along with a boolean
    /// indicating whether an overflow happened. If `self` is the minimum value
    /// (e.g., `i32::MIN` for values of type `i32`), then the minimum value will
    /// be returned again and `true` will be returned for an overflow happening.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(2).overflowing_neg(), (I256::new(-2), false));
    /// assert_eq!(I256::MIN.overflowing_neg(), (I256::MIN, true));
    /// ```
    #[inline]
    pub fn overflowing_neg(self) -> (Self, bool) {
        if self == Self::MIN {
            (Self::MIN, true)
        } else {
            (-self, false)
        }
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
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(1).overflowing_shl(4), (I256::new(0x10), false));
    /// assert_eq!(I256::new(1).overflowing_shl(260), (I256::new(0x10), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
        (self.wrapping_shl(rhs), (rhs > 255))
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
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(0x10).overflowing_shr(4), (I256::new(0x1), false));
    /// assert_eq!(I256::new(0x10).overflowing_shr(260), (I256::new(0x1), true));
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
        (self.wrapping_shr(rhs), (rhs > 255))
    }

    /// Computes the absolute value of `self`.
    ///
    /// Returns a tuple of the absolute version of self along with a boolean
    /// indicating whether an overflow happened. If self is the minimum value
    /// (e.g., I256::MIN for values of type I256), then the minimum value will
    /// be returned again and true will be returned for an overflow happening.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).overflowing_abs(), (I256::new(10), false));
    /// assert_eq!(I256::new(-10).overflowing_abs(), (I256::new(10), false));
    /// assert_eq!(I256::MIN.overflowing_abs(), (I256::MIN, true));
    /// ```
    #[inline]
    pub fn overflowing_abs(self) -> (Self, bool) {
        (self.wrapping_abs(), self == Self::MIN)
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
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(3).overflowing_pow(4), (I256::new(81), false));
    /// assert_eq!(
    ///     I256::new(10).overflowing_pow(77),
    ///     (
    ///         I256::from_words(
    ///             -46408779215366586471190473126206792002,
    ///             -113521875028918879454725857041952276480,
    ///         ),
    ///         true,
    ///     )
    /// );
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn overflowing_pow(self, mut exp: u32) -> (Self, bool) {
        if exp == 0 {
            return (Self::ONE, false);
        }
        let mut base = self;
        let mut acc = Self::ONE;
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

        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        r = acc.overflowing_mul(base);
        r.1 |= overflown;
        r
    }

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    ///
    /// assert_eq!(I256::new(2).pow(5), 32);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn pow(self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut base = self;
        let mut acc = Self::ONE;

        while exp > 1 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            exp /= 2;
            base = base * base;
        }

        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        acc * base
    }

    /// Calculates the quotient of Euclidean division of `self` by `rhs`.
    ///
    /// This computes the integer `q` such that `self = q * rhs + r`, with
    /// `r = self.rem_euclid(rhs)` and `0 <= r < abs(rhs)`.
    ///
    /// In other words, the result is `self / rhs` rounded to the integer `q`
    /// such that `self >= q * rhs`.
    /// If `self > 0`, this is equal to round towards zero (the default in
    /// Rust); if `self < 0`, this is equal to round towards +/- infinity.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0 or the division results in
    /// overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let a = I256::new(7);
    /// let b = I256::new(4);
    ///
    /// assert_eq!(a.div_euclid(b), 1); // 7 >= 4 * 1
    /// assert_eq!(a.div_euclid(-b), -1); // 7 >= -4 * -1
    /// assert_eq!((-a).div_euclid(b), -2); // -7 >= 4 * -2
    /// assert_eq!((-a).div_euclid(-b), 2); // -7 >= -4 * 2
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn div_euclid(self, rhs: Self) -> Self {
        let q = self / rhs;
        if self % rhs < 0 {
            return if rhs > 0 { q - 1 } else { q + 1 };
        }
        q
    }

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// This is done as if by the Euclidean division algorithm -- given
    /// `r = self.rem_euclid(rhs)`, `self = rhs * self.div_euclid(rhs) + r`, and
    /// `0 <= r < abs(rhs)`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0 or the division results in
    /// overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// let a = I256::new(7);
    /// let b = I256::new(4);
    ///
    /// assert_eq!(a.rem_euclid(b), 3);
    /// assert_eq!((-a).rem_euclid(b), 1);
    /// assert_eq!(a.rem_euclid(-b), 3);
    /// assert_eq!((-a).rem_euclid(-b), 1);
    /// ```
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        if r < 0 {
            if rhs < 0 {
                r - rhs
            } else {
                r + rhs
            }
        } else {
            r
        }
    }

    /// Computes the absolute value of `self`.
    ///
    /// # Overflow behavior
    ///
    /// The absolute value of
    /// `I256::MIN`
    /// cannot be represented as an
    /// `I256`,
    /// and attempting to calculate it will cause an overflow. This means
    /// that code in debug mode will trigger a panic on this case and
    /// optimized code will return
    /// `I256::MIN`
    /// without a panic.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).abs(), 10);
    /// assert_eq!(I256::new(-10).abs(), 10);
    /// ```
    #[allow(unused_attributes)]
    #[inline]
    pub fn abs(self) -> Self {
        if self.is_negative() {
            -self
        } else {
            self
        }
    }

    /// Returns a number representing sign of `self`.
    ///
    ///  - `0` if the number is zero
    ///  - `1` if the number is positive
    ///  - `-1` if the number is negative
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).signum(), 1);
    /// assert_eq!(I256::new(0).signum(), 0);
    /// assert_eq!(I256::new(-10).signum(), -1);
    /// ```
    #[inline(always)]
    pub const fn signum(self) -> Self {
        I256::new(self.signum128())
    }

    /// Returns a number representing sign of `self` as a 128 bit signed integer.
    ///
    ///  - `0` if the number is zero
    ///  - `1` if the number is positive
    ///  - `-1` if the number is negative
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert_eq!(I256::new(10).signum128(), 1i128);
    /// assert_eq!(I256::new(0).signum128(), 0i128);
    /// assert_eq!(I256::new(-10).signum128(), -1i128);
    /// ```
    #[inline]
    pub const fn signum128(self) -> i128 {
        let (hi, lo) = self.into_words();
        hi.signum() | (lo != 0) as i128
    }

    /// Returns `true` if `self` is positive and `false` if the number is zero
    /// or negative.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert!(I256::new(10).is_positive());
    /// assert!(!I256::new(-10).is_positive());
    /// ```
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.signum128() > 0
    }

    /// Returns `true` if `self` is negative and `false` if the number is zero
    /// or positive.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::I256;
    /// assert!(I256::new(-10).is_negative());
    /// assert!(!I256::new(10).is_negative());
    /// ```
    #[inline(always)]
    pub const fn is_negative(self) -> bool {
        let (a, _) = self.into_words();
        a < 0
    }

    /// Return the memory representation of this integer as a byte array in
    /// big-endian (network) byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let bytes = I256::from_words(
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
    pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()] {
        self.to_be().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in
    /// little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let bytes = I256::from_words(
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
    pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()] {
        self.to_le().to_ne_bytes()
    }

    /// Return the memory representation of this integer as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code
    /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate,
    /// instead.
    ///
    /// [`to_be_bytes`]: Self::to_be_bytes
    /// [`to_le_bytes`]: Self::to_le_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let bytes = I256::from_words(
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
    pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()] {
        // SAFETY: integers are plain old datatypes so we can always transmute them to
        // arrays of bytes
        unsafe { mem::transmute(self) }
    }

    /// Create an integer value from its representation as a byte array in
    /// big endian.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let value = I256::from_be_bytes([
    ///     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ///     0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    /// ]);
    /// assert_eq!(
    ///     value,
    ///     I256::from_words(
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
    /// # use ethnum::I256;
    /// fn read_be_i256(input: &mut &[u8]) -> I256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<I256>());
    ///     *input = rest;
    ///     I256::from_be_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        Self::from_be(Self::from_ne_bytes(bytes))
    }

    /// Create an integer value from its representation as a byte array in
    /// little endian.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let value = I256::from_le_bytes([
    ///     0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
    ///     0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    /// ]);
    /// assert_eq!(
    ///     value,
    ///     I256::from_words(
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
    /// # use ethnum::I256;
    /// fn read_le_i256(input: &mut &[u8]) -> I256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<I256>());
    ///     *input = rest;
    ///     I256::from_le_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        Self::from_le(Self::from_ne_bytes(bytes))
    }

    /// Create an integer value from its memory representation as a byte
    /// array in native endianness.
    ///
    /// As the target platform's native endianness is used, portable code
    /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
    /// appropriate instead.
    ///
    /// [`from_be_bytes`]: Self::from_be_bytes
    /// [`from_le_bytes`]: Self::from_le_bytes
    ///
    /// # Examples
    ///
    /// ```
    /// # use ethnum::I256;
    /// let value = I256::from_ne_bytes(if cfg!(target_endian = "big") {
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
    ///     I256::from_words(
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
    /// # use ethnum::I256;
    /// fn read_ne_i256(input: &mut &[u8]) -> I256 {
    ///     let (int_bytes, rest) = input.split_at(std::mem::size_of::<I256>());
    ///     *input = rest;
    ///     I256::from_ne_bytes(int_bytes.try_into().unwrap())
    /// }
    /// ```
    #[inline]
    pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
        // SAFETY: integers are plain old datatypes so we can always transmute to them
        unsafe { mem::transmute(bytes) }
    }
}
