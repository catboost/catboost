//! This crate implements 256-bit integer types.
//!
//! The implementation tries to follow as closely as possible to primitive
//! integer types, and should implement all the common methods and traits as the
//! primitive integer types.

#![deny(missing_docs)]
#![no_std]

#[cfg(test)]
extern crate alloc;

#[macro_use]
mod macros {
    #[macro_use]
    pub mod cmp;
    #[macro_use]
    pub mod fmt;
    #[macro_use]
    pub mod iter;
    #[macro_use]
    pub mod ops;
    #[macro_use]
    pub mod parse;
}

mod error;
mod fmt;
mod int;
pub mod intrinsics;
mod parse;
#[cfg(feature = "serde")]
pub mod serde;
mod uint;

/// Macro for 256-bit signed integer literal.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// # use ethnum::{int, I256};
/// assert_eq!(
///     int!(
///         "-57896044618658097711785492504343953926634992332820282019728792003956564819968"
///     ),
///     I256::MIN,
/// );
/// ```
///
/// Additionally, this macro accepts `0b` for binary, `0o` for octal, and `0x`
/// for hexadecimal literals. Using `_` for spacing is also permitted.
///
/// ```
/// # use ethnum::{int, I256};
/// assert_eq!(
///     int!(
///         "0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff
///            ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
///     ),
///     I256::MAX,
/// );
/// assert_eq!(int!("0b101010"), 42);
/// assert_eq!(int!("-0o52"), -42);
/// ```
#[macro_export]
macro_rules! int {
    ($integer:literal) => {{
        const VALUE: $crate::I256 = $crate::I256::const_from_str_prefixed($integer);
        VALUE
    }};
}

/// Macro for 256-bit unsigned integer literal.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// # use ethnum::{uint, U256};
/// assert_eq!(
///     uint!(
///         "115792089237316195423570985008687907852837564279074904382605163141518161494337"
///     ),
///     U256::from_words(
///         0xfffffffffffffffffffffffffffffffe,
///         0xbaaedce6af48a03bbfd25e8cd0364141,
///     ),
/// );
/// ```
///
/// Additionally, this macro accepts `0b` for binary, `0o` for octal, and `0x`
/// for hexadecimal literals. Using `_` for spacing is also permitted.
///
/// ```
/// # use ethnum::{uint, U256};
/// assert_eq!(
///     uint!(
///         "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff
///            ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"
///     ),
///     U256::MAX,
/// );
/// assert_eq!(uint!("0b101010"), 42);
/// assert_eq!(uint!("0o52"), 42);
/// ```
#[macro_export]
macro_rules! uint {
    ($integer:literal) => {{
        const VALUE: $crate::U256 = $crate::U256::const_from_str_prefixed($integer);
        VALUE
    }};
}

/// Convenience re-export of 256-integer types and as- conversion traits.
pub mod prelude {
    pub use crate::{AsI256, AsU256, I256, U256};
}

pub use crate::{
    int::{AsI256, I256},
    uint::{AsU256, U256},
};

/// A 256-bit signed integer type.
#[allow(non_camel_case_types)]
pub type i256 = I256;

/// A 256-bit unsigned integer type.
#[allow(non_camel_case_types)]
pub type u256 = U256;
