//! # Fast `&[u8]` to integer parser
//!
//! SIMD (fast) parsing is supported on x86_64 (SSE4.1, AVX2) and on Arm64 (aarch64, Neon),
//! but this library works even if you don't have a SIMD supported cpu (and it will be still faster than str::parse).
//!
//! Supports negative values and validates the input.
//!
//! Supported output types: u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize.
//!
//! Has good test coverage, and can be considered safe.
//!
//! To enable SIMD it needs the `target-feature` or `target-cpu` flags set, or it will fallback to non-SIMD functions.
//! You can copy the `./.cargo/config.toml` to your project, or use one of the following environment variables:
//!
//! -   `RUSTFLAGS="-C target-feature=+sse2,+sse3,+sse4.1,+ssse3,+avx,+avx2"` for x86_64
//!
//! -   `RUSTFLAGS="-C target-feature=+neon"` for Arm64
//!
//! -   `RUSTFLAGS="-C target-cpu=native"` will optimize for your current cpu
//!
//! If you have `&str` then use `.as_bytes()`
//!
//! Supports `no_std` with `--no-default-features`
//!
//! # Examples
//!
//! ```
//! let val: u64 = atoi_simd::parse(b"1234").unwrap();
//! assert_eq!(val, 1234_u64);
//!
//! assert_eq!(atoi_simd::parse::<i64>(b"-2345"), Ok(-2345_i64));
//!
//! assert_eq!(atoi_simd::parse_any::<u64>(b"123something_else"), Ok((123_u64, 3)));
//!
//! // a drop-in replacement for `str::parse`
//! assert_eq!(atoi_simd::parse_skipped::<u64>(b"+000000000000000000001234"), Ok(1234_u64));
//! ```
#![allow(clippy::comparison_chain)]
#![cfg_attr(not(feature = "std"), no_std)]
// #![feature(stdsimd)]

mod error;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_feature = "sse2",
        target_feature = "sse3",
        target_feature = "sse4.1",
        target_feature = "ssse3",
        target_feature = "avx",
        target_feature = "avx2"
    ),
)))]
mod fallback;
mod linker;
mod short;
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(
        target_feature = "sse2",
        target_feature = "sse",
        target_feature = "sse4.1",
        target_feature = "ssse3"
    ),
))]
mod simd;

pub use crate::{
    error::AtoiSimdError,
    linker::{Parse, ParseNeg},
};

/// Parses a slice of digits, and checks the first '-' char for signed integers.
#[inline]
pub fn parse<T: Parse>(s: &[u8]) -> Result<T, AtoiSimdError> {
    T::atoi_simd_parse(s)
}

/// Parses a positive integer.
#[inline]
pub fn parse_pos<T: Parse>(s: &[u8]) -> Result<T, AtoiSimdError> {
    T::atoi_simd_parse_pos(s)
}

/// Parses a negative integer. Slice must not contain '-' sign.
#[inline]
pub fn parse_neg<T: ParseNeg>(s: &[u8]) -> Result<T, AtoiSimdError> {
    T::atoi_simd_parse_neg(s)
}

/// Parses a slice of digits until it reaches invalid character, and checks the first '-' char for signed integers.
/// Returns the parsed value and the parsed size of the slice.
#[inline]
pub fn parse_any<T: Parse>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    T::atoi_simd_parse_any(s)
}

/// Parses a positive integer until it reaches an invalid character.
/// Returns the parsed value and the parsed size of the slice.
#[inline]
pub fn parse_any_pos<T: Parse>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    T::atoi_simd_parse_any_pos(s)
}

/// Parses a negative integer until it reaches an invalid character. Slice must not contain '-' sign.
/// Returns the parsed value and the parsed size of the slice.
#[inline]
pub fn parse_any_neg<T: ParseNeg>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    T::atoi_simd_parse_any_neg(s)
}

/// Parses a slice of digits. Has been made to be used as a drop-in replacement for `str::parse`.
/// Checks the first '-' char for signed integers.
/// Skips the '+' char and extra zeroes at the beginning.
/// It's slower than `parse()`.
#[inline]
pub fn parse_skipped<T: Parse>(s: &[u8]) -> Result<T, AtoiSimdError> {
    T::atoi_simd_parse_skipped(s)
}

#[deprecated(since = "0.16.0", note = "Use `parse_any` instead")]
#[inline]
pub fn parse_until_invalid<T: Parse>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    parse_any(s)
}

#[deprecated(since = "0.16.0", note = "Use `parse_any_pos` instead")]
#[inline]
pub fn parse_until_invalid_pos<T: Parse>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    parse_any_pos(s)
}

#[deprecated(since = "0.16.0", note = "Use `parse_any_neg` instead")]
#[inline]
pub fn parse_until_invalid_neg<T: ParseNeg>(s: &[u8]) -> Result<(T, usize), AtoiSimdError> {
    parse_any_neg(s)
}
