//! Serde serialization implementation for 256-bit integer types.
//!
//! This implementation is very JSON-centric in that it serializes the integer
//! types as `QUANTITIES` as specified in the Ethereum RPC. That is, integers
//! are encoded as `"0x"` prefixed strings without extrenuous leading `0`s. For
//! negative signed integers, the string is prefixed with a `"-"` sign.
//!
//! Note that this module contains alternative serialization schemes that can
//! be used with `#[serde(with = "...")]`.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```text
//! #[derive(Deserialize, Serialize)]
//! struct Example {
//!     a: U256, // "0x2a"
//!     #[serde(with = "ethnum::serde::decimal")]
//!     b: I256, // "-42"
//!     #[serde(with = "ethnum::serde::prefixed")]
//!     c: U256, // "0x2a" or "42"
//!     #[serde(with = "ethnum::serde::permissive")]
//!     d: I256, // "-0x2a" or "-42" or -42
//!     #[serde(with = "ethnum::serde::bytes::be")]
//!     e: U256, // [0x2a, 0x00, ..., 0x00]
//!     #[serde(with = "ethnum::serde::bytes::le")]
//!     f: I256, // [0xd6, 0xff, ..., 0xff]
//!     #[serde(with = "ethnum::serde::compressed_bytes::be")]
//!     g: U256, // [0x2a]
//!     #[serde(with = "ethnum::serde::compressed_bytes::le")]
//!     h: I256, // [0xd6]
//! }
//! ```

use crate::{int::I256, uint::U256};
use core::{
    fmt::{self, Display, Formatter, Write},
    mem::MaybeUninit,
    ptr, slice, str,
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};

impl Serialize for I256 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut f = FormatBuffer::hex();
        write!(f, "{self:-#x}").expect("unexpected formatting failure");
        serializer.serialize_str(f.as_str())
    }
}

impl Serialize for U256 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut f = FormatBuffer::hex();
        write!(f, "{self:#x}").expect("unexpected formatting failure");
        serializer.serialize_str(f.as_str())
    }
}

impl<'de> Deserialize<'de> for I256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FormatVisitor(Self::from_str_hex))
    }
}

impl<'de> Deserialize<'de> for U256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FormatVisitor(Self::from_str_hex))
    }
}

/// Module for use with `#[serde(with = "ethnum::serde::decimal")]` to specify
/// decimal string serialization for 256-bit integer types.
pub mod decimal {
    use super::*;
    use core::num::ParseIntError;

    #[doc(hidden)]
    pub trait Decimal: Sized {
        fn from_str_decimal(src: &str) -> Result<Self, ParseIntError>;
        fn write_decimal(&self, f: &mut impl Write);
    }

    impl Decimal for I256 {
        fn from_str_decimal(src: &str) -> Result<Self, ParseIntError> {
            Self::from_str_radix(src, 10)
        }
        fn write_decimal(&self, f: &mut impl Write) {
            write!(f, "{self}").expect("unexpected formatting error")
        }
    }

    impl Decimal for U256 {
        fn from_str_decimal(src: &str) -> Result<Self, ParseIntError> {
            Self::from_str_radix(src, 10)
        }
        fn write_decimal(&self, f: &mut impl Write) {
            write!(f, "{self}").expect("unexpected formatting error")
        }
    }

    #[doc(hidden)]
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Decimal,
        S: Serializer,
    {
        let mut f = FormatBuffer::decimal();
        value.write_decimal(&mut f);
        serializer.serialize_str(f.as_str())
    }

    #[doc(hidden)]
    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: Decimal,
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FormatVisitor(T::from_str_decimal))
    }
}

/// Module for use with `#[serde(with = "ethnum::serde::prefixed")]` to specify
/// prefixed string serialization for 256-bit integer types.
///
/// This allows serialization to look for an optional `0x` prefix to determine
/// if it is a hexadecimal string or decimal string.
pub mod prefixed {
    use super::*;
    use core::num::ParseIntError;

    #[doc(hidden)]
    pub trait Prefixed: Serialize + Sized {
        fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError>;
    }

    impl Prefixed for I256 {
        fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError> {
            Self::from_str_prefixed(src)
        }
    }

    impl Prefixed for U256 {
        fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError> {
            Self::from_str_prefixed(src)
        }
    }

    #[doc(hidden)]
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Prefixed,
        S: Serializer,
    {
        value.serialize(serializer)
    }

    #[doc(hidden)]
    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: Prefixed,
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FormatVisitor(T::from_str_prefixed))
    }
}

/// Module for use with `#[serde(with = "ethnum::serde::permissive")]` to
/// specify extremely permissive serialization for 256-bit integer types.
///
/// This allows serialization to also accept standard numerical types as values
/// in addition to prefixed strings.
pub mod permissive {
    use super::{prefixed::Prefixed, FormatVisitor};
    use crate::{AsI256 as _, I256, U256};
    use core::fmt::{self, Formatter};
    use core::marker::PhantomData;
    use serde::{
        de::{self, Deserializer, Visitor},
        Serializer,
    };

    #[doc(hidden)]
    pub trait Permissive: Prefixed {
        fn cast(value: I256) -> Self;
    }

    impl Permissive for I256 {
        fn cast(value: I256) -> Self {
            value
        }
    }

    impl Permissive for U256 {
        fn cast(value: I256) -> Self {
            value.as_u256()
        }
    }

    #[doc(hidden)]
    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Permissive,
        S: Serializer,
    {
        value.serialize(serializer)
    }

    struct PermissiveVisitor<T>(PhantomData<T>);

    impl<T> Visitor<'_> for PermissiveVisitor<T>
    where
        T: Permissive,
    {
        type Value = T;

        fn expecting(&self, f: &mut Formatter) -> fmt::Result {
            f.write_str("number, decimal string or '0x-' prefixed hexadecimal string")
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(T::cast(v.as_i256()))
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(T::cast(v.as_i256()))
        }

        fn visit_i128<E>(self, v: i128) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(T::cast(v.as_i256()))
        }

        fn visit_u128<E>(self, v: u128) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(T::cast(v.as_i256()))
        }

        fn visit_f32<E>(self, v: f32) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            const N: f32 = (1_u64 << 24) as _;
            if !(-N..N).contains(&v) {
                return Err(de::Error::custom(
                    "invalid conversion from single precision floating point \
                     number outside of valid integer range (-2^24, 2^24)",
                ));
            }

            self.visit_f64(v as _)
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            const N: f64 = (1_u64 << 53) as _;
            if !(-N..N).contains(&v) {
                return Err(de::Error::custom(
                    "invalid conversion from double precision floating point \
                     number outside of valid integer range (-2^53, 2^53)",
                ));
            }

            // SOUNDNESS: `#[no_std]` does not have `f64::fract`, so work around
            // it by casting to and from an integer type. This is sound because
            // we already verified that the `f64` is within a "safe" range.
            let i = v as i64;
            if i as f64 != v {
                return Err(de::Error::custom(
                    "invalid conversion from floating point number \
                     with fractional part to 256-bit integer",
                ));
            }

            Ok(T::cast(i.as_i256()))
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            FormatVisitor(T::from_str_prefixed).visit_str(v)
        }
    }

    #[doc(hidden)]
    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: Permissive,
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(PermissiveVisitor(PhantomData))
    }
}

/// Serde byte serialization for 256-bit integer types.
pub mod bytes {
    macro_rules! endianness {
        ($name:literal; $to:ident, $from:ident) => {
            use crate::{I256, U256};
            use core::{
                fmt::{self, Formatter},
                marker::PhantomData,
                mem::{self, MaybeUninit},
            };
            use serde::{
                de::{self, Deserializer, Visitor},
                Serializer,
            };

            #[doc(hidden)]
            pub trait Bytes: Sized + Copy {
                fn to_bytes(self) -> [u8; 32];
                fn from_bytes(bytes: [u8; 32]) -> Self;
            }

            impl Bytes for I256 {
                fn to_bytes(self) -> [u8; 32] {
                    self.$to()
                }

                fn from_bytes(bytes: [u8; 32]) -> Self {
                    I256::$from(bytes)
                }
            }

            impl Bytes for U256 {
                fn to_bytes(self) -> [u8; 32] {
                    self.$to()
                }

                fn from_bytes(bytes: [u8; 32]) -> Self {
                    U256::$from(bytes)
                }
            }

            #[doc(hidden)]
            pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
            where
                T: Bytes,
                S: Serializer,
            {
                let bytes = value.to_bytes();
                serializer.serialize_bytes(&bytes)
            }

            struct BytesVisitor<T>(PhantomData<T>);

            impl<'de, T> Visitor<'de> for BytesVisitor<T>
            where
                T: Bytes,
            {
                type Value = T;

                fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                    f.write_str(concat!("32 bytes in ", $name, " endian"))
                }

                fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                where
                    E: de::Error,
                {
                    let bytes = v
                        .try_into()
                        .map_err(|_| E::invalid_length(v.len(), &self))?;

                    Ok(T::from_bytes(bytes))
                }

                fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
                where
                    S: de::SeqAccess<'de>,
                {
                    match seq.size_hint() {
                        Some(len) if len != 32 => {
                            return Err(de::Error::invalid_length(len, &self))
                        }
                        _ => {}
                    }

                    let mut bytes = [MaybeUninit::<u8>::uninit(); 32];
                    for i in 0..32 {
                        bytes[i].write(
                            seq.next_element()?
                                .ok_or(de::Error::invalid_length(i, &self))?,
                        );
                    }
                    if seq.next_element::<u8>()?.is_some() {
                        return Err(de::Error::invalid_length(33, &self));
                    }

                    // SAFETY: all bytes have been initialized in for loop.
                    let bytes = unsafe { mem::transmute::<[MaybeUninit<u8>; 32], [u8; 32]>(bytes) };

                    Ok(T::from_bytes(bytes))
                }
            }

            #[doc(hidden)]
            pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
            where
                T: Bytes,
                D: Deserializer<'de>,
            {
                deserializer.deserialize_bytes(BytesVisitor(PhantomData))
            }
        };
    }

    /// Module for use with `#[serde(with = "ethnum::serde::bytes::le")]` to
    /// specify little endian byte serialization for 256-bit integer types.
    pub mod le {
        endianness!("little"; to_le_bytes, from_le_bytes);
    }

    /// Module for use with `#[serde(with = "ethnum::serde::bytes::be")]` to
    /// specify big endian byte serialization for 256-bit integer types.
    pub mod be {
        endianness!("big"; to_be_bytes, from_be_bytes);
    }

    /// Module for use with `#[serde(with = "ethnum::serde::bytes::ne")]` to
    /// specify native endian byte serialization for 256-bit integer types.
    pub mod ne {
        #[cfg(target_endian = "little")]
        #[doc(hidden)]
        pub use super::le::{deserialize, serialize};

        #[cfg(target_endian = "big")]
        #[doc(hidden)]
        pub use super::be::{deserialize, serialize};
    }
}

/// Serde compressed byte serialization for 256-bit integer types.
pub mod compressed_bytes {
    use crate::{I256, U256};

    #[doc(hidden)]
    pub trait CompressedBytes {
        fn leading_bits(&self) -> u32;
        fn extend(msb: u8) -> u8;
    }

    impl CompressedBytes for I256 {
        fn leading_bits(&self) -> u32 {
            match self.is_negative() {
                true => self.leading_ones() - 1,
                false => self.leading_zeros(),
            }
        }

        fn extend(msb: u8) -> u8 {
            ((msb as i8) >> 7) as _
        }
    }

    impl CompressedBytes for U256 {
        fn leading_bits(&self) -> u32 {
            self.leading_zeros()
        }

        fn extend(_: u8) -> u8 {
            0
        }
    }

    macro_rules! endianness {
        ($name:literal; $parent:ident, |$tb:ident| $to:block, |$fb:ident| $from:block) => {
            use super::CompressedBytes;
            use crate::serde::bytes::$parent::Bytes;
            use core::{
                fmt::{self, Formatter},
                marker::PhantomData,
                mem::MaybeUninit,
            };
            use serde::{
                de::{self, Deserializer, Visitor},
                Serializer,
            };

            #[doc(hidden)]
            pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
            where
                T: Bytes + CompressedBytes,
                S: Serializer,
            {
                let bytes = value.to_bytes();
                let $tb = (value.leading_bits() as usize) / 8;
                let index = { $to };
                serializer.serialize_bytes(&bytes[index])
            }

            struct CompressedBytesVisitor<T>(PhantomData<T>);

            impl<'de, T> Visitor<'de> for CompressedBytesVisitor<T>
            where
                T: Bytes + CompressedBytes,
            {
                type Value = T;

                fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                    f.write_str(concat!("bytes in ", $name, " endian"))
                }

                fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                where
                    E: de::Error,
                {
                    if v.len() > 32 {
                        return Err(E::invalid_length(v.len(), &self));
                    }

                    let extend = T::extend(v.last().copied().unwrap_or_default());
                    let mut bytes = [extend; 32];
                    let $fb = v.len();
                    let index = { $from };
                    bytes[index].copy_from_slice(v);

                    Ok(T::from_bytes(bytes))
                }

                fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
                where
                    S: de::SeqAccess<'de>,
                {
                    match seq.size_hint() {
                        Some(len) if len > 32 => return Err(de::Error::invalid_length(len, &self)),
                        _ => {}
                    }

                    let mut bytes = [MaybeUninit::<u8>::uninit(); 32];
                    let mut i = 0;
                    while i < 32 {
                        let b = match seq.next_element()? {
                            Some(b) => b,
                            None => break,
                        };
                        bytes[i].write(b);
                        i += 1;
                    }
                    if i == 32 && seq.next_element::<u8>()?.is_some() {
                        return Err(de::Error::invalid_length(33, &self));
                    }

                    // SAFETY: bytes up to `i` have been initialized in while
                    // loop.
                    let bytes = unsafe { &*(&bytes[..i] as *const _ as *const _) };

                    self.visit_bytes(bytes)
                }
            }

            #[doc(hidden)]
            pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
            where
                T: Bytes + CompressedBytes,
                D: Deserializer<'de>,
            {
                deserializer.deserialize_bytes(CompressedBytesVisitor(PhantomData))
            }
        };
    }

    /// Module for `#[serde(with = "ethnum::serde::compressed_bytes::le")]`
    /// to specify compressed little endian byte serialization for 256-bit
    /// integer types. This will serialize integer types with as few bytes as
    /// possible.
    pub mod le {
        endianness!("little"; le, |l| { ..32 - l }, |l| { ..l });
    }

    /// Module for `#[serde(with = "ethnum::serde::compressed_bytes::be")]`
    /// to specify compressed big endian byte serialization for 256-bit
    /// integer types. This will serialize integer types with as few bytes as
    /// possible.
    pub mod be {
        endianness!("big"; be, |l| { l.. }, |l| { 32 - l.. });
    }

    /// Module for `#[serde(with = "ethnum::serde::compressed_bytes::ne")]`
    /// to specify compressed native endian byte serialization for 256-bit
    /// integer types. This will serialize integer types with as few bytes as
    /// possible.
    pub mod ne {
        #[cfg(target_endian = "little")]
        #[doc(hidden)]
        pub use super::le::{deserialize, serialize};

        #[cfg(target_endian = "big")]
        #[doc(hidden)]
        pub use super::be::{deserialize, serialize};
    }
}

/// Internal visitor struct implementation to facilitate implementing different
/// serialization formats.
struct FormatVisitor<F>(F);

impl<T, E, F> Visitor<'_> for FormatVisitor<F>
where
    E: Display,
    F: FnOnce(&str) -> Result<T, E>,
{
    type Value = T;

    fn expecting(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("a formatted 256-bit integer")
    }

    fn visit_str<E_>(self, v: &str) -> Result<Self::Value, E_>
    where
        E_: de::Error,
    {
        self.0(v).map_err(de::Error::custom)
    }

    fn visit_bytes<E_>(self, v: &[u8]) -> Result<Self::Value, E_>
    where
        E_: de::Error,
    {
        let string = str::from_utf8(v)
            .map_err(|_| de::Error::invalid_value(de::Unexpected::Bytes(v), &self))?;
        self.visit_str(string)
    }
}

/// A stack-allocated buffer that can be used for writing formatted strings.
///
/// This allows us to leverage existing `fmt` implementations on integer types
/// without requiring heap allocations (i.e. writing to a `String` buffer).
struct FormatBuffer<const N: usize> {
    offset: usize,
    buffer: [MaybeUninit<u8>; N],
}

impl<const N: usize> FormatBuffer<N> {
    /// Creates a new formatting buffer.
    fn new() -> Self {
        Self {
            offset: 0,
            buffer: [MaybeUninit::uninit(); N],
        }
    }

    /// Returns a `str` to the currently written data.
    fn as_str(&self) -> &str {
        // SAFETY: We only ever write valid UTF-8 strings to the buffer, so the
        // resulting string will always be valid.
        unsafe {
            let buffer = slice::from_raw_parts(self.buffer[0].as_ptr(), self.offset);
            str::from_utf8_unchecked(buffer)
        }
    }
}

impl FormatBuffer<78> {
    /// Allocates a formatting buffer large enough to hold any possible decimal
    /// encoded 256-bit value.
    fn decimal() -> Self {
        Self::new()
    }
}

impl FormatBuffer<67> {
    /// Allocates a formatting buffer large enough to hold any possible
    /// hexadecimal encoded 256-bit value.
    fn hex() -> Self {
        Self::new()
    }
}

impl<const N: usize> Write for FormatBuffer<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.offset.checked_add(s.len()).ok_or(fmt::Error)?;

        // Make sure there is enough space in the buffer.
        if end > N {
            return Err(fmt::Error);
        }

        // SAFETY: We checked that there is enough space in the buffer to fit
        // the string `s` starting from `offset`, and the pointers cannot be
        // overlapping because of Rust ownership semantics (i.e. `s` cannot
        // overlap with `buffer` because we have a mutable reference to `self`
        // and by extension `buffer`).
        unsafe {
            let buffer = self.buffer[0].as_mut_ptr().add(self.offset);
            ptr::copy_nonoverlapping(s.as_ptr(), buffer, s.len());
        }
        self.offset = end;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{
        boxed::Box,
        fmt::{Display, LowerHex},
        format,
        string::String,
        vec,
        vec::Vec,
    };
    use serde::{
        de::{value, IntoDeserializer},
        ser::Impossible,
    };

    #[test]
    fn serialize_integers() {
        macro_rules! ser {
            ($method:expr, $value:expr) => {{
                let value = $value;
                ($method)(&value, StringSerializer).unwrap()
            }};
        }

        macro_rules! bin_ser {
            ($method:expr, $value:expr) => {{
                let value = $value;
                ($method)(&value, BytesSerializer).unwrap()
            }};
        }

        assert_eq!(
            ser!(I256::serialize, I256::MIN),
            "-0x8000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(ser!(I256::serialize, I256::new(-1)), "-0x1");
        assert_eq!(ser!(I256::serialize, I256::new(0)), "0x0");
        assert_eq!(ser!(I256::serialize, I256::new(42)), "0x2a");

        assert_eq!(ser!(U256::serialize, U256::new(0)), "0x0");
        assert_eq!(ser!(U256::serialize, U256::new(4919)), "0x1337");
        assert_eq!(
            ser!(U256::serialize, U256::MAX),
            "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        );

        assert_eq!(
            ser!(decimal::serialize, I256::MIN),
            "-57896044618658097711785492504343953926634992332820282019728792003956564819968",
        );
        assert_eq!(ser!(decimal::serialize, I256::new(-1)), "-1");
        assert_eq!(ser!(decimal::serialize, I256::new(0)), "0");
        assert_eq!(ser!(decimal::serialize, I256::new(42)), "42");

        assert_eq!(ser!(decimal::serialize, U256::new(0)), "0");
        assert_eq!(ser!(decimal::serialize, U256::new(4919)), "4919");
        assert_eq!(
            ser!(decimal::serialize, U256::MAX),
            "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        );

        assert_eq!(ser!(prefixed::serialize, I256::new(42)), "0x2a");
        assert_eq!(ser!(permissive::serialize, I256::new(42)), "0x2a");

        assert_eq!(bin_ser!(bytes::le::serialize, U256::ZERO), vec![0x00; 32]);
        assert_eq!(bin_ser!(bytes::le::serialize, U256::MAX), vec![0xff; 32]);
        assert_eq!(bin_ser!(bytes::le::serialize, U256::new(0x4215)), {
            let mut v = vec![0x15, 0x42];
            v.resize(32, 0x00);
            v
        });

        assert_eq!(
            bin_ser!(bytes::le::serialize, I256::new(-1)),
            vec![0xff; 32]
        );
        assert_eq!(bin_ser!(bytes::le::serialize, I256::new(-424242)), {
            let mut v = vec![0xce, 0x86, 0xf9];
            v.resize(32, 0xff);
            v
        });

        assert_eq!(bin_ser!(bytes::be::serialize, U256::ZERO), vec![0x00; 32]);
        assert_eq!(bin_ser!(bytes::be::serialize, U256::MAX), vec![0xff; 32]);
        assert_eq!(bin_ser!(bytes::be::serialize, U256::new(0x4215)), {
            let mut v = vec![0x00; 32];
            v[30..].copy_from_slice(&[0x42, 0x15]);
            v
        });

        assert_eq!(
            bin_ser!(bytes::be::serialize, I256::new(-1)),
            vec![0xff; 32]
        );
        assert_eq!(bin_ser!(bytes::be::serialize, I256::new(-424242)), {
            let mut v = vec![0xff; 32];
            v[29..].copy_from_slice(&[0xf9, 0x86, 0xce]);
            v
        });

        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, U256::ZERO),
            vec![]
        );
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, U256::MAX),
            vec![0xff; 32],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, U256::new(0x4215)),
            vec![0x15, 0x42],
        );

        assert_eq!(bin_ser!(compressed_bytes::le::serialize, I256::MIN), {
            let mut v = vec![0; 32];
            v[31] = 0x80;
            v
        });
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, I256::ZERO),
            vec![]
        );
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, I256::new(-1)),
            vec![0xff],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, I256::new(-0x8000)),
            vec![0x00, 0x80],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::le::serialize, I256::new(-424242)),
            vec![0xce, 0x86, 0xf9],
        );

        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, U256::ZERO),
            vec![]
        );
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, U256::MAX),
            vec![0xff; 32],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, U256::new(0x4215)),
            vec![0x42, 0x15],
        );

        assert_eq!(bin_ser!(compressed_bytes::be::serialize, I256::MIN), {
            let mut v = vec![0; 32];
            v[0] = 0x80;
            v
        });
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, I256::ZERO),
            vec![]
        );
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, I256::new(-1)),
            vec![0xff],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, I256::new(-0x8000)),
            vec![0x80, 0x00],
        );
        assert_eq!(
            bin_ser!(compressed_bytes::be::serialize, I256::new(-424242)),
            vec![0xf9, 0x86, 0xce],
        );
    }

    #[test]
    fn deserialize_integers() {
        macro_rules! de {
            ($method:expr, $src:expr) => {{
                let deserializer = IntoDeserializer::<value::Error>::into_deserializer($src);
                ($method)(deserializer).unwrap()
            }};
            (err; $method:expr, $src:expr) => {{
                let deserializer = IntoDeserializer::<value::Error>::into_deserializer($src);
                ($method)(deserializer).is_err()
            }};
        }

        macro_rules! assert_de_bytes {
            ($method:expr, $src:expr; eq: $exp:expr) => {{
                let src = $src;
                let exp = $exp;

                assert_eq!(de!($method, src.as_slice()), exp);

                let seq =
                    value::SeqDeserializer::<_, value::Error>::new(src.into_iter());
                assert_eq!(($method)(seq).unwrap(), exp);
            }};
            ($method:expr, $src:expr; err) => {{
                let src = $src;
                assert!(de!(err; $method, src.as_slice()));
                let seq =
                    value::SeqDeserializer::<_, value::Error>::new(src.into_iter());
                assert!(($method)(seq).is_err());
            }};
        }

        assert_eq!(
            de!(
                I256::deserialize,
                "-0x8000000000000000000000000000000000000000000000000000000000000000"
            ),
            I256::MIN
        );
        assert_eq!(de!(I256::deserialize, "-0x1337"), I256::new(-4919));
        assert_eq!(de!(I256::deserialize, "0x0"), I256::new(0));
        assert_eq!(de!(I256::deserialize, "0x2a"), I256::new(42));
        assert_eq!(de!(I256::deserialize, "0x2A"), I256::new(42));

        assert_eq!(de!(U256::deserialize, "0x0"), U256::new(0));
        assert_eq!(de!(U256::deserialize, "0x2a"), U256::new(42));
        assert_eq!(de!(U256::deserialize, "0x2A"), U256::new(42));
        assert_eq!(
            de!(
                U256::deserialize,
                "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            ),
            U256::MAX
        );

        assert_eq!(
            de!(
                decimal::deserialize::<I256, _>,
                "-57896044618658097711785492504343953926634992332820282019728792003956564819968"
            ),
            I256::MIN
        );
        assert_eq!(de!(decimal::deserialize::<I256, _>, "-1"), I256::new(-1));
        assert_eq!(de!(decimal::deserialize::<I256, _>, "0"), I256::new(0));
        assert_eq!(de!(decimal::deserialize::<I256, _>, "42"), I256::new(42));

        assert_eq!(de!(decimal::deserialize::<U256, _>, "0"), U256::new(0));
        assert_eq!(de!(decimal::deserialize::<U256, _>, "42"), U256::new(42));
        assert_eq!(
            de!(
                decimal::deserialize::<U256, _>,
                "115792089237316195423570985008687907853269984665640564039457584007913129639935"
            ),
            U256::MAX
        );

        assert_eq!(de!(prefixed::deserialize::<I256, _>, "-1"), I256::new(-1));
        assert_eq!(de!(prefixed::deserialize::<I256, _>, "-0x1"), I256::new(-1));
        assert_eq!(de!(prefixed::deserialize::<I256, _>, "42"), I256::new(42));
        assert_eq!(de!(prefixed::deserialize::<I256, _>, "0x2a"), I256::new(42));
        assert_eq!(de!(prefixed::deserialize::<I256, _>, "0x2A"), I256::new(42));

        assert_eq!(de!(prefixed::deserialize::<U256, _>, "42"), U256::new(42));
        assert_eq!(de!(prefixed::deserialize::<U256, _>, "0x2a"), U256::new(42));
        assert_eq!(de!(prefixed::deserialize::<U256, _>, "0x2A"), U256::new(42));

        assert_eq!(
            de!(permissive::deserialize::<I256, _>, -42_i64),
            I256::new(-42)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, 42_u64),
            I256::new(42)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, -1337_i128),
            I256::new(-1337)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, 1337_u128),
            I256::new(1337)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, 100.0_f32),
            I256::new(100)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, -100.0_f64),
            I256::new(-100)
        );
        assert_eq!(de!(permissive::deserialize::<I256, _>, "-1"), I256::new(-1));
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, "1000"),
            I256::new(1000)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, "0x42"),
            I256::new(0x42)
        );
        assert_eq!(
            de!(permissive::deserialize::<I256, _>, "-0x2a"),
            I256::new(-42)
        );
        assert_eq!(
            de!(
                permissive::deserialize::<I256, _>,
                "0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            ),
            I256::MAX
        );
        assert_eq!(
            de!(
                permissive::deserialize::<I256, _>,
                "-0x8000000000000000000000000000000000000000000000000000000000000000"
            ),
            I256::MIN
        );

        assert_eq!(
            de!(permissive::deserialize::<U256, _>, 42_u64),
            U256::new(42)
        );
        assert_eq!(
            de!(permissive::deserialize::<U256, _>, 1337_u128),
            U256::new(1337)
        );
        assert_eq!(
            de!(permissive::deserialize::<U256, _>, 100.0_f32),
            U256::new(100)
        );
        assert_eq!(
            de!(permissive::deserialize::<U256, _>, 100.0_f64),
            U256::new(100)
        );
        assert_eq!(
            de!(permissive::deserialize::<U256, _>, "1000"),
            U256::new(1000)
        );
        assert_eq!(
            de!(permissive::deserialize::<U256, _>, "0x42"),
            U256::new(0x42)
        );
        assert_eq!(
            de!(
                permissive::deserialize::<U256, _>,
                "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            ),
            U256::MAX
        );

        assert!(de!(err; permissive::deserialize::<I256, _>, 4.2_f32));
        assert!(de!(err; permissive::deserialize::<I256, _>, 16777216.0_f32));
        assert!(de!(err; permissive::deserialize::<I256, _>, -13.37_f64));
        assert!(de!(err; permissive::deserialize::<I256, _>, 9007199254740992.0_f32));
        assert!(
            de!(err; permissive::deserialize::<I256, _>, "0x8000000000000000000000000000000000000000000000000000000000000000")
        );
        assert!(
            de!(err; permissive::deserialize::<I256, _>, "-0x8000000000000000000000000000000000000000000000000000000000000001"
            )
        );

        assert!(de!(err; permissive::deserialize::<U256, _>, 4.2_f32));
        assert!(de!(err; permissive::deserialize::<U256, _>, 16777216.0_f32));
        assert!(de!(err; permissive::deserialize::<U256, _>, 13.37_f64));
        assert!(de!(err; permissive::deserialize::<U256, _>, 9007199254740992.0_f32));
        assert!(
            de!(err; permissive::deserialize::<U256, _>, "0x10000000000000000000000000000000000000000000000000000000000000000")
        );

        assert_de_bytes!(
            bytes::le::deserialize::<U256, _>, [0x00; 32];
            eq: U256::ZERO
        );
        assert_de_bytes!(
            bytes::le::deserialize::<U256, _>, [0xff; 32];
            eq: U256::MAX
        );

        assert_de_bytes!(
            bytes::le::deserialize::<I256, _>, [0x00; 32];
            eq: I256::ZERO
        );
        assert_de_bytes!(
            bytes::le::deserialize::<I256, _>, [0xff; 32];
            eq: I256::new(-1)
        );

        let forty_two = {
            let mut v = [0x00; 32];
            v[0] = 0x2a;
            v
        };
        assert_de_bytes!(
            bytes::le::deserialize::<U256, _>, forty_two;
            eq: U256::new(42)
        );
        assert_de_bytes!(
            bytes::le::deserialize::<I256, _>, forty_two;
            eq: I256::new(42)
        );

        assert_de_bytes!(
            bytes::le::deserialize::<U256, _>, [0xff; 31];
            err
        );
        assert_de_bytes!(
            bytes::le::deserialize::<U256, _>, [0xff; 33];
            err
        );
        assert_de_bytes!(
            bytes::le::deserialize::<I256, _>, [0xff; 31];
            err
        );
        assert_de_bytes!(
            bytes::le::deserialize::<I256, _>, [0xff; 33];
            err
        );

        assert_de_bytes!(
            bytes::be::deserialize::<U256, _>, [0x00; 32];
            eq: U256::ZERO
        );
        assert_de_bytes!(
            bytes::be::deserialize::<U256, _>, [0xff; 32];
            eq: U256::MAX
        );

        assert_de_bytes!(
            bytes::be::deserialize::<I256, _>, [0x00; 32];
            eq: I256::ZERO
        );
        assert_de_bytes!(
            bytes::be::deserialize::<I256, _>, [0xff; 32];
            eq: I256::new(-1)
        );

        let forty_two = {
            let mut v = [0x00; 32];
            v[31] = 0x2a;
            v
        };
        assert_de_bytes!(
            bytes::be::deserialize::<U256, _>, forty_two;
            eq: U256::new(42)
        );
        assert_de_bytes!(
            bytes::be::deserialize::<I256, _>, forty_two;
            eq: I256::new(42)
        );

        assert_de_bytes!(
            bytes::be::deserialize::<U256, _>, [0xff; 31];
            err
        );
        assert_de_bytes!(
            bytes::be::deserialize::<U256, _>, [0xff; 33];
            err
        );
        assert_de_bytes!(
            bytes::be::deserialize::<I256, _>, [0xff; 31];
            err
        );
        assert_de_bytes!(
            bytes::be::deserialize::<I256, _>, [0xff; 33];
            err
        );

        assert_de_bytes!(
            compressed_bytes::le::deserialize::<U256, _>, [];
            eq: U256::ZERO
        );
        assert_de_bytes!(
            compressed_bytes::le::deserialize::<U256, _>, [0xff; 32];
            eq: U256::MAX
        );

        assert_de_bytes!(
            compressed_bytes::le::deserialize::<U256, _>, [0x2a];
            eq: U256::new(42)
        );
        assert_de_bytes!(
            compressed_bytes::le::deserialize::<U256, _>, [0xee, 0xff];
            eq: U256::new(0xffee)
        );
        assert_de_bytes!(
            compressed_bytes::le::deserialize::<I256, _>, [];
            eq: I256::ZERO
        );
        assert_de_bytes!(
            compressed_bytes::le::deserialize::<I256, _>, [0xff];
            eq: I256::new(-1)
        );

        assert_de_bytes!(
            compressed_bytes::le::deserialize::<U256, _>, [0xff; 33];
            err
        );
        assert_de_bytes!(
            compressed_bytes::le::deserialize::<I256, _>, [0xff; 33];
            err
        );

        assert_de_bytes!(
            compressed_bytes::be::deserialize::<U256, _>, [];
            eq: U256::ZERO
        );
        assert_de_bytes!(
            compressed_bytes::be::deserialize::<U256, _>, [0xff; 32];
            eq: U256::MAX
        );

        assert_de_bytes!(
            compressed_bytes::be::deserialize::<U256, _>, [0x2a];
            eq: U256::new(42)
        );
        assert_de_bytes!(
            compressed_bytes::be::deserialize::<U256, _>, [0xff, 0xee];
            eq: U256::new(0xffee)
        );
        assert_de_bytes!(
            compressed_bytes::be::deserialize::<I256, _>, [];
            eq: I256::ZERO
        );
        assert_de_bytes!(
            compressed_bytes::be::deserialize::<I256, _>, [0xfe];
            eq: I256::new(-2)
        );

        assert_de_bytes!(
            compressed_bytes::be::deserialize::<U256, _>, [0xff; 33];
            err
        );
        assert_de_bytes!(
            compressed_bytes::be::deserialize::<I256, _>, [0xff; 33];
            err
        );
    }

    #[test]
    fn formatting_buffer() {
        for value in [
            Box::new(I256::MIN) as Box<dyn Display>,
            Box::new(I256::MAX),
            Box::new(U256::MIN),
            Box::new(U256::MAX),
        ] {
            let mut f = FormatBuffer::decimal();
            write!(f, "{value}").unwrap();
            assert_eq!(f.as_str(), format!("{value}"));
        }

        for value in [
            Box::new(I256::MIN) as Box<dyn LowerHex>,
            Box::new(I256::MAX),
            Box::new(U256::MIN),
            Box::new(U256::MAX),
        ] {
            let mut f = FormatBuffer::hex();
            let value = &*value;
            write!(f, "{value:-#x}").unwrap();
            assert_eq!(f.as_str(), format!("{value:-#x}"));
        }
    }

    /// A string serializer used for testing.
    struct StringSerializer;

    impl Serializer for StringSerializer {
        type Ok = String;
        type Error = fmt::Error;
        type SerializeSeq = Impossible<String, fmt::Error>;
        type SerializeTuple = Impossible<String, fmt::Error>;
        type SerializeTupleStruct = Impossible<String, fmt::Error>;
        type SerializeTupleVariant = Impossible<String, fmt::Error>;
        type SerializeMap = Impossible<String, fmt::Error>;
        type SerializeStruct = Impossible<String, fmt::Error>;
        type SerializeStructVariant = Impossible<String, fmt::Error>;
        fn serialize_bool(self, _: bool) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i8(self, _: i8) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i16(self, _: i16) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i32(self, _: i32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i64(self, _: i64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u8(self, _: u8) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u16(self, _: u16) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u32(self, _: u32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u64(self, _: u64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_f32(self, _: f32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_f64(self, _: f64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_char(self, _: char) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
            Ok(v.into())
        }
        fn serialize_bytes(self, _: &[u8]) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_some<T>(self, _: &T) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_unit_struct(self, _: &'static str) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_unit_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
        ) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_newtype_struct<T>(
            self,
            _: &'static str,
            _: &T,
        ) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_newtype_variant<T>(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: &T,
        ) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple_struct(
            self,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleStruct, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleVariant, Self::Error> {
            unimplemented!()
        }
        fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
            unimplemented!()
        }
        fn serialize_struct(
            self,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStruct, Self::Error> {
            unimplemented!()
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStructVariant, Self::Error> {
            unimplemented!()
        }
        fn collect_str<T>(self, _: &T) -> Result<Self::Ok, Self::Error>
        where
            T: Display + ?Sized,
        {
            unimplemented!()
        }
    }

    /// A string serializer used for testing.
    struct BytesSerializer;

    impl Serializer for BytesSerializer {
        type Ok = Vec<u8>;
        type Error = fmt::Error;
        type SerializeSeq = Impossible<Vec<u8>, fmt::Error>;
        type SerializeTuple = Impossible<Vec<u8>, fmt::Error>;
        type SerializeTupleStruct = Impossible<Vec<u8>, fmt::Error>;
        type SerializeTupleVariant = Impossible<Vec<u8>, fmt::Error>;
        type SerializeMap = Impossible<Vec<u8>, fmt::Error>;
        type SerializeStruct = Impossible<Vec<u8>, fmt::Error>;
        type SerializeStructVariant = Impossible<Vec<u8>, fmt::Error>;
        fn serialize_bool(self, _: bool) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i8(self, _: i8) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i16(self, _: i16) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i32(self, _: i32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_i64(self, _: i64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u8(self, _: u8) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u16(self, _: u16) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u32(self, _: u32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_u64(self, _: u64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_f32(self, _: f32) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_f64(self, _: f64) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_char(self, _: char) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_str(self, _: &str) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
            Ok(v.to_vec())
        }
        fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_some<T>(self, _: &T) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_unit_struct(self, _: &'static str) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_unit_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
        ) -> Result<Self::Ok, Self::Error> {
            unimplemented!()
        }
        fn serialize_newtype_struct<T>(
            self,
            _: &'static str,
            _: &T,
        ) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_newtype_variant<T>(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: &T,
        ) -> Result<Self::Ok, Self::Error>
        where
            T: Serialize + ?Sized,
        {
            unimplemented!()
        }
        fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple_struct(
            self,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleStruct, Self::Error> {
            unimplemented!()
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleVariant, Self::Error> {
            unimplemented!()
        }
        fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
            unimplemented!()
        }
        fn serialize_struct(
            self,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStruct, Self::Error> {
            unimplemented!()
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStructVariant, Self::Error> {
            unimplemented!()
        }
        fn collect_str<T>(self, _: &T) -> Result<Self::Ok, Self::Error>
        where
            T: Display + ?Sized,
        {
            unimplemented!()
        }
    }
}
