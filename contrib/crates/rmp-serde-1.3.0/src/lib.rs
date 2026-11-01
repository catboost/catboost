#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, missing_docs)]

use std::fmt::{self, Display, Formatter};
use std::str::{self, Utf8Error};

use serde::de;
use serde::{Deserialize, Serialize};

#[allow(deprecated)]
pub use crate::decode::from_read_ref;
pub use crate::decode::{from_read, Deserializer};
pub use crate::encode::{to_vec, to_vec_named, Serializer};

pub use crate::decode::from_slice;

mod bytes;
pub mod config;
pub mod decode;
pub mod encode;

/// Hack used to serialize MessagePack Extension types.
///
/// A special `ExtStruct` type is used to represent
/// extension types. This struct is renamed in serde.
///
/// Name of Serde newtype struct to Represent Msgpack's Ext
/// Msgpack Ext: `Ext(tag, binary)`
/// Serde data model: `_ExtStruct((tag, binary))`
///
/// Example Serde impl for custom type:
///
/// ```ignore
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// #[serde(rename = "_ExtStruct")]
/// struct ExtStruct((i8, serde_bytes::ByteBuf));
///
/// test_round(ExtStruct((2, serde_bytes::ByteBuf::from(vec![5]))),
///            Value::Ext(2, vec![5]));
/// ```
pub const MSGPACK_EXT_STRUCT_NAME: &str = "_ExtStruct";

/// Helper that allows both to encode and decode strings no matter whether they contain valid or
/// invalid UTF-8.
///
/// Regardless of validity the UTF-8 content this type will always be serialized as a string.
#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct Raw {
    s: Result<String, (Vec<u8>, Utf8Error)>,
}

impl Raw {
    /// Constructs a new `Raw` from the UTF-8 string.
    #[inline]
    #[must_use]
    pub fn new(v: String) -> Self {
        Self { s: Ok(v) }
    }

    /// DO NOT USE. See <https://github.com/3Hren/msgpack-rust/issues/305>
    #[deprecated(note = "This feature has been removed")]
    #[must_use]
    pub fn from_utf8(v: Vec<u8>) -> Self {
        match String::from_utf8(v) {
            Ok(v) => Raw::new(v),
            Err(err) => {
                let e = err.utf8_error();
                Self {
                    s: Err((err.into_bytes(), e)),
                }
            }
        }
    }

    /// Returns `true` if the raw is valid UTF-8.
    #[inline]
    #[must_use]
    pub fn is_str(&self) -> bool {
        self.s.is_ok()
    }

    /// Returns `true` if the raw contains invalid UTF-8 sequence.
    #[inline]
    #[must_use]
    pub fn is_err(&self) -> bool {
        self.s.is_err()
    }

    /// Returns the string reference if the raw is valid UTF-8, or else `None`.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self.s {
            Ok(ref s) => Some(s.as_str()),
            Err(..) => None,
        }
    }

    /// Returns the underlying `Utf8Error` if the raw contains invalid UTF-8 sequence, or
    /// else `None`.
    #[inline]
    #[must_use]
    pub fn as_err(&self) -> Option<&Utf8Error> {
        match self.s {
            Ok(..) => None,
            Err((_, ref err)) => Some(err),
        }
    }

    /// Returns a byte slice of this raw's contents.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self.s {
            Ok(ref s) => s.as_bytes(),
            Err(ref err) => &err.0[..],
        }
    }

    /// Consumes this object, yielding the string if the raw is valid UTF-8, or else `None`.
    #[inline]
    #[must_use]
    pub fn into_str(self) -> Option<String> {
        self.s.ok()
    }

    /// Converts a `Raw` into a byte vector.
    #[inline]
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        match self.s {
            Ok(s) => s.into_bytes(),
            Err(err) => err.0,
        }
    }
}

impl Serialize for Raw {
    fn serialize<S>(&self, se: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.s {
            Ok(ref s) => se.serialize_str(s),
            Err((ref b, ..)) => se.serialize_bytes(b),
        }
    }
}

struct RawVisitor;

impl<'de> de::Visitor<'de> for RawVisitor {
    type Value = Raw;

    #[cold]
    fn expecting(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        "string or bytes".fmt(fmt)
    }

    #[inline]
    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(Raw { s: Ok(v) })
    }

    #[inline]
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where E: de::Error
    {
        Ok(Raw { s: Ok(v.into()) })
    }

    #[inline]
    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match str::from_utf8(v) {
            Ok(s) => Ok(s.into()),
            Err(err) => Err((v.into(), err)),
        };

        Ok(Raw { s })
    }

    #[inline]
    fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match String::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(err) => {
                let e = err.utf8_error();
                Err((err.into_bytes(), e))
            }
        };

        Ok(Raw { s })
    }
}

impl<'de> Deserialize<'de> for Raw {
    #[inline]
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
        where D: de::Deserializer<'de>
    {
        de.deserialize_any(RawVisitor)
    }
}

/// Helper that allows both to encode and decode strings no matter whether they contain valid or
/// invalid UTF-8.
///
/// Regardless of validity the UTF-8 content this type will always be serialized as a string.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct RawRef<'a> {
    s: Result<&'a str, (&'a [u8], Utf8Error)>,
}

impl<'a> RawRef<'a> {
    /// Constructs a new `RawRef` from the UTF-8 string.
    #[inline]
    #[must_use]
    pub fn new(v: &'a str) -> Self {
        Self { s: Ok(v) }
    }

    #[deprecated(note = "This feature has been removed")]
    #[must_use]
    pub fn from_utf8(v: &'a [u8]) -> Self {
        match str::from_utf8(v) {
            Ok(v) => RawRef::new(v),
            Err(err) => {
                Self {
                    s: Err((v, err))
                }
            }
        }
    }

    /// Returns `true` if the raw is valid UTF-8.
    #[inline]
    #[must_use]
    pub fn is_str(&self) -> bool {
        self.s.is_ok()
    }

    /// Returns `true` if the raw contains invalid UTF-8 sequence.
    #[inline]
    #[must_use]
    pub fn is_err(&self) -> bool {
        self.s.is_err()
    }

    /// Returns the string reference if the raw is valid UTF-8, or else `None`.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self.s {
            Ok(s) => Some(s),
            Err(..) => None,
        }
    }

    /// Returns the underlying `Utf8Error` if the raw contains invalid UTF-8 sequence, or
    /// else `None`.
    #[inline]
    #[must_use]
    pub fn as_err(&self) -> Option<&Utf8Error> {
        match self.s {
            Ok(..) => None,
            Err((_, ref err)) => Some(err),
        }
    }

    /// Returns a byte slice of this raw's contents.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self.s {
            Ok(s) => s.as_bytes(),
            Err((bytes, _err)) => bytes,
        }
    }
}

impl<'a> Serialize for RawRef<'a> {
    fn serialize<S>(&self, se: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.s {
            Ok(s) => se.serialize_str(s),
            Err((b, ..)) => se.serialize_bytes(b),
        }
    }
}

struct RawRefVisitor;

impl<'de> de::Visitor<'de> for RawRefVisitor {
    type Value = RawRef<'de>;

    #[cold]
    fn expecting(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        "string or bytes".fmt(fmt)
    }

    #[inline]
    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
        where E: de::Error
    {
        Ok(RawRef { s: Ok(v) })
    }

    #[inline]
    fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
        where E: de::Error
    {
        let s = match str::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(err) => Err((v, err)),
        };

        Ok(RawRef { s })
    }
}

impl<'de> Deserialize<'de> for RawRef<'de> {
    #[inline]
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
        where D: de::Deserializer<'de>
    {
        de.deserialize_any(RawRefVisitor)
    }
}
