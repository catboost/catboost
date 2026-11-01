//! A crate providing generalised value traits for working with
//! `JSONesque` values.
#![warn(unused_extern_crates)]
#![cfg_attr(feature = "portable", feature(portable_simd))]
#![deny(
    clippy::all,
    clippy::unwrap_used,
    clippy::unnecessary_unwrap,
    clippy::pedantic,
    missing_docs
)]
// We might want to revisit inline_always
#![allow(clippy::module_name_repetitions)]

#[cfg(all(feature = "128bit", feature = "c-abi"))]
compile_error!(
    "Combining the features `128bit` and `c-abi` is impossible because i128's \
    ABI is unstable (see \
    https://github.com/rust-lang/unsafe-code-guidelines/issues/119). Please \
    use only one of them in order to compile this crate. If you don't know \
    where this error is coming from, it's possible that you depend on \
    value-trait twice indirectly, once with the `c-abi` feature, and once with \
    the `128bit` feature, and that they have been merged by Cargo."
);

#[cfg(all(feature = "c-abi", feature = "ordered-float"))]
compile_error!(
    "Combining the features `c-abi` and `ordered-float` is impossible because ordered_float::OrderedFloat does not implement `StableAbi`"
);

use std::borrow::Cow;
use std::fmt;

mod array;
/// Traits for serializing JSON
pub mod generator;
mod impls;
mod node;
mod object;
mod option;
/// Prelude for traits
pub mod prelude;

/// Traits that provide basic interactions, they do have no auto-implementations
pub mod base;

/// Traits that have derived implementations relying on `base` traitsa
pub mod derived;

pub use node::StaticNode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An access error for `ValueType`
pub enum AccessError {
    /// An access attempt to a Value was made under the
    /// assumption that it is an Object - the Value however
    /// wasn't.
    NotAnObject,
    /// An access attempt to a Value was made under the
    /// assumption that it is an Array - the Value however
    /// wasn't.
    NotAnArray,
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnArray => write!(f, "The value is not an array"),
            Self::NotAnObject => write!(f, "The value is not an object"),
        }
    }
}
impl std::error::Error for AccessError {}

/// Extended types that have no native representation in JSON
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendedValueType {
    /// A 32 bit signed integer value
    I32,
    /// A 16 bit signed integer value
    I16,
    /// A 8 bit signed integer value
    I8,
    /// A 32 bit unsigned integer value
    U32,
    /// A 16 bit unsigned integer value
    U16,
    /// A 8 bit unsigned integer value
    U8,
    /// A useize value
    Usize,
    /// A 32 bit floating point value
    F32,
    /// A single utf-8 character
    Char,
    /// Not a value at all
    None,
}

impl Default for ExtendedValueType {
    fn default() -> Self {
        Self::None
    }
}

impl fmt::Display for ExtendedValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::I16 => write!(f, "i16"),
            Self::I8 => write!(f, "i8"),
            Self::U32 => write!(f, "u32"),
            Self::U16 => write!(f, "u16"),
            Self::U8 => write!(f, "u8"),
            Self::Usize => write!(f, "usize"),
            Self::F32 => write!(f, "f32"),
            Self::Char => write!(f, "char"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Types of JSON values
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueType {
    /// null
    Null,
    /// a boolean
    Bool,
    /// a signed integer type
    I64,
    /// a 128 bit signed integer
    I128,
    /// a unsigned integer type
    U64,
    /// a 128 bit unsigned integer
    U128,
    /// a float type
    F64,
    /// a string type
    String,
    /// an array
    Array,
    /// an object
    Object,
    /// Extended types that do not have a real representation in JSON
    Extended(ExtendedValueType),
    #[cfg(feature = "custom-types")]
    /// a custom type
    Custom(&'static str),
}
impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool => write!(f, "bool"),
            Self::I64 => write!(f, "i64"),
            Self::I128 => write!(f, "i128"),
            Self::U64 => write!(f, "u64"),
            Self::U128 => write!(f, "u128"),
            Self::F64 => write!(f, "f64"),
            Self::String => write!(f, "string"),
            Self::Array => write!(f, "array"),
            Self::Object => write!(f, "object"),
            Self::Extended(ty) => write!(f, "{ty}"),
            #[cfg(feature = "custom-types")]
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl Default for ValueType {
    fn default() -> Self {
        Self::Null
    }
}

#[allow(clippy::trait_duplication_in_bounds)] // This is a bug From<()> is counted as duplicate
/// Support of builder methods for traits.
pub trait ValueBuilder<'input>:
    Default
    + From<StaticNode>
    + From<i8>
    + From<i16>
    + From<i32>
    + From<i64>
    + From<u8>
    + From<u16>
    + From<u32>
    + From<u64>
    + From<f32>
    + From<f64>
    + From<bool>
    + From<()>
    + From<String>
    + From<&'input str>
    + From<Cow<'input, str>>
{
    /// Returns an empty array with a given capacity
    fn array_with_capacity(capacity: usize) -> Self;
    /// Returns an empty object with a given capacity
    fn object_with_capacity(capacity: usize) -> Self;
    /// Returns an empty array
    #[must_use]
    fn array() -> Self {
        Self::array_with_capacity(0)
    }
    /// Returns an empty object
    #[must_use]
    fn object() -> Self {
        Self::object_with_capacity(0)
    }
    /// Returns anull value
    fn null() -> Self;
}

/// A type error thrown by the `try_*` functions
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryTypeError {
    /// The expected value type
    pub expected: ValueType,
    /// The actual value type
    pub got: ValueType,
}

impl std::fmt::Display for TryTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expected type {}, got {}", self.expected, self.got)
    }
}

impl std::error::Error for TryTypeError {}

// /// The `Value` exposes common interface for values, this allows using both/// `BorrowedValue` and `OwnedValue` nearly interchangeable
// pub trait Value:
//     Sized
//     + Index<usize>
//     + PartialEq<i8>
//     + PartialEq<i16>
//     + PartialEq<i32>
//     + PartialEq<i64>
//     + PartialEq<i128>
//     + PartialEq<u8>
//     + PartialEq<u16>
//     + PartialEq<u32>
//     + PartialEq<u64>
//     + PartialEq<u128>
//     + PartialEq<f32>
//     + PartialEq<f64>
//     + PartialEq<String>
//     + PartialEq<bool>
//     + PartialEq<()>
//     + derived::ValueTryAsScalar
//     + base::ValueAsContainer
// {
// }
