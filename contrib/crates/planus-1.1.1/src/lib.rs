#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
mod backvec;
mod builder;
mod impls;
mod slice_helpers;
mod traits;

/// Error types for serialization/deserialization
pub mod errors;
/// Types for interacting with vectors of unions in serialized data
pub mod union_vectors;
/// Types for interacting with vectors in serialized data
pub mod vectors;

#[cfg(any(
    feature = "vtable-cache",
    feature = "string-cache",
    feature = "bytes-cache"
))]
mod builder_cache;

#[doc(hidden)]
pub extern crate alloc;
#[doc(hidden)]
pub mod table_reader;
#[doc(hidden)]
pub mod table_writer;

pub use crate::{
    builder::Builder,
    errors::Error,
    slice_helpers::{ArrayWithStartOffset, SliceWithStartOffset},
    traits::*,
    union_vectors::UnionVector,
    vectors::Vector,
};

#[doc(hidden)]
pub const fn check_version_compatibility(s: &str) {
    match s.as_bytes() {
        b"planus-1.1.1" => (),
        _ => panic!(
            "Your generated code is out of date, please regenerate using planus version 1.1.1"
        ),
    }
}

/// A type alias for [`Result`] with a Planus error
///
/// It is recommended to handle reading of serialized data in functions
/// returning this result type to avoid boilerplate error handling using
/// the ? operator.
///
/// [`Result`]: core::result::Result
pub type Result<T> = core::result::Result<T, Error>;
#[doc(hidden)]
pub type Cursor<'a, const N: usize> = array_init_cursor::Cursor<'a, u8, N>;

#[doc(hidden)]
pub enum Void {}

#[doc(hidden)]
/// Used in the union-builders in generated code
pub struct Uninitialized;

#[doc(hidden)]
/// Used in the union-builders in generated code
pub struct Initialized<const N: u8, T>(pub T);

#[doc(hidden)]
/// Used in the tables-builders in generated code
pub struct DefaultValue;

impl<P: Primitive, D: ?Sized> WriteAsDefault<P, D> for DefaultValue {
    type Prepared = Void;
    fn prepare(&self, _builder: &mut Builder, _default: &D) -> Option<Self::Prepared> {
        None
    }
}

impl<P> WriteAsDefaultUnionVector<P> for DefaultValue {
    fn prepare(&self, _builder: &mut Builder) -> Option<UnionVectorOffset<P>> {
        None
    }
}

impl From<Void> for crate::Error {
    fn from(v: Void) -> Self {
        match v {}
    }
}

/// An offset to a serialized value of type T inside a buffer currently being built.
pub struct Offset<T: ?Sized> {
    offset: u32,
    phantom: core::marker::PhantomData<T>,
}
impl<T: ?Sized> Copy for Offset<T> {}
impl<T: ?Sized> Clone for Offset<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Offset<T> {
    #[doc(hidden)]
    pub fn downcast(&self) -> Offset<()> {
        Offset {
            offset: self.offset,
            phantom: core::marker::PhantomData,
        }
    }
}

/// An offset to a serialized union value of type T inside a buffer currently being built.
pub struct UnionOffset<T: ?Sized> {
    tag: u8,
    offset: Offset<()>,
    phantom: core::marker::PhantomData<T>,
}
impl<T: ?Sized> Copy for UnionOffset<T> {}
impl<T: ?Sized> Clone for UnionOffset<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> UnionOffset<T> {
    #[doc(hidden)]
    #[inline]
    pub fn new(tag: u8, offset: Offset<()>) -> Self {
        Self {
            tag,
            offset,
            phantom: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn tag(&self) -> u8 {
        self.tag
    }

    #[doc(hidden)]
    #[inline]
    pub fn offset(&self) -> Offset<()> {
        self.offset
    }
}

/// An offset to a serialized vector of union values of type T and vector of union tags inside a buffer currently being built
pub struct UnionVectorOffset<T: ?Sized> {
    tags_offset: Offset<[u8]>,
    values_offset: Offset<[Offset<()>]>,
    phantom: core::marker::PhantomData<T>,
}
impl<T: ?Sized> Copy for UnionVectorOffset<T> {}
impl<T: ?Sized> Clone for UnionVectorOffset<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> UnionVectorOffset<T> {
    #[doc(hidden)]
    #[inline]
    pub fn new(tags_offset: Offset<[u8]>, values_offset: Offset<[Offset<()>]>) -> Self {
        Self {
            tags_offset,
            values_offset,
            phantom: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn tags_offset(&self) -> Offset<[u8]> {
        self.tags_offset
    }

    #[doc(hidden)]
    #[inline]
    pub fn values_offset(&self) -> Offset<[Offset<()>]> {
        self.values_offset
    }
}
