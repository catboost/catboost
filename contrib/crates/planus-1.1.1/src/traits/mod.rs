use core::mem::MaybeUninit;

use crate::{
    builder::Builder, errors::ErrorKind, slice_helpers::SliceWithStartOffset, Cursor, Offset,
    Result, UnionOffset, UnionVectorOffset,
};

#[doc(hidden)]
/// # Safety
/// `ALIGNMENT` match the actual alignment requirements of the type. It most likely is a power of two.
/// `SIZE` match the actual size of the type. For primitive types, that is core::mem::size_of::<Self>().
pub unsafe trait Primitive {
    const ALIGNMENT: usize;
    const ALIGNMENT_MASK: usize = Self::ALIGNMENT - 1;
    const SIZE: usize;
}

/// Interface for getting a view into serialized data.
///
/// To get an owned variant use [`TryInto`] on the `Ref` type. Note that for
/// nested types with lots of sharing the owned variants can be much larger than
/// the serialized representation.
///
/// # Examples
///
/// ```no_run
/// use std::error::Error;
/// use planus::ReadAsRoot;
/// use planus_example::monster_generated::my_game::sample::{Monster, MonsterRef};
///
/// fn main() -> Result<(), Box<dyn Error>> {
///     let buf = std::fs::read("monster.bin")?;
///     let monster: MonsterRef<'_> = MonsterRef::read_as_root(&buf)?;
///     let monster_health = monster.hp()?;
///     let owned_monster: Monster = monster.try_into().expect("invalid monster");
///     Ok(())
/// }
pub trait ReadAsRoot<'a>: Sized {
    /// Takes a slice assumed to be of this type and returns a view into it.
    ///
    /// If the data is not valid for this type the field accessors will give
    /// errors or invalid values, but will still be memory safe.
    fn read_as_root(slice: &'a [u8]) -> Result<Self>;
}

/// Trait used by generated code to serialize primitive types.
pub trait WriteAs<P: Primitive> {
    #[doc(hidden)]
    type Prepared: WriteAsPrimitive<P>;
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Self::Prepared;
}

/// Trait used by generated code to serialize primitive types with default values.
pub trait WriteAsDefault<P: Primitive, D: ?Sized> {
    #[doc(hidden)]
    type Prepared: WriteAsPrimitive<P>;
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder, default: &D) -> Option<Self::Prepared>;
}

/// Trait used by generated code to serialize optional primitive types.
pub trait WriteAsOptional<P: Primitive> {
    #[doc(hidden)]
    type Prepared: WriteAsPrimitive<P>;
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Option<Self::Prepared>;
}

/// Trait used by generated code to serialize offsets to already serialized data.
pub trait WriteAsOffset<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Offset<T>;
}

/// Trait used by generated code to serialize offsets to unions.
pub trait WriteAsUnion<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> UnionOffset<T>;
}

/// Trait used by generated code to serialize offsets to optional unions.
pub trait WriteAsOptionalUnion<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionOffset<T>>;
}

/// Trait used by generated code to serialize offsets to unions.
pub trait WriteAsUnionVector<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> UnionVectorOffset<T>;
}

/// Trait used by generated code to serialize offsets to union.
pub trait WriteAsDefaultUnionVector<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<T>>;
}

/// Trait used by generated code to serialize offsets to optional unions.
pub trait WriteAsOptionalUnionVector<T: ?Sized> {
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<T>>;
}

#[doc(hidden)]
pub trait WriteAsPrimitive<P> {
    fn write<const N: usize>(&self, cursor: Cursor<'_, N>, buffer_position: u32);
}

#[doc(hidden)]
pub trait TableRead<'buf>: Sized {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> core::result::Result<Self, ErrorKind>;
}

#[doc(hidden)]
pub trait TableReadUnion<'buf>: 'buf + Sized {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        tag: u8,
        offset: usize,
    ) -> core::result::Result<Self, ErrorKind>;
}

#[doc(hidden)]
pub trait TableReadUnionVector<'buf>: Sized {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        tags_offset: usize,
        values_offset: usize,
    ) -> core::result::Result<Self, ErrorKind>;
}

/// Trait used by generated code to read elements from vectors.
pub trait VectorRead<'buf>: 'buf {
    #[doc(hidden)]
    const STRIDE: usize;
    #[doc(hidden)]
    unsafe fn from_buffer(buffer: SliceWithStartOffset<'buf>, offset: usize) -> Self;
}

#[doc(hidden)]
pub trait VectorReadUnion<'buf>: 'buf + Sized + TableReadUnion<'buf> {
    const VECTOR_NAME: &'static str;
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        tag: u8,
        offset: usize,
    ) -> crate::Result<Self> {
        <Self as TableReadUnion>::from_buffer(buffer, tag, offset)
            .map_err(|e| e.with_error_location(Self::VECTOR_NAME, "get", buffer.offset_from_start))
    }
}

/// This trait is a hack to get around the coherence restriction.
/// Ideally we would want to be able to do an `impl VectorRead<'buf> for planus::Result<MyType>`
/// in our generated code, however instead we do something like this:
///   impl<T: VectorReadInner<'buf>, E> VectorRead<'buf> for Result<T, E>
#[doc(hidden)]
pub trait VectorReadInner<'buf>: 'buf + Sized {
    #[doc(hidden)]
    type Error: Sized;
    #[doc(hidden)]
    const STRIDE: usize;
    #[doc(hidden)]
    unsafe fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> core::result::Result<Self, Self::Error>;
}

/// Trait used by generated code to write elements to vectors.
///
/// # Safety
/// The implementation of write_values should initialize the bytes as
/// downstream code will assume so.
pub unsafe trait VectorWrite<P> {
    #[doc(hidden)]
    const STRIDE: usize;
    #[doc(hidden)]
    type Value: WriteAsPrimitive<P> + Sized;
    #[doc(hidden)]
    fn prepare(&self, builder: &mut Builder) -> Self::Value;
    #[doc(hidden)]
    unsafe fn write_values(
        values: &[Self::Value],
        bytes: *mut MaybeUninit<u8>,
        buffer_position: u32,
    );
}
