use core::mem::MaybeUninit;

use crate::{builder::Builder, traits::*, Cursor, Offset, UnionOffset, UnionVectorOffset};

impl<P: Primitive, T: ?Sized + WriteAsPrimitive<P>> WriteAsPrimitive<P> for &T {
    #[inline]
    fn write<const N: usize>(&self, cursor: Cursor<'_, N>, buffer_position: u32) {
        T::write(*self, cursor, buffer_position)
    }
}

impl<T1: ?Sized, T2: ?Sized + WriteAsOffset<T1>> WriteAsOffset<T1> for &T2 {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<T1> {
        T2::prepare(self, builder)
    }
}

impl<P: Primitive, T: ?Sized + WriteAs<P>> WriteAs<P> for &T {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> T::Prepared {
        T::prepare(self, builder)
    }
}

impl<P: Primitive, D: ?Sized, T: ?Sized + WriteAsDefault<P, D>> WriteAsDefault<P, D> for &T {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder, default: &D) -> Option<T::Prepared> {
        T::prepare(self, builder, default)
    }
}

impl<P: Primitive, T: ?Sized + WriteAsOptional<P>> WriteAsOptional<P> for &T {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<T::Prepared> {
        T::prepare(self, builder)
    }
}

impl<T1: ?Sized, T2: ?Sized + WriteAsUnion<T1>> WriteAsUnion<T1> for &T2 {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> UnionOffset<T1> {
        T2::prepare(self, builder)
    }
}

impl<T1: ?Sized, T2: ?Sized + WriteAsOptionalUnion<T1>> WriteAsOptionalUnion<T1> for &T2 {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionOffset<T1>> {
        T2::prepare(self, builder)
    }
}

impl<P, T: ?Sized + WriteAsUnionVector<P>> WriteAsUnionVector<P> for &T {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> UnionVectorOffset<P> {
        T::prepare(self, builder)
    }
}

impl<P, T: ?Sized + WriteAsDefaultUnionVector<P>> WriteAsDefaultUnionVector<P> for &T {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<P>> {
        T::prepare(self, builder)
    }
}

impl<P, T: ?Sized + WriteAsOptionalUnionVector<P>> WriteAsOptionalUnionVector<P> for &T {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<P>> {
        T::prepare(self, builder)
    }
}

/// # Safety
/// `T` must implement `VectorWrite` following the safety requirements from the trait.
unsafe impl<P: Primitive, T: ?Sized + VectorWrite<P>> VectorWrite<P> for &T {
    const STRIDE: usize = T::STRIDE;
    type Value = T::Value;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Self::Value {
        T::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[Self::Value],
        bytes: *mut MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        T::write_values(values, bytes, buffer_position);
    }
}
