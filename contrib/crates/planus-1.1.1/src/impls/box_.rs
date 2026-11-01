use crate::{builder::Builder, traits::*, Offset, UnionOffset};

impl<P, T: ?Sized + WriteAsOffset<P>> WriteAsOffset<P> for alloc::boxed::Box<T> {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<P> {
        T::prepare(self, builder)
    }
}

impl<P: Primitive, T: ?Sized + WriteAs<P>> WriteAs<P> for alloc::boxed::Box<T> {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> T::Prepared {
        T::prepare(self, builder)
    }
}

impl<P: Primitive, D: ?Sized, T: ?Sized + WriteAsDefault<P, D>> WriteAsDefault<P, D>
    for alloc::boxed::Box<T>
{
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder, default: &D) -> Option<T::Prepared> {
        T::prepare(self, builder, default)
    }
}

impl<P: Primitive, T: ?Sized + WriteAsOptional<P>> WriteAsOptional<P> for alloc::boxed::Box<T> {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<T::Prepared> {
        T::prepare(self, builder)
    }
}

impl<T1: ?Sized, T2: ?Sized + WriteAsUnion<T1>> WriteAsUnion<T1> for alloc::boxed::Box<T2> {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> UnionOffset<T1> {
        T2::prepare(self, builder)
    }
}

impl<T1: ?Sized, T2: ?Sized + WriteAsOptionalUnion<T1>> WriteAsOptionalUnion<T1>
    for alloc::boxed::Box<T2>
{
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionOffset<T1>> {
        T2::prepare(self, builder)
    }
}
