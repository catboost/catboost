use crate::{
    builder::Builder,
    traits::{Primitive, WriteAsOptional, WriteAsOptionalUnion, WriteAsPrimitive},
    Cursor, UnionOffset, UnionVectorOffset, Void, WriteAsOptionalUnionVector,
};

impl<P: Primitive> WriteAsPrimitive<P> for Void {
    #[inline]
    fn write<const N: usize>(&self, _cursor: Cursor<'_, N>, _buffer_position: u32) {
        match *self {}
    }
}

impl<P: Primitive> WriteAsOptional<P> for () {
    type Prepared = Void;
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<Void> {
        None
    }
}

impl<T: ?Sized> WriteAsOptionalUnion<T> for () {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<UnionOffset<T>> {
        None
    }
}

impl<T: ?Sized> WriteAsOptionalUnionVector<T> for () {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<UnionVectorOffset<T>> {
        None
    }
}
