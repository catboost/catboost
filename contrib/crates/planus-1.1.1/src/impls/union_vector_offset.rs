use crate::{builder::Builder, traits::*, UnionVectorOffset};

impl<T: ?Sized> WriteAsUnionVector<T> for UnionVectorOffset<T> {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Self {
        *self
    }
}

impl<T: ?Sized> WriteAsOptionalUnionVector<T> for UnionVectorOffset<T> {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<Self> {
        Some(*self)
    }
}

impl<T: ?Sized> WriteAsDefaultUnionVector<T> for UnionVectorOffset<T> {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<Self> {
        Some(*self)
    }
}
