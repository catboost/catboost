use crate::{builder::Builder, traits::*, UnionOffset};

impl<T: ?Sized> WriteAsUnion<T> for UnionOffset<T> {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Self {
        *self
    }
}

impl<T: ?Sized> WriteAsOptionalUnion<T> for UnionOffset<T> {
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<Self> {
        Some(*self)
    }
}
