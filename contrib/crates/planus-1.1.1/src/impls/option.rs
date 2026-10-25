use crate::{builder::Builder, traits::*, UnionOffset, UnionVectorOffset};

impl<P: Primitive, T: WriteAsOptional<P>> WriteAsOptional<P> for Option<T> {
    type Prepared = T::Prepared;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<T::Prepared> {
        self.as_ref()?.prepare(builder)
    }
}

impl<T1, T2: WriteAsOptionalUnion<T1>> WriteAsOptionalUnion<T1> for Option<T2> {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionOffset<T1>> {
        self.as_ref()?.prepare(builder)
    }
}

impl<T1, T2: WriteAsOptionalUnionVector<T1>> WriteAsOptionalUnionVector<T1> for Option<T2> {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<T1>> {
        self.as_ref()?.prepare(builder)
    }
}
