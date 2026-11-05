use crate::{builder::Builder, traits::*, Offset, UnionVectorOffset};

impl<T, P> WriteAsOffset<[P]> for alloc::vec::Vec<T>
where
    [T]: WriteAsOffset<[P]>,
{
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<[P]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<T, P> WriteAs<Offset<[P]>> for alloc::vec::Vec<T>
where
    [T]: WriteAsOffset<[P]>,
{
    type Prepared = Offset<[P]>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<[P]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<T, P> WriteAsDefault<Offset<[P]>, ()> for alloc::vec::Vec<T>
where
    [T]: WriteAsOffset<[P]>,
{
    type Prepared = Offset<[P]>;

    #[inline]
    fn prepare(&self, builder: &mut Builder, _default: &()) -> Option<Offset<[P]>> {
        if self.is_empty() {
            None
        } else {
            Some(WriteAsOffset::prepare(self.as_slice(), builder))
        }
    }
}

impl<T, P> WriteAsOptional<Offset<[P]>> for alloc::vec::Vec<T>
where
    [T]: WriteAsOffset<[P]>,
{
    type Prepared = Offset<[P]>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<Offset<[P]>> {
        Some(WriteAsOffset::prepare(self.as_slice(), builder))
    }
}

impl<T, P> WriteAsUnionVector<P> for alloc::vec::Vec<T>
where
    T: WriteAsUnion<P>,
{
    fn prepare(&self, builder: &mut Builder) -> UnionVectorOffset<P> {
        WriteAsUnionVector::prepare(self.as_slice(), builder)
    }
}

impl<T, P> WriteAsDefaultUnionVector<P> for alloc::vec::Vec<T>
where
    T: WriteAsUnion<P>,
{
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<P>> {
        if self.is_empty() {
            None
        } else {
            Some(WriteAsUnionVector::prepare(self.as_slice(), builder))
        }
    }
}

impl<T, P> WriteAsOptionalUnionVector<P> for alloc::vec::Vec<T>
where
    T: WriteAsUnion<P>,
{
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<UnionVectorOffset<P>> {
        Some(WriteAsUnionVector::prepare(self.as_slice(), builder))
    }
}
