use crate::imp_prelude::*;

/// Arrays and similar that can be split along an axis
pub(crate) trait SplitAt
{
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    where Self: Sized;
}

pub(crate) trait SplitPreference: SplitAt
{
    #[allow(dead_code)] // used only when Rayon support is enabled
    fn can_split(&self) -> bool;
    fn split_preference(&self) -> (Axis, usize);
    fn split(self) -> (Self, Self)
    where Self: Sized
    {
        let (axis, index) = self.split_preference();
        self.split_at(axis, index)
    }
}

impl<D> SplitAt for D
where D: Dimension
{
    fn split_at(self, axis: Axis, index: Ix) -> (Self, Self)
    {
        let mut d1 = self;
        let mut d2 = d1.clone();
        let i = axis.index();
        let len = d1[i];
        d1[i] = index;
        d2[i] = len - index;
        (d1, d2)
    }
}

impl<'a, A, D> SplitAt for ArrayViewMut<'a, A, D>
where D: Dimension
{
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}

impl<A, D> SplitAt for RawArrayViewMut<A, D>
where D: Dimension
{
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}
