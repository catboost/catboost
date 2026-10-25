use std::marker::PhantomData;

use super::LanesIter;
use super::LanesIterMut;
use crate::imp_prelude::*;
use crate::{Layout, NdProducer};

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Lanes {
        base,
        inner_len,
        inner_stride,
    }
    Lanes<'a, A, D> {
        type Item = ArrayView<'a, A, Ix1>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

/// See [`.lanes()`](ArrayBase::lanes)
/// for more information.
pub struct Lanes<'a, A, D>
{
    base: ArrayView<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}

impl<'a, A, D: Dimension> Lanes<'a, A, D>
{
    pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
    where Di: Dimension<Smaller = D>
    {
        let ndim = v.ndim();
        let len;
        let stride;
        let iter_v = if ndim == 0 {
            len = 1;
            stride = 1;
            v.try_remove_axis(Axis(0))
        } else {
            let i = axis.index();
            len = v.dim[i];
            stride = v.strides[i] as isize;
            v.try_remove_axis(axis)
        };
        Lanes {
            inner_len: len,
            inner_stride: stride,
            base: iter_v,
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone =>]
    LanesMut {
        base,
        inner_len,
        inner_stride,
    }
    LanesMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, Ix1>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

impl<'a, A, D> IntoIterator for Lanes<'a, A, D>
where D: Dimension
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LanesIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter
    {
        LanesIter {
            iter: self.base.into_base_iter(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
            life: PhantomData,
        }
    }
}

/// See [`.lanes_mut()`](ArrayBase::lanes_mut)
/// for more information.
pub struct LanesMut<'a, A, D>
{
    base: ArrayViewMut<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}

impl<'a, A, D: Dimension> LanesMut<'a, A, D>
{
    pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
    where Di: Dimension<Smaller = D>
    {
        let ndim = v.ndim();
        let len;
        let stride;
        let iter_v = if ndim == 0 {
            len = 1;
            stride = 1;
            v.try_remove_axis(Axis(0))
        } else {
            let i = axis.index();
            len = v.dim[i];
            stride = v.strides[i] as isize;
            v.try_remove_axis(axis)
        };
        LanesMut {
            inner_len: len,
            inner_stride: stride,
            base: iter_v,
        }
    }
}

impl<'a, A, D> IntoIterator for LanesMut<'a, A, D>
where D: Dimension
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LanesIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter
    {
        LanesIterMut {
            iter: self.base.into_base_iter(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
            life: PhantomData,
        }
    }
}
