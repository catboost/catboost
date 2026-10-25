use std::fmt::Debug;

use super::{stride_offset, stride_offset_checked};
use crate::itertools::zip;
use crate::{Dim, Dimension, IntoDimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl};

/// Tuple or fixed size arrays that can be used to index an array.
///
/// ```
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[0, 1],
///                    [2, 3]]);
/// assert_eq!(a[[0, 1]], 1);
/// assert_eq!(a[[1, 1]], 3);
/// a[[1, 1]] += 1;
/// assert_eq!(a[(1, 1)], 4);
/// ```
#[allow(clippy::missing_safety_doc)] // TODO: Add doc
pub unsafe trait NdIndex<E>: Debug
{
    #[doc(hidden)]
    fn index_checked(&self, dim: &E, strides: &E) -> Option<isize>;
    #[doc(hidden)]
    fn index_unchecked(&self, strides: &E) -> isize;
}

unsafe impl<D> NdIndex<D> for D
where D: Dimension
{
    fn index_checked(&self, dim: &D, strides: &D) -> Option<isize>
    {
        dim.stride_offset_checked(strides, self)
    }
    fn index_unchecked(&self, strides: &D) -> isize
    {
        D::stride_offset(self, strides)
    }
}

unsafe impl NdIndex<Ix0> for ()
{
    #[inline]
    fn index_checked(&self, dim: &Ix0, strides: &Ix0) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &Ix0())
    }
    #[inline(always)]
    fn index_unchecked(&self, _strides: &Ix0) -> isize
    {
        0
    }
}

unsafe impl NdIndex<Ix2> for (Ix, Ix)
{
    #[inline]
    fn index_checked(&self, dim: &Ix2, strides: &Ix2) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &Ix2(self.0, self.1))
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix2) -> isize
    {
        stride_offset(self.0, get!(strides, 0)) + stride_offset(self.1, get!(strides, 1))
    }
}
unsafe impl NdIndex<Ix3> for (Ix, Ix, Ix)
{
    #[inline]
    fn index_checked(&self, dim: &Ix3, strides: &Ix3) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }

    #[inline]
    fn index_unchecked(&self, strides: &Ix3) -> isize
    {
        stride_offset(self.0, get!(strides, 0))
            + stride_offset(self.1, get!(strides, 1))
            + stride_offset(self.2, get!(strides, 2))
    }
}

unsafe impl NdIndex<Ix4> for (Ix, Ix, Ix, Ix)
{
    #[inline]
    fn index_checked(&self, dim: &Ix4, strides: &Ix4) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix4) -> isize
    {
        zip(strides.ix(), self.into_dimension().ix())
            .map(|(&s, &i)| stride_offset(i, s))
            .sum()
    }
}
unsafe impl NdIndex<Ix5> for (Ix, Ix, Ix, Ix, Ix)
{
    #[inline]
    fn index_checked(&self, dim: &Ix5, strides: &Ix5) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix5) -> isize
    {
        zip(strides.ix(), self.into_dimension().ix())
            .map(|(&s, &i)| stride_offset(i, s))
            .sum()
    }
}

unsafe impl NdIndex<Ix6> for (Ix, Ix, Ix, Ix, Ix, Ix)
{
    #[inline]
    fn index_checked(&self, dim: &Ix6, strides: &Ix6) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix6) -> isize
    {
        zip(strides.ix(), self.into_dimension().ix())
            .map(|(&s, &i)| stride_offset(i, s))
            .sum()
    }
}

unsafe impl NdIndex<Ix1> for Ix
{
    #[inline]
    fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize>
    {
        dim.stride_offset_checked(strides, &Ix1(*self))
    }
    #[inline(always)]
    fn index_unchecked(&self, strides: &Ix1) -> isize
    {
        stride_offset(*self, get!(strides, 0))
    }
}

unsafe impl NdIndex<IxDyn> for Ix
{
    #[inline]
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize>
    {
        debug_assert_eq!(dim.ndim(), 1);
        stride_offset_checked(dim.ix(), strides.ix(), &[*self])
    }
    #[inline(always)]
    fn index_unchecked(&self, strides: &IxDyn) -> isize
    {
        debug_assert_eq!(strides.ndim(), 1);
        stride_offset(*self, get!(strides, 0))
    }
}

macro_rules! ndindex_with_array {
    ($([$n:expr, $ix_n:ident $($index:tt)*])+) => {
        $(
        // implement NdIndex<Ix2> for [Ix; 2] and so on
        unsafe impl NdIndex<$ix_n> for [Ix; $n] {
            #[inline]
            fn index_checked(&self, dim: &$ix_n, strides: &$ix_n) -> Option<isize> {
                dim.stride_offset_checked(strides, &self.into_dimension())
            }

            #[inline]
            fn index_unchecked(&self, _strides: &$ix_n) -> isize {
                $(
                stride_offset(self[$index], get!(_strides, $index)) +
                )*
                0
            }
        }
        )+
    };
}

ndindex_with_array! {
    [0, Ix0]
    [1, Ix1 0]
    [2, Ix2 0 1]
    [3, Ix3 0 1 2]
    [4, Ix4 0 1 2 3]
    [5, Ix5 0 1 2 3 4]
    [6, Ix6 0 1 2 3 4 5]
}

// implement NdIndex<IxDyn> for Dim<[Ix; 2]> and so on
unsafe impl<const N: usize> NdIndex<IxDyn> for Dim<[Ix; N]>
{
    #[inline]
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize>
    {
        debug_assert_eq!(
            strides.ndim(),
            N,
            "Attempted to index with {:?} in array with {} axes",
            self,
            strides.ndim()
        );
        stride_offset_checked(dim.ix(), strides.ix(), self.ix())
    }

    #[inline]
    fn index_unchecked(&self, strides: &IxDyn) -> isize
    {
        debug_assert_eq!(
            strides.ndim(),
            N,
            "Attempted to index with {:?} in array with {} axes",
            self,
            strides.ndim()
        );
        (0..N)
            .map(|i| stride_offset(get!(self, i), get!(strides, i)))
            .sum()
    }
}

// implement NdIndex<IxDyn> for [Ix; 2] and so on
unsafe impl<const N: usize> NdIndex<IxDyn> for [Ix; N]
{
    #[inline]
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize>
    {
        debug_assert_eq!(
            strides.ndim(),
            N,
            "Attempted to index with {:?} in array with {} axes",
            self,
            strides.ndim()
        );
        stride_offset_checked(dim.ix(), strides.ix(), self)
    }

    #[inline]
    fn index_unchecked(&self, strides: &IxDyn) -> isize
    {
        debug_assert_eq!(
            strides.ndim(),
            N,
            "Attempted to index with {:?} in array with {} axes",
            self,
            strides.ndim()
        );
        (0..N)
            .map(|i| stride_offset(self[i], get!(strides, i)))
            .sum()
    }
}

impl<'a> IntoDimension for &'a [Ix]
{
    type Dim = IxDyn;
    fn into_dimension(self) -> Self::Dim
    {
        Dim(IxDynImpl::from(self))
    }
}

unsafe impl<'a> NdIndex<IxDyn> for &'a IxDyn
{
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize>
    {
        (**self).index_checked(dim, strides)
    }
    fn index_unchecked(&self, strides: &IxDyn) -> isize
    {
        (**self).index_unchecked(strides)
    }
}

unsafe impl<'a> NdIndex<IxDyn> for &'a [Ix]
{
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize>
    {
        stride_offset_checked(dim.ix(), strides.ix(), self)
    }
    fn index_unchecked(&self, strides: &IxDyn) -> isize
    {
        zip(strides.ix(), *self)
            .map(|(&s, &i)| stride_offset(i, s))
            .sum()
    }
}
