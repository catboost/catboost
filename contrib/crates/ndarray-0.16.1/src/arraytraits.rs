// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::hash;
use std::mem;
use std::mem::size_of;
use std::ops::{Index, IndexMut};
use std::{iter::FromIterator, slice};

use crate::imp_prelude::*;
use crate::Arc;

use crate::{
    dimension,
    iter::{Iter, IterMut},
    numeric_util,
    FoldWhile,
    NdIndex,
    OwnedArcRepr,
    Zip,
};

#[cold]
#[inline(never)]
pub(crate) fn array_out_of_bounds() -> !
{
    panic!("ndarray: index out of bounds");
}

#[inline(always)]
pub fn debug_bounds_check<S, D, I>(_a: &ArrayBase<S, D>, _index: &I)
where
    D: Dimension,
    I: NdIndex<D>,
    S: Data,
{
    debug_bounds_check!(_a, *_index);
}

/// Access the element at **index**.
///
/// **Panics** if index is out of bounds.
impl<S, D, I> Index<I> for ArrayBase<S, D>
where
    D: Dimension,
    I: NdIndex<D>,
    S: Data,
{
    type Output = S::Elem;
    #[inline]
    fn index(&self, index: I) -> &S::Elem
    {
        debug_bounds_check!(self, index);
        unsafe {
            &*self.ptr.as_ptr().offset(
                index
                    .index_checked(&self.dim, &self.strides)
                    .unwrap_or_else(|| array_out_of_bounds()),
            )
        }
    }
}

/// Access the element at **index** mutably.
///
/// **Panics** if index is out of bounds.
impl<S, D, I> IndexMut<I> for ArrayBase<S, D>
where
    D: Dimension,
    I: NdIndex<D>,
    S: DataMut,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut S::Elem
    {
        debug_bounds_check!(self, index);
        unsafe {
            &mut *self.as_mut_ptr().offset(
                index
                    .index_checked(&self.dim, &self.strides)
                    .unwrap_or_else(|| array_out_of_bounds()),
            )
        }
    }
}

/// Return `true` if the array shapes and all elements of `self` and
/// `rhs` are equal. Return `false` otherwise.
impl<A, B, S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
where
    A: PartialEq<B>,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D: Dimension,
{
    fn eq(&self, rhs: &ArrayBase<S2, D>) -> bool
    {
        if self.shape() != rhs.shape() {
            return false;
        }
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_eq(self_s, rhs_s);
            }
        }
        Zip::from(self)
            .and(rhs)
            .fold_while(true, |_, a, b| {
                if a != b {
                    FoldWhile::Done(false)
                } else {
                    FoldWhile::Continue(true)
                }
            })
            .into_inner()
    }
}

/// Return `true` if the array shapes and all elements of `self` and
/// `rhs` are equal. Return `false` otherwise.
#[allow(clippy::unconditional_recursion)] // false positive
impl<'a, A, B, S, S2, D> PartialEq<&'a ArrayBase<S2, D>> for ArrayBase<S, D>
where
    A: PartialEq<B>,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D: Dimension,
{
    fn eq(&self, rhs: &&ArrayBase<S2, D>) -> bool
    {
        *self == **rhs
    }
}

/// Return `true` if the array shapes and all elements of `self` and
/// `rhs` are equal. Return `false` otherwise.
#[allow(clippy::unconditional_recursion)] // false positive
impl<'a, A, B, S, S2, D> PartialEq<ArrayBase<S2, D>> for &'a ArrayBase<S, D>
where
    A: PartialEq<B>,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D: Dimension,
{
    fn eq(&self, rhs: &ArrayBase<S2, D>) -> bool
    {
        **self == *rhs
    }
}

impl<S, D> Eq for ArrayBase<S, D>
where
    D: Dimension,
    S: Data,
    S::Elem: Eq,
{
}

impl<A, S> From<Box<[A]>> for ArrayBase<S, Ix1>
where S: DataOwned<Elem = A>
{
    /// Create a one-dimensional array from a boxed slice (no copying needed).
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    fn from(b: Box<[A]>) -> Self
    {
        Self::from_vec(b.into_vec())
    }
}

impl<A, S> From<Vec<A>> for ArrayBase<S, Ix1>
where S: DataOwned<Elem = A>
{
    /// Create a one-dimensional array from a vector (no copying needed).
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::Array;
    ///
    /// let array = Array::from(vec![1., 2., 3., 4.]);
    /// ```
    fn from(v: Vec<A>) -> Self
    {
        Self::from_vec(v)
    }
}

impl<A, S> FromIterator<A> for ArrayBase<S, Ix1>
where S: DataOwned<Elem = A>
{
    /// Create a one-dimensional array from an iterable.
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::{Array, arr1};
    ///
    /// // Either use `from_iter` directly or use `Iterator::collect`.
    /// let array = Array::from_iter((0..5).map(|x| x * x));
    /// assert!(array == arr1(&[0, 1, 4, 9, 16]))
    /// ```
    fn from_iter<I>(iterable: I) -> ArrayBase<S, Ix1>
    where I: IntoIterator<Item = A>
    {
        Self::from_iter(iterable)
    }
}

impl<'a, S, D> IntoIterator for &'a ArrayBase<S, D>
where
    D: Dimension,
    S: Data,
{
    type Item = &'a S::Elem;
    type IntoIter = Iter<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter()
    }
}

impl<'a, S, D> IntoIterator for &'a mut ArrayBase<S, D>
where
    D: Dimension,
    S: DataMut,
{
    type Item = &'a mut S::Elem;
    type IntoIter = IterMut<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter_mut()
    }
}

impl<'a, A, D> IntoIterator for ArrayView<'a, A, D>
where D: Dimension
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.into_iter_()
    }
}

impl<'a, A, D> IntoIterator for ArrayViewMut<'a, A, D>
where D: Dimension
{
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.into_iter_()
    }
}

impl<S, D> hash::Hash for ArrayBase<S, D>
where
    D: Dimension,
    S: Data,
    S::Elem: hash::Hash,
{
    // Note: elements are hashed in the logical order
    fn hash<H: hash::Hasher>(&self, state: &mut H)
    {
        self.shape().hash(state);
        if let Some(self_s) = self.as_slice() {
            hash::Hash::hash_slice(self_s, state);
        } else {
            for row in self.rows() {
                if let Some(row_s) = row.as_slice() {
                    hash::Hash::hash_slice(row_s, state);
                } else {
                    for elt in row {
                        elt.hash(state)
                    }
                }
            }
        }
    }
}

// NOTE: ArrayBase keeps an internal raw pointer that always
// points into the storage. This is Sync & Send as long as we
// follow the usual inherited mutability rules, as we do with
// Vec, &[] and &mut []

/// `ArrayBase` is `Sync` when the storage type is.
unsafe impl<S, D> Sync for ArrayBase<S, D>
where
    S: Sync + Data,
    D: Sync,
{
}

/// `ArrayBase` is `Send` when the storage type is.
unsafe impl<S, D> Send for ArrayBase<S, D>
where
    S: Send + Data,
    D: Send,
{
}

#[cfg(feature = "serde")]
// Use version number so we can add a packed format later.
pub const ARRAY_FORMAT_VERSION: u8 = 1u8;

// use "raw" form instead of type aliases here so that they show up in docs
/// Implementation of `ArrayView::from(&S)` where `S` is a slice or sliceable.
///
/// **Panics** if the length of the slice overflows `isize`. (This can only
/// occur if `A` is zero-sized, because slices cannot contain more than
/// `isize::MAX` number of bytes.)
impl<'a, A, Slice: ?Sized> From<&'a Slice> for ArrayView<'a, A, Ix1>
where Slice: AsRef<[A]>
{
    /// Create a one-dimensional read-only array view of the data in `slice`.
    ///
    /// **Panics** if the slice length is greater than `isize::MAX`.
    fn from(slice: &'a Slice) -> Self
    {
        aview1(slice.as_ref())
    }
}

/// Implementation of ArrayView2::from(&[[A; N]; M])
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This can only occur if A
/// is zero-sized because slices cannot contain more than `isize::MAX` number of bytes).
/// **Panics** if N == 0 and the number of rows is greater than isize::MAX.
impl<'a, A, const M: usize, const N: usize> From<&'a [[A; N]; M]> for ArrayView<'a, A, Ix2>
{
    /// Create a two-dimensional read-only array view of the data in `slice`
    fn from(xs: &'a [[A; N]; M]) -> Self
    {
        Self::from(&xs[..])
    }
}

/// Implementation of ArrayView2::from(&[[A; N]])
///
/// **Panics** if the product of non-zero axis lengths overflows `isize`. (This
/// can only occur if A is zero-sized or if `N` is zero, because slices cannot
/// contain more than `isize::MAX` number of bytes.)
impl<'a, A, const N: usize> From<&'a [[A; N]]> for ArrayView<'a, A, Ix2>
{
    /// Create a two-dimensional read-only array view of the data in `slice`
    fn from(xs: &'a [[A; N]]) -> Self
    {
        aview2(xs)
    }
}

/// Implementation of `ArrayView::from(&A)` where `A` is an array.
impl<'a, A, S, D> From<&'a ArrayBase<S, D>> for ArrayView<'a, A, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Create a read-only array view of the array.
    fn from(array: &'a ArrayBase<S, D>) -> Self
    {
        array.view()
    }
}

/// Implementation of `ArrayViewMut::from(&mut S)` where `S` is a slice or sliceable.
impl<'a, A, Slice: ?Sized> From<&'a mut Slice> for ArrayViewMut<'a, A, Ix1>
where Slice: AsMut<[A]>
{
    /// Create a one-dimensional read-write array view of the data in `slice`.
    ///
    /// **Panics** if the slice length is greater than `isize::MAX`.
    fn from(slice: &'a mut Slice) -> Self
    {
        let xs = slice.as_mut();
        if mem::size_of::<A>() == 0 {
            assert!(
                xs.len() <= isize::MAX as usize,
                "Slice length must fit in `isize`.",
            );
        }
        unsafe { Self::from_shape_ptr(xs.len(), xs.as_mut_ptr()) }
    }
}

/// Implementation of ArrayViewMut2::from(&mut [[A; N]; M])
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This can only occur if A
/// is zero-sized because slices cannot contain more than `isize::MAX` number of bytes).
/// **Panics** if N == 0 and the number of rows is greater than isize::MAX.
impl<'a, A, const M: usize, const N: usize> From<&'a mut [[A; N]; M]> for ArrayViewMut<'a, A, Ix2>
{
    /// Create a two-dimensional read-write array view of the data in `slice`
    fn from(xs: &'a mut [[A; N]; M]) -> Self
    {
        Self::from(&mut xs[..])
    }
}

/// Implementation of ArrayViewMut2::from(&mut [[A; N]])
///
/// **Panics** if the product of non-zero axis lengths overflows `isize`. (This
/// can only occur if `A` is zero-sized or if `N` is zero, because slices
/// cannot contain more than `isize::MAX` number of bytes.)
impl<'a, A, const N: usize> From<&'a mut [[A; N]]> for ArrayViewMut<'a, A, Ix2>
{
    /// Create a two-dimensional read-write array view of the data in `slice`
    fn from(xs: &'a mut [[A; N]]) -> Self
    {
        let cols = N;
        let rows = xs.len();
        let dim = Ix2(rows, cols);
        if size_of::<A>() == 0 {
            dimension::size_of_shape_checked(&dim).expect("Product of non-zero axis lengths must not overflow isize.");
        } else if N == 0 {
            assert!(
                xs.len() <= isize::MAX as usize,
                "Product of non-zero axis lengths must not overflow isize.",
            );
        }

        // `cols * rows` is guaranteed to fit in `isize` because we checked that it fits in
        // `isize::MAX`
        unsafe {
            let data = slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut A, cols * rows);
            ArrayViewMut::from_shape_ptr(dim, data.as_mut_ptr())
        }
    }
}

/// Implementation of `ArrayViewMut::from(&mut A)` where `A` is an array.
impl<'a, A, S, D> From<&'a mut ArrayBase<S, D>> for ArrayViewMut<'a, A, D>
where
    S: DataMut<Elem = A>,
    D: Dimension,
{
    /// Create a read-write array view of the array.
    fn from(array: &'a mut ArrayBase<S, D>) -> Self
    {
        array.view_mut()
    }
}

impl<A, D> From<Array<A, D>> for ArcArray<A, D>
where D: Dimension
{
    fn from(arr: Array<A, D>) -> ArcArray<A, D>
    {
        let data = OwnedArcRepr(Arc::new(arr.data));
        // safe because: equivalent unmoved data, ptr and dims remain valid
        unsafe { ArrayBase::from_data_ptr(data, arr.ptr).with_strides_dim(arr.strides, arr.dim) }
    }
}

/// Argument conversion into an array view
///
/// The trait is parameterized over `A`, the element type, and `D`, the
/// dimensionality of the array. `D` defaults to one-dimensional.
///
/// Use `.into()` to do the conversion.
///
/// ```
/// use ndarray::AsArray;
///
/// fn sum<'a, V: AsArray<'a, f64>>(data: V) -> f64 {
///     let array_view = data.into();
///     array_view.sum()
/// }
///
/// assert_eq!(
///     sum(&[1., 2., 3.]),
///     6.
/// );
///
/// ```
pub trait AsArray<'a, A: 'a, D = Ix1>: Into<ArrayView<'a, A, D>>
where D: Dimension
{
}
impl<'a, A: 'a, D, T> AsArray<'a, A, D> for T
where
    T: Into<ArrayView<'a, A, D>>,
    D: Dimension,
{
}

/// Create an owned array with a default state.
///
/// The array is created with dimension `D::default()`, which results
/// in for example dimensions `0` and `(0, 0)` with zero elements for the
/// one-dimensional and two-dimensional cases respectively.
///
/// The default dimension for `IxDyn` is `IxDyn(&[0])` (array has zero
/// elements). And the default for the dimension `()` is `()` (array has
/// one element).
///
/// Since arrays cannot grow, the intention is to use the default value as
/// placeholder.
impl<A, S, D> Default for ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
    A: Default,
{
    // NOTE: We can implement Default for non-zero dimensional array views by
    // using an empty slice, however we need a trait for nonzero Dimension.
    fn default() -> Self
    {
        ArrayBase::default(D::default())
    }
}
