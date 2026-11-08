// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::slice;
#[allow(unused_imports)]
use rawpointer::PointerExt;
use std::mem::MaybeUninit;

use crate::imp_prelude::*;

use crate::{Baseiter, ElementsBase, ElementsBaseMut, Iter, IterMut};

use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
use crate::iter::{self, AxisIter, AxisIterMut};
use crate::math_cell::MathCell;
use crate::IndexLonger;

/// Methods for read-only array views.
impl<'a, A, D> ArrayView<'a, A, D>
where D: Dimension
{
    /// Convert the view into an `ArrayView<'b, A, D>` where `'b` is a lifetime
    /// outlived by `'a'`.
    pub fn reborrow<'b>(self) -> ArrayView<'b, A, D>
    where 'a: 'b
    {
        unsafe { ArrayView::new(self.ptr, self.dim, self.strides) }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    ///
    /// Note that while the method is similar to [`ArrayBase::as_slice()`], this method transfers
    /// the view's lifetime to the slice, so it is a bit more powerful.
    pub fn to_slice(&self) -> Option<&'a [A]>
    {
        if self.is_standard_layout() {
            unsafe { Some(slice::from_raw_parts(self.ptr.as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice, if it is contiguous.
    /// Return `None` otherwise.
    ///
    /// Note that while the method is similar to
    /// [`ArrayBase::as_slice_memory_order()`], this method transfers the view's
    /// lifetime to the slice, so it is a bit more powerful.
    pub fn to_slice_memory_order(&self) -> Option<&'a [A]>
    {
        if self.is_contiguous() {
            let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
            unsafe { Some(slice::from_raw_parts(self.ptr.sub(offset).as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Converts to a raw array view.
    #[inline]
    pub(crate) fn into_raw_view(self) -> RawArrayView<A, D>
    {
        unsafe { RawArrayView::new(self.ptr, self.dim, self.strides) }
    }
}

/// Methods specific to `ArrayView0`.
///
/// ***See also all methods for [`ArrayView`] and [`ArrayBase`]***
impl<'a, A> ArrayView<'a, A, Ix0>
{
    /// Consume the view and return a reference to the single element in the array.
    ///
    /// The lifetime of the returned reference matches the lifetime of the data
    /// the array view was pointing to.
    ///
    /// ```
    /// use ndarray::{arr0, Array0};
    ///
    /// // `Foo` doesn't implement `Clone`.
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct Foo;
    ///
    /// let array: Array0<Foo> = arr0(Foo);
    /// let view = array.view();
    /// let scalar: &Foo = view.into_scalar();
    /// assert_eq!(scalar, &Foo);
    /// ```
    pub fn into_scalar(self) -> &'a A
    {
        self.index(Ix0())
    }
}

/// Methods specific to `ArrayViewMut0`.
///
/// ***See also all methods for [`ArrayViewMut`] and [`ArrayBase`]***
impl<'a, A> ArrayViewMut<'a, A, Ix0>
{
    /// Consume the mutable view and return a mutable reference to the single element in the array.
    ///
    /// The lifetime of the returned reference matches the lifetime of the data
    /// the array view was pointing to.
    ///
    /// ```
    /// use ndarray::{arr0, Array0};
    ///
    /// let mut array: Array0<f64> = arr0(5.);
    /// let view = array.view_mut();
    /// let scalar = view.into_scalar();
    /// *scalar = 7.;
    /// assert_eq!(scalar, &7.);
    /// assert_eq!(array[()], 7.);
    /// ```
    pub fn into_scalar(self) -> &'a mut A
    {
        self.index(Ix0())
    }
}

/// Methods for read-write array views.
impl<'a, A, D> ArrayViewMut<'a, A, D>
where D: Dimension
{
    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    ///
    /// Note that while this is similar to [`ArrayBase::as_slice_mut()`], this method transfers the
    /// view's lifetime to the slice.
    pub fn into_slice(self) -> Option<&'a mut [A]>
    {
        self.try_into_slice().ok()
    }

    /// Return the array’s data as a slice, if it is contiguous.
    /// Return `None` otherwise.
    ///
    /// Note that while this is similar to
    /// [`ArrayBase::as_slice_memory_order_mut()`], this method transfers the
    /// view's lifetime to the slice.
    pub fn into_slice_memory_order(self) -> Option<&'a mut [A]>
    {
        self.try_into_slice_memory_order().ok()
    }

    /// Return a shared view of the array with elements as if they were embedded in cells.
    ///
    /// The cell view itself can be copied and accessed without exclusivity.
    ///
    /// The view acts "as if" the elements are temporarily in cells, and elements
    /// can be changed through shared references using the regular cell methods.
    pub fn into_cell_view(self) -> ArrayView<'a, MathCell<A>, D>
    {
        // safety: valid because
        // A and MathCell<A> have the same representation
        // &'a mut T is interchangeable with &'a Cell<T> -- see method Cell::from_mut in std
        unsafe {
            self.into_raw_view_mut()
                .cast::<MathCell<A>>()
                .deref_into_view()
        }
    }

    /// Return the array view as a view of `MaybeUninit<A>` elements
    ///
    /// This conversion leaves the elements as they were (presumably initialized), but
    /// they are represented with the `MaybeUninit<A>` type. Effectively this means that
    /// the elements can be overwritten without dropping the old element in its place.
    /// (In some situations this is not what you want, while for `Copy` elements it makes
    /// no difference at all.)
    ///
    /// # Safety
    ///
    /// This method allows writing uninitialized data into the view, which could leave any
    /// original array that we borrow from in an inconsistent state. This is not allowed
    /// when using the resulting array view.
    pub(crate) unsafe fn into_maybe_uninit(self) -> ArrayViewMut<'a, MaybeUninit<A>, D>
    {
        // Safe because: A and MaybeUninit<A> have the same representation;
        // and we can go from initialized to (maybe) not unconditionally in terms of
        // representation. However, the user must be careful to not write uninit elements
        // through the view.
        self.into_raw_view_mut()
            .cast::<MaybeUninit<A>>()
            .deref_into_view_mut()
    }
}

/// Private raw array view methods
impl<A, D> RawArrayView<A, D>
where D: Dimension
{
    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D>
    {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }
}

impl<A, D> RawArrayViewMut<A, D>
where D: Dimension
{
    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D>
    {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }
}

/// Private array view methods
impl<'a, A, D> ArrayView<'a, A, D>
where D: Dimension
{
    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D>
    {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBase<'a, A, D>
    {
        ElementsBase::new(self)
    }

    pub(crate) fn into_iter_(self) -> Iter<'a, A, D>
    {
        Iter::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note = "This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIter<'a, A, D::Smaller>
    where D: RemoveAxis
    {
        AxisIter::new(self, Axis(0))
    }
}

impl<'a, A, D> ArrayViewMut<'a, A, D>
where D: Dimension
{
    // Convert into a read-only view
    pub(crate) fn into_view(self) -> ArrayView<'a, A, D>
    {
        unsafe { ArrayView::new(self.ptr, self.dim, self.strides) }
    }

    /// Converts to a mutable raw array view.
    pub(crate) fn into_raw_view_mut(self) -> RawArrayViewMut<A, D>
    {
        unsafe { RawArrayViewMut::new(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D>
    {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBaseMut<'a, A, D>
    {
        ElementsBaseMut::new(self)
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Otherwise return self in the Err branch of the result.
    pub(crate) fn try_into_slice(self) -> Result<&'a mut [A], Self>
    {
        if self.is_standard_layout() {
            unsafe { Ok(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len())) }
        } else {
            Err(self)
        }
    }

    /// Return the array’s data as a slice, if it is contiguous.
    /// Otherwise return self in the Err branch of the result.
    fn try_into_slice_memory_order(self) -> Result<&'a mut [A], Self>
    {
        if self.is_contiguous() {
            let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
            unsafe { Ok(slice::from_raw_parts_mut(self.ptr.sub(offset).as_ptr(), self.len())) }
        } else {
            Err(self)
        }
    }

    pub(crate) fn into_iter_(self) -> IterMut<'a, A, D>
    {
        IterMut::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note = "This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIterMut<'a, A, D::Smaller>
    where D: RemoveAxis
    {
        AxisIterMut::new(self, Axis(0))
    }
}
