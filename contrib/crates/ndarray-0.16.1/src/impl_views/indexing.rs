// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::arraytraits::array_out_of_bounds;
use crate::imp_prelude::*;
use crate::NdIndex;

/// Extra indexing methods for array views
///
/// These methods are very similar to regular indexing or calling of the
/// `get`/`get_mut` methods that we can use on any array or array view. The
/// difference here is in the length of lifetime in the resulting reference.
///
/// **Note** that the `ArrayView` (read-only) and `ArrayViewMut` (read-write) differ
/// in how they are allowed implement this trait -- `ArrayView`'s implementation
/// is usual. If you put in a `ArrayView<'a, T, D>` here, you get references
/// `&'a T` out.
///
/// For `ArrayViewMut` to obey the borrowing rules we have to consume the
/// view if we call any of these methods. (The equivalent of reborrow is
/// `.view_mut()` for read-write array views, but if you can use that,
/// then the regular indexing / `get_mut` should suffice, too.)
///
/// ```
/// use ndarray::IndexLonger;
/// use ndarray::ArrayView;
///
/// let data = [0.; 256];
/// let long_life_ref = {
///     // make a 16 × 16 array view
///     let view = ArrayView::from(&data[..]).into_shape_with_order((16, 16)).unwrap();
///
///     // index the view and with `IndexLonger`.
///     // Note here that we get a reference with a life that is derived from
///     // `data`, the base data, instead of being derived from the view
///     IndexLonger::index(&view, [0, 1])
/// };
///
/// // view goes out of scope
///
/// assert_eq!(long_life_ref, &0.);
///
/// ```
pub trait IndexLonger<I>
{
    /// The type of the reference to the element that is produced, including
    /// its lifetime.
    type Output;
    /// Get a reference of a element through the view.
    ///
    /// This method is like `Index::index` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::get
    ///
    /// **Panics** if index is out of bounds.
    #[track_caller]
    fn index(self, index: I) -> Self::Output;

    /// Get a reference of a element through the view.
    ///
    /// This method is like `ArrayBase::get` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] (and [`get_mut`][2]) which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::get
    /// [2]: ArrayBase::get_mut
    ///
    /// **Panics** if index is out of bounds.
    #[track_caller]
    fn get(self, index: I) -> Option<Self::Output>;

    /// Get a reference of a element through the view without boundary check
    ///
    /// This method is like `elem` with a longer lifetime (matching the array
    /// view); which we can't do for general arrays.
    ///
    /// See also [the `uget` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::uget
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is in-bounds.
    unsafe fn uget(self, index: I) -> Self::Output;
}

impl<'a, 'b, I, A, D> IndexLonger<I> for &'b ArrayView<'a, A, D>
where
    I: NdIndex<D>,
    D: Dimension,
{
    type Output = &'a A;

    /// Get a reference of a element through the view.
    ///
    /// This method is like `Index::index` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::get
    ///
    /// **Panics** if index is out of bounds.
    #[track_caller]
    fn index(self, index: I) -> &'a A
    {
        debug_bounds_check!(self, index);
        unsafe { &*self.get_ptr(index).unwrap_or_else(|| array_out_of_bounds()) }
    }

    fn get(self, index: I) -> Option<&'a A>
    {
        unsafe { self.get_ptr(index).map(|ptr| &*ptr) }
    }

    /// Get a reference of a element through the view without boundary check
    ///
    /// This method is like `elem` with a longer lifetime (matching the array
    /// view); which we can't do for general arrays.
    ///
    /// See also [the `uget` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::uget
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    unsafe fn uget(self, index: I) -> &'a A
    {
        debug_bounds_check!(self, index);
        &*self.as_ptr().offset(index.index_unchecked(&self.strides))
    }
}

impl<'a, I, A, D> IndexLonger<I> for ArrayViewMut<'a, A, D>
where
    I: NdIndex<D>,
    D: Dimension,
{
    type Output = &'a mut A;

    /// Convert a mutable array view to a mutable reference of a element.
    ///
    /// This method is like `IndexMut::index_mut` but with a longer lifetime
    /// (matching the array view); which we can only do for the array view and
    /// not in the `Index` trait.
    ///
    /// See also [the `get_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::get_mut
    ///
    /// **Panics** if index is out of bounds.
    #[track_caller]
    fn index(mut self, index: I) -> &'a mut A
    {
        debug_bounds_check!(self, index);
        unsafe {
            match self.get_mut_ptr(index) {
                Some(ptr) => &mut *ptr,
                None => array_out_of_bounds(),
            }
        }
    }

    /// Convert a mutable array view to a mutable reference of a element, with
    /// checked access.
    ///
    /// See also [the `get_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::get_mut
    ///
    fn get(mut self, index: I) -> Option<&'a mut A>
    {
        debug_bounds_check!(self, index);
        unsafe {
            match self.get_mut_ptr(index) {
                Some(ptr) => Some(&mut *ptr),
                None => None,
            }
        }
    }

    /// Convert a mutable array view to a mutable reference of a element without
    /// boundary check.
    ///
    /// See also [the `uget_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: ArrayBase::uget_mut
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    unsafe fn uget(mut self, index: I) -> &'a mut A
    {
        debug_bounds_check!(self, index);
        &mut *self
            .as_mut_ptr()
            .offset(index.index_unchecked(&self.strides))
    }
}
