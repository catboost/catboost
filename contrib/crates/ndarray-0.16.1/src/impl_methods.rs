// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::slice;
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[allow(unused_imports)]
use rawpointer::PointerExt;
use std::mem::{size_of, ManuallyDrop};

use crate::imp_prelude::*;

use crate::argument_traits::AssignElem;
use crate::dimension;
use crate::dimension::broadcast::co_broadcast;
use crate::dimension::reshape_dim;
use crate::dimension::IntoDimension;
use crate::dimension::{
    abs_index,
    axes_of,
    do_slice,
    merge_axes,
    move_min_stride_axis_to_last,
    offset_from_low_addr_ptr_to_logical_ptr,
    size_of_shape_checked,
    stride_offset,
    Axes,
};
use crate::error::{self, from_kind, ErrorKind, ShapeError};
use crate::itertools::zip;
use crate::math_cell::MathCell;
use crate::order::Order;
use crate::shape_builder::ShapeArg;
use crate::zip::{IntoNdProducer, Zip};
use crate::AxisDescription;
use crate::{arraytraits, DimMax};

use crate::iter::{
    AxisChunksIter,
    AxisChunksIterMut,
    AxisIter,
    AxisIterMut,
    AxisWindows,
    ExactChunks,
    ExactChunksMut,
    IndexedIter,
    IndexedIterMut,
    Iter,
    IterMut,
    Lanes,
    LanesMut,
    Windows,
};
use crate::slice::{MultiSliceArg, SliceArg};
use crate::stacking::concatenate;
use crate::{NdIndex, Slice, SliceInfoElem};

/// # Methods For All Array Types
impl<A, S, D> ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    /// Return the total number of elements in the array.
    pub fn len(&self) -> usize
    {
        self.dim.size()
    }

    /// Return the length of `axis`.
    ///
    /// The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the
    /// number of dimensions (axes) of the array.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn len_of(&self, axis: Axis) -> usize
    {
        self.dim[axis.index()]
    }

    /// Return whether the array has any elements
    pub fn is_empty(&self) -> bool
    {
        self.len() == 0
    }

    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> usize
    {
        self.dim.ndim()
    }

    /// Return the shape of the array in its “pattern” form,
    /// an integer in the one-dimensional case, tuple in the n-dimensional cases
    /// and so on.
    pub fn dim(&self) -> D::Pattern
    {
        self.dim.clone().into_pattern()
    }

    /// Return the shape of the array as it's stored in the array.
    ///
    /// This is primarily useful for passing to other `ArrayBase`
    /// functions, such as when creating another array of the same
    /// shape and dimensionality.
    ///
    /// ```
    /// use ndarray::Array;
    ///
    /// let a = Array::from_elem((2, 3), 5.);
    ///
    /// // Create an array of zeros that's the same shape and dimensionality as `a`.
    /// let b = Array::<f64, _>::zeros(a.raw_dim());
    /// ```
    pub fn raw_dim(&self) -> D
    {
        self.dim.clone()
    }

    /// Return the shape of the array as a slice.
    ///
    /// Note that you probably don't want to use this to create an array of the
    /// same shape as another array because creating an array with e.g.
    /// [`Array::zeros()`](ArrayBase::zeros) using a shape of type `&[usize]`
    /// results in a dynamic-dimensional array. If you want to create an array
    /// that has the same shape and dimensionality as another array, use
    /// [`.raw_dim()`](ArrayBase::raw_dim) instead:
    ///
    /// ```rust
    /// use ndarray::{Array, Array2};
    ///
    /// let a = Array2::<i32>::zeros((3, 4));
    /// let shape = a.shape();
    /// assert_eq!(shape, &[3, 4]);
    ///
    /// // Since `a.shape()` returned `&[usize]`, we get an `ArrayD` instance:
    /// let b = Array::zeros(shape);
    /// assert_eq!(a.clone().into_dyn(), b);
    ///
    /// // To get the same dimension type, use `.raw_dim()` instead:
    /// let c = Array::zeros(a.raw_dim());
    /// assert_eq!(a, c);
    /// ```
    pub fn shape(&self) -> &[usize]
    {
        self.dim.slice()
    }

    /// Return the strides of the array as a slice.
    pub fn strides(&self) -> &[isize]
    {
        let s = self.strides.slice();
        // reinterpret unsigned integer as signed
        unsafe { slice::from_raw_parts(s.as_ptr() as *const _, s.len()) }
    }

    /// Return the stride of `axis`.
    ///
    /// The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the
    /// number of dimensions (axes) of the array.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn stride_of(&self, axis: Axis) -> isize
    {
        // strides are reinterpreted as isize
        self.strides[axis.index()] as isize
    }

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrayView<'_, A, D>
    where S: Data
    {
        debug_assert!(self.pointer_is_inbounds());
        unsafe { ArrayView::new(self.ptr, self.dim.clone(), self.strides.clone()) }
    }

    /// Return a read-write view of the array
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
    where S: DataMut
    {
        self.ensure_unique();
        unsafe { ArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone()) }
    }

    /// Return a shared view of the array with elements as if they were embedded in cells.
    ///
    /// The cell view requires a mutable borrow of the array. Once borrowed the
    /// cell view itself can be copied and accessed without exclusivity.
    ///
    /// The view acts "as if" the elements are temporarily in cells, and elements
    /// can be changed through shared references using the regular cell methods.
    pub fn cell_view(&mut self) -> ArrayView<'_, MathCell<A>, D>
    where S: DataMut
    {
        self.view_mut().into_cell_view()
    }

    /// Return an uniquely owned copy of the array.
    ///
    /// If the input array is contiguous, then the output array will have the same
    /// memory layout. Otherwise, the layout of the output array is unspecified.
    /// If you need a particular layout, you can allocate a new array with the
    /// desired memory layout and [`.assign()`](Self::assign) the data.
    /// Alternatively, you can collectan iterator, like this for a result in
    /// standard layout:
    ///
    /// ```
    /// # use ndarray::prelude::*;
    /// # let arr = Array::from_shape_vec((2, 2).f(), vec![1, 2, 3, 4]).unwrap();
    /// # let owned = {
    /// Array::from_shape_vec(arr.raw_dim(), arr.iter().cloned().collect()).unwrap()
    /// # };
    /// # assert!(owned.is_standard_layout());
    /// # assert_eq!(arr, owned);
    /// ```
    ///
    /// or this for a result in column-major (Fortran) layout:
    ///
    /// ```
    /// # use ndarray::prelude::*;
    /// # let arr = Array::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
    /// # let owned = {
    /// Array::from_shape_vec(arr.raw_dim().f(), arr.t().iter().cloned().collect()).unwrap()
    /// # };
    /// # assert!(owned.t().is_standard_layout());
    /// # assert_eq!(arr, owned);
    /// ```
    pub fn to_owned(&self) -> Array<A, D>
    where
        A: Clone,
        S: Data,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            unsafe { Array::from_shape_vec_unchecked(self.dim.clone().strides(self.strides.clone()), slc.to_vec()) }
        } else {
            self.map(A::clone)
        }
    }

    /// Return a shared ownership (copy on write) array, cloning the array
    /// elements if necessary.
    pub fn to_shared(&self) -> ArcArray<A, D>
    where
        A: Clone,
        S: Data,
    {
        S::to_shared(self)
    }

    /// Turn the array into a uniquely owned array, cloning the array elements
    /// if necessary.
    pub fn into_owned(self) -> Array<A, D>
    where
        A: Clone,
        S: Data,
    {
        S::into_owned(self)
    }

    /// Converts the array into `Array<A, D>` if this is possible without
    /// cloning the array elements. Otherwise, returns `self` unchanged.
    ///
    /// ```
    /// use ndarray::{array, rcarr2, ArcArray2, Array2};
    ///
    /// // Reference-counted, clone-on-write `ArcArray`.
    /// let a: ArcArray2<_> = rcarr2(&[[1., 2.], [3., 4.]]);
    /// {
    ///     // Another reference to the same data.
    ///     let b: ArcArray2<_> = a.clone();
    ///     // Since there are two references to the same data, `.into_owned()`
    ///     // would require cloning the data, so `.try_into_owned_nocopy()`
    ///     // returns `Err`.
    ///     assert!(b.try_into_owned_nocopy().is_err());
    /// }
    /// // Here, since the second reference has been dropped, the `ArcArray`
    /// // can be converted into an `Array` without cloning the data.
    /// let unique: Array2<_> = a.try_into_owned_nocopy().unwrap();
    /// assert_eq!(unique, array![[1., 2.], [3., 4.]]);
    /// ```
    pub fn try_into_owned_nocopy(self) -> Result<Array<A, D>, Self>
    where S: Data
    {
        S::try_into_owned_nocopy(self)
    }

    /// Turn the array into a shared ownership (copy on write) array,
    /// cloning the array elements if necessary.
    ///
    /// If you want to generalize over `Array` and `ArcArray` inputs but avoid
    /// an `A: Clone` bound, use `Into::<ArcArray<A, D>>::into` instead of this
    /// method.
    pub fn into_shared(self) -> ArcArray<A, D>
    where
        A: Clone,
        S: DataOwned,
    {
        S::into_shared(self)
    }

    /// Returns a reference to the first element of the array, or `None` if it
    /// is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::Array3;
    ///
    /// let mut a = Array3::<f64>::zeros([3, 4, 2]);
    /// a[[0, 0, 0]] = 42.;
    /// assert_eq!(a.first(), Some(&42.));
    ///
    /// let b = Array3::<f64>::zeros([3, 0, 5]);
    /// assert_eq!(b.first(), None);
    /// ```
    pub fn first(&self) -> Option<&A>
    where S: Data
    {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &*self.as_ptr() })
        }
    }

    /// Returns a mutable reference to the first element of the array, or
    /// `None` if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::Array3;
    ///
    /// let mut a = Array3::<f64>::zeros([3, 4, 2]);
    /// *a.first_mut().unwrap() = 42.;
    /// assert_eq!(a[[0, 0, 0]], 42.);
    ///
    /// let mut b = Array3::<f64>::zeros([3, 0, 5]);
    /// assert_eq!(b.first_mut(), None);
    /// ```
    pub fn first_mut(&mut self) -> Option<&mut A>
    where S: DataMut
    {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { &mut *self.as_mut_ptr() })
        }
    }

    /// Returns a reference to the last element of the array, or `None` if it
    /// is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::Array3;
    ///
    /// let mut a = Array3::<f64>::zeros([3, 4, 2]);
    /// a[[2, 3, 1]] = 42.;
    /// assert_eq!(a.last(), Some(&42.));
    ///
    /// let b = Array3::<f64>::zeros([3, 0, 5]);
    /// assert_eq!(b.last(), None);
    /// ```
    pub fn last(&self) -> Option<&A>
    where S: Data
    {
        if self.is_empty() {
            None
        } else {
            let mut index = self.raw_dim();
            for ax in 0..index.ndim() {
                index[ax] -= 1;
            }
            Some(unsafe { self.uget(index) })
        }
    }

    /// Returns a mutable reference to the last element of the array, or `None`
    /// if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::Array3;
    ///
    /// let mut a = Array3::<f64>::zeros([3, 4, 2]);
    /// *a.last_mut().unwrap() = 42.;
    /// assert_eq!(a[[2, 3, 1]], 42.);
    ///
    /// let mut b = Array3::<f64>::zeros([3, 0, 5]);
    /// assert_eq!(b.last_mut(), None);
    /// ```
    pub fn last_mut(&mut self) -> Option<&mut A>
    where S: DataMut
    {
        if self.is_empty() {
            None
        } else {
            let mut index = self.raw_dim();
            for ax in 0..index.ndim() {
                index[ax] -= 1;
            }
            Some(unsafe { self.uget_mut(index) })
        }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Elements are visited in the *logical order* of the array, which
    /// is where the rightmost index is varying the fastest.
    ///
    /// Iterator element type is `&A`.
    pub fn iter(&self) -> Iter<'_, A, D>
    where S: Data
    {
        debug_assert!(self.pointer_is_inbounds());
        self.view().into_iter_()
    }

    /// Return an iterator of mutable references to the elements of the array.
    ///
    /// Elements are visited in the *logical order* of the array, which
    /// is where the rightmost index is varying the fastest.
    ///
    /// Iterator element type is `&mut A`.
    pub fn iter_mut(&mut self) -> IterMut<'_, A, D>
    where S: DataMut
    {
        self.view_mut().into_iter_()
    }

    /// Return an iterator of indexes and references to the elements of the array.
    ///
    /// Elements are visited in the *logical order* of the array, which
    /// is where the rightmost index is varying the fastest.
    ///
    /// Iterator element type is `(D::Pattern, &A)`.
    ///
    /// See also [`Zip::indexed`]
    pub fn indexed_iter(&self) -> IndexedIter<'_, A, D>
    where S: Data
    {
        IndexedIter::new(self.view().into_elements_base())
    }

    /// Return an iterator of indexes and mutable references to the elements of the array.
    ///
    /// Elements are visited in the *logical order* of the array, which
    /// is where the rightmost index is varying the fastest.
    ///
    /// Iterator element type is `(D::Pattern, &mut A)`.
    pub fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, A, D>
    where S: DataMut
    {
        IndexedIterMut::new(self.view_mut().into_elements_base())
    }

    /// Return a sliced view of the array.
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`s!`], [`SliceArg`], and [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// (**Panics** if `D` is `IxDyn` and `info` does not match the number of array axes.)
    #[track_caller]
    pub fn slice<I>(&self, info: I) -> ArrayView<'_, A, I::OutDim>
    where
        I: SliceArg<D>,
        S: Data,
    {
        self.view().slice_move(info)
    }

    /// Return a sliced read-write view of the array.
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`s!`], [`SliceArg`], and [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// (**Panics** if `D` is `IxDyn` and `info` does not match the number of array axes.)
    #[track_caller]
    pub fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, A, I::OutDim>
    where
        I: SliceArg<D>,
        S: DataMut,
    {
        self.view_mut().slice_move(info)
    }

    /// Return multiple disjoint, sliced, mutable views of the array.
    ///
    /// See [*Slicing*](#slicing) for full documentation. See also
    /// [`MultiSliceArg`], [`s!`], [`SliceArg`], and
    /// [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** if any of the following occur:
    ///
    /// * if any of the views would intersect (i.e. if any element would appear in multiple slices)
    /// * if an index is out of bounds or step size is zero
    /// * if `D` is `IxDyn` and `info` does not match the number of array axes
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{arr2, s};
    ///
    /// let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);
    /// let (mut edges, mut middle) = a.multi_slice_mut((s![.., ..;2], s![.., 1]));
    /// edges.fill(1);
    /// middle.fill(0);
    /// assert_eq!(a, arr2(&[[1, 0, 1], [1, 0, 1]]));
    /// ```
    #[track_caller]
    pub fn multi_slice_mut<'a, M>(&'a mut self, info: M) -> M::Output
    where
        M: MultiSliceArg<'a, A, D>,
        S: DataMut,
    {
        info.multi_slice_move(self.view_mut())
    }

    /// Slice the array, possibly changing the number of dimensions.
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`s!`], [`SliceArg`], and [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// (**Panics** if `D` is `IxDyn` and `info` does not match the number of array axes.)
    #[track_caller]
    pub fn slice_move<I>(mut self, info: I) -> ArrayBase<S, I::OutDim>
    where I: SliceArg<D>
    {
        assert_eq!(
            info.in_ndim(),
            self.ndim(),
            "The input dimension of `info` must match the array to be sliced.",
        );
        let out_ndim = info.out_ndim();
        let mut new_dim = I::OutDim::zeros(out_ndim);
        let mut new_strides = I::OutDim::zeros(out_ndim);

        let mut old_axis = 0;
        let mut new_axis = 0;
        info.as_ref().iter().for_each(|&ax_info| match ax_info {
            SliceInfoElem::Slice { start, end, step } => {
                // Slice the axis in-place to update the `dim`, `strides`, and `ptr`.
                self.slice_axis_inplace(Axis(old_axis), Slice { start, end, step });
                // Copy the sliced dim and stride to corresponding axis.
                new_dim[new_axis] = self.dim[old_axis];
                new_strides[new_axis] = self.strides[old_axis];
                old_axis += 1;
                new_axis += 1;
            }
            SliceInfoElem::Index(index) => {
                // Collapse the axis in-place to update the `ptr`.
                let i_usize = abs_index(self.len_of(Axis(old_axis)), index);
                self.collapse_axis(Axis(old_axis), i_usize);
                // Skip copying the axis since it should be removed. Note that
                // removing this axis is safe because `.collapse_axis()` panics
                // if the index is out-of-bounds, so it will panic if the axis
                // is zero length.
                old_axis += 1;
            }
            SliceInfoElem::NewAxis => {
                // Set the dim and stride of the new axis.
                new_dim[new_axis] = 1;
                new_strides[new_axis] = 0;
                new_axis += 1;
            }
        });
        debug_assert_eq!(old_axis, self.ndim());
        debug_assert_eq!(new_axis, out_ndim);

        // safe because new dimension, strides allow access to a subset of old data
        unsafe { self.with_strides_dim(new_strides, new_dim) }
    }

    /// Slice the array in place without changing the number of dimensions.
    ///
    /// In particular, if an axis is sliced with an index, the axis is
    /// collapsed, as in [`.collapse_axis()`], rather than removed, as in
    /// [`.slice_move()`] or [`.index_axis_move()`].
    ///
    /// [`.collapse_axis()`]: Self::collapse_axis
    /// [`.slice_move()`]: Self::slice_move
    /// [`.index_axis_move()`]: Self::index_axis_move
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`s!`], [`SliceArg`], and [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** in the following cases:
    ///
    /// - if an index is out of bounds
    /// - if a step size is zero
    /// - if [`SliceInfoElem::NewAxis`] is in `info`, e.g. if [`NewAxis`] was
    ///   used in the [`s!`] macro
    /// - if `D` is `IxDyn` and `info` does not match the number of array axes
    #[track_caller]
    pub fn slice_collapse<I>(&mut self, info: I)
    where I: SliceArg<D>
    {
        assert_eq!(
            info.in_ndim(),
            self.ndim(),
            "The input dimension of `info` must match the array to be sliced.",
        );
        let mut axis = 0;
        info.as_ref().iter().for_each(|&ax_info| match ax_info {
            SliceInfoElem::Slice { start, end, step } => {
                self.slice_axis_inplace(Axis(axis), Slice { start, end, step });
                axis += 1;
            }
            SliceInfoElem::Index(index) => {
                let i_usize = abs_index(self.len_of(Axis(axis)), index);
                self.collapse_axis(Axis(axis), i_usize);
                axis += 1;
            }
            SliceInfoElem::NewAxis => panic!("`slice_collapse` does not support `NewAxis`."),
        });
        debug_assert_eq!(axis, self.ndim());
    }

    /// Return a view of the array, sliced along the specified axis.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    #[must_use = "slice_axis returns an array view with the sliced result"]
    pub fn slice_axis(&self, axis: Axis, indices: Slice) -> ArrayView<'_, A, D>
    where S: Data
    {
        let mut view = self.view();
        view.slice_axis_inplace(axis, indices);
        view
    }

    /// Return a mutable view of the array, sliced along the specified axis.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    #[must_use = "slice_axis_mut returns an array view with the sliced result"]
    pub fn slice_axis_mut(&mut self, axis: Axis, indices: Slice) -> ArrayViewMut<'_, A, D>
    where S: DataMut
    {
        let mut view_mut = self.view_mut();
        view_mut.slice_axis_inplace(axis, indices);
        view_mut
    }

    /// Slice the array in place along the specified axis.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn slice_axis_inplace(&mut self, axis: Axis, indices: Slice)
    {
        let offset =
            do_slice(&mut self.dim.slice_mut()[axis.index()], &mut self.strides.slice_mut()[axis.index()], indices);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Slice the array in place along the specified axis, then return the sliced array.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[must_use = "slice_axis_move returns an array with the sliced result"]
    pub fn slice_axis_move(mut self, axis: Axis, indices: Slice) -> Self
    {
        self.slice_axis_inplace(axis, indices);
        self
    }

    /// Return a view of a slice of the array, with a closure specifying the
    /// slice for each axis.
    ///
    /// This is especially useful for code which is generic over the
    /// dimensionality of the array.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.
    #[track_caller]
    pub fn slice_each_axis<F>(&self, f: F) -> ArrayView<'_, A, D>
    where
        F: FnMut(AxisDescription) -> Slice,
        S: Data,
    {
        let mut view = self.view();
        view.slice_each_axis_inplace(f);
        view
    }

    /// Return a mutable view of a slice of the array, with a closure
    /// specifying the slice for each axis.
    ///
    /// This is especially useful for code which is generic over the
    /// dimensionality of the array.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.
    #[track_caller]
    pub fn slice_each_axis_mut<F>(&mut self, f: F) -> ArrayViewMut<'_, A, D>
    where
        F: FnMut(AxisDescription) -> Slice,
        S: DataMut,
    {
        let mut view = self.view_mut();
        view.slice_each_axis_inplace(f);
        view
    }

    /// Slice the array in place, with a closure specifying the slice for each
    /// axis.
    ///
    /// This is especially useful for code which is generic over the
    /// dimensionality of the array.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.
    #[track_caller]
    pub fn slice_each_axis_inplace<F>(&mut self, mut f: F)
    where F: FnMut(AxisDescription) -> Slice
    {
        for ax in 0..self.ndim() {
            self.slice_axis_inplace(
                Axis(ax),
                f(AxisDescription {
                    axis: Axis(ax),
                    len: self.dim[ax],
                    stride: self.strides[ax] as isize,
                }),
            )
        }
    }

    /// Return a reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// Arrays also support indexing syntax: `array[index]`.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    ///
    /// assert!(
    ///     a.get((0, 1)) == Some(&2.) &&
    ///     a.get((0, 2)) == None &&
    ///     a[(0, 1)] == 2. &&
    ///     a[[0, 1]] == 2.
    /// );
    /// ```
    pub fn get<I>(&self, index: I) -> Option<&A>
    where
        S: Data,
        I: NdIndex<D>,
    {
        unsafe { self.get_ptr(index).map(|ptr| &*ptr) }
    }

    /// Return a raw pointer to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    ///
    /// let v = a.raw_view();
    /// let p = a.get_ptr((0, 1)).unwrap();
    ///
    /// assert_eq!(unsafe { *p }, 2.);
    /// ```
    pub fn get_ptr<I>(&self, index: I) -> Option<*const A>
    where I: NdIndex<D>
    {
        let ptr = self.ptr;
        index
            .index_checked(&self.dim, &self.strides)
            .map(move |offset| unsafe { ptr.as_ptr().offset(offset) as *const _ })
    }

    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut A>
    where
        S: DataMut,
        I: NdIndex<D>,
    {
        unsafe { self.get_mut_ptr(index).map(|ptr| &mut *ptr) }
    }

    /// Return a raw pointer to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let mut a = arr2(&[[1., 2.], [3., 4.]]);
    ///
    /// let v = a.raw_view_mut();
    /// let p = a.get_mut_ptr((0, 1)).unwrap();
    ///
    /// unsafe {
    ///     *p = 5.;
    /// }
    ///
    /// assert_eq!(a.get((0, 1)), Some(&5.));
    /// ```
    pub fn get_mut_ptr<I>(&mut self, index: I) -> Option<*mut A>
    where
        S: RawDataMut,
        I: NdIndex<D>,
    {
        // const and mut are separate to enforce &mutness as well as the
        // extra code in as_mut_ptr
        let ptr = self.as_mut_ptr();
        index
            .index_checked(&self.dim, &self.strides)
            .map(move |offset| unsafe { ptr.offset(offset) })
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a reference to the element at `index`.
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is in-bounds.
    #[inline]
    pub unsafe fn uget<I>(&self, index: I) -> &A
    where
        S: Data,
        I: NdIndex<D>,
    {
        arraytraits::debug_bounds_check(self, &index);
        let off = index.index_unchecked(&self.strides);
        &*self.ptr.as_ptr().offset(off)
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a mutable reference to the element at `index`.
    ///
    /// **Note:** Only unchecked for non-debug builds of ndarray.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// 1. the index is in-bounds and
    ///
    /// 2. the data is uniquely held by the array. (This property is guaranteed
    ///    for `Array` and `ArrayViewMut`, but not for `ArcArray` or `CowArray`.)
    #[inline]
    pub unsafe fn uget_mut<I>(&mut self, index: I) -> &mut A
    where
        S: DataMut,
        I: NdIndex<D>,
    {
        debug_assert!(self.data.is_unique());
        arraytraits::debug_bounds_check(self, &index);
        let off = index.index_unchecked(&self.strides);
        &mut *self.ptr.as_ptr().offset(off)
    }

    /// Swap elements at indices `index1` and `index2`.
    ///
    /// Indices may be equal.
    ///
    /// ***Panics*** if an index is out of bounds.
    #[track_caller]
    pub fn swap<I>(&mut self, index1: I, index2: I)
    where
        S: DataMut,
        I: NdIndex<D>,
    {
        let ptr = self.as_mut_ptr();
        let offset1 = index1.index_checked(&self.dim, &self.strides);
        let offset2 = index2.index_checked(&self.dim, &self.strides);
        if let Some(offset1) = offset1 {
            if let Some(offset2) = offset2 {
                unsafe {
                    std::ptr::swap(ptr.offset(offset1), ptr.offset(offset2));
                }
                return;
            }
        }
        panic!("swap: index out of bounds for indices {:?} {:?}", index1, index2);
    }

    /// Swap elements *unchecked* at indices `index1` and `index2`.
    ///
    /// Indices may be equal.
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// 1. both `index1` and `index2` are in-bounds and
    ///
    /// 2. the data is uniquely held by the array. (This property is guaranteed
    ///    for `Array` and `ArrayViewMut`, but not for `ArcArray` or `CowArray`.)
    pub unsafe fn uswap<I>(&mut self, index1: I, index2: I)
    where
        S: DataMut,
        I: NdIndex<D>,
    {
        debug_assert!(self.data.is_unique());
        arraytraits::debug_bounds_check(self, &index1);
        arraytraits::debug_bounds_check(self, &index2);
        let off1 = index1.index_unchecked(&self.strides);
        let off2 = index2.index_unchecked(&self.strides);
        std::ptr::swap(self.ptr.as_ptr().offset(off1), self.ptr.as_ptr().offset(off2));
    }

    // `get` for zero-dimensional arrays
    // panics if dimension is not zero. otherwise an element is always present.
    fn get_0d(&self) -> &A
    where S: Data
    {
        assert!(self.ndim() == 0);
        unsafe { &*self.as_ptr() }
    }

    /// Returns a view restricted to `index` along the axis, with the axis
    /// removed.
    ///
    /// See [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, ArrayView, Axis};
    ///
    /// let a = arr2(&[[1., 2. ],    // ... axis 0, row 0
    ///                [3., 4. ],    // --- axis 0, row 1
    ///                [5., 6. ]]);  // ... axis 0, row 2
    /// //               .   \
    /// //                .   axis 1, column 1
    /// //                 axis 1, column 0
    /// assert!(
    ///     a.index_axis(Axis(0), 1) == ArrayView::from(&[3., 4.]) &&
    ///     a.index_axis(Axis(1), 1) == ArrayView::from(&[2., 4., 6.])
    /// );
    /// ```
    #[track_caller]
    pub fn index_axis(&self, axis: Axis, index: usize) -> ArrayView<'_, A, D::Smaller>
    where
        S: Data,
        D: RemoveAxis,
    {
        self.view().index_axis_move(axis, index)
    }

    /// Returns a mutable view restricted to `index` along the axis, with the
    /// axis removed.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, aview2, Axis};
    ///
    /// let mut a = arr2(&[[1., 2. ],
    ///                    [3., 4. ]]);
    /// //                   .   \
    /// //                    .   axis 1, column 1
    /// //                     axis 1, column 0
    ///
    /// {
    ///     let mut column1 = a.index_axis_mut(Axis(1), 1);
    ///     column1 += 10.;
    /// }
    ///
    /// assert!(
    ///     a == aview2(&[[1., 12.],
    ///                   [3., 14.]])
    /// );
    /// ```
    #[track_caller]
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> ArrayViewMut<'_, A, D::Smaller>
    where
        S: DataMut,
        D: RemoveAxis,
    {
        self.view_mut().index_axis_move(axis, index)
    }

    /// Collapses the array to `index` along the axis and removes the axis.
    ///
    /// See [`.index_axis()`](Self::index_axis) and [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    #[track_caller]
    pub fn index_axis_move(mut self, axis: Axis, index: usize) -> ArrayBase<S, D::Smaller>
    where D: RemoveAxis
    {
        self.collapse_axis(axis, index);
        let dim = self.dim.remove_axis(axis);
        let strides = self.strides.remove_axis(axis);
        // safe because new dimension, strides allow access to a subset of old data
        unsafe { self.with_strides_dim(strides, dim) }
    }

    /// Selects `index` along the axis, collapsing the axis into length one.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    #[track_caller]
    pub fn collapse_axis(&mut self, axis: Axis, index: usize)
    {
        let offset = dimension::do_collapse_axis(&mut self.dim, &self.strides, axis.index(), index);
        self.ptr = unsafe { self.ptr.offset(offset) };
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Along `axis`, select arbitrary subviews corresponding to `indices`
    /// and and copy them into a new array.
    ///
    /// **Panics** if `axis` or an element of `indices` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, Axis};
    ///
    /// let x = arr2(&[[0., 1.],
    ///                [2., 3.],
    ///                [4., 5.],
    ///                [6., 7.],
    ///                [8., 9.]]);
    ///
    /// let r = x.select(Axis(0), &[0, 4, 3]);
    /// assert!(
    ///         r == arr2(&[[0., 1.],
    ///                     [8., 9.],
    ///                     [6., 7.]])
    ///);
    /// ```
    #[track_caller]
    pub fn select(&self, axis: Axis, indices: &[Ix]) -> Array<A, D>
    where
        A: Clone,
        S: Data,
        D: RemoveAxis,
    {
        if self.ndim() == 1 {
            // using .len_of(axis) means that we check if `axis` is in bounds too.
            let axis_len = self.len_of(axis);
            // bounds check the indices first
            if let Some(max_index) = indices.iter().cloned().max() {
                if max_index >= axis_len {
                    panic!("ndarray: index {} is out of bounds in array of len {}",
                           max_index, self.len_of(axis));
                }
            } // else: indices empty is ok
            let view = self.view().into_dimensionality::<Ix1>().unwrap();
            Array::from_iter(indices.iter().map(move |&index| {
                // Safety: bounds checked indexes
                unsafe { view.uget(index).clone() }
            }))
            .into_dimensionality::<D>()
            .unwrap()
        } else {
            let mut subs = vec![self.view(); indices.len()];
            for (&i, sub) in zip(indices, &mut subs[..]) {
                sub.collapse_axis(axis, i);
            }
            if subs.is_empty() {
                let mut dim = self.raw_dim();
                dim.set_axis(axis, 0);
                unsafe { Array::from_shape_vec_unchecked(dim, vec![]) }
            } else {
                concatenate(axis, &subs).unwrap()
            }
        }
    }

    /// Return a producer and iterable that traverses over the *generalized*
    /// rows of the array. For a 2D array these are the regular rows.
    ///
    /// This is equivalent to `.lanes(Axis(n - 1))` where *n* is `self.ndim()`.
    ///
    /// For an array of dimensions *a* × *b* × *c* × ... × *l* × *m*
    /// it has *a* × *b* × *c* × ... × *l* rows each of length *m*.
    ///
    /// For example, in a 2 × 2 × 3 array, each row is 3 elements long
    /// and there are 2 × 2 = 4 rows in total.
    ///
    /// Iterator element is `ArrayView1<A>` (1D array view).
    ///
    /// ```
    /// use ndarray::arr3;
    ///
    /// let a = arr3(&[[[ 0,  1,  2],    // -- row 0, 0
    ///                 [ 3,  4,  5]],   // -- row 0, 1
    ///                [[ 6,  7,  8],    // -- row 1, 0
    ///                 [ 9, 10, 11]]]); // -- row 1, 1
    ///
    /// // `rows` will yield the four generalized rows of the array.
    /// for row in a.rows() {
    ///     /* loop body */
    /// }
    /// ```
    pub fn rows(&self) -> Lanes<'_, A, D::Smaller>
    where S: Data
    {
        let mut n = self.ndim();
        if n == 0 {
            n += 1;
        }
        Lanes::new(self.view(), Axis(n - 1))
    }

    /// Return a producer and iterable that traverses over the *generalized*
    /// rows of the array and yields mutable array views.
    ///
    /// Iterator element is `ArrayView1<A>` (1D read-write array view).
    pub fn rows_mut(&mut self) -> LanesMut<'_, A, D::Smaller>
    where S: DataMut
    {
        let mut n = self.ndim();
        if n == 0 {
            n += 1;
        }
        LanesMut::new(self.view_mut(), Axis(n - 1))
    }

    /// Return a producer and iterable that traverses over the *generalized*
    /// columns of the array. For a 2D array these are the regular columns.
    ///
    /// This is equivalent to `.lanes(Axis(0))`.
    ///
    /// For an array of dimensions *a* × *b* × *c* × ... × *l* × *m*
    /// it has *b* × *c* × ... × *l* × *m* columns each of length *a*.
    ///
    /// For example, in a 2 × 2 × 3 array, each column is 2 elements long
    /// and there are 2 × 3 = 6 columns in total.
    ///
    /// Iterator element is `ArrayView1<A>` (1D array view).
    ///
    /// ```
    /// use ndarray::arr3;
    ///
    /// // The generalized columns of a 3D array:
    /// // are directed along the 0th axis: 0 and 6, 1 and 7 and so on...
    /// let a = arr3(&[[[ 0,  1,  2], [ 3,  4,  5]],
    ///                [[ 6,  7,  8], [ 9, 10, 11]]]);
    ///
    /// // Here `columns` will yield the six generalized columns of the array.
    /// for column in a.columns() {
    ///     /* loop body */
    /// }
    /// ```
    pub fn columns(&self) -> Lanes<'_, A, D::Smaller>
    where S: Data
    {
        Lanes::new(self.view(), Axis(0))
    }

    /// Return a producer and iterable that traverses over the *generalized*
    /// columns of the array and yields mutable array views.
    ///
    /// Iterator element is `ArrayView1<A>` (1D read-write array view).
    pub fn columns_mut(&mut self) -> LanesMut<'_, A, D::Smaller>
    where S: DataMut
    {
        LanesMut::new(self.view_mut(), Axis(0))
    }

    /// Return a producer and iterable that traverses over all 1D lanes
    /// pointing in the direction of `axis`.
    ///
    /// When pointing in the direction of the first axis, they are *columns*,
    /// in the direction of the last axis *rows*; in general they are all
    /// *lanes* and are one dimensional.
    ///
    /// Iterator element is `ArrayView1<A>` (1D array view).
    ///
    /// ```
    /// use ndarray::{arr3, aview1, Axis};
    ///
    /// let a = arr3(&[[[ 0,  1,  2],
    ///                 [ 3,  4,  5]],
    ///                [[ 6,  7,  8],
    ///                 [ 9, 10, 11]]]);
    ///
    /// let inner0 = a.lanes(Axis(0));
    /// let inner1 = a.lanes(Axis(1));
    /// let inner2 = a.lanes(Axis(2));
    ///
    /// // The first lane for axis 0 is [0, 6]
    /// assert_eq!(inner0.into_iter().next().unwrap(), aview1(&[0, 6]));
    /// // The first lane for axis 1 is [0, 3]
    /// assert_eq!(inner1.into_iter().next().unwrap(), aview1(&[0, 3]));
    /// // The first lane for axis 2 is [0, 1, 2]
    /// assert_eq!(inner2.into_iter().next().unwrap(), aview1(&[0, 1, 2]));
    /// ```
    pub fn lanes(&self, axis: Axis) -> Lanes<'_, A, D::Smaller>
    where S: Data
    {
        Lanes::new(self.view(), axis)
    }

    /// Return a producer and iterable that traverses over all 1D lanes
    /// pointing in the direction of `axis`.
    ///
    /// Iterator element is `ArrayViewMut1<A>` (1D read-write array view).
    pub fn lanes_mut(&mut self, axis: Axis) -> LanesMut<'_, A, D::Smaller>
    where S: DataMut
    {
        LanesMut::new(self.view_mut(), axis)
    }

    /// Return an iterator that traverses over the outermost dimension
    /// and yields each subview.
    ///
    /// This is equivalent to `.axis_iter(Axis(0))`.
    ///
    /// Iterator element is `ArrayView<A, D::Smaller>` (read-only array view).
    #[allow(deprecated)]
    pub fn outer_iter(&self) -> AxisIter<'_, A, D::Smaller>
    where
        S: Data,
        D: RemoveAxis,
    {
        self.view().into_outer_iter()
    }

    /// Return an iterator that traverses over the outermost dimension
    /// and yields each subview.
    ///
    /// This is equivalent to `.axis_iter_mut(Axis(0))`.
    ///
    /// Iterator element is `ArrayViewMut<A, D::Smaller>` (read-write array view).
    #[allow(deprecated)]
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<'_, A, D::Smaller>
    where
        S: DataMut,
        D: RemoveAxis,
    {
        self.view_mut().into_outer_iter()
    }

    /// Return an iterator that traverses over `axis`
    /// and yields each subview along it.
    ///
    /// For example, in a 3 × 4 × 5 array, with `axis` equal to `Axis(2)`,
    /// the iterator element
    /// is a 3 × 4 subview (and there are 5 in total), as shown
    /// in the picture below.
    ///
    /// Iterator element is `ArrayView<A, D::Smaller>` (read-only array view).
    ///
    /// See [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// <img src="https://rust-ndarray.github.io/ndarray/images/axis_iter_3_4_5.svg" height="250px">
    #[track_caller]
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D::Smaller>
    where
        S: Data,
        D: RemoveAxis,
    {
        AxisIter::new(self.view(), axis)
    }

    /// Return an iterator that traverses over `axis`
    /// and yields each mutable subview along it.
    ///
    /// Iterator element is `ArrayViewMut<A, D::Smaller>`
    /// (read-write array view).
    ///
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, A, D::Smaller>
    where
        S: DataMut,
        D: RemoveAxis,
    {
        AxisIterMut::new(self.view_mut(), axis)
    }

    /// Return an iterator that traverses over `axis` by chunks of `size`,
    /// yielding non-overlapping views along that axis.
    ///
    /// Iterator element is `ArrayView<A, D>`
    ///
    /// The last view may have less elements if `size` does not divide
    /// the axis' dimension.
    ///
    /// **Panics** if `axis` is out of bounds or if `size` is zero.
    ///
    /// ```
    /// use ndarray::Array;
    /// use ndarray::{arr3, Axis};
    ///
    /// let a = Array::from_iter(0..28).into_shape_with_order((2, 7, 2)).unwrap();
    /// let mut iter = a.axis_chunks_iter(Axis(1), 2);
    ///
    /// // first iteration yields a 2 × 2 × 2 view
    /// assert_eq!(iter.next().unwrap(),
    ///            arr3(&[[[ 0,  1], [ 2, 3]],
    ///                   [[14, 15], [16, 17]]]));
    ///
    /// // however the last element is a 2 × 1 × 2 view since 7 % 2 == 1
    /// assert_eq!(iter.next_back().unwrap(), arr3(&[[[12, 13]],
    ///                                              [[26, 27]]]));
    /// ```
    #[track_caller]
    pub fn axis_chunks_iter(&self, axis: Axis, size: usize) -> AxisChunksIter<'_, A, D>
    where S: Data
    {
        AxisChunksIter::new(self.view(), axis, size)
    }

    /// Return an iterator that traverses over `axis` by chunks of `size`,
    /// yielding non-overlapping read-write views along that axis.
    ///
    /// Iterator element is `ArrayViewMut<A, D>`
    ///
    /// **Panics** if `axis` is out of bounds or if `size` is zero.
    #[track_caller]
    pub fn axis_chunks_iter_mut(&mut self, axis: Axis, size: usize) -> AxisChunksIterMut<'_, A, D>
    where S: DataMut
    {
        AxisChunksIterMut::new(self.view_mut(), axis, size)
    }

    /// Return an exact chunks producer (and iterable).
    ///
    /// It produces the whole chunks of a given n-dimensional chunk size,
    /// skipping the remainder along each dimension that doesn't fit evenly.
    ///
    /// The produced element is a `ArrayView<A, D>` with exactly the dimension
    /// `chunk_size`.
    ///
    /// **Panics** if any dimension of `chunk_size` is zero<br>
    /// (**Panics** if `D` is `IxDyn` and `chunk_size` does not match the
    /// number of array axes.)
    #[track_caller]
    pub fn exact_chunks<E>(&self, chunk_size: E) -> ExactChunks<'_, A, D>
    where
        E: IntoDimension<Dim = D>,
        S: Data,
    {
        ExactChunks::new(self.view(), chunk_size)
    }

    /// Return an exact chunks producer (and iterable).
    ///
    /// It produces the whole chunks of a given n-dimensional chunk size,
    /// skipping the remainder along each dimension that doesn't fit evenly.
    ///
    /// The produced element is a `ArrayViewMut<A, D>` with exactly
    /// the dimension `chunk_size`.
    ///
    /// **Panics** if any dimension of `chunk_size` is zero<br>
    /// (**Panics** if `D` is `IxDyn` and `chunk_size` does not match the
    /// number of array axes.)
    ///
    /// ```rust
    /// use ndarray::Array;
    /// use ndarray::arr2;
    /// let mut a = Array::zeros((6, 7));
    ///
    /// // Fill each 2 × 2 chunk with the index of where it appeared in iteration
    /// for (i, mut chunk) in a.exact_chunks_mut((2, 2)).into_iter().enumerate() {
    ///     chunk.fill(i);
    /// }
    ///
    /// // The resulting array is:
    /// assert_eq!(
    ///   a,
    ///   arr2(&[[0, 0, 1, 1, 2, 2, 0],
    ///          [0, 0, 1, 1, 2, 2, 0],
    ///          [3, 3, 4, 4, 5, 5, 0],
    ///          [3, 3, 4, 4, 5, 5, 0],
    ///          [6, 6, 7, 7, 8, 8, 0],
    ///          [6, 6, 7, 7, 8, 8, 0]]));
    /// ```
    #[track_caller]
    pub fn exact_chunks_mut<E>(&mut self, chunk_size: E) -> ExactChunksMut<'_, A, D>
    where
        E: IntoDimension<Dim = D>,
        S: DataMut,
    {
        ExactChunksMut::new(self.view_mut(), chunk_size)
    }

    /// Return a window producer and iterable.
    ///
    /// The windows are all distinct overlapping views of size `window_size`
    /// that fit into the array's shape.
    ///
    /// This is essentially equivalent to [`.windows_with_stride()`] with unit stride.
    #[track_caller]
    pub fn windows<E>(&self, window_size: E) -> Windows<'_, A, D>
    where
        E: IntoDimension<Dim = D>,
        S: Data,
    {
        Windows::new(self.view(), window_size)
    }

    /// Return a window producer and iterable.
    ///
    /// The windows are all distinct views of size `window_size`
    /// that fit into the array's shape.
    ///
    /// The stride is ordered by the outermost axis.<br>
    /// Hence, a (x₀, x₁, ..., xₙ) stride will be applied to
    /// (A₀, A₁, ..., Aₙ) where Aₓ stands for `Axis(x)`.
    ///
    /// This produces all windows that fit within the array for the given stride,
    /// assuming the window size is not larger than the array size.
    ///
    /// The produced element is an `ArrayView<A, D>` with exactly the dimension
    /// `window_size`.
    ///
    /// Note that passing a stride of only ones is similar to
    /// calling [`ArrayBase::windows()`].
    ///
    /// **Panics** if any dimension of `window_size` or `stride` is zero.<br>
    /// (**Panics** if `D` is `IxDyn` and `window_size` or `stride` does not match the
    /// number of array axes.)
    ///
    /// This is the same illustration found in [`ArrayBase::windows()`],
    /// 2×2 windows in a 3×4 array, but now with a (1, 2) stride:
    ///
    /// ```text
    ///          ──▶ Axis(1)
    ///
    ///      │   ┏━━━━━┳━━━━━┱─────┬─────┐   ┌─────┬─────┲━━━━━┳━━━━━┓
    ///      ▼   ┃ a₀₀ ┃ a₀₁ ┃     │     │   │     │     ┃ a₀₂ ┃ a₀₃ ┃
    /// Axis(0)  ┣━━━━━╋━━━━━╉─────┼─────┤   ├─────┼─────╊━━━━━╋━━━━━┫
    ///          ┃ a₁₀ ┃ a₁₁ ┃     │     │   │     │     ┃ a₁₂ ┃ a₁₃ ┃
    ///          ┡━━━━━╇━━━━━╃─────┼─────┤   ├─────┼─────╄━━━━━╇━━━━━┩
    ///          │     │     │     │     │   │     │     │     │     │
    ///          └─────┴─────┴─────┴─────┘   └─────┴─────┴─────┴─────┘
    ///
    ///          ┌─────┬─────┬─────┬─────┐   ┌─────┬─────┬─────┬─────┐
    ///          │     │     │     │     │   │     │     │     │     │
    ///          ┢━━━━━╈━━━━━╅─────┼─────┤   ├─────┼─────╆━━━━━╈━━━━━┪
    ///          ┃ a₁₀ ┃ a₁₁ ┃     │     │   │     │     ┃ a₁₂ ┃ a₁₃ ┃
    ///          ┣━━━━━╋━━━━━╉─────┼─────┤   ├─────┼─────╊━━━━━╋━━━━━┫
    ///          ┃ a₂₀ ┃ a₂₁ ┃     │     │   │     │     ┃ a₂₂ ┃ a₂₃ ┃
    ///          ┗━━━━━┻━━━━━┹─────┴─────┘   └─────┴─────┺━━━━━┻━━━━━┛
    /// ```
    #[track_caller]
    pub fn windows_with_stride<E>(&self, window_size: E, stride: E) -> Windows<'_, A, D>
    where
        E: IntoDimension<Dim = D>,
        S: Data,
    {
        Windows::new_with_stride(self.view(), window_size, stride)
    }

    /// Returns a producer which traverses over all windows of a given length along an axis.
    ///
    /// The windows are all distinct, possibly-overlapping views. The shape of each window
    /// is the shape of `self`, with the length of `axis` replaced with `window_size`.
    ///
    /// **Panics** if `axis` is out-of-bounds or if `window_size` is zero.
    ///
    /// ```
    /// use ndarray::{Array3, Axis, s};
    ///
    /// let arr = Array3::from_shape_fn([4, 5, 2], |(i, j, k)| i * 100 + j * 10 + k);
    /// let correct = vec![
    ///     arr.slice(s![.., 0..3, ..]),
    ///     arr.slice(s![.., 1..4, ..]),
    ///     arr.slice(s![.., 2..5, ..]),
    /// ];
    /// for (window, correct) in arr.axis_windows(Axis(1), 3).into_iter().zip(&correct) {
    ///     assert_eq!(window, correct);
    ///     assert_eq!(window.shape(), &[4, 3, 2]);
    /// }
    /// ```
    pub fn axis_windows(&self, axis: Axis, window_size: usize) -> AxisWindows<'_, A, D>
    where S: Data
    {
        let axis_index = axis.index();

        ndassert!(
            axis_index < self.ndim(),
            concat!(
                "Window axis {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            axis_index,
            self.ndim(),
            self.shape()
        );

        AxisWindows::new(self.view(), axis, window_size)
    }

    // Return (length, stride) for diagonal
    fn diag_params(&self) -> (Ix, Ixs)
    {
        /* empty shape has len 1 */
        let len = self.dim.slice().iter().cloned().min().unwrap_or(1);
        let stride = self.strides().iter().sum();
        (len, stride)
    }

    /// Return a view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by *(0, 0, .., 0)*,
    /// *(1, 1, ..., 1)* etc as long as all axes have elements.
    pub fn diag(&self) -> ArrayView1<'_, A>
    where S: Data
    {
        self.view().into_diag()
    }

    /// Return a read-write view over the diagonal elements of the array.
    pub fn diag_mut(&mut self) -> ArrayViewMut1<'_, A>
    where S: DataMut
    {
        self.view_mut().into_diag()
    }

    /// Return the diagonal as a one-dimensional array.
    pub fn into_diag(self) -> ArrayBase<S, Ix1>
    {
        let (len, stride) = self.diag_params();
        // safe because new len stride allows access to a subset of the current elements
        unsafe { self.with_strides_dim(Ix1(stride as Ix), Ix1(len)) }
    }

    /// Try to make the array unshared.
    ///
    /// This is equivalent to `.ensure_unique()` if `S: DataMut`.
    ///
    /// This method is mostly only useful with unsafe code.
    fn try_ensure_unique(&mut self)
    where S: RawDataMut
    {
        debug_assert!(self.pointer_is_inbounds());
        S::try_ensure_unique(self);
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Make the array unshared.
    ///
    /// This method is mostly only useful with unsafe code.
    fn ensure_unique(&mut self)
    where S: DataMut
    {
        debug_assert!(self.pointer_is_inbounds());
        S::ensure_unique(self);
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Return `true` if the array data is laid out in contiguous “C order” in
    /// memory (where the last index is the most rapidly varying).
    ///
    /// Return `false` otherwise, i.e. the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool
    {
        dimension::is_layout_c(&self.dim, &self.strides)
    }

    /// Return true if the array is known to be contiguous.
    pub(crate) fn is_contiguous(&self) -> bool
    {
        D::is_contiguous(&self.dim, &self.strides)
    }

    /// Return a standard-layout array containing the data, cloning if
    /// necessary.
    ///
    /// If `self` is in standard layout, a COW view of the data is returned
    /// without cloning. Otherwise, the data is cloned, and the returned array
    /// owns the cloned data.
    ///
    /// ```
    /// use ndarray::Array2;
    ///
    /// let standard = Array2::<f64>::zeros((3, 4));
    /// assert!(standard.is_standard_layout());
    /// let cow_view = standard.as_standard_layout();
    /// assert!(cow_view.is_view());
    /// assert!(cow_view.is_standard_layout());
    ///
    /// let fortran = standard.reversed_axes();
    /// assert!(!fortran.is_standard_layout());
    /// let cow_owned = fortran.as_standard_layout();
    /// assert!(cow_owned.is_owned());
    /// assert!(cow_owned.is_standard_layout());
    /// ```
    pub fn as_standard_layout(&self) -> CowArray<'_, A, D>
    where
        S: Data<Elem = A>,
        A: Clone,
    {
        if self.is_standard_layout() {
            CowArray::from(self.view())
        } else {
            let v = crate::iterators::to_vec_mapped(self.iter(), A::clone);
            let dim = self.dim.clone();
            debug_assert_eq!(v.len(), dim.size());

            unsafe {
                // Safe because the shape and element type are from the existing array
                // and the strides are the default strides.
                CowArray::from(Array::from_shape_vec_unchecked(dim, v))
            }
        }
    }

    /// Return a pointer to the first element in the array.
    ///
    /// Raw access to array elements needs to follow the strided indexing
    /// scheme: an element at multi-index *I* in an array with strides *S* is
    /// located at offset
    ///
    /// *Σ<sub>0 ≤ k < d</sub> I<sub>k</sub> × S<sub>k</sub>*
    ///
    /// where *d* is `self.ndim()`.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const A
    {
        self.ptr.as_ptr() as *const A
    }

    /// Return a mutable pointer to the first element in the array.
    ///
    /// This method attempts to unshare the data. If `S: DataMut`, then the
    /// data is guaranteed to be uniquely held on return.
    ///
    /// # Warning
    ///
    /// When accessing elements through this pointer, make sure to use strides
    /// obtained *after* calling this method, since the process of unsharing
    /// the data may change the strides.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut A
    where S: RawDataMut
    {
        self.try_ensure_unique(); // for ArcArray
        self.ptr.as_ptr()
    }

    /// Return a raw view of the array.
    #[inline]
    pub fn raw_view(&self) -> RawArrayView<A, D>
    {
        unsafe { RawArrayView::new(self.ptr, self.dim.clone(), self.strides.clone()) }
    }

    /// Return a raw mutable view of the array.
    ///
    /// This method attempts to unshare the data. If `S: DataMut`, then the
    /// data is guaranteed to be uniquely held on return.
    #[inline]
    pub fn raw_view_mut(&mut self) -> RawArrayViewMut<A, D>
    where S: RawDataMut
    {
        self.try_ensure_unique(); // for ArcArray
        unsafe { RawArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone()) }
    }

    /// Return a raw mutable view of the array.
    ///
    /// Safety: The caller must ensure that the owned array is unshared when this is called
    #[inline]
    pub(crate) unsafe fn raw_view_mut_unchecked(&mut self) -> RawArrayViewMut<A, D>
    where S: DataOwned
    {
        RawArrayViewMut::new(self.ptr, self.dim.clone(), self.strides.clone())
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    ///
    /// If this function returns `Some(_)`, then the element order in the slice
    /// corresponds to the logical order of the array’s elements.
    pub fn as_slice(&self) -> Option<&[A]>
    where S: Data
    {
        if self.is_standard_layout() {
            unsafe { Some(slice::from_raw_parts(self.ptr.as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    pub fn as_slice_mut(&mut self) -> Option<&mut [A]>
    where S: DataMut
    {
        if self.is_standard_layout() {
            self.ensure_unique();
            unsafe { Some(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice if it is contiguous,
    /// return `None` otherwise.
    ///
    /// If this function returns `Some(_)`, then the elements in the slice
    /// have whatever order the elements have in memory.
    pub fn as_slice_memory_order(&self) -> Option<&[A]>
    where S: Data
    {
        if self.is_contiguous() {
            let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
            unsafe { Some(slice::from_raw_parts(self.ptr.sub(offset).as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice if it is contiguous,
    /// return `None` otherwise.
    ///
    /// In the contiguous case, in order to return a unique reference, this
    /// method unshares the data if necessary, but it preserves the existing
    /// strides.
    pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [A]>
    where S: DataMut
    {
        self.try_as_slice_memory_order_mut().ok()
    }

    /// Return the array’s data as a slice if it is contiguous, otherwise
    /// return `self` in the `Err` variant.
    pub(crate) fn try_as_slice_memory_order_mut(&mut self) -> Result<&mut [A], &mut Self>
    where S: DataMut
    {
        if self.is_contiguous() {
            self.ensure_unique();
            let offset = offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides);
            unsafe { Ok(slice::from_raw_parts_mut(self.ptr.sub(offset).as_ptr(), self.len())) }
        } else {
            Err(self)
        }
    }

    /// Transform the array into `new_shape`; any shape with the same number of elements is
    /// accepted.
    ///
    /// `order` specifies the *logical* order in which the array is to be read and reshaped.
    /// The array is returned as a `CowArray`; a view if possible, otherwise an owned array.
    ///
    /// For example, when starting from the one-dimensional sequence 1 2 3 4 5 6, it would be
    /// understood as a 2 x 3 array in row major ("C") order this way:
    ///
    /// ```text
    /// 1 2 3
    /// 4 5 6
    /// ```
    ///
    /// and as 2 x 3 in column major ("F") order this way:
    ///
    /// ```text
    /// 1 3 5
    /// 2 4 6
    /// ```
    ///
    /// This example should show that any time we "reflow" the elements in the array to a different
    /// number of rows and columns (or more axes if applicable), it is important to pick an index
    /// ordering, and that's the reason for the function parameter for `order`.
    ///
    /// The `new_shape` parameter should be a dimension and an optional order like these examples:
    ///
    /// ```text
    /// (3, 4)                          // Shape 3 x 4 with default order (RowMajor)
    /// ((3, 4), Order::RowMajor))      // use specific order
    /// ((3, 4), Order::ColumnMajor))   // use specific order
    /// ((3, 4), Order::C))             // use shorthand for order - shorthands C and F
    /// ```
    ///
    /// **Errors** if the new shape doesn't have the same number of elements as the array's current
    /// shape.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray::Order;
    ///
    /// assert!(
    ///     array![1., 2., 3., 4., 5., 6.].to_shape(((2, 3), Order::RowMajor)).unwrap()
    ///     == array![[1., 2., 3.],
    ///               [4., 5., 6.]]
    /// );
    ///
    /// assert!(
    ///     array![1., 2., 3., 4., 5., 6.].to_shape(((2, 3), Order::ColumnMajor)).unwrap()
    ///     == array![[1., 3., 5.],
    ///               [2., 4., 6.]]
    /// );
    /// ```
    pub fn to_shape<E>(&self, new_shape: E) -> Result<CowArray<'_, A, E::Dim>, ShapeError>
    where
        E: ShapeArg,
        A: Clone,
        S: Data,
    {
        let (shape, order) = new_shape.into_shape_and_order();
        self.to_shape_order(shape, order.unwrap_or(Order::RowMajor))
    }

    fn to_shape_order<E>(&self, shape: E, order: Order) -> Result<CowArray<'_, A, E>, ShapeError>
    where
        E: Dimension,
        A: Clone,
        S: Data,
    {
        let len = self.dim.size();
        if size_of_shape_checked(&shape) != Ok(len) {
            return Err(error::incompatible_shapes(&self.dim, &shape));
        }

        // Create a view if the length is 0, safe because the array and new shape is empty.
        if len == 0 {
            unsafe {
                return Ok(CowArray::from(ArrayView::from_shape_ptr(shape, self.as_ptr())));
            }
        }

        // Try to reshape the array as a view into the existing data
        match reshape_dim(&self.dim, &self.strides, &shape, order) {
            Ok(to_strides) => unsafe {
                return Ok(CowArray::from(ArrayView::new(self.ptr, shape, to_strides)));
            },
            Err(err) if err.kind() == ErrorKind::IncompatibleShape => {
                return Err(error::incompatible_shapes(&self.dim, &shape));
            }
            _otherwise => {}
        }

        // otherwise create a new array and copy the elements
        unsafe {
            let (shape, view) = match order {
                Order::RowMajor => (shape.set_f(false), self.view()),
                Order::ColumnMajor => (shape.set_f(true), self.t()),
            };
            Ok(CowArray::from(Array::from_shape_trusted_iter_unchecked(shape, view.into_iter(), A::clone)))
        }
    }

    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted, but the source array must be contiguous.
    ///
    /// If an index ordering is not specified, the default is `RowMajor`.
    /// The operation will only succeed if the array's memory layout is compatible with
    /// the index ordering, so that the array elements can be rearranged in place.
    ///
    /// If required use `.to_shape()` or `.into_shape_clone` instead for more flexible reshaping of
    /// arrays, which allows copying elements if required.
    ///
    /// **Errors** if the shapes don't have the same number of elements.<br>
    /// **Errors** if order RowMajor is given but input is not c-contiguous.
    /// **Errors** if order ColumnMajor is given but input is not f-contiguous.
    ///
    /// If shape is not given: use memory layout of incoming array. Row major arrays are
    /// reshaped using row major index ordering, column major arrays with column major index
    /// ordering.
    ///
    /// The `new_shape` parameter should be a dimension and an optional order like these examples:
    ///
    /// ```text
    /// (3, 4)                          // Shape 3 x 4 with default order (RowMajor)
    /// ((3, 4), Order::RowMajor))      // use specific order
    /// ((3, 4), Order::ColumnMajor))   // use specific order
    /// ((3, 4), Order::C))             // use shorthand for order - shorthands C and F
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    /// use ndarray::Order;
    ///
    /// assert!(
    ///     aview1(&[1., 2., 3., 4.]).into_shape_with_order((2, 2)).unwrap()
    ///     == aview2(&[[1., 2.],
    ///                 [3., 4.]])
    /// );
    ///
    /// assert!(
    ///     aview1(&[1., 2., 3., 4.]).into_shape_with_order(((2, 2), Order::ColumnMajor)).unwrap()
    ///     == aview2(&[[1., 3.],
    ///                 [2., 4.]])
    /// );
    /// ```
    pub fn into_shape_with_order<E>(self, shape: E) -> Result<ArrayBase<S, E::Dim>, ShapeError>
    where E: ShapeArg
    {
        let (shape, order) = shape.into_shape_and_order();
        self.into_shape_with_order_impl(shape, order.unwrap_or(Order::RowMajor))
    }

    fn into_shape_with_order_impl<E>(self, shape: E, order: Order) -> Result<ArrayBase<S, E>, ShapeError>
    where E: Dimension
    {
        let shape = shape.into_dimension();
        if size_of_shape_checked(&shape) != Ok(self.dim.size()) {
            return Err(error::incompatible_shapes(&self.dim, &shape));
        }

        // Check if contiguous, then we can change shape
        unsafe {
            // safe because arrays are contiguous and len is unchanged
            match order {
                Order::RowMajor if self.is_standard_layout() =>
                    Ok(self.with_strides_dim(shape.default_strides(), shape)),
                Order::ColumnMajor if self.raw_view().reversed_axes().is_standard_layout() =>
                    Ok(self.with_strides_dim(shape.fortran_strides(), shape)),
                _otherwise => Err(error::from_kind(error::ErrorKind::IncompatibleLayout)),
            }
        }
    }

    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted, but the source array or view must be in standard
    /// or column-major (Fortran) layout.
    ///
    /// **Note** that `.into_shape()` "moves" elements differently depending on if the input array
    /// is C-contig or F-contig, it follows the index order that corresponds to the memory order.
    /// Prefer to use `.to_shape()` or `.into_shape_with_order()`.
    ///
    /// Because of this, the method **is deprecated**. That reshapes depend on memory order is not
    /// intuitive.
    ///
    /// **Errors** if the shapes don't have the same number of elements.<br>
    /// **Errors** if the input array is not c- or f-contiguous.
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    ///
    /// assert!(
    ///     aview1(&[1., 2., 3., 4.]).into_shape((2, 2)).unwrap()
    ///     == aview2(&[[1., 2.],
    ///                 [3., 4.]])
    /// );
    /// ```
    #[deprecated(note = "Use `.into_shape_with_order()` or `.to_shape()`", since = "0.16.0")]
    pub fn into_shape<E>(self, shape: E) -> Result<ArrayBase<S, E::Dim>, ShapeError>
    where E: IntoDimension
    {
        let shape = shape.into_dimension();
        if size_of_shape_checked(&shape) != Ok(self.dim.size()) {
            return Err(error::incompatible_shapes(&self.dim, &shape));
        }
        // Check if contiguous, if not => copy all, else just adapt strides
        unsafe {
            // safe because arrays are contiguous and len is unchanged
            if self.is_standard_layout() {
                Ok(self.with_strides_dim(shape.default_strides(), shape))
            } else if self.ndim() > 1 && self.raw_view().reversed_axes().is_standard_layout() {
                Ok(self.with_strides_dim(shape.fortran_strides(), shape))
            } else {
                Err(error::from_kind(error::ErrorKind::IncompatibleLayout))
            }
        }
    }

    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted. Array elements are reordered in place if
    /// possible, otherwise they are copied to create a new array.
    ///
    /// If an index ordering is not specified, the default is `RowMajor`.
    ///
    /// # `.to_shape` vs `.into_shape_clone`
    ///
    /// - `to_shape` supports views and outputting views
    /// - `to_shape` borrows the original array, `into_shape_clone` consumes the original
    /// - `into_shape_clone` preserves array type (Array vs ArcArray), but does not support views.
    ///
    /// **Errors** if the shapes don't have the same number of elements.<br>
    pub fn into_shape_clone<E>(self, shape: E) -> Result<ArrayBase<S, E::Dim>, ShapeError>
    where
        S: DataOwned,
        A: Clone,
        E: ShapeArg,
    {
        let (shape, order) = shape.into_shape_and_order();
        let order = order.unwrap_or(Order::RowMajor);
        self.into_shape_clone_order(shape, order)
    }

    fn into_shape_clone_order<E>(self, shape: E, order: Order) -> Result<ArrayBase<S, E>, ShapeError>
    where
        S: DataOwned,
        A: Clone,
        E: Dimension,
    {
        let len = self.dim.size();
        if size_of_shape_checked(&shape) != Ok(len) {
            return Err(error::incompatible_shapes(&self.dim, &shape));
        }

        // Safe because the array and new shape is empty.
        if len == 0 {
            unsafe {
                return Ok(self.with_strides_dim(shape.default_strides(), shape));
            }
        }

        // Try to reshape the array's current data
        match reshape_dim(&self.dim, &self.strides, &shape, order) {
            Ok(to_strides) => unsafe {
                return Ok(self.with_strides_dim(to_strides, shape));
            },
            Err(err) if err.kind() == ErrorKind::IncompatibleShape => {
                return Err(error::incompatible_shapes(&self.dim, &shape));
            }
            _otherwise => {}
        }

        // otherwise, clone and allocate a new array
        unsafe {
            let (shape, view) = match order {
                Order::RowMajor => (shape.set_f(false), self.view()),
                Order::ColumnMajor => (shape.set_f(true), self.t()),
            };

            Ok(ArrayBase::from_shape_trusted_iter_unchecked(shape, view.into_iter(), A::clone))
        }
    }

    /// *Note: Reshape is for `ArcArray` only. Use `.into_shape_with_order()` for
    /// other arrays and array views.*
    ///
    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted.
    ///
    /// May clone all elements if needed to arrange elements in standard
    /// layout (and break sharing).
    ///
    /// **Panics** if shapes are incompatible.
    ///
    /// *This method is obsolete, because it is inflexible in how logical order
    /// of the array is handled. See [`.to_shape()`].*
    ///
    /// ```
    /// use ndarray::{rcarr1, rcarr2};
    ///
    /// assert!(
    ///     rcarr1(&[1., 2., 3., 4.]).reshape((2, 2))
    ///     == rcarr2(&[[1., 2.],
    ///                 [3., 4.]])
    /// );
    /// ```
    #[track_caller]
    #[deprecated(note = "Use `.into_shape_with_order()` or `.to_shape()`", since = "0.16.0")]
    pub fn reshape<E>(&self, shape: E) -> ArrayBase<S, E::Dim>
    where
        S: DataShared + DataOwned,
        A: Clone,
        E: IntoDimension,
    {
        let shape = shape.into_dimension();
        if size_of_shape_checked(&shape) != Ok(self.dim.size()) {
            panic!(
                "ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                self.dim.slice(),
                shape.slice()
            )
        }
        // Check if contiguous, if not => copy all, else just adapt strides
        if self.is_standard_layout() {
            let cl = self.clone();
            // safe because array is contiguous and shape has equal number of elements
            unsafe { cl.with_strides_dim(shape.default_strides(), shape) }
        } else {
            let v = self.iter().cloned().collect::<Vec<A>>();
            unsafe { ArrayBase::from_shape_vec_unchecked(shape, v) }
        }
    }

    /// Flatten the array to a one-dimensional array.
    ///
    /// The array is returned as a `CowArray`; a view if possible, otherwise an owned array.
    ///
    /// ```
    /// use ndarray::{arr1, arr3};
    ///
    /// let array = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    /// let flattened = array.flatten();
    /// assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
    /// ```
    pub fn flatten(&self) -> CowArray<'_, A, Ix1>
    where
        A: Clone,
        S: Data,
    {
        self.flatten_with_order(Order::RowMajor)
    }

    /// Flatten the array to a one-dimensional array.
    ///
    /// `order` specifies the *logical* order in which the array is to be read and reshaped.
    /// The array is returned as a `CowArray`; a view if possible, otherwise an owned array.
    ///
    /// ```
    /// use ndarray::{arr1, arr2};
    /// use ndarray::Order;
    ///
    /// let array = arr2(&[[1, 2], [3, 4], [5, 6], [7, 8]]);
    /// let flattened = array.flatten_with_order(Order::RowMajor);
    /// assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
    /// let flattened = array.flatten_with_order(Order::ColumnMajor);
    /// assert_eq!(flattened, arr1(&[1, 3, 5, 7, 2, 4, 6, 8]));
    /// ```
    pub fn flatten_with_order(&self, order: Order) -> CowArray<'_, A, Ix1>
    where
        A: Clone,
        S: Data,
    {
        self.to_shape((self.len(), order)).unwrap()
    }

    /// Flatten the array to a one-dimensional array, consuming the array.
    ///
    /// If possible, no copy is made, and the new array use the same memory as the original array.
    /// Otherwise, a new array is allocated and the elements are copied.
    ///
    /// ```
    /// use ndarray::{arr1, arr3};
    ///
    /// let array = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    /// let flattened = array.into_flat();
    /// assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
    /// ```
    pub fn into_flat(self) -> ArrayBase<S, Ix1>
    where
        A: Clone,
        S: DataOwned,
    {
        let len = self.len();
        self.into_shape_clone(Ix1(len)).unwrap()
    }

    /// Convert any array or array view to a dynamic dimensional array or
    /// array view (respectively).
    ///
    /// ```
    /// use ndarray::{arr2, ArrayD};
    ///
    /// let array: ArrayD<i32> = arr2(&[[1, 2],
    ///                                 [3, 4]]).into_dyn();
    /// ```
    pub fn into_dyn(self) -> ArrayBase<S, IxDyn>
    {
        // safe because new dims equivalent
        unsafe {
            ArrayBase::from_data_ptr(self.data, self.ptr).with_strides_dim(self.strides.into_dyn(), self.dim.into_dyn())
        }
    }

    /// Convert an array or array view to another with the same type, but different dimensionality
    /// type. Errors if the dimensions don't agree (the number of axes must match).
    ///
    /// Note that conversion to a dynamic dimensional array will never fail (and is equivalent to
    /// the `into_dyn` method).
    ///
    /// ```
    /// use ndarray::{ArrayD, Ix2, IxDyn};
    ///
    /// // Create a dynamic dimensionality array and convert it to an Array2
    /// // (Ix2 dimension type).
    ///
    /// let array = ArrayD::<f64>::zeros(IxDyn(&[10, 10]));
    ///
    /// assert!(array.into_dimensionality::<Ix2>().is_ok());
    /// ```
    pub fn into_dimensionality<D2>(self) -> Result<ArrayBase<S, D2>, ShapeError>
    where D2: Dimension
    {
        unsafe {
            if D::NDIM == D2::NDIM {
                // safe because D == D2
                let dim = unlimited_transmute::<D, D2>(self.dim);
                let strides = unlimited_transmute::<D, D2>(self.strides);
                return Ok(ArrayBase::from_data_ptr(self.data, self.ptr).with_strides_dim(strides, dim));
            } else if D::NDIM.is_none() || D2::NDIM.is_none() {
                // one is dynamic dim
                // safe because dim, strides are equivalent under a different type
                if let Some(dim) = D2::from_dimension(&self.dim) {
                    if let Some(strides) = D2::from_dimension(&self.strides) {
                        return Ok(self.with_strides_dim(strides, dim));
                    }
                }
            }
        }
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    }

    /// Act like a larger size and/or shape array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if shapes can not be broadcast together.
    ///
    /// ***Background***
    ///
    ///  * Two axes are compatible if they are equal, or one of them is 1.
    ///  * In this instance, only the axes of the smaller side (self) can be 1.
    ///
    /// Compare axes beginning with the *last* axis of each shape.
    ///
    /// For example (1, 2, 4) can be broadcast into (7, 6, 2, 4)
    /// because its axes are either equal or 1 (or missing);
    /// while (2, 2) can *not* be broadcast into (2, 4).
    ///
    /// The implementation creates a view with strides set to zero for the
    /// axes that are to be repeated.
    ///
    /// The broadcasting documentation for Numpy has more information.
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    ///
    /// assert!(
    ///     aview1(&[1., 0.]).broadcast((10, 2)).unwrap()
    ///     == aview2(&[[1., 0.]; 10])
    /// );
    /// ```
    pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<'_, A, E::Dim>>
    where
        E: IntoDimension,
        S: Data,
    {
        /// Return new stride when trying to grow `from` into shape `to`
        ///
        /// Broadcasting works by returning a "fake stride" where elements
        /// to repeat are in axes with 0 stride, so that several indexes point
        /// to the same element.
        ///
        /// **Note:** Cannot be used for mutable iterators, since repeating
        /// elements would create aliasing pointers.
        fn upcast<D: Dimension, E: Dimension>(to: &D, from: &E, stride: &E) -> Option<D>
        {
            // Make sure the product of non-zero axis lengths does not exceed
            // `isize::MAX`. This is the only safety check we need to perform
            // because all the other constraints of `ArrayBase` are guaranteed
            // to be met since we're starting from a valid `ArrayBase`.
            let _ = size_of_shape_checked(to).ok()?;

            let mut new_stride = to.clone();
            // begin at the back (the least significant dimension)
            // size of the axis has to either agree or `from` has to be 1
            if to.ndim() < from.ndim() {
                return None;
            }

            {
                let mut new_stride_iter = new_stride.slice_mut().iter_mut().rev();
                for ((er, es), dr) in from
                    .slice()
                    .iter()
                    .rev()
                    .zip(stride.slice().iter().rev())
                    .zip(new_stride_iter.by_ref())
                {
                    /* update strides */
                    if *dr == *er {
                        /* keep stride */
                        *dr = *es;
                    } else if *er == 1 {
                        /* dead dimension, zero stride */
                        *dr = 0
                    } else {
                        return None;
                    }
                }

                /* set remaining strides to zero */
                for dr in new_stride_iter {
                    *dr = 0;
                }
            }
            Some(new_stride)
        }
        let dim = dim.into_dimension();

        // Note: zero strides are safe precisely because we return an read-only view
        let broadcast_strides = match upcast(&dim, &self.dim, &self.strides) {
            Some(st) => st,
            None => return None,
        };
        unsafe { Some(ArrayView::new(self.ptr, dim, broadcast_strides)) }
    }

    /// For two arrays or views, find their common shape if possible and
    /// broadcast them as array views into that shape.
    ///
    /// Return `ShapeError` if their shapes can not be broadcast together.
    #[allow(clippy::type_complexity)]
    pub(crate) fn broadcast_with<'a, 'b, B, S2, E>(
        &'a self, other: &'b ArrayBase<S2, E>,
    ) -> Result<(ArrayView<'a, A, DimMaxOf<D, E>>, ArrayView<'b, B, DimMaxOf<D, E>>), ShapeError>
    where
        S: Data<Elem = A>,
        S2: Data<Elem = B>,
        D: Dimension + DimMax<E>,
        E: Dimension,
    {
        let shape = co_broadcast::<D, E, <D as DimMax<E>>::Output>(&self.dim, &other.dim)?;
        let view1 = if shape.slice() == self.dim.slice() {
            self.view()
                .into_dimensionality::<<D as DimMax<E>>::Output>()
                .unwrap()
        } else if let Some(view1) = self.broadcast(shape.clone()) {
            view1
        } else {
            return Err(from_kind(ErrorKind::IncompatibleShape));
        };
        let view2 = if shape.slice() == other.dim.slice() {
            other
                .view()
                .into_dimensionality::<<D as DimMax<E>>::Output>()
                .unwrap()
        } else if let Some(view2) = other.broadcast(shape) {
            view2
        } else {
            return Err(from_kind(ErrorKind::IncompatibleShape));
        };
        Ok((view1, view2))
    }

    /// Swap axes `ax` and `bx`.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions
    /// and strides.
    ///
    /// **Panics** if the axes are out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let mut a = arr2(&[[1., 2., 3.]]);
    /// a.swap_axes(0, 1);
    /// assert!(
    ///     a == arr2(&[[1.], [2.], [3.]])
    /// );
    /// ```
    #[track_caller]
    pub fn swap_axes(&mut self, ax: usize, bx: usize)
    {
        self.dim.slice_mut().swap(ax, bx);
        self.strides.slice_mut().swap(ax, bx);
    }

    /// Permute the axes.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions
    /// and strides.
    ///
    /// *i* in the *j*-th place in the axes sequence means `self`'s *i*-th axis
    /// becomes `self.permuted_axes()`'s *j*-th axis
    ///
    /// **Panics** if any of the axes are out of bounds, if an axis is missing,
    /// or if an axis is repeated more than once.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{arr2, Array3};
    ///
    /// let a = arr2(&[[0, 1], [2, 3]]);
    /// assert_eq!(a.view().permuted_axes([1, 0]), a.t());
    ///
    /// let b = Array3::<u8>::zeros((1, 2, 3));
    /// assert_eq!(b.permuted_axes([1, 0, 2]).shape(), &[2, 1, 3]);
    /// ```
    #[track_caller]
    pub fn permuted_axes<T>(self, axes: T) -> ArrayBase<S, D>
    where T: IntoDimension<Dim = D>
    {
        let axes = axes.into_dimension();
        // Ensure that each axis is used exactly once.
        let mut usage_counts = D::zeros(self.ndim());
        for axis in axes.slice() {
            usage_counts[*axis] += 1;
        }
        for count in usage_counts.slice() {
            assert_eq!(*count, 1, "each axis must be listed exactly once");
        }
        // Determine the new shape and strides.
        let mut new_dim = usage_counts; // reuse to avoid an allocation
        let mut new_strides = D::zeros(self.ndim());
        {
            let dim = self.dim.slice();
            let strides = self.strides.slice();
            for (new_axis, &axis) in axes.slice().iter().enumerate() {
                new_dim[new_axis] = dim[axis];
                new_strides[new_axis] = strides[axis];
            }
        }
        // safe because axis invariants are checked above; they are a permutation of the old
        unsafe { self.with_strides_dim(new_strides, new_dim) }
    }

    /// Transpose the array by reversing axes.
    ///
    /// Transposition reverses the order of the axes (dimensions and strides)
    /// while retaining the same data.
    pub fn reversed_axes(mut self) -> ArrayBase<S, D>
    {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
        self
    }

    /// Return a transposed view of the array.
    ///
    /// This is a shorthand for `self.view().reversed_axes()`.
    ///
    /// See also the more general methods `.reversed_axes()` and `.swap_axes()`.
    pub fn t(&self) -> ArrayView<'_, A, D>
    where S: Data
    {
        self.view().reversed_axes()
    }

    /// Return an iterator over the length and stride of each axis.
    pub fn axes(&self) -> Axes<'_, D>
    {
        axes_of(&self.dim, &self.strides)
    }

    /*
    /// Return the axis with the least stride (by absolute value)
    pub fn min_stride_axis(&self) -> Axis {
        self.dim.min_stride_axis(&self.strides)
    }
    */

    /// Return the axis with the greatest stride (by absolute value),
    /// preferring axes with len > 1.
    pub fn max_stride_axis(&self) -> Axis
    {
        self.dim.max_stride_axis(&self.strides)
    }

    /// Reverse the stride of `axis`.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn invert_axis(&mut self, axis: Axis)
    {
        unsafe {
            let s = self.strides.axis(axis) as Ixs;
            let m = self.dim.axis(axis);
            if m != 0 {
                self.ptr = self.ptr.offset(stride_offset(m - 1, s as Ix));
            }
            self.strides.set_axis(axis, (-s) as Ix);
        }
    }

    /// If possible, merge in the axis `take` to `into`.
    ///
    /// Returns `true` iff the axes are now merged.
    ///
    /// This method merges the axes if movement along the two original axes
    /// (moving fastest along the `into` axis) can be equivalently represented
    /// as movement along one (merged) axis. Merging the axes preserves this
    /// order in the merged axis. If `take` and `into` are the same axis, then
    /// the axis is "merged" if its length is ≤ 1.
    ///
    /// If the return value is `true`, then the following hold:
    ///
    /// * The new length of the `into` axis is the product of the original
    ///   lengths of the two axes.
    ///
    /// * The new length of the `take` axis is 0 if the product of the original
    ///   lengths of the two axes is 0, and 1 otherwise.
    ///
    /// If the return value is `false`, then merging is not possible, and the
    /// original shape and strides have been preserved.
    ///
    /// Note that the ordering constraint means that if it's possible to merge
    /// `take` into `into`, it's usually not possible to merge `into` into
    /// `take`, and vice versa.
    ///
    /// ```
    /// use ndarray::Array3;
    /// use ndarray::Axis;
    ///
    /// let mut a = Array3::<f64>::zeros((2, 3, 4));
    /// assert!(a.merge_axes(Axis(1), Axis(2)));
    /// assert_eq!(a.shape(), &[2, 1, 12]);
    /// ```
    ///
    /// ***Panics*** if an axis is out of bounds.
    #[track_caller]
    pub fn merge_axes(&mut self, take: Axis, into: Axis) -> bool
    {
        merge_axes(&mut self.dim, &mut self.strides, take, into)
    }

    /// Insert new array axis at `axis` and return the result.
    ///
    /// ```
    /// use ndarray::{Array3, Axis, arr1, arr2};
    ///
    /// // Convert a 1-D array into a row vector (2-D).
    /// let a = arr1(&[1, 2, 3]);
    /// let row = a.insert_axis(Axis(0));
    /// assert_eq!(row, arr2(&[[1, 2, 3]]));
    ///
    /// // Convert a 1-D array into a column vector (2-D).
    /// let b = arr1(&[1, 2, 3]);
    /// let col = b.insert_axis(Axis(1));
    /// assert_eq!(col, arr2(&[[1], [2], [3]]));
    ///
    /// // The new axis always has length 1.
    /// let b = Array3::<f64>::zeros((3, 4, 5));
    /// assert_eq!(b.insert_axis(Axis(2)).shape(), &[3, 4, 1, 5]);
    /// ```
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn insert_axis(self, axis: Axis) -> ArrayBase<S, D::Larger>
    {
        assert!(axis.index() <= self.ndim());
        // safe because a new axis of length one does not affect memory layout
        unsafe {
            let strides = self.strides.insert_axis(axis);
            let dim = self.dim.insert_axis(axis);
            self.with_strides_dim(strides, dim)
        }
    }

    /// Remove array axis `axis` and return the result.
    ///
    /// This is equivalent to `.index_axis_move(axis, 0)` and makes most sense to use if the
    /// axis to remove is of length 1.
    ///
    /// **Panics** if the axis is out of bounds or its length is zero.
    #[track_caller]
    pub fn remove_axis(self, axis: Axis) -> ArrayBase<S, D::Smaller>
    where D: RemoveAxis
    {
        self.index_axis_move(axis, 0)
    }

    pub(crate) fn pointer_is_inbounds(&self) -> bool
    {
        self.data._is_pointer_inbounds(self.as_ptr())
    }

    /// Perform an elementwise assigment to `self` from `rhs`.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    #[track_caller]
    pub fn assign<E: Dimension, S2>(&mut self, rhs: &ArrayBase<S2, E>)
    where
        S: DataMut,
        A: Clone,
        S2: Data<Elem = A>,
    {
        self.zip_mut_with(rhs, |x, y| x.clone_from(y));
    }

    /// Perform an elementwise assigment of values cloned from `self` into array or producer `to`.
    ///
    /// The destination `to` can be another array or a producer of assignable elements.
    /// [`AssignElem`] determines how elements are assigned.
    ///
    /// **Panics** if shapes disagree.
    #[track_caller]
    pub fn assign_to<P>(&self, to: P)
    where
        S: Data,
        P: IntoNdProducer<Dim = D>,
        P::Item: AssignElem<A>,
        A: Clone,
    {
        Zip::from(self).map_assign_into(to, A::clone);
    }

    /// Perform an elementwise assigment to `self` from element `x`.
    pub fn fill(&mut self, x: A)
    where
        S: DataMut,
        A: Clone,
    {
        self.map_inplace(move |elt| elt.clone_from(&x));
    }

    pub(crate) fn zip_mut_with_same_shape<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
    where
        S: DataMut,
        S2: Data<Elem = B>,
        E: Dimension,
        F: FnMut(&mut A, &B),
    {
        debug_assert_eq!(self.shape(), rhs.shape());

        if self.dim.strides_equivalent(&self.strides, &rhs.strides) {
            if let Some(self_s) = self.as_slice_memory_order_mut() {
                if let Some(rhs_s) = rhs.as_slice_memory_order() {
                    for (s, r) in self_s.iter_mut().zip(rhs_s) {
                        f(s, r);
                    }
                    return;
                }
            }
        }

        // Otherwise, fall back to the outer iter
        self.zip_mut_with_by_rows(rhs, f);
    }

    // zip two arrays where they have different layout or strides
    #[inline(always)]
    fn zip_mut_with_by_rows<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
    where
        S: DataMut,
        S2: Data<Elem = B>,
        E: Dimension,
        F: FnMut(&mut A, &B),
    {
        debug_assert_eq!(self.shape(), rhs.shape());
        debug_assert_ne!(self.ndim(), 0);

        // break the arrays up into their inner rows
        let n = self.ndim();
        let dim = self.raw_dim();
        Zip::from(LanesMut::new(self.view_mut(), Axis(n - 1)))
            .and(Lanes::new(rhs.broadcast_assume(dim), Axis(n - 1)))
            .for_each(move |s_row, r_row| Zip::from(s_row).and(r_row).for_each(&mut f));
    }

    fn zip_mut_with_elem<B, F>(&mut self, rhs_elem: &B, mut f: F)
    where
        S: DataMut,
        F: FnMut(&mut A, &B),
    {
        self.map_inplace(move |elt| f(elt, rhs_elem));
    }

    /// Traverse two arrays in unspecified order, in lock step,
    /// calling the closure `f` on each element pair.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    #[track_caller]
    #[inline]
    pub fn zip_mut_with<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, f: F)
    where
        S: DataMut,
        S2: Data<Elem = B>,
        E: Dimension,
        F: FnMut(&mut A, &B),
    {
        if rhs.dim.ndim() == 0 {
            // Skip broadcast from 0-dim array
            self.zip_mut_with_elem(rhs.get_0d(), f);
        } else if self.dim.ndim() == rhs.dim.ndim() && self.shape() == rhs.shape() {
            self.zip_mut_with_same_shape(rhs, f);
        } else {
            let rhs_broadcast = rhs.broadcast_unwrap(self.raw_dim());
            self.zip_mut_with_by_rows(&rhs_broadcast, f);
        }
    }

    /// Traverse the array elements and apply a fold,
    /// returning the resulting value.
    ///
    /// Elements are visited in arbitrary order.
    pub fn fold<'a, F, B>(&'a self, init: B, f: F) -> B
    where
        F: FnMut(B, &'a A) -> B,
        A: 'a,
        S: Data,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            slc.iter().fold(init, f)
        } else {
            let mut v = self.view();
            move_min_stride_axis_to_last(&mut v.dim, &mut v.strides);
            v.into_elements_base().fold(init, f)
        }
    }

    /// Call `f` by reference on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[ 0., 1.],
    ///                [-1., 2.]]);
    /// assert!(
    ///     a.map(|x| *x >= 1.0)
    ///     == arr2(&[[false, true],
    ///               [false, true]])
    /// );
    /// ```
    pub fn map<'a, B, F>(&'a self, f: F) -> Array<B, D>
    where
        F: FnMut(&'a A) -> B,
        A: 'a,
        S: Data,
    {
        unsafe {
            if let Some(slc) = self.as_slice_memory_order() {
                ArrayBase::from_shape_trusted_iter_unchecked(
                    self.dim.clone().strides(self.strides.clone()),
                    slc.iter(),
                    f,
                )
            } else {
                ArrayBase::from_shape_trusted_iter_unchecked(self.dim.clone(), self.iter(), f)
            }
        }
    }

    /// Call `f` on a mutable reference of each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    pub fn map_mut<'a, B, F>(&'a mut self, f: F) -> Array<B, D>
    where
        F: FnMut(&'a mut A) -> B,
        A: 'a,
        S: DataMut,
    {
        let dim = self.dim.clone();
        if self.is_contiguous() {
            let strides = self.strides.clone();
            let slc = self.as_slice_memory_order_mut().unwrap();
            unsafe { ArrayBase::from_shape_trusted_iter_unchecked(dim.strides(strides), slc.iter_mut(), f) }
        } else {
            unsafe { ArrayBase::from_shape_trusted_iter_unchecked(dim, self.iter_mut(), f) }
        }
    }

    /// Call `f` by **v**alue on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[ 0., 1.],
    ///                [-1., 2.]]);
    /// assert!(
    ///     a.mapv(f32::abs) == arr2(&[[0., 1.],
    ///                                [1., 2.]])
    /// );
    /// ```
    pub fn mapv<B, F>(&self, mut f: F) -> Array<B, D>
    where
        F: FnMut(A) -> B,
        A: Clone,
        S: Data,
    {
        self.map(move |x| f(x.clone()))
    }

    /// Call `f` by **v**alue on each element, update the array with the new values
    /// and return it.
    ///
    /// Elements are visited in arbitrary order.
    pub fn mapv_into<F>(mut self, f: F) -> Self
    where
        S: DataMut,
        F: FnMut(A) -> A,
        A: Clone,
    {
        self.mapv_inplace(f);
        self
    }

    /// Consume the array, call `f` by **v**alue on each element, and return an
    /// owned array with the new values. Works for **any** `F: FnMut(A)->B`.
    ///
    /// If `A` and `B` are the same type then the map is performed by delegating
    /// to [`mapv_into`] and then converting into an owned array. This avoids
    /// unnecessary memory allocations in [`mapv`].
    ///
    /// If `A` and `B` are different types then a new array is allocated and the
    /// map is performed as in [`mapv`].
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// [`mapv_into`]: ArrayBase::mapv_into
    /// [`mapv`]: ArrayBase::mapv
    pub fn mapv_into_any<B, F>(self, mut f: F) -> Array<B, D>
    where
        S: DataMut,
        F: FnMut(A) -> B,
        A: Clone + 'static,
        B: 'static,
    {
        if core::any::TypeId::of::<A>() == core::any::TypeId::of::<B>() {
            // A and B are the same type.
            // Wrap f in a closure of type FnMut(A) -> A .
            let f = |a| {
                let b = f(a);
                // Safe because A and B are the same type.
                unsafe { unlimited_transmute::<B, A>(b) }
            };
            // Delegate to mapv_into() using the wrapped closure.
            // Convert output to a uniquely owned array of type Array<A, D>.
            let output = self.mapv_into(f).into_owned();
            // Change the return type from Array<A, D> to Array<B, D>.
            // Again, safe because A and B are the same type.
            unsafe { unlimited_transmute::<Array<A, D>, Array<B, D>>(output) }
        } else {
            // A and B are not the same type.
            // Fallback to mapv().
            self.mapv(f)
        }
    }

    /// Modify the array in place by calling `f` by mutable reference on each element.
    ///
    /// Elements are visited in arbitrary order.
    pub fn map_inplace<'a, F>(&'a mut self, f: F)
    where
        S: DataMut,
        A: 'a,
        F: FnMut(&'a mut A),
    {
        match self.try_as_slice_memory_order_mut() {
            Ok(slc) => slc.iter_mut().for_each(f),
            Err(arr) => {
                let mut v = arr.view_mut();
                move_min_stride_axis_to_last(&mut v.dim, &mut v.strides);
                v.into_elements_base().for_each(f);
            }
        }
    }

    /// Modify the array in place by calling `f` by **v**alue on each element.
    /// The array is updated with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// ```
    /// # #[cfg(feature = "approx")] {
    /// use approx::assert_abs_diff_eq;
    /// use ndarray::arr2;
    ///
    /// let mut a = arr2(&[[ 0., 1.],
    ///                    [-1., 2.]]);
    /// a.mapv_inplace(f32::exp);
    /// assert_abs_diff_eq!(
    ///     a,
    ///     arr2(&[[1.00000, 2.71828],
    ///            [0.36788, 7.38906]]),
    ///     epsilon = 1e-5,
    /// );
    /// # }
    /// ```
    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        S: DataMut,
        F: FnMut(A) -> A,
        A: Clone,
    {
        self.map_inplace(move |x| *x = f(x.clone()));
    }

    /// Call `f` for each element in the array.
    ///
    /// Elements are visited in arbitrary order.
    pub fn for_each<'a, F>(&'a self, mut f: F)
    where
        F: FnMut(&'a A),
        A: 'a,
        S: Data,
    {
        self.fold((), move |(), elt| f(elt))
    }

    /// Fold along an axis.
    ///
    /// Combine the elements of each subview with the previous using the `fold`
    /// function and initial value `init`.
    ///
    /// Return the result as an `Array`.
    ///
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn fold_axis<B, F>(&self, axis: Axis, init: B, mut fold: F) -> Array<B, D::Smaller>
    where
        D: RemoveAxis,
        F: FnMut(&B, &A) -> B,
        B: Clone,
        S: Data,
    {
        let mut res = Array::from_elem(self.raw_dim().remove_axis(axis), init);
        for subview in self.axis_iter(axis) {
            res.zip_mut_with(&subview, |x, y| *x = fold(x, y));
        }
        res
    }

    /// Reduce the values along an axis into just one value, producing a new
    /// array with one less dimension.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return the result as an `Array`.
    ///
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn map_axis<'a, B, F>(&'a self, axis: Axis, mut mapping: F) -> Array<B, D::Smaller>
    where
        D: RemoveAxis,
        F: FnMut(ArrayView1<'a, A>) -> B,
        A: 'a,
        S: Data,
    {
        if self.len_of(axis) == 0 {
            let new_dim = self.dim.remove_axis(axis);
            Array::from_shape_simple_fn(new_dim, move || mapping(ArrayView::from(&[])))
        } else {
            Zip::from(self.lanes(axis)).map_collect(mapping)
        }
    }

    /// Reduce the values along an axis into just one value, producing a new
    /// array with one less dimension.
    /// 1-dimensional lanes are passed as mutable references to the reducer,
    /// allowing for side-effects.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return the result as an `Array`.
    ///
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn map_axis_mut<'a, B, F>(&'a mut self, axis: Axis, mut mapping: F) -> Array<B, D::Smaller>
    where
        D: RemoveAxis,
        F: FnMut(ArrayViewMut1<'a, A>) -> B,
        A: 'a,
        S: DataMut,
    {
        if self.len_of(axis) == 0 {
            let new_dim = self.dim.remove_axis(axis);
            Array::from_shape_simple_fn(new_dim, move || mapping(ArrayViewMut::from(&mut [])))
        } else {
            Zip::from(self.lanes_mut(axis)).map_collect(mapping)
        }
    }

    /// Remove the `index`th elements along `axis` and shift down elements from higher indexes.
    ///
    /// Note that this "removes" the elements by swapping them around to the end of the axis and
    /// shortening the length of the axis; the elements are not deinitialized or dropped by this,
    /// just moved out of view (this only matters for elements with ownership semantics). It's
    /// similar to slicing an owned array in place.
    ///
    /// Decreases the length of `axis` by one.
    ///
    /// ***Panics*** if `axis` is out of bounds<br>
    /// ***Panics*** if not `index < self.len_of(axis)`.
    pub fn remove_index(&mut self, axis: Axis, index: usize)
    where S: DataOwned + DataMut
    {
        assert!(index < self.len_of(axis), "index {} must be less than length of Axis({})",
                index, axis.index());
        let (_, mut tail) = self.view_mut().split_at(axis, index);
        // shift elements to the front
        Zip::from(tail.lanes_mut(axis)).for_each(|mut lane| lane.rotate1_front());
        // then slice the axis in place to cut out the removed final element
        self.slice_axis_inplace(axis, Slice::new(0, Some(-1), 1));
    }

    /// Iterates over pairs of consecutive elements along the axis.
    ///
    /// The first argument to the closure is an element, and the second
    /// argument is the next element along the axis. Iteration is guaranteed to
    /// proceed in order along the specified axis, but in all other respects
    /// the iteration order is unspecified.
    ///
    /// # Example
    ///
    /// For example, this can be used to compute the cumulative sum along an
    /// axis:
    ///
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let mut arr = array![
    ///     [[1, 2], [3, 4], [5, 6]],
    ///     [[7, 8], [9, 10], [11, 12]],
    /// ];
    /// arr.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    /// assert_eq!(
    ///     arr,
    ///     array![
    ///         [[1, 2], [4, 6], [9, 12]],
    ///         [[7, 8], [16, 18], [27, 30]],
    ///     ],
    /// );
    /// ```
    pub fn accumulate_axis_inplace<F>(&mut self, axis: Axis, mut f: F)
    where
        F: FnMut(&A, &mut A),
        S: DataMut,
    {
        if self.len_of(axis) <= 1 {
            return;
        }
        let mut curr = self.raw_view_mut(); // mut borrow of the array here
        let mut prev = curr.raw_view(); // derive further raw views from the same borrow
        prev.slice_axis_inplace(axis, Slice::from(..-1));
        curr.slice_axis_inplace(axis, Slice::from(1..));
        // This implementation relies on `Zip` iterating along `axis` in order.
        Zip::from(prev).and(curr).for_each(|prev, curr| unsafe {
            // These pointer dereferences and borrows are safe because:
            //
            // 1. They're pointers to elements in the array.
            //
            // 2. `S: DataMut` guarantees that elements are safe to borrow
            //    mutably and that they don't alias.
            //
            // 3. The lifetimes of the borrows last only for the duration
            //    of the call to `f`, so aliasing across calls to `f`
            //    cannot occur.
            f(&*prev, &mut *curr)
        });
    }
}

/// Transmute from A to B.
///
/// Like transmute, but does not have the compile-time size check which blocks
/// using regular transmute in some cases.
///
/// **Panics** if the size of A and B are different.
#[track_caller]
#[inline]
unsafe fn unlimited_transmute<A, B>(data: A) -> B
{
    // safe when sizes are equal and caller guarantees that representations are equal
    assert_eq!(size_of::<A>(), size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}

type DimMaxOf<A, B> = <A as DimMax<B>>::Output;

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::arr3;

    #[test]
    fn test_flatten()
    {
        let array = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let flattened = array.flatten();
        assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_flatten_with_order()
    {
        let array = arr2(&[[1, 2], [3, 4], [5, 6], [7, 8]]);
        let flattened = array.flatten_with_order(Order::RowMajor);
        assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let flattened = array.flatten_with_order(Order::ColumnMajor);
        assert_eq!(flattened, arr1(&[1, 3, 5, 7, 2, 4, 6, 8]));
    }

    #[test]
    fn test_into_flat()
    {
        let array = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        let flattened = array.into_flat();
        assert_eq!(flattened, arr1(&[1, 2, 3, 4, 5, 6, 7, 8]));
    }
}
