#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ptr::NonNull;
use std::mem;
use std::mem::MaybeUninit;

#[allow(unused_imports)] // Needed for Rust 1.64
use rawpointer::PointerExt;

use crate::imp_prelude::*;

use crate::dimension;
use crate::error::{ErrorKind, ShapeError};
use crate::iterators::Baseiter;
use crate::low_level_util::AbortIfPanic;
use crate::OwnedRepr;
use crate::Zip;

/// Methods specific to `Array0`.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A> Array<A, Ix0>
{
    /// Returns the single element in the array without cloning it.
    ///
    /// ```
    /// use ndarray::{arr0, Array0};
    ///
    /// // `Foo` doesn't implement `Clone`.
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct Foo;
    ///
    /// let array: Array0<Foo> = arr0(Foo);
    /// let scalar: Foo = array.into_scalar();
    /// assert_eq!(scalar, Foo);
    /// ```
    pub fn into_scalar(self) -> A
    {
        let size = mem::size_of::<A>();
        if size == 0 {
            // Any index in the `Vec` is fine since all elements are identical.
            self.data.into_vec().remove(0)
        } else {
            // Find the index in the `Vec` corresponding to `self.ptr`.
            // (This is necessary because the element in the array might not be
            // the first element in the `Vec`, such as if the array was created
            // by `array![1, 2, 3, 4].slice_move(s![2])`.)
            let first = self.ptr.as_ptr() as usize;
            let base = self.data.as_ptr() as usize;
            let index = (first - base) / size;
            debug_assert_eq!((first - base) % size, 0);
            // Remove the element at the index and return it.
            self.data.into_vec().remove(index)
        }
    }
}

/// Methods specific to `Array`.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A, D> Array<A, D>
where D: Dimension
{
    /// Returns the offset (in units of `A`) from the start of the allocation
    /// to the first element, or `None` if the array is empty.
    fn offset_from_alloc_to_logical_ptr(&self) -> Option<usize>
    {
        if self.is_empty() {
            return None;
        }
        if std::mem::size_of::<A>() == 0 {
            Some(dimension::offset_from_low_addr_ptr_to_logical_ptr(&self.dim, &self.strides))
        } else {
            let offset = unsafe { self.as_ptr().offset_from(self.data.as_ptr()) };
            debug_assert!(offset >= 0);
            Some(offset as usize)
        }
    }

    /// Return a vector of the elements in the array, in the way they are
    /// stored internally, and the index in the vector corresponding to the
    /// logically first element of the array (or 0 if the array is empty).
    ///
    /// If the array is in standard memory layout, the logical element order
    /// of the array (`.iter()` order) and of the returned vector will be the same.
    ///
    /// ```
    /// use ndarray::{array, Array2, Axis};
    ///
    /// let mut arr: Array2<f64> = array![[1., 2.], [3., 4.], [5., 6.]];
    /// arr.slice_axis_inplace(Axis(0), (1..).into());
    /// assert_eq!(arr[[0, 0]], 3.);
    /// let copy = arr.clone();
    ///
    /// let shape = arr.shape().to_owned();
    /// let strides = arr.strides().to_owned();
    /// let (v, offset) = arr.into_raw_vec_and_offset();
    ///
    /// assert_eq!(v, &[1., 2., 3., 4., 5., 6.]);
    /// assert_eq!(offset, Some(2));
    /// assert_eq!(v[offset.unwrap()], 3.);
    /// for row in 0..shape[0] {
    ///     for col in 0..shape[1] {
    ///         let index = (
    ///             offset.unwrap() as isize
    ///             + row as isize * strides[0]
    ///             + col as isize * strides[1]
    ///         ) as usize;
    ///         assert_eq!(v[index], copy[[row, col]]);
    ///     }
    /// }
    /// ```
    ///
    /// In the case of zero-sized elements, the offset to the logically first
    /// element is somewhat meaningless. For convenience, an offset will be
    /// returned such that all indices computed using the offset, shape, and
    /// strides will be in-bounds for the `Vec<A>`. Note that this offset won't
    /// necessarily be the same as the offset for an array of nonzero-sized
    /// elements sliced in the same way.
    ///
    /// ```
    /// use ndarray::{array, Array2, Axis};
    ///
    /// let mut arr: Array2<()> = array![[(), ()], [(), ()], [(), ()]];
    /// arr.slice_axis_inplace(Axis(0), (1..).into());
    ///
    /// let shape = arr.shape().to_owned();
    /// let strides = arr.strides().to_owned();
    /// let (v, offset) = arr.into_raw_vec_and_offset();
    ///
    /// assert_eq!(v, &[(), (), (), (), (), ()]);
    /// for row in 0..shape[0] {
    ///     for col in 0..shape[1] {
    ///         let index = (
    ///             offset.unwrap() as isize
    ///             + row as isize * strides[0]
    ///             + col as isize * strides[1]
    ///         ) as usize;
    ///         assert_eq!(v[index], ());
    ///     }
    /// }
    /// ```
    pub fn into_raw_vec_and_offset(self) -> (Vec<A>, Option<usize>)
    {
        let offset = self.offset_from_alloc_to_logical_ptr();
        (self.data.into_vec(), offset)
    }

    /// Return a vector of the elements in the array, in the way they are
    /// stored internally.
    ///
    /// Depending on slicing and strides, the logically first element of the
    /// array can be located at an offset. Because of this, prefer to use
    /// `.into_raw_vec_and_offset()` instead.
    #[deprecated(note = "Use .into_raw_vec_and_offset() instead", since = "0.16.0")]
    pub fn into_raw_vec(self) -> Vec<A>
    {
        self.into_raw_vec_and_offset().0
    }
}

/// Methods specific to `Array2`.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A> Array<A, Ix2>
{
    /// Append a row to an array
    ///
    /// The elements from `row` are cloned and added as a new row in the array.
    ///
    /// ***Errors*** with a shape error if the length of the row does not match the length of the
    /// rows in the array.
    ///
    /// The memory layout of the `self` array matters for ensuring that the append is efficient.
    /// Appending automatically changes memory layout of the array so that it is appended to
    /// along the "growing axis". However, if the memory layout needs adjusting, the array must
    /// reallocate and move memory.
    ///
    /// The operation leaves the existing data in place and is most efficent if one of these is
    /// true:
    ///
    /// - The axis being appended to is the longest stride axis, i.e the array is in row major
    ///   ("C") layout.
    /// - The array has 0 or 1 rows (It is converted to row major)
    ///
    /// Ensure appending is efficient by, for example, appending to an empty array and then always
    /// pushing/appending along the same axis. For pushing rows, ndarray's default layout (C order)
    /// is efficient.
    ///
    /// When repeatedly appending to a single axis, the amortized average complexity of each
    /// append is O(m), where *m* is the length of the row.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array};
    ///
    /// // create an empty array and append
    /// let mut a = Array::zeros((0, 4));
    /// a.push_row(ArrayView::from(&[ 1.,  2.,  3.,  4.])).unwrap();
    /// a.push_row(ArrayView::from(&[-1., -2., -3., -4.])).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[ 1.,  2.,  3.,  4.],
    ///            [-1., -2., -3., -4.]]);
    /// ```
    pub fn push_row(&mut self, row: ArrayView<A, Ix1>) -> Result<(), ShapeError>
    where A: Clone
    {
        self.append(Axis(0), row.insert_axis(Axis(0)))
    }

    /// Append a column to an array
    ///
    /// The elements from `column` are cloned and added as a new column in the array.
    ///
    /// ***Errors*** with a shape error if the length of the column does not match the length of
    /// the columns in the array.
    ///
    /// The memory layout of the `self` array matters for ensuring that the append is efficient.
    /// Appending automatically changes memory layout of the array so that it is appended to
    /// along the "growing axis". However, if the memory layout needs adjusting, the array must
    /// reallocate and move memory.
    ///
    /// The operation leaves the existing data in place and is most efficent if one of these is
    /// true:
    ///
    /// - The axis being appended to is the longest stride axis, i.e the array is in column major
    ///   ("F") layout.
    /// - The array has 0 or 1 columns (It is converted to column major)
    ///
    /// Ensure appending is efficient by, for example, appending to an empty array and then always
    /// pushing/appending along the same axis. For pushing columns, column major layout (F order)
    /// is efficient.
    ///
    /// When repeatedly appending to a single axis, the amortized average complexity of each append
    /// is O(m), where *m* is the length of the column.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array};
    ///
    /// // create an empty array and append
    /// let mut a = Array::zeros((2, 0));
    /// a.push_column(ArrayView::from(&[1., 2.])).unwrap();
    /// a.push_column(ArrayView::from(&[-1., -2.])).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[1., -1.],
    ///            [2., -2.]]);
    /// ```
    pub fn push_column(&mut self, column: ArrayView<A, Ix1>) -> Result<(), ShapeError>
    where A: Clone
    {
        self.append(Axis(1), column.insert_axis(Axis(1)))
    }

    /// Reserve capacity to grow array by at least `additional` rows.
    ///
    /// Existing elements of `array` are untouched and the backing storage is grown by
    /// calling the underlying `reserve` method of the `OwnedRepr`.
    ///
    /// This is useful when pushing or appending repeatedly to an array to avoid multiple
    /// allocations.
    ///
    /// ***Errors*** with a shape error if the resultant capacity is larger than the addressable
    /// bounds; that is, the product of non-zero axis lengths once `axis` has been extended by
    /// `additional` exceeds `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::Array2;
    /// let mut a = Array2::<i32>::zeros((2,4));
    /// a.reserve_rows(1000).unwrap();
    /// assert!(a.into_raw_vec().capacity() >= 4*1002);
    /// ```
    pub fn reserve_rows(&mut self, additional: usize) -> Result<(), ShapeError>
    {
        self.reserve(Axis(0), additional)
    }

    /// Reserve capacity to grow array by at least `additional` columns.
    ///
    /// Existing elements of `array` are untouched and the backing storage is grown by
    /// calling the underlying `reserve` method of the `OwnedRepr`.
    ///
    /// This is useful when pushing or appending repeatedly to an array to avoid multiple
    /// allocations.
    ///
    /// ***Errors*** with a shape error if the resultant capacity is larger than the addressable
    /// bounds; that is, the product of non-zero axis lengths once `axis` has been extended by
    /// `additional` exceeds `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::Array2;
    /// let mut a = Array2::<i32>::zeros((2,4));
    /// a.reserve_columns(1000).unwrap();
    /// assert!(a.into_raw_vec().capacity() >= 2*1002);
    /// ```
    pub fn reserve_columns(&mut self, additional: usize) -> Result<(), ShapeError>
    {
        self.reserve(Axis(1), additional)
    }
}

impl<A, D> Array<A, D>
where D: Dimension
{
    /// Move all elements from self into `new_array`, which must be of the same shape but
    /// can have a different memory layout. The destination is overwritten completely.
    ///
    /// The destination should be a mut reference to an array or an `ArrayViewMut` with
    /// `A` elements.
    ///
    /// ***Panics*** if the shapes don't agree.
    ///
    /// ## Example
    ///
    /// ```
    /// use ndarray::Array;
    ///
    /// // Usage example of move_into in safe code
    /// let mut a = Array::default((10, 10));
    /// let b = Array::from_shape_fn((10, 10), |(i, j)| (i + j).to_string());
    /// b.move_into(&mut a);
    /// ```
    pub fn move_into<'a, AM>(self, new_array: AM)
    where
        AM: Into<ArrayViewMut<'a, A, D>>,
        A: 'a,
    {
        // Remove generic parameter P and call the implementation
        let new_array = new_array.into();
        if mem::needs_drop::<A>() {
            self.move_into_needs_drop(new_array);
        } else {
            // If `A` doesn't need drop, we can overwrite the destination.
            // Safe because: move_into_uninit only writes initialized values
            unsafe { self.move_into_uninit(new_array.into_maybe_uninit()) }
        }
    }

    fn move_into_needs_drop(mut self, new_array: ArrayViewMut<A, D>)
    {
        // Simple case where `A` has a destructor: just swap values between self and new_array.
        // Afterwards, `self` drops full of initialized values and dropping works as usual.
        // This avoids moving out of owned values in `self` while at the same time managing
        // the dropping if the values being overwritten in `new_array`.
        Zip::from(&mut self)
            .and(new_array)
            .for_each(|src, dst| mem::swap(src, dst));
    }

    /// Move all elements from self into `new_array`, which must be of the same shape but
    /// can have a different memory layout. The destination is overwritten completely.
    ///
    /// The destination should be a mut reference to an array or an `ArrayViewMut` with
    /// `MaybeUninit<A>` elements (which are overwritten without dropping any existing value).
    ///
    /// Minor implementation note: Owned arrays like `self` may be sliced in place and own elements
    /// that are not part of their active view; these are dropped at the end of this function,
    /// after all elements in the "active view" are moved into `new_array`. If there is a panic in
    /// drop of any such element, other elements may be leaked.
    ///
    /// ***Panics*** if the shapes don't agree.
    ///
    /// ## Example
    ///
    /// ```
    /// use ndarray::Array;
    ///
    /// let a = Array::from_iter(0..100).into_shape_with_order((10, 10)).unwrap();
    /// let mut b = Array::uninit((10, 10));
    /// a.move_into_uninit(&mut b);
    /// unsafe {
    ///     // we can now promise we have fully initialized `b`.
    ///     let b = b.assume_init();
    /// }
    /// ```
    pub fn move_into_uninit<'a, AM>(self, new_array: AM)
    where
        AM: Into<ArrayViewMut<'a, MaybeUninit<A>, D>>,
        A: 'a,
    {
        // Remove generic parameter AM and call the implementation
        self.move_into_impl(new_array.into())
    }

    fn move_into_impl(mut self, new_array: ArrayViewMut<MaybeUninit<A>, D>)
    {
        unsafe {
            // Safety: copy_to_nonoverlapping cannot panic
            let guard = AbortIfPanic(&"move_into: moving out of owned value");
            // Move all reachable elements; we move elements out of `self`.
            // and thus must not panic for the whole section until we call `self.data.set_len(0)`.
            Zip::from(self.raw_view_mut())
                .and(new_array)
                .for_each(|src, dst| {
                    src.copy_to_nonoverlapping(dst.as_mut_ptr(), 1);
                });
            guard.defuse();
            // Drop all unreachable elements
            self.drop_unreachable_elements();
        }
    }

    /// This drops all "unreachable" elements in the data storage of self.
    ///
    /// That means those elements that are not visible in the slicing of the array.
    /// *Reachable elements are assumed to already have been moved from.*
    ///
    /// # Safety
    ///
    /// This is a panic critical section since `self` is already moved-from.
    fn drop_unreachable_elements(mut self) -> OwnedRepr<A>
    {
        let self_len = self.len();

        // "deconstruct" self; the owned repr releases ownership of all elements and we
        // and carry on with raw view methods
        let data_len = self.data.len();

        let has_unreachable_elements = self_len != data_len;
        if !has_unreachable_elements || mem::size_of::<A>() == 0 || !mem::needs_drop::<A>() {
            unsafe {
                self.data.set_len(0);
            }
            self.data
        } else {
            self.drop_unreachable_elements_slow()
        }
    }

    #[inline(never)]
    #[cold]
    fn drop_unreachable_elements_slow(mut self) -> OwnedRepr<A>
    {
        // "deconstruct" self; the owned repr releases ownership of all elements and we
        // carry on with raw view methods
        let data_len = self.data.len();
        let data_ptr = self.data.as_nonnull_mut();

        unsafe {
            // Safety: self.data releases ownership of the elements. Any panics below this point
            // will result in leaking elements instead of double drops.
            let self_ = self.raw_view_mut();
            self.data.set_len(0);

            drop_unreachable_raw(self_, data_ptr, data_len);
        }

        self.data
    }

    /// Create an empty array with an all-zeros shape
    ///
    /// ***Panics*** if D is zero-dimensional, because it can't be empty
    pub(crate) fn empty() -> Array<A, D>
    {
        assert_ne!(D::NDIM, Some(0));
        let ndim = D::NDIM.unwrap_or(1);
        Array::from_shape_simple_fn(D::zeros(ndim), || unreachable!())
    }

    /// Create new_array with the right layout for appending to `growing_axis`
    #[cold]
    fn change_to_contig_append_layout(&mut self, growing_axis: Axis)
    {
        let ndim = self.ndim();
        let mut dim = self.raw_dim();

        // The array will be created with 0 (C) or ndim-1 (F) as the biggest stride
        // axis. Rearrange the shape so that `growing_axis` is the biggest stride axis
        // afterwards.
        let mut new_array;
        if growing_axis == Axis(ndim - 1) {
            new_array = Self::uninit(dim.f());
        } else {
            dim.slice_mut()[..=growing_axis.index()].rotate_right(1);
            new_array = Self::uninit(dim);
            new_array.dim.slice_mut()[..=growing_axis.index()].rotate_left(1);
            new_array.strides.slice_mut()[..=growing_axis.index()].rotate_left(1);
        }

        // self -> old_self.
        // dummy array -> self.
        // old_self elements are moved -> new_array.
        let old_self = std::mem::replace(self, Self::empty());
        old_self.move_into_uninit(new_array.view_mut());

        // new_array -> self.
        unsafe {
            *self = new_array.assume_init();
        }
    }

    /// Append an array to the array along an axis.
    ///
    /// The elements of `array` are cloned and extend the axis `axis` in the present array;
    /// `self` will grow in size by 1 along `axis`.
    ///
    /// Append to the array, where the array being pushed to the array has one dimension less than
    /// the `self` array. This method is equivalent to [append](ArrayBase::append) in this way:
    /// `self.append(axis, array.insert_axis(axis))`.
    ///
    /// ***Errors*** with a shape error if the shape of self does not match the array-to-append;
    /// all axes *except* the axis along which it being appended matter for this check:
    /// the shape of `self` with `axis` removed must be the same as the shape of `array`.
    ///
    /// The memory layout of the `self` array matters for ensuring that the append is efficient.
    /// Appending automatically changes memory layout of the array so that it is appended to
    /// along the "growing axis". However, if the memory layout needs adjusting, the array must
    /// reallocate and move memory.
    ///
    /// The operation leaves the existing data in place and is most efficent if `axis` is a
    /// "growing axis" for the array, i.e. one of these is true:
    ///
    /// - The axis is the longest stride axis, for example the 0th axis in a C-layout or the
    ///   *n-1*th axis in an F-layout array.
    /// - The axis has length 0 or 1 (It is converted to the new growing axis)
    ///
    /// Ensure appending is efficient by for example starting from an empty array and/or always
    /// appending to an array along the same axis.
    ///
    /// The amortized average complexity of the append, when appending along its growing axis, is
    /// O(*m*) where *m* is the number of individual elements to append.
    ///
    /// The memory layout of the argument `array` does not matter to the same extent.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array, Axis};
    ///
    /// // create an empty array and push rows to it
    /// let mut a = Array::zeros((0, 4));
    /// let ones  = ArrayView::from(&[1.; 4]);
    /// let zeros = ArrayView::from(&[0.; 4]);
    /// a.push(Axis(0), ones).unwrap();
    /// a.push(Axis(0), zeros).unwrap();
    /// a.push(Axis(0), ones).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[1., 1., 1., 1.],
    ///            [0., 0., 0., 0.],
    ///            [1., 1., 1., 1.]]);
    /// ```
    pub fn push(&mut self, axis: Axis, array: ArrayView<A, D::Smaller>) -> Result<(), ShapeError>
    where
        A: Clone,
        D: RemoveAxis,
    {
        // same-dimensionality conversion
        self.append(axis, array.insert_axis(axis).into_dimensionality::<D>().unwrap())
    }

    /// Append an array to the array along an axis.
    ///
    /// The elements of `array` are cloned and extend the axis `axis` in the present array;
    /// `self` will grow in size by `array.len_of(axis)` along `axis`.
    ///
    /// ***Errors*** with a shape error if the shape of self does not match the array-to-append;
    /// all axes *except* the axis along which it being appended matter for this check:
    /// the shape of `self` with `axis` removed must be the same as the shape of `array` with
    /// `axis` removed.
    ///
    /// The memory layout of the `self` array matters for ensuring that the append is efficient.
    /// Appending automatically changes memory layout of the array so that it is appended to
    /// along the "growing axis". However, if the memory layout needs adjusting, the array must
    /// reallocate and move memory.
    ///
    /// The operation leaves the existing data in place and is most efficent if `axis` is a
    /// "growing axis" for the array, i.e. one of these is true:
    ///
    /// - The axis is the longest stride axis, for example the 0th axis in a C-layout or the
    ///   *n-1*th axis in an F-layout array.
    /// - The axis has length 0 or 1 (It is converted to the new growing axis)
    ///
    /// Ensure appending is efficient by for example starting from an empty array and/or always
    /// appending to an array along the same axis.
    ///
    /// The amortized average complexity of the append, when appending along its growing axis, is
    /// O(*m*) where *m* is the number of individual elements to append.
    ///
    /// The memory layout of the argument `array` does not matter to the same extent.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array, Axis};
    ///
    /// // create an empty array and append two rows at a time
    /// let mut a = Array::zeros((0, 4));
    /// let ones  = ArrayView::from(&[1.; 8]).into_shape_with_order((2, 4)).unwrap();
    /// let zeros = ArrayView::from(&[0.; 8]).into_shape_with_order((2, 4)).unwrap();
    /// a.append(Axis(0), ones).unwrap();
    /// a.append(Axis(0), zeros).unwrap();
    /// a.append(Axis(0), ones).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[1., 1., 1., 1.],
    ///            [1., 1., 1., 1.],
    ///            [0., 0., 0., 0.],
    ///            [0., 0., 0., 0.],
    ///            [1., 1., 1., 1.],
    ///            [1., 1., 1., 1.]]);
    /// ```
    pub fn append(&mut self, axis: Axis, mut array: ArrayView<A, D>) -> Result<(), ShapeError>
    where
        A: Clone,
        D: RemoveAxis,
    {
        if self.ndim() == 0 {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        let current_axis_len = self.len_of(axis);
        let self_dim = self.raw_dim();
        let array_dim = array.raw_dim();
        let remaining_shape = self_dim.remove_axis(axis);
        let array_rem_shape = array_dim.remove_axis(axis);

        if remaining_shape != array_rem_shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        let len_to_append = array.len();

        let mut res_dim = self_dim;
        res_dim[axis.index()] += array_dim[axis.index()];
        let new_len = dimension::size_of_shape_checked(&res_dim)?;

        if len_to_append == 0 {
            // There are no elements to append and shapes are compatible:
            // either the dimension increment is zero, or there is an existing
            // zero in another axis in self.
            debug_assert_eq!(self.len(), new_len);
            self.dim = res_dim;
            return Ok(());
        }

        let self_is_empty = self.is_empty();
        let mut incompatible_layout = false;

        // array must be empty or have `axis` as the outermost (longest stride) axis
        if !self_is_empty && current_axis_len > 1 {
            // `axis` must be max stride axis or equal to its stride
            let axis_stride = self.stride_of(axis);
            if axis_stride < 0 {
                incompatible_layout = true;
            } else {
                for ax in self.axes() {
                    if ax.axis == axis {
                        continue;
                    }
                    if ax.len > 1 && ax.stride.abs() > axis_stride {
                        incompatible_layout = true;
                        break;
                    }
                }
            }
        }

        // array must be be "full" (contiguous and have no exterior holes)
        if self.len() != self.data.len() {
            incompatible_layout = true;
        }

        if incompatible_layout {
            self.change_to_contig_append_layout(axis);
            // safety-check parameters after remodeling
            debug_assert_eq!(self_is_empty, self.is_empty());
            debug_assert_eq!(current_axis_len, self.len_of(axis));
        }

        let strides = if self_is_empty {
            // recompute strides - if the array was previously empty, it could have zeros in
            // strides.
            // The new order is based on c/f-contig but must have `axis` as outermost axis.
            if axis == Axis(self.ndim() - 1) {
                // prefer f-contig when appending to the last axis
                // Axis n - 1 is outermost axis
                res_dim.fortran_strides()
            } else {
                // standard axis order except for the growing axis;
                // anticipates that it's likely that `array` has standard order apart from the
                // growing axis.
                res_dim.slice_mut()[..=axis.index()].rotate_right(1);
                let mut strides = res_dim.default_strides();
                res_dim.slice_mut()[..=axis.index()].rotate_left(1);
                strides.slice_mut()[..=axis.index()].rotate_left(1);
                strides
            }
        } else if current_axis_len == 1 {
            // This is the outermost/longest stride axis; so we find the max across the other axes
            let new_stride = self.axes().fold(1, |acc, ax| {
                if ax.axis == axis || ax.len <= 1 {
                    acc
                } else {
                    let this_ax = ax.len as isize * ax.stride.abs();
                    if this_ax > acc {
                        this_ax
                    } else {
                        acc
                    }
                }
            });
            let mut strides = self.strides.clone();
            strides[axis.index()] = new_stride as usize;
            strides
        } else {
            self.strides.clone()
        };

        // grow backing storage and update head ptr
        self.reserve(axis, array_dim[axis.index()])?;

        unsafe {
            // clone elements from view to the array now
            //
            // To be robust for panics and drop the right elements, we want
            // to fill the tail in memory order, so that we can drop the right elements on panic.
            //
            // We have: Zip::from(tail_view).and(array)
            // Transform tail_view into standard order by inverting and moving its axes.
            // Keep the Zip traversal unchanged by applying the same axis transformations to
            // `array`. This ensures the Zip traverses the underlying memory in order.
            //
            // XXX It would be possible to skip this transformation if the element
            // doesn't have drop. However, in the interest of code coverage, all elements
            // use this code initially.

            // Invert axes in tail_view by inverting strides
            let mut tail_strides = strides.clone();
            if tail_strides.ndim() > 1 {
                for i in 0..tail_strides.ndim() {
                    let s = tail_strides[i] as isize;
                    if s < 0 {
                        tail_strides.set_axis(Axis(i), -s as usize);
                        array.invert_axis(Axis(i));
                    }
                }
            }

            // With > 0 strides, the current end of data is the correct base pointer for tail_view
            let tail_ptr = self.data.as_end_nonnull();
            let mut tail_view = RawArrayViewMut::new(tail_ptr, array_dim, tail_strides);

            if tail_view.ndim() > 1 {
                sort_axes_in_default_order_tandem(&mut tail_view, &mut array);
                debug_assert!(tail_view.is_standard_layout(),
                              "not std layout dim: {:?}, strides: {:?}",
                              tail_view.shape(), tail_view.strides());
            }

            // Keep track of currently filled length of `self.data` and update it
            // on scope exit (panic or loop finish). This "indirect" way to
            // write the length is used to help the compiler, the len store to self.data may
            // otherwise be mistaken to alias with other stores in the loop.
            struct SetLenOnDrop<'a, A: 'a>
            {
                len: usize,
                data: &'a mut OwnedRepr<A>,
            }

            impl<A> Drop for SetLenOnDrop<'_, A>
            {
                fn drop(&mut self)
                {
                    unsafe {
                        self.data.set_len(self.len);
                    }
                }
            }

            let mut data_length_guard = SetLenOnDrop {
                len: self.data.len(),
                data: &mut self.data,
            };

            // Safety: tail_view is constructed to have the same shape as array
            Zip::from(tail_view)
                .and_unchecked(array)
                .debug_assert_c_order()
                .for_each(|to, from| {
                    to.write(from.clone());
                    data_length_guard.len += 1;
                });
            drop(data_length_guard);

            // update array dimension
            self.strides = strides;
            self.dim = res_dim;
        }
        // multiple assertions after pointer & dimension update
        debug_assert_eq!(self.data.len(), self.len());
        debug_assert_eq!(self.len(), new_len);
        debug_assert!(self.pointer_is_inbounds());

        Ok(())
    }

    /// Reserve capacity to grow array along `axis` by at least `additional` elements.
    ///
    /// The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the
    /// number of dimensions (axes) of the array.
    ///
    /// Existing elements of `array` are untouched and the backing storage is grown by
    /// calling the underlying `reserve` method of the `OwnedRepr`.
    ///
    /// This is useful when pushing or appending repeatedly to an array to avoid multiple
    /// allocations.
    ///
    /// ***Panics*** if the axis is out of bounds.
    ///
    /// ***Errors*** with a shape error if the resultant capacity is larger than the addressable
    /// bounds; that is, the product of non-zero axis lengths once `axis` has been extended by
    /// `additional` exceeds `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::{Array3, Axis};
    /// let mut a = Array3::<i32>::zeros((0,2,4));
    /// a.reserve(Axis(0), 1000).unwrap();
    /// assert!(a.into_raw_vec().capacity() >= 2*4*1000);
    /// ```
    ///
    pub fn reserve(&mut self, axis: Axis, additional: usize) -> Result<(), ShapeError>
    where D: RemoveAxis
    {
        debug_assert!(axis.index() < self.ndim());
        let self_dim = self.raw_dim();
        let remaining_shape = self_dim.remove_axis(axis);

        // Make sure added capacity doesn't overflow usize::MAX
        let len_to_append = remaining_shape
            .size()
            .checked_mul(additional)
            .ok_or(ShapeError::from_kind(ErrorKind::Overflow))?;

        // Make sure new capacity is still in bounds
        let mut res_dim = self_dim;
        res_dim[axis.index()] += additional;
        let new_len = dimension::size_of_shape_checked(&res_dim)?;

        // Check whether len_to_append would cause an overflow
        debug_assert_eq!(self.len().checked_add(len_to_append).unwrap(), new_len);

        unsafe {
            // grow backing storage and update head ptr
            let data_to_array_offset = if std::mem::size_of::<A>() != 0 {
                self.as_ptr().offset_from(self.data.as_ptr())
            } else {
                0
            };
            debug_assert!(data_to_array_offset >= 0);
            self.ptr = self
                .data
                .reserve(len_to_append)
                .offset(data_to_array_offset);
        }

        debug_assert!(self.pointer_is_inbounds());

        Ok(())
    }
}

/// This drops all "unreachable" elements in `self_` given the data pointer and data length.
///
/// # Safety
///
/// This is an internal function for use by move_into and IntoIter only, safety invariants may need
/// to be upheld across the calls from those implementations.
pub(crate) unsafe fn drop_unreachable_raw<A, D>(
    mut self_: RawArrayViewMut<A, D>, data_ptr: NonNull<A>, data_len: usize,
) where D: Dimension
{
    let self_len = self_.len();

    for i in 0..self_.ndim() {
        if self_.stride_of(Axis(i)) < 0 {
            self_.invert_axis(Axis(i));
        }
    }
    sort_axes_in_default_order(&mut self_);
    // with uninverted axes this is now the element with lowest address
    let array_memory_head_ptr = self_.ptr;
    let data_end_ptr = data_ptr.add(data_len);
    debug_assert!(data_ptr <= array_memory_head_ptr);
    debug_assert!(array_memory_head_ptr <= data_end_ptr);

    // The idea is simply this: the iterator will yield the elements of self_ in
    // increasing address order.
    //
    // The pointers produced by the iterator are those that we *do not* touch.
    // The pointers *not mentioned* by the iterator are those we have to drop.
    //
    // We have to drop elements in the range from `data_ptr` until (not including)
    // `data_end_ptr`, except those that are produced by `iter`.

    // As an optimization, the innermost axis is removed if it has stride 1, because
    // we then have a long stretch of contiguous elements we can skip as one.
    let inner_lane_len;
    if self_.ndim() > 1 && self_.strides.last_elem() == 1 {
        self_.dim.slice_mut().rotate_right(1);
        self_.strides.slice_mut().rotate_right(1);
        inner_lane_len = self_.dim[0];
        self_.dim[0] = 1;
        self_.strides[0] = 1;
    } else {
        inner_lane_len = 1;
    }

    // iter is a raw pointer iterator traversing the array in memory order now with the
    // sorted axes.
    let mut iter = Baseiter::new(self_.ptr, self_.dim, self_.strides);
    let mut dropped_elements = 0;

    let mut last_ptr = data_ptr;

    while let Some(elem_ptr) = iter.next() {
        // The interval from last_ptr up until (not including) elem_ptr
        // should now be dropped. This interval may be empty, then we just skip this loop.
        while last_ptr != elem_ptr {
            debug_assert!(last_ptr < data_end_ptr);
            std::ptr::drop_in_place(last_ptr.as_mut());
            last_ptr = last_ptr.add(1);
            dropped_elements += 1;
        }
        // Next interval will continue one past the current lane
        last_ptr = elem_ptr.add(inner_lane_len);
    }

    while last_ptr < data_end_ptr {
        std::ptr::drop_in_place(last_ptr.as_mut());
        last_ptr = last_ptr.add(1);
        dropped_elements += 1;
    }

    assert_eq!(data_len, dropped_elements + self_len,
               "Internal error: inconsistency in move_into");
}

/// Sort axes to standard order, i.e Axis(0) has biggest stride and Axis(n - 1) least stride
///
/// The axes should have stride >= 0 before calling this method.
fn sort_axes_in_default_order<S, D>(a: &mut ArrayBase<S, D>)
where
    S: RawData,
    D: Dimension,
{
    if a.ndim() <= 1 {
        return;
    }
    sort_axes1_impl(&mut a.dim, &mut a.strides);
}

fn sort_axes1_impl<D>(adim: &mut D, astrides: &mut D)
where D: Dimension
{
    debug_assert!(adim.ndim() > 1);
    debug_assert_eq!(adim.ndim(), astrides.ndim());
    // bubble sort axes
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..adim.ndim() - 1 {
            let axis_i = i;
            let next_axis = i + 1;

            // make sure higher stride axes sort before.
            debug_assert!(astrides.slice()[axis_i] as isize >= 0);
            if (astrides.slice()[axis_i] as isize) < astrides.slice()[next_axis] as isize {
                changed = true;
                adim.slice_mut().swap(axis_i, next_axis);
                astrides.slice_mut().swap(axis_i, next_axis);
            }
        }
    }
}

/// Sort axes to standard order, i.e Axis(0) has biggest stride and Axis(n - 1) least stride
///
/// Axes in a and b are sorted by the strides of `a`, and `a`'s axes should have stride >= 0 before
/// calling this method.
fn sort_axes_in_default_order_tandem<S, S2, D>(a: &mut ArrayBase<S, D>, b: &mut ArrayBase<S2, D>)
where
    S: RawData,
    S2: RawData,
    D: Dimension,
{
    if a.ndim() <= 1 {
        return;
    }
    sort_axes2_impl(&mut a.dim, &mut a.strides, &mut b.dim, &mut b.strides);
}

fn sort_axes2_impl<D>(adim: &mut D, astrides: &mut D, bdim: &mut D, bstrides: &mut D)
where D: Dimension
{
    debug_assert!(adim.ndim() > 1);
    debug_assert_eq!(adim.ndim(), bdim.ndim());
    // bubble sort axes
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..adim.ndim() - 1 {
            let axis_i = i;
            let next_axis = i + 1;

            // make sure higher stride axes sort before.
            debug_assert!(astrides.slice()[axis_i] as isize >= 0);
            if (astrides.slice()[axis_i] as isize) < astrides.slice()[next_axis] as isize {
                changed = true;
                adim.slice_mut().swap(axis_i, next_axis);
                astrides.slice_mut().swap(axis_i, next_axis);
                bdim.slice_mut().swap(axis_i, next_axis);
                bstrides.slice_mut().swap(axis_i, next_axis);
            }
        }
    }
}
