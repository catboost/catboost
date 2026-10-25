// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constructor methods for ndarray
//!
//!

#![allow(clippy::match_wild_err_arm)]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use num_traits::Float;
use num_traits::{One, Zero};
use std::mem;
use std::mem::MaybeUninit;

use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
use crate::dimension::{self, CanIndexCheckMode};
use crate::error::{self, ShapeError};
use crate::extension::nonnull::nonnull_from_vec_data;
use crate::imp_prelude::*;
use crate::indexes;
use crate::indices;
#[cfg(feature = "std")]
use crate::iterators::to_vec;
use crate::iterators::to_vec_mapped;
use crate::iterators::TrustedIterator;
use crate::StrideShape;
#[cfg(feature = "std")]
use crate::{geomspace, linspace, logspace};
#[allow(unused_imports)]
use rawpointer::PointerExt;

/// # Constructor Methods for Owned Arrays
///
/// Note that the constructor methods apply to `Array` and `ArcArray`,
/// the two array types that have owned storage.
///
/// ## Constructor methods for one-dimensional arrays.
impl<S, A> ArrayBase<S, Ix1>
where S: DataOwned<Elem = A>
{
    /// Create a one-dimensional array from a vector (no copying needed).
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::Array;
    ///
    /// let array = Array::from_vec(vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_vec(v: Vec<A>) -> Self
    {
        if mem::size_of::<A>() == 0 {
            assert!(
                v.len() <= isize::MAX as usize,
                "Length must fit in `isize`.",
            );
        }
        unsafe { Self::from_shape_vec_unchecked(v.len() as Ix, v) }
    }

    /// Create a one-dimensional array from an iterator or iterable.
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::Array;
    ///
    /// let array = Array::from_iter(0..10);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: IntoIterator<Item = A>>(iterable: I) -> Self
    {
        Self::from_vec(iterable.into_iter().collect())
    }

    /// Create a one-dimensional array with `n` evenly spaced elements from
    /// `start` to `end` (inclusive). `A` must be a floating point type.
    ///
    /// Note that if `start > end`, the first element will still be `start`,
    /// and the following elements will be decreasing. This is different from
    /// the behavior of `std::ops::RangeInclusive`, which interprets `start >
    /// end` to mean that the range is empty.
    ///
    /// **Panics** if `n` is greater than `isize::MAX` or if converting `n - 1`
    /// to type `A` fails.
    ///
    /// ```rust
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::linspace(0., 1., 5);
    /// assert!(array == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
    /// ```
    #[cfg(feature = "std")]
    pub fn linspace(start: A, end: A, n: usize) -> Self
    where A: Float
    {
        Self::from(to_vec(linspace::linspace(start, end, n)))
    }

    /// Create a one-dimensional array with elements from `start` to `end`
    /// (exclusive), incrementing by `step`. `A` must be a floating point type.
    ///
    /// **Panics** if the length is greater than `isize::MAX`.
    ///
    /// ```rust
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::range(0., 5., 1.);
    /// assert!(array == arr1(&[0., 1., 2., 3., 4.]))
    /// ```
    #[cfg(feature = "std")]
    pub fn range(start: A, end: A, step: A) -> Self
    where A: Float
    {
        Self::from(to_vec(linspace::range(start, end, step)))
    }

    /// Create a one-dimensional array with `n` logarithmically spaced
    /// elements, with the starting value being `base.powf(start)` and the
    /// final one being `base.powf(end)`. `A` must be a floating point type.
    ///
    /// If `base` is negative, all values will be negative.
    ///
    /// **Panics** if `n` is greater than `isize::MAX` or if converting `n - 1`
    /// to type `A` fails.
    ///
    /// ```rust
    /// # #[cfg(feature = "approx")] {
    /// use approx::assert_abs_diff_eq;
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::logspace(10.0, 0.0, 3.0, 4);
    /// assert_abs_diff_eq!(array, arr1(&[1e0, 1e1, 1e2, 1e3]));
    ///
    /// let array = Array::logspace(-10.0, 3.0, 0.0, 4);
    /// assert_abs_diff_eq!(array, arr1(&[-1e3, -1e2, -1e1, -1e0]));
    /// # }
    /// ```
    #[cfg(feature = "std")]
    pub fn logspace(base: A, start: A, end: A, n: usize) -> Self
    where A: Float
    {
        Self::from(to_vec(logspace::logspace(base, start, end, n)))
    }

    /// Create a one-dimensional array with `n` geometrically spaced elements
    /// from `start` to `end` (inclusive). `A` must be a floating point type.
    ///
    /// Returns `None` if `start` and `end` have different signs or if either
    /// one is zero. Conceptually, this means that in order to obtain a `Some`
    /// result, `end / start` must be positive.
    ///
    /// **Panics** if `n` is greater than `isize::MAX` or if converting `n - 1`
    /// to type `A` fails.
    ///
    /// ```rust
    /// # fn example() -> Option<()> {
    /// # #[cfg(feature = "approx")] {
    /// use approx::assert_abs_diff_eq;
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::geomspace(1e0, 1e3, 4)?;
    /// assert_abs_diff_eq!(array, arr1(&[1e0, 1e1, 1e2, 1e3]), epsilon = 1e-12);
    ///
    /// let array = Array::geomspace(-1e3, -1e0, 4)?;
    /// assert_abs_diff_eq!(array, arr1(&[-1e3, -1e2, -1e1, -1e0]), epsilon = 1e-12);
    /// # }
    /// # Some(())
    /// # }
    /// #
    /// # example().unwrap();
    /// ```
    #[cfg(feature = "std")]
    pub fn geomspace(start: A, end: A, n: usize) -> Option<Self>
    where A: Float
    {
        Some(Self::from(to_vec(geomspace::geomspace(start, end, n)?)))
    }
}

/// ## Constructor methods for two-dimensional arrays.
impl<S, A> ArrayBase<S, Ix2>
where S: DataOwned<Elem = A>
{
    /// Create an identity matrix of size `n` (square 2D array).
    ///
    /// **Panics** if `n * n` would overflow `isize`.
    pub fn eye(n: Ix) -> Self
    where
        S: DataMut,
        A: Clone + Zero + One,
    {
        let mut eye = Self::zeros((n, n));
        for a_ii in eye.diag_mut() {
            *a_ii = A::one();
        }
        eye
    }

    /// Create a 2D matrix from its diagonal
    ///
    /// **Panics** if `diag.len() * diag.len()` would overflow `isize`.
    ///
    /// ```rust
    /// use ndarray::{Array2, arr1, arr2};
    ///
    /// let diag = arr1(&[1, 2]);
    /// let array = Array2::from_diag(&diag);
    /// assert_eq!(array, arr2(&[[1, 0], [0, 2]]));
    /// ```
    pub fn from_diag<S2>(diag: &ArrayBase<S2, Ix1>) -> Self
    where
        A: Clone + Zero,
        S: DataMut,
        S2: Data<Elem = A>,
    {
        let n = diag.len();
        let mut arr = Self::zeros((n, n));
        arr.diag_mut().assign(diag);
        arr
    }

    /// Create a square 2D matrix of the specified size, with the specified
    /// element along the diagonal and zeros elsewhere.
    ///
    /// **Panics** if `n * n` would overflow `isize`.
    ///
    /// ```rust
    /// use ndarray::{array, Array2};
    ///
    /// let array = Array2::from_diag_elem(2, 5.);
    /// assert_eq!(array, array![[5., 0.], [0., 5.]]);
    /// ```
    pub fn from_diag_elem(n: usize, elem: A) -> Self
    where
        S: DataMut,
        A: Clone + Zero,
    {
        let mut eye = Self::zeros((n, n));
        for a_ii in eye.diag_mut() {
            *a_ii = elem.clone();
        }
        eye
    }
}

#[cfg(not(debug_assertions))]
#[allow(clippy::match_wild_err_arm)]
macro_rules! size_of_shape_checked_unwrap {
    ($dim:expr) => {
        match dimension::size_of_shape_checked($dim) {
            Ok(sz) => sz,
            Err(_) => {
                panic!("ndarray: Shape too large, product of non-zero axis lengths overflows isize")
            }
        }
    };
}

#[cfg(debug_assertions)]
macro_rules! size_of_shape_checked_unwrap {
    ($dim:expr) => {
        match dimension::size_of_shape_checked($dim) {
            Ok(sz) => sz,
            Err(_) => panic!(
                "ndarray: Shape too large, product of non-zero axis lengths \
                 overflows isize in shape {:?}",
                $dim
            ),
        }
    };
}

/// ## Constructor methods for n-dimensional arrays.
///
/// The `shape` argument can be an integer or a tuple of integers to specify
/// a static size. For example `10` makes a length 10 one-dimensional array
/// (dimension type `Ix1`) and `(5, 6)` a 5 × 6 array (dimension type `Ix2`).
///
/// With the trait `ShapeBuilder` in scope, there is the method `.f()` to select
/// column major (“f” order) memory layout instead of the default row major.
/// For example `Array::zeros((5, 6).f())` makes a column major 5 × 6 array.
///
/// Use [`type@IxDyn`] for the shape to create an array with dynamic
/// number of axes.
///
/// Finally, the few constructors that take a completely general
/// `Into<StrideShape>` argument *optionally* support custom strides, for
/// example a shape given like `(10, 2, 2).strides((1, 10, 20))` is valid.
impl<S, A, D> ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    /// Create an array with copies of `elem`, shape `shape`.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    ///
    /// ```
    /// use ndarray::{Array, arr3, ShapeBuilder};
    ///
    /// let a = Array::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// assert!(a.strides() == &[4, 2, 1]);
    ///
    /// let b = Array::from_elem((2, 2, 2).f(), 1.);
    /// assert!(b.strides() == &[1, 2, 4]);
    /// ```
    pub fn from_elem<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let size = size_of_shape_checked_unwrap!(&shape.dim);
        let v = vec![elem; size];
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }

    /// Create an array with zeros, shape `shape`.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(shape, A::zero())
    }

    /// Create an array with ones, shape `shape`.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elem(shape, A::one())
    }

    /// Create an array with default values, shape `shape`
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_shape_simple_fn(shape, A::default)
    }

    /// Create an array with values created by the function `f`.
    ///
    /// `f` is called with no argument, and it should return the element to
    /// create. If the precise index of the element to create is needed,
    /// use [`from_shape_fn`](ArrayBase::from_shape_fn) instead.
    ///
    /// This constructor can be useful if the element order is not important,
    /// for example if they are identical or random.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn from_shape_simple_fn<Sh, F>(shape: Sh, mut f: F) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut() -> A,
    {
        let shape = shape.into_shape_with_order();
        let len = size_of_shape_checked_unwrap!(&shape.dim);
        let v = to_vec_mapped(0..len, move |_| f());
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }

    /// Create an array with values created by the function `f`.
    ///
    /// `f` is called with the index of the element to create; the elements are
    /// visited in arbitrary order.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    ///
    /// ```
    /// use ndarray::{Array, arr2};
    ///
    /// // Create a table of i × j (with i and j from 1 to 3)
    /// let ij_table = Array::from_shape_fn((3, 3), |(i, j)| (1 + i) * (1 + j));
    ///
    /// assert_eq!(
    ///     ij_table,
    ///     arr2(&[[1, 2, 3],
    ///            [2, 4, 6],
    ///            [3, 6, 9]])
    /// );
    /// ```
    pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
    {
        let shape = shape.into_shape_with_order();
        let _ = size_of_shape_checked_unwrap!(&shape.dim);
        if shape.is_c() {
            let v = to_vec_mapped(indices(shape.dim.clone()).into_iter(), f);
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
        } else {
            let dim = shape.dim.clone();
            let v = to_vec_mapped(indexes::indices_iter_f(dim), f);
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
        }
    }

    /// Create an array with the given shape from a vector. (No cloning of
    /// elements needed.)
    ///
    /// ----
    ///
    /// For a contiguous c- or f-order shape, the following applies:
    ///
    /// **Errors** if `shape` does not correspond to the number of elements in
    /// `v` or if the shape/strides would result in overflowing `isize`.
    ///
    /// ----
    ///
    /// For custom strides, the following applies:
    ///
    /// **Errors** if strides and dimensions can point out of bounds of `v`, if
    /// strides allow multiple indices to point to the same element, or if the
    /// shape/strides would result in overflowing `isize`.
    ///
    /// ```
    /// use ndarray::Array;
    /// use ndarray::ShapeBuilder; // Needed for .strides() method
    /// use ndarray::arr2;
    ///
    /// let a = Array::from_shape_vec((2, 2), vec![1., 2., 3., 4.]);
    /// assert!(a.is_ok());
    ///
    /// let b = Array::from_shape_vec((2, 2).strides((1, 2)),
    ///                               vec![1., 2., 3., 4.]).unwrap();
    /// assert!(
    ///     b == arr2(&[[1., 3.],
    ///                 [2., 4.]])
    /// );
    /// ```
    pub fn from_shape_vec<Sh>(shape: Sh, v: Vec<A>) -> Result<Self, ShapeError>
    where Sh: Into<StrideShape<D>>
    {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_vec_impl(shape.into(), v)
    }

    fn from_shape_vec_impl(shape: StrideShape<D>, v: Vec<A>) -> Result<Self, ShapeError>
    {
        let dim = shape.dim;
        let is_custom = shape.strides.is_custom();
        dimension::can_index_slice_with_strides(&v, &dim, &shape.strides, dimension::CanIndexCheckMode::OwnedMutable)?;
        if !is_custom && dim.size() != v.len() {
            return Err(error::incompatible_shapes(&Ix1(v.len()), &dim));
        }
        let strides = shape.strides.strides_for_dim(&dim);
        unsafe { Ok(Self::from_vec_dim_stride_unchecked(dim, strides, v)) }
    }

    /// Creates an array from a vector and interpret it according to the
    /// provided shape and strides. (No cloning of elements needed.)
    ///
    /// # Safety
    ///
    /// The caller must ensure that the following conditions are met:
    ///
    /// 1. The ndim of `dim` and `strides` must be the same.
    ///
    /// 2. The product of non-zero axis lengths must not exceed `isize::MAX`.
    ///
    /// 3. For axes with length > 1, the pointer cannot move outside the
    ///    slice.
    ///
    /// 4. If the array will be empty (any axes are zero-length), the
    ///    difference between the least address and greatest address accessible
    ///    by moving along all axes must be ≤ `v.len()`.
    ///
    ///    If the array will not be empty, the difference between the least
    ///    address and greatest address accessible by moving along all axes
    ///    must be < `v.len()`.
    ///
    /// 5. The strides must not allow any element to be referenced by two different
    ///    indices.
    pub unsafe fn from_shape_vec_unchecked<Sh>(shape: Sh, v: Vec<A>) -> Self
    where Sh: Into<StrideShape<D>>
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides.strides_for_dim(&dim);
        Self::from_vec_dim_stride_unchecked(dim, strides, v)
    }

    unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>) -> Self
    {
        // debug check for issues that indicates wrong use of this constructor
        debug_assert!(dimension::can_index_slice(&v, &dim, &strides, CanIndexCheckMode::OwnedMutable).is_ok());

        let ptr = nonnull_from_vec_data(&mut v).add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides));
        ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim(strides, dim)
    }

    /// Creates an array from an iterator, mapped by `map` and interpret it according to the
    /// provided shape and strides.
    ///
    /// # Safety
    ///
    /// See from_shape_vec_unchecked
    pub(crate) unsafe fn from_shape_trusted_iter_unchecked<Sh, I, F>(shape: Sh, iter: I, map: F) -> Self
    where
        Sh: Into<StrideShape<D>>,
        I: TrustedIterator + ExactSizeIterator,
        F: FnMut(I::Item) -> A,
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides.strides_for_dim(&dim);
        let v = to_vec_mapped(iter, map);
        Self::from_vec_dim_stride_unchecked(dim, strides, v)
    }

    /// Create an array with uninitialized elements, shape `shape`.
    ///
    /// The uninitialized elements of type `A` are represented by the type `MaybeUninit<A>`,
    /// an easier way to handle uninit values correctly.
    ///
    /// Only *when* the array is completely initialized with valid elements, can it be
    /// converted to an array of `A` elements using [`.assume_init()`].
    ///
    /// **Panics** if the number of elements in `shape` would overflow isize.
    ///
    /// ### Safety
    ///
    /// The whole of the array must be initialized before it is converted
    /// using [`.assume_init()`] or otherwise traversed/read with the element type `A`.
    ///
    /// ### Examples
    ///
    /// It is possible to assign individual values through `*elt = MaybeUninit::new(value)`
    /// and so on.
    ///
    /// [`.assume_init()`]: ArrayBase::assume_init
    ///
    /// ```
    /// use ndarray::{s, Array2};
    ///
    /// // Example Task: Let's create a column shifted copy of the input
    ///
    /// fn shift_by_two(a: &Array2<f32>) -> Array2<f32> {
    ///     // create an uninitialized array
    ///     let mut b = Array2::uninit(a.dim());
    ///
    ///     // two first columns in b are two last in a
    ///     // rest of columns in b are the initial columns in a
    ///
    ///     a.slice(s![.., -2..]).assign_to(b.slice_mut(s![.., ..2]));
    ///     a.slice(s![.., 2..]).assign_to(b.slice_mut(s![.., ..-2]));
    ///
    ///     // Now we can promise that `b` is safe to use with all operations
    ///     unsafe {
    ///         b.assume_init()
    ///     }
    /// }
    ///
    /// # let _ = shift_by_two;
    /// ```
    pub fn uninit<Sh>(shape: Sh) -> ArrayBase<S::MaybeUninit, D>
    where Sh: ShapeBuilder<Dim = D>
    {
        unsafe {
            let shape = shape.into_shape_with_order();
            let size = size_of_shape_checked_unwrap!(&shape.dim);
            let mut v = Vec::with_capacity(size);
            v.set_len(size);
            ArrayBase::from_shape_vec_unchecked(shape, v)
        }
    }

    /// Create an array with uninitialized elements, shape `shape`.
    ///
    /// The uninitialized elements of type `A` are represented by the type `MaybeUninit<A>`,
    /// an easier way to handle uninit values correctly.
    ///
    /// The `builder` closure gets unshared access to the array through a view and can use it to
    /// modify the array before it is returned. This allows initializing the array for any owned
    /// array type (avoiding clone requirements for copy-on-write, because the array is unshared
    /// when initially created).
    ///
    /// Only *when* the array is completely initialized with valid elements, can it be
    /// converted to an array of `A` elements using [`.assume_init()`].
    ///
    /// **Panics** if the number of elements in `shape` would overflow isize.
    ///
    /// ### Safety
    ///
    /// The whole of the array must be initialized before it is converted
    /// using [`.assume_init()`] or otherwise traversed/read with the element type `A`.
    ///
    /// [`.assume_init()`]: ArrayBase::assume_init
    pub fn build_uninit<Sh, F>(shape: Sh, builder: F) -> ArrayBase<S::MaybeUninit, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnOnce(ArrayViewMut<MaybeUninit<A>, D>),
    {
        let mut array = Self::uninit(shape);
        // Safe because: the array is unshared here
        unsafe {
            builder(array.raw_view_mut_unchecked().deref_into_view_mut());
        }
        array
    }
}
