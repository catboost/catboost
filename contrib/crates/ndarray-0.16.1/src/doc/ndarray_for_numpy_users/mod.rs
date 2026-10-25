//! `ndarray` for NumPy users.
//!
//! This is an introductory guide to `ndarray` for people with experience using
//! NumPy, although it may also be useful to others. For a more general
//! introduction to `ndarray`'s array type `ArrayBase`, see the [`ArrayBase`
//! docs][ArrayBase].
//!
//! # Contents
//!
//! * [Similarities](#similarities)
//! * [Some key differences](#some-key-differences)
//! * [The ndarray ecosystem](#the-ndarray-ecosystem)
//! * [Other Rust array/matrix crates](#other-rust-arraymatrix-crates)
//! * [Rough `ndarray`–NumPy equivalents](#rough-ndarraynumpy-equivalents)
//!
//!   * [Array creation](#array-creation)
//!   * [Indexing and slicing](#indexing-and-slicing)
//!   * [Shape and strides](#shape-and-strides)
//!   * [Mathematics](#mathematics)
//!   * [Array manipulation](#array-manipulation)
//!   * [Iteration](#iteration)
//!   * [Type conversions](#type-conversions)
//!   * [Convenience methods for 2-D arrays](#convenience-methods-for-2-d-arrays)
//!
//! # Similarities
//!
//! `ndarray`'s array type ([`ArrayBase`]), is very similar to
//! NumPy's array type (`numpy.ndarray`):
//!
//! * Arrays have a single element type.
//! * Arrays can have arbitrarily many dimensions.
//! * Arrays can have arbitrary strides.
//! * Indexing starts at zero, not one.
//! * The default memory layout is row-major, and the default iterators follow
//!   row-major order (also called "logical order" in the documentation).
//! * Arithmetic operators work elementwise. (For example, `a * b` performs
//!   elementwise multiplication, not matrix multiplication.)
//! * Owned arrays are contiguous in memory.
//! * Many operations, such as slicing, are very cheap because they can return
//!   a view of an array instead of copying the data.
//!
//! NumPy has many features that `ndarray` doesn't have yet, such as:
//!
//! * [index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays)
//! * [mask index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays)
//! * co-broadcasting (`ndarray` only supports broadcasting the right-hand array in a binary operation.)
//!
//! # Some key differences
//!
//! <table>
//! <tr>
//! <th>
//!
//! NumPy
//!
//! </th>
//! <th>
//!
//! `ndarray`
//!
//! </th>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! In NumPy, there is no distinction between owned arrays, views, and mutable
//! views. There can be multiple arrays (instances of `numpy.ndarray`) that
//! mutably reference the same data.
//!
//! </td>
//! <td>
//!
//! In `ndarray`, all arrays are instances of [`ArrayBase`], but
//! `ArrayBase` is generic over the ownership of the data. [`Array`]
//! owns its data; [`ArrayView`] is a view;
//! [`ArrayViewMut`] is a mutable view; [`CowArray`]
//! either owns its data or is a view (with copy-on-write mutation of the view
//! variant); and [`ArcArray`] has a reference-counted pointer to its
//! data (with copy-on-write mutation). Arrays and views follow Rust's aliasing
//! rules.
//!
//! </td>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! In NumPy, all arrays are dynamic-dimensional.
//!
//! </td>
//! <td>
//!
//! In `ndarray`, you can create fixed-dimension arrays, such as
//! [`Array2`]. This takes advantage of the type system to help you
//! write correct code and also avoids small heap allocations for the shape and
//! strides.
//!
//! </td>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! When slicing in NumPy, the indices are `start`, `start + step`, `start +
//! 2*step`, … until reaching `end` (exclusive).
//!
//! </td>
//! <td>
//!
//! When slicing in `ndarray`, the axis is first sliced with `start..end`. Then if
//! `step` is positive, the first index is the front of the slice; if `step` is
//! negative, the first index is the back of the slice. This means that the
//! behavior is different from NumPy when `step < 0`. See the docs for the
//! [`s![]` macro][s!] for more details.
//!
//! </td>
//! </tr>
//! </table>
//!
//! # The ndarray ecosystem
//!
//! `ndarray` does not provide advanced linear algebra routines out of the box (e.g. SVD decomposition).
//! Most of the routines that you can find in `scipy.linalg`/`numpy.linalg` are provided by another crate,
//! [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg).
//!
//! The same holds for statistics: `ndarray` provides some basic functionalities (e.g. `mean`)
//! but more advanced routines can be found in [`ndarray-stats`](https://crates.io/crates/ndarray-stats).
//!
//! If you are looking to generate random arrays instead, check out [`ndarray-rand`](https://crates.io/crates/ndarray-rand).
//!
//! It is also possible to serialize `NumPy` arrays in `.npy`/`.npz` format and deserialize them as `ndarray` arrays (and vice versa)
//! using [`ndarray-npy`](https://crates.io/crates/ndarray-npy).
//!
//! # Other Rust array/matrix crates
//!
//! Of the array/matrix types in Rust crates, the `ndarray` array type is probably
//! the most similar to NumPy's arrays and is the most flexible. However, if your
//! use-case is constrained to linear algebra on 1-D and 2-D vectors and matrices,
//! it might be worth considering other crates:
//!
//! * [`nalgebra`](https://crates.io/crates/nalgebra) provides 1-D and 2-D
//!   column-major vector and matrix types for linear algebra. Vectors and matrices
//!   can have constant or dynamic shapes, and `nalgebra` uses the type system to
//!   provide compile-time checking of shapes, not just the number of dimensions.
//!   `nalgebra` provides convenient functionality for geometry (e.g. coordinate
//!   transformations) and linear algebra.
//! * [`cgmath`](https://crates.io/crates/cgmath) provides 1-D and 2-D column-major
//!   types of shape 4×4 or smaller. It's primarily designed for computer graphics
//!   and provides convenient functionality for geometry (e.g. coordinate
//!   transformations). Similar to `nalgebra`, `cgmath` uses the type system to
//!   provide compile-time checking of shapes.
//! * [`rulinalg`](https://crates.io/crates/rulinalg) provides 1-D and 2-D
//!   row-major vector and matrix types with dynamic shapes. Similar to `ndarray`,
//!   `rulinalg` provides compile-time checking of the number of dimensions, but
//!   not shapes. `rulinalg` provides pure-Rust implementations of linear algebra
//!   operations.
//! * If there's another crate that should be listed here, please let us know.
//!
//! In contrast to these crates, `ndarray` provides an *n*-dimensional array type,
//! so it's not restricted to 1-D and 2-D vectors and matrices. Also, operators
//! operate elementwise by default, so the multiplication operator `*` performs
//! elementwise multiplication instead of matrix multiplication. (You have to
//! specifically call `.dot()` if you want matrix multiplication.)
//!
//! # Rough `ndarray`–NumPy equivalents
//!
//! These tables provide some rough equivalents of NumPy operations in `ndarray`.
//! There are a variety of other methods that aren't included in these tables,
//! including shape-manipulation, array creation, and iteration routines.
//!
//! It's assumed that you've imported NumPy like this:
//!
//! ```python
//! import numpy as np
//! ```
//!
//! and `ndarray` like this:
//!
//! ```
//! use ndarray::prelude::*;
//! #
//! # fn main() { let _ = arr0(1); }
//! ```
//!
//! ## Array creation
//!
//! This table contains ways to create arrays from scratch. For creating arrays by
//! operations on other arrays (e.g. arithmetic), see the other tables. Also see
//! the [`::from_vec()`][::from_vec()], [`::from_iter()`][::from_iter()],
//! [`::default()`][::default()], [`::from_shape_fn()`][::from_shape_fn()], and
//! [`::from_shape_vec_unchecked()`][::from_shape_vec_unchecked()] methods.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `np.array([[1.,2.,3.], [4.,5.,6.]])` | [`array![[1.,2.,3.], [4.,5.,6.]]`][array!] or [`arr2(&[[1.,2.,3.], [4.,5.,6.]])`][arr2()] | 2×3 floating-point array literal
//! `np.arange(0., 10., 0.5)` or `np.r_[:10.:0.5]` | [`Array::range(0., 10., 0.5)`][::range()] | create a 1-D array with values `0.`, `0.5`, …, `9.5`
//! `np.linspace(0., 10., 11)` or `np.r_[:10.:11j]` | [`Array::linspace(0., 10., 11)`][::linspace()] | create a 1-D array with 11 elements with values `0.`, …, `10.`
//! `np.logspace(2.0, 3.0, num=4, base=10.0)` | [`Array::logspace(10.0, 2.0, 3.0, 4)`][::logspace()] | create a 1-D array with 4 elements with values `100.`, `215.4`, `464.1`, `1000.`
//! `np.geomspace(1., 1000., num=4)` | [`Array::geomspace(1e0, 1e3, 4)`][::geomspace()] | create a 1-D array with 4 elements with values `1.`, `10.`, `100.`, `1000.`
//! `np.ones((3, 4, 5))` | [`Array::ones((3, 4, 5))`][::ones()] | create a 3×4×5 array filled with ones (inferring the element type)
//! `np.zeros((3, 4, 5))` | [`Array::zeros((3, 4, 5))`][::zeros()] | create a 3×4×5 array filled with zeros (inferring the element type)
//! `np.zeros((3, 4, 5), order='F')` | [`Array::zeros((3, 4, 5).f())`][::zeros()] | create a 3×4×5 array with Fortran (column-major) memory layout filled with zeros (inferring the element type)
//! `np.zeros_like(a, order='C')` | [`Array::zeros(a.raw_dim())`][::zeros()] | create an array of zeros of the shape shape as `a`, with row-major memory layout (unlike NumPy, this infers the element type from context instead of duplicating `a`'s element type)
//! `np.full((3, 4), 7.)` | [`Array::from_elem((3, 4), 7.)`][::from_elem()] | create a 3×4 array filled with the value `7.`
//! `np.eye(3)` | [`Array::eye(3)`][::eye()] | create a 3×3 identity matrix (inferring the element type)
//! `np.diag(np.array([1, 2, 3]))` | [`Array2::from_diag(&arr1(&[1, 2, 3]))`][::from_diag()] | create a 3×3 matrix with `[1, 2, 3]` as diagonal and zeros elsewhere (inferring the element type)
//! `np.array([1, 2, 3, 4]).reshape((2, 2))` | [`Array::from_shape_vec((2, 2), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec`
//! `np.array([1, 2, 3, 4]).reshape((2, 2), order='F')` | [`Array::from_shape_vec((2, 2).f(), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec` using Fortran (column-major) order
//! `np.random` | See the [`ndarray-rand`](https://crates.io/crates/ndarray-rand) crate. | create arrays of random numbers
//!
//! Note that the examples in the table rely on the compiler inferring the
//! element type and dimensionality from context, which is usually sufficient.
//! However, if the compiler cannot infer the types, you can specify them
//! manually. These are examples of creating a 3-D Fortran-layout array of
//! `f64`s:
//!
//! ```
//! # use ndarray::prelude::*;
//! #
//! // This is an example where the compiler can infer the element type
//! // because `f64::sin` can only be called on `f64` elements:
//! let arr1 = Array::zeros((3, 2, 4).f());
//! arr1.mapv(f64::sin);
//!
//! // Specify just the element type and infer the dimensionality:
//! let arr2 = Array::<f64, _>::zeros((3, 2, 4).f());
//! let arr3: Array<f64, _> = Array::zeros((3, 2, 4).f());
//!
//! // Specify both the element type and dimensionality:
//! let arr4 = Array3::<f64>::zeros((3, 2, 4).f());
//! let arr5: Array3<f64> = Array::zeros((3, 2, 4).f());
//! let arr6 = Array::<f64, Ix3>::zeros((3, 2, 4).f());
//! let arr7: Array<f64, Ix3> = Array::zeros((3, 2, 4).f());
//! ```
//!
//! ## Indexing and slicing
//!
//! A few notes:
//!
//! * Indices start at 0. For example, "row 1" is the second row in the array.
//!
//! * Some methods have multiple variants in terms of ownership and mutability.
//!   Only the non-mutable methods that take the array by reference are listed in
//!   this table. For example, [`.slice()`][.slice()] also has corresponding
//!   methods [`.slice_mut()`][.slice_mut()], [`.slice_move()`][.slice_move()], and
//!   [`.slice_collapse()`][.slice_collapse()].
//!
//! * The behavior of slicing is different from NumPy for slices with
//!   `step < 0`. See the docs for the [`s![]` macro][s!] for more details.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a[-1]` | [`a[a.len() - 1]`][.index()] | access the last element in 1-D array `a`
//! `a[1, 4]` | [`a[[1, 4]]`][.index()] | access the element in row 1, column 4
//! `a[1]` or `a[1, :, :]` | [`a.slice(s![1, .., ..])`][.slice()] or [`a.index_axis(Axis(0), 1)`][.index_axis()] | get a 2-D subview of a 3-D array at index 1 of axis 0
//! `a[0:5]` or `a[:5]` or `a[0:5, :]` | [`a.slice(s![0..5, ..])`][.slice()] or [`a.slice(s![..5, ..])`][.slice()] or [`a.slice_axis(Axis(0), Slice::from(0..5))`][.slice_axis()] | get the first 5 rows of a 2-D array
//! `a[-5:]` or `a[-5:, :]` | [`a.slice(s![-5.., ..])`][.slice()] or [`a.slice_axis(Axis(0), Slice::from(-5..))`][.slice_axis()] | get the last 5 rows of a 2-D array
//! `a[:3, 4:9]` | [`a.slice(s![..3, 4..9])`][.slice()] | columns 4, 5, 6, 7, and 8 of the first 3 rows
//! `a[1:4:2, ::-1]` | [`a.slice(s![1..4;2, ..;-1])`][.slice()] | rows 1 and 3 with the columns in reverse order
//! `a.take([4, 2])` | `a.select(Axis(0), &[4, 2])` | rows 4 and 2 of the array
//!
//! ## Shape and strides
//!
//! Note that [`a.shape()`][.shape()], [`a.dim()`][.dim()], and
//! [`a.raw_dim()`][.raw_dim()] all return the shape of the array, but as
//! different types. `a.shape()` returns the shape as `&[Ix]`, (where
//! [`Ix`] is `usize`) which is useful for general operations on the shape.
//! `a.dim()` returns the shape as `D::Pattern`, which is useful for
//! pattern-matching shapes. `a.raw_dim()` returns the shape as `D`, which is
//! useful for creating other arrays of the same shape.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `np.ndim(a)` or `a.ndim` | [`a.ndim()`][.ndim()] | get the number of dimensions of array `a`
//! `np.size(a)` or `a.size` | [`a.len()`][.len()] | get the number of elements in array `a`
//! `np.shape(a)` or `a.shape` | [`a.shape()`][.shape()] or [`a.dim()`][.dim()] | get the shape of array `a`
//! `a.shape[axis]` | [`a.len_of(Axis(axis))`][.len_of()] | get the length of an axis
//! `a.strides` | [`a.strides()`][.strides()] | get the strides of array `a`
//! `np.size(a) == 0` or `a.size == 0` | [`a.is_empty()`][.is_empty()] | check if the array has zero elements
//!
//! ## Mathematics
//!
//! Note that [`.mapv()`][.mapv()] has corresponding methods [`.map()`][.map()],
//! [`.mapv_into()`][.mapv_into()], [`.map_inplace()`][.map_inplace()], and
//! [`.mapv_inplace()`][.mapv_inplace()]. Also look at [`.fold()`][.fold()],
//! [`.for_each()`][.for_each()], [`.fold_axis()`][.fold_axis()], and
//! [`.map_axis()`][.map_axis()].
//!
//! <table>
//! <tr><th>
//!
//! NumPy
//!
//! </th><th>
//!
//! `ndarray`
//!
//! </th><th>
//!
//! Notes
//!
//! </th></tr>
//!
//! <tr><td>
//!
//! `a.transpose()` or `a.T`
//!
//! </td><td>
//!
//! [`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()]
//!
//! </td><td>
//!
//! transpose of array `a` (view for `.t()` or by-move for `.reversed_axes()`)
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `mat1.dot(mat2)`
//!
//! </td><td>
//!
//! [`mat1.dot(&mat2)`][matrix-* dot]
//!
//! </td><td>
//!
//! 2-D matrix multiply
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `mat.dot(vec)`
//!
//! </td><td>
//!
//! [`mat.dot(&vec)`][matrix-* dot]
//!
//! </td><td>
//!
//! 2-D matrix dot 1-D column vector
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `vec.dot(mat)`
//!
//! </td><td>
//!
//! [`vec.dot(&mat)`][vec-* dot]
//!
//! </td><td>
//!
//! 1-D row vector dot 2-D matrix
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `vec1.dot(vec2)`
//!
//! </td><td>
//!
//! [`vec1.dot(&vec2)`][vec-* dot]
//!
//! </td><td>
//!
//! vector dot product
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a * b`, `a + b`, etc.
//!
//! </td><td>
//!
//! [`a * b`, `a + b`, etc.](ArrayBase#arithmetic-operations)
//!
//! </td><td>
//!
//! element-wise arithmetic operations
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a**3`
//!
//! </td><td>
//!
//! [`a.mapv(|a| a.powi(3))`][.mapv()]
//!
//! </td><td>
//!
//! element-wise power of 3
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sqrt(a)`
//!
//! </td><td>
//!
//! [`a.mapv(f64::sqrt)`][.mapv()]
//!
//! </td><td>
//!
//! element-wise square root for `f64` array
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `(a>0.5)`
//!
//! </td><td>
//!
//! [`a.mapv(|a| a > 0.5)`][.mapv()]
//!
//! </td><td>
//!
//! array of `bool`s of same shape as `a` with `true` where `a > 0.5` and `false` elsewhere
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sum(a)` or `a.sum()`
//!
//! </td><td>
//!
//! [`a.sum()`][.sum()]
//!
//! </td><td>
//!
//! sum the elements in `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sum(a, axis=2)` or `a.sum(axis=2)`
//!
//! </td><td>
//!
//! [`a.sum_axis(Axis(2))`][.sum_axis()]
//!
//! </td><td>
//!
//! sum the elements in `a` along axis 2
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.mean(a)` or `a.mean()`
//!
//! </td><td>
//!
//! [`a.mean().unwrap()`][.mean()]
//! </td><td>
//!
//! calculate the mean of the elements in `f64` array `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.mean(a, axis=2)` or `a.mean(axis=2)`
//!
//! </td><td>
//!
//! [`a.mean_axis(Axis(2))`][.mean_axis()]
//!
//! </td><td>
//!
//! calculate the mean of the elements in `a` along axis 2
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.allclose(a, b, atol=1e-8)`
//!
//! </td><td>
//!
//! [`a.abs_diff_eq(&b, 1e-8)`][.abs_diff_eq()]
//!
//! </td><td>
//!
//! check if the arrays' elementwise differences are within an absolute tolerance (it requires the `approx` feature-flag)
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.diag(a)`
//!
//! </td><td>
//!
//! [`a.diag()`][.diag()]
//!
//! </td><td>
//!
//! view the diagonal of `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.linalg`
//!
//! </td><td>
//!
//! See [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg)
//!
//! </td><td>
//!
//! linear algebra (matrix inverse, solving, decompositions, etc.)
//!
//! </td></tr>
//! </table>
//!
//! ## Array manipulation
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a[:] = 3.` | [`a.fill(3.)`][.fill()] | set all array elements to the same scalar value
//! `a[:] = b` | [`a.assign(&b)`][.assign()] | copy the data from array `b` into array `a`
//! `a[:5, 2] = 3.` | [`a.slice_mut(s![..5, 2]).fill(3.)`][.fill()] | set a portion of the array to the same scalar value
//! `a[:5, 2] = b` | [`a.slice_mut(s![..5, 2]).assign(&b)`][.assign()] | copy the data from array `b` into part of array `a`
//! `np.concatenate((a,b), axis=1)` | [`concatenate![Axis(1), a, b]`][concatenate!] or [`concatenate(Axis(1), &[a.view(), b.view()])`][concatenate()] | concatenate arrays `a` and `b` along axis 1
//! `np.stack((a,b), axis=1)` | [`stack![Axis(1), a, b]`][stack!] or [`stack(Axis(1), vec![a.view(), b.view()])`][stack()] | stack arrays `a` and `b` along axis 1
//! `a[:,np.newaxis]` or `np.expand_dims(a, axis=1)` | [`a.slice(s![.., NewAxis])`][.slice()] or [`a.insert_axis(Axis(1))`][.insert_axis()] | create an view of 1-D array `a`, inserting a new axis 1
//! `a.transpose()` or `a.T` | [`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()] | transpose of array `a` (view for `.t()` or by-move for `.reversed_axes()`)
//! `np.diag(a)` | [`a.diag()`][.diag()] | view the diagonal of `a`
//! `a.flatten()` | [`use std::iter::FromIterator; Array::from_iter(a.iter().cloned())`][::from_iter()] | create a 1-D array by flattening `a`
//!
//! ## Iteration
//!
//! `ndarray` has lots of interesting iterators/producers that implement the
//! [`NdProducer`](crate::NdProducer) trait, which is a generalization of `Iterator`
//! to multiple dimensions. This makes it possible to correctly and efficiently
//! zip together slices/subviews of arrays in multiple dimensions with
//! [`Zip`] or [`azip!()`]. The purpose of this is similar to
//! [`np.nditer`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nditer.html),
//! but [`Zip`] is implemented and used somewhat differently.
//!
//! This table lists some of the iterators/producers which have a direct
//! equivalent in NumPy. For a more complete introduction to producers and
//! iterators, see [*Loops, Producers, and
//! Iterators*](ArrayBase#loops-producers-and-iterators).
//! Note that there are also variants of these iterators (with a `_mut` suffix)
//! that yield `ArrayViewMut` instead of `ArrayView`.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a.flat` | [`a.iter()`][.iter()] | iterator over the array elements in logical order
//! `np.ndenumerate(a)` | [`a.indexed_iter()`][.indexed_iter()] | flat iterator yielding the index along with each element reference
//! `iter(a)` | [`a.outer_iter()`][.outer_iter()] or [`a.axis_iter(Axis(0))`][.axis_iter()] | iterator over the first (outermost) axis, yielding each subview
//!
//! ## Type conversions
//!
//! In `ndarray`, conversions between datatypes are done with `mapv()` by
//! passing a closure to convert every element independently.
//! For the conversion itself, we have several options:
//! - `std::convert::From` ensures lossless, safe conversions at compile-time
//!   and is generally recommended.
//! - `std::convert::TryFrom` can be used for potentially unsafe conversions. It
//!   will return a `Result` which can be handled or `unwrap()`ed to panic if
//!   any value at runtime cannot be converted losslessly.
//! - The `as` keyword compiles to lossless/lossy conversions depending on the
//!   source and target datatypes. It can be useful when `TryFrom` is a
//!   performance issue or does not apply. A notable difference to NumPy is that
//!   `as` performs a [*saturating* cast][sat_conv] when casting
//!   from floats to integers. Further information can be found in the
//!   [reference on type cast expressions][as_typecast].
//!
//! For details, be sure to check out the type conversion examples.
//!

//! <table>
//! <tr><th>
//!
//! NumPy
//!
//! </th><th>
//!
//! `ndarray`
//!
//! </th><th>
//!
//! Notes
//!
//! </th></tr>
//!
//! <tr><td>
//!
//! `a.astype(np.float32)`
//!
//! </td><td>
//!
//! `a.mapv(f32::from)`
//!
//! </td><td>
//!
//! convert `u8` array infallibly to `f32` array with `std::convert::From`, generally recommended
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a.astype(np.int32)`
//!
//! </td><td>
//!
//! `a.mapv(i32::from)`
//!
//! </td><td>
//!
//! upcast `u8` array to `i32` array with `std::convert::From`, preferable over `as` because it ensures at compile-time that the conversion is lossless
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a.astype(np.uint8)`
//!
//! </td><td>
//!
//! `a.mapv(|x| u8::try_from(x).unwrap())`
//!
//! </td><td>
//!
//! try to convert `i8` array to `u8` array, panic if any value cannot be converted lossless at runtime (e.g. negative value)
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a.astype(np.int32)`
//!
//! </td><td>
//!
//! `a.mapv(|x| x as i32)`
//!
//! </td><td>
//!
//! convert `f32` array to `i32` array with ["saturating" conversion][sat_conv]; care needed because it can be a lossy conversion or result in non-finite values! See [the reference for information][as_typecast].
//!
//! </td></tr>
//! </table>
//!
//! [as_conv]: https://doc.rust-lang.org/rust-by-example/types/cast.html
//! [sat_conv]: https://blog.rust-lang.org/2020/07/16/Rust-1.45.0.html#fixing-unsoundness-in-casts
//! [as_typecast]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
//!
//! ## Convenience methods for 2-D arrays
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `len(a)` or `a.shape[0]` | [`a.nrows()`][.nrows()] | get the number of rows in a 2-D array
//! `a.shape[1]` | [`a.ncols()`][.ncols()] | get the number of columns in a 2-D array
//! `a[1]` or `a[1,:]` | [`a.row(1)`][.row()] or [`a.row_mut(1)`][.row_mut()] | view (or mutable view) of row 1 in a 2-D array
//! `a[:,4]` | [`a.column(4)`][.column()] or [`a.column_mut(4)`][.column_mut()] | view (or mutable view) of column 4 in a 2-D array
//! `a.shape[0] == a.shape[1]` | [`a.is_square()`][.is_square()] | check if the array is square
//!
//! [.abs_diff_eq()]: ArrayBase#impl-AbsDiffEq<ArrayBase<S2%2C%20D>>
//! [.assign()]: ArrayBase::assign
//! [.axis_iter()]: ArrayBase::axis_iter
//! [.ncols()]: ArrayBase::ncols
//! [.column()]: ArrayBase::column
//! [.column_mut()]: ArrayBase::column_mut
//! [concatenate()]: crate::concatenate()
//! [concatenate!]: crate::concatenate!
//! [stack!]: crate::stack!
//! [::default()]: ArrayBase::default
//! [.diag()]: ArrayBase::diag
//! [.dim()]: ArrayBase::dim
//! [::eye()]: ArrayBase::eye
//! [.fill()]: ArrayBase::fill
//! [.fold()]: ArrayBase::fold
//! [.fold_axis()]: ArrayBase::fold_axis
//! [::from_elem()]: ArrayBase::from_elem
//! [::from_iter()]: ArrayBase::from_iter
//! [::from_diag()]: ArrayBase::from_diag
//! [::from_shape_fn()]: ArrayBase::from_shape_fn
//! [::from_shape_vec()]: ArrayBase::from_shape_vec
//! [::from_shape_vec_unchecked()]: ArrayBase::from_shape_vec_unchecked
//! [::from_vec()]: ArrayBase::from_vec
//! [.index()]: ArrayBase#impl-Index<I>
//! [.indexed_iter()]: ArrayBase::indexed_iter
//! [.insert_axis()]: ArrayBase::insert_axis
//! [.is_empty()]: ArrayBase::is_empty
//! [.is_square()]: ArrayBase::is_square
//! [.iter()]: ArrayBase::iter
//! [.len()]: ArrayBase::len
//! [.len_of()]: ArrayBase::len_of
//! [::linspace()]: ArrayBase::linspace
//! [::logspace()]: ArrayBase::logspace
//! [::geomspace()]: ArrayBase::geomspace
//! [.map()]: ArrayBase::map
//! [.map_axis()]: ArrayBase::map_axis
//! [.map_inplace()]: ArrayBase::map_inplace
//! [.mapv()]: ArrayBase::mapv
//! [.mapv_inplace()]: ArrayBase::mapv_inplace
//! [.mapv_into()]: ArrayBase::mapv_into
//! [matrix-* dot]: ArrayBase::dot-1
//! [.mean()]: ArrayBase::mean
//! [.mean_axis()]: ArrayBase::mean_axis
//! [.ndim()]: ArrayBase::ndim
//! [::ones()]: ArrayBase::ones
//! [.outer_iter()]: ArrayBase::outer_iter
//! [::range()]: ArrayBase::range
//! [.raw_dim()]: ArrayBase::raw_dim
//! [.reversed_axes()]: ArrayBase::reversed_axes
//! [.row()]: ArrayBase::row
//! [.row_mut()]: ArrayBase::row_mut
//! [.nrows()]: ArrayBase::nrows
//! [.sum()]: ArrayBase::sum
//! [.slice()]: ArrayBase::slice
//! [.slice_axis()]: ArrayBase::slice_axis
//! [.slice_collapse()]: ArrayBase::slice_collapse
//! [.slice_move()]: ArrayBase::slice_move
//! [.slice_mut()]: ArrayBase::slice_mut
//! [.shape()]: ArrayBase::shape
//! [stack()]: crate::stack()
//! [.strides()]: ArrayBase::strides
//! [.index_axis()]: ArrayBase::index_axis
//! [.sum_axis()]: ArrayBase::sum_axis
//! [.t()]: ArrayBase::t
//! [vec-* dot]: ArrayBase::dot
//! [.for_each()]: ArrayBase::for_each
//! [::zeros()]: ArrayBase::zeros
//! [`Zip`]: crate::Zip

pub mod coord_transform;
pub mod rk_step;
pub mod simple_math;

// This is to avoid putting `crate::` everywhere
#[allow(unused_imports)]
use crate::imp_prelude::*;
