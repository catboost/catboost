// Copyright 2014-2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "ndarray"]
#![doc(html_root_url = "https://docs.rs/ndarray/0.15/")]
#![doc(html_logo_url = "https://rust-ndarray.github.io/images/rust-ndarray_logo.svg")]
#![allow(
    unstable_name_collisions, // our `PointerExt` collides with upcoming inherent methods on `NonNull`
    clippy::deref_addrof,
    clippy::manual_map, // is not an error
    clippy::while_let_on_iterator, // is not an error
    clippy::from_iter_instead_of_collect, // using from_iter is good style
    clippy::incompatible_msrv, // false positive PointerExt::offset
)]
#![doc(test(attr(deny(warnings))))]
#![doc(test(attr(allow(unused_variables))))]
#![doc(test(attr(allow(deprecated))))]
#![cfg_attr(not(feature = "std"), no_std)]

//! The `ndarray` crate provides an *n*-dimensional container for general elements
//! and for numerics.
//!
//! In *n*-dimensional we include, for example, 1-dimensional rows or columns,
//! 2-dimensional matrices, and higher dimensional arrays. If the array has *n*
//! dimensions, then an element in the array is accessed by using that many indices.
//! Each dimension is also called an *axis*.
//!
//! - **[`ArrayBase`]**:
//!   The *n*-dimensional array type itself.<br>
//!   It is used to implement both the owned arrays and the views; see its docs
//!   for an overview of all array features.<br>
//! - The main specific array type is **[`Array`]**, which owns
//!   its elements.
//!
//! ## Highlights
//!
//! - Generic *n*-dimensional array
//! - [Slicing](ArrayBase#slicing), also with arbitrary step size, and negative
//!   indices to mean elements from the end of the axis.
//! - Views and subviews of arrays; iterators that yield subviews.
//! - Higher order operations and arithmetic are performant
//! - Array views can be used to slice and mutate any `[T]` data using
//!   `ArrayView::from` and `ArrayViewMut::from`.
//! - [`Zip`] for lock step function application across two or more arrays or other
//!   item producers ([`NdProducer`] trait).
//!
//! ## Crate Status
//!
//! - Still iterating on and evolving the crate
//!   + The crate is continuously developing, and breaking changes are expected
//!     during evolution from version to version. We adopt the newest stable
//!     rust features if we need them.
//!   + Note that functions/methods/traits/etc. hidden from the docs are not
//!     considered part of the public API, so changes to them are not
//!     considered breaking changes.
//! - Performance:
//!   + Prefer higher order methods and arithmetic operations on arrays first,
//!     then iteration, and as a last priority using indexed algorithms.
//!   + The higher order functions like [`.map()`](ArrayBase::map),
//!     [`.map_inplace()`](ArrayBase::map_inplace), [`.zip_mut_with()`](ArrayBase::zip_mut_with),
//!     [`Zip`] and [`azip!()`](azip) are the most efficient ways
//!     to perform single traversal and lock step traversal respectively.
//!   + Performance of an operation depends on the memory layout of the array
//!     or array view. Especially if it's a binary operation, which
//!     needs matching memory layout to be efficient (with some exceptions).
//!   + Efficient floating point matrix multiplication even for very large
//!     matrices; can optionally use BLAS to improve it further.
//!
//! - **MSRV: Requires Rust 1.64 or later**
//!
//! ## Crate Feature Flags
//!
//! The following crate feature flags are available. They are configured in your
//! `Cargo.toml`. See [`doc::crate_feature_flags`] for more information.
//!
//! - `std`: Rust standard library-using functionality (enabled by default)
//! - `serde`: serialization support for serde 1.x
//! - `rayon`: Parallel iterators, parallelized methods, the [`parallel`] module and [`par_azip!`].
//! - `approx` Implementations of traits from the [`approx`] crate.
//! - `blas`: transparent BLAS support for matrix multiplication, needs configuration.
//! - `matrixmultiply-threading`: Use threading from `matrixmultiply`.
//!
//! ## Documentation
//!
//! * The docs for [`ArrayBase`] provide an overview of
//!   the *n*-dimensional array type. Other good pages to look at are the
//!   documentation for the [`s![]`](s!) and
//!   [`azip!()`](azip!) macros.
//!
//! * If you have experience with NumPy, you may also be interested in
//!   [`ndarray_for_numpy_users`](doc::ndarray_for_numpy_users).
//!
//! ## The ndarray ecosystem
//!
//! `ndarray` provides a lot of functionality, but it's not a one-stop solution.
//!
//! `ndarray` includes matrix multiplication and other binary/unary operations out of the box.
//! More advanced linear algebra routines (e.g. SVD decomposition or eigenvalue computation)
//! can be found in [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg).
//!
//! The same holds for statistics: `ndarray` provides some basic functionalities (e.g. `mean`)
//! but more advanced routines can be found in [`ndarray-stats`](https://crates.io/crates/ndarray-stats).
//!
//! If you are looking to generate random arrays instead, check out [`ndarray-rand`](https://crates.io/crates/ndarray-rand).
//!
//! For conversion between `ndarray`, [`nalgebra`](https://crates.io/crates/nalgebra) and
//! [`image`](https://crates.io/crates/image) check out [`nshare`](https://crates.io/crates/nshare).

extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "blas")]
extern crate cblas_sys;

#[cfg(feature = "docs")]
pub mod doc;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use std::marker::PhantomData;

pub use crate::dimension::dim::*;
pub use crate::dimension::{Axis, AxisDescription, Dimension, IntoDimension, RemoveAxis};
pub use crate::dimension::{DimAdd, DimMax};

pub use crate::dimension::IxDynImpl;
pub use crate::dimension::NdIndex;
pub use crate::error::{ErrorKind, ShapeError};
pub use crate::indexes::{indices, indices_of};
pub use crate::order::Order;
pub use crate::slice::{MultiSliceArg, NewAxis, Slice, SliceArg, SliceInfo, SliceInfoElem, SliceNextDim};

use crate::iterators::Baseiter;
use crate::iterators::{ElementsBase, ElementsBaseMut, Iter, IterMut};

pub use crate::arraytraits::AsArray;
pub use crate::linalg_traits::LinalgScalar;
#[cfg(feature = "std")]
pub use crate::linalg_traits::NdFloat;

pub use crate::stacking::{concatenate, stack};

pub use crate::impl_views::IndexLonger;
pub use crate::math_cell::MathCell;
pub use crate::shape_builder::{Shape, ShapeArg, ShapeBuilder, StrideShape};

#[macro_use]
mod macro_utils;
#[macro_use]
mod private;
mod aliases;
#[macro_use]
mod itertools;
mod argument_traits;
#[cfg(feature = "serde")]
mod array_serde;
mod arrayformat;
mod arraytraits;
pub use crate::argument_traits::AssignElem;
mod data_repr;
mod data_traits;

pub use crate::aliases::*;

pub use crate::data_traits::{Data, DataMut, DataOwned, DataShared, RawData, RawDataClone, RawDataMut, RawDataSubst};

mod free_functions;
pub use crate::free_functions::*;
pub use crate::iterators::iter;

mod error;
mod extension;
mod geomspace;
mod indexes;
mod iterators;
mod layout;
mod linalg_traits;
mod linspace;
#[cfg(feature = "std")]
pub use crate::linspace::{linspace, range, Linspace};
mod logspace;
#[cfg(feature = "std")]
pub use crate::logspace::{logspace, Logspace};
mod math_cell;
mod numeric_util;
mod order;
mod partial;
mod shape_builder;
#[macro_use]
mod slice;
mod split_at;
mod stacking;
mod low_level_util;
#[macro_use]
mod zip;

mod dimension;

pub use crate::zip::{FoldWhile, IntoNdProducer, NdProducer, Zip};

pub use crate::layout::Layout;

/// Implementation's prelude. Common types used everywhere.
mod imp_prelude
{
    pub use crate::dimension::DimensionExt;
    pub use crate::prelude::*;
    pub use crate::ArcArray;
    pub use crate::{
        CowRepr,
        Data,
        DataMut,
        DataOwned,
        DataShared,
        Ix,
        Ixs,
        RawData,
        RawDataMut,
        RawViewRepr,
        RemoveAxis,
        ViewRepr,
    };
}

pub mod prelude;

/// Array index type
pub type Ix = usize;
/// Array index type (signed)
pub type Ixs = isize;

/// An *n*-dimensional array.
///
/// The array is a general container of elements.
/// The array supports arithmetic operations by applying them elementwise, if the
/// elements are numeric, but it supports non-numeric elements too.
///
/// The arrays rarely grow or shrink, since those operations can be costly. On
/// the other hand there is a rich set of methods and operations for taking views,
/// slices, and making traversals over one or more arrays.
///
/// In *n*-dimensional we include for example 1-dimensional rows or columns,
/// 2-dimensional matrices, and higher dimensional arrays. If the array has *n*
/// dimensions, then an element is accessed by using that many indices.
///
/// The `ArrayBase<S, D>` is parameterized by `S` for the data container and
/// `D` for the dimensionality.
///
/// Type aliases [`Array`], [`ArcArray`], [`CowArray`], [`ArrayView`], and
/// [`ArrayViewMut`] refer to `ArrayBase` with different types for the data
/// container: arrays with different kinds of ownership or different kinds of array views.
///
/// ## Contents
///
/// + [Array](#array)
/// + [ArcArray](#arcarray)
/// + [CowArray](#cowarray)
/// + [Array Views](#array-views)
/// + [Indexing and Dimension](#indexing-and-dimension)
/// + [Loops, Producers and Iterators](#loops-producers-and-iterators)
/// + [Slicing](#slicing)
/// + [Subviews](#subviews)
/// + [Arithmetic Operations](#arithmetic-operations)
/// + [Broadcasting](#broadcasting)
/// + [Conversions](#conversions)
/// + [Constructor Methods for Owned Arrays](#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](#methods-for-all-array-types)
/// + [Methods For 1-D Arrays](#methods-for-1-d-arrays)
/// + [Methods For 2-D Arrays](#methods-for-2-d-arrays)
/// + [Methods for Dynamic-Dimensional Arrays](#methods-for-dynamic-dimensional-arrays)
/// + [Numerical Methods for Arrays](#numerical-methods-for-arrays)
///
/// ## `Array`
///
/// [`Array`] is an owned array that owns the underlying array
/// elements directly (just like a `Vec`) and it is the default way to create and
/// store n-dimensional data. `Array<A, D>` has two type parameters: `A` for
/// the element type, and `D` for the dimensionality. A particular
/// dimensionality's type alias like `Array3<A>` just has the type parameter
/// `A` for element type.
///
/// An example:
///
/// ```
/// // Create a three-dimensional f64 array, initialized with zeros
/// use ndarray::Array3;
/// let mut temperature = Array3::<f64>::zeros((3, 4, 5));
/// // Increase the temperature in this location
/// temperature[[2, 2, 2]] += 0.5;
/// ```
///
/// ## `ArcArray`
///
/// [`ArcArray`] is an owned array with reference counted
/// data (shared ownership).
/// Sharing requires that it uses copy-on-write for mutable operations.
/// Calling a method for mutating elements on `ArcArray`, for example
/// [`view_mut()`](Self::view_mut) or [`get_mut()`](Self::get_mut),
/// will break sharing and require a clone of the data (if it is not uniquely held).
///
/// ## `CowArray`
///
/// [`CowArray`] is analogous to [`std::borrow::Cow`].
/// It can represent either an immutable view or a uniquely owned array. If a
/// `CowArray` instance is the immutable view variant, then calling a method
/// for mutating elements in the array will cause it to be converted into the
/// owned variant (by cloning all the elements) before the modification is
/// performed.
///
/// ## Array Views
///
/// [`ArrayView`] and [`ArrayViewMut`] are read-only and read-write array views
/// respectively. They use dimensionality, indexing, and almost all other
/// methods the same way as the other array types.
///
/// Methods for `ArrayBase` apply to array views too, when the trait bounds
/// allow.
///
/// Please see the documentation for the respective array view for an overview
/// of methods specific to array views: [`ArrayView`], [`ArrayViewMut`].
///
/// A view is created from an array using [`.view()`](ArrayBase::view),
/// [`.view_mut()`](ArrayBase::view_mut), using
/// slicing ([`.slice()`](ArrayBase::slice), [`.slice_mut()`](ArrayBase::slice_mut)) or from one of
/// the many iterators that yield array views.
///
/// You can also create an array view from a regular slice of data not
/// allocated with `Array` — see array view methods or their `From` impls.
///
/// Note that all `ArrayBase` variants can change their view (slicing) of the
/// data freely, even when their data can’t be mutated.
///
/// ## Indexing and Dimension
///
/// The dimensionality of the array determines the number of *axes*, for example
/// a 2D array has two axes. These are listed in “big endian” order, so that
/// the greatest dimension is listed first, the lowest dimension with the most
/// rapidly varying index is the last.
///
/// In a 2D array the index of each element is `[row, column]` as seen in this
/// 4 × 3 example:
///
/// ```ignore
/// [[ [0, 0], [0, 1], [0, 2] ],  // row 0
///  [ [1, 0], [1, 1], [1, 2] ],  // row 1
///  [ [2, 0], [2, 1], [2, 2] ],  // row 2
///  [ [3, 0], [3, 1], [3, 2] ]]  // row 3
/// //    \       \       \
/// //   column 0  \     column 2
/// //            column 1
/// ```
///
/// The number of axes for an array is fixed by its `D` type parameter: `Ix1`
/// for a 1D array, `Ix2` for a 2D array etc. The dimension type `IxDyn` allows
/// a dynamic number of axes.
///
/// A fixed size array (`[usize; N]`) of the corresponding dimensionality is
/// used to index the `Array`, making the syntax `array[[` i, j,  ...`]]`
///
/// ```
/// use ndarray::Array2;
/// let mut array = Array2::zeros((4, 3));
/// array[[1, 1]] = 7;
/// ```
///
/// Important traits and types for dimension and indexing:
///
/// - A [`struct@Dim`] value represents a dimensionality or index.
/// - Trait [`Dimension`] is implemented by all
///   dimensionalities. It defines many operations for dimensions and indices.
/// - Trait [`IntoDimension`] is used to convert into a
///   `Dim` value.
/// - Trait [`ShapeBuilder`] is an extension of
///   `IntoDimension` and is used when constructing an array. A shape describes
///   not just the extent of each axis but also their strides.
/// - Trait [`NdIndex`] is an extension of `Dimension` and is
///   for values that can be used with indexing syntax.
///
///
/// The default memory order of an array is *row major* order (a.k.a “c” order),
/// where each row is contiguous in memory.
/// A *column major* (a.k.a. “f” or fortran) memory order array has
/// columns (or, in general, the outermost axis) with contiguous elements.
///
/// The logical order of any array’s elements is the row major order
/// (the rightmost index is varying the fastest).
/// The iterators `.iter(), .iter_mut()` always adhere to this order, for example.
///
/// ## Loops, Producers and Iterators
///
/// Using [`Zip`] is the most general way to apply a procedure
/// across one or several arrays or *producers*.
///
/// [`NdProducer`] is like an iterable but for
/// multidimensional data. All producers have dimensions and axes, like an
/// array view, and they can be split and used with parallelization using `Zip`.
///
/// For example, `ArrayView<A, D>` is a producer, it has the same dimensions
/// as the array view and for each iteration it produces a reference to
/// the array element (`&A` in this case).
///
/// Another example, if we have a 10 × 10 array and use `.exact_chunks((2, 2))`
/// we get a producer of chunks which has the dimensions 5 × 5 (because
/// there are *10 / 2 = 5* chunks in either direction). The 5 × 5 chunks producer
/// can be paired with any other producers of the same dimension with `Zip`, for
/// example 5 × 5 arrays.
///
/// ### `.iter()` and `.iter_mut()`
///
/// These are the element iterators of arrays and they produce an element
/// sequence in the logical order of the array, that means that the elements
/// will be visited in the sequence that corresponds to increasing the
/// last index first: *0, ..., 0,  0*; *0, ..., 0, 1*; *0, ...0, 2* and so on.
///
/// ### `.outer_iter()` and `.axis_iter()`
///
/// These iterators produce array views of one smaller dimension.
///
/// For example, for a 2D array, `.outer_iter()` will produce the 1D rows.
/// For a 3D array, `.outer_iter()` produces 2D subviews.
///
/// `.axis_iter()` is like `outer_iter()` but allows you to pick which
/// axis to traverse.
///
/// The `outer_iter` and `axis_iter` are one dimensional producers.
///
/// ## `.rows()`, `.columns()` and `.lanes()`
///
/// [`.rows()`][gr] is a producer (and iterable) of all rows in an array.
///
/// ```
/// use ndarray::Array;
///
/// // 1. Loop over the rows of a 2D array
/// let mut a = Array::zeros((10, 10));
/// for mut row in a.rows_mut() {
///     row.fill(1.);
/// }
///
/// // 2. Use Zip to pair each row in 2D `a` with elements in 1D `b`
/// use ndarray::Zip;
/// let mut b = Array::zeros(a.nrows());
///
/// Zip::from(a.rows())
///     .and(&mut b)
///     .for_each(|a_row, b_elt| {
///         *b_elt = a_row[a.ncols() - 1] - a_row[0];
///     });
/// ```
///
/// The *lanes* of an array are 1D segments along an axis and when pointed
/// along the last axis they are *rows*, when pointed along the first axis
/// they are *columns*.
///
/// A *m* × *n* array has *m* rows each of length *n* and conversely
/// *n* columns each of length *m*.
///
/// To generalize this, we say that an array of dimension *a* × *m* × *n*
/// has *a m* rows. It's composed of *a* times the previous array, so it
/// has *a* times as many rows.
///
/// All methods: [`.rows()`][gr], [`.rows_mut()`][grm],
/// [`.columns()`][gc], [`.columns_mut()`][gcm],
/// [`.lanes(axis)`][l], [`.lanes_mut(axis)`][lm].
///
/// [gr]: Self::rows
/// [grm]: Self::rows_mut
/// [gc]: Self::columns
/// [gcm]: Self::columns_mut
/// [l]: Self::lanes
/// [lm]: Self::lanes_mut
///
/// Yes, for 2D arrays `.rows()` and `.outer_iter()` have about the same
/// effect:
///
///  + `rows()` is a producer with *n* - 1 dimensions of 1 dimensional items
///  + `outer_iter()` is a producer with 1 dimension of *n* - 1 dimensional items
///
/// ## Slicing
///
/// You can use slicing to create a view of a subset of the data in
/// the array. Slicing methods include [`.slice()`], [`.slice_mut()`],
/// [`.slice_move()`], and [`.slice_collapse()`].
///
/// The slicing argument can be passed using the macro [`s![]`](s!),
/// which will be used in all examples. (The explicit form is an instance of
/// [`SliceInfo`] or another type which implements [`SliceArg`]; see their docs
/// for more information.)
///
/// If a range is used, the axis is preserved. If an index is used, that index
/// is selected and the axis is removed; this selects a subview. See
/// [*Subviews*](#subviews) for more information about subviews. If a
/// [`NewAxis`] instance is used, a new axis is inserted. Note that
/// [`.slice_collapse()`] panics on `NewAxis` elements and behaves like
/// [`.collapse_axis()`] by preserving the number of dimensions.
///
/// [`.slice()`]: Self::slice
/// [`.slice_mut()`]: Self::slice_mut
/// [`.slice_move()`]: Self::slice_move
/// [`.slice_collapse()`]: Self::slice_collapse
///
/// When slicing arrays with generic dimensionality, creating an instance of
/// [`SliceInfo`] to pass to the multi-axis slicing methods like [`.slice()`]
/// is awkward. In these cases, it's usually more convenient to use
/// [`.slice_each_axis()`]/[`.slice_each_axis_mut()`]/[`.slice_each_axis_inplace()`]
/// or to create a view and then slice individual axes of the view using
/// methods such as [`.slice_axis_inplace()`] and [`.collapse_axis()`].
///
/// [`.slice_each_axis()`]: Self::slice_each_axis
/// [`.slice_each_axis_mut()`]: Self::slice_each_axis_mut
/// [`.slice_each_axis_inplace()`]: Self::slice_each_axis_inplace
/// [`.slice_axis_inplace()`]: Self::slice_axis_inplace
/// [`.collapse_axis()`]: Self::collapse_axis
///
/// It's possible to take multiple simultaneous *mutable* slices with
/// [`.multi_slice_mut()`] or (for [`ArrayViewMut`] only)
/// [`.multi_slice_move()`].
///
/// [`.multi_slice_mut()`]: Self::multi_slice_mut
/// [`.multi_slice_move()`]: ArrayViewMut#method.multi_slice_move
///
/// ```
/// use ndarray::{arr2, arr3, s, ArrayBase, DataMut, Dimension, NewAxis, Slice};
///
/// // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`.
///
/// let a = arr3(&[[[ 1,  2,  3],     // -- 2 rows  \_
///                 [ 4,  5,  6]],    // --         /
///                [[ 7,  8,  9],     //            \_ 2 submatrices
///                 [10, 11, 12]]]);  //            /
/// //  3 columns ..../.../.../
///
/// assert_eq!(a.shape(), &[2, 2, 3]);
///
/// // Let’s create a slice with
/// //
/// // - Both of the submatrices of the greatest dimension: `..`
/// // - Only the first row in each submatrix: `0..1`
/// // - Every element in each row: `..`
///
/// let b = a.slice(s![.., 0..1, ..]);
/// let c = arr3(&[[[ 1,  2,  3]],
///                [[ 7,  8,  9]]]);
/// assert_eq!(b, c);
/// assert_eq!(b.shape(), &[2, 1, 3]);
///
/// // Let’s create a slice with
/// //
/// // - Both submatrices of the greatest dimension: `..`
/// // - The last row in each submatrix: `-1..`
/// // - Row elements in reverse order: `..;-1`
/// let d = a.slice(s![.., -1.., ..;-1]);
/// let e = arr3(&[[[ 6,  5,  4]],
///                [[12, 11, 10]]]);
/// assert_eq!(d, e);
/// assert_eq!(d.shape(), &[2, 1, 3]);
///
/// // Let’s create a slice while selecting a subview and inserting a new axis with
/// //
/// // - Both submatrices of the greatest dimension: `..`
/// // - The last row in each submatrix, removing that axis: `-1`
/// // - Row elements in reverse order: `..;-1`
/// // - A new axis at the end.
/// let f = a.slice(s![.., -1, ..;-1, NewAxis]);
/// let g = arr3(&[[ [6],  [5],  [4]],
///                [[12], [11], [10]]]);
/// assert_eq!(f, g);
/// assert_eq!(f.shape(), &[2, 3, 1]);
///
/// // Let's take two disjoint, mutable slices of a matrix with
/// //
/// // - One containing all the even-index columns in the matrix
/// // - One containing all the odd-index columns in the matrix
/// let mut h = arr2(&[[0, 1, 2, 3],
///                    [4, 5, 6, 7]]);
/// let (s0, s1) = h.multi_slice_mut((s![.., ..;2], s![.., 1..;2]));
/// let i = arr2(&[[0, 2],
///                [4, 6]]);
/// let j = arr2(&[[1, 3],
///                [5, 7]]);
/// assert_eq!(s0, i);
/// assert_eq!(s1, j);
///
/// // Generic function which assigns the specified value to the elements which
/// // have indices in the lower half along all axes.
/// fn fill_lower<S, D>(arr: &mut ArrayBase<S, D>, x: S::Elem)
/// where
///     S: DataMut,
///     S::Elem: Clone,
///     D: Dimension,
/// {
///     arr.slice_each_axis_mut(|ax| Slice::from(0..ax.len / 2)).fill(x);
/// }
/// fill_lower(&mut h, 9);
/// let k = arr2(&[[9, 9, 2, 3],
///                [4, 5, 6, 7]]);
/// assert_eq!(h, k);
/// ```
///
/// ## Subviews
///
/// Subview methods allow you to restrict the array view while removing one
/// axis from the array. Methods for selecting individual subviews include
/// [`.index_axis()`], [`.index_axis_mut()`], [`.index_axis_move()`], and
/// [`.index_axis_inplace()`]. You can also select a subview by using a single
/// index instead of a range when slicing. Some other methods, such as
/// [`.fold_axis()`], [`.axis_iter()`], [`.axis_iter_mut()`],
/// [`.outer_iter()`], and [`.outer_iter_mut()`] operate on all the subviews
/// along an axis.
///
/// A related method is [`.collapse_axis()`], which modifies the view in the
/// same way as [`.index_axis()`] except for removing the collapsed axis, since
/// it operates *in place*. The length of the axis becomes 1.
///
/// Methods for selecting an individual subview take two arguments: `axis` and
/// `index`.
///
/// [`.axis_iter()`]: Self::axis_iter
/// [`.axis_iter_mut()`]: Self::axis_iter_mut
/// [`.fold_axis()`]: Self::fold_axis
/// [`.index_axis()`]: Self::index_axis
/// [`.index_axis_inplace()`]: Self::index_axis_inplace
/// [`.index_axis_mut()`]: Self::index_axis_mut
/// [`.index_axis_move()`]: Self::index_axis_move
/// [`.collapse_axis()`]: Self::collapse_axis
/// [`.outer_iter()`]: Self::outer_iter
/// [`.outer_iter_mut()`]: Self::outer_iter_mut
///
/// ```
///
/// use ndarray::{arr3, aview1, aview2, s, Axis};
///
///
/// // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`.
///
/// let a = arr3(&[[[ 1,  2,  3],    // \ axis 0, submatrix 0
///                 [ 4,  5,  6]],   // /
///                [[ 7,  8,  9],    // \ axis 0, submatrix 1
///                 [10, 11, 12]]]); // /
///         //        \
///         //         axis 2, column 0
///
/// assert_eq!(a.shape(), &[2, 2, 3]);
///
/// // Let’s take a subview along the greatest dimension (axis 0),
/// // taking submatrix 0, then submatrix 1
///
/// let sub_0 = a.index_axis(Axis(0), 0);
/// let sub_1 = a.index_axis(Axis(0), 1);
///
/// assert_eq!(sub_0, aview2(&[[ 1,  2,  3],
///                            [ 4,  5,  6]]));
/// assert_eq!(sub_1, aview2(&[[ 7,  8,  9],
///                            [10, 11, 12]]));
/// assert_eq!(sub_0.shape(), &[2, 3]);
///
/// // This is the subview picking only axis 2, column 0
/// let sub_col = a.index_axis(Axis(2), 0);
///
/// assert_eq!(sub_col, aview2(&[[ 1,  4],
///                              [ 7, 10]]));
///
/// // You can take multiple subviews at once (and slice at the same time)
/// let double_sub = a.slice(s![1, .., 0]);
/// assert_eq!(double_sub, aview1(&[7, 10]));
/// ```
///
/// ## Arithmetic Operations
///
/// Arrays support all arithmetic operations the same way: they apply elementwise.
///
/// Since the trait implementations are hard to overview, here is a summary.
///
/// ### Binary Operators with Two Arrays
///
/// Let `A` be an array or view of any kind. Let `B` be an array
/// with owned storage (either `Array` or `ArcArray`).
/// Let `C` be an array with mutable data (either `Array`, `ArcArray`
/// or `ArrayViewMut`).
/// The following combinations of operands
/// are supported for an arbitrary binary operator denoted by `@` (it can be
/// `+`, `-`, `*`, `/` and so on).
///
/// - `&A @ &A` which produces a new `Array`
/// - `B @ A` which consumes `B`, updates it with the result, and returns it
/// - `B @ &A` which consumes `B`, updates it with the result, and returns it
/// - `C @= &A` which performs an arithmetic operation in place
///
/// Note that the element type needs to implement the operator trait and the
/// `Clone` trait.
///
/// ```
/// use ndarray::{array, ArrayView1};
///
/// let owned1 = array![1, 2];
/// let owned2 = array![3, 4];
/// let view1 = ArrayView1::from(&[5, 6]);
/// let view2 = ArrayView1::from(&[7, 8]);
/// let mut mutable = array![9, 10];
///
/// let sum1 = &view1 + &view2;   // Allocates a new array. Note the explicit `&`.
/// // let sum2 = view1 + &view2; // This doesn't work because `view1` is not an owned array.
/// let sum3 = owned1 + view1;    // Consumes `owned1`, updates it, and returns it.
/// let sum4 = owned2 + &view2;   // Consumes `owned2`, updates it, and returns it.
/// mutable += &view2;            // Updates `mutable` in-place.
/// ```
///
/// ### Binary Operators with Array and Scalar
///
/// The trait [`ScalarOperand`] marks types that can be used in arithmetic
/// with arrays directly. For a scalar `K` the following combinations of operands
/// are supported (scalar can be on either the left or right side, but
/// `ScalarOperand` docs has the detailed conditions).
///
/// - `&A @ K` or `K @ &A` which produces a new `Array`
/// - `B @ K` or `K @ B` which consumes `B`, updates it with the result and returns it
/// - `C @= K` which performs an arithmetic operation in place
///
/// ### Unary Operators
///
/// Let `A` be an array or view of any kind. Let `B` be an array with owned
/// storage (either `Array` or `ArcArray`). The following operands are supported
/// for an arbitrary unary operator denoted by `@` (it can be `-` or `!`).
///
/// - `@&A` which produces a new `Array`
/// - `@B` which consumes `B`, updates it with the result, and returns it
///
/// ## Broadcasting
///
/// Arrays support limited *broadcasting*, where arithmetic operations with
/// array operands of different sizes can be carried out by repeating the
/// elements of the smaller dimension array. See
/// [`.broadcast()`](Self::broadcast) for a more detailed
/// description.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1., 1.],
///                [1., 2.],
///                [0., 3.],
///                [0., 4.]]);
///
/// let b = arr2(&[[0., 1.]]);
///
/// let c = arr2(&[[1., 2.],
///                [1., 3.],
///                [0., 4.],
///                [0., 5.]]);
/// // We can add because the shapes are compatible even if not equal.
/// // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
/// assert!(
///     c == a + b
/// );
/// ```
///
/// ## Conversions
///
/// ### Conversions Between Array Types
///
/// This table is a summary of the conversions between arrays of different
/// ownership, dimensionality, and element type. All of the conversions in this
/// table preserve the shape of the array.
///
/// <table>
/// <tr>
/// <th rowspan="2">Output</th>
/// <th colspan="5">Input</th>
/// </tr>
///
/// <tr>
/// <td>
///
/// `Array<A, D>`
///
/// </td>
/// <td>
///
/// `ArcArray<A, D>`
///
/// </td>
/// <td>
///
/// `CowArray<'a, A, D>`
///
/// </td>
/// <td>
///
/// `ArrayView<'a, A, D>`
///
/// </td>
/// <td>
///
/// `ArrayViewMut<'a, A, D>`
///
/// </td>
/// </tr>
///
/// <!--Conversions to `Array<A, D>`-->
///
/// <tr>
/// <td>
///
/// `Array<A, D>`
///
/// </td>
/// <td>
///
/// no-op
///
/// </td>
/// <td>
///
/// [`a.into_owned()`][.into_owned()]
///
/// </td>
/// <td>
///
/// [`a.into_owned()`][.into_owned()]
///
/// </td>
/// <td>
///
/// [`a.to_owned()`][.to_owned()]
///
/// </td>
/// <td>
///
/// [`a.to_owned()`][.to_owned()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to `ArcArray<A, D>`-->
///
/// <tr>
/// <td>
///
/// `ArcArray<A, D>`
///
/// </td>
/// <td>
///
/// [`a.into_shared()`][.into_shared()]
///
/// </td>
/// <td>
///
/// no-op
///
/// </td>
/// <td>
///
/// [`a.into_owned().into_shared()`][.into_shared()]
///
/// </td>
/// <td>
///
/// [`a.to_owned().into_shared()`][.into_shared()]
///
/// </td>
/// <td>
///
/// [`a.to_owned().into_shared()`][.into_shared()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to `CowArray<'a, A, D>`-->
///
/// <tr>
/// <td>
///
/// `CowArray<'a, A, D>`
///
/// </td>
/// <td>
///
/// [`CowArray::from(a)`](CowArray#impl-From<ArrayBase<OwnedRepr<A>%2C%20D>>)
///
/// </td>
/// <td>
///
/// [`CowArray::from(a.into_owned())`](CowArray#impl-From<ArrayBase<OwnedRepr<A>%2C%20D>>)
///
/// </td>
/// <td>
///
/// no-op
///
/// </td>
/// <td>
///
/// [`CowArray::from(a)`](CowArray#impl-From<ArrayBase<ViewRepr<%26%27a%20A>%2C%20D>>)
///
/// </td>
/// <td>
///
/// [`CowArray::from(a.view())`](CowArray#impl-From<ArrayBase<ViewRepr<%26%27a%20A>%2C%20D>>)
///
/// </td>
/// </tr>
///
/// <!--Conversions to `ArrayView<'b, A, D>`-->
///
/// <tr>
/// <td>
///
/// `ArrayView<'b, A, D>`
///
/// </td>
/// <td>
///
/// [`a.view()`][.view()]
///
/// </td>
/// <td>
///
/// [`a.view()`][.view()]
///
/// </td>
/// <td>
///
/// [`a.view()`][.view()]
///
/// </td>
/// <td>
///
/// [`a.view()`][.view()] or [`a.reborrow()`][ArrayView::reborrow()]
///
/// </td>
/// <td>
///
/// [`a.view()`][.view()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to `ArrayViewMut<'b, A, D>`-->
///
/// <tr>
/// <td>
///
/// `ArrayViewMut<'b, A, D>`
///
/// </td>
/// <td>
///
/// [`a.view_mut()`][.view_mut()]
///
/// </td>
/// <td>
///
/// [`a.view_mut()`][.view_mut()]
///
/// </td>
/// <td>
///
/// [`a.view_mut()`][.view_mut()]
///
/// </td>
/// <td>
///
/// illegal
///
/// </td>
/// <td>
///
/// [`a.view_mut()`][.view_mut()] or [`a.reborrow()`][ArrayViewMut::reborrow()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to equivalent with dim `D2`-->
///
/// <tr>
/// <td>
///
/// equivalent with dim `D2` (e.g. converting from dynamic dim to const dim)
///
/// </td>
/// <td colspan="5">
///
/// [`a.into_dimensionality::<D2>()`][.into_dimensionality()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to equivalent with dim `IxDyn`-->
///
/// <tr>
/// <td>
///
/// equivalent with dim `IxDyn`
///
/// </td>
/// <td colspan="5">
///
/// [`a.into_dyn()`][.into_dyn()]
///
/// </td>
/// </tr>
///
/// <!--Conversions to `Array<B, D>`-->
///
/// <tr>
/// <td>
///
/// `Array<B, D>` (new element type)
///
/// </td>
/// <td colspan="5">
///
/// [`a.map(|x| x.do_your_conversion())`][.map()]
///
/// </td>
/// </tr>
/// </table>
///
/// ### Conversions Between Arrays and `Vec`s/Slices/Scalars
///
/// This is a table of the safe conversions between arrays and
/// `Vec`s/slices/scalars. Note that some of the return values are actually
/// `Result`/`Option` wrappers around the indicated output types.
///
/// Input | Output | Methods
/// ------|--------|--------
/// `Vec<A>` | `ArrayBase<S: DataOwned, Ix1>` | [`::from_vec()`](Self::from_vec)
/// `Vec<A>` | `ArrayBase<S: DataOwned, D>` | [`::from_shape_vec()`](Self::from_shape_vec)
/// `&[A]` | `ArrayView1<A>` | [`::from()`](ArrayView#method.from)
/// `&[A]` | `ArrayView<A, D>` | [`::from_shape()`](ArrayView#method.from_shape)
/// `&mut [A]` | `ArrayViewMut1<A>` | [`::from()`](ArrayViewMut#method.from)
/// `&mut [A]` | `ArrayViewMut<A, D>` | [`::from_shape()`](ArrayViewMut#method.from_shape)
/// `&ArrayBase<S, Ix1>` | `Vec<A>` | [`.to_vec()`](Self::to_vec)
/// `Array<A, D>` | `Vec<A>` | [`.into_raw_vec()`](Array#method.into_raw_vec)<sup>[1](#into_raw_vec)</sup>
/// `&ArrayBase<S, D>` | `&[A]` | [`.as_slice()`](Self::as_slice)<sup>[2](#req_contig_std)</sup>, [`.as_slice_memory_order()`](Self::as_slice_memory_order)<sup>[3](#req_contig)</sup>
/// `&mut ArrayBase<S: DataMut, D>` | `&mut [A]` | [`.as_slice_mut()`](Self::as_slice_mut)<sup>[2](#req_contig_std)</sup>, [`.as_slice_memory_order_mut()`](Self::as_slice_memory_order_mut)<sup>[3](#req_contig)</sup>
/// `ArrayView<A, D>` | `&[A]` | [`.to_slice()`](ArrayView#method.to_slice)<sup>[2](#req_contig_std)</sup>
/// `ArrayViewMut<A, D>` | `&mut [A]` | [`.into_slice()`](ArrayViewMut#method.into_slice)<sup>[2](#req_contig_std)</sup>
/// `Array0<A>` | `A` | [`.into_scalar()`](Array#method.into_scalar)
///
/// <sup><a name="into_raw_vec">1</a></sup>Returns the data in memory order.
///
/// <sup><a name="req_contig_std">2</a></sup>Works only if the array is
/// contiguous and in standard order.
///
/// <sup><a name="req_contig">3</a></sup>Works only if the array is contiguous.
///
/// The table above does not include all the constructors; it only shows
/// conversions to/from `Vec`s/slices. See
/// [below](#constructor-methods-for-owned-arrays) for more constructors.
///
/// [ArrayView::reborrow()]: ArrayView#method.reborrow
/// [ArrayViewMut::reborrow()]: ArrayViewMut#method.reborrow
/// [.into_dimensionality()]: Self::into_dimensionality
/// [.into_dyn()]: Self::into_dyn
/// [.into_owned()]: Self::into_owned
/// [.into_shared()]: Self::into_shared
/// [.to_owned()]: Self::to_owned
/// [.map()]: Self::map
/// [.view()]: Self::view
/// [.view_mut()]: Self::view_mut
///
/// ### Conversions from Nested `Vec`s/`Array`s
///
/// It's generally a good idea to avoid nested `Vec`/`Array` types, such as
/// `Vec<Vec<A>>` or `Vec<Array2<A>>` because:
///
/// * they require extra heap allocations compared to a single `Array`,
///
/// * they can scatter data all over memory (because of multiple allocations),
///
/// * they cause unnecessary indirection (traversing multiple pointers to reach
///   the data),
///
/// * they don't enforce consistent shape within the nested
///   `Vec`s/`ArrayBase`s, and
///
/// * they are generally more difficult to work with.
///
/// The most common case where users might consider using nested
/// `Vec`s/`Array`s is when creating an array by appending rows/subviews in a
/// loop, where the rows/subviews are computed within the loop. However, there
/// are better ways than using nested `Vec`s/`Array`s.
///
/// If you know ahead-of-time the shape of the final array, the cleanest
/// solution is to allocate the final array before the loop, and then assign
/// the data to it within the loop, like this:
///
/// ```rust
/// use ndarray::{array, Array2, Axis};
///
/// let mut arr = Array2::zeros((2, 3));
/// for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
///     // Perform calculations and assign to `row`; this is a trivial example:
///     row.fill(i);
/// }
/// assert_eq!(arr, array![[0, 0, 0], [1, 1, 1]]);
/// ```
///
/// If you don't know ahead-of-time the shape of the final array, then the
/// cleanest solution is generally to append the data to a flat `Vec`, and then
/// convert it to an `Array` at the end with
/// [`::from_shape_vec()`](Self::from_shape_vec). You just have to be careful
/// that the layout of the data (the order of the elements in the flat `Vec`)
/// is correct.
///
/// ```rust
/// use ndarray::{array, Array2};
///
/// let ncols = 3;
/// let mut data = Vec::new();
/// let mut nrows = 0;
/// for i in 0..2 {
///     // Compute `row` and append it to `data`; this is a trivial example:
///     let row = vec![i; ncols];
///     data.extend_from_slice(&row);
///     nrows += 1;
/// }
/// let arr = Array2::from_shape_vec((nrows, ncols), data)?;
/// assert_eq!(arr, array![[0, 0, 0], [1, 1, 1]]);
/// # Ok::<(), ndarray::ShapeError>(())
/// ```
///
/// If neither of these options works for you, and you really need to convert
/// nested `Vec`/`Array` instances to an `Array`, the cleanest solution is
/// generally to use [`Iterator::flatten()`]
/// to get a flat `Vec`, and then convert the `Vec` to an `Array` with
/// [`::from_shape_vec()`](Self::from_shape_vec), like this:
///
/// ```rust
/// use ndarray::{array, Array2, Array3};
///
/// let nested: Vec<Array2<i32>> = vec![
///     array![[1, 2, 3], [4, 5, 6]],
///     array![[7, 8, 9], [10, 11, 12]],
/// ];
/// let inner_shape = nested[0].dim();
/// let shape = (nested.len(), inner_shape.0, inner_shape.1);
/// let flat: Vec<i32> = nested.iter().flatten().cloned().collect();
/// let arr = Array3::from_shape_vec(shape, flat)?;
/// assert_eq!(arr, array![
///     [[1, 2, 3], [4, 5, 6]],
///     [[7, 8, 9], [10, 11, 12]],
/// ]);
/// # Ok::<(), ndarray::ShapeError>(())
/// ```
///
/// Note that this implementation assumes that the nested `Vec`s are all the
/// same shape and that the `Vec` is non-empty. Depending on your application,
/// it may be a good idea to add checks for these assumptions and possibly
/// choose a different way to handle the empty case.
///
// # For implementors
//
// All methods must uphold the following constraints:
//
// 1. `data` must correctly represent the data buffer / ownership information,
//    `ptr` must point into the data represented by `data`, and the `dim` and
//    `strides` must be consistent with `data`. For example,
//
//    * If `data` is `OwnedRepr<A>`, all elements represented by `ptr`, `dim`,
//      and `strides` must be owned by the `Vec` and not aliased by multiple
//      indices.
//
//    * If `data` is `ViewRepr<&'a mut A>`, all elements represented by `ptr`,
//      `dim`, and `strides` must be exclusively borrowed and not aliased by
//      multiple indices.
//
// 2. If the type of `data` implements `Data`, then `ptr` must be aligned.
//
// 3. `ptr` must be non-null, and it must be safe to [`.offset()`] `ptr` by
//    zero.
//
// 4. It must be safe to [`.offset()`] the pointer repeatedly along all axes
//    and calculate the `count`s for the `.offset()` calls without overflow,
//    even if the array is empty or the elements are zero-sized.
//
//    More specifically, the set of all possible (signed) offset counts
//    relative to `ptr` can be determined by the following (the casts and
//    arithmetic must not overflow):
//
//    ```rust
//    /// Returns all the possible offset `count`s relative to `ptr`.
//    fn all_offset_counts(shape: &[usize], strides: &[isize]) -> BTreeSet<isize> {
//        assert_eq!(shape.len(), strides.len());
//        let mut all_offsets = BTreeSet::<isize>::new();
//        all_offsets.insert(0);
//        for axis in 0..shape.len() {
//            let old_offsets = all_offsets.clone();
//            for index in 0..shape[axis] {
//                assert!(index <= isize::MAX as usize);
//                let off = (index as isize).checked_mul(strides[axis]).unwrap();
//                for &old_offset in &old_offsets {
//                    all_offsets.insert(old_offset.checked_add(off).unwrap());
//                }
//            }
//        }
//        all_offsets
//    }
//    ```
//
//    Note that it must be safe to offset the pointer *repeatedly* along all
//    axes, so in addition for it being safe to offset `ptr` by each of these
//    counts, the difference between the least and greatest address reachable
//    by these offsets in units of `A` and in units of bytes must not be
//    greater than `isize::MAX`.
//
//    In other words,
//
//    * All possible pointers generated by moving along all axes must be in
//      bounds or one byte past the end of a single allocation with element
//      type `A`. The only exceptions are if the array is empty or the element
//      type is zero-sized. In these cases, `ptr` may be dangling, but it must
//      still be safe to [`.offset()`] the pointer along the axes.
//
//    * The offset in units of bytes between the least address and greatest
//      address by moving along all axes must not exceed `isize::MAX`. This
//      constraint prevents the computed offset, in bytes, from overflowing
//      `isize` regardless of the starting point due to past offsets.
//
//    * The offset in units of `A` between the least address and greatest
//      address by moving along all axes must not exceed `isize::MAX`. This
//      constraint prevents overflow when calculating the `count` parameter to
//      [`.offset()`] regardless of the starting point due to past offsets.
//
//    For example, if the shape is [2, 0, 3] and the strides are [3, 6, -1],
//    the offsets of interest relative to `ptr` are -2, -1, 0, 1, 2, 3. So,
//    `ptr.offset(-2)`, `ptr.offset(-1)`, …, `ptr.offset(3)` must be pointers
//    within a single allocation with element type `A`; `(3 - (-2)) *
//    size_of::<A>()` must not exceed `isize::MAX`, and `3 - (-2)` must not
//    exceed `isize::MAX`. Note that this is a requirement even though the
//    array is empty (axis 1 has length 0).
//
//    A dangling pointer can be used when creating an empty array, but this
//    usually means all the strides have to be zero. A dangling pointer that
//    can safely be offset by zero bytes can be constructed with
//    `::std::ptr::NonNull::<A>::dangling().as_ptr()`. (It isn't entirely clear
//    from the documentation that a pointer created this way is safe to
//    `.offset()` at all, even by zero bytes, but the implementation of
//    `Vec<A>` does this, so we can too. See rust-lang/rust#54857 for details.)
//
// 5. The product of non-zero axis lengths must not exceed `isize::MAX`. (This
//    also implies that the length of any individual axis must not exceed
//    `isize::MAX`, and an array can contain at most `isize::MAX` elements.)
//    This constraint makes various calculations easier because they don't have
//    to worry about overflow and axis lengths can be freely cast to `isize`.
//
// Constraints 2–5 are carefully designed such that if they're upheld for the
// array, they're also upheld for any subset of axes of the array as well as
// slices/subviews/reshapes of the array. This is important for iterators that
// produce subviews (and other similar cases) to be safe without extra (easy to
// forget) checks for zero-length axes. Constraint 1 is similarly upheld for
// any subset of axes and slices/subviews/reshapes, except when removing a
// zero-length axis (since if the other axes are non-zero-length, that would
// allow accessing elements that should not be possible to access).
//
// Method/function implementations can rely on these constraints being upheld.
// The constraints can be temporarily violated within a method/function
// implementation since `ArrayBase` doesn't implement `Drop` and `&mut
// ArrayBase` is `!UnwindSafe`, but the implementation must not call
// methods/functions on the array while it violates the constraints.
//
// Users of the `ndarray` crate cannot rely on these constraints because they
// may change in the future.
//
// [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset-1
pub struct ArrayBase<S, D>
where S: RawData
{
    /// Data buffer / ownership information. (If owned, contains the data
    /// buffer; if borrowed, contains the lifetime and mutability.)
    data: S,
    /// A non-null pointer into the buffer held by `data`; may point anywhere
    /// in its range. If `S: Data`, this pointer must be aligned.
    ptr: std::ptr::NonNull<S::Elem>,
    /// The lengths of the axes.
    dim: D,
    /// The element count stride per axis. To be parsed as `isize`.
    strides: D,
}

/// An array where the data has shared ownership and is copy on write.
///
/// The `ArcArray<A, D>` is parameterized by `A` for the element type and `D` for
/// the dimensionality.
///
/// It can act as both an owner as the data as well as a shared reference (view
/// like).
/// Calling a method for mutating elements on `ArcArray`, for example
/// [`view_mut()`](ArrayBase::view_mut) or
/// [`get_mut()`](ArrayBase::get_mut), will break sharing and
/// require a clone of the data (if it is not uniquely held).
///
/// `ArcArray` uses atomic reference counting like `Arc`, so it is `Send` and
/// `Sync` (when allowed by the element type of the array too).
///
/// **[`ArrayBase`]** is used to implement both the owned
/// arrays and the views; see its docs for an overview of all array features.
///
/// See also:
///
/// + [Constructor Methods for Owned Arrays](ArrayBase#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](ArrayBase#methods-for-all-array-types)
pub type ArcArray<A, D> = ArrayBase<OwnedArcRepr<A>, D>;

/// An array that owns its data uniquely.
///
/// `Array` is the main n-dimensional array type, and it owns all its array
/// elements.
///
/// The `Array<A, D>` is parameterized by `A` for the element type and `D` for
/// the dimensionality.
///
/// **[`ArrayBase`]** is used to implement both the owned
/// arrays and the views; see its docs for an overview of all array features.
///
/// See also:
///
/// + [Constructor Methods for Owned Arrays](ArrayBase#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](ArrayBase#methods-for-all-array-types)
/// + Dimensionality-specific type alises
///   [`Array1`],
///   [`Array2`],
///   [`Array3`], ...,
///   [`ArrayD`],
///   and so on.
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;

/// An array with copy-on-write behavior.
///
/// An `CowArray` represents either a uniquely owned array or a view of an
/// array. The `'a` corresponds to the lifetime of the view variant.
///
/// This type is analogous to [`std::borrow::Cow`].
/// If a `CowArray` instance is the immutable view variant, then calling a
/// method for mutating elements in the array will cause it to be converted
/// into the owned variant (by cloning all the elements) before the
/// modification is performed.
///
/// Array views have all the methods of an array (see [`ArrayBase`]).
///
/// See also [`ArcArray`], which also provides
/// copy-on-write behavior but has a reference-counted pointer to the data
/// instead of either a view or a uniquely owned copy.
pub type CowArray<'a, A, D> = ArrayBase<CowRepr<'a, A>, D>;

/// A read-only array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// The `ArrayView<'a, A, D>` is parameterized by `'a` for the scope of the
/// borrow, `A` for the element type and `D` for the dimensionality.
///
/// Array views have all the methods of an array (see [`ArrayBase`]).
///
/// See also [`ArrayViewMut`].
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;

/// A read-write array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// The `ArrayViewMut<'a, A, D>` is parameterized by `'a` for the scope of the
/// borrow, `A` for the element type and `D` for the dimensionality.
///
/// Array views have all the methods of an array (see [`ArrayBase`]).
///
/// See also [`ArrayView`].
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;

/// A read-only array view without a lifetime.
///
/// This is similar to [`ArrayView`] but does not carry any lifetime or
/// ownership information, and its data cannot be read without an unsafe
/// conversion into an [`ArrayView`]. The relationship between `RawArrayView`
/// and [`ArrayView`] is somewhat analogous to the relationship between `*const
/// T` and `&T`, but `RawArrayView` has additional requirements that `*const T`
/// does not, such as non-nullness.
///
/// The `RawArrayView<A, D>` is parameterized by `A` for the element type and
/// `D` for the dimensionality.
///
/// Raw array views have all the methods of an array (see
/// [`ArrayBase`]).
///
/// See also [`RawArrayViewMut`].
///
/// # Warning
///
/// You can't use this type with an arbitrary raw pointer; see
/// [`from_shape_ptr`](#method.from_shape_ptr) for details.
pub type RawArrayView<A, D> = ArrayBase<RawViewRepr<*const A>, D>;

/// A mutable array view without a lifetime.
///
/// This is similar to [`ArrayViewMut`] but does not carry any lifetime or
/// ownership information, and its data cannot be read/written without an
/// unsafe conversion into an [`ArrayViewMut`]. The relationship between
/// `RawArrayViewMut` and [`ArrayViewMut`] is somewhat analogous to the
/// relationship between `*mut T` and `&mut T`, but `RawArrayViewMut` has
/// additional requirements that `*mut T` does not, such as non-nullness.
///
/// The `RawArrayViewMut<A, D>` is parameterized by `A` for the element type
/// and `D` for the dimensionality.
///
/// Raw array views have all the methods of an array (see
/// [`ArrayBase`]).
///
/// See also [`RawArrayView`].
///
/// # Warning
///
/// You can't use this type with an arbitrary raw pointer; see
/// [`from_shape_ptr`](#method.from_shape_ptr) for details.
pub type RawArrayViewMut<A, D> = ArrayBase<RawViewRepr<*mut A>, D>;

pub use data_repr::OwnedRepr;

/// ArcArray's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`ArcArray`] for the array type!*
#[derive(Debug)]
pub struct OwnedArcRepr<A>(Arc<OwnedRepr<A>>);

impl<A> Clone for OwnedArcRepr<A>
{
    fn clone(&self) -> Self
    {
        OwnedArcRepr(self.0.clone())
    }
}

/// Array pointer’s representation.
///
/// *Don’t use this type directly—use the type aliases
/// [`RawArrayView`] / [`RawArrayViewMut`] for the array type!*
#[derive(Copy, Clone)]
// This is just a marker type, to carry the mutability and element type.
pub struct RawViewRepr<A>
{
    ptr: PhantomData<A>,
}

impl<A> RawViewRepr<A>
{
    #[inline(always)]
    const fn new() -> Self
    {
        RawViewRepr { ptr: PhantomData }
    }
}

/// Array view’s representation.
///
/// *Don’t use this type directly—use the type aliases
/// [`ArrayView`] / [`ArrayViewMut`] for the array type!*
#[derive(Copy, Clone)]
// This is just a marker type, to carry the lifetime parameter.
pub struct ViewRepr<A>
{
    life: PhantomData<A>,
}

impl<A> ViewRepr<A>
{
    #[inline(always)]
    const fn new() -> Self
    {
        ViewRepr { life: PhantomData }
    }
}

/// CowArray's representation.
///
/// *Don't use this type directly—use the type alias
/// [`CowArray`] for the array type!*
pub enum CowRepr<'a, A>
{
    /// Borrowed data.
    View(ViewRepr<&'a A>),
    /// Owned data.
    Owned(OwnedRepr<A>),
}

impl<'a, A> CowRepr<'a, A>
{
    /// Returns `true` iff the data is the `View` variant.
    pub fn is_view(&self) -> bool
    {
        match self {
            CowRepr::View(_) => true,
            CowRepr::Owned(_) => false,
        }
    }

    /// Returns `true` iff the data is the `Owned` variant.
    pub fn is_owned(&self) -> bool
    {
        match self {
            CowRepr::View(_) => false,
            CowRepr::Owned(_) => true,
        }
    }
}

// NOTE: The order of modules decides in which order methods on the type ArrayBase
// (mainly mentioning that as the most relevant type) show up in the documentation.
// Consider the doc effect of ordering modules here.
mod impl_clone;

mod impl_internal_constructors;
mod impl_constructors;

mod impl_methods;
mod impl_owned_array;
mod impl_special_element_types;

/// Private Methods
impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    #[inline]
    fn broadcast_unwrap<E>(&self, dim: E) -> ArrayView<'_, A, E>
    where E: Dimension
    {
        #[cold]
        #[inline(never)]
        fn broadcast_panic<D, E>(from: &D, to: &E) -> !
        where
            D: Dimension,
            E: Dimension,
        {
            panic!(
                "ndarray: could not broadcast array from shape: {:?} to: {:?}",
                from.slice(),
                to.slice()
            )
        }

        match self.broadcast(dim.clone()) {
            Some(it) => it,
            None => broadcast_panic(&self.dim, &dim),
        }
    }

    // Broadcast to dimension `E`, without checking that the dimensions match
    // (Checked in debug assertions).
    #[inline]
    fn broadcast_assume<E>(&self, dim: E) -> ArrayView<'_, A, E>
    where E: Dimension
    {
        let dim = dim.into_dimension();
        debug_assert_eq!(self.shape(), dim.slice());
        let ptr = self.ptr;
        let mut strides = dim.clone();
        strides.slice_mut().copy_from_slice(self.strides.slice());
        unsafe { ArrayView::new(ptr, dim, strides) }
    }

    /// Remove array axis `axis` and return the result.
    fn try_remove_axis(self, axis: Axis) -> ArrayBase<S, D::Smaller>
    {
        let d = self.dim.try_remove_axis(axis);
        let s = self.strides.try_remove_axis(axis);
        // safe because new dimension, strides allow access to a subset of old data
        unsafe { self.with_strides_dim(s, d) }
    }
}

// parallel methods
#[cfg(feature = "rayon")]
pub mod parallel;

mod impl_1d;
mod impl_2d;
mod impl_dyn;

mod numeric;

pub mod linalg;

mod impl_ops;
pub use crate::impl_ops::ScalarOperand;

#[cfg(feature = "approx")]
mod array_approx;

// Array view methods
mod impl_views;

// Array raw view methods
mod impl_raw_views;

// Copy-on-write array methods
mod impl_cow;

// Arc array methods
mod impl_arc_array;

/// Returns `true` if the pointer is aligned.
pub(crate) fn is_aligned<T>(ptr: *const T) -> bool
{
    (ptr as usize) % ::std::mem::align_of::<T>() == 0
}

// Triangular constructors
mod tri;
