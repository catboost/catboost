//! This crate provides Rust interfaces for [NumPy C APIs][c-api],
//! especially for the [ndarray][ndarray] class.
//!
//! It uses [`pyo3`] for Rust bindings to CPython, and uses
//! [`ndarray`] as the Rust matrix library.
//!
//! To resolve its dependency on NumPy, it calls `import numpy.core` internally.
//! This means that this crate should work if you can use NumPy in your Python environment,
//! e.g. after installing it by `pip install numpy`. It does not matter whether you use
//! the system environment or a dedicated virtual environment.
//!
//! Loading NumPy is done automatically and on demand. So if it is not installed, the functions
//! provided by this crate will panic instead of returning a result.
//!
#![cfg_attr(
    feature = "nalgebra",
    doc = "Integration with [`nalgebra`] is provided via an implementation of [`ToPyArray`] for [`nalgebra::Matrix`] to convert nalgebra matrices into NumPy arrays
as well as the [`PyReadonlyArray::try_as_matrix`] and [`PyReadwriteArray::try_as_matrix_mut`] methods to treat NumPy array as nalgebra matrix slices.
"
)]
//! # Example
//!
//! ```
//! use numpy::pyo3::Python;
//! use numpy::ndarray::array;
//! use numpy::{ToPyArray, PyArray, PyArrayMethods};
//!
//! Python::with_gil(|py| {
//!     let py_array = array![[1i64, 2], [3, 4]].to_pyarray(py);
//!
//!     assert_eq!(
//!         py_array.readonly().as_array(),
//!         array![[1i64, 2], [3, 4]]
//!     );
//! });
//! ```
//!
#![cfg_attr(feature = "nalgebra", doc = "```")]
#![cfg_attr(not(feature = "nalgebra"), doc = "```rust,ignore")]
//! use numpy::pyo3::Python;
//! use numpy::nalgebra::Matrix3;
//! use numpy::{pyarray, ToPyArray, PyArrayMethods};
//!
//! Python::with_gil(|py| {
//!     let py_array = pyarray![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];
//!
//!     let py_array_square;
//!
//!     {
//!         let py_array = py_array.readwrite();
//!         let mut na_matrix = py_array.as_matrix_mut();
//!
//!         na_matrix.add_scalar_mut(1);
//!
//!         py_array_square = na_matrix.pow(2).to_pyarray(py);
//!     }
//!
//!     assert_eq!(
//!         py_array.readonly().as_matrix(),
//!         Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9)
//!     );
//!
//!     assert_eq!(
//!         py_array_square.readonly().as_matrix(),
//!         Matrix3::new(30, 36, 42, 66, 81, 96, 102, 126, 150)
//!     );
//! });
#![doc = "```"]
//!
//! [c-api]: https://numpy.org/doc/stable/reference/c-api
//! [ndarray]: https://numpy.org/doc/stable/reference/arrays.ndarray.html
#![deny(missing_docs)]

pub mod array;
mod array_like;
pub mod borrow;
pub mod convert;
pub mod datetime;
mod dtype;
mod error;
pub mod npyffi;
mod slice_container;
mod strings;
mod sum_products;
mod untyped_array;

pub use ndarray;
pub use pyo3;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

pub use crate::array::{
    get_array_module, PyArray, PyArray0, PyArray0Methods, PyArray1, PyArray2, PyArray3, PyArray4,
    PyArray5, PyArray6, PyArrayDyn, PyArrayMethods,
};
pub use crate::array_like::{
    AllowTypeChange, PyArrayLike, PyArrayLike0, PyArrayLike1, PyArrayLike2, PyArrayLike3,
    PyArrayLike4, PyArrayLike5, PyArrayLike6, PyArrayLikeDyn, TypeMustMatch,
};
pub use crate::borrow::{
    PyReadonlyArray, PyReadonlyArray0, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArray4, PyReadonlyArray5, PyReadonlyArray6, PyReadonlyArrayDyn, PyReadwriteArray,
    PyReadwriteArray0, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4,
    PyReadwriteArray5, PyReadwriteArray6, PyReadwriteArrayDyn,
};
pub use crate::convert::{IntoPyArray, NpyIndex, ToNpyDims, ToPyArray};
pub use crate::dtype::{dtype, Complex32, Complex64, Element, PyArrayDescr, PyArrayDescrMethods};
pub use crate::error::{BorrowError, FromVecError, NotContiguousError};
pub use crate::npyffi::{PY_ARRAY_API, PY_UFUNC_API};
pub use crate::strings::{PyFixedString, PyFixedUnicode};
pub use crate::sum_products::{dot, einsum, inner};
pub use crate::untyped_array::{PyUntypedArray, PyUntypedArrayMethods};

pub use ndarray::{array, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// A prelude
///
/// The purpose of this module is to avoid direct imports of
/// the method traits defined by this crate via a glob import:
///
/// ```
/// # #![allow(unused_imports)]
/// use numpy::prelude::*;
/// ```
pub mod prelude {
    pub use crate::array::{PyArray0Methods, PyArrayMethods};
    pub use crate::convert::{IntoPyArray, ToPyArray};
    pub use crate::dtype::PyArrayDescrMethods;
    pub use crate::untyped_array::PyUntypedArrayMethods;
}

#[cfg(doctest)]
mod doctest {
    macro_rules! doc_comment {
        ($doc_string:expr, $mod_name:ident) => {
            #[doc = $doc_string]
            mod $mod_name {}
        };
    }
    doc_comment!(include_str!("../README.md"), readme);
}

#[cold]
#[inline(always)]
fn cold() {}

/// Create a [`PyArray`] with one, two or three dimensions.
///
/// This macro is backed by [`ndarray::array`].
///
/// # Example
///
/// ```
/// use numpy::pyo3::Python;
/// use numpy::ndarray::array;
/// use numpy::{pyarray, PyArrayMethods};
///
/// Python::with_gil(|py| {
///     let array = pyarray![py, [1, 2], [3, 4]];
///
///     assert_eq!(
///         array.readonly().as_array(),
///         array![[1, 2], [3, 4]]
///     );
/// });
#[macro_export]
macro_rules! pyarray {
    ($py: ident, $([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$([$([$($x,)*],)*],)*], $py)
    }};
    ($py: ident, $([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$([$($x,)*],)*], $py)
    }};
    ($py: ident, $($x:expr),* $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$($x,)*], $py)
    }};
}
