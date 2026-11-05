//! Defines error types.

use std::error::Error;
use std::fmt;

use pyo3::{
    conversion::IntoPyObject, exceptions::PyTypeError, Bound, Py, PyErr, PyErrArguments, PyObject,
    Python,
};

use crate::dtype::PyArrayDescr;

/// Array dimensionality should be limited by [`NPY_MAXDIMS`][NPY_MAXDIMS] which is currently 32.´
///
/// [NPY_MAXDIMS]: https://github.com/numpy/numpy/blob/4c60b3263ac50e5e72f6a909e156314fc3c9cba0/numpy/core/include/numpy/ndarraytypes.h#L40
pub(crate) const MAX_DIMENSIONALITY_ERR: &str = "unexpected dimensionality: NumPy is expected to limit arrays to 32 or fewer dimensions.\nPlease report a bug against the `rust-numpy` crate.";

pub(crate) const DIMENSIONALITY_MISMATCH_ERR: &str = "inconsistent dimensionalities: The dimensionality expected by `PyArray` does not match that given by NumPy.\nPlease report a bug against the `rust-numpy` crate.";

macro_rules! impl_pyerr {
    ($err_type:ty) => {
        impl Error for $err_type {}

        impl PyErrArguments for $err_type {
            fn arguments<'py>(self, py: Python<'py>) -> PyObject {
                self.to_string()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind()
            }
        }

        impl From<$err_type> for PyErr {
            fn from(err: $err_type) -> PyErr {
                PyTypeError::new_err(err)
            }
        }
    };
}

/// Represents that dimensionalities of the given arrays do not match.
#[derive(Debug)]
pub struct DimensionalityError {
    from: usize,
    to: usize,
}

impl DimensionalityError {
    pub(crate) fn new(from: usize, to: usize) -> Self {
        Self { from, to }
    }
}

impl fmt::Display for DimensionalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dimensionality mismatch:\n from={}, to={}",
            self.from, self.to
        )
    }
}

impl_pyerr!(DimensionalityError);

/// Represents that types of the given arrays do not match.
#[derive(Debug)]
pub struct TypeError<'py> {
    from: Bound<'py, PyArrayDescr>,
    to: Bound<'py, PyArrayDescr>,
}

impl<'py> TypeError<'py> {
    pub(crate) fn new(from: Bound<'py, PyArrayDescr>, to: Bound<'py, PyArrayDescr>) -> Self {
        Self { from, to }
    }
}

impl fmt::Display for TypeError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type mismatch:\n from={}, to={}", self.from, self.to)
    }
}

impl Error for TypeError<'_> {}

struct TypeErrorArguments {
    from: Py<PyArrayDescr>,
    to: Py<PyArrayDescr>,
}

impl PyErrArguments for TypeErrorArguments {
    fn arguments<'py>(self, py: Python<'py>) -> PyObject {
        let err = TypeError {
            from: self.from.into_bound(py),
            to: self.to.into_bound(py),
        };

        err.to_string()
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind()
    }
}

impl From<TypeError<'_>> for PyErr {
    fn from(err: TypeError<'_>) -> PyErr {
        let args = TypeErrorArguments {
            from: err.from.into(),
            to: err.to.into(),
        };

        PyTypeError::new_err(args)
    }
}

/// Represents that given `Vec` cannot be treated as an array.
#[derive(Debug)]
pub struct FromVecError {
    len: usize,
    exp_len: usize,
}

impl FromVecError {
    pub(crate) fn new(len: usize, exp_len: usize) -> Self {
        Self { len, exp_len }
    }
}

impl fmt::Display for FromVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid length: {}, but expected {}",
            self.len, self.exp_len
        )
    }
}

impl_pyerr!(FromVecError);

/// Represents that the given array is not contiguous.
#[derive(Debug)]
pub struct NotContiguousError;

impl fmt::Display for NotContiguousError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The given array is not contiguous")
    }
}

impl_pyerr!(NotContiguousError);

/// Inidcates why borrowing an array failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum BorrowError {
    /// The given array is already borrowed
    AlreadyBorrowed,
    /// The given array is not writeable
    NotWriteable,
}

impl fmt::Display for BorrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyBorrowed => write!(f, "The given array is already borrowed"),
            Self::NotWriteable => write!(f, "The given array is not writeable"),
        }
    }
}

impl_pyerr!(BorrowError);

/// An internal type used to ignore certain error conditions
///
/// This is beneficial when those errors will never reach a public API anyway
/// but dropping them will improve performance.
pub(crate) struct IgnoreError;

impl<E> From<E> for IgnoreError
where
    PyErr: From<E>,
{
    fn from(_err: E) -> Self {
        Self
    }
}
