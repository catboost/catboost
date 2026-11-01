use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::ptr::null_mut;

use ndarray::{Dimension, IxDyn};
use pyo3::types::PyAnyMethods;
use pyo3::{Borrowed, Bound, FromPyObject, PyResult};

use crate::array::PyArray;
use crate::dtype::Element;
use crate::npyffi::{array::PY_ARRAY_API, NPY_CASTING, NPY_ORDER};

/// Return value of a function that can yield either an array or a scalar.
pub trait ArrayOrScalar<'py, T>: FromPyObject<'py> {}

impl<'py, T, D> ArrayOrScalar<'py, T> for Bound<'py, PyArray<T, D>>
where
    T: Element,
    D: Dimension,
{
}

impl<'py, T> ArrayOrScalar<'py, T> for T where T: Element + FromPyObject<'py> {}

/// Return the inner product of two arrays.
///
/// [NumPy's documentation][inner] has the details.
///
/// # Examples
///
/// Note that this function can either return a scalar...
///
/// ```
/// use pyo3::Python;
/// use numpy::{inner, pyarray, PyArray0};
///
/// Python::with_gil(|py| {
///     let vector = pyarray![py, 1.0, 2.0, 3.0];
///     let result: f64 = inner(&vector, &vector).unwrap();
///     assert_eq!(result, 14.0);
/// });
/// ```
///
/// ...or an array depending on its arguments.
///
/// ```
/// use pyo3::{Python, Bound};
/// use numpy::prelude::*;
/// use numpy::{inner, pyarray, PyArray0};
///
/// Python::with_gil(|py| {
///     let vector = pyarray![py, 1, 2, 3];
///     let result: Bound<'_, PyArray0<_>> = inner(&vector, &vector).unwrap();
///     assert_eq!(result.item(), 14);
/// });
/// ```
///
/// [inner]: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
pub fn inner<'py, T, DIN1, DIN2, OUT>(
    array1: &Bound<'py, PyArray<T, DIN1>>,
    array2: &Bound<'py, PyArray<T, DIN2>>,
) -> PyResult<OUT>
where
    T: Element,
    DIN1: Dimension,
    DIN2: Dimension,
    OUT: ArrayOrScalar<'py, T>,
{
    let py = array1.py();
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_InnerProduct(py, array1.as_ptr(), array2.as_ptr());
        Bound::from_owned_ptr_or_err(py, result)?
    };
    obj.extract()
}

/// Return the dot product of two arrays.
///
/// [NumPy's documentation][dot] has the details.
///
/// # Examples
///
/// Note that this function can either return an array...
///
/// ```
/// use pyo3::{Python, Bound};
/// use ndarray::array;
/// use numpy::{dot, pyarray, PyArray2, PyArrayMethods};
///
/// Python::with_gil(|py| {
///     let matrix = pyarray![py, [1, 0], [0, 1]];
///     let another_matrix = pyarray![py, [4, 1], [2, 2]];
///
///     let result: Bound<'_, PyArray2<_>> = dot(&matrix, &another_matrix).unwrap();
///
///     assert_eq!(
///         result.readonly().as_array(),
///         array![[4, 1], [2, 2]]
///     );
/// });
/// ```
///
/// ...or a scalar depending on its arguments.
///
/// ```
/// use pyo3::Python;
/// use numpy::{dot, pyarray, PyArray0};
///
/// Python::with_gil(|py| {
///     let vector = pyarray![py, 1.0, 2.0, 3.0];
///     let result: f64 = dot(&vector, &vector).unwrap();
///     assert_eq!(result, 14.0);
/// });
/// ```
///
/// [dot]: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
pub fn dot<'py, T, DIN1, DIN2, OUT>(
    array1: &Bound<'py, PyArray<T, DIN1>>,
    array2: &Bound<'py, PyArray<T, DIN2>>,
) -> PyResult<OUT>
where
    T: Element,
    DIN1: Dimension,
    DIN2: Dimension,
    OUT: ArrayOrScalar<'py, T>,
{
    let py = array1.py();
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_MatrixProduct(py, array1.as_ptr(), array2.as_ptr());
        Bound::from_owned_ptr_or_err(py, result)?
    };
    obj.extract()
}

/// Return the Einstein summation convention of given tensors.
///
/// This is usually invoked via the the [`einsum!`][crate::einsum!] macro.
pub fn einsum<'py, T, OUT>(
    subscripts: &str,
    arrays: &[Borrowed<'_, 'py, PyArray<T, IxDyn>>],
) -> PyResult<OUT>
where
    T: Element,
    OUT: ArrayOrScalar<'py, T>,
{
    let subscripts = match CStr::from_bytes_with_nul(subscripts.as_bytes()) {
        Ok(subscripts) => Cow::Borrowed(subscripts),
        Err(_) => Cow::Owned(CString::new(subscripts).unwrap()),
    };

    let py = arrays[0].py();
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_EinsteinSum(
            py,
            subscripts.as_ptr() as _,
            arrays.len() as _,
            arrays.as_ptr() as _,
            null_mut(),
            NPY_ORDER::NPY_KEEPORDER,
            NPY_CASTING::NPY_NO_CASTING,
            null_mut(),
        );
        Bound::from_owned_ptr_or_err(py, result)?
    };
    obj.extract()
}

/// Return the Einstein summation convention of given tensors.
///
/// For more about the Einstein summation convention, please refer to
/// [NumPy's documentation][einsum].
///
/// # Example
///
/// ```
/// use pyo3::{Python, Bound};
/// use ndarray::array;
/// use numpy::{einsum, pyarray, PyArray, PyArray2, PyArrayMethods};
///
/// Python::with_gil(|py| {
///     let tensor = PyArray::arange(py, 0, 2 * 3 * 4, 1).reshape([2, 3, 4]).unwrap();
///     let another_tensor = pyarray![py, [20, 30], [40, 50], [60, 70]];
///
///     let result: Bound<'_, PyArray2<_>> = einsum!("ijk,ji->ik", tensor, another_tensor).unwrap();
///
///     assert_eq!(
///         result.readonly().as_array(),
///         array![[640,  760,  880, 1000], [2560, 2710, 2860, 3010]]
///     );
/// });
/// ```
///
/// [einsum]: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
#[macro_export]
macro_rules! einsum {
    ($subscripts:literal $(,$array:ident)+ $(,)*) => {{
        let arrays = [$($array.to_dyn().as_borrowed(),)+];
        $crate::einsum(concat!($subscripts, "\0"), &arrays)
    }};
}
