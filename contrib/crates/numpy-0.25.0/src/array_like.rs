use std::marker::PhantomData;
use std::ops::Deref;

use ndarray::{Array1, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use pyo3::{
    intern,
    sync::GILOnceCell,
    types::{PyAnyMethods, PyDict},
    Bound, FromPyObject, Py, PyAny, PyResult,
};

use crate::array::PyArrayMethods;
use crate::{get_array_module, Element, IntoPyArray, PyArray, PyReadonlyArray};

pub trait Coerce: Sealed {
    const VAL: bool;
}

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

/// Marker type to indicate that the element type received via [`PyArrayLike`] must match the specified type exactly.
#[derive(Debug)]
pub struct TypeMustMatch;

impl Sealed for TypeMustMatch {}

impl Coerce for TypeMustMatch {
    const VAL: bool = false;
}

/// Marker type to indicate that the element type received via [`PyArrayLike`] can be cast to the specified type by NumPy's [`asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
#[derive(Debug)]
pub struct AllowTypeChange;

impl Sealed for AllowTypeChange {}

impl Coerce for AllowTypeChange {
    const VAL: bool = true;
}

/// Receiver for arrays or array-like types.
///
/// When building API using NumPy in Python, it is common for functions to additionally accept any array-like type such as `list[float]` as arguments.
/// `PyArrayLike` enables the same pattern in Rust extensions, i.e. by taking this type as the argument of a `#[pyfunction]`,
/// one will always get access to a [`PyReadonlyArray`] that will either reference to the NumPy array originally passed into the function
/// or a temporary one created by converting the input type into a NumPy array.
///
/// Depending on whether [`TypeMustMatch`] or [`AllowTypeChange`] is used for the `C` type parameter,
/// the element type must either match the specific type `T` exactly or will be cast to it by NumPy's [`asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// # Example
///
/// `PyArrayLike1<'py, T, TypeMustMatch>` will enable you to receive both NumPy arrays and sequences
///
/// ```rust
/// # use pyo3::prelude::*;
/// use pyo3::py_run;
/// use numpy::{get_array_module, PyArrayLike1, TypeMustMatch};
///
/// #[pyfunction]
/// fn sum_up<'py>(py: Python<'py>, array: PyArrayLike1<'py, f64, TypeMustMatch>) -> f64 {
///     array.as_array().sum()
/// }
///
/// Python::with_gil(|py| {
///     let np = get_array_module(py).unwrap();
///     let sum_up = wrap_pyfunction!(sum_up)(py).unwrap();
///
///     py_run!(py, np sum_up, r"assert sum_up(np.array([1., 2., 3.])) == 6.");
///     py_run!(py, np sum_up, r"assert sum_up((1., 2., 3.)) == 6.");
/// });
/// ```
///
/// but it will not cast the element type if that is required
///
/// ```rust,should_panic
/// use pyo3::prelude::*;
/// use pyo3::py_run;
/// use numpy::{get_array_module, PyArrayLike1, TypeMustMatch};
///
/// #[pyfunction]
/// fn sum_up<'py>(py: Python<'py>, array: PyArrayLike1<'py, i32, TypeMustMatch>) -> i32 {
///     array.as_array().sum()
/// }
///
/// Python::with_gil(|py| {
///     let np = get_array_module(py).unwrap();
///     let sum_up = wrap_pyfunction!(sum_up)(py).unwrap();
///
///     py_run!(py, np sum_up, r"assert sum_up((1., 2., 3.)) == 6");
/// });
/// ```
///
/// whereas `PyArrayLike1<'py, T, AllowTypeChange>` will do even at the cost loosing precision
///
/// ```rust
/// use pyo3::prelude::*;
/// use pyo3::py_run;
/// use numpy::{get_array_module, AllowTypeChange, PyArrayLike1};
///
/// #[pyfunction]
/// fn sum_up<'py>(py: Python<'py>, array: PyArrayLike1<'py, i32, AllowTypeChange>) -> i32 {
///     array.as_array().sum()
/// }
///
/// Python::with_gil(|py| {
///     let np = get_array_module(py).unwrap();
///     let sum_up = wrap_pyfunction!(sum_up)(py).unwrap();
///
///     py_run!(py, np sum_up, r"assert sum_up((1.5, 2.5)) == 3");
/// });
/// ```
#[derive(Debug)]
#[repr(transparent)]
pub struct PyArrayLike<'py, T, D, C = TypeMustMatch>(PyReadonlyArray<'py, T, D>, PhantomData<C>)
where
    T: Element,
    D: Dimension,
    C: Coerce;

impl<'py, T, D, C> Deref for PyArrayLike<'py, T, D, C>
where
    T: Element,
    D: Dimension,
    C: Coerce,
{
    type Target = PyReadonlyArray<'py, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'py, T, D, C> FromPyObject<'py> for PyArrayLike<'py, T, D, C>
where
    T: Element + 'py,
    D: Dimension + 'py,
    C: Coerce,
    Vec<T>: FromPyObject<'py>,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = ob.downcast::<PyArray<T, D>>() {
            return Ok(Self(array.readonly(), PhantomData));
        }

        let py = ob.py();

        if matches!(D::NDIM, None | Some(1)) {
            if let Ok(vec) = ob.extract::<Vec<T>>() {
                let array = Array1::from(vec)
                    .into_dimensionality()
                    .expect("D being compatible to Ix1")
                    .into_pyarray(py)
                    .readonly();
                return Ok(Self(array, PhantomData));
            }
        }

        static AS_ARRAY: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

        let as_array = AS_ARRAY
            .get_or_try_init(py, || {
                get_array_module(py)?.getattr("asarray").map(Into::into)
            })?
            .bind(py);

        let kwargs = if C::VAL {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "dtype"), T::get_dtype(py))?;
            Some(kwargs)
        } else {
            None
        };

        let array = as_array.call((ob,), kwargs.as_ref())?.extract()?;
        Ok(Self(array, PhantomData))
    }
}

/// Receiver for zero-dimensional arrays or array-like types.
pub type PyArrayLike0<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix0, C>;

/// Receiver for one-dimensional arrays or array-like types.
pub type PyArrayLike1<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix1, C>;

/// Receiver for two-dimensional arrays or array-like types.
pub type PyArrayLike2<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix2, C>;

/// Receiver for three-dimensional arrays or array-like types.
pub type PyArrayLike3<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix3, C>;

/// Receiver for four-dimensional arrays or array-like types.
pub type PyArrayLike4<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix4, C>;

/// Receiver for five-dimensional arrays or array-like types.
pub type PyArrayLike5<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix5, C>;

/// Receiver for six-dimensional arrays or array-like types.
pub type PyArrayLike6<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, Ix6, C>;

/// Receiver for arrays or array-like types whose dimensionality is determined at runtime.
pub type PyArrayLikeDyn<'py, T, C = TypeMustMatch> = PyArrayLike<'py, T, IxDyn, C>;
