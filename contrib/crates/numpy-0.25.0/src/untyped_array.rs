//! Safe, untyped interface for NumPy's [N-dimensional arrays][ndarray]
//!
//! [ndarray]: https://numpy.org/doc/stable/reference/arrays.ndarray.html
use std::slice;

use pyo3::{
    ffi, pyobject_native_type_named, types::PyAnyMethods, Bound, PyAny, PyTypeInfo, Python,
};

use crate::array::{PyArray, PyArrayMethods};
use crate::cold;
use crate::dtype::PyArrayDescr;
use crate::npyffi;

/// A safe, untyped wrapper for NumPy's [`ndarray`] class.
///
/// Unlike [`PyArray<T,D>`][crate::PyArray], this type does not constrain either element type `T` nor the dimensionality `D`.
/// This can be useful to inspect function arguments, but it prevents operating on the elements without further downcasts.
///
/// When both element type `T` and index type `D` are known, these values can be downcast to `PyArray<T, D>`. In addition,
/// `PyArray<T, D>` can be dereferenced to a `PyUntypedArray` and can therefore automatically access its methods.
///
/// # Example
///
/// Taking `PyUntypedArray` can be helpful to implement polymorphic entry points:
///
/// ```
/// # use pyo3::prelude::*;
/// use pyo3::exceptions::PyTypeError;
/// use numpy::{Element, PyUntypedArray, PyArray1, dtype};
/// use numpy::{PyUntypedArrayMethods, PyArrayMethods, PyArrayDescrMethods};
///
/// #[pyfunction]
/// fn entry_point(py: Python<'_>, array: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
///     fn implementation<T: Element>(array: &Bound<'_, PyArray1<T>>) -> PyResult<()> {
///         /* .. */
///
///         Ok(())
///     }
///
///     let element_type = array.dtype();
///
///     if element_type.is_equiv_to(&dtype::<f32>(py)) {
///         let array = array.downcast::<PyArray1<f32>>()?;
///
///         implementation(array)
///     } else if element_type.is_equiv_to(&dtype::<f64>(py)) {
///         let array = array.downcast::<PyArray1<f64>>()?;
///
///         implementation(array)
///     } else {
///         Err(PyTypeError::new_err(format!("Unsupported element type: {}", element_type)))
///     }
/// }
/// #
/// # Python::with_gil(|py| {
/// #   let array = PyArray1::<f64>::zeros(py, 42, false);
/// #   entry_point(py, array.as_untyped())
/// # }).unwrap();
/// ```
#[repr(transparent)]
pub struct PyUntypedArray(PyAny);

unsafe impl PyTypeInfo for PyUntypedArray {
    const NAME: &'static str = "PyUntypedArray";
    const MODULE: Option<&'static str> = Some("numpy");

    fn type_object_raw<'py>(py: Python<'py>) -> *mut ffi::PyTypeObject {
        unsafe { npyffi::PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type) }
    }

    fn is_type_of(ob: &Bound<'_, PyAny>) -> bool {
        unsafe { npyffi::PyArray_Check(ob.py(), ob.as_ptr()) != 0 }
    }
}

pyobject_native_type_named!(PyUntypedArray);

/// Implementation of functionality for [`PyUntypedArray`].
#[doc(alias = "PyUntypedArray")]
pub trait PyUntypedArrayMethods<'py>: Sealed {
    /// Returns a raw pointer to the underlying [`PyArrayObject`][npyffi::PyArrayObject].
    fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject;

    /// Returns the `dtype` of the array.
    ///
    /// See also [`ndarray.dtype`][ndarray-dtype] and [`PyArray_DTYPE`][PyArray_DTYPE].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::prelude::*;
    /// use numpy::{dtype, PyArray};
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///    let array = PyArray::from_vec(py, vec![1_i32, 2, 3]);
    ///
    ///    assert!(array.dtype().is_equiv_to(&dtype::<i32>(py)));
    /// });
    /// ```
    ///
    /// [ndarray-dtype]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dtype.html
    /// [PyArray_DTYPE]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_DTYPE
    fn dtype(&self) -> Bound<'py, PyArrayDescr>;

    /// Returns `true` if the internal data of the array is contiguous,
    /// indepedently of whether C-style/row-major or Fortran-style/column-major.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{PyArray1, PyUntypedArrayMethods};
    /// use pyo3::{types::{IntoPyDict, PyAnyMethods}, Python, ffi::c_str};
    ///
    /// # fn main() -> pyo3::PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let array = PyArray1::arange(py, 0, 10, 1);
    ///     assert!(array.is_contiguous());
    ///
    ///     let view = py
    ///         .eval(c_str!("array[::2]"), None, Some(&[("array", array)].into_py_dict(py)?))?
    ///         .downcast_into::<PyArray1<i32>>()?;
    ///     assert!(!view.is_contiguous());
    /// #   Ok(())
    /// })
    /// # }
    /// ```
    fn is_contiguous(&self) -> bool {
        unsafe {
            check_flags(
                &*self.as_array_ptr(),
                npyffi::NPY_ARRAY_C_CONTIGUOUS | npyffi::NPY_ARRAY_F_CONTIGUOUS,
            )
        }
    }

    /// Returns `true` if the internal data of the array is Fortran-style/column-major contiguous.
    fn is_fortran_contiguous(&self) -> bool {
        unsafe { check_flags(&*self.as_array_ptr(), npyffi::NPY_ARRAY_F_CONTIGUOUS) }
    }

    /// Returns `true` if the internal data of the array is C-style/row-major contiguous.
    fn is_c_contiguous(&self) -> bool {
        unsafe { check_flags(&*self.as_array_ptr(), npyffi::NPY_ARRAY_C_CONTIGUOUS) }
    }

    /// Returns the number of dimensions of the array.
    ///
    /// See also [`ndarray.ndim`][ndarray-ndim] and [`PyArray_NDIM`][PyArray_NDIM].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{PyArray3, PyUntypedArrayMethods};
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let arr = PyArray3::<f64>::zeros(py, [4, 5, 6], false);
    ///
    ///     assert_eq!(arr.ndim(), 3);
    /// });
    /// ```
    ///
    /// [ndarray-ndim]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
    /// [PyArray_NDIM]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
    #[inline]
    fn ndim(&self) -> usize {
        unsafe { (*self.as_array_ptr()).nd as usize }
    }

    /// Returns a slice indicating how many bytes to advance when iterating along each axis.
    ///
    /// See also [`ndarray.strides`][ndarray-strides] and [`PyArray_STRIDES`][PyArray_STRIDES].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{PyArray3, PyUntypedArrayMethods};
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let arr = PyArray3::<f64>::zeros(py, [4, 5, 6], false);
    ///
    ///     assert_eq!(arr.strides(), &[240, 48, 8]);
    /// });
    /// ```
    /// [ndarray-strides]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
    /// [PyArray_STRIDES]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_STRIDES
    #[inline]
    fn strides(&self) -> &[isize] {
        let n = self.ndim();
        if n == 0 {
            cold();
            return &[];
        }
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).strides;
            slice::from_raw_parts(p, n)
        }
    }

    /// Returns a slice which contains dimmensions of the array.
    ///
    /// See also [`ndarray.shape`][ndaray-shape] and [`PyArray_DIMS`][PyArray_DIMS].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{PyArray3, PyUntypedArrayMethods};
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let arr = PyArray3::<f64>::zeros(py, [4, 5, 6], false);
    ///
    ///     assert_eq!(arr.shape(), &[4, 5, 6]);
    /// });
    /// ```
    ///
    /// [ndarray-shape]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
    /// [PyArray_DIMS]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_DIMS
    #[inline]
    fn shape(&self) -> &[usize] {
        let n = self.ndim();
        if n == 0 {
            cold();
            return &[];
        }
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).dimensions as *mut usize;
            slice::from_raw_parts(p, n)
        }
    }

    /// Calculates the total number of elements in the array.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns `true` if the there are no elements in the array.
    fn is_empty(&self) -> bool {
        self.shape().iter().any(|dim| *dim == 0)
    }
}

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

fn check_flags(obj: &npyffi::PyArrayObject, flags: i32) -> bool {
    obj.flags & flags != 0
}

impl<'py> PyUntypedArrayMethods<'py> for Bound<'py, PyUntypedArray> {
    #[inline]
    fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.as_ptr().cast()
    }

    fn dtype(&self) -> Bound<'py, PyArrayDescr> {
        unsafe {
            let descr_ptr = (*self.as_array_ptr()).descr;
            Bound::from_borrowed_ptr(self.py(), descr_ptr.cast()).downcast_into_unchecked()
        }
    }
}

impl Sealed for Bound<'_, PyUntypedArray> {}

// We won't be able to provide a `Deref` impl from `Bound<'_, PyArray<T, D>>` to
// `Bound<'_, PyUntypedArray>`, so this seems to be the next best thing to do
impl<'py, T, D> PyUntypedArrayMethods<'py> for Bound<'py, PyArray<T, D>> {
    #[inline]
    fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.as_untyped().as_array_ptr()
    }

    #[inline]
    fn dtype(&self) -> Bound<'py, PyArrayDescr> {
        self.as_untyped().dtype()
    }
}

impl<T, D> Sealed for Bound<'_, PyArray<T, D>> {}
