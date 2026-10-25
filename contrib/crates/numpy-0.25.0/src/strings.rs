//! Types to support arrays of [ASCII][ascii] and [UCS4][ucs4] strings
//!
//! [ascii]: https://numpy.org/doc/stable/reference/c-api/dtype.html#c.NPY_STRING
//! [ucs4]: https://numpy.org/doc/stable/reference/c-api/dtype.html#c.NPY_UNICODE

use std::collections::hash_map::Entry;
use std::fmt;
use std::mem::size_of;
use std::os::raw::c_char;
use std::str;
use std::sync::Mutex;

use pyo3::sync::MutexExt;
use pyo3::{
    ffi::{Py_UCS1, Py_UCS4},
    Bound, Py, Python,
};
use rustc_hash::FxHashMap;

use crate::dtype::{clone_methods_impl, Element, PyArrayDescr, PyArrayDescrMethods};
use crate::npyffi::PyDataType_SET_ELSIZE;
use crate::npyffi::NPY_TYPES;

/// A newtype wrapper around [`[u8; N]`][Py_UCS1] to handle [`byte` scalars][numpy-bytes] while satisfying coherence.
///
/// Note that when creating arrays of ASCII strings without an explicit `dtype`,
/// NumPy will automatically determine the smallest possible array length at runtime.
///
/// For example,
///
/// ```python
/// array = numpy.array([b"foo", b"bar", b"foobar"])
/// ```
///
/// yields `S6` for `array.dtype`.
///
/// On the Rust side however, the length `N` of `PyFixedString<N>` must always be given
/// explicitly and as a compile-time constant. For this work reliably, the Python code
/// should set the `dtype` explicitly, e.g.
///
/// ```python
/// numpy.array([b"foo", b"bar", b"foobar"], dtype='S12')
/// ```
///
/// always matching `PyArray1<PyFixedString<12>>`.
///
/// # Example
///
/// ```rust
/// # use pyo3::Python;
/// use numpy::{PyArray1, PyUntypedArrayMethods, PyFixedString};
///
/// # Python::with_gil(|py| {
/// let array = PyArray1::<PyFixedString<3>>::from_vec(py, vec![[b'f', b'o', b'o'].into()]);
///
/// assert!(array.dtype().to_string().contains("S3"));
/// # });
/// ```
///
/// [numpy-bytes]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bytes_
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyFixedString<const N: usize>(pub [Py_UCS1; N]);

impl<const N: usize> fmt::Display for PyFixedString<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(str::from_utf8(&self.0).unwrap().trim_end_matches('\0'))
    }
}

impl<const N: usize> From<[Py_UCS1; N]> for PyFixedString<N> {
    fn from(val: [Py_UCS1; N]) -> Self {
        Self(val)
    }
}

unsafe impl<const N: usize> Element for PyFixedString<N> {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        static DTYPES: TypeDescriptors = TypeDescriptors::new();

        unsafe { DTYPES.from_size(py, NPY_TYPES::NPY_STRING, b'|' as _, size_of::<Self>()) }
    }

    clone_methods_impl!(Self);
}

/// A newtype wrapper around [`[PyUCS4; N]`][Py_UCS4] to handle [`str_` scalars][numpy-str] while satisfying coherence.
///
/// Note that when creating arrays of Unicode strings without an explicit `dtype`,
/// NumPy will automatically determine the smallest possible array length at runtime.
///
/// For example,
///
/// ```python
/// numpy.array(["foo🐍", "bar🦀", "foobar"])
/// ```
///
/// yields `U6` for `array.dtype`.
///
/// On the Rust side however, the length `N` of `PyFixedUnicode<N>` must always be given
/// explicitly and as a compile-time constant. For this work reliably, the Python code
/// should set the `dtype` explicitly, e.g.
///
/// ```python
/// numpy.array(["foo🐍", "bar🦀", "foobar"], dtype='U12')
/// ```
///
/// always matching `PyArray1<PyFixedUnicode<12>>`.
///
/// # Example
///
/// ```rust
/// # use pyo3::Python;
/// use numpy::{PyArray1, PyUntypedArrayMethods, PyFixedUnicode};
///
/// # Python::with_gil(|py| {
/// let array = PyArray1::<PyFixedUnicode<3>>::from_vec(py, vec![[b'b' as _, b'a' as _, b'r' as _].into()]);
///
/// assert!(array.dtype().to_string().contains("U3"));
/// # });
/// ```
///
/// [numpy-str]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.str_
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyFixedUnicode<const N: usize>(pub [Py_UCS4; N]);

impl<const N: usize> fmt::Display for PyFixedUnicode<N> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        for character in self.0 {
            if character == 0 {
                break;
            }

            write!(fmt, "{}", char::from_u32(character).unwrap())?;
        }

        Ok(())
    }
}

impl<const N: usize> From<[Py_UCS4; N]> for PyFixedUnicode<N> {
    fn from(val: [Py_UCS4; N]) -> Self {
        Self(val)
    }
}

unsafe impl<const N: usize> Element for PyFixedUnicode<N> {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        static DTYPES: TypeDescriptors = TypeDescriptors::new();

        unsafe { DTYPES.from_size(py, NPY_TYPES::NPY_UNICODE, b'=' as _, size_of::<Self>()) }
    }

    clone_methods_impl!(Self);
}

struct TypeDescriptors {
    dtypes: Mutex<Option<FxHashMap<usize, Py<PyArrayDescr>>>>,
}

impl TypeDescriptors {
    const fn new() -> Self {
        Self {
            dtypes: Mutex::new(None),
        }
    }

    /// `npy_type` must be either `NPY_STRING` or `NPY_UNICODE` with matching `byteorder` and `size`
    #[allow(clippy::wrong_self_convention)]
    unsafe fn from_size<'py>(
        &self,
        py: Python<'py>,
        npy_type: NPY_TYPES,
        byteorder: c_char,
        size: usize,
    ) -> Bound<'py, PyArrayDescr> {
        let mut dtypes = self
            .dtypes
            .lock_py_attached(py)
            .expect("dtype cache poisoned");

        let dtype = match dtypes.get_or_insert_with(Default::default).entry(size) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let dtype = PyArrayDescr::new_from_npy_type(py, npy_type);

                let descr = &mut *dtype.as_dtype_ptr();
                PyDataType_SET_ELSIZE(py, descr, size.try_into().unwrap());
                descr.byteorder = byteorder;

                entry.insert(dtype.into())
            }
        };

        dtype.bind(py).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_fixed_string() {
        assert_eq!(
            PyFixedString([b'f', b'o', b'o', 0, 0, 0]).to_string(),
            "foo"
        );
        assert_eq!(
            PyFixedString([b'f', b'o', b'o', b'b', b'a', b'r']).to_string(),
            "foobar"
        );
    }

    #[test]
    fn format_fixed_unicode() {
        assert_eq!(
            PyFixedUnicode([b'f' as _, b'o' as _, b'o' as _, 0, 0, 0]).to_string(),
            "foo"
        );
        assert_eq!(
            PyFixedUnicode([0x1F980, 0x1F40D, 0, 0, 0, 0]).to_string(),
            "🦀🐍"
        );
        assert_eq!(
            PyFixedUnicode([b'f' as _, b'o' as _, b'o' as _, b'b' as _, b'a' as _, b'r' as _])
                .to_string(),
            "foobar"
        );
    }
}
