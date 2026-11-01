use std::mem::size_of;
use std::os::raw::{c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong, c_ushort};
use std::ptr;

#[cfg(feature = "half")]
use half::{bf16, f16};
use num_traits::{Bounded, Zero};
#[cfg(feature = "half")]
use pyo3::sync::GILOnceCell;
use pyo3::{
    conversion::IntoPyObject,
    exceptions::{PyIndexError, PyValueError},
    ffi::{self, PyTuple_Size},
    pyobject_native_type_named,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyTuple, PyType},
    Borrowed, Bound, Py, PyAny, PyObject, PyResult, PyTypeInfo, Python,
};

use crate::npyffi::{
    NpyTypes, PyArray_Descr, PyDataType_ALIGNMENT, PyDataType_ELSIZE, PyDataType_FIELDS,
    PyDataType_FLAGS, PyDataType_NAMES, PyDataType_SUBARRAY, NPY_ALIGNED_STRUCT,
    NPY_BYTEORDER_CHAR, NPY_ITEM_HASOBJECT, NPY_TYPES, PY_ARRAY_API,
};

pub use num_complex::{Complex32, Complex64};

/// Binding of [`numpy.dtype`][dtype].
///
/// # Example
///
/// ```
/// use numpy::{dtype, get_array_module, PyArrayDescr, PyArrayDescrMethods};
/// use numpy::pyo3::{types::{IntoPyDict, PyAnyMethods}, Python, ffi::c_str};
///
/// # fn main() -> pyo3::PyResult<()> {
/// Python::with_gil(|py| {
///     let locals = [("np", get_array_module(py)?)].into_py_dict(py)?;
///
///     let dt = py
///         .eval(c_str!("np.array([1, 2, 3.0]).dtype"), Some(&locals), None)?
///         .downcast_into::<PyArrayDescr>()?;
///
///     assert!(dt.is_equiv_to(&dtype::<f64>(py)));
/// #   Ok(())
/// })
/// # }
/// ```
///
/// [dtype]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
#[repr(transparent)]
pub struct PyArrayDescr(PyAny);

pyobject_native_type_named!(PyArrayDescr);

unsafe impl PyTypeInfo for PyArrayDescr {
    const NAME: &'static str = "PyArrayDescr";
    const MODULE: Option<&'static str> = Some("numpy");

    #[inline]
    fn type_object_raw<'py>(py: Python<'py>) -> *mut ffi::PyTypeObject {
        unsafe { PY_ARRAY_API.get_type_object(py, NpyTypes::PyArrayDescr_Type) }
    }
}

/// Returns the type descriptor ("dtype") for a registered type.
#[inline]
pub fn dtype<'py, T: Element>(py: Python<'py>) -> Bound<'py, PyArrayDescr> {
    T::get_dtype(py)
}

impl PyArrayDescr {
    /// Creates a new type descriptor ("dtype") object from an arbitrary object.
    ///
    /// Equivalent to invoking the constructor of [`numpy.dtype`][dtype].
    ///
    /// [dtype]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.html
    #[inline]
    pub fn new<'a, 'py, T>(py: Python<'py>, ob: T) -> PyResult<Bound<'py, Self>>
    where
        T: IntoPyObject<'py>,
    {
        fn inner<'py>(
            py: Python<'py>,
            obj: Borrowed<'_, 'py, PyAny>,
        ) -> PyResult<Bound<'py, PyArrayDescr>> {
            let mut descr: *mut PyArray_Descr = ptr::null_mut();
            unsafe {
                // None is an invalid input here and is not converted to NPY_DEFAULT_TYPE
                PY_ARRAY_API.PyArray_DescrConverter2(py, obj.as_ptr(), &mut descr);
                Bound::from_owned_ptr_or_err(py, descr.cast())
                    .map(|any| any.downcast_into_unchecked())
            }
        }

        inner(
            py,
            ob.into_pyobject(py)
                .map_err(Into::into)?
                .into_any()
                .as_borrowed(),
        )
    }

    /// Shortcut for creating a type descriptor of `object` type.
    #[inline]
    pub fn object(py: Python<'_>) -> Bound<'_, Self> {
        Self::from_npy_type(py, NPY_TYPES::NPY_OBJECT)
    }

    /// Returns the type descriptor for a registered type.
    #[inline]
    pub fn of<'py, T: Element>(py: Python<'py>) -> Bound<'py, Self> {
        T::get_dtype(py)
    }

    fn from_npy_type<'py>(py: Python<'py>, npy_type: NPY_TYPES) -> Bound<'py, Self> {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(py, npy_type as _);
            Bound::from_owned_ptr(py, descr.cast()).downcast_into_unchecked()
        }
    }

    pub(crate) fn new_from_npy_type<'py>(py: Python<'py>, npy_type: NPY_TYPES) -> Bound<'py, Self> {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrNewFromType(py, npy_type as _);
            Bound::from_owned_ptr(py, descr.cast()).downcast_into_unchecked()
        }
    }
}

/// Implementation of functionality for [`PyArrayDescr`].
#[doc(alias = "PyArrayDescr")]
pub trait PyArrayDescrMethods<'py>: Sealed {
    /// Returns `self` as `*mut PyArray_Descr`.
    fn as_dtype_ptr(&self) -> *mut PyArray_Descr;

    /// Returns `self` as `*mut PyArray_Descr` while increasing the reference count.
    ///
    /// Useful in cases where the descriptor is stolen by the API.
    fn into_dtype_ptr(self) -> *mut PyArray_Descr;

    /// Returns true if two type descriptors are equivalent.
    fn is_equiv_to(&self, other: &Self) -> bool;

    /// Returns the [array scalar][arrays-scalars] corresponding to this type descriptor.
    ///
    /// Equivalent to [`numpy.dtype.type`][dtype-type].
    ///
    /// [arrays-scalars]: https://numpy.org/doc/stable/reference/arrays.scalars.html
    /// [dtype-type]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.type.html
    fn typeobj(&self) -> Bound<'py, PyType>;

    /// Returns a unique number for each of the 21 different built-in
    /// [enumerated types][enumerated-types].
    ///
    /// These are roughly ordered from least-to-most precision.
    ///
    /// Equivalent to [`numpy.dtype.num`][dtype-num].
    ///
    /// [enumerated-types]: https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types
    /// [dtype-num]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.num.html
    fn num(&self) -> c_int {
        unsafe { &*self.as_dtype_ptr() }.type_num
    }

    /// Returns the element size of this type descriptor.
    ///
    /// Equivalent to [`numpy.dtype.itemsize`][dtype-itemsize].
    ///
    /// [dtype-itemsiize]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.itemsize.html
    fn itemsize(&self) -> usize;

    /// Returns the required alignment (bytes) of this type descriptor according to the compiler.
    ///
    /// Equivalent to [`numpy.dtype.alignment`][dtype-alignment].
    ///
    /// [dtype-alignment]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.alignment.html
    fn alignment(&self) -> usize;

    /// Returns an ASCII character indicating the byte-order of this type descriptor object.
    ///
    /// All built-in data-type objects have byteorder either `=` or `|`.
    ///
    /// Equivalent to [`numpy.dtype.byteorder`][dtype-byteorder].
    ///
    /// [dtype-byteorder]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
    fn byteorder(&self) -> u8 {
        unsafe { &*self.as_dtype_ptr() }.byteorder.max(0) as _
    }

    /// Returns a unique ASCII character for each of the 21 different built-in types.
    ///
    /// Note that structured data types are categorized as `V` (void).
    ///
    /// Equivalent to [`numpy.dtype.char`][dtype-char].
    ///
    /// [dtype-char]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.char.html
    fn char(&self) -> u8 {
        unsafe { &*self.as_dtype_ptr() }.type_.max(0) as _
    }

    /// Returns an ASCII character (one of `biufcmMOSUV`) identifying the general kind of data.
    ///
    /// Note that structured data types are categorized as `V` (void).
    ///
    /// Equivalent to [`numpy.dtype.kind`][dtype-kind].
    ///
    /// [dtype-kind]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
    fn kind(&self) -> u8 {
        unsafe { &*self.as_dtype_ptr() }.kind.max(0) as _
    }

    /// Returns bit-flags describing how this type descriptor is to be interpreted.
    ///
    /// Equivalent to [`numpy.dtype.flags`][dtype-flags].
    ///
    /// [dtype-flags]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.flags.html
    fn flags(&self) -> u64;

    /// Returns the number of dimensions if this type descriptor represents a sub-array, and zero otherwise.
    ///
    /// Equivalent to [`numpy.dtype.ndim`][dtype-ndim].
    ///
    /// [dtype-ndim]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.ndim.html
    fn ndim(&self) -> usize;

    /// Returns the type descriptor for the base element of subarrays, regardless of their dimension or shape.
    ///
    /// If the dtype is not a subarray, returns self.
    ///
    /// Equivalent to [`numpy.dtype.base`][dtype-base].
    ///
    /// [dtype-base]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.base.html
    fn base(&self) -> Bound<'py, PyArrayDescr>;

    /// Returns the shape of the sub-array.
    ///
    /// If the dtype is not a sub-array, an empty vector is returned.
    ///
    /// Equivalent to [`numpy.dtype.shape`][dtype-shape].
    ///
    /// [dtype-shape]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.shape.html
    fn shape(&self) -> Vec<usize>;

    /// Returns true if the type descriptor contains any reference-counted objects in any fields or sub-dtypes.
    ///
    /// Equivalent to [`numpy.dtype.hasobject`][dtype-hasobject].
    ///
    /// [dtype-hasobject]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.hasobject.html
    fn has_object(&self) -> bool {
        self.flags() & NPY_ITEM_HASOBJECT != 0
    }

    /// Returns true if the type descriptor is a struct which maintains field alignment.
    ///
    /// This flag is sticky, so when combining multiple structs together, it is preserved
    /// and produces new dtypes which are also aligned.
    ///
    /// Equivalent to [`numpy.dtype.isalignedstruct`][dtype-isalignedstruct].
    ///
    /// [dtype-isalignedstruct]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.isalignedstruct.html
    fn is_aligned_struct(&self) -> bool {
        self.flags() & NPY_ALIGNED_STRUCT != 0
    }

    /// Returns true if the type descriptor is a sub-array.
    ///
    /// Equivalent to PyDataType_HASSUBARRAY(self).
    fn has_subarray(&self) -> bool;

    /// Returns true if the type descriptor is a structured type.
    ///
    /// Equivalent to PyDataType_HASFIELDS(self).
    fn has_fields(&self) -> bool;

    /// Returns true if type descriptor byteorder is native, or `None` if not applicable.
    fn is_native_byteorder(&self) -> Option<bool> {
        // based on PyArray_ISNBO(self->byteorder)
        match self.byteorder() {
            b'=' => Some(true),
            b'|' => None,
            byteorder => Some(byteorder == NPY_BYTEORDER_CHAR::NPY_NATBYTE as u8),
        }
    }

    /// Returns an ordered list of field names, or `None` if there are no fields.
    ///
    /// The names are ordered according to increasing byte offset.
    ///
    /// Equivalent to [`numpy.dtype.names`][dtype-names].
    ///
    /// [dtype-names]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.names.html
    fn names(&self) -> Option<Vec<String>>;

    /// Returns the type descriptor and offset of the field with the given name.
    ///
    /// This method will return an error if this type descriptor is not structured,
    /// or if it does not contain a field with a given name.
    ///
    /// The list of all names can be found via [`PyArrayDescr::names`].
    ///
    /// Equivalent to retrieving a single item from [`numpy.dtype.fields`][dtype-fields].
    ///
    /// [dtype-fields]: https://numpy.org/doc/stable/reference/generated/numpy.dtype.fields.html
    fn get_field(&self, name: &str) -> PyResult<(Bound<'py, PyArrayDescr>, usize)>;
}

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

impl<'py> PyArrayDescrMethods<'py> for Bound<'py, PyArrayDescr> {
    fn as_dtype_ptr(&self) -> *mut PyArray_Descr {
        self.as_ptr() as _
    }

    fn into_dtype_ptr(self) -> *mut PyArray_Descr {
        self.into_ptr() as _
    }

    fn is_equiv_to(&self, other: &Self) -> bool {
        let self_ptr = self.as_dtype_ptr();
        let other_ptr = other.as_dtype_ptr();

        unsafe {
            self_ptr == other_ptr
                || PY_ARRAY_API.PyArray_EquivTypes(self.py(), self_ptr, other_ptr) != 0
        }
    }

    fn typeobj(&self) -> Bound<'py, PyType> {
        let dtype_type_ptr = unsafe { &*self.as_dtype_ptr() }.typeobj;
        unsafe { PyType::from_borrowed_type_ptr(self.py(), dtype_type_ptr) }
    }

    fn itemsize(&self) -> usize {
        unsafe { PyDataType_ELSIZE(self.py(), self.as_dtype_ptr()).max(0) as _ }
    }

    fn alignment(&self) -> usize {
        unsafe { PyDataType_ALIGNMENT(self.py(), self.as_dtype_ptr()).max(0) as _ }
    }

    fn flags(&self) -> u64 {
        unsafe { PyDataType_FLAGS(self.py(), self.as_dtype_ptr()) as _ }
    }

    fn ndim(&self) -> usize {
        let subarray = unsafe { PyDataType_SUBARRAY(self.py(), self.as_dtype_ptr()).as_ref() };
        match subarray {
            None => 0,
            Some(subarray) => unsafe { PyTuple_Size(subarray.shape) }.max(0) as _,
        }
    }

    fn base(&self) -> Bound<'py, PyArrayDescr> {
        let subarray = unsafe { PyDataType_SUBARRAY(self.py(), self.as_dtype_ptr()).as_ref() };
        match subarray {
            None => self.clone(),
            Some(subarray) => unsafe {
                Bound::from_borrowed_ptr(self.py(), subarray.base.cast()).downcast_into_unchecked()
            },
        }
    }

    fn shape(&self) -> Vec<usize> {
        let subarray = unsafe { PyDataType_SUBARRAY(self.py(), self.as_dtype_ptr()).as_ref() };
        match subarray {
            None => Vec::new(),
            Some(subarray) => {
                // NumPy guarantees that shape is a tuple of non-negative integers so this should never panic.
                let shape = unsafe { Borrowed::from_ptr(self.py(), subarray.shape) };
                shape.extract().unwrap()
            }
        }
    }

    fn has_subarray(&self) -> bool {
        unsafe { !PyDataType_SUBARRAY(self.py(), self.as_dtype_ptr()).is_null() }
    }

    fn has_fields(&self) -> bool {
        unsafe { !PyDataType_NAMES(self.py(), self.as_dtype_ptr()).is_null() }
    }

    fn names(&self) -> Option<Vec<String>> {
        if !self.has_fields() {
            return None;
        }
        let names = unsafe {
            Borrowed::from_ptr(self.py(), PyDataType_NAMES(self.py(), self.as_dtype_ptr()))
        };
        names.extract().ok()
    }

    fn get_field(&self, name: &str) -> PyResult<(Bound<'py, PyArrayDescr>, usize)> {
        if !self.has_fields() {
            return Err(PyValueError::new_err(
                "cannot get field information: type descriptor has no fields",
            ));
        }
        let dict = unsafe {
            Borrowed::from_ptr(self.py(), PyDataType_FIELDS(self.py(), self.as_dtype_ptr()))
        };
        let dict = unsafe { dict.downcast_unchecked::<PyDict>() };
        // NumPy guarantees that fields are tuples of proper size and type, so this should never panic.
        let tuple = dict
            .get_item(name)?
            .ok_or_else(|| PyIndexError::new_err(name.to_owned()))?
            .downcast_into::<PyTuple>()
            .unwrap();
        // Note that we cannot just extract the entire tuple since the third element can be a title.
        let dtype = tuple
            .get_item(0)
            .unwrap()
            .downcast_into::<PyArrayDescr>()
            .unwrap();
        let offset = tuple.get_item(1).unwrap().extract().unwrap();
        Ok((dtype, offset))
    }
}

impl Sealed for Bound<'_, PyArrayDescr> {}

/// Represents that a type can be an element of `PyArray`.
///
/// Currently, only integer/float/complex/object types are supported. The [NumPy documentation][enumerated-types]
/// list the other built-in types which we are not yet implemented.
///
/// Note that NumPy's integer types like `numpy.int_` and `numpy.uint` are based on C's integer hierarchy
/// which implies that their widths change depending on the platform's [data model][data-models].
/// For example, `numpy.int_` matches C's `long` which is 32 bits wide on Windows (using the LLP64 data model)
/// but 64 bits wide on Linux (using the LP64 data model).
///
/// In contrast, Rust's [`isize`] and [`usize`] types are defined to have the same width as a pointer
/// and are therefore always 64 bits wide on 64-bit platforms. If you want to match NumPy's behaviour,
/// consider using the [`c_long`][std::ffi::c_long] and [`c_ulong`][std::ffi::c_ulong] type aliases.
///
/// # Safety
///
/// A type `T` that implements this trait should be safe when managed by a NumPy
/// array, thus implementing this trait is marked unsafe. Data types that don't
/// contain Python objects (i.e., either the object type itself or record types
/// containing object-type fields) are assumed to be trivially copyable, which
/// is reflected in the `IS_COPY` flag. Furthermore, it is assumed that for
/// the object type the elements are pointers into the Python heap and that the
/// corresponding `Clone` implemenation will never panic as it only increases
/// the reference count.
///
/// # Custom element types
///
/// Note that we cannot safely store `Py<T>` where `T: PyClass`, because the type information would be
/// eliminated in the resulting NumPy array.
/// In other words, objects are always treated as `Py<PyAny>` (a.k.a. `PyObject`) by Python code,
/// and only `Py<PyAny>` can be stored in a type safe manner.
///
/// You can however create [`Array<Py<T>, D>`][ndarray::Array] and turn that into a NumPy array
/// safely and efficiently using [`from_owned_object_array`][crate::PyArray::from_owned_object_array].
///
/// [enumerated-types]: https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types
/// [data-models]: https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
pub unsafe trait Element: Sized + Send + Sync {
    /// Flag that indicates whether this type is trivially copyable.
    ///
    /// It should be set to true for all trivially copyable types (like scalar types
    /// and record/array types only containing trivially copyable fields and elements).
    ///
    /// This flag should *always* be set to `false` for object types or record types
    /// that contain object-type fields.
    const IS_COPY: bool;

    /// Returns the associated type descriptor ("dtype") for the given element type.
    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr>;

    /// Create a clone of the value while the GIL is guaranteed to be held.
    fn clone_ref(&self, py: Python<'_>) -> Self;

    /// Create an owned copy of the slice while the GIL is guaranteed to be held.
    ///
    /// Some types may provide implementations of this method that are more efficient
    /// than simply mapping the `py_clone` method to each element in the slice.
    #[inline]
    fn vec_from_slice(py: Python<'_>, slc: &[Self]) -> Vec<Self> {
        slc.iter().map(|elem| elem.clone_ref(py)).collect()
    }

    /// Create an owned copy of the array while the GIL is guaranteed to be held.
    ///
    /// Some types may provide implementations of this method that are more efficient
    /// than simply mapping the `py_clone` method to each element in the view.
    #[inline]
    fn array_from_view<D>(
        py: Python<'_>,
        view: ::ndarray::ArrayView<'_, Self, D>,
    ) -> ::ndarray::Array<Self, D>
    where
        D: ::ndarray::Dimension,
    {
        view.map(|elem| elem.clone_ref(py))
    }
}

fn npy_int_type_lookup<T, T0, T1, T2>(npy_types: [NPY_TYPES; 3]) -> NPY_TYPES {
    // `npy_common.h` defines the integer aliases. In order, it checks:
    // NPY_BITSOF_LONG, NPY_BITSOF_LONGLONG, NPY_BITSOF_INT, NPY_BITSOF_SHORT, NPY_BITSOF_CHAR
    // and assigns the alias to the first matching size, so we should check in this order.
    match size_of::<T>() {
        x if x == size_of::<T0>() => npy_types[0],
        x if x == size_of::<T1>() => npy_types[1],
        x if x == size_of::<T2>() => npy_types[2],
        _ => panic!("Unable to match integer type descriptor: {:?}", npy_types),
    }
}

fn npy_int_type<T: Bounded + Zero + Sized + PartialEq>() -> NPY_TYPES {
    let is_unsigned = T::min_value() == T::zero();
    let bit_width = 8 * size_of::<T>();

    match (is_unsigned, bit_width) {
        (false, 8) => NPY_TYPES::NPY_BYTE,
        (false, 16) => NPY_TYPES::NPY_SHORT,
        (false, 32) => npy_int_type_lookup::<i32, c_long, c_int, c_short>([
            NPY_TYPES::NPY_LONG,
            NPY_TYPES::NPY_INT,
            NPY_TYPES::NPY_SHORT,
        ]),
        (false, 64) => npy_int_type_lookup::<i64, c_long, c_longlong, c_int>([
            NPY_TYPES::NPY_LONG,
            NPY_TYPES::NPY_LONGLONG,
            NPY_TYPES::NPY_INT,
        ]),
        (true, 8) => NPY_TYPES::NPY_UBYTE,
        (true, 16) => NPY_TYPES::NPY_USHORT,
        (true, 32) => npy_int_type_lookup::<u32, c_ulong, c_uint, c_ushort>([
            NPY_TYPES::NPY_ULONG,
            NPY_TYPES::NPY_UINT,
            NPY_TYPES::NPY_USHORT,
        ]),
        (true, 64) => npy_int_type_lookup::<u64, c_ulong, c_ulonglong, c_uint>([
            NPY_TYPES::NPY_ULONG,
            NPY_TYPES::NPY_ULONGLONG,
            NPY_TYPES::NPY_UINT,
        ]),
        _ => unreachable!(),
    }
}

// Invoke within the `Element` impl for a `Clone` type to provide an efficient
// implementation of the cloning methods
macro_rules! clone_methods_impl {
    ($Self:ty) => {
        #[inline]
        fn clone_ref(&self, _py: ::pyo3::Python<'_>) -> $Self {
            ::std::clone::Clone::clone(self)
        }

        #[inline]
        fn vec_from_slice(_py: ::pyo3::Python<'_>, slc: &[$Self]) -> Vec<$Self> {
            ::std::borrow::ToOwned::to_owned(slc)
        }

        #[inline]
        fn array_from_view<D>(
            _py: ::pyo3::Python<'_>,
            view: ::ndarray::ArrayView<'_, $Self, D>,
        ) -> ::ndarray::Array<$Self, D>
        where
            D: ::ndarray::Dimension,
        {
            ::ndarray::ArrayView::to_owned(&view)
        }
    };
}
pub(crate) use clone_methods_impl;
use pyo3::BoundObject;

macro_rules! impl_element_scalar {
    (@impl: $ty:ty, $npy_type:expr $(,#[$meta:meta])*) => {
        $(#[$meta])*
        unsafe impl Element for $ty {
            const IS_COPY: bool = true;

            fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
                PyArrayDescr::from_npy_type(py, $npy_type)
            }

            clone_methods_impl!($ty);
        }
    };
    ($ty:ty => $npy_type:ident $(,#[$meta:meta])*) => {
        impl_element_scalar!(@impl: $ty, NPY_TYPES::$npy_type $(,#[$meta])*);
    };
    ($($tys:ty),+) => {
        $(impl_element_scalar!(@impl: $tys, npy_int_type::<$tys>());)+
    };
}

impl_element_scalar!(bool => NPY_BOOL);

impl_element_scalar!(i8, i16, i32, i64);
impl_element_scalar!(u8, u16, u32, u64);

impl_element_scalar!(f32 => NPY_FLOAT);
impl_element_scalar!(f64 => NPY_DOUBLE);

#[cfg(feature = "half")]
impl_element_scalar!(f16 => NPY_HALF);

#[cfg(feature = "half")]
unsafe impl Element for bf16 {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        static DTYPE: GILOnceCell<Py<PyArrayDescr>> = GILOnceCell::new();

        DTYPE
            .get_or_init(py, || {
                PyArrayDescr::new(py, "bfloat16").expect("A package which provides a `bfloat16` data type for NumPy is required to use the `half::bf16` element type.").unbind()
            })
            .clone_ref(py)
            .into_bound(py)
    }

    clone_methods_impl!(Self);
}

impl_element_scalar!(Complex32 => NPY_CFLOAT,
    #[doc = "Complex type with `f32` components which maps to `numpy.csingle` (`numpy.complex64`)."]);
impl_element_scalar!(Complex64 => NPY_CDOUBLE,
    #[doc = "Complex type with `f64` components which maps to `numpy.cdouble` (`numpy.complex128`)."]);

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl_element_scalar!(usize, isize);

unsafe impl Element for PyObject {
    const IS_COPY: bool = false;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object(py)
    }

    #[inline]
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Py::clone_ref(self, py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::types::PyString;
    use pyo3::{py_run, types::PyTypeMethods};

    use crate::npyffi::{is_numpy_2, NPY_NEEDS_PYAPI};

    #[test]
    fn test_dtype_new() {
        Python::with_gil(|py| {
            assert!(PyArrayDescr::new(py, "float64")
                .unwrap()
                .is(dtype::<f64>(py)));

            let dt = PyArrayDescr::new(py, [("a", "O"), ("b", "?")].as_ref()).unwrap();
            assert_eq!(dt.names(), Some(vec!["a".to_owned(), "b".to_owned()]));
            assert!(dt.has_object());
            assert!(dt.get_field("a").unwrap().0.is(dtype::<PyObject>(py)));
            assert!(dt.get_field("b").unwrap().0.is(dtype::<bool>(py)));

            assert!(PyArrayDescr::new(py, 123_usize).is_err());
        });
    }

    #[test]
    fn test_dtype_names() {
        fn type_name<T: Element>(py: Python<'_>) -> Bound<'_, PyString> {
            dtype::<T>(py).typeobj().qualname().unwrap()
        }
        Python::with_gil(|py| {
            if is_numpy_2(py) {
                assert_eq!(type_name::<bool>(py), "bool");
            } else {
                assert_eq!(type_name::<bool>(py), "bool_");
            }

            assert_eq!(type_name::<i8>(py), "int8");
            assert_eq!(type_name::<i16>(py), "int16");
            assert_eq!(type_name::<i32>(py), "int32");
            assert_eq!(type_name::<i64>(py), "int64");
            assert_eq!(type_name::<u8>(py), "uint8");
            assert_eq!(type_name::<u16>(py), "uint16");
            assert_eq!(type_name::<u32>(py), "uint32");
            assert_eq!(type_name::<u64>(py), "uint64");
            assert_eq!(type_name::<f32>(py), "float32");
            assert_eq!(type_name::<f64>(py), "float64");

            assert_eq!(type_name::<Complex32>(py), "complex64");
            assert_eq!(type_name::<Complex64>(py), "complex128");

            #[cfg(target_pointer_width = "32")]
            {
                assert_eq!(type_name::<usize>(py), "uint32");
                assert_eq!(type_name::<isize>(py), "int32");
            }

            #[cfg(target_pointer_width = "64")]
            {
                assert_eq!(type_name::<usize>(py), "uint64");
                assert_eq!(type_name::<isize>(py), "int64");
            }
        });
    }

    #[test]
    fn test_dtype_methods_scalar() {
        Python::with_gil(|py| {
            let dt = dtype::<f64>(py);

            assert_eq!(dt.num(), NPY_TYPES::NPY_DOUBLE as c_int);
            assert_eq!(dt.flags(), 0);
            assert_eq!(dt.typeobj().qualname().unwrap(), "float64");
            assert_eq!(dt.char(), b'd');
            assert_eq!(dt.kind(), b'f');
            assert_eq!(dt.byteorder(), b'=');
            assert_eq!(dt.is_native_byteorder(), Some(true));
            assert_eq!(dt.itemsize(), 8);
            assert_eq!(dt.alignment(), 8);
            assert!(!dt.has_object());
            assert!(dt.names().is_none());
            assert!(!dt.has_fields());
            assert!(!dt.is_aligned_struct());
            assert!(!dt.has_subarray());
            assert!(dt.base().is_equiv_to(&dt));
            assert_eq!(dt.ndim(), 0);
            assert_eq!(dt.shape(), Vec::<usize>::new());
        });
    }

    #[test]
    fn test_dtype_methods_subarray() {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            py_run!(
                py,
                *locals,
                "dtype = __import__('numpy').dtype(('f8', (2, 3)))"
            );
            let dt = locals
                .get_item("dtype")
                .unwrap()
                .unwrap()
                .downcast_into::<PyArrayDescr>()
                .unwrap();

            assert_eq!(dt.num(), NPY_TYPES::NPY_VOID as c_int);
            assert_eq!(dt.flags(), 0);
            assert_eq!(dt.typeobj().qualname().unwrap(), "void");
            assert_eq!(dt.char(), b'V');
            assert_eq!(dt.kind(), b'V');
            assert_eq!(dt.byteorder(), b'|');
            assert_eq!(dt.is_native_byteorder(), None);
            assert_eq!(dt.itemsize(), 48);
            assert_eq!(dt.alignment(), 8);
            assert!(!dt.has_object());
            assert!(dt.names().is_none());
            assert!(!dt.has_fields());
            assert!(!dt.is_aligned_struct());
            assert!(dt.has_subarray());
            assert_eq!(dt.ndim(), 2);
            assert_eq!(dt.shape(), vec![2, 3]);
            assert!(dt.base().is_equiv_to(&dtype::<f64>(py)));
        });
    }

    #[test]
    fn test_dtype_methods_record() {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            py_run!(
                py,
                *locals,
                "dtype = __import__('numpy').dtype([('x', 'u1'), ('y', 'f8'), ('z', 'O')], align=True)"
            );
            let dt = locals
                .get_item("dtype")
                .unwrap()
                .unwrap()
                .downcast_into::<PyArrayDescr>()
                .unwrap();

            assert_eq!(dt.num(), NPY_TYPES::NPY_VOID as c_int);
            assert_ne!(dt.flags() & NPY_ITEM_HASOBJECT, 0);
            assert_ne!(dt.flags() & NPY_NEEDS_PYAPI, 0);
            assert_ne!(dt.flags() & NPY_ALIGNED_STRUCT, 0);
            assert_eq!(dt.typeobj().qualname().unwrap(), "void");
            assert_eq!(dt.char(), b'V');
            assert_eq!(dt.kind(), b'V');
            assert_eq!(dt.byteorder(), b'|');
            assert_eq!(dt.is_native_byteorder(), None);
            assert_eq!(dt.itemsize(), 24);
            assert_eq!(dt.alignment(), 8);
            assert!(dt.has_object());
            assert_eq!(
                dt.names(),
                Some(vec!["x".to_owned(), "y".to_owned(), "z".to_owned()])
            );
            assert!(dt.has_fields());
            assert!(dt.is_aligned_struct());
            assert!(!dt.has_subarray());
            assert_eq!(dt.ndim(), 0);
            assert_eq!(dt.shape(), Vec::<usize>::new());
            assert!(dt.base().is_equiv_to(&dt));
            let x = dt.get_field("x").unwrap();
            assert!(x.0.is_equiv_to(&dtype::<u8>(py)));
            assert_eq!(x.1, 0);
            let y = dt.get_field("y").unwrap();
            assert!(y.0.is_equiv_to(&dtype::<f64>(py)));
            assert_eq!(y.1, 8);
            let z = dt.get_field("z").unwrap();
            assert!(z.0.is_equiv_to(&dtype::<PyObject>(py)));
            assert_eq!(z.1, 16);
        });
    }
}
