//! Types to safely create references into NumPy arrays
//!
//! It is assumed that unchecked code - which includes unsafe Rust and Python - is validated by its author
//! which together with the dynamic borrow checking performed by this crate ensures that
//! safe Rust code cannot cause undefined behaviour by creating references into NumPy arrays.
//!
//! With these borrows established, [references to individual elements][PyReadonlyArray::get] or [reference-based views of whole array][PyReadonlyArray::as_array]
//! can be created safely. These are then the starting point for algorithms iteraing over and operating on the elements of the array.
//!
//! # Examples
//!
//! The first example shows that dynamic borrow checking works to constrain
//! both what safe Rust code can invoke and how it is invoked.
//!
//! ```rust
//! # use std::panic::{catch_unwind, AssertUnwindSafe};
//! #
//! use numpy::{PyArray1, PyArrayMethods, npyffi::flags};
//! use ndarray::Zip;
//! use pyo3::{Python, Bound};
//!
//! fn add(x: &Bound<'_, PyArray1<f64>>, y: &Bound<'_, PyArray1<f64>>, z: &Bound<'_, PyArray1<f64>>) {
//!     let x1 = x.readonly();
//!     let y1 = y.readonly();
//!     let mut z1 = z.readwrite();
//!
//!     let x2 = x1.as_array();
//!     let y2 = y1.as_array();
//!     let z2 = z1.as_array_mut();
//!
//!     Zip::from(x2)
//!         .and(y2)
//!         .and(z2)
//!         .for_each(|x3, y3, z3| *z3 = x3 + y3);
//!
//!     // Will fail at runtime due to conflict with `x1`.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _x4 = x.readwrite();
//!     }));
//!     assert!(res.is_err());
//! }
//!
//! Python::with_gil(|py| {
//!     let x = PyArray1::<f64>::zeros(py, 42, false);
//!     let y = PyArray1::<f64>::zeros(py, 42, false);
//!     let z = PyArray1::<f64>::zeros(py, 42, false);
//!
//!     // Will work as the three arrays are distinct.
//!     add(&x, &y, &z);
//!
//!     // Will work as `x1` and `y1` are compatible borrows.
//!     add(&x, &x, &z);
//!
//!     // Will fail at runtime due to conflict between `y1` and `z1`.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         add(&x, &y, &y);
//!     }));
//!     assert!(res.is_err());
//! });
//! ```
//!
//! The second example shows that non-overlapping and interleaved views are also supported.
//!
//! ```rust
//! use numpy::{PyArray1, PyArrayMethods};
//! use pyo3::{types::{IntoPyDict, PyAnyMethods}, Python, ffi::c_str};
//!
//! # fn main() -> pyo3::PyResult<()> {
//! Python::with_gil(|py| {
//!     let array = PyArray1::arange(py, 0.0, 10.0, 1.0);
//!     let locals = [("array", array)].into_py_dict(py)?;
//!
//!     let view1 = py.eval(c_str!("array[:5]"), None, Some(&locals))?.downcast_into::<PyArray1<f64>>()?;
//!     let view2 = py.eval(c_str!("array[5:]"), None, Some(&locals))?.downcast_into::<PyArray1<f64>>()?;
//!     let view3 = py.eval(c_str!("array[::2]"), None, Some(&locals))?.downcast_into::<PyArray1<f64>>()?;
//!     let view4 = py.eval(c_str!("array[1::2]"), None, Some(&locals))?.downcast_into::<PyArray1<f64>>()?;
//!
//!     {
//!         let _view1 = view1.readwrite();
//!         let _view2 = view2.readwrite();
//!     }
//!
//!     {
//!         let _view3 = view3.readwrite();
//!         let _view4 = view4.readwrite();
//!     }
//! #   Ok(())
//! })
//! # }
//! ```
//!
//! The third example shows that some views are incorrectly rejected since the borrows are over-approximated.
//!
//! ```rust
//! # use std::panic::{catch_unwind, AssertUnwindSafe};
//! #
//! use numpy::{PyArray2, PyArrayMethods};
//! use pyo3::{types::{IntoPyDict, PyAnyMethods}, Python, ffi::c_str};
//!
//! # fn main() -> pyo3::PyResult<()> {
//! Python::with_gil(|py| {
//!     let array = PyArray2::<f64>::zeros(py, (10, 10), false);
//!     let locals = [("array", array)].into_py_dict(py)?;
//!
//!     let view1 = py.eval(c_str!("array[:, ::3]"), None, Some(&locals))?.downcast_into::<PyArray2<f64>>()?;
//!     let view2 = py.eval(c_str!("array[:, 1::3]"), None, Some(&locals))?.downcast_into::<PyArray2<f64>>()?;
//!
//!     // A false conflict as the views do not actually share any elements.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _view1 = view1.readwrite();
//!         let _view2 = view2.readwrite();
//!     }));
//!     assert!(res.is_err());
//! #   Ok(())
//! })
//! # }
//! ```
//!
//! # Rationale
//!
//! Rust references require aliasing discipline to be maintained, i.e. there must always
//! exist only a single mutable (aka exclusive) reference or multiple immutable (aka shared) references
//! for each object, otherwise the program contains undefined behaviour.
//!
//! The aim of this module is to ensure that safe Rust code is unable to violate these requirements on its own.
//! We cannot prevent unchecked code - this includes unsafe Rust, Python or other native code like C or Fortran -
//! from violating them. Therefore the responsibility to avoid this lies with the author of that code instead of the compiler.
//! However, assuming that the unchecked code is correct, we can ensure that safe Rust is unable to introduce mistakes
//! into an otherwise correct program by dynamically checking which arrays are currently borrowed and in what manner.
//!
//! This means that we follow the [base object chain][base] of each array to the original allocation backing it and
//! track which parts of that allocation are covered by the array and thereby ensure that only a single read-write array
//! or multiple read-only arrays overlapping with that region are borrowed at any time.
//!
//! In contrast to Rust references, the mere existence of Python references or raw pointers is not an issue
//! because these values are not assumed to follow aliasing discipline by the Rust compiler.
//!
//! This cannot prevent unchecked code from concurrently modifying an array via callbacks or using multiple threads,
//! but that would lead to incorrect results even if the code that is interfered with is implemented in another language
//! which does not require aliasing discipline.
//!
//! Concerning multi-threading in particular: While the GIL needs to be acquired to create borrows, they are not bound to the GIL
//! and will stay active after the GIL is released, for example by calling [`allow_threads`][pyo3::Python::allow_threads].
//! Borrows also do not provide synchronization, i.e. multiple threads borrowing the same array will lead to runtime panics,
//! it will not block those threads until already active borrows are released.
//!
//! In summary, this crate takes the position that all unchecked code - unsafe Rust, Python, C, Fortran, etc. - must be checked for correctness by its author.
//! Safe Rust code can then rely on this correctness, but should not be able to introduce memory safety issues on its own. Additionally, dynamic borrow checking
//! can catch _some_ mistakes introduced by unchecked code, e.g. Python calling a function with the same array as an input and as an output argument.
//!
//! # Limitations
//!
//! Note that the current implementation of this is an over-approximation: It will consider borrows
//! potentially conflicting if the initial arrays have the same object at the end of their [base object chain][base].
//! Then, multiple conditions which are sufficient but not necessary to show the absence of conflicts are checked.
//!
//! While this is sufficient to handle common situations like slicing an array with a non-unit step size which divides
//! the dimension along that axis, there are also cases which it does not handle. For example, if the step size does
//! not divide the dimension along the sliced axis. Under such conditions, borrows are rejected even though the arrays
//! do not actually share any elements.
//!
//! This does limit the set of programs that can be written using safe Rust in way similar to rustc itself
//! which ensures that all accepted programs are memory safe but does not necessarily accept all memory safe programs.
//! However, the unsafe method [`PyArray::as_array_mut`] can be used as an escape hatch.
//! More involved cases like the example from above may be supported in the future.
//!
//! [base]: https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.NPY_AO.base

mod shared;

use std::any::type_name;
use std::fmt;
use std::ops::Deref;

use ndarray::{
    ArrayView, ArrayViewMut, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
};
use pyo3::{types::PyAnyMethods, Bound, FromPyObject, PyAny, PyResult};

use crate::array::{PyArray, PyArrayMethods};
use crate::convert::NpyIndex;
use crate::dtype::Element;
use crate::error::{BorrowError, NotContiguousError};
use crate::npyffi::flags;
use crate::untyped_array::PyUntypedArrayMethods;

use shared::{acquire, acquire_mut, release, release_mut};

/// Read-only borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadwriteArray`],
/// i.e. that only shared references into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
#[repr(transparent)]
pub struct PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    array: Bound<'py, PyArray<T, D>>,
}

/// Read-only borrow of a zero-dimensional array.
pub type PyReadonlyArray0<'py, T> = PyReadonlyArray<'py, T, Ix0>;

/// Read-only borrow of a one-dimensional array.
pub type PyReadonlyArray1<'py, T> = PyReadonlyArray<'py, T, Ix1>;

/// Read-only borrow of a two-dimensional array.
pub type PyReadonlyArray2<'py, T> = PyReadonlyArray<'py, T, Ix2>;

/// Read-only borrow of a three-dimensional array.
pub type PyReadonlyArray3<'py, T> = PyReadonlyArray<'py, T, Ix3>;

/// Read-only borrow of a four-dimensional array.
pub type PyReadonlyArray4<'py, T> = PyReadonlyArray<'py, T, Ix4>;

/// Read-only borrow of a five-dimensional array.
pub type PyReadonlyArray5<'py, T> = PyReadonlyArray<'py, T, Ix5>;

/// Read-only borrow of a six-dimensional array.
pub type PyReadonlyArray6<'py, T> = PyReadonlyArray<'py, T, Ix6>;

/// Read-only borrow of an array whose dimensionality is determined at runtime.
pub type PyReadonlyArrayDyn<'py, T> = PyReadonlyArray<'py, T, IxDyn>;

impl<'py, T, D> Deref for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    type Target = Bound<'py, PyArray<T, D>>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for PyReadonlyArray<'py, T, D> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let array = obj.downcast::<PyArray<T, D>>()?;
        Ok(array.readonly())
    }
}

impl<'py, T, D> PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    pub(crate) fn try_new(array: Bound<'py, PyArray<T, D>>) -> Result<Self, BorrowError> {
        acquire(array.py(), array.as_array_ptr())?;

        Ok(Self { array })
    }

    /// Provides an immutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array(&self) -> ArrayView<'_, T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_array() }
    }

    /// Provide an immutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_slice() }
    }

    /// Provide an immutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<&T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.array.get(index) }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N, D> PyReadonlyArray<'py, N, D>
where
    N: nalgebra::Scalar + Element,
    D: Dimension,
{
    /// Try to convert this array into a [`nalgebra::MatrixView`] using the given shape and strides.
    ///
    /// Note that nalgebra's types default to Fortan/column-major standard strides whereas NumPy creates C/row-major strides by default.
    /// Furthermore, array views created by slicing into existing arrays will often have non-standard strides.
    ///
    /// If you do not fully control the memory layout of a given array, e.g. at your API entry points,
    /// it can be useful to opt into nalgebra's support for [dynamic strides][nalgebra::Dyn], for example
    ///
    /// ```rust
    /// # use pyo3::prelude::*;
    /// use pyo3::{py_run, ffi::c_str};
    /// use numpy::{get_array_module, PyReadonlyArray2};
    /// use nalgebra::{MatrixView, Const, Dyn};
    ///
    /// #[pyfunction]
    /// fn sum_standard_layout<'py>(py: Python<'py>, array: PyReadonlyArray2<'py, f64>) -> Option<f64> {
    ///     let matrix: Option<MatrixView<f64, Const<2>, Const<2>>> = array.try_as_matrix();
    ///     matrix.map(|matrix| matrix.sum())
    /// }
    ///
    /// #[pyfunction]
    /// fn sum_dynamic_strides<'py>(py: Python<'py>, array: PyReadonlyArray2<'py, f64>) -> Option<f64> {
    ///     let matrix: Option<MatrixView<f64, Const<2>, Const<2>, Dyn, Dyn>> = array.try_as_matrix();
    ///     matrix.map(|matrix| matrix.sum())
    /// }
    ///
    /// # fn main() -> pyo3::PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let np = py.eval(c_str!("__import__('numpy')"), None, None)?;
    ///     let sum_standard_layout = wrap_pyfunction!(sum_standard_layout)(py)?;
    ///     let sum_dynamic_strides = wrap_pyfunction!(sum_dynamic_strides)(py)?;
    ///
    ///     py_run!(py, np sum_standard_layout, r"assert sum_standard_layout(np.ones((2, 2), order='F')) == 4.");
    ///     py_run!(py, np sum_standard_layout, r"assert sum_standard_layout(np.ones((2, 2, 2))[:,:,0]) is None");
    ///
    ///     py_run!(py, np sum_dynamic_strides, r"assert sum_dynamic_strides(np.ones((2, 2), order='F')) == 4.");
    ///     py_run!(py, np sum_dynamic_strides, r"assert sum_dynamic_strides(np.ones((2, 2, 2))[:,:,0]) == 4.");
    /// #   Ok(())
    /// })
    /// # }
    /// ```
    #[doc(alias = "nalgebra")]
    pub fn try_as_matrix<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixView<'_, N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        unsafe { self.array.try_as_matrix() }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadonlyArray<'py, N, Ix1>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this one-dimensional array into a [`nalgebra::DMatrixView`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix(&self) -> nalgebra::DMatrixView<'_, N, nalgebra::Dyn, nalgebra::Dyn> {
        self.try_as_matrix().unwrap()
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadonlyArray<'py, N, Ix2>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this two-dimensional array into a [`nalgebra::DMatrixView`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix(&self) -> nalgebra::DMatrixView<'_, N, nalgebra::Dyn, nalgebra::Dyn> {
        self.try_as_matrix().unwrap()
    }
}

impl<'py, T, D> Clone for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn clone(&self) -> Self {
        acquire(self.array.py(), self.array.as_array_ptr()).unwrap();

        Self {
            array: self.array.clone(),
        }
    }
}

impl<'py, T, D> Drop for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn drop(&mut self) {
        release(self.array.py(), self.array.as_array_ptr());
    }
}

impl<'py, T, D> fmt::Debug for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!(
            "PyReadonlyArray<{}, {}>",
            type_name::<T>(),
            type_name::<D>()
        );

        f.debug_struct(&name).finish()
    }
}

/// Read-write borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadonlyArray`] and no other instances of [`PyReadwriteArray`],
/// i.e. that only a single exclusive reference into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
#[repr(transparent)]
pub struct PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    array: Bound<'py, PyArray<T, D>>,
}

/// Read-write borrow of a zero-dimensional array.
pub type PyReadwriteArray0<'py, T> = PyReadwriteArray<'py, T, Ix0>;

/// Read-write borrow of a one-dimensional array.
pub type PyReadwriteArray1<'py, T> = PyReadwriteArray<'py, T, Ix1>;

/// Read-write borrow of a two-dimensional array.
pub type PyReadwriteArray2<'py, T> = PyReadwriteArray<'py, T, Ix2>;

/// Read-write borrow of a three-dimensional array.
pub type PyReadwriteArray3<'py, T> = PyReadwriteArray<'py, T, Ix3>;

/// Read-write borrow of a four-dimensional array.
pub type PyReadwriteArray4<'py, T> = PyReadwriteArray<'py, T, Ix4>;

/// Read-write borrow of a five-dimensional array.
pub type PyReadwriteArray5<'py, T> = PyReadwriteArray<'py, T, Ix5>;

/// Read-write borrow of a six-dimensional array.
pub type PyReadwriteArray6<'py, T> = PyReadwriteArray<'py, T, Ix6>;

/// Read-write borrow of an array whose dimensionality is determined at runtime.
pub type PyReadwriteArrayDyn<'py, T> = PyReadwriteArray<'py, T, IxDyn>;

impl<'py, T, D> Deref for PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    type Target = PyReadonlyArray<'py, T, D>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Exclusive references decay implictly into shared references.
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}
impl<'py, T, D> From<PyReadwriteArray<'py, T, D>> for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn from(value: PyReadwriteArray<'py, T, D>) -> Self {
        let array = value.array.clone();
        ::std::mem::drop(value);
        Self::try_new(array)
            .expect("releasing an exclusive reference should immediately permit a shared reference")
    }
}

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for PyReadwriteArray<'py, T, D> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let array = obj.downcast::<PyArray<T, D>>()?;
        Ok(array.readwrite())
    }
}

impl<'py, T, D> PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    pub(crate) fn try_new(array: Bound<'py, PyArray<T, D>>) -> Result<Self, BorrowError> {
        acquire_mut(array.py(), array.as_array_ptr())?;

        Ok(Self { array })
    }

    /// Provides a mutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> ArrayViewMut<'_, T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_array_mut() }
    }

    /// Provide a mutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> Result<&mut [T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_slice_mut() }
    }

    /// Provide a mutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.array.get_mut(index) }
    }

    /// Clear the [`WRITEABLE` flag][writeable] from the underlying NumPy array.
    ///
    /// Calling this will prevent any further [PyReadwriteArray]s from being taken out.  Python
    /// space can reset this flag, unless the additional flag [`OWNDATA`][owndata] is unset.  Such
    /// an array can be created from Rust space by using [PyArray::borrow_from_array_bound].
    ///
    /// [writeable]: https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_WRITEABLE
    /// [owndata]: https://numpy.org/doc/stable/reference/c-api/array.html#c.NPY_ARRAY_OWNDATA
    pub fn make_nonwriteable(self) -> PyReadonlyArray<'py, T, D> {
        // SAFETY: consuming the only extant mutable reference guarantees we cannot invalidate an
        // existing reference, nor allow the caller to keep hold of one.
        unsafe {
            (*self.as_array_ptr()).flags &= !flags::NPY_ARRAY_WRITEABLE;
        }
        self.into()
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N, D> PyReadwriteArray<'py, N, D>
where
    N: nalgebra::Scalar + Element,
    D: Dimension,
{
    /// Try to convert this array into a [`nalgebra::MatrixViewMut`] using the given shape and strides.
    ///
    /// See [`PyReadonlyArray::try_as_matrix`] for a discussion of the memory layout requirements.
    #[doc(alias = "nalgebra")]
    pub fn try_as_matrix_mut<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixViewMut<'_, N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        unsafe { self.array.try_as_matrix_mut() }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadwriteArray<'py, N, Ix1>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this one-dimensional array into a [`nalgebra::DMatrixViewMut`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix_mut(&self) -> nalgebra::DMatrixViewMut<'_, N, nalgebra::Dyn, nalgebra::Dyn> {
        self.try_as_matrix_mut().unwrap()
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadwriteArray<'py, N, Ix2>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this two-dimensional array into a [`nalgebra::DMatrixViewMut`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix_mut(&self) -> nalgebra::DMatrixViewMut<'_, N, nalgebra::Dyn, nalgebra::Dyn> {
        self.try_as_matrix_mut().unwrap()
    }
}

impl<'py, T> PyReadwriteArray<'py, T, Ix1>
where
    T: Element,
{
    /// Extends or truncates the dimensions of an array.
    ///
    /// Safe wrapper for [`PyArray::resize`].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{PyArray, PyArrayMethods, PyUntypedArrayMethods};
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 10, 1);
    ///     assert_eq!(pyarray.len(), 10);
    ///
    ///     let pyarray = pyarray.readwrite();
    ///     let pyarray = pyarray.resize(100).unwrap();
    ///     assert_eq!(pyarray.len(), 100);
    /// });
    /// ```
    pub fn resize<ID: IntoDimension>(self, dims: ID) -> PyResult<Self> {
        // SAFETY: Ownership of `self` proves exclusive access to the interior of the array.
        unsafe {
            self.array.resize(dims)?;
        }

        let py = self.array.py();
        let ptr = self.array.as_array_ptr();

        // Update the borrow metadata to match the shape change.
        release_mut(py, ptr);
        acquire_mut(py, ptr).unwrap();

        Ok(self)
    }
}

impl<'py, T, D> Drop for PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn drop(&mut self) {
        release_mut(self.array.py(), self.array.as_array_ptr());
    }
}

impl<'py, T, D> fmt::Debug for PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!(
            "PyReadwriteArray<{}, {}>",
            type_name::<T>(),
            type_name::<D>()
        );

        f.debug_struct(&name).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::{types::IntoPyDict, Python};

    use crate::array::PyArray1;
    use pyo3::ffi::c_str;

    #[test]
    fn test_debug_formatting() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            {
                let shared = array.readonly();

                assert_eq!(
                    format!("{:?}", shared),
                    "PyReadonlyArray<f64, ndarray::dimension::dim::Dim<[usize; 3]>>"
                );
            }

            {
                let exclusive = array.readwrite();

                assert_eq!(
                    format!("{:?}", exclusive),
                    "PyReadwriteArray<f64, ndarray::dimension::dim::Dim<[usize; 3]>>"
                );
            }
        });
    }

    #[test]
    #[should_panic(expected = "AlreadyBorrowed")]
    fn cannot_clone_exclusive_borrow_via_deref() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (3, 2, 1), false);

            let exclusive = array.readwrite();
            let _shared = exclusive.clone();
        });
    }

    #[test]
    fn failed_resize_does_not_double_release() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);

            // The view will make the internal reference check of `PyArray_Resize` fail.
            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let _view = py
                .eval(c_str!("array[:]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();

            let exclusive = array.readwrite();
            assert!(exclusive.resize(100).is_err());
        });
    }

    #[test]
    fn ineffective_resize_does_not_conflict() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);

            let exclusive = array.readwrite();
            assert!(exclusive.resize(10).is_ok());
        });
    }
}
