//! Defines conversion traits between Rust types and NumPy data types.

use std::{mem, os::raw::c_int, ptr};

use ndarray::{ArrayBase, Data, Dim, Dimension, IntoDimension, Ix1, OwnedRepr};
use pyo3::{Bound, Python};

use crate::array::{PyArray, PyArrayMethods};
use crate::dtype::Element;
use crate::error::MAX_DIMENSIONALITY_ERR;
use crate::npyffi::{self, npy_intp};
use crate::slice_container::PySliceContainer;

/// Conversion trait from owning Rust types into [`PyArray`].
///
/// This trait takes ownership of `self`, which means it holds a pointer into the Rust heap.
///
/// In addition, some destructive methods like `resize` cannot be used with NumPy arrays constructed using this trait.
///
/// # Example
///
/// ```
/// use numpy::{PyArray, IntoPyArray, PyArrayMethods};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].into_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
///
///     // Array cannot be resized when its data is owned by Rust.
///     unsafe {
///         assert!(py_array.resize(100).is_err());
///     }
/// });
/// ```
pub trait IntoPyArray: Sized {
    /// The element type of resulting array.
    type Item: Element;
    /// The dimension type of the resulting array.
    type Dim: Dimension;

    /// Consumes `self` and moves its data into a NumPy array.
    fn into_pyarray<'py>(self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>>;
}

impl<T: Element> IntoPyArray for Box<[T]> {
    type Item = T;
    type Dim = Ix1;

    fn into_pyarray<'py>(self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        let container = PySliceContainer::from(self);
        let dims = Dim([container.len]);
        let strides = [mem::size_of::<T>() as npy_intp];
        // The data pointer is derived only after dissolving `Box` into `PySliceContainer`
        // to avoid unsound aliasing of Box<[T]> which is currently noalias,
        // c.f. https://github.com/rust-lang/unsafe-code-guidelines/issues/326
        let data_ptr = container.ptr as *mut T;
        unsafe { PyArray::from_raw_parts(py, dims, strides.as_ptr(), data_ptr, container) }
    }
}

impl<T: Element> IntoPyArray for Vec<T> {
    type Item = T;
    type Dim = Ix1;

    fn into_pyarray<'py>(mut self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        let dims = Dim([self.len()]);
        let strides = [mem::size_of::<T>() as npy_intp];
        let data_ptr = self.as_mut_ptr();
        unsafe {
            PyArray::from_raw_parts(
                py,
                dims,
                strides.as_ptr(),
                data_ptr,
                PySliceContainer::from(self),
            )
        }
    }
}

impl<A, D> IntoPyArray for ArrayBase<OwnedRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    type Item = A;
    type Dim = D;

    fn into_pyarray<'py>(self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        PyArray::from_owned_array(py, self)
    }
}

/// Conversion trait from borrowing Rust types to [`PyArray`].
///
/// This trait takes `&self` by reference, which means it allocates in Python heap and then copies the elements there.
///
/// # Examples
///
/// ```
/// use numpy::{PyArray, ToPyArray, PyArrayMethods};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].to_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
/// });
/// ```
///
/// Due to copying the elments, this method converts non-contiguous arrays to C-order contiguous arrays.
///
/// ```
/// use numpy::prelude::*;
/// use numpy::{PyArray, ToPyArray};
/// use ndarray::{arr3, s};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let array = arr3(&[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
///     let py_array = array.slice(s![.., 0..1, ..]).to_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_array(), arr3(&[[[1, 2, 3]], [[7, 8, 9]]]));
///     assert!(py_array.is_c_contiguous());
/// });
/// ```
pub trait ToPyArray {
    /// The element type of resulting array.
    type Item: Element;
    /// The dimension type of the resulting array.
    type Dim: Dimension;

    /// Copies the content pointed to by `&self` into a newly allocated NumPy array.
    fn to_pyarray<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>>;
}

impl<T: Element> ToPyArray for [T] {
    type Item = T;
    type Dim = Ix1;

    fn to_pyarray<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        PyArray::from_slice(py, self)
    }
}

impl<S, D, A> ToPyArray for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: Element,
{
    type Item = A;
    type Dim = D;

    fn to_pyarray<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        let len = self.len();
        match self.order() {
            Some(flag) if A::IS_COPY => {
                // if the array is contiguous, copy it by `copy_nonoverlapping`.
                let strides = self.npy_strides();
                unsafe {
                    let array = PyArray::new_uninit(py, self.raw_dim(), strides.as_ptr(), flag);
                    ptr::copy_nonoverlapping(self.as_ptr(), array.data(), len);
                    array
                }
            }
            _ => {
                // if the array is not contiguous, copy all elements by `ArrayBase::iter`.
                let dim = self.raw_dim();
                unsafe {
                    let array = PyArray::<A, _>::new(py, dim, false);
                    let mut data_ptr = array.data();
                    for item in self.iter() {
                        data_ptr.write(item.clone_ref(py));
                        data_ptr = data_ptr.add(1);
                    }
                    array
                }
            }
        }
    }
}

#[cfg(feature = "nalgebra")]
impl<N, R, C, S> ToPyArray for nalgebra::Matrix<N, R, C, S>
where
    N: nalgebra::Scalar + Element,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::Storage<N, R, C>,
{
    type Item = N;
    type Dim = crate::Ix2;

    /// Note that the NumPy array always has Fortran memory layout
    /// matching the [memory layout][memory-layout] used by [`nalgebra`].
    ///
    /// [memory-layout]: https://nalgebra.org/docs/faq/#what-is-the-memory-layout-of-matrices
    fn to_pyarray<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<Self::Item, Self::Dim>> {
        unsafe {
            let array = PyArray::<N, _>::new(py, (self.nrows(), self.ncols()), true);
            let mut data_ptr = array.data();
            if self.data.is_contiguous() {
                ptr::copy_nonoverlapping(self.data.ptr(), data_ptr, self.len());
            } else {
                for item in self.iter() {
                    data_ptr.write(item.clone_ref(py));
                    data_ptr = data_ptr.add(1);
                }
            }
            array
        }
    }
}

pub(crate) trait ArrayExt {
    fn npy_strides(&self) -> [npyffi::npy_intp; 32];
    fn order(&self) -> Option<c_int>;
}

impl<A, S, D> ArrayExt for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn npy_strides(&self) -> [npyffi::npy_intp; 32] {
        let strides = self.strides();
        let itemsize = mem::size_of::<A>() as isize;

        assert!(strides.len() <= 32, "{}", MAX_DIMENSIONALITY_ERR);

        let mut new_strides = [0; 32];

        for i in 0..strides.len() {
            new_strides[i] = (strides[i] * itemsize) as npyffi::npy_intp;
        }

        new_strides
    }

    fn order(&self) -> Option<c_int> {
        if self.is_standard_layout() {
            Some(npyffi::NPY_ORDER::NPY_CORDER as _)
        } else if self.ndim() > 1 && self.raw_view().reversed_axes().is_standard_layout() {
            Some(npyffi::NPY_ORDER::NPY_FORTRANORDER as _)
        } else {
            None
        }
    }
}

/// Utility trait to specify the dimensions of an array.
pub trait ToNpyDims: Dimension + Sealed {
    #[doc(hidden)]
    fn ndim_cint(&self) -> c_int {
        self.ndim() as c_int
    }
    #[doc(hidden)]
    fn as_dims_ptr(&mut self) -> *mut npyffi::npy_intp {
        self.slice_mut().as_ptr() as *mut npyffi::npy_intp
    }
    #[doc(hidden)]
    fn to_npy_dims(&mut self) -> npyffi::PyArray_Dims {
        npyffi::PyArray_Dims {
            ptr: self.as_dims_ptr(),
            len: self.ndim_cint(),
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

impl<D> ToNpyDims for D where D: Dimension {}

/// Trait implemented by types that can be used to index an array.
///
/// This is equivalent to [`ndarray::NdIndex`] but accounts for
/// NumPy strides being in units of bytes instead of elements.
///
/// All types which implement [`IntoDimension`] implement this trait as well.
/// This includes at least
/// - [tuple](https://doc.rust-lang.org/stable/std/primitive.tuple.html)
/// - [array](https://doc.rust-lang.org/stable/std/primitive.array.html)
/// - [slice](https://doc.rust-lang.org/stable/std/primitive.slice.html)
pub trait NpyIndex: IntoDimension + Sealed {
    #[doc(hidden)]
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize>;
    #[doc(hidden)]
    fn get_unchecked<T>(self, strides: &[isize]) -> isize;
}

impl<D: IntoDimension> Sealed for D {}

impl<D: IntoDimension> NpyIndex for D {
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize> {
        let indices = self.into_dimension();
        let indices = indices.slice();

        if indices.len() != dims.len() {
            return None;
        }
        if indices.iter().zip(dims).any(|(i, d)| i >= d) {
            return None;
        }

        Some(get_unchecked_impl::<T>(indices, strides))
    }

    fn get_unchecked<T>(self, strides: &[isize]) -> isize {
        let indices = self.into_dimension();
        let indices = indices.slice();
        get_unchecked_impl::<T>(indices, strides)
    }
}

fn get_unchecked_impl<T>(indices: &[usize], strides: &[isize]) -> isize {
    let size = mem::size_of::<T>() as isize;

    indices
        .iter()
        .zip(strides)
        .map(|(&i, stride)| stride * i as isize / size)
        .sum()
}
