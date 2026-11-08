use std::{mem, ptr};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use pyo3::pyclass;

/// Utility type to safely store `Box<[_]>` or `Vec<_>` on the Python heap
#[pyclass(frozen)]
#[derive(Debug)]
pub(crate) struct PySliceContainer {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
    cap: usize,
    drop: unsafe fn(*mut u8, usize, usize),
}

// This resembles `unsafe impl<T: Send> Send for PySliceContainer<T> {}` if we
// were allow to use a generic there.
// SAFETY: Every construction below enforces `T: Send` fulfilling the ideal bound above
unsafe impl Send for PySliceContainer {}

// This resembles `unsafe impl<T: Sync> Sync for PySliceContainer<T> {}` if we
// were allow to use a generic there.
// SAFETY: Every construction below enforces `T: Sync` fulfilling the ideal bound above
unsafe impl Sync for PySliceContainer {}

impl<T: Send + Sync> From<Box<[T]>> for PySliceContainer {
    fn from(data: Box<[T]>) -> Self {
        unsafe fn drop_boxed_slice<T>(ptr: *mut u8, len: usize, _cap: usize) {
            let _ = Box::from_raw(ptr::slice_from_raw_parts_mut(ptr as *mut T, len));
        }

        // FIXME(adamreichold): Use `Box::into_raw` when
        // `*mut [T]::{as_mut_ptr, len}` become stable and compatible with our MSRV.
        let mut data = mem::ManuallyDrop::new(data);

        let ptr = data.as_mut_ptr() as *mut u8;
        let len = data.len();
        let cap = 0;
        let drop = drop_boxed_slice::<T>;

        Self {
            ptr,
            len,
            cap,
            drop,
        }
    }
}

impl<T: Send + Sync> From<Vec<T>> for PySliceContainer {
    fn from(data: Vec<T>) -> Self {
        unsafe fn drop_vec<T>(ptr: *mut u8, len: usize, cap: usize) {
            let _ = Vec::from_raw_parts(ptr as *mut T, len, cap);
        }

        // FIXME(adamreichold): Use `Vec::into_raw_parts`
        // when it becomes stable and compatible with our MSRV.
        let mut data = mem::ManuallyDrop::new(data);

        let ptr = data.as_mut_ptr() as *mut u8;
        let len = data.len();
        let cap = data.capacity();
        let drop = drop_vec::<T>;

        Self {
            ptr,
            len,
            cap,
            drop,
        }
    }
}

impl<A, D> From<ArrayBase<OwnedRepr<A>, D>> for PySliceContainer
where
    A: Send + Sync,
    D: Dimension,
{
    fn from(data: ArrayBase<OwnedRepr<A>, D>) -> Self {
        #[allow(deprecated)]
        Self::from(data.into_raw_vec())
    }
}

impl Drop for PySliceContainer {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr, self.len, self.cap);
        }
    }
}
