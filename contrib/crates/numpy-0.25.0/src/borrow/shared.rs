use std::collections::hash_map::Entry;
use std::ffi::{c_void, CString};
use std::mem::forget;
use std::os::raw::{c_char, c_int};
use std::slice::from_raw_parts;
use std::sync::Mutex;

use num_integer::gcd;
use pyo3::types::{PyAnyMethods, PyCapsuleMethods};
use pyo3::{exceptions::PyTypeError, sync::GILOnceCell, types::PyCapsule, PyResult, Python};
use rustc_hash::FxHashMap;

use crate::array::get_array_module;
use crate::cold;
use crate::error::BorrowError;
use crate::npyffi::{PyArrayObject, PyArray_Check, PyDataType_ELSIZE, NPY_ARRAY_WRITEABLE};

/// Defines the shared C API used for borrow checking
///
/// This structure will be placed into a capsule at
/// `numpy.core.multiarray._RUST_NUMPY_BORROW_CHECKING_API`.
///
/// Versions are assumed to be backwards-compatible, i.e.
/// an extension which knows version N will work using
/// any API version M as long as M >= N holds.
///
/// Put differently, the only valid changes are adding
/// fields (data or functions) at the end of the structure.
#[repr(C)]
struct Shared {
    version: u64,
    flags: *mut c_void,
    acquire: unsafe extern "C" fn(flags: *mut c_void, array: *mut PyArrayObject) -> c_int,
    acquire_mut: unsafe extern "C" fn(flags: *mut c_void, array: *mut PyArrayObject) -> c_int,
    release: unsafe extern "C" fn(flags: *mut c_void, array: *mut PyArrayObject),
    release_mut: unsafe extern "C" fn(flags: *mut c_void, array: *mut PyArrayObject),
}

unsafe impl Send for Shared {}

// These are the entry points which implement the shared borrow checking API:

unsafe extern "C" fn acquire_shared(flags: *mut c_void, array: *mut PyArrayObject) -> c_int {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let py = Python::assume_gil_acquired();
    let flags = &*(flags as *mut BorrowFlags);

    let address = base_address(py, array);
    let key = borrow_key(py, array);

    match flags.acquire(address, key) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

unsafe extern "C" fn acquire_mut_shared(flags: *mut c_void, array: *mut PyArrayObject) -> c_int {
    if (*array).flags & NPY_ARRAY_WRITEABLE == 0 {
        return -2;
    }

    // SAFETY: GIL must be held when calling `acquire_shared`.
    let py = Python::assume_gil_acquired();
    let flags = &*(flags as *mut BorrowFlags);

    let address = base_address(py, array);
    let key = borrow_key(py, array);

    match flags.acquire_mut(address, key) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

unsafe extern "C" fn release_shared(flags: *mut c_void, array: *mut PyArrayObject) {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let py = Python::assume_gil_acquired();
    let flags = &*(flags as *mut BorrowFlags);
    let address = base_address(py, array);
    let key = borrow_key(py, array);

    flags.release(address, key);
}

unsafe extern "C" fn release_mut_shared(flags: *mut c_void, array: *mut PyArrayObject) {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let py = Python::assume_gil_acquired();
    let flags = &*(flags as *mut BorrowFlags);

    let address = base_address(py, array);
    let key = borrow_key(py, array);

    flags.release_mut(address, key);
}

// This global state is a cache used to access the shared borrow checking API from this extension:

struct SharedPtr(GILOnceCell<*const Shared>);

unsafe impl Send for SharedPtr {}

unsafe impl Sync for SharedPtr {}

static SHARED: SharedPtr = SharedPtr(GILOnceCell::new());

fn get_or_insert_shared<'py>(py: Python<'py>) -> PyResult<&'py Shared> {
    let shared = SHARED.0.get_or_try_init(py, || insert_shared(py))?;

    // SAFETY: We inserted the capsule if it was missing
    // and verified that it contains a compatible version.
    Ok(unsafe { &**shared })
}

// This function will publish this extension's version of the shared borrow checking API
// as a capsule placed at `numpy.core.multiarray._RUST_NUMPY_BORROW_CHECKING_API` and
// immediately initialize the cache used access it from this extension.

#[cold]
fn insert_shared<'py>(py: Python<'py>) -> PyResult<*const Shared> {
    let module = get_array_module(py)?;

    let capsule = match module.getattr("_RUST_NUMPY_BORROW_CHECKING_API") {
        Ok(capsule) => capsule.downcast_into::<PyCapsule>()?,
        Err(_err) => {
            let flags: *mut BorrowFlags = Box::into_raw(Box::default());

            let shared = Shared {
                version: 1,
                flags: flags as *mut c_void,
                acquire: acquire_shared,
                acquire_mut: acquire_mut_shared,
                release: release_shared,
                release_mut: release_mut_shared,
            };

            let capsule = PyCapsule::new_with_destructor(
                py,
                shared,
                Some(CString::new("_RUST_NUMPY_BORROW_CHECKING_API").unwrap()),
                |shared, _ctx| {
                    // SAFETY: `shared.flags` was initialized using `Box::into_raw`.
                    let _ = unsafe { Box::from_raw(shared.flags as *mut BorrowFlags) };
                },
            )?;
            module.setattr("_RUST_NUMPY_BORROW_CHECKING_API", &capsule)?;
            capsule
        }
    };

    // SAFETY: All versions of the shared borrow checking API start with a version field.
    let version = unsafe { *capsule.pointer().cast::<u64>() };
    if version < 1 {
        return Err(PyTypeError::new_err(format!(
            "Version {} of borrow checking API is not supported by this version of rust-numpy",
            version
        )));
    }

    let ptr = capsule.pointer();

    // Intentionally leak a reference to the capsule
    // so we can safely cache a pointer into its interior.
    forget(capsule);

    Ok(ptr.cast())
}

// These entry points will be used to access the shared borrow checking API from this extension:

pub fn acquire<'py>(py: Python<'py>, array: *mut PyArrayObject) -> Result<(), BorrowError> {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    let rc = unsafe { (shared.acquire)(shared.flags, array) };

    match rc {
        0 => Ok(()),
        -1 => Err(BorrowError::AlreadyBorrowed),
        rc => panic!("Unexpected return code {} from borrow checking API", rc),
    }
}

pub fn acquire_mut<'py>(py: Python<'py>, array: *mut PyArrayObject) -> Result<(), BorrowError> {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    let rc = unsafe { (shared.acquire_mut)(shared.flags, array) };

    match rc {
        0 => Ok(()),
        -1 => Err(BorrowError::AlreadyBorrowed),
        -2 => Err(BorrowError::NotWriteable),
        rc => panic!("Unexpected return code {} from borrow checking API", rc),
    }
}

pub fn release<'py>(py: Python<'py>, array: *mut PyArrayObject) {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    unsafe {
        (shared.release)(shared.flags, array);
    }
}

pub fn release_mut<'py>(py: Python<'py>, array: *mut PyArrayObject) {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    unsafe {
        (shared.release_mut)(shared.flags, array);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BorrowKey {
    /// exclusive range of lowest and highest address covered by array
    pub range: (*mut c_char, *mut c_char),
    /// the data address on which address computations are based
    pub data_ptr: *mut c_char,
    /// the greatest common divisor of the strides of the array
    pub gcd_strides: isize,
}

impl BorrowKey {
    fn conflicts(&self, other: &Self) -> bool {
        debug_assert!(self.range.0 <= self.range.1);
        debug_assert!(other.range.0 <= other.range.1);

        if other.range.0 >= self.range.1 || self.range.0 >= other.range.1 {
            return false;
        }

        // The Diophantine equation which describes whether any integers can combine the data pointers and strides of the two arrays s.t.
        // they yield the same element has a solution if and only if the GCD of all strides divides the difference of the data pointers.
        //
        // That solution could be out of bounds which mean that this is still an over-approximation.
        // It appears sufficient to handle typical cases like the color channels of an image,
        // but fails when slicing an array with a step size that does not divide the dimension along that axis.
        //
        // https://users.rust-lang.org/t/math-for-borrow-checking-numpy-arrays/73303
        let ptr_diff = unsafe { self.data_ptr.offset_from(other.data_ptr).abs() };
        let gcd_strides = gcd(self.gcd_strides, other.gcd_strides);

        if ptr_diff % gcd_strides != 0 {
            return false;
        }

        // By default, a conflict is assumed as it is the safe choice without actually solving the aliasing equation.
        true
    }
}

type BorrowFlagsInner = FxHashMap<*mut c_void, FxHashMap<BorrowKey, isize>>;

#[derive(Default)]
struct BorrowFlags(Mutex<BorrowFlagsInner>);

impl BorrowFlags {
    fn acquire(&self, address: *mut c_void, key: BorrowKey) -> Result<(), ()> {
        let mut borrow_flags = self.0.lock().unwrap();
        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(readers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*readers, 0);

                    let new_readers = readers.wrapping_add(1);

                    if new_readers <= 0 {
                        cold();
                        return Err(());
                    }

                    *readers = new_readers;
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, readers)| key.conflicts(other) && *readers < 0)
                    {
                        cold();
                        return Err(());
                    }

                    same_base_arrays.insert(key, 1);
                }
            }
            Entry::Vacant(entry) => {
                let mut same_base_arrays =
                    FxHashMap::with_capacity_and_hasher(1, Default::default());
                same_base_arrays.insert(key, 1);
                entry.insert(same_base_arrays);
            }
        }

        Ok(())
    }

    fn release(&self, address: *mut c_void, key: BorrowKey) {
        let mut borrow_flags = self.0.lock().unwrap();

        let same_base_arrays = borrow_flags.get_mut(&address).unwrap();
        let readers = same_base_arrays.get_mut(&key).unwrap();

        *readers -= 1;

        if *readers == 0 {
            if same_base_arrays.len() > 1 {
                same_base_arrays.remove(&key).unwrap();
            } else {
                borrow_flags.remove(&address).unwrap();
            }
        }
    }

    fn acquire_mut(&self, address: *mut c_void, key: BorrowKey) -> Result<(), ()> {
        let mut borrow_flags = self.0.lock().unwrap();

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(writers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*writers, 0);

                    cold();
                    return Err(());
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, writers)| key.conflicts(other) && *writers != 0)
                    {
                        cold();
                        return Err(());
                    }

                    same_base_arrays.insert(key, -1);
                }
            }
            Entry::Vacant(entry) => {
                let mut same_base_arrays =
                    FxHashMap::with_capacity_and_hasher(1, Default::default());
                same_base_arrays.insert(key, -1);
                entry.insert(same_base_arrays);
            }
        }

        Ok(())
    }

    fn release_mut(&self, address: *mut c_void, key: BorrowKey) {
        let mut borrow_flags = self.0.lock().unwrap();

        let same_base_arrays = borrow_flags.get_mut(&address).unwrap();

        if same_base_arrays.len() > 1 {
            same_base_arrays.remove(&key).unwrap();
        } else {
            borrow_flags.remove(&address);
        }
    }
}

fn base_address<'py>(py: Python<'py>, mut array: *mut PyArrayObject) -> *mut c_void {
    loop {
        let base = unsafe { (*array).base };

        if base.is_null() {
            return array as *mut c_void;
        } else if unsafe { PyArray_Check(py, base) } != 0 {
            array = base as *mut PyArrayObject;
        } else {
            return base as *mut c_void;
        }
    }
}

fn borrow_key<'py>(py: Python<'py>, array: *mut PyArrayObject) -> BorrowKey {
    let range = data_range(py, array);

    let data_ptr = unsafe { (*array).data };
    let gcd_strides = gcd_strides(array);

    BorrowKey {
        range,
        data_ptr,
        gcd_strides,
    }
}

fn data_range<'py>(py: Python<'py>, array: *mut PyArrayObject) -> (*mut c_char, *mut c_char) {
    let nd = unsafe { (*array).nd } as usize;
    let data = unsafe { (*array).data };

    if nd == 0 {
        return (data, data);
    }

    let shape = unsafe { from_raw_parts((*array).dimensions as *mut usize, nd) };
    let strides = unsafe { from_raw_parts((*array).strides, nd) };

    let itemsize = unsafe { PyDataType_ELSIZE(py, (*array).descr) } as isize;

    let mut start = 0;
    let mut end = 0;

    if shape.iter().all(|dim| *dim != 0) {
        for (&dim, &stride) in shape.iter().zip(strides) {
            let offset = (dim - 1) as isize * stride;

            if offset >= 0 {
                end += offset;
            } else {
                start += offset;
            }
        }

        end += itemsize;
    }

    let start = unsafe { data.offset(start) };
    let end = unsafe { data.offset(end) };

    (start, end)
}

fn gcd_strides(array: *mut PyArrayObject) -> isize {
    let nd = unsafe { (*array).nd } as usize;

    if nd == 0 {
        return 1;
    }

    let strides = unsafe { from_raw_parts((*array).strides, nd) };

    strides.iter().copied().reduce(gcd).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array;
    use pyo3::types::IntoPyDict;

    use crate::array::{PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods};
    use crate::convert::IntoPyArray;
    use crate::untyped_array::PyUntypedArrayMethods;
    use pyo3::ffi::c_str;

    struct BorrowFlagsState {
        #[cfg(not(Py_GIL_DISABLED))]
        n_flags: usize,
        n_arrays: usize,
        flag: Option<isize>,
    }

    fn get_borrow_flags_state<'py>(
        py: Python<'py>,
        base: *mut c_void,
        key: &BorrowKey,
    ) -> BorrowFlagsState {
        let shared = get_or_insert_shared(py).unwrap();
        assert_eq!(shared.version, 1);
        let inner = unsafe { &(*(shared.flags as *mut BorrowFlags)).0 }
            .lock()
            .unwrap();
        if let Some(base_arrays) = inner.get(&base) {
            BorrowFlagsState {
                #[cfg(not(Py_GIL_DISABLED))]
                n_flags: inner.len(),
                n_arrays: base_arrays.len(),
                flag: base_arrays.get(key).copied(),
            }
        } else {
            BorrowFlagsState {
                #[cfg(not(Py_GIL_DISABLED))]
                n_flags: 0,
                n_arrays: 0,
                flag: None,
            }
        }
    }

    #[test]
    fn without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(base.is_null());

            let base_address = base_address(py, array.as_array_ptr());
            assert_eq!(base_address, array.as_ptr().cast());

            let data_range = data_range(py, array.as_array_ptr());
            assert_eq!(data_range.0, array.data() as *mut c_char);
            assert_eq!(data_range.1, unsafe { array.data().add(6) } as *mut c_char);
        });
    }

    #[test]
    fn with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(py, array.as_array_ptr());
            assert_ne!(base_address, array.as_ptr().cast());
            assert_eq!(base_address, base.cast::<c_void>());

            let data_range = data_range(py, array.as_array_ptr());
            assert_eq!(data_range.0, array.data().cast::<c_char>());
            assert_eq!(data_range.1, unsafe {
                array.data().add(6).cast::<c_char>()
            });
        });
    }

    #[test]
    fn view_without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let view = py
                .eval(c_str!("array[:,:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(
                view.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>()
            );

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as *mut c_void, array.as_ptr().cast::<c_void>());

            let base_address = base_address(py, view.as_array_ptr());
            assert_ne!(base_address, view.as_ptr().cast::<c_void>());
            assert_eq!(base_address, base.cast::<c_void>());

            let data_range = data_range(py, view.as_array_ptr());
            assert_eq!(data_range.0, array.data() as *mut c_char);
            assert_eq!(data_range.1, unsafe { array.data().add(4) } as *mut c_char);
        });
    }

    #[test]
    fn view_with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let view = py
                .eval(c_str!("array[:,:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(
                view.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>(),
            );

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base.cast::<c_void>(), array.as_ptr().cast::<c_void>());

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(py, view.as_array_ptr());
            assert_ne!(base_address, view.as_ptr().cast::<c_void>());
            assert_ne!(base_address, array.as_ptr().cast::<c_void>());
            assert_eq!(base_address, base.cast::<c_void>());

            let data_range = data_range(py, view.as_array_ptr());
            assert_eq!(data_range.0, array.data().cast::<c_char>());
            assert_eq!(data_range.1, unsafe {
                array.data().add(4).cast::<c_char>()
            });
        });
    }

    #[test]
    fn view_of_view_without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let view1 = py
                .eval(c_str!("array[:,:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(
                view1.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>()
            );

            let locals = [("view1", &view1)].into_py_dict(py).unwrap();
            let view2 = py
                .eval(c_str!("view1[:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(
                view2.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>()
            );
            assert_ne!(
                view2.as_ptr().cast::<c_void>(),
                view1.as_ptr().cast::<c_void>()
            );

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base as *mut c_void, array.as_ptr().cast::<c_void>());

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base as *mut c_void, array.as_ptr().cast::<c_void>());

            let base_address = base_address(py, view2.as_array_ptr());
            assert_ne!(base_address, view2.as_ptr().cast::<c_void>());
            assert_ne!(base_address, view1.as_ptr().cast::<c_void>());
            assert_eq!(base_address, base as *mut c_void);

            let data_range = data_range(py, view2.as_array_ptr());
            assert_eq!(data_range.0, array.data() as *mut c_char);
            assert_eq!(data_range.1, unsafe { array.data().add(1) } as *mut c_char);
        });
    }

    #[test]
    fn view_of_view_with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let view1 = py
                .eval(c_str!("array[:,:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(
                view1.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>(),
            );

            let locals = [("view1", &view1)].into_py_dict(py).unwrap();
            let view2 = py
                .eval(c_str!("view1[:,0]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(
                view2.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>(),
            );
            assert_ne!(
                view2.as_ptr().cast::<c_void>(),
                view1.as_ptr().cast::<c_void>(),
            );

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base.cast::<c_void>(), array.as_ptr().cast::<c_void>());

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base.cast::<c_void>(), array.as_ptr().cast::<c_void>());

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(py, view2.as_array_ptr());
            assert_ne!(base_address, view2.as_ptr().cast::<c_void>());
            assert_ne!(base_address, view1.as_ptr().cast::<c_void>());
            assert_ne!(base_address, array.as_ptr().cast::<c_void>());
            assert_eq!(base_address, base.cast::<c_void>());

            let data_range = data_range(py, view2.as_array_ptr());
            assert_eq!(data_range.0, array.data().cast::<c_char>());
            assert_eq!(data_range.1, unsafe {
                array.data().add(1).cast::<c_char>()
            });
        });
    }

    #[test]
    fn view_with_negative_strides() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", &array)].into_py_dict(py).unwrap();
            let view = py
                .eval(c_str!("array[::-1,:,::-1]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray3<f64>>()
                .unwrap();
            assert_ne!(
                view.as_ptr().cast::<c_void>(),
                array.as_ptr().cast::<c_void>()
            );

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base.cast::<c_void>(), array.as_ptr().cast::<c_void>());

            let base_address = base_address(py, view.as_array_ptr());
            assert_ne!(base_address, view.as_ptr().cast::<c_void>());
            assert_eq!(base_address, base.cast::<c_void>());

            let data_range = data_range(py, view.as_array_ptr());
            assert_eq!(view.data(), unsafe { array.data().offset(2) });
            assert_eq!(data_range.0, unsafe { view.data().offset(-2) }
                as *mut c_char);
            assert_eq!(data_range.1, unsafe { view.data().offset(4) }
                as *mut c_char);
        });
    }

    #[test]
    fn array_with_zero_dimensions() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 0, 3), false);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(base.is_null());

            let base_address = base_address(py, array.as_array_ptr());
            assert_eq!(base_address, array.as_ptr().cast::<c_void>());

            let data_range = data_range(py, array.as_array_ptr());
            assert_eq!(data_range.0, array.data() as *mut c_char);
            assert_eq!(data_range.1, array.data() as *mut c_char);
        });
    }

    #[test]
    fn view_with_non_dividing_strides() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (10, 10), false);
            let locals = [("array", array)].into_py_dict(py).unwrap();

            let view1 = py
                .eval(c_str!("array[:,::3]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();

            let key1 = borrow_key(py, view1.as_array_ptr());

            assert_eq!(view1.strides(), &[80, 24]);
            assert_eq!(key1.gcd_strides, 8);

            let view2 = py
                .eval(c_str!("array[:,1::3]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();

            let key2 = borrow_key(py, view2.as_array_ptr());

            assert_eq!(view2.strides(), &[80, 24]);
            assert_eq!(key2.gcd_strides, 8);

            let view3 = py
                .eval(c_str!("array[:,::2]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();

            let key3 = borrow_key(py, view3.as_array_ptr());

            assert_eq!(view3.strides(), &[80, 16]);
            assert_eq!(key3.gcd_strides, 16);

            let view4 = py
                .eval(c_str!("array[:,1::2]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray2<f64>>()
                .unwrap();

            let key4 = borrow_key(py, view4.as_array_ptr());

            assert_eq!(view4.strides(), &[80, 16]);
            assert_eq!(key4.gcd_strides, 16);

            assert!(!key3.conflicts(&key4));
            assert!(key1.conflicts(&key3));
            assert!(key2.conflicts(&key4));

            // This is a false conflict where all aliasing indices like (0,7) and (2,0) are out of bounds.
            assert!(key1.conflicts(&key2));
        });
    }

    #[test]
    fn borrow_multiple_arrays() {
        Python::with_gil(|py| {
            let array1 = PyArray::<f64, _>::zeros(py, 10, false);
            let array2 = PyArray::<f64, _>::zeros(py, 10, false);

            let base1 = base_address(py, array1.as_array_ptr());
            let base2 = base_address(py, array2.as_array_ptr());

            let key1 = borrow_key(py, array1.as_array_ptr());
            let _exclusive1 = array1.readwrite();

            {
                let state = get_borrow_flags_state(py, base1, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);

                assert_eq!(state.n_arrays, 1);
                assert_eq!(state.flag, Some(-1));
            }

            let key2 = borrow_key(py, array2.as_array_ptr());
            let _shared2 = array2.readonly();

            {
                let state = get_borrow_flags_state(py, base1, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 2);

                assert_eq!(state.n_arrays, 1);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base2, &key2);
                assert_eq!(state.n_arrays, 1);
                assert_eq!(state.flag, Some(1));
            }
        });
    }

    #[test]
    fn borrow_multiple_views() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);
            let base = base_address(py, array.as_array_ptr());

            let locals = [("array", array)].into_py_dict(py).unwrap();

            let view1 = py
                .eval(c_str!("array[:5]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();

            let key1 = borrow_key(py, view1.as_array_ptr());
            let exclusive1 = view1.readwrite();

            {
                let state = get_borrow_flags_state(py, base, &key1);

                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 1);
                assert_eq!(state.flag, Some(-1));
            }

            let view2 = py
                .eval(c_str!("array[5:]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();

            let key2 = borrow_key(py, view2.as_array_ptr());
            let shared2 = view2.readonly();

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 2);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, Some(1));
            }

            let view3 = py
                .eval(c_str!("array[5:]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();

            let key3 = borrow_key(py, view3.as_array_ptr());
            let shared3 = view3.readonly();

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 2);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, Some(2));

                let state = get_borrow_flags_state(py, base, &key3);
                assert_eq!(state.flag, Some(2));
            }

            let view4 = py
                .eval(c_str!("array[7:]"), None, Some(&locals))
                .unwrap()
                .downcast_into::<PyArray1<f64>>()
                .unwrap();

            let key4 = borrow_key(py, view4.as_array_ptr());
            let shared4 = view4.readonly();

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 3);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, Some(2));

                let state = get_borrow_flags_state(py, base, &key3);
                assert_eq!(state.flag, Some(2));

                let state = get_borrow_flags_state(py, base, &key4);
                assert_eq!(state.flag, Some(1));
            }

            drop(shared2);

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 3);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, Some(1));

                let state = get_borrow_flags_state(py, base, &key3);
                assert_eq!(state.flag, Some(1));

                let state = get_borrow_flags_state(py, base, &key4);
                assert_eq!(state.flag, Some(1));
            }

            drop(shared3);

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 2);
                assert_eq!(state.flag, Some(-1));

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, None);

                let state = get_borrow_flags_state(py, base, &key3);
                assert_eq!(state.flag, None);

                let state = get_borrow_flags_state(py, base, &key4);
                assert_eq!(state.flag, Some(1));
            }

            drop(exclusive1);

            {
                let state = get_borrow_flags_state(py, base, &key1);
                #[cfg(not(Py_GIL_DISABLED))]
                // borrow checking state is shared and other tests might have registered a borrow
                assert_eq!(state.n_flags, 1);
                assert_eq!(state.n_arrays, 1);
                assert_eq!(state.flag, None);

                let state = get_borrow_flags_state(py, base, &key2);
                assert_eq!(state.flag, None);

                let state = get_borrow_flags_state(py, base, &key3);
                assert_eq!(state.flag, None);

                let state = get_borrow_flags_state(py, base, &key4);
                assert_eq!(state.flag, Some(1));
            }

            drop(shared4);

            #[cfg(not(Py_GIL_DISABLED))]
            // borrow checking state is shared and other tests might have registered a borrow
            {
                assert_eq!(get_borrow_flags_state(py, base, &key1).n_flags, 0);
            }
        });
    }
}
