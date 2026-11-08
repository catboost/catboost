//! Low-Level bindings for NumPy C API.
//!
//! <https://numpy.org/doc/stable/reference/c-api>
#![allow(
    non_camel_case_types,
    missing_docs,
    missing_debug_implementations,
    clippy::too_many_arguments,
    clippy::missing_safety_doc
)]

use std::mem::forget;
use std::os::raw::{c_uint, c_void};

use pyo3::{
    sync::GILOnceCell,
    types::{PyAnyMethods, PyCapsule, PyCapsuleMethods, PyModule},
    PyResult, Python,
};

pub const API_VERSION_2_0: c_uint = 0x00000012;

static API_VERSION: GILOnceCell<c_uint> = GILOnceCell::new();

fn get_numpy_api<'py>(
    py: Python<'py>,
    module: &str,
    capsule: &str,
) -> PyResult<*const *const c_void> {
    let module = PyModule::import(py, module)?;
    let capsule = module.getattr(capsule)?.downcast_into::<PyCapsule>()?;

    let api = capsule.pointer() as *const *const c_void;

    // Intentionally leak a reference to the capsule
    // so we can safely cache a pointer into its interior.
    forget(capsule);

    Ok(api)
}

/// Returns whether the runtime `numpy` version is 2.0 or greater.
pub fn is_numpy_2<'py>(py: Python<'py>) -> bool {
    let api_version = *API_VERSION.get_or_init(py, || unsafe {
        PY_ARRAY_API.PyArray_GetNDArrayCFeatureVersion(py)
    });
    api_version >= API_VERSION_2_0
}

// Implements wrappers for NumPy's Array and UFunc API
macro_rules! impl_api {
    // API available on all versions
    [$offset: expr; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            let fptr = self.get(py, $offset) as *const extern "C" fn ($($arg : $t), *) $(-> $ret)*;
            (*fptr)($($arg), *)
        }
    };

    // API with version constraints, checked at runtime
    [$offset: expr; NumPy1; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            assert!(
                !is_numpy_2(py),
                "{} requires API < {:08X} (NumPy 1) but the runtime version is API {:08X}",
                stringify!($fname),
                API_VERSION_2_0,
                *API_VERSION.get(py).expect("API_VERSION is initialized"),
            );
            let fptr = self.get(py, $offset) as *const extern "C" fn ($($arg: $t), *) $(-> $ret)*;
            (*fptr)($($arg), *)
        }

    };
    [$offset: expr; NumPy2; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            assert!(
                is_numpy_2(py),
                "{} requires API {:08X} or greater (NumPy 2) but the runtime version is API {:08X}",
                stringify!($fname),
                API_VERSION_2_0,
                *API_VERSION.get(py).expect("API_VERSION is initialized"),
            );
            let fptr = self.get(py, $offset) as *const extern "C" fn ($($arg: $t), *) $(-> $ret)*;
            (*fptr)($($arg), *)
        }

    };
}

pub mod array;
pub mod flags;
pub mod objects;
pub mod types;
pub mod ufunc;

pub use self::array::*;
pub use self::flags::*;
pub use self::objects::*;
pub use self::types::*;
pub use self::ufunc::*;
