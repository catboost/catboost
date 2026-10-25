//! Support datetimes and timedeltas
//!
//! This module provides wrappers for NumPy's [`datetime64`][scalars-datetime64] and [`timedelta64`][scalars-timedelta64] types
//! which are used for time keeping with with an emphasis on scientific applications.
//! This means that while these types differentiate absolute and relative quantities, they ignore calendars (a month is always 30.44 days) and time zones.
//! On the other hand, their flexible units enable them to support either a large range (up to 2<sup>64</sup> years) or high precision (down to 10<sup>-18</sup> seconds).
//!
//! [The corresponding section][datetime] of the NumPy documentation contains more information.
//!
//! # Example
//!
//! ```
//! use numpy::{datetime::{units, Datetime, Timedelta}, PyArray1, PyArrayMethods};
//! use pyo3::{Python, types::PyAnyMethods, ffi::c_str};
//! # use pyo3::types::PyDict;
//!
//! # fn main() -> pyo3::PyResult<()> {
//! Python::with_gil(|py| {
//! #    let locals = py
//! #        .eval(c_str!("{ 'np': __import__('numpy') }"), None, None)?
//! #        .downcast_into::<PyDict>()?;
//! #
//!     let array = py
//!         .eval(
//!             c_str!("np.array([np.datetime64('2017-04-21')])"),
//!             None,
//!             Some(&locals),
//!         )?
//!         .downcast_into::<PyArray1<Datetime<units::Days>>>()?;
//!
//!     assert_eq!(
//!         array.get_owned(0).unwrap(),
//!         Datetime::<units::Days>::from(17_277)
//!     );
//!
//!     let array = py
//!         .eval(
//!             c_str!("np.array([np.datetime64('2022-03-29')]) - np.array([np.datetime64('2017-04-21')])"),
//!             None,
//!             Some(&locals),
//!         )?
//!         .downcast_into::<PyArray1<Timedelta<units::Days>>>()?;
//!
//!     assert_eq!(
//!         array.get_owned(0).unwrap(),
//!         Timedelta::<units::Days>::from(1_803)
//!     );
//! #   Ok(())
//! })
//! # }
//! ```
//!
//! [datetime]: https://numpy.org/doc/stable/reference/arrays.datetime.html
//! [scalars-datetime64]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64
//! [scalars-timedelta64]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.timedelta64

use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Mutex;

use pyo3::sync::MutexExt;
use pyo3::{Bound, Py, Python};
use rustc_hash::FxHashMap;

use crate::dtype::{clone_methods_impl, Element, PyArrayDescr, PyArrayDescrMethods};
use crate::npyffi::{
    PyArray_DatetimeDTypeMetaData, PyDataType_C_METADATA, NPY_DATETIMEUNIT, NPY_TYPES,
};

/// Represents the [datetime units][datetime-units] supported by NumPy
///
/// [datetime-units]: https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
pub trait Unit: Send + Sync + Clone + Copy + PartialEq + Eq + Hash + PartialOrd + Ord {
    /// The matching NumPy [datetime unit code][NPY_DATETIMEUNIT]
    ///
    /// [NPY_DATETIMEUNIT]: https://github.com/numpy/numpy/blob/4c60b3263ac50e5e72f6a909e156314fc3c9cba0/numpy/core/include/numpy/ndarraytypes.h#L276
    const UNIT: NPY_DATETIMEUNIT;

    /// The abbrevation used for debug formatting
    const ABBREV: &'static str;
}

macro_rules! define_units {
    ($($(#[$meta:meta])* $struct:ident => $unit:ident $abbrev:literal,)+) => {
        $(

        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $struct;

        impl Unit for $struct {
            const UNIT: NPY_DATETIMEUNIT = NPY_DATETIMEUNIT::$unit;

            const ABBREV: &'static str = $abbrev;
        }

        )+
    };
}

/// Predefined implementors of the [`Unit`] trait
pub mod units {
    use super::*;

    define_units!(
        #[doc = "Years, i.e. 12 months"]
        Years => NPY_FR_Y "a",
        #[doc = "Months, i.e. 30 days"]
        Months => NPY_FR_M "mo",
        #[doc = "Weeks, i.e. 7 days"]
        Weeks => NPY_FR_W "w",
        #[doc = "Days, i.e. 24 hours"]
        Days => NPY_FR_D "d",
        #[doc = "Hours, i.e. 60 minutes"]
        Hours => NPY_FR_h "h",
        #[doc = "Minutes, i.e. 60 seconds"]
        Minutes => NPY_FR_m "min",
        #[doc = "Seconds"]
        Seconds => NPY_FR_s "s",
        #[doc = "Milliseconds, i.e. 10^-3 seconds"]
        Milliseconds => NPY_FR_ms "ms",
        #[doc = "Microseconds, i.e. 10^-6 seconds"]
        Microseconds => NPY_FR_us "µs",
        #[doc = "Nanoseconds, i.e. 10^-9 seconds"]
        Nanoseconds => NPY_FR_ns "ns",
        #[doc = "Picoseconds, i.e. 10^-12 seconds"]
        Picoseconds => NPY_FR_ps "ps",
        #[doc = "Femtoseconds, i.e. 10^-15 seconds"]
        Femtoseconds => NPY_FR_fs "fs",
        #[doc = "Attoseconds, i.e. 10^-18 seconds"]
        Attoseconds => NPY_FR_as "as",
    );
}

/// Corresponds to the [`datetime64`][scalars-datetime64] scalar type
///
/// [scalars-datetime64]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.datetime64
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Datetime<U: Unit>(i64, PhantomData<U>);

impl<U: Unit> From<i64> for Datetime<U> {
    fn from(val: i64) -> Self {
        Self(val, PhantomData)
    }
}

impl<U: Unit> From<Datetime<U>> for i64 {
    fn from(val: Datetime<U>) -> Self {
        val.0
    }
}

unsafe impl<U: Unit> Element for Datetime<U> {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        static DTYPES: TypeDescriptors = unsafe { TypeDescriptors::new(NPY_TYPES::NPY_DATETIME) };

        DTYPES.from_unit(py, U::UNIT)
    }

    clone_methods_impl!(Self);
}

impl<U: Unit> fmt::Debug for Datetime<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Datetime({} {})", self.0, U::ABBREV)
    }
}

/// Corresponds to the [`timedelta64`][scalars-datetime64] scalar type
///
/// [scalars-timedelta64]: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.timedelta64
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Timedelta<U: Unit>(i64, PhantomData<U>);

impl<U: Unit> From<i64> for Timedelta<U> {
    fn from(val: i64) -> Self {
        Self(val, PhantomData)
    }
}

impl<U: Unit> From<Timedelta<U>> for i64 {
    fn from(val: Timedelta<U>) -> Self {
        val.0
    }
}

unsafe impl<U: Unit> Element for Timedelta<U> {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        static DTYPES: TypeDescriptors = unsafe { TypeDescriptors::new(NPY_TYPES::NPY_TIMEDELTA) };

        DTYPES.from_unit(py, U::UNIT)
    }

    clone_methods_impl!(Self);
}

impl<U: Unit> fmt::Debug for Timedelta<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timedelta({} {})", self.0, U::ABBREV)
    }
}

struct TypeDescriptors {
    npy_type: NPY_TYPES,
    dtypes: Mutex<Option<FxHashMap<NPY_DATETIMEUNIT, Py<PyArrayDescr>>>>,
}

impl TypeDescriptors {
    /// `npy_type` must be either `NPY_DATETIME` or `NPY_TIMEDELTA`.
    const unsafe fn new(npy_type: NPY_TYPES) -> Self {
        Self {
            npy_type,
            dtypes: Mutex::new(None),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_unit<'py>(&self, py: Python<'py>, unit: NPY_DATETIMEUNIT) -> Bound<'py, PyArrayDescr> {
        let mut dtypes = self
            .dtypes
            .lock_py_attached(py)
            .expect("dtype cache poisoned");

        let dtype = match dtypes.get_or_insert_with(Default::default).entry(unit) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let dtype = PyArrayDescr::new_from_npy_type(py, self.npy_type);

                // SAFETY: `self.npy_type` is either `NPY_DATETIME` or `NPY_TIMEDELTA` which implies the type of `c_metadata`.
                unsafe {
                    let metadata = &mut *(PyDataType_C_METADATA(py, dtype.as_dtype_ptr())
                        as *mut PyArray_DatetimeDTypeMetaData);

                    metadata.meta.base = unit;
                    metadata.meta.num = 1;
                }

                entry.insert(dtype.into())
            }
        };

        dtype.bind(py).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::{
        ffi::c_str,
        py_run,
        types::{PyAnyMethods, PyDict, PyModule},
    };

    use crate::array::{PyArray1, PyArrayMethods};

    #[test]
    fn from_python_to_rust() {
        Python::with_gil(|py| {
            let locals = py
                .eval(c_str!("{ 'np': __import__('numpy') }"), None, None)
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();

            let array = py
                .eval(
                    c_str!("np.array([np.datetime64('1970-01-01')])"),
                    None,
                    Some(&locals),
                )
                .unwrap()
                .downcast_into::<PyArray1<Datetime<units::Days>>>()
                .unwrap();

            let value: i64 = array.get_owned(0).unwrap().into();
            assert_eq!(value, 0);
        });
    }

    #[test]
    fn from_rust_to_python() {
        Python::with_gil(|py| {
            let array = PyArray1::<Timedelta<units::Minutes>>::zeros(py, 1, false);

            *array.readwrite().get_mut(0).unwrap() = Timedelta::<units::Minutes>::from(5);

            let np = py
                .eval(c_str!("__import__('numpy')"), None, None)
                .unwrap()
                .downcast_into::<PyModule>()
                .unwrap();

            py_run!(py, array np, "assert array.dtype == np.dtype('timedelta64[m]')");
            py_run!(py, array np, "assert array[0] == np.timedelta64(5, 'm')");
        });
    }

    #[test]
    fn debug_formatting() {
        assert_eq!(
            format!("{:?}", Datetime::<units::Days>::from(28)),
            "Datetime(28 d)"
        );

        assert_eq!(
            format!("{:?}", Timedelta::<units::Milliseconds>::from(160)),
            "Timedelta(160 ms)"
        );
    }

    #[test]
    fn unit_conversion() {
        #[track_caller]
        fn convert<'py, S: Unit, D: Unit>(py: Python<'py>, expected_value: i64) {
            let array = PyArray1::<Timedelta<S>>::from_slice(py, &[Timedelta::<S>::from(1)]);
            let array = array.cast::<Timedelta<D>>(false).unwrap();

            let value: i64 = array.get_owned(0).unwrap().into();
            assert_eq!(value, expected_value);
        }

        Python::with_gil(|py| {
            convert::<units::Years, units::Days>(py, (97 + 400 * 365) / 400);
            convert::<units::Months, units::Days>(py, (97 + 400 * 365) / 400 / 12);

            convert::<units::Weeks, units::Seconds>(py, 7 * 24 * 60 * 60);
            convert::<units::Days, units::Seconds>(py, 24 * 60 * 60);
            convert::<units::Hours, units::Seconds>(py, 60 * 60);
            convert::<units::Minutes, units::Seconds>(py, 60);

            convert::<units::Seconds, units::Milliseconds>(py, 1_000);
            convert::<units::Seconds, units::Microseconds>(py, 1_000_000);
            convert::<units::Seconds, units::Nanoseconds>(py, 1_000_000_000);
            convert::<units::Seconds, units::Picoseconds>(py, 1_000_000_000_000);
            convert::<units::Seconds, units::Femtoseconds>(py, 1_000_000_000_000_000);

            convert::<units::Femtoseconds, units::Attoseconds>(py, 1_000);
        });
    }
}
