#[cfg(feature = "float")]
use num_traits::float::FloatCore;
use num_traits::PrimInt;

use super::super::dtype_strategy::Int;
/// The DTypeStrategy for which we implement the ScalarArgMinMax trait
#[cfg(any(feature = "float", feature = "half"))]
use super::super::dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};

/// Helper trait to initialize the min and max values & check if we should return
/// This will be implemented for all:
/// - ints - Int DTypeStrategy (see 1st impl block below)
/// - uints - Int DTypeStrategy (see 1st impl block below)
/// - floats: returning NaNs - FloatReturnNan DTypeStrategy (see 2nd impl block below)
/// - floats: ignoring NaNs - FloatIgnoreNaN DTypeStrategy (see 3rd impl block below)
///
trait SCALARInit<ScalarDType: Copy + PartialOrd> {
    const _RETURN_AT_NAN: bool;

    /// Initialize the initial value for the min and max values

    fn _init_min(start_value: ScalarDType) -> ScalarDType;

    fn _init_max(start_value: ScalarDType) -> ScalarDType;

    /// Check if we should allow the updating the value(s) with the first non-NaN value

    fn _allow_first_non_nan_update(start_value: ScalarDType) -> bool;

    /// Nan check

    fn _nan_check(v: ScalarDType) -> bool;
}

/// A trait providing the scalar implementation of the argminmax operations.
///
// This trait will be implemented for the different DTypeStrategy
pub trait ScalarArgMinMax<ScalarDType: Copy + PartialOrd> {
    /// Get the index of the minimum and maximum values in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the slice
    /// `(min_index, max_index)`.
    ///
    fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    /// Get the index of the minimum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the minimum value in the slice.
    ///
    fn argmin(data: &[ScalarDType]) -> usize;

    /// Get the index of the maximum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the maximum value in the slice.
    ///
    fn argmax(data: &[ScalarDType]) -> usize;
}

/// Type that implements the [ScalarArgMinMax](crate::ScalarArgMinMax) trait.
///
/// This struct implements the ScalarArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
// See the impl_scalar! macro below for the implementation of the ScalarArgMinMax trait
pub struct SCALAR<DTypeStrategy> {
    pub(crate) _dtype_strategy: std::marker::PhantomData<DTypeStrategy>,
}

/// ------- Implement the SCALARInit trait for the different DTypeStrategy -------

impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<Int>
where
    ScalarDType: PrimInt,
{
    const _RETURN_AT_NAN: bool = false;

    #[inline(always)]
    fn _init_min(start_value: ScalarDType) -> ScalarDType {
        start_value
    }

    #[inline(always)]
    fn _init_max(start_value: ScalarDType) -> ScalarDType {
        start_value
    }

    #[inline(always)]
    fn _allow_first_non_nan_update(_start_value: ScalarDType) -> bool {
        false
    }

    #[inline(always)]
    fn _nan_check(_v: ScalarDType) -> bool {
        false
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatReturnNaN>
where
    ScalarDType: FloatCore,
{
    const _RETURN_AT_NAN: bool = true;

    #[inline(always)]
    fn _init_min(start_value: ScalarDType) -> ScalarDType {
        start_value
    }

    #[inline(always)]
    fn _init_max(start_value: ScalarDType) -> ScalarDType {
        start_value
    }

    #[inline(always)]
    fn _allow_first_non_nan_update(_start_value: ScalarDType) -> bool {
        false
    }

    #[inline(always)]
    fn _nan_check(v: ScalarDType) -> bool {
        v.is_nan()
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatIgnoreNaN>
where
    ScalarDType: FloatCore,
{
    const _RETURN_AT_NAN: bool = false;

    #[inline(always)]
    fn _init_min(start_value: ScalarDType) -> ScalarDType {
        if start_value.is_nan() {
            ScalarDType::infinity()
        } else {
            start_value
        }
    }

    #[inline(always)]
    fn _init_max(start_value: ScalarDType) -> ScalarDType {
        if start_value.is_nan() {
            ScalarDType::neg_infinity()
        } else {
            start_value
        }
    }

    #[inline(always)]
    fn _allow_first_non_nan_update(start_value: ScalarDType) -> bool {
        start_value.is_nan()
    }

    #[inline(always)]
    fn _nan_check(v: ScalarDType) -> bool {
        v.is_nan()
    }
}

/// ------- Implement the ScalarArgMinMax trait for the different DTypeStrategy -------

macro_rules! impl_scalar {
    ($dtype_strategy:ty, $($dtype:ty),*) => {
        $(
            impl ScalarArgMinMax<$dtype> for SCALAR<$dtype_strategy>
            {
                #[inline(always)]
                fn argminmax(arr: &[$dtype]) -> (usize, usize) {
                    assert!(!arr.is_empty());
                    let mut low_index: usize = 0;
                    let mut high_index: usize = 0;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let start_value: $dtype = unsafe { *arr.get_unchecked(0) };
                    let mut low: $dtype = Self::_init_min(start_value);
                    let mut high: $dtype = Self::_init_max(start_value);
                    let mut first_non_nan_update: bool = Self::_allow_first_non_nan_update(start_value);
                    for i in 0..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return (i, i); // -> return the index
                        }
                        if first_non_nan_update {
                            // If we allow the first non-nan update (only for FloatIgnoreNaN)
                            if !Self::_nan_check(v) {
                                // Update the low and high
                                low = v;
                                low_index = i;
                                high = v;
                                high_index = i;
                                // And disable the first_non_nan_update update
                                first_non_nan_update = false;
                            }
                        } else if v < low {
                            low = v;
                            low_index = i;
                        } else if v > high {
                            high = v;
                            high_index = i;
                        }
                    }
                    (low_index, high_index)
                }

                #[inline(always)]
                fn argmin(arr: &[$dtype]) -> usize {
                    assert!(!arr.is_empty());
                    let mut low_index: usize = 0;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let start_value: $dtype = unsafe { *arr.get_unchecked(0) };
                    let mut low: $dtype = Self::_init_min(start_value);
                    let mut first_non_nan_update: bool = Self::_allow_first_non_nan_update(start_value);
                    for i in 0..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return i; // -> return the index
                        }
                        if first_non_nan_update {
                            // If we allow the first non-nan update (only for FloatIgnoreNaN)
                            if !Self::_nan_check(v) {
                                // Update the low
                                low = v;
                                low_index = i;
                                // And disable the first_non_nan_update update
                                first_non_nan_update = false;
                            }
                        } else if v < low {
                            low = v;
                            low_index = i;
                        }
                    }
                    low_index
                }

                #[inline(always)]
                fn argmax(arr: &[$dtype]) -> usize {
                    assert!(!arr.is_empty());
                    let mut high_index: usize = 0;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let start_value: $dtype = unsafe { *arr.get_unchecked(0) };
                    let mut high: $dtype = Self::_init_max(start_value);
                    let mut first_non_nan_update: bool = Self::_allow_first_non_nan_update(start_value);
                    for i in 0..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return i; // -> return the index
                        }
                        if first_non_nan_update {
                            // If we allow the first non-nan update (only for FloatIgnoreNaN)
                            if !Self::_nan_check(v) {
                                // Update the high
                                high = v;
                                high_index = i;
                                // And disable the first_non_nan_update update
                                first_non_nan_update = false;
                            }
                        } else if v > high {
                            high = v;
                            high_index = i;
                        }
                    }
                    high_index
                }
            }
        )*
    };
}

impl_scalar!(Int, i8, i16, i32, i64, u8, u16, u32, u64);
#[cfg(feature = "float")]
impl_scalar!(FloatReturnNaN, f32, f64);
#[cfg(feature = "float")]
impl_scalar!(FloatIgnoreNaN, f32, f64);

// --- Optional data types

#[cfg(feature = "half")]
use super::scalar_f16::{
    scalar_argmax_f16_ignore_nan, scalar_argmin_f16_ignore_nan, scalar_argminmax_f16_ignore_nan,
};
#[cfg(feature = "half")]
use super::scalar_f16::{
    scalar_argmax_f16_return_nan, scalar_argmin_f16_return_nan, scalar_argminmax_f16_return_nan,
};

#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "half")]
impl ScalarArgMinMax<f16> for SCALAR<FloatReturnNaN> {
    #[inline(always)]
    fn argminmax(arr: &[f16]) -> (usize, usize) {
        scalar_argminmax_f16_return_nan(arr)
    }

    #[inline(always)]
    fn argmin(arr: &[f16]) -> usize {
        scalar_argmin_f16_return_nan(arr)
    }

    #[inline(always)]
    fn argmax(arr: &[f16]) -> usize {
        scalar_argmax_f16_return_nan(arr)
    }
}

#[cfg(feature = "half")]
impl ScalarArgMinMax<f16> for SCALAR<FloatIgnoreNaN> {
    #[inline(always)]
    fn argminmax(arr: &[f16]) -> (usize, usize) {
        scalar_argminmax_f16_ignore_nan(arr)
    }

    #[inline(always)]
    fn argmin(arr: &[f16]) -> usize {
        scalar_argmin_f16_ignore_nan(arr)
    }

    #[inline(always)]
    fn argmax(arr: &[f16]) -> usize {
        scalar_argmax_f16_ignore_nan(arr)
    }
}
