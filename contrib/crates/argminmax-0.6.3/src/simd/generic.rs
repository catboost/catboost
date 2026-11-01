use num_traits::AsPrimitive;

use super::task::*;
use crate::scalar::ScalarArgMinMax;

// ---------------------------------- SIMD operations ----------------------------------

/// Raw SIMD operations
/// These operations are used by the SIMD algorithm and have to be implemented for each
/// data type - SIMD instruction set combination.
/// The operations are implemented in the `simd_*.rs` files.
///
/// Note that for floating point dataypes two implementations are required:
/// - one for the ignore NaN case (uses a floating point SIMDVecDtype)
///   (see the `simd_f*_ignore_nan.rs` files)
/// - one for the return NaN case (uses an integer SIMDVecDtype - as we use the
///   ord_transform to view the floating point data as ordinal integer data).
///   (see the `simd_f*_return_nan.rs` files)
///
#[doc(hidden)]
pub trait SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Integers > this value **cannot** be accurately represented in SIMDVecDtype
    const MAX_INDEX: usize;
    /// Initial index value for the SIMD vector
    const INITIAL_INDEX: SIMDVecDtype;
    /// Increment value for the SIMD vector
    const INDEX_INCREMENT: SIMDVecDtype;

    /// Convert a SIMD register to array
    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    /// Load a SIMD register from memory
    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    /// Add two SIMD registers
    unsafe fn _mm_add(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDVecDtype;

    /// Compare two SIMD registers for greater-than (gt): a > b
    /// Returns a SIMD mask
    unsafe fn _mm_cmpgt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Compare two SIMD registers for less-than (lt): a < b
    unsafe fn _mm_cmplt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Blend two SIMD registers using a SIMD mask (selects elements from a or b)
    unsafe fn _mm_blendv(a: SIMDVecDtype, b: SIMDVecDtype, mask: SIMDMaskDtype) -> SIMDVecDtype;

    /// Horizontal min: get the minimum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_min(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal min
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
        (min_index.as_(), min_value)
    }

    /// Horizontal max: get the maximum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_max(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal max
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
        (max_index.as_(), max_value)
    }

    /// Get the largest multiple of LANE_SIZE that is <= MAX_INDEX
    #[inline(always)]
    fn _get_overflow_lane_size_limit() -> usize {
        Self::MAX_INDEX - Self::MAX_INDEX % LANE_SIZE
    }

    /// ----------------- SIMD operations necessary for ignoring NaNs ------------------

    #[inline(always)]
    unsafe fn _mm_set1(_value: ScalarDType) -> SIMDVecDtype {
        // This is a dummy method that is only used for the ignore NaN case.
        // For the Integer and Float return NaN case, this method is not used.
        unreachable!()
    }
}

/// SIMD initialization operations
/// These operations are used by the SIMD algorithm and have to be implemented for each
/// data type - SIMD instruction set combination.
///
/// These operations are implemented in the `simd_*.rs` files through calling one of the
/// three macros below:
/// - `impl_SIMDInit_Int!`
///     - called in the `simd_i*.rs` files
///     - called in the `simd_u*.rs` files
/// - `impl_SIMDInit_FloatReturnNaN!`
///     - see the `simd_f*_return_nan.rs` files
/// - `impl_SIMDInit_FloatIgnoreNaN!`
///     - see the `simd_f*_ignore_nan.rs` files
///
/// The current (default) implementation is for the Int case - see `impl_SIMDInit_Int!`
/// macro below for more details.
/// For the Float Return NaN case,only the _return_check method is changed - see
/// `impl_SIMDInit_FloatReturnNaN!` macro below for more details.
/// For the Float Ignore NaN case, all the initialization methods are changed - see
/// `impl_SIMDInit_FloatIgnoreNaN!` macro below for more details. Note that for this
/// case, the SIMDOps should implement the `_mm_set1` method.
///
#[doc(hidden)]
pub trait SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    const IGNORE_NAN: bool = false;

    /// Initialization for _core_argminmax

    #[inline(always)]
    unsafe fn _initialize_index_values_low(
        arr_ptr: *const ScalarDType,
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        // Initialize the index and value SIMD registers
        (Self::INITIAL_INDEX, Self::_mm_loadu(arr_ptr))
    }

    #[inline(always)]
    unsafe fn _initialize_index_values_high(
        arr_ptr: *const ScalarDType,
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        // Initialize the index and value SIMD registers
        (Self::INITIAL_INDEX, Self::_mm_loadu(arr_ptr))
    }

    /// Initialization for _overflow_safe_core_argminmax

    #[inline(always)]
    fn _initialize_min_value(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _initialize_max_value(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    /// Checks

    /// Return case for the algorithm
    #[inline(always)]
    fn _return_check(_v: ScalarDType) -> bool {
        false
    }

    /// Check if the value is NaN
    #[inline(always)]
    fn _nan_check(_v: ScalarDType) -> bool {
        false
    }
}

// --------------- Int (signed and unsigned)

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_Int {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use the default implementation
        }
    };
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_Int; // Now classic paths Just Work™

// --------------- Float Return NaNs

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_FloatReturnNaN {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use all initialization methods from the default implementation

            /// Return when a NaN is found
            #[inline(always)]
            fn _return_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }

            #[inline(always)]
            fn _nan_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }
        }
    };
}

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_FloatReturnNaN; // Now classic paths Just Work™

// --------------- Float Ignore NaNs

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_FloatIgnoreNaN {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use the _return_check method from the default implementation

            const IGNORE_NAN: bool = true;

            #[inline(always)]
            unsafe fn _initialize_index_values_low(
                arr_ptr: *const $scalar_dtype,
            ) -> ($simd_vec_dtype, $simd_vec_dtype) {
                // Initialize the index and value SIMD registers
                let new_values = Self::_mm_loadu(arr_ptr);
                let mask_low =
                    Self::_mm_cmplt(new_values, Self::_mm_set1(<$scalar_dtype>::INFINITY));
                let values_low = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::INFINITY),
                    new_values,
                    mask_low,
                );
                let index_low = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::zero()),
                    Self::INITIAL_INDEX,
                    mask_low,
                );
                (index_low, values_low)
            }

            #[inline(always)]
            unsafe fn _initialize_index_values_high(
                arr_ptr: *const $scalar_dtype,
            ) -> ($simd_vec_dtype, $simd_vec_dtype) {
                // Initialize the index and value SIMD registers
                let new_values = Self::_mm_loadu(arr_ptr);
                let mask_high =
                    Self::_mm_cmpgt(new_values, Self::_mm_set1(<$scalar_dtype>::NEG_INFINITY));
                let values_high = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::NEG_INFINITY),
                    new_values,
                    mask_high,
                );
                let index_high = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::zero()),
                    Self::INITIAL_INDEX,
                    mask_high,
                );
                (index_high, values_high)
            }

            #[inline(always)]
            fn _initialize_min_value(_: &[$scalar_dtype]) -> $scalar_dtype {
                <$scalar_dtype>::INFINITY
            }

            #[inline(always)]
            fn _initialize_max_value(_: &[$scalar_dtype]) -> $scalar_dtype {
                <$scalar_dtype>::NEG_INFINITY
            }

            #[inline(always)]
            fn _nan_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }
        }
    };
}

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_FloatIgnoreNaN; // Now classic paths Just Work™

// ---------------------------------- SIMD algorithm -----------------------------------

/// The SIMDCore trait (for all data types).
/// This trait contains the core of the argminmax algorithm.
///
/// This trait is auto-implemented below for all structs - iff the SIMDOps and the
/// SIMDInit traits are implemented for the struct
///
#[doc(hidden)]
pub trait SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
    + SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argminmax` method is overflow safe.
    ///
    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let mut new_index = Self::INITIAL_INDEX; // Index we will increment in the loop
        let (mut index_low, mut values_low) = Self::_initialize_index_values_low(arr_ptr);
        let (mut index_high, mut values_high) = Self::_initialize_index_values_high(arr_ptr);

        // This is (40%-5%) slower than the loop below (depending on the data type)
        // arr.chunks_exact(LANE_SIZE)
        //     .into_iter()
        //     .skip(1)
        //     .for_each(|step| {
        //         new_index = Self::_mm_add(new_index, increment);

        //         let new_values = Self::_mm_loadu(step.as_ptr());

        //         let lt_mask = Self::_mm_cmplt(new_values, values_low);
        //         let gt_mask = Self::_mm_cmpgt(new_values, values_high);

        //         index_low = Self::_mm_blendv(index_low, new_index, lt_mask);
        //         index_high = Self::_mm_blendv(index_high, new_index, gt_mask);

        //         values_low = Self::_mm_blendv(values_low, new_values, lt_mask);
        //         values_high = Self::_mm_blendv(values_high, new_values, gt_mask);
        //     });

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask_low = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask_low);
            index_low = Self::_mm_blendv(index_low, new_index, mask_low);

            // Update the highest values and index
            let mask_high = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask_high);
            index_high = Self::_mm_blendv(index_high, new_index, mask_high);
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

    /// Core argmin algorithm - returns (argmin, min)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argmin` method is overflow safe.
    ///
    #[inline(always)]
    unsafe fn _core_argmin(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let mut new_index = Self::INITIAL_INDEX; // Index we will increment in the loop
        let (mut index_low, mut values_low) = Self::_initialize_index_values_low(arr_ptr);

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask_low = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask_low);
            index_low = Self::_mm_blendv(index_low, new_index, mask_low);
        }

        // Get the min index and corresponding value from the SIMD vectors and return
        Self::_horiz_min(index_low, values_low)
    }

    /// Core argmax algorithm - returns (argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argmax` method is overflow safe.
    ///
    #[inline(always)]
    unsafe fn _core_argmax(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let mut new_index = Self::INITIAL_INDEX; // Index we will increment in the loop
        let (mut index_high, mut values_high) = Self::_initialize_index_values_high(arr_ptr);

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the highest values and index
            let mask_high = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask_high);
            index_high = Self::_mm_blendv(index_high, new_index, mask_high);
        }

        // Get the max index and corresponding value from the SIMD vectors and return
        Self::_horiz_max(index_high, values_high)
    }

    /// Overflow-safe core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: &[ScalarDType],
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        // let n_loops = (arr.len() + dtype_max - 1) / dtype_max; // ceil division
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = Self::_initialize_min_value(arr);
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = Self::_initialize_max_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(min_value) || Self::_return_check(max_value) {
                // We can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..start + dtype_max]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(min_value) || Self::_return_check(max_value) {
                // We can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value, max_index, max_value)
    }

    /// Overflow-safe core argmin algorithm - returns (argmin, min)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argmin(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = Self::_initialize_min_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(min_value) {
                // We can return immediately
                return (min_index, min_value);
            }
            let (min_index_, min_value_) = Self::_core_argmin(&arr[start..start + dtype_max]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(min_value) {
                // We can return immediately
                return (min_index, min_value);
            }
            let (min_index_, min_value_) = Self::_core_argmin(&arr[start..]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value)
    }

    /// Overflow-safe core argmax algorithm - returns (argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argmax(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = Self::_initialize_max_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(max_value) {
                // We can return immediately
                return (max_index, max_value);
            }
            let (max_index_, max_value_) = Self::_core_argmax(&arr[start..start + dtype_max]);
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(max_value) {
                // We can return immediately
                return (max_index, max_value);
            }
            let (max_index_, max_value_) = Self::_core_argmax(&arr[start..]);
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (max_index, max_value)
    }
}

/// Implement SIMDCore where SIMDOps & SIMDInit are implemented
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> for T
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    T: SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
        + SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>,
{
    // Implement the SIMDCore trait
}

// -------------------------------- ArgMinMax SIMD TRAIT -------------------------------

/// A trait providing the SIMD implementation of the argminmax operations.
///
// This trait its methods should be implemented for all structs that implement `SIMDOps`
// for the same generics.
// This trait is implemented in the `simd_*.rs` files calling the `impl_SIMDArgMinMax!`
// macro. With the exception of the `simd_f*_return_nan.rs` files, which implement this
// trait themselves (as these return .argminmax().0 and .argminmax().1 respectively
// instead of .argmin() and .argmax()).
//
pub trait SIMDArgMinMax<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize, SCALAR>:
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    SCALAR: ScalarArgMinMax<ScalarDType>,
{
    /// Get the index of the minimum and maximum values in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the slice
    /// `(min_index, max_index)`.
    ///
    /// # Safety
    /// This function is unsafe because unsafe SIMD operations are used.  
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    // Is necessary to have a separate function for this so we can call it in the
    // argminmax function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argminmax(data: &[ScalarDType]) -> (usize, usize)
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argminmax, // SIMD operation
            SCALAR::argminmax,                   // Scalar operation
            Self::_nan_check,                    // NaN check - true if value is NaN
            Self::IGNORE_NAN,                    // Ignore NaNs - if false -> return NaN
        )
    }

    /// Get the index of the minimum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the minimum value in the slice.
    ///
    /// # Safety
    /// This function is unsafe because unsafe SIMD operations are used.  
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argmin(data: &[ScalarDType]) -> usize;

    // Is necessary to have a separate function for this so we can call it in the
    // argmin function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argmin(data: &[ScalarDType]) -> usize
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argmin_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argmin, // SIMD operation
            SCALAR::argmin,                   // Scalar operation
            Self::_nan_check,                 // NaN check - true if value is NaN
            Self::IGNORE_NAN,                 // Ignore NaNs - if false -> return NaN
        )
    }

    /// Get the index of the maximum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the maximum value in the slice.
    ///
    /// # Safety
    /// This function is unsafe because unsafe SIMD operations are used.  
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argmax(data: &[ScalarDType]) -> usize;

    // Is necessary to have a separate function for this so we can call it in the
    // argmax function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argmax(data: &[ScalarDType]) -> usize
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argmax, // SIMD operation
            SCALAR::argmax,                   // Scalar operation
            Self::_nan_check,                 // NaN check - true if value is NaN
            Self::IGNORE_NAN,                 // Ignore NaNs - if false -> return NaN
        )
    }
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDArgMinMax {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $scalar_struct:ty, $simd_struct:ty, $target:expr) => {
        impl
            SIMDArgMinMax<
                $scalar_dtype,
                $simd_vec_dtype,
                $simd_mask_dtype,
                $lane_size,
                $scalar_struct,
            > for $simd_struct
        {
            #[target_feature(enable = $target)]
            unsafe fn argminmax(data: &[$scalar_dtype]) -> (usize, usize) {
                Self::_argminmax(data)
            }

            #[target_feature(enable = $target)]
            unsafe fn argmin(data: &[$scalar_dtype]) -> usize {
                Self::_argmin(data)
                // TODO: test if this is same speed as _argmin
                // Self::_argminmax(data).0
            }

            #[target_feature(enable = $target)]
            unsafe fn argmax(data: &[$scalar_dtype]) -> usize {
                Self::_argmax(data)
                // Self::_argminmax(data).1
            }
        }
    };
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDArgMinMax; // Now classic paths Just Work™

// --------------------------------- Unimplement Macros --------------------------------

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDOps {
    ($scalar_type:ty, $reg:ty, $simd_struct:ty) => {
        impl SIMDOps<$scalar_type, $reg, $reg, 0> for $simd_struct {
            const INITIAL_INDEX: $reg = 0;
            const INDEX_INCREMENT: $reg = 0;
            const MAX_INDEX: usize = 0;

            unsafe fn _reg_to_arr(_reg: $reg) -> [$scalar_type; 0] {
                unimplemented!()
            }

            unsafe fn _mm_loadu(_data: *const $scalar_type) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_add(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmpgt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmplt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_blendv(_a: $reg, _b: $reg, _mask: $reg) -> $reg {
                unimplemented!()
            }
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDInit {
    ($scalar_type:ty, $reg:ty, $simd_struct:ty) => {
        impl SIMDInit<$scalar_type, $reg, $reg, 0> for $simd_struct {
            // Use the default implementation
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDArgMinMax {
    ($scalar_type:ty, $reg:ty, $scalar:ty, $simd_struct:ty) => {
        impl SIMDArgMinMax<$scalar_type, $reg, $reg, 0, $scalar> for $simd_struct {
            unsafe fn argminmax(_data: &[$scalar_type]) -> (usize, usize) {
                unimplemented!()
            }

            unsafe fn argmin(_data: &[$scalar_type]) -> usize {
                unimplemented!()
            }

            unsafe fn argmax(_data: &[$scalar_type]) -> usize {
                unimplemented!()
            }
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDArgMinMax; // Now classic paths Just Work™

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDInit; // Now classic paths Just Work™

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDOps; // Now classic paths Just Work™
