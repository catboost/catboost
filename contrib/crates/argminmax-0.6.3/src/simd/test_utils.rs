#[cfg(any(feature = "float", feature = "half"))]
use num_traits::float::FloatCore;
use num_traits::AsPrimitive;
use num_traits::{Bounded, One};

use crate::{SIMDArgMinMax, ScalarArgMinMax};

// ------- Generic tests for argminmax

/// The generic tests check whether the scalar and SIMD function return the same result.

#[cfg(test)]
const LONG_ARR_LEN: usize = 8193; // 8192 + 1

#[cfg(test)]
const NB_RUNS: usize = 10_000;
#[cfg(test)]
const RANDOM_RUN_ARR_LEN: usize = 32 * 4 + 1;

/// Tests whether the scalar and SIMD function return the same result.
/// - tests for a long array of random DType values whether the scalar and SIMD function
///   return the same result.
/// - tests for many arrays of random DType values whether the scalar and SIMD function
///   return the same result.
#[cfg(test)]
pub(crate) fn test_return_same_result_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // 1. Test for a long array
    let data: &[DType] = &get_data(LONG_ARR_LEN);
    assert_eq!(data.len() % 64, 1); // assert that data does not fully fit in a register

    // argminmax
    let (argmin_index, argmax_index) = SCALAR::argminmax(data);
    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(data) };
    // argmin
    let argmin_index_single = SCALAR::argmin(data);
    let argmin_simd_index_single = unsafe { SIMD::argmin(data) };
    // argmax
    let argmax_index_single = SCALAR::argmax(data);
    let argmax_simd_index_single = unsafe { SIMD::argmax(data) };

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // 2. Test for many arrays
    for _ in 0..NB_RUNS {
        let data: &[DType] = &get_data(RANDOM_RUN_ARR_LEN);
        // argminmax
        let (argmin_index, argmax_index) = SCALAR::argminmax(data);
        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(data) };
        // argmin
        let argmin_index_single = SCALAR::argmin(data);
        let argmin_simd_index_single = unsafe { SIMD::argmin(data) };
        // argmax
        let argmax_index_single = SCALAR::argmax(data);
        let argmax_simd_index_single = unsafe { SIMD::argmax(data) };

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);
    }
}

/// Test if the first index is returned when the MIN/MAX value occurs multiple times.
#[cfg(test)]
pub(crate) fn test_first_index_identical_values_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + One + Bounded,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    let mut data: [DType; 64] = [DType::one(); 64]; // multiple of lane size

    // Case 1: all elements are identical
    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 2: all elements are identical except for a couple of MIN/MAX values
    // Add multiple MIN values to the array
    data[5] = DType::min_value();
    data[13] = DType::min_value();
    data[41] = DType::min_value();

    // Add multiple MAX values to the array
    data[7] = DType::max_value();
    data[17] = DType::max_value();
    data[31] = DType::max_value();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 5);
    assert_eq!(argmin_index_single, 5);
    assert_eq!(argmax_index, 7);
    assert_eq!(argmax_index_single, 7);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 5);
    assert_eq!(argmin_simd_index_single, 5);
    assert_eq!(argmax_simd_index, 7);
    assert_eq!(argmax_simd_index_single, 7);
}

// ------- Overflow test

/// Test wheter no overflow occurs when the array is too long.
/// The array length is 2^(size_of(DType) * 8 + 1), unless `arr_len` is specified.
#[cfg(test)]
pub(crate) fn test_no_overflow_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
    arr_len: Option<usize>,
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // left shift 1 by the number of bits in DType + 1
    let shift_size = {
        #[cfg(target_arch = "arm")] // clip shift size to 31 for armv7 (32-bit usize)
        {
            std::cmp::min(std::mem::size_of::<DType>() * 8 + 1, 31)
        }
        #[cfg(not(target_arch = "arm"))]
        {
            std::mem::size_of::<DType>() * 8 + 1 // #bits + 1
        }
    };
    let arr_len = arr_len.unwrap_or(1 << shift_size);
    let data: &[DType] = &get_data(arr_len);

    // argminmax
    let (argmin_index, argmax_index) = SCALAR::argminmax(data);
    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(data) };
    // argmin
    let argmin_index_single = SCALAR::argmin(data);
    let argmin_simd_index_single = unsafe { SIMD::argmin(data) };
    // argmax
    let argmax_index_single = SCALAR::argmax(data);
    let argmax_simd_index_single = unsafe { SIMD::argmax(data) };

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
}

// ------- Float tests for argminmax

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
const FLOAT_ARR_LEN: usize = 1024 + 3;

/// Test whether infinities are handled correctly.
/// -> infinities should be returned as the argmin/argmax
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_return_infs_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    // Case 1: all elements are +inf
    for i in 0..data.len() {
        data[i] = DType::infinity();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 2: all elements are -inf
    for i in 0..data.len() {
        data[i] = DType::neg_infinity();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 3: add some +inf and -inf in the middle
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[100] = DType::infinity();
    data[200] = DType::neg_infinity();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 200);
    assert_eq!(argmin_index_single, 200);
    assert_eq!(argmax_index, 100);
    assert_eq!(argmax_index_single, 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 200);
    assert_eq!(argmin_simd_index_single, 200);
    assert_eq!(argmax_simd_index, 100);
    assert_eq!(argmax_simd_index_single, 100);
}

/// Test whether NaNs are handled correctly - in this case, they should be ignored.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_ignore_nans_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // Case 1: NaN is the first element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert!(argmin_index != 0);
    assert!(argmin_index_single != 0);
    assert!(argmax_index != 0);
    assert!(argmax_index_single != 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert!(argmin_simd_index != 0);
    assert!(argmin_simd_index_single != 0);
    assert!(argmax_simd_index != 0);
    assert!(argmax_simd_index_single != 0);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // Case 1.1 - NaN is the first element, other values are all the same
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();
    for i in 1..data.len() {
        data[i] = DType::from(1.0).unwrap();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 1);
    assert_eq!(argmin_index_single, 1);
    assert_eq!(argmax_index, 1);
    assert_eq!(argmax_index_single, 1);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 1);
    assert_eq!(argmin_simd_index_single, 1);
    assert_eq!(argmax_simd_index, 1);
    assert_eq!(argmax_simd_index_single, 1);

    // Case 1.2 - NaN is the first element, other values are monotonic increasing
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();
    for i in 1..data.len() {
        data[i] = DType::from(i as f64).unwrap();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 1);
    assert_eq!(argmin_index_single, 1);
    assert_eq!(argmax_index, FLOAT_ARR_LEN - 1);
    assert_eq!(argmax_index_single, FLOAT_ARR_LEN - 1);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 1);
    assert_eq!(argmin_simd_index_single, 1);
    assert_eq!(argmax_simd_index, FLOAT_ARR_LEN - 1);
    assert_eq!(argmax_simd_index_single, FLOAT_ARR_LEN - 1);

    // Case 1.3 - NaN is the first element, other values are monotonic decreasing
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();
    for i in 1..data.len() {
        data[i] = DType::from((FLOAT_ARR_LEN - i) as f64).unwrap();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, FLOAT_ARR_LEN - 1);
    assert_eq!(argmin_index_single, FLOAT_ARR_LEN - 1);
    assert_eq!(argmax_index, 1);
    assert_eq!(argmax_index_single, 1);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, FLOAT_ARR_LEN - 1);
    assert_eq!(argmin_simd_index_single, FLOAT_ARR_LEN - 1);
    assert_eq!(argmax_simd_index, 1);
    assert_eq!(argmax_simd_index_single, 1);

    // Case 2: first 100 elements are NaN
    for i in 0..100 {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert!(argmin_index > 99);
    assert!(argmin_index_single > 99);
    assert!(argmax_index > 99);
    assert!(argmax_index_single > 99);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert!(argmin_simd_index > 99);
    assert!(argmin_simd_index_single > 99);
    assert!(argmax_simd_index > 99);
    assert!(argmax_simd_index_single > 99);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // Case 3: NaN is the last element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[FLOAT_ARR_LEN - 1] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert!(argmin_index != 1026);
    assert!(argmin_index_single != 1026);
    assert!(argmax_index != 1026);
    assert!(argmax_index_single != 1026);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert!(argmin_simd_index != 1026);
    assert!(argmin_simd_index_single != 1026);
    assert!(argmax_simd_index != 1026);
    assert!(argmax_simd_index_single != 1026);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // Case 4: last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert!(argmin_index < FLOAT_ARR_LEN - 100);
    assert!(argmin_index_single < FLOAT_ARR_LEN - 100);
    assert!(argmax_index < FLOAT_ARR_LEN - 100);
    assert!(argmax_index_single < FLOAT_ARR_LEN - 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert!(argmin_simd_index < FLOAT_ARR_LEN - 100);
    assert!(argmin_simd_index_single < FLOAT_ARR_LEN - 100);
    assert!(argmax_simd_index < FLOAT_ARR_LEN - 100);
    assert!(argmax_simd_index_single < FLOAT_ARR_LEN - 100);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // Case 5: NaN is somewhere in the middle element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[123] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert!(argmin_index != 123);
    assert!(argmin_index_single != 123);
    assert!(argmax_index != 123);
    assert!(argmax_index_single != 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert!(argmin_simd_index != 123);
    assert!(argmin_simd_index_single != 123);
    assert!(argmax_simd_index != 123);
    assert!(argmax_simd_index_single != 123);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);

    // Case 6: all elements are NaN
    for i in 0..data.len() {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);
}

/// Test whether NaNs are handled correctly - in this case, the index of the first NaN
/// should be returned.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_return_nans_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // Case 1: NaN is the first element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 2: first 100 elements are NaN
    for i in 0..100 {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 3: NaN is the last element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[FLOAT_ARR_LEN - 1] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 1026);
    assert_eq!(argmin_index_single, 1026);
    assert_eq!(argmax_index, 1026);
    assert_eq!(argmax_index_single, 1026);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 1026);
    assert_eq!(argmin_simd_index_single, 1026);
    assert_eq!(argmax_simd_index, 1026);
    assert_eq!(argmax_simd_index_single, 1026);

    // Case 4: last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmin_index_single, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_index_single, FLOAT_ARR_LEN - 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmin_simd_index_single, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_simd_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_simd_index_single, FLOAT_ARR_LEN - 100);

    // Case 5: NaN is somewhere in the middle element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[123] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 123);
    assert_eq!(argmin_index_single, 123);
    assert_eq!(argmax_index, 123);
    assert_eq!(argmax_index_single, 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 123);
    assert_eq!(argmin_simd_index_single, 123);
    assert_eq!(argmax_simd_index, 123);
    assert_eq!(argmax_simd_index_single, 123);

    // Case 6: NaN in the middle of the array and last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 123);
    assert_eq!(argmin_index_single, 123);
    assert_eq!(argmax_index, 123);
    assert_eq!(argmax_index_single, 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 123);
    assert_eq!(argmin_simd_index_single, 123);
    assert_eq!(argmax_simd_index, 123);
    assert_eq!(argmax_simd_index_single, 123);

    // Case 7: all elements are NaN
    for i in 0..data.len() {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 8: array exact multiple of LANE_SIZE and only 1 element is NaN
    let mut data: Vec<DType> = get_data(128);
    data[17] = DType::nan();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 17);
    assert_eq!(argmin_index_single, 17);
    assert_eq!(argmax_index, 17);
    assert_eq!(argmax_index_single, 17);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 17);
    assert_eq!(argmin_simd_index_single, 17);
    assert_eq!(argmax_simd_index, 17);
    assert_eq!(argmax_simd_index_single, 17);
}
