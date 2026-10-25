/// Implementation of the scalar argminmax operations for f16.
///
/// As f16 is not hardware supported on most x86 CPUs, we aim to facilitate efficient
/// implementation of argminmax operations on f16 arrays through transforming the f16
/// values to i16ord. (more details in simd/simd_f16_return_nan.rs)
///
use half::f16;

#[inline(always)]
fn f16_to_i16ord(x: f16) -> i16 {
    let x = unsafe { std::mem::transmute::<f16, i16>(x) };
    ((x >> 15) & 0x7FFF) ^ x
}

// ------- Float Return NaN -------

// TODO: commented this (see the TODO below)
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// #[inline(never)]
pub(crate) fn scalar_argminmax_f16_return_nan(arr: &[f16]) -> (usize, usize) {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = f16_to_i16ord(unsafe { *arr.get_unchecked(low_index) });
    let mut high: i16 = f16_to_i16ord(unsafe { *arr.get_unchecked(high_index) });
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return (i, i);
        }
        let v: i16 = f16_to_i16ord(v);
        if v < low {
            low = v;
            low_index = i;
        } else if v > high {
            high = v;
            high_index = i;
        }
    }
    (low_index, high_index)
}

pub(crate) fn scalar_argmin_f16_return_nan(arr: &[f16]) -> usize {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = f16_to_i16ord(unsafe { *arr.get_unchecked(low_index) });
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return i;
        }
        let v: i16 = f16_to_i16ord(v);
        if v < low {
            low = v;
            low_index = i;
        }
    }
    low_index
}

pub(crate) fn scalar_argmax_f16_return_nan(arr: &[f16]) -> usize {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut high: i16 = f16_to_i16ord(unsafe { *arr.get_unchecked(high_index) });
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return i;
        }
        let v: i16 = f16_to_i16ord(v);
        if v > high {
            high = v;
            high_index = i;
        }
    }
    high_index
}

pub(crate) fn scalar_argminmax_f16_ignore_nan(arr: &[f16]) -> (usize, usize) {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = f16_to_i16ord(f16::INFINITY);
    let mut high: i16 = f16_to_i16ord(f16::NEG_INFINITY);
    let mut first_non_nan_update = true;
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = f16_to_i16ord(v);
            if first_non_nan_update {
                low = v;
                high = v;
                low_index = i;
                high_index = i;
                first_non_nan_update = false;
            } else if v < low {
                low = v;
                low_index = i;
            } else if v > high {
                high = v;
                high_index = i;
            }
        }
    }
    (low_index, high_index)
}

pub(crate) fn scalar_argmin_f16_ignore_nan(arr: &[f16]) -> usize {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = f16_to_i16ord(f16::INFINITY);
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = f16_to_i16ord(v);
            if v < low {
                low = v;
                low_index = i;
            }
        }
    }
    low_index
}

pub(crate) fn scalar_argmax_f16_ignore_nan(arr: &[f16]) -> usize {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut high: i16 = f16_to_i16ord(f16::NEG_INFINITY);
    for i in 0..arr.len() {
        let v: f16 = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = f16_to_i16ord(v);
            if v > high {
                high = v;
                high_index = i;
            }
        }
    }
    high_index
}

// TODO: previously we had dedicated non x86_64 code for f16 (see below)

// #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
// // #[inline(never)]
// pub(crate) fn scalar_argminmax_f16(arr: &[f16]) -> (usize, usize) {
//     // f16 is transformed to i16ord
//     //   benchmarks  show:
//     //     1. this is 7-10x faster than using raw f16
//     //     2. this is 3x faster than transforming to f32 or f64
//     assert!(!arr.is_empty());
//     // This is 3% slower on x86_64, but 12% faster on aarch64.
//     let minmax_tuple: (usize, i16, usize, i16) = arr.iter().enumerate().fold(
//         (0, f16_to_i16ord(arr[0]), 0, f16_to_i16ord(arr[0])),
//         |(low_index, low, high_index, high), (i, item)| {
//             if item.is_nan() {
//                 // Return the index of the first NaN value
//                 return (i, i);
//             }
//             let item = f16_to_i16ord(*item);
//             if item < low {
//                 (i, item, high_index, high)
//             } else if item > high {
//                 (low_index, low, i, item)
//             } else {
//                 (low_index, low, high_index, high)
//             }
//         },
//     );
//     (minmax_tuple.0, minmax_tuple.2)
// }

// ======================================= TESTS =======================================

#[cfg(all(feature = "float", feature = "half"))]
#[cfg(test)]
mod tests {
    use super::{
        scalar_argmax_f16_ignore_nan, scalar_argmin_f16_ignore_nan, scalar_argminmax_f16_ignore_nan,
    };
    use super::{
        scalar_argmax_f16_return_nan, scalar_argmin_f16_return_nan, scalar_argminmax_f16_return_nan,
    };
    use crate::{FloatIgnoreNaN, FloatReturnNaN, ScalarArgMinMax, SCALAR};

    use half::f16;

    use dev_utils::utils;

    const ARR_LEN: usize = 1025;

    fn get_arrays(len: usize) -> (Vec<f32>, Vec<f16>) {
        let vec_f16: Vec<f16> = utils::SampleUniformFullRange::get_random_array(len);
        let vec_f32: Vec<f32> = vec_f16.iter().map(|x| x.to_f32()).collect();
        (vec_f32, vec_f16)
    }

    #[test]
    fn test_generic_and_specific_impl_return_the_same_results() {
        for _ in 0..100 {
            let (vec_f32, vec_f16) = get_arrays(ARR_LEN);
            let data_f32: &[f32] = &vec_f32;
            let data_f16: &[f16] = &vec_f16;
            // Return NaN
            let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(data_f32);
            let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_return_nan(data_f16);
            let argmin_index_f16_single = scalar_argmin_f16_return_nan(data_f16);
            let argmax_index_f16_single = scalar_argmax_f16_return_nan(data_f16);
            assert_eq!(argmin_index, argmin_index_f16);
            assert_eq!(argmax_index, argmax_index_f16_single);
            assert_eq!(argmax_index, argmax_index_f16);
            assert_eq!(argmin_index, argmin_index_f16_single);
            // Ignore NaN
            let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(data_f32);
            let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_ignore_nan(data_f16);
            let argmin_index_f16_single = scalar_argmin_f16_ignore_nan(data_f16);
            let argmax_index_f16_single = scalar_argmax_f16_ignore_nan(data_f16);
            assert_eq!(argmin_index, argmin_index_f16);
            assert_eq!(argmin_index, argmin_index_f16_single);
            assert_eq!(argmax_index, argmax_index_f16);
            assert_eq!(argmax_index, argmax_index_f16_single);
        }
    }

    #[test]
    fn test_generic_and_specific_impl_return_nans() {
        // first, middle, last element
        let nan_pos: [usize; 3] = [0, ARR_LEN / 2, ARR_LEN - 1];
        for pos in nan_pos.iter() {
            let (vec_f32, vec_f16) = get_arrays(ARR_LEN);
            let mut data_f32: Vec<f32> = vec_f32;
            let mut data_f16: Vec<f16> = vec_f16;
            data_f32[*pos] = f32::NAN;
            data_f16[*pos] = f16::NAN;
            let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(&data_f32);
            let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_return_nan(&data_f16);
            let argmin_index_f16_single = scalar_argmin_f16_return_nan(&data_f16);
            let argmax_index_f16_single = scalar_argmax_f16_return_nan(&data_f16);
            assert_eq!(argmin_index, argmin_index_f16);
            assert_eq!(argmin_index, argmin_index_f16_single);
            assert_eq!(argmax_index, argmax_index_f16);
            assert_eq!(argmax_index, argmax_index_f16_single);
        }

        // all elements are NaN
        let (mut vec_f32, mut vec_f16) = get_arrays(ARR_LEN);
        vec_f32.iter_mut().for_each(|x| *x = f32::NAN);
        vec_f16.iter_mut().for_each(|x| *x = f16::NAN);
        let data_f32: &[f32] = &vec_f32;
        let data_f16: &[f16] = &vec_f16;
        let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(data_f32);
        let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_return_nan(data_f16);
        let argmin_index_f16_single = scalar_argmin_f16_return_nan(data_f16);
        let argmax_index_f16_single = scalar_argmax_f16_return_nan(data_f16);
        assert_eq!(argmin_index, argmin_index_f16);
        assert_eq!(argmin_index, argmin_index_f16_single);
        assert_eq!(argmax_index, argmax_index_f16);
        assert_eq!(argmax_index, argmax_index_f16_single);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmax_index, 0);
    }

    #[test]
    fn test_generic_and_specific_impl_ignore_nans() {
        // first, middle, last element
        let nan_pos: [usize; 3] = [0, ARR_LEN / 2, ARR_LEN - 1];
        for pos in nan_pos.iter() {
            let (vec_f32, vec_f16) = get_arrays(ARR_LEN);
            let mut data_f32: Vec<f32> = vec_f32;
            let mut data_f16: Vec<f16> = vec_f16;
            data_f32[*pos] = f32::NAN;
            data_f16[*pos] = f16::NAN;
            let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(&data_f32);
            let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_ignore_nan(&data_f16);
            let argmin_index_f16_single = scalar_argmin_f16_ignore_nan(&data_f16);
            let argmax_index_f16_single = scalar_argmax_f16_ignore_nan(&data_f16);
            assert_eq!(argmin_index, argmin_index_f16);
            assert_eq!(argmin_index, argmin_index_f16_single);
            assert_eq!(argmax_index, argmax_index_f16);
            assert_eq!(argmax_index, argmax_index_f16_single);
        }

        // all elements are NaN
        let (mut vec_f32, mut vec_f16) = get_arrays(ARR_LEN);
        vec_f32.iter_mut().for_each(|x| *x = f32::NAN);
        vec_f16.iter_mut().for_each(|x| *x = f16::NAN);
        let data_f32: &[f32] = &vec_f32;
        let data_f16: &[f16] = &vec_f16;
        let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(data_f32);
        let (argmin_index_f16, argmax_index_f16) = scalar_argminmax_f16_ignore_nan(data_f16);
        let argmin_index_f16_single = scalar_argmin_f16_ignore_nan(data_f16);
        let argmax_index_f16_single = scalar_argmax_f16_ignore_nan(data_f16);
        assert_eq!(argmin_index, argmin_index_f16);
        assert_eq!(argmin_index, argmin_index_f16_single);
        assert_eq!(argmax_index, argmax_index_f16);
        assert_eq!(argmax_index, argmax_index_f16_single);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmax_index, 0);
    }
}
