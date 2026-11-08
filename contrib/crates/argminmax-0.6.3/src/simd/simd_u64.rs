/// Implementation of the argminmax operations for u64.
/// As there are no SIMD instructions for uints (on x86 & x86_64) we transform the u64
/// values to i64 ordinal values:
///     ord_i64 = v ^ -0x8000000000000000
///
/// This transformation is a bijection, i.e. it is reversible:
///     v = ord_i64 ^ -0x8000000000000000
///
/// Through this transformation we can perform the argminmax operations using SIMD on
/// the ordinal integer values and then transform the result back to the original u64
/// values.
///

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use super::config::SIMDInstructionSet;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use super::generic::{impl_SIMDArgMinMax, impl_SIMDInit_Int};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::generic::{SIMDArgMinMax, SIMDInit, SIMDOps};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use crate::SCALAR;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on u64 data: (default) Int
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::super::dtype_strategy::Int;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::task::{max_index_value, min_index_value};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const XOR_VALUE: i64 = -0x8000000000000000; // i64::MIN

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unnecessary_transmutes)]
fn _i64ord_to_u64(ord_i64: i64) -> u64 {
    // let v = ord_i64 ^ -0x8000000000000000;
    unsafe { std::mem::transmute::<i64, u64>(ord_i64 ^ XOR_VALUE) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64",))]
const MAX_INDEX: usize = i64::MAX as usize; // SIMD operations on signed ints
#[cfg(target_arch = "aarch64")]
const MAX_INDEX: usize = u64::MAX as usize; // SIMD operations on unsigned ints

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<Int>::LANE_SIZE_64;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u64_as_m256i_to_i64ord(u64_as_m256i: __m256i) -> __m256i {
        // on a scalar: v ^ -0x8000000000000000
        // transforms to monotonically increasing order
        _mm256_xor_si256(u64_as_m256i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m256i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u64, __m256i, __m256i, LANE_SIZE> for AVX2<Int> {
        const INITIAL_INDEX: __m256i = unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [u64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m256i {
            _u64_as_m256i_to_i64ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_u64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_u64(max_value))
        }
    }

    impl_SIMDInit_Int!(u64, __m256i, __m256i, LANE_SIZE, AVX2<Int>);

    impl_SIMDArgMinMax!(
        u64,
        __m256i,
        __m256i,
        LANE_SIZE,
        SCALAR<Int>,
        AVX2<Int>,
        "avx2"
    );
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<Int>::LANE_SIZE_64;
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u64_as_m128i_to_i64ord(u64_as_m128i: __m128i) -> __m128i {
        // on a scalar: v ^ -0x8000000000000000
        // transforms to monotonically increasing order
        _mm_xor_si128(u64_as_m128i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m128i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u64, __m128i, __m128i, LANE_SIZE> for SSE<Int> {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i64, 1i64]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [u64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m128i {
            _u64_as_m128i_to_i64ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_u64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_u64(max_value))
        }
    }

    impl_SIMDInit_Int!(u64, __m128i, __m128i, LANE_SIZE, SSE<Int>);

    impl_SIMDArgMinMax!(
        u64,
        __m128i,
        __m128i,
        LANE_SIZE,
        SCALAR<Int>,
        SSE<Int>,
        "sse4.2"
    );
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<Int>::LANE_SIZE_64;
    const XOR_MASK: __m512i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u64_as_m512i_to_i64ord(u64_as_m512i: __m512i) -> __m512i {
        // on a scalar: v ^ -0x8000000000000000
        // transforms to monotonically increasing order
        _mm512_xor_si512(u64_as_m512i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m512i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u64, __m512i, u8, LANE_SIZE> for AVX512<Int> {
        const INITIAL_INDEX: __m512i =
            unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64]) };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [u64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m512i {
            _u64_as_m512i_to_i64ord(_mm512_loadu_epi64(data as *const i64))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmpgt_epi64_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmplt_epi64_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u8) -> __m512i {
            _mm512_mask_blend_epi64(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_u64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, u64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_u64(max_value))
        }
    }

    impl_SIMDInit_Int!(u64, __m512i, u8, LANE_SIZE, AVX512<Int>);

    impl_SIMDArgMinMax!(
        u64,
        __m512i,
        u8,
        LANE_SIZE,
        SCALAR<Int>,
        AVX512<Int>,
        "avx512f"
    );
}

// --------------------------------------- NEON ----------------------------------------

// There are NEON SIMD intrinsics for u64, but
//  - for arm we miss the vcgt_ and vclt_ intrinsics.
//  - for aarch64 the required intrinsics are present (on nightly)

#[cfg(target_arch = "arm")]
#[cfg(feature = "nightly_simd")]
mod neon {
    use super::super::config::NEON;
    use super::super::generic::{unimpl_SIMDArgMinMax, unimpl_SIMDInit, unimpl_SIMDOps};
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimpl_SIMDOps!(u64, usize, NEON<Int>);
    unimpl_SIMDInit!(u64, usize, NEON<Int>);
    unimpl_SIMDArgMinMax!(u64, usize, SCALAR<Int>, NEON<Int>);
}

#[cfg(target_arch = "aarch64")] // stable for AArch64
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<Int>::LANE_SIZE_64;

    impl SIMDOps<u64, uint64x2_t, uint64x2_t, LANE_SIZE> for NEON<Int> {
        const INITIAL_INDEX: uint64x2_t = unsafe { std::mem::transmute([0u64, 1u64]) };
        const INDEX_INCREMENT: uint64x2_t =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: uint64x2_t) -> [u64; LANE_SIZE] {
            std::mem::transmute::<uint64x2_t, [u64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> uint64x2_t {
            vld1q_u64(data)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
            vaddq_u64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
            vcgtq_u64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
            vcltq_u64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: uint64x2_t, b: uint64x2_t, mask: uint64x2_t) -> uint64x2_t {
            vbslq_u64(mask, b, a)
        }
    }

    impl_SIMDInit_Int!(u64, uint64x2_t, uint64x2_t, LANE_SIZE, NEON<Int>);

    impl_SIMDArgMinMax!(
        u64,
        uint64x2_t,
        uint64x2_t,
        LANE_SIZE,
        SCALAR<Int>,
        NEON<Int>,
        "neon"
    );
}

// ======================================= TESTS =======================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use std::marker::PhantomData;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly_simd")]
    use crate::simd::config::AVX512;
    #[cfg(target_arch = "aarch64")]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, SSE};
    use crate::{Int, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_return_same_result_argminmax,
    };

    use dev_utils::utils;

    fn get_array_u64(n: usize) -> Vec<u64> {
        utils::SampleUniformFullRange::get_random_array(n)
    }

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<Int> = SCALAR {
        _dtype_strategy: PhantomData::<Int>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("sse4.2"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("avx2"))]
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("avx512f")))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // --------------- Template for AArch64 ---------------

    #[cfg(target_arch = "aarch64")]
    #[template]
    #[rstest]
    #[case::neon(NEON {_dtype_strategy: PhantomData::<Int>}, true)]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ----------------- The actual tests -----------------

    #[apply(simd_implementations)]
    fn test_first_index_is_returned_when_identical_values_found<
        T,
        SIMDV,
        SIMDM,
        const LANE_SIZE: usize,
    >(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<u64, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_first_index_identical_values_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<u64, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_u64, SCALAR_STRATEGY, simd);
    }
}
