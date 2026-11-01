/// Implementation of the argminmax operations for u32.
/// As there are no SIMD instructions for uints (on x86 & x86_64) we transform the u32
/// values to i32 ordinal values:
///     ord_i32 = v ^ -0x80000000
///
/// This transformation is a bijection, i.e. it is reversible:
///     v = ord_i32 ^ -0x80000000
///
/// Through this transformation we can perform the argminmax operations using SIMD on
/// the ordinal integer values and then transform the result back to the original u32
/// values.
///
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::config::SIMDInstructionSet;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::generic::{impl_SIMDArgMinMax, impl_SIMDInit_Int, SIMDArgMinMax, SIMDInit, SIMDOps};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use crate::SCALAR;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on u32 data: (default) Int
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
const XOR_VALUE: i32 = -0x80000000; // i32::MIN

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unnecessary_transmutes)]
fn _i32ord_to_u32(ord_i32: i32) -> u32 {
    // let v = ord_i32 ^ -0x80000000;
    unsafe { std::mem::transmute::<i32, u32>(ord_i32 ^ XOR_VALUE) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MAX_INDEX: usize = i32::MAX as usize; // SIMD operations on signed ints
#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
const MAX_INDEX: usize = u8::MAX as usize; // SIMD operations on unsigned ints

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<Int>::LANE_SIZE_32;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u32_as_m256i_to_i32ord(u32_as_m256i: __m256i) -> __m256i {
        // on a scalar: v ^ -0x80000000
        // transforms to monotonically increasing order
        _mm256_xor_si256(u32_as_m256i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m256i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u32, __m256i, __m256i, LANE_SIZE> for AVX2<Int> {
        const INITIAL_INDEX: __m256i =
            unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [u32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u32) -> __m256i {
            _u32_as_m256i_to_i32ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_u32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_u32(max_value))
        }
    }

    impl_SIMDInit_Int!(u32, __m256i, __m256i, LANE_SIZE, AVX2<Int>);

    impl_SIMDArgMinMax!(
        u32,
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

    const LANE_SIZE: usize = SSE::<Int>::LANE_SIZE_32;
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u32_as_m128i_to_i32ord(u32_as_m128i: __m128i) -> __m128i {
        // on a scalar: v ^ -0x80000000
        // transforms to monotonically increasing order
        _mm_xor_si128(u32_as_m128i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m128i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u32, __m128i, __m128i, LANE_SIZE> for SSE<Int> {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [u32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u32) -> __m128i {
            _u32_as_m128i_to_i32ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_u32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_u32(max_value))
        }
    }

    impl_SIMDInit_Int!(u32, __m128i, __m128i, LANE_SIZE, SSE<Int>);

    impl_SIMDArgMinMax!(
        u32,
        __m128i,
        __m128i,
        LANE_SIZE,
        SCALAR<Int>,
        SSE<Int>,
        "sse4.1"
    );
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<Int>::LANE_SIZE_32;
    const XOR_MASK: __m512i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u32_as_m512i_to_i32ord(u32_as_m512i: __m512i) -> __m512i {
        // on scalar: v ^ -0x80000000
        // transforms to monotonically increasing order
        _mm512_xor_si512(u32_as_m512i, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m512i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<u32, __m512i, u16, LANE_SIZE> for AVX512<Int> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32, 9i32, 10i32, 11i32, 12i32,
                13i32, 14i32, 15i32,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [u32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to signed integers.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u32) -> __m512i {
            _u32_as_m512i_to_i32ord(_mm512_loadu_epi32(data as *const i32))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmpgt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmplt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u16) -> __m512i {
            _mm512_mask_blend_epi32(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_u32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, u32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_u32(max_value))
        }
    }

    impl_SIMDInit_Int!(u32, __m512i, u16, LANE_SIZE, AVX512<Int>);

    impl_SIMDArgMinMax!(
        u32,
        __m512i,
        u16,
        LANE_SIZE,
        SCALAR<Int>,
        AVX512<Int>,
        "avx512f"
    );
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64" // stable for AArch64
))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<Int>::LANE_SIZE_32;

    impl SIMDOps<u32, uint32x4_t, uint32x4_t, LANE_SIZE> for NEON<Int> {
        const INITIAL_INDEX: uint32x4_t = unsafe { std::mem::transmute([0u32, 1u32, 2u32, 3u32]) };
        const INDEX_INCREMENT: uint32x4_t =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: uint32x4_t) -> [u32; LANE_SIZE] {
            std::mem::transmute::<uint32x4_t, [u32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u32) -> uint32x4_t {
            vld1q_u32(data)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            vaddq_u32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            vcgtq_u32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            vcltq_u32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: uint32x4_t, b: uint32x4_t, mask: uint32x4_t) -> uint32x4_t {
            vbslq_u32(mask, b, a)
        }
    }

    impl_SIMDInit_Int!(u32, uint32x4_t, uint32x4_t, LANE_SIZE, NEON<Int>);

    impl_SIMDArgMinMax!(
        u32,
        uint32x4_t,
        uint32x4_t,
        LANE_SIZE,
        SCALAR<Int>,
        NEON<Int>,
        "neon"
    );
}

// ======================================= TESTS =======================================

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use std::marker::PhantomData;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly_simd")]
    use crate::simd::config::AVX512;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, SSE};
    use crate::{Int, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_return_same_result_argminmax,
    };

    use dev_utils::utils;

    fn get_array_u32(n: usize) -> Vec<u32> {
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
    #[case::sse(SSE {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("avx2"))]
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("avx512f")))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
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
        T: SIMDArgMinMax<u32, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
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
        T: SIMDArgMinMax<u32, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_u32, SCALAR_STRATEGY, simd);
    }
}
