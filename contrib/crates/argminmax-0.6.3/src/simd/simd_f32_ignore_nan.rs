/// Implementation of the argminmax operations for f32 that ignores NaN values.
/// This implementation returns the index of the minimum and maximum values.
/// However, unexpected behavior may occur when there are
/// - *only* NaN values in the array
/// - *only* +/- infinity values in the array
/// - *only* NaN and +/- infinity values in the array
/// In these cases, index 0 is returned.
///
/// NaN values are ignored and treated as if they are not present in the array.
/// To realize this we create an initial SIMD register with values +/- infinity.
/// As comparisons with NaN always return false, it is guaranteed that no NaN values
/// are added to the accumulating SIMD register.
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
use super::generic::{
    impl_SIMDArgMinMax, impl_SIMDInit_FloatIgnoreNaN, SIMDArgMinMax, SIMDInit, SIMDOps,
};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use crate::SCALAR;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use num_traits::Zero;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on f32 data: ignore NaN values
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::super::dtype_strategy::FloatIgnoreNaN;

// https://stackoverflow.com/a/3793950
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MAX_INDEX: usize = 1 << f32::MANTISSA_DIGITS;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx_ignore_nan {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<FloatIgnoreNaN>::LANE_SIZE_32;

    impl SIMDOps<f32, __m256, __m256, LANE_SIZE> for AVX2<FloatIgnoreNaN> {
        const INITIAL_INDEX: __m256 = unsafe {
            std::mem::transmute([
                0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32,
            ])
        };
        const INDEX_INCREMENT: __m256 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m256, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m256 {
            _mm256_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(b, a, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256, b: __m256, mask: __m256) -> __m256 {
            _mm256_blendv_ps(a, b, mask)
        }

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m256 {
            _mm256_set1_ps(a)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f32, __m256, __m256, LANE_SIZE, AVX2<FloatIgnoreNaN>);

    // Requires just AVX (and thus not necessarily AVX2)
    impl_SIMDArgMinMax!(
        f32,
        __m256,
        __m256,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        AVX2<FloatIgnoreNaN>,
        "avx"
    );
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse_ignore_nan {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<FloatIgnoreNaN>::LANE_SIZE_32;

    impl SIMDOps<f32, __m128, __m128, LANE_SIZE> for SSE<FloatIgnoreNaN> {
        const INITIAL_INDEX: __m128 =
            unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };
        const INDEX_INCREMENT: __m128 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m128, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m128 {
            _mm_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128, b: __m128) -> __m128 {
            _mm_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128, b: __m128) -> __m128 {
            _mm_cmpgt_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128, b: __m128) -> __m128 {
            _mm_cmplt_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128, b: __m128, mask: __m128) -> __m128 {
            _mm_blendv_ps(a, b, mask)
        }

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN!

        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m128 {
            _mm_set1_ps(a)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f32, __m128, __m128, LANE_SIZE, SSE<FloatIgnoreNaN>);

    impl_SIMDArgMinMax!(
        f32,
        __m128,
        __m128,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        SSE<FloatIgnoreNaN>,
        "sse4.1"
    );
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
mod avx512_ignore_nan {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<FloatIgnoreNaN>::LANE_SIZE_32;

    impl SIMDOps<f32, __m512, u16, LANE_SIZE> for AVX512<FloatIgnoreNaN> {
        const INITIAL_INDEX: __m512 = unsafe {
            std::mem::transmute([
                0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
                10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32,
            ])
        };
        const INDEX_INCREMENT: __m512 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m512, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m512 {
            _mm512_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512, b: __m512, mask: u16) -> __m512 {
            _mm512_mask_blend_ps(mask, a, b)
        }

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m512 {
            _mm512_set1_ps(a)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f32, __m512, u16, LANE_SIZE, AVX512<FloatIgnoreNaN>);

    impl_SIMDArgMinMax!(
        f32,
        __m512,
        u16,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        AVX512<FloatIgnoreNaN>,
        "avx512f"
    );
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64" // stable for AArch64
))]
mod neon_ignore_nan {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<FloatIgnoreNaN>::LANE_SIZE_32;

    impl SIMDOps<f32, float32x4_t, uint32x4_t, LANE_SIZE> for NEON<FloatIgnoreNaN> {
        const INITIAL_INDEX: float32x4_t =
            unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };
        const INDEX_INCREMENT: float32x4_t =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: float32x4_t) -> [f32; LANE_SIZE] {
            std::mem::transmute::<float32x4_t, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> float32x4_t {
            vld1q_f32(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vaddq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcgtq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcltq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: float32x4_t, b: float32x4_t, mask: uint32x4_t) -> float32x4_t {
            vbslq_f32(mask, b, a)
        }

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> float32x4_t {
            vdupq_n_f32(a)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(
        f32,
        float32x4_t,
        uint32x4_t,
        LANE_SIZE,
        NEON<FloatIgnoreNaN>
    );

    impl_SIMDArgMinMax!(
        f32,
        float32x4_t,
        uint32x4_t,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        NEON<FloatIgnoreNaN>,
        "neon"
    );
}

// ======================================= TESTS =======================================

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
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
    use crate::{FloatIgnoreNaN, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_no_overflow_argminmax,
        test_return_same_result_argminmax,
    };
    // Float specific tests
    use super::super::test_utils::{test_ignore_nans_argminmax, test_return_infs_argminmax};

    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Vec<f32> {
        utils::SampleUniformFullRange::get_random_array(n)
    }

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<FloatIgnoreNaN> = SCALAR {
        _dtype_strategy: PhantomData::<FloatIgnoreNaN>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("avx"))]
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("avx512f")))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[template]
    #[rstest]
    #[case::neon(NEON {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, true)]
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
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
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
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_no_overflow<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_no_overflow_argminmax(get_array_f32, SCALAR_STRATEGY, simd, Some(1 << 25));
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_ignore_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_ignore_nans_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }
}
