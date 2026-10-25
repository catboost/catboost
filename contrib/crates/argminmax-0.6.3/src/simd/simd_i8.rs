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

/// The dtype-strategy for performing operations on i8 data: (default) Int
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::super::dtype_strategy::Int;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MAX_INDEX: usize = i8::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<Int>::LANE_SIZE_8;

    impl SIMDOps<i8, __m256i, __m256i, LANE_SIZE> for AVX2<Int> {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8, 16i8, 17i8, 18i8, 19i8, 20i8, 21i8, 22i8, 23i8, 24i8, 25i8, 26i8, 27i8, 28i8,
                29i8, 30i8, 31i8,
            ])
        };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi8(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m256i = value;
            vmin = _mm256_min_epi8(vmin, _mm256_permute2x128_si256(vmin, vmin, 1));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 8));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 4));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 2));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm256_extract_epi8(vmin, 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm256_cmpeq_epi8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi8(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 2));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm256_extract_epi8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m256i = value;
            vmax = _mm256_max_epi8(vmax, _mm256_permute2x128_si256(vmax, vmax, 1));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm256_extract_epi8(vmax, 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm256_cmpeq_epi8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi8(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 2));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm256_extract_epi8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    impl_SIMDInit_Int!(i8, __m256i, __m256i, LANE_SIZE, AVX2<Int>);

    impl_SIMDArgMinMax!(
        i8,
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

    const LANE_SIZE: usize = SSE::<Int>::LANE_SIZE_8;

    impl SIMDOps<i8, __m128i, __m128i, LANE_SIZE> for SSE<Int> {
        const INITIAL_INDEX: __m128i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8,
            ])
        };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m128i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m128i = value;
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 8));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 4));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 2));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm_extract_epi8(vmin, 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm_cmpeq_epi8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                  // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 2));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm_extract_epi8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m128i = value;
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 8));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 4));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 2));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm_extract_epi8(vmax, 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm_cmpeq_epi8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                  // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 2));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm_extract_epi8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    impl_SIMDInit_Int!(i8, __m128i, __m128i, LANE_SIZE, SSE<Int>);

    impl_SIMDArgMinMax!(
        i8,
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

    const LANE_SIZE: usize = AVX512::<Int>::LANE_SIZE_8;

    impl SIMDOps<i8, __m512i, u64, LANE_SIZE> for AVX512<Int> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8, 16i8, 17i8, 18i8, 19i8, 20i8, 21i8, 22i8, 23i8, 24i8, 25i8, 26i8, 27i8, 28i8,
                29i8, 30i8, 31i8, 32i8, 33i8, 34i8, 35i8, 36i8, 37i8, 38i8, 39i8, 40i8, 41i8, 42i8,
                43i8, 44i8, 45i8, 46i8, 47i8, 48i8, 49i8, 50i8, 51i8, 52i8, 53i8, 54i8, 55i8, 56i8,
                57i8, 58i8, 59i8, 60i8, 61i8, 62i8, 63i8,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m512i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m512i {
            _mm512_loadu_epi8(data as *const i8)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u64 {
            _mm512_cmpgt_epi8_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u64 {
            _mm512_cmplt_epi8_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u64) -> __m512i {
            _mm512_mask_blend_epi8(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m512i = value;
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi32(vmin, vmin, 8));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi32(vmin, vmin, 4));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 8));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 4));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 2));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm_extract_epi8(_mm512_castsi512_si128(vmin), 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm512_cmpeq_epi8_mask(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi8(
                mask,
                _mm512_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
            );
            // 3. Find the minimum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 2));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm_extract_epi8(_mm512_castsi512_si128(imin), 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m512i = value;
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi32(vmax, vmax, 8));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi32(vmax, vmax, 4));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 8));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 4));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 2));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm_extract_epi8(_mm512_castsi512_si128(vmax), 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm512_cmpeq_epi8_mask(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi8(
                mask,
                _mm512_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
            );
            // 3. Find the maximum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 2));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm_extract_epi8(_mm512_castsi512_si128(imin), 0) as usize;

            (max_index, max_value)
        }
    }

    impl_SIMDInit_Int!(i8, __m512i, u64, LANE_SIZE, AVX512<Int>);

    impl_SIMDArgMinMax!(
        i8,
        __m512i,
        u64,
        LANE_SIZE,
        SCALAR<Int>,
        AVX512<Int>,
        "avx512bw"
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

    const LANE_SIZE: usize = NEON::<Int>::LANE_SIZE_8;

    impl SIMDOps<i8, int8x16_t, uint8x16_t, LANE_SIZE> for NEON<Int> {
        const INITIAL_INDEX: int8x16_t = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8,
            ])
        };
        const INDEX_INCREMENT: int8x16_t =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: int8x16_t) -> [i8; LANE_SIZE] {
            std::mem::transmute::<int8x16_t, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> int8x16_t {
            // TODO: requires v7
            vld1q_s8(data as *const i8)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            vaddq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
            vcgtq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
            vcltq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int8x16_t, b: int8x16_t, mask: uint8x16_t) -> int8x16_t {
            vbslq_s8(mask, b, a)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: int8x16_t, value: int8x16_t) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: int8x16_t = value;
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 8));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 4));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 2));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 1));
            let min_value: i8 = vgetq_lane_s8(vmin, 0);

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = vceqq_s8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = vbslq_s8(
                mask,
                index,               // if mask is 1, use index
                vdupq_n_s8(i8::MAX), // if mask is 0, use i8::MAX
            );
            // 3. Find the minimum index
            let mut imin: int8x16_t = search_index;
            imin = vminq_s8(imin, vextq_s8(imin, imin, 8));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 4));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 2));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 1));
            let min_index: usize = vgetq_lane_s8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int8x16_t, value: int8x16_t) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: int8x16_t = value;
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 8));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 4));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 2));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 1));
            let max_value: i8 = vgetq_lane_s8(vmax, 0);

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = vceqq_s8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = vbslq_s8(
                mask,
                index,               // if mask is 1, use index
                vdupq_n_s8(i8::MAX), // if mask is 0, use i8::MAX
            );
            // 3. Find the maximum index
            let mut imin: int8x16_t = search_index;
            imin = vminq_s8(imin, vextq_s8(imin, imin, 8));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 4));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 2));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 1));
            let max_index: usize = vgetq_lane_s8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    impl_SIMDInit_Int!(i8, int8x16_t, uint8x16_t, LANE_SIZE, NEON<Int>);

    impl_SIMDArgMinMax!(
        i8,
        int8x16_t,
        uint8x16_t,
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
    use crate::{Int, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_no_overflow_argminmax,
        test_return_same_result_argminmax,
    };

    use dev_utils::utils;

    fn get_array_i8(n: usize) -> Vec<i8> {
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
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<Int>}, is_x86_feature_detected!("avx512bw")))]
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
        T: SIMDArgMinMax<i8, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
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
        T: SIMDArgMinMax<i8, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_i8, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_no_overflow<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<i8, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_no_overflow_argminmax(get_array_i8, SCALAR_STRATEGY, simd, None);
    }
}
