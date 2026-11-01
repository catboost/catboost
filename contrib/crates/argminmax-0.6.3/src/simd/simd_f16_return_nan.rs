/// Implementation of the argminmax operations for f16 where NaN values take precedence.
/// This implementation returns the index of the first* NaN value if any are present,
/// otherwise it returns the index of the minimum and maximum values.
///
/// To serve this functionality we transform the f16 values to ordinal i16 values:
///     ord_i16 = ((v >> 15) & 0x7FFFFFFF) ^ v
///
/// This transformation is a bijection, i.e. it is reversible:
///     v = ((ord_i16 >> 15) & 0x7FFFFFFF) ^ ord_i16
///
/// Through this transformation we can perform the argminmax operations on the ordinal
/// integer values and then transform the result back to the original f16 values.
/// This transformation is necessary because comparisons with NaN values are always false.
/// So unless we perform ! <=  as gt and ! >=  as lt the argminmax operations will not
/// add NaN values to the accumulating SIMD register. And as le and ge are significantly
/// more expensive than lt and gt we use this efficient bitwise transformation.
///
/// Note that most x86 CPUs do not support f16 instructions - making this implementation
/// multitudes (up to 300x) faster than trying to use a vanilla scalar implementation.
///
///
/// ---
///
/// *Note: the first NaN value is only returned iff all NaN values have the same bit
/// representation. When NaN values have different bit representations then the index of
/// the highest / lowest ord_i16 is returned for the
/// SIMDOps::_get_overflow_lane_size_limit() chunk of the data - which is not
/// necessarily the index of the first NaN value.
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
use super::generic::{impl_SIMDInit_FloatReturnNaN, SIMDArgMinMax, SIMDInit, SIMDOps};
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

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use half::f16;

/// The dtype-strategy for performing operations on f16 data: return NaN index
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::super::dtype_strategy::FloatReturnNaN;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const BIT_SHIFT: i32 = 15;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MASK_VALUE: i16 = 0x7FFF; // i16::MAX - masks everything but the sign bit

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
#[inline(always)]
fn _i16ord_to_f16(ord_i16: i16) -> f16 {
    let v = ((ord_i16 >> BIT_SHIFT) & MASK_VALUE) ^ ord_i16;
    f16::from_bits(v as u16)
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MAX_INDEX: usize = i16::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: __m256i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m256i_to_i16ord(f16_as_m256i: __m256i) -> __m256i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm256_srai_epi16(f16_as_m256i, BIT_SHIFT);
        let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, LOWER_15_MASK);
        _mm256_xor_si256(sign_bit_masked, f16_as_m256i)
        // TODO: investigate if this is faster
        // _mm256_xor_si256(
        //     _mm256_srai_epi16(f16_as_m256i, 15),
        //     _mm256_and_si256(f16_as_m256i, LOWER_15_MASK),
        // )
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m256i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m256i, __m256i, LANE_SIZE> for AVX2<FloatReturnNaN> {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m256i {
            _f16_as_m256i_to_i16ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m256i = value;
            vmin = _mm256_min_epi16(vmin, _mm256_permute2x128_si256(vmin, vmin, 1));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 8));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 4));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm256_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm256_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m256i = value;
            vmax = _mm256_max_epi16(vmax, _mm256_permute2x128_si256(vmax, vmax, 1));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm256_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm256_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m256i, __m256i, LANE_SIZE, AVX2<FloatReturnNaN>);

    impl SIMDArgMinMax<f16, __m256i, __m256i, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for AVX2<FloatReturnNaN>
    {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f16]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f16]) -> usize {
            Self::argminmax(data).1
        }
    }
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: __m128i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m128i_to_i16ord(f16_as_m128i: __m128i) -> __m128i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm_srai_epi16(f16_as_m128i, BIT_SHIFT);
        let sign_bit_masked = _mm_and_si128(sign_bit_shifted, LOWER_15_MASK);
        _mm_xor_si128(sign_bit_masked, f16_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m128i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m128i, __m128i, LANE_SIZE> for SSE<FloatReturnNaN> {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m128i {
            _f16_as_m128i_to_i16ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m128i = value;
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 8));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 4));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m128i = value;
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 8));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 4));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i8::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m128i, __m128i, LANE_SIZE, SSE<FloatReturnNaN>);

    impl SIMDArgMinMax<f16, __m128i, __m128i, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for SSE<FloatReturnNaN>
    {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f16]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f16]) -> usize {
            Self::argminmax(data).1
        }
    }
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: __m512i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m521i_to_i16ord(f16_as_m512i: __m512i) -> __m512i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm512_srai_epi16(f16_as_m512i, BIT_SHIFT as u32);
        let sign_bit_masked = _mm512_and_si512(sign_bit_shifted, LOWER_15_MASK);
        _mm512_xor_si512(f16_as_m512i, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m512i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m512i, u32, LANE_SIZE> for AVX512<FloatReturnNaN> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m512i {
            _f16_as_m521i_to_i16ord(_mm512_loadu_epi16(data as *const i16))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmpgt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmplt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u32) -> __m512i {
            _mm512_mask_blend_epi16(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m512i = value;
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmin), 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the minimum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m512i = value;
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmax), 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the maximum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m512i, u32, LANE_SIZE, AVX512<FloatReturnNaN>);

    impl SIMDArgMinMax<f16, __m512i, u32, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for AVX512<FloatReturnNaN>
    {
        #[target_feature(enable = "avx512bw")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f16]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f16]) -> usize {
            Self::argminmax(data).1
        }
    }
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64" // stable for AArch64
))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: int16x8_t = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_int16x8_to_i16ord(f16_as_int16x8: int16x8_t) -> int16x8_t {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = vshrq_n_s16(f16_as_int16x8, BIT_SHIFT);
        let sign_bit_masked = vandq_s16(sign_bit_shifted, LOWER_15_MASK);
        veorq_s16(f16_as_int16x8, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: int16x8_t) -> [i16; LANE_SIZE] {
        std::mem::transmute::<int16x8_t, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, int16x8_t, uint16x8_t, LANE_SIZE> for NEON<FloatReturnNaN> {
        const INITIAL_INDEX: int16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: int16x8_t =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: int16x8_t) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> int16x8_t {
            _f16_as_int16x8_to_i16ord(vld1q_s16(data as *const i16))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int16x8_t, b: int16x8_t) -> int16x8_t {
            vaddq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcgtq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcltq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int16x8_t, b: int16x8_t, mask: uint16x8_t) -> int16x8_t {
            vbslq_s16(mask, b, a)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: int16x8_t, value: int16x8_t) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: int16x8_t = value;
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 4));
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 2));
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 1));
            let min_value: i16 = vgetq_lane_s16(vmin, 0);

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = vceqq_s16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = vbslq_s16(
                mask,
                index,                 // if mask is 1, use index
                vdupq_n_s16(i16::MAX), // if mask is 0, use i16::MAX
            );
            // 3. Find the minimum index
            let mut imin: int16x8_t = search_index;
            imin = vminq_s16(imin, vextq_s16(imin, imin, 4));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 2));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 1));
            let min_index: usize = vgetq_lane_s16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int16x8_t, value: int16x8_t) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: int16x8_t = value;
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 4));
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 2));
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 1));
            let max_value: i16 = vgetq_lane_s16(vmax, 0);

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = vceqq_s16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = vbslq_s16(
                mask,
                index,                 // if mask is 1, use index
                vdupq_n_s16(i16::MAX), // if mask is 0, use i16::MAX
            );
            // 3. Find the maximum index
            let mut imin: int16x8_t = search_index;
            imin = vminq_s16(imin, vextq_s16(imin, imin, 4));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 2));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 1));
            let max_index: usize = vgetq_lane_s16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, int16x8_t, uint16x8_t, LANE_SIZE, NEON<FloatReturnNaN>);

    impl SIMDArgMinMax<f16, int16x8_t, uint16x8_t, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for NEON<FloatReturnNaN>
    {
        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f16]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f16]) -> usize {
            Self::argminmax(data).1
        }
    }
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

    use half::f16;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "nightly_simd")]
    use crate::simd::config::AVX512;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, SSE};
    use crate::{FloatReturnNaN, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_no_overflow_argminmax,
        test_return_same_result_argminmax,
    };
    // Float specific tests
    use super::super::test_utils::{test_return_infs_argminmax, test_return_nans_argminmax};

    use dev_utils::utils;

    fn get_array_f16(n: usize) -> Vec<f16> {
        utils::SampleUniformFullRange::get_random_array(n)
    }

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<FloatReturnNaN> = SCALAR {
        _dtype_strategy: PhantomData::<FloatReturnNaN>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("avx2"))]
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("avx512bw")))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[template]
    #[rstest]
    #[case::neon(NEON {_dtype_strategy: PhantomData::<FloatReturnNaN>}, true)]
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
        T: SIMDArgMinMax<f16, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
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
        T: SIMDArgMinMax<f16, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_f16, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_no_overflow<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f16, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_no_overflow_argminmax(get_array_f16, SCALAR_STRATEGY, simd, None);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f16, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f16, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f16, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_nans_argminmax(get_array_f16, SCALAR_STRATEGY, simd);
    }
}
