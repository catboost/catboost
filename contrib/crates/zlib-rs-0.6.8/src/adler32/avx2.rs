//! # Safety
//!
//! The functions in this module should only be executed on x86 machines with the AVX2 extension.
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_permutevar8x32_epi32, _mm256_sad_epu8, _mm256_slli_epi32,
    _mm256_zextsi128_si256, _mm_add_epi32, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_shuffle_epi32,
    _mm_unpackhi_epi64,
};

use crate::adler32::{
    generic::{adler32_len_16, adler32_len_64},
    BASE, NMAX,
};

const fn __m256i_literal(bytes: [u8; 32]) -> __m256i {
    // SAFETY: any valid [u8; 32] represents a valid __m256i
    unsafe { core::mem::transmute(bytes) }
}

const DOT2V: __m256i = __m256i_literal([
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
    8, 7, 6, 5, 4, 3, 2, 1,
]);

const DOT3V: __m256i = __m256i_literal([
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
]);

const ZERO: __m256i = __m256i_literal([0u8; 32]);

/// 32 bit horizontal sum, adapted from Agner Fog's vector library.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(x: __m256i) -> u32 {
    #[allow(unused_unsafe)] // because target features 1.1
    unsafe {
        let sum1 = _mm_add_epi32(_mm256_extracti128_si256(x, 1), _mm256_castsi256_si128(x));
        let sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
        let sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
        _mm_cvtsi128_si32(sum3) as u32
    }
}

#[target_feature(enable = "avx2")]
unsafe fn partial_hsum256(x: __m256i) -> u32 {
    const PERM_VEC: __m256i = __m256i_literal([
        0, 0, 0, 0, //
        2, 0, 0, 0, //
        4, 0, 0, 0, //
        6, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
        1, 0, 0, 0, //
    ]);

    #[allow(unused_unsafe)] // because target features 1.1
    unsafe {
        let non_zero = _mm256_permutevar8x32_epi32(x, PERM_VEC);
        let non_zero_sse = _mm256_castsi256_si128(non_zero);
        let sum2 = _mm_add_epi32(non_zero_sse, _mm_unpackhi_epi64(non_zero_sse, non_zero_sse));
        let sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
        _mm_cvtsi128_si32(sum3) as u32
    }
}

pub fn adler32_avx2(adler: u32, src: &[u8]) -> u32 {
    assert!(crate::cpu_features::is_enabled_avx2_and_bmi2());
    // SAFETY: the assertion above ensures this code is not executed unless the CPU has AVX2.
    unsafe { adler32_avx2_help(adler, src) }
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
#[target_feature(enable = "bmi1")]
unsafe fn adler32_avx2_help(adler: u32, src: &[u8]) -> u32 {
    if src.is_empty() {
        return adler;
    }

    // SAFETY: [u8; 32] safely transmutes into __m256i.
    let (before, middle, after) = unsafe { src.align_to::<__m256i>() };

    let mut adler1 = (adler >> 16) & 0xffff;
    let mut adler0 = adler & 0xffff;

    let adler = if before.len() < 16 {
        adler32_len_16(adler0, before, adler1)
    } else if before.len() < 32 {
        adler32_len_64(adler0, before, adler1)
    } else {
        adler
    };

    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

    // use largest step possible (without causing overflow)
    for chunk in middle.chunks(NMAX as usize / 32) {
        (adler0, adler1) = unsafe { helper_32_bytes(adler0, adler1, chunk) };
    }

    if !after.is_empty() {
        if after.len() < 16 {
            return adler32_len_16(adler0, after, adler1);
        } else if after.len() < 32 {
            return adler32_len_64(adler0, after, adler1);
        } else {
            unreachable!()
        }
    }

    adler0 | (adler1 << 16)
}

#[target_feature(enable = "avx2")]
unsafe fn helper_32_bytes(mut adler0: u32, mut adler1: u32, src: &[__m256i]) -> (u32, u32) {
    unsafe {
        let mut vs1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler0 as i32));
        let mut vs2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(adler1 as i32));

        let mut vs1_0 = vs1;
        let mut vs3 = ZERO;

        for vbuf in src.iter().copied() {
            let vs1_sad = _mm256_sad_epu8(vbuf, ZERO); // Sum of abs diff, resulting in 2 x int32's

            vs1 = _mm256_add_epi32(vs1, vs1_sad);
            vs3 = _mm256_add_epi32(vs3, vs1_0);
            let v_short_sum2 = _mm256_maddubs_epi16(vbuf, DOT2V); // sum 32 uint8s to 16 shorts
            let vsum2 = _mm256_madd_epi16(v_short_sum2, DOT3V); // sum 16 shorts to 8 uint32s
            vs2 = _mm256_add_epi32(vsum2, vs2);
            vs1_0 = vs1;
        }

        /* Defer the multiplication with 32 to outside of the loop */
        vs3 = _mm256_slli_epi32(vs3, 5);
        vs2 = _mm256_add_epi32(vs2, vs3);

        adler0 = partial_hsum256(vs1) % BASE;
        adler1 = hsum256(vs2) % BASE;

        (adler0, adler1)
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx2")]
mod test {
    use super::*;

    #[test]
    fn empty_input() {
        let avx2 = adler32_avx2(0, &[]);
        let rust = crate::adler32::generic::adler32_rust(0, &[]);

        assert_eq!(rust, avx2);
    }

    quickcheck::quickcheck! {
        fn adler32_avx2_is_adler32_rust(v: Vec<u8>, start: u32) -> bool {
            let avx2 = adler32_avx2(start, &v);
            let rust = crate::adler32::generic::adler32_rust(start, &v);

            rust == avx2
        }
    }

    const INPUT: [u8; 1024] = {
        let mut array = [0; 1024];
        let mut i = 0;
        while i < array.len() {
            array[i] = i as u8;
            i += 1;
        }

        array
    };

    #[test]
    fn start_alignment() {
        // SIMD algorithm is sensitive to alignment;
        for i in 0..16 {
            for start in [crate::ADLER32_INITIAL_VALUE as u32, 42] {
                let avx2 = adler32_avx2(start, &INPUT[i..]);
                let rust = crate::adler32::generic::adler32_rust(start, &INPUT[i..]);

                assert_eq!(avx2, rust, "offset = {i}, start = {start}");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn large_input() {
        const DEFAULT: &[u8] = include_bytes!("../deflate/test-data/paper-100k.pdf");

        let avx2 = adler32_avx2(42, DEFAULT);
        let rust = crate::adler32::generic::adler32_rust(42, DEFAULT);

        assert_eq!(avx2, rust);
    }
}
