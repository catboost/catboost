use core::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_sad_epu8,
    _mm512_setzero_si512, _mm512_slli_epi32, _mm512_zextsi128_si512, _mm_cvtsi32_si128,
};

use super::avx512::{_mm512_reduce_add_epu32, partial_hsum};
use crate::adler32::{BASE, NMAX};

const fn __m512i_literal(bytes: [u8; 64]) -> __m512i {
    // SAFETY: any valid [u8; 64] represents a valid __m512i
    unsafe { core::mem::transmute(bytes) }
}

const DOT2V: __m512i = __m512i_literal({
    let mut arr = [0; 64];

    // generates [64, 63, ..., 2, 1]
    let mut i = 64;
    while i > 0 {
        i -= 1;
        arr[i] = (64 - i) as u8;
    }

    arr
});

pub fn adler32_avx512(adler: u32, src: &[u8]) -> u32 {
    assert!(cfg!(target_feature = "avx512bw"));
    assert!(cfg!(target_feature = "avx512vnni"));
    // SAFETY: the assertion above ensures this code is not executed unless the CPU has avx512.
    unsafe { adler32_avx512_vnni(adler, src) }
}

#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512vnni")]
pub(super) fn adler32_avx512_vnni(mut adler: u32, mut src: &[u8]) -> u32 {
    if src.is_empty() {
        return adler;
    }

    let mut adler0;
    let mut adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff;

    unsafe {
        'rem_peel: loop {
            if src.len() < 32 {
                return super::avx2::adler32_avx2(adler, src);
            }

            if src.len() < 64 {
                return super::avx2::adler32_avx2(adler, src);
            }

            let dot2v = DOT2V;

            let zero = _mm512_setzero_si512();
            let mut vs1;
            let mut vs2;

            while src.len() >= 64 {
                vs1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler0 as i32));
                vs2 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler1 as i32));
                let mut k: usize = Ord::min(src.len(), NMAX as usize);
                k -= k % 64;
                let mut vs1_0 = vs1;
                let mut vs3 = _mm512_setzero_si512();
                /* We might get a tad bit more ILP here if we sum to a second register in the loop */
                let mut vs2_1 = _mm512_setzero_si512();
                let mut vbuf0;
                let mut vbuf1;

                /* Remainder peeling */
                if (k % 128) != 0 {
                    vbuf1 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>());

                    src = &src[64..];
                    k -= 64;

                    let vs1_sad = _mm512_sad_epu8(vbuf1, zero);
                    vs1 = _mm512_add_epi32(vs1, vs1_sad);
                    vs3 = _mm512_add_epi32(vs3, vs1_0);
                    vs2 = _mm512_dpbusd_epi32(vs2, vbuf1, dot2v);
                    vs1_0 = vs1;
                }

                /* Manually unrolled this loop by 2 for an decent amount of ILP */
                while k >= 128 {
                    /*
                       vs1 = adler + sum(c[i])
                       vs2 = sum2 + 64 vs1 + sum( (64-i+1) c[i] )
                    */
                    vbuf0 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>());
                    vbuf1 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(1));
                    src = &src[128..];
                    k -= 128;

                    let mut vs1_sad = _mm512_sad_epu8(vbuf0, zero);
                    vs1 = _mm512_add_epi32(vs1, vs1_sad);
                    vs3 = _mm512_add_epi32(vs3, vs1_0);
                    /* multiply-add, resulting in 16 ints. Fuse with sum stage from prior versions, as we now have the dp
                     * instructions to eliminate them */
                    vs2 = _mm512_dpbusd_epi32(vs2, vbuf0, dot2v);

                    vs3 = _mm512_add_epi32(vs3, vs1);
                    vs1_sad = _mm512_sad_epu8(vbuf1, zero);
                    vs1 = _mm512_add_epi32(vs1, vs1_sad);
                    vs2_1 = _mm512_dpbusd_epi32(vs2_1, vbuf1, dot2v);
                    vs1_0 = vs1;
                }

                vs3 = _mm512_slli_epi32(vs3, 6);
                vs2 = _mm512_add_epi32(vs2, vs3);
                vs2 = _mm512_add_epi32(vs2, vs2_1);

                adler0 = partial_hsum(vs1) % BASE;
                adler1 = _mm512_reduce_add_epu32(vs2) % BASE;
            }

            adler = adler0 | (adler1 << 16);

            /* Process tail (len < 64). */
            if !src.is_empty() {
                continue 'rem_peel;
            }

            return adler;
        }
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx512vnni")]
#[cfg(target_feature = "avx512bw")]
mod test {
    use super::*;
    use core::arch::x86_64::__m256i;

    #[test]
    fn empty_input() {
        let avx512 = unsafe { adler32_avx512(0, &[]) };
        let rust = crate::adler32::generic::adler32_rust(0, &[]);

        assert_eq!(rust, avx512);
    }

    #[test]
    fn zero_chunks() {
        let input = &[
            1u8, 39, 76, 148, 0, 58, 0, 14, 255, 59, 1, 229, 1, 83, 5, 84, 207, 152, 188,
        ];
        let avx512 = unsafe { adler32_avx512(0, input) };
        let rust = crate::adler32::generic::adler32_rust(0, input);

        assert_eq!(rust, avx512);
    }

    #[test]
    fn one_chunk() {
        let input: [u8; 85] = core::array::from_fn(|i| i as u8);
        let avx512 = unsafe { adler32_avx512(0, &input) };
        let rust = crate::adler32::generic::adler32_rust(0, &input);

        assert_eq!(rust, avx512);
    }

    #[test]
    fn foobar() {
        let v = vec![
            0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 255, 209, 220, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let start = 0;
        let avx512 = unsafe { adler32_avx512(start, &v) };
        let rust = crate::adler32::generic::adler32_rust(start, &v);

        assert_eq!(rust, avx512);
    }

    quickcheck::quickcheck! {
        fn adler32_avx512_is_adler32_rust(v: Vec<u8>, start: u32) -> bool {
            let avx512 = unsafe { adler32_avx512(start, &v) };
            let rust = crate::adler32::generic::adler32_rust(start, &v);

            rust == avx512
        }
    }

    const INPUT: [u8; 128] = {
        let mut array = [0; 128];
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
                let avx512 = unsafe { adler32_avx512(start, &INPUT[i..]) };
                let rust = crate::adler32::generic::adler32_rust(start, &INPUT[i..]);

                assert_eq!(avx512, rust, "offset = {i}, start = {start}");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn large_input() {
        const DEFAULT: &[u8] = include_bytes!("../deflate/test-data/paper-100k.pdf");

        let avx512 = unsafe { adler32_avx512(42, DEFAULT) };
        let rust = crate::adler32::generic::adler32_rust(42, DEFAULT);

        assert_eq!(avx512, rust);
    }
}
