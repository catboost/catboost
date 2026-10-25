use crate::CRC32_INITIAL_VALUE;
use core::arch::x86_64::{
    __m512i, _mm512_clmulepi64_epi128, _mm512_extracti32x4_epi32, _mm512_inserti32x4,
    _mm512_loadu_si512, _mm512_set4_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32, _mm512_xor_si512, _mm512_zextsi128_si512, _mm_cvtsi32_si128,
};

impl super::pclmulqdq::Accumulator {
    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub(super) unsafe fn fold_16_vpclmulqdq(
        &mut self,
        dst: &mut [u8],
        src: &mut &[u8],
        init_crc: &mut u32,
    ) -> usize {
        unsafe { self.fold_help_vpclmulqdq::<false>(dst, src, init_crc) }
    }

    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    pub(super) unsafe fn fold_16_vpclmulqdq_copy(
        &mut self,
        dst: &mut [u8],
        src: &mut &[u8],
    ) -> usize {
        let mut init_crc = CRC32_INITIAL_VALUE;
        unsafe { self.fold_help_vpclmulqdq::<true>(dst, src, &mut init_crc) }
    }

    #[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
    unsafe fn fold_help_vpclmulqdq<const COPY: bool>(
        &mut self,
        mut dst: &mut [u8],
        src: &mut &[u8],
        init_crc: &mut u32,
    ) -> usize {
        let [xmm_crc0, xmm_crc1, xmm_crc2, xmm_crc3] = &mut self.fold;
        let start_len = src.len();

        unsafe {
            let zmm_fold4 =
                _mm512_set4_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596u32 as i32);
            let zmm_fold16 = _mm512_set4_epi32(0x00000001, 0x1542778a, 0x00000001, 0x322d1430);

            // zmm register init
            let zmm_crc0 = _mm512_setzero_si512();
            let mut zmm_t0 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>());

            if !COPY && *init_crc != CRC32_INITIAL_VALUE {
                let xmm_initial = _mm_cvtsi32_si128(*init_crc as i32);
                let zmm_initial = _mm512_zextsi128_si512(xmm_initial);
                zmm_t0 = _mm512_xor_si512(zmm_t0, zmm_initial);
                *init_crc = CRC32_INITIAL_VALUE;
            }

            let mut zmm_crc1 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(1));
            let mut zmm_crc2 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(2));
            let mut zmm_crc3 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(3));

            /* already have intermediate CRC in xmm registers
             * fold4 with 4 xmm_crc to get zmm_crc0
             */
            let mut zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc0, 0);
            zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc1, 1);
            zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc2, 2);
            zmm_crc0 = _mm512_inserti32x4(zmm_crc0, *xmm_crc3, 3);
            let mut z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
            zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
            zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);

            if COPY {
                _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>(), zmm_t0);
                _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(1), zmm_crc1);
                _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(2), zmm_crc2);
                _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(3), zmm_crc3);
                dst = &mut dst[256..];
            }

            *src = &src[256..];

            // fold-16 loops
            while src.len() >= 256 {
                let zmm_t0 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>());
                let zmm_t1 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(1));
                let zmm_t2 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(2));
                let zmm_t3 = _mm512_loadu_si512(src.as_ptr().cast::<__m512i>().add(3));

                let z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x01);
                let z1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x01);
                let z2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x01);
                let z3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x01);

                zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold16, 0x10);
                zmm_crc1 = _mm512_clmulepi64_epi128(zmm_crc1, zmm_fold16, 0x10);
                zmm_crc2 = _mm512_clmulepi64_epi128(zmm_crc2, zmm_fold16, 0x10);
                zmm_crc3 = _mm512_clmulepi64_epi128(zmm_crc3, zmm_fold16, 0x10);

                zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_t0, 0x96);
                zmm_crc1 = _mm512_ternarylogic_epi32(zmm_crc1, z1, zmm_t1, 0x96);
                zmm_crc2 = _mm512_ternarylogic_epi32(zmm_crc2, z2, zmm_t2, 0x96);
                zmm_crc3 = _mm512_ternarylogic_epi32(zmm_crc3, z3, zmm_t3, 0x96);

                if COPY {
                    _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>(), zmm_t0);
                    _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(1), zmm_t1);
                    _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(2), zmm_t2);
                    _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>().add(3), zmm_t3);
                    dst = &mut dst[256..];
                }

                *src = &src[256..];
            }

            // zmm_crc[0,1,2,3] -> zmm_crc0
            z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
            zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
            zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc1, 0x96);

            z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
            zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
            zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc2, 0x96);

            z0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x01);
            zmm_crc0 = _mm512_clmulepi64_epi128(zmm_crc0, zmm_fold4, 0x10);
            zmm_crc0 = _mm512_ternarylogic_epi32(zmm_crc0, z0, zmm_crc3, 0x96);

            // zmm_crc0 -> xmm_crc[0, 1, 2, 3]
            *xmm_crc0 = _mm512_extracti32x4_epi32(zmm_crc0, 0);
            *xmm_crc1 = _mm512_extracti32x4_epi32(zmm_crc0, 1);
            *xmm_crc2 = _mm512_extracti32x4_epi32(zmm_crc0, 2);
            *xmm_crc3 = _mm512_extracti32x4_epi32(zmm_crc0, 3);

            // return n bytes processed
            start_len - src.len()
        }
    }
}
