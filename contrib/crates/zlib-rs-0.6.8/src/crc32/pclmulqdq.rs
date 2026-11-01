use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_clmulepi64_si128, _mm_extract_epi32, _mm_load_si128,
    _mm_loadu_si128, _mm_or_si128, _mm_shuffle_epi8, _mm_slli_si128, _mm_srli_si128,
    _mm_storeu_si128, _mm_xor_si128,
};

use crate::CRC32_INITIAL_VALUE;

#[derive(Debug)]
#[repr(C, align(16))]
struct Align16<T>(T);

#[cfg(target_arch = "x86_64")]
const fn reg(input: [u32; 4]) -> __m128i {
    // SAFETY: any valid [u32; 4] represents a valid __m128i
    unsafe { core::mem::transmute(input) }
}

/// # Safety
///
/// The methods of this struct should only be used on x86_64 systems that support the `pclmulqdq`
/// instruction, and SSE2 and SSE4.1 instruction sets.
#[derive(Debug, Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct Accumulator {
    pub(super) fold: [__m128i; 4],
}

#[cfg(target_arch = "x86_64")]
impl Accumulator {
    const XMM_FOLD4: __m128i = reg([0xc6e41596u32, 0x00000001u32, 0x54442bd4u32, 0x00000001u32]);

    pub const fn new() -> Self {
        let xmm_crc0 = reg([0x9db42487, 0, 0, 0]);
        let xmm_zero = reg([0, 0, 0, 0]);

        Self {
            fold: [xmm_crc0, xmm_zero, xmm_zero, xmm_zero],
        }
    }

    pub unsafe fn fold(&mut self, src: &[u8], start: u32) {
        unsafe { self.fold_help::<false>(&mut [], src, start) }
    }

    pub unsafe fn fold_copy(&mut self, dst: &mut [u8], src: &[u8]) {
        unsafe { self.fold_help::<true>(dst, src, 0) }
    }

    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    pub unsafe fn finish(self) -> u32 {
        const CRC_MASK1: __m128i =
            reg([0xFFFFFFFFu32, 0xFFFFFFFFu32, 0x00000000u32, 0x00000000u32]);

        const CRC_MASK2: __m128i =
            reg([0x00000000u32, 0xFFFFFFFFu32, 0xFFFFFFFFu32, 0xFFFFFFFFu32]);

        const RK1_RK2: __m128i = reg([
            0xccaa009e, 0x00000000, /* rk1 */
            0x751997d0, 0x00000001, /* rk2 */
        ]);

        const RK5_RK6: __m128i = reg([
            0xccaa009e, 0x00000000, /* rk5 */
            0x63cd6124, 0x00000001, /* rk6 */
        ]);

        const RK7_RK8: __m128i = reg([
            0xf7011640, 0x00000001, /* rk7 */
            0xdb710640, 0x00000001, /* rk8 */
        ]);

        let [mut xmm_crc0, mut xmm_crc1, mut xmm_crc2, mut xmm_crc3] = self.fold;

        /*
         * k1
         */
        let mut crc_fold = RK1_RK2;

        #[allow(unused_unsafe)] // because target features 1.1
        unsafe {
            let x_tmp0 = _mm_clmulepi64_si128(xmm_crc0, crc_fold, 0x10);
            xmm_crc0 = _mm_clmulepi64_si128(xmm_crc0, crc_fold, 0x01);
            xmm_crc1 = _mm_xor_si128(xmm_crc1, x_tmp0);
            xmm_crc1 = _mm_xor_si128(xmm_crc1, xmm_crc0);

            let x_tmp1 = _mm_clmulepi64_si128(xmm_crc1, crc_fold, 0x10);
            xmm_crc1 = _mm_clmulepi64_si128(xmm_crc1, crc_fold, 0x01);
            xmm_crc2 = _mm_xor_si128(xmm_crc2, x_tmp1);
            xmm_crc2 = _mm_xor_si128(xmm_crc2, xmm_crc1);

            let x_tmp2 = _mm_clmulepi64_si128(xmm_crc2, crc_fold, 0x10);
            xmm_crc2 = _mm_clmulepi64_si128(xmm_crc2, crc_fold, 0x01);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, x_tmp2);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);

            /*
             * k5
             */
            crc_fold = RK5_RK6;

            xmm_crc0 = xmm_crc3;
            xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0);
            xmm_crc0 = _mm_srli_si128(xmm_crc0, 8);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc0);

            xmm_crc0 = xmm_crc3;
            xmm_crc3 = _mm_slli_si128(xmm_crc3, 4);
            xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0x10);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc0);
            xmm_crc3 = _mm_and_si128(xmm_crc3, CRC_MASK2);

            /*
             * k7
             */
            xmm_crc1 = xmm_crc3;
            xmm_crc2 = xmm_crc3;
            crc_fold = RK7_RK8;

            xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);
            xmm_crc3 = _mm_and_si128(xmm_crc3, CRC_MASK1);

            xmm_crc2 = xmm_crc3;
            xmm_crc3 = _mm_clmulepi64_si128(xmm_crc3, crc_fold, 0x10);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc2);
            xmm_crc3 = _mm_xor_si128(xmm_crc3, xmm_crc1);

            !(_mm_extract_epi32(xmm_crc3, 2) as u32)
        }
    }

    unsafe fn fold_step<const N: usize>(&mut self) {
        self.fold = core::array::from_fn(|i| match self.fold.get(i + N) {
            Some(v) => *v,
            None => unsafe { Self::step(self.fold[(i + N) - 4]) },
        });
    }

    #[inline(always)]
    unsafe fn step(input: __m128i) -> __m128i {
        unsafe {
            _mm_xor_si128(
                _mm_clmulepi64_si128(input, Self::XMM_FOLD4, 0x01),
                _mm_clmulepi64_si128(input, Self::XMM_FOLD4, 0x10),
            )
        }
    }

    unsafe fn partial_fold(&mut self, xmm_crc_part: __m128i, len: usize) {
        const PSHUFB_SHF_TABLE: [__m128i; 15] = [
            reg([0x84838281, 0x88878685, 0x8c8b8a89, 0x008f8e8d]), /* shl 15 (16 - 1)/shr1 */
            reg([0x85848382, 0x89888786, 0x8d8c8b8a, 0x01008f8e]), /* shl 14 (16 - 3)/shr2 */
            reg([0x86858483, 0x8a898887, 0x8e8d8c8b, 0x0201008f]), /* shl 13 (16 - 4)/shr3 */
            reg([0x87868584, 0x8b8a8988, 0x8f8e8d8c, 0x03020100]), /* shl 12 (16 - 4)/shr4 */
            reg([0x88878685, 0x8c8b8a89, 0x008f8e8d, 0x04030201]), /* shl 11 (16 - 5)/shr5 */
            reg([0x89888786, 0x8d8c8b8a, 0x01008f8e, 0x05040302]), /* shl 10 (16 - 6)/shr6 */
            reg([0x8a898887, 0x8e8d8c8b, 0x0201008f, 0x06050403]), /* shl  9 (16 - 7)/shr7 */
            reg([0x8b8a8988, 0x8f8e8d8c, 0x03020100, 0x07060504]), /* shl  8 (16 - 8)/shr8 */
            reg([0x8c8b8a89, 0x008f8e8d, 0x04030201, 0x08070605]), /* shl  7 (16 - 9)/shr9 */
            reg([0x8d8c8b8a, 0x01008f8e, 0x05040302, 0x09080706]), /* shl  6 (16 -10)/shr10*/
            reg([0x8e8d8c8b, 0x0201008f, 0x06050403, 0x0a090807]), /* shl  5 (16 -11)/shr11*/
            reg([0x8f8e8d8c, 0x03020100, 0x07060504, 0x0b0a0908]), /* shl  4 (16 -12)/shr12*/
            reg([0x008f8e8d, 0x04030201, 0x08070605, 0x0c0b0a09]), /* shl  3 (16 -13)/shr13*/
            reg([0x01008f8e, 0x05040302, 0x09080706, 0x0d0c0b0a]), /* shl  2 (16 -14)/shr14*/
            reg([0x0201008f, 0x06050403, 0x0a090807, 0x0e0d0c0b]), /* shl  1 (16 -15)/shr15*/
        ];

        unsafe {
            let xmm_shl = PSHUFB_SHF_TABLE[len - 1];
            let xmm_shr = _mm_xor_si128(xmm_shl, reg([0x80808080u32; 4]));

            let xmm_a0 = Self::step(_mm_shuffle_epi8(self.fold[0], xmm_shl));

            self.fold[0] = _mm_shuffle_epi8(self.fold[0], xmm_shr);
            let xmm_tmp1 = _mm_shuffle_epi8(self.fold[1], xmm_shl);
            self.fold[0] = _mm_or_si128(self.fold[0], xmm_tmp1);

            self.fold[1] = _mm_shuffle_epi8(self.fold[1], xmm_shr);
            let xmm_tmp2 = _mm_shuffle_epi8(self.fold[2], xmm_shl);
            self.fold[1] = _mm_or_si128(self.fold[1], xmm_tmp2);

            self.fold[2] = _mm_shuffle_epi8(self.fold[2], xmm_shr);
            let xmm_tmp3 = _mm_shuffle_epi8(self.fold[3], xmm_shl);
            self.fold[2] = _mm_or_si128(self.fold[2], xmm_tmp3);

            self.fold[3] = _mm_shuffle_epi8(self.fold[3], xmm_shr);
            let xmm_crc_part = _mm_shuffle_epi8(xmm_crc_part, xmm_shl);
            self.fold[3] = _mm_or_si128(self.fold[3], xmm_crc_part);

            // zlib-ng uses casts and a floating-point xor instruction here. There is a theory that
            // this breaks dependency chains on some CPUs and gives better throughput. Other sources
            // claim that casting between integer and float has a cost and should be avoided. We can't
            // measure the difference, and choose the shorter code.
            self.fold[3] = _mm_xor_si128(self.fold[3], xmm_a0)
        }
    }

    #[allow(clippy::needless_range_loop)]
    unsafe fn progress<const N: usize, const COPY: bool>(
        &mut self,
        dst: &mut [u8],
        src: &mut &[u8],
        init_crc: &mut u32,
    ) -> usize {
        let mut it = src.chunks_exact(16);
        let mut input: [_; N] = core::array::from_fn(|_| unsafe {
            _mm_load_si128(it.next().unwrap().as_ptr() as *const __m128i)
        });

        *src = &src[N * 16..];

        if COPY {
            for (s, d) in input[..N].iter().zip(dst.chunks_exact_mut(16)) {
                unsafe { _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, *s) };
            }
        } else if *init_crc != CRC32_INITIAL_VALUE {
            let xmm_initial = reg([*init_crc, 0, 0, 0]);
            input[0] = unsafe { _mm_xor_si128(input[0], xmm_initial) };
            *init_crc = CRC32_INITIAL_VALUE;
        }

        unsafe { self.fold_step::<N>() };

        for i in 0..N {
            self.fold[i + (4 - N)] = unsafe { _mm_xor_si128(self.fold[i + (4 - N)], input[i]) };
        }

        if COPY {
            N * 16
        } else {
            0
        }
    }

    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    unsafe fn fold_help<const COPY: bool>(
        &mut self,
        mut dst: &mut [u8],
        mut src: &[u8],
        mut init_crc: u32,
    ) {
        let mut xmm_crc_part = reg([0; 4]);

        let mut partial_buf = Align16([0u8; 16]);

        // Technically the CRC functions don't even call this for input < 64, but a bare minimum of 31
        // bytes of input is needed for the aligning load that occurs.  If there's an initial CRC, to
        // carry it forward through the folded CRC there must be 16 - src % 16 + 16 bytes available, which
        // by definition can be up to 15 bytes + one full vector load. */
        let first = init_crc != CRC32_INITIAL_VALUE;
        assert!(src.len() >= 31 || !first);

        // However due to how alignment and loads work out, the 31 minimum is too tight. We use a
        // simpler algorithm anyway for small input because it is faster, so to prevent accidental
        // future UB be more strict here.
        assert!(src.len() >= 64 || !first);

        if COPY {
            assert_eq!(dst.len(), src.len(), "dst and src must be the same length")
        }

        if src.len() < 16 {
            if src.is_empty() {
                return;
            }

            partial_buf.0[..src.len()].copy_from_slice(src);
            xmm_crc_part = unsafe { _mm_load_si128(partial_buf.0.as_mut_ptr() as *mut __m128i) };
            if COPY {
                dst[..src.len()].copy_from_slice(&partial_buf.0[..src.len()]);
            }
        } else {
            // SAFETY: [u8; 16] can safely transmute into _m128i.
            let (before, _, _) = unsafe { src.align_to::<__m128i>() };

            #[allow(unused_unsafe)] // because target features 1.1
            if !before.is_empty() {
                xmm_crc_part = unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) };
                if COPY {
                    unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, xmm_crc_part) };
                    dst = &mut dst[before.len()..];
                } else {
                    let is_initial = init_crc == CRC32_INITIAL_VALUE;

                    if !is_initial {
                        let xmm_initial = reg([init_crc, 0, 0, 0]);
                        xmm_crc_part = unsafe { _mm_xor_si128(xmm_crc_part, xmm_initial) };
                        init_crc = CRC32_INITIAL_VALUE;
                    }

                    if before.len() < 4 && !is_initial {
                        let xmm_t0 = xmm_crc_part;
                        xmm_crc_part =
                            unsafe { _mm_loadu_si128((src.as_ptr() as *const __m128i).add(1)) };

                        unsafe { self.fold_step::<1>() };

                        self.fold[3] = unsafe { _mm_xor_si128(self.fold[3], xmm_t0) };
                        src = &src[16..];
                    }
                }

                unsafe { self.partial_fold(xmm_crc_part, before.len()) };

                src = &src[before.len()..];
            }

            #[cfg(feature = "vpclmulqdq")]
            #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
            if src.len() >= 256 {
                let n;
                if COPY {
                    n = unsafe { self.fold_16_vpclmulqdq_copy(dst, &mut src) };
                    dst = &mut dst[n..];
                } else {
                    unsafe { self.fold_16_vpclmulqdq(dst, &mut src, &mut init_crc) };
                }
            }

            while src.len() >= 64 {
                let n = unsafe { self.progress::<4, COPY>(dst, &mut src, &mut init_crc) };
                dst = &mut dst[n..];
            }

            if src.len() >= 48 {
                let n = unsafe { self.progress::<3, COPY>(dst, &mut src, &mut init_crc) };
                dst = &mut dst[n..];
            } else if src.len() >= 32 {
                let n = unsafe { self.progress::<2, COPY>(dst, &mut src, &mut init_crc) };
                dst = &mut dst[n..];
            } else if src.len() >= 16 {
                let n = unsafe { self.progress::<1, COPY>(dst, &mut src, &mut init_crc) };
                dst = &mut dst[n..];
            }
        }

        if !src.is_empty() {
            debug_assert!(src.len() <= 16);
            unsafe {
                // SAFETY: src and xmm_crc_part don't overlap. xmm_crc_part is statically
                // guaranteed to be 16 bytes, and in earlier code we advanced the src slice such
                // that it's less than 16 bytes.
                core::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    &mut xmm_crc_part as *mut _ as *mut u8,
                    src.len(),
                );
                if COPY {
                    _mm_storeu_si128(partial_buf.0.as_mut_ptr() as *mut __m128i, xmm_crc_part);
                    core::ptr::copy_nonoverlapping(
                        partial_buf.0.as_ptr(),
                        dst.as_mut_ptr(),
                        src.len(),
                    );
                }
                self.partial_fold(xmm_crc_part, src.len());
            }
        }
    }
}
