//! # Safety
//!
//! The functions in this module should only be executed on aarch64 machines with the Neon
//! extension.
use core::arch::aarch64::{
    uint16x8_t, uint16x8x2_t, uint16x8x4_t, uint8x16_t, vaddq_u32, vaddw_high_u8, vaddw_u8,
    vdupq_n_u16, vdupq_n_u32, vextq_u8, vget_high_u32, vget_lane_u32, vget_low_u16, vget_low_u32,
    vget_low_u8, vld1q_u8_x4, vmlal_high_u16, vmlal_u16, vpadalq_u16, vpadalq_u8, vpadd_u32,
    vpaddlq_u8, vsetq_lane_u32, vshlq_n_u32,
};

use crate::adler32::{
    generic::{adler32_len_1, adler32_len_16},
    BASE, NMAX,
};

const TAPS: [uint16x8x4_t; 2] = unsafe {
    core::mem::transmute::<[u16; 64], [uint16x8x4_t; 2]>([
        64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
        41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
        18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ])
};

pub fn adler32_neon(adler: u32, buf: &[u8]) -> u32 {
    assert!(crate::cpu_features::is_enabled_neon());
    // SAFETY: the assertion above ensures this code is not executed unless the CPU has Neon
    // extensions.
    unsafe { adler32_neon_internal(adler, buf) }
}

#[target_feature(enable = "neon")]
unsafe fn adler32_neon_internal(mut adler: u32, buf: &[u8]) -> u32 {
    /* split Adler-32 into component sums */
    let sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if buf.len() == 1 {
        return adler32_len_1(adler, buf, sum2);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if buf.is_empty() {
        return adler | (sum2 << 16);
    }

    /* in case short lengths are provided, keep it somewhat fast */
    if buf.len() < 16 {
        return adler32_len_16(adler, buf, sum2);
    }

    // Split Adler-32 into component sums, it can be supplied by the caller sites (e.g. in a PNG file).
    let mut pair = (adler, sum2);

    // If memory is not SIMD aligned, do scalar sums to an aligned
    // offset, provided that doing so doesn't completely eliminate
    // SIMD operation. Aligned loads are still faster on ARM, even
    // though there's no explicit aligned load instruction
    const _: () = assert!(core::mem::align_of::<uint8x16_t>() == 16);
    let (before, middle, after) = unsafe { buf.align_to::<uint8x16_t>() };

    pair = handle_tail(pair, before);

    // NOTE: zlib-ng adjusts the chunk size to account for already having consumed some input. We
    // instead just mod the pair by BASE and then use the full chunk width below. Failing to do so
    // can result in overflow.
    pair.0 %= BASE;
    pair.1 %= BASE;

    for chunk in middle.chunks(NMAX as usize / core::mem::size_of::<uint8x16_t>()) {
        pair = unsafe { accum32(pair, chunk) };
        pair.0 %= BASE;
        pair.1 %= BASE;
    }

    if !after.is_empty() {
        pair = handle_tail(pair, after);
        pair.0 %= BASE;
        pair.1 %= BASE;
    }

    // D = B * 65536 + A, see: https://en.wikipedia.org/wiki/Adler-32.
    (pair.1 << 16) | pair.0
}

fn handle_tail(mut pair: (u32, u32), buf: &[u8]) -> (u32, u32) {
    for x in buf {
        pair.0 += *x as u32;
        pair.1 += pair.0;
    }

    pair
}

#[target_feature(enable = "neon")]
unsafe fn accum32(s: (u32, u32), buf: &[uint8x16_t]) -> (u32, u32) {
    // No target features 1.1 yet...
    unsafe {
        let mut adacc = vdupq_n_u32(0);
        let mut s2acc = vdupq_n_u32(0);

        adacc = vsetq_lane_u32(s.0, adacc, 0);
        s2acc = vsetq_lane_u32(s.1, s2acc, 0);

        let mut s3acc = vdupq_n_u32(0);
        let mut adacc_prev = adacc;

        let mut s2_0 = vdupq_n_u16(0);
        let mut s2_1 = vdupq_n_u16(0);
        let mut s2_2 = vdupq_n_u16(0);
        let mut s2_3 = vdupq_n_u16(0);

        let mut s2_4 = vdupq_n_u16(0);
        let mut s2_5 = vdupq_n_u16(0);
        let mut s2_6 = vdupq_n_u16(0);
        let mut s2_7 = vdupq_n_u16(0);

        let mut it = buf.chunks_exact(4);

        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn swap_64bit_lanes_when_be(v: uint8x16_t) -> uint8x16_t {
            #[rustfmt::skip]
            crate::cfg_select! {
                target_endian = "big" => { vextq_u8(v, v, 8) }
                target_endian = "little" => { v }
            }
        }

        for chunk in &mut it {
            // SAFETY: the chunks_exact iterator ensures chunk always references a 16x4 block within buf.
            let d0_d3 = {
                let mut tmp = unsafe { vld1q_u8_x4(chunk.as_ptr() as *const u8) };

                tmp.0 = swap_64bit_lanes_when_be(tmp.0);
                tmp.1 = swap_64bit_lanes_when_be(tmp.1);
                tmp.2 = swap_64bit_lanes_when_be(tmp.2);
                tmp.3 = swap_64bit_lanes_when_be(tmp.3);

                tmp
            };

            // Unfortunately it doesn't look like there's a direct sum 8 bit to 32
            // bit instruction, we'll have to make due summing to 16 bits first
            let hsum = uint16x8x2_t(vpaddlq_u8(d0_d3.0), vpaddlq_u8(d0_d3.1));

            let hsum_fold = uint16x8x2_t(vpadalq_u8(hsum.0, d0_d3.2), vpadalq_u8(hsum.1, d0_d3.3));

            adacc = vpadalq_u16(adacc, hsum_fold.0);
            s3acc = vaddq_u32(s3acc, adacc_prev);
            adacc = vpadalq_u16(adacc, hsum_fold.1);

            // If we do straight widening additions to the 16 bit values, we don't incur
            // the usual penalties of a pairwise add. We can defer the multiplications
            // until the very end. These will not overflow because we are incurring at
            // most 408 loop iterations (NMAX / 64), and a given lane is only going to be
            // summed into once. This means for the maximum input size, the largest value
            // we will see is 255 * 102 = 26010, safely under uint16 max
            s2_0 = vaddw_u8(s2_0, vget_low_u8(d0_d3.0));
            s2_1 = vaddw_high_u8(s2_1, d0_d3.0);
            s2_2 = vaddw_u8(s2_2, vget_low_u8(d0_d3.1));
            s2_3 = vaddw_high_u8(s2_3, d0_d3.1);
            s2_4 = vaddw_u8(s2_4, vget_low_u8(d0_d3.2));
            s2_5 = vaddw_high_u8(s2_5, d0_d3.2);
            s2_6 = vaddw_u8(s2_6, vget_low_u8(d0_d3.3));
            s2_7 = vaddw_high_u8(s2_7, d0_d3.3);

            adacc_prev = adacc;
        }

        s3acc = vshlq_n_u32(s3acc, 6);

        let remainder = it.remainder();

        if !remainder.is_empty() {
            let mut s3acc_0 = vdupq_n_u32(0);
            for d0 in remainder.iter().copied() {
                let d0 = swap_64bit_lanes_when_be(d0);
                let adler: uint16x8_t = vpaddlq_u8(d0);
                s2_6 = vaddw_u8(s2_6, vget_low_u8(d0));
                s2_7 = vaddw_high_u8(s2_7, d0);
                adacc = vpadalq_u16(adacc, adler);
                s3acc_0 = vaddq_u32(s3acc_0, adacc_prev);
                adacc_prev = adacc;
            }

            s3acc_0 = vshlq_n_u32(s3acc_0, 4);
            s3acc = vaddq_u32(s3acc_0, s3acc);
        }

        let t0_t3 = TAPS[0];
        let t4_t7 = TAPS[1];

        let mut s2acc_0 = vdupq_n_u32(0);
        let mut s2acc_1 = vdupq_n_u32(0);
        let mut s2acc_2 = vdupq_n_u32(0);

        s2acc = vmlal_high_u16(s2acc, t0_t3.0, s2_0);
        s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t0_t3.0), vget_low_u16(s2_0));
        s2acc_1 = vmlal_high_u16(s2acc_1, t0_t3.1, s2_1);
        s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t0_t3.1), vget_low_u16(s2_1));

        s2acc = vmlal_high_u16(s2acc, t0_t3.2, s2_2);
        s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t0_t3.2), vget_low_u16(s2_2));
        s2acc_1 = vmlal_high_u16(s2acc_1, t0_t3.3, s2_3);
        s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t0_t3.3), vget_low_u16(s2_3));

        s2acc = vmlal_high_u16(s2acc, t4_t7.0, s2_4);
        s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t4_t7.0), vget_low_u16(s2_4));
        s2acc_1 = vmlal_high_u16(s2acc_1, t4_t7.1, s2_5);
        s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t4_t7.1), vget_low_u16(s2_5));

        s2acc = vmlal_high_u16(s2acc, t4_t7.2, s2_6);
        s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t4_t7.2), vget_low_u16(s2_6));
        s2acc_1 = vmlal_high_u16(s2acc_1, t4_t7.3, s2_7);
        s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t4_t7.3), vget_low_u16(s2_7));

        s2acc = vaddq_u32(s2acc_0, s2acc);
        s2acc_2 = vaddq_u32(s2acc_1, s2acc_2);
        s2acc = vaddq_u32(s2acc, s2acc_2);

        let s2acc = vaddq_u32(s2acc, s3acc);
        let adacc2 = vpadd_u32(vget_low_u32(adacc), vget_high_u32(adacc));
        let s2acc2 = vpadd_u32(vget_low_u32(s2acc), vget_high_u32(s2acc));
        let as_ = vpadd_u32(adacc2, s2acc2);

        (vget_lane_u32(as_, 0), vget_lane_u32(as_, 1))
    }
}

#[cfg(all(test, feature = "std", any(miri, target_feature = "neon")))]
mod tests {
    use super::*;

    quickcheck::quickcheck! {
        fn adler32_neon_is_adler32_rust(v: Vec<u8>, start: u32) -> bool {
            let neon = adler32_neon(start, &v);
            let rust = crate::adler32::generic::adler32_rust(start, &v);

            rust == neon
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
                let neon = adler32_neon(start, &INPUT[i..]);
                let rust = crate::adler32::generic::adler32_rust(start, &INPUT[i..]);

                assert_eq!(neon, rust, "offset = {i}, start = {start}");
            }
        }
    }

    #[test]
    fn large_input() {
        const DEFAULT: &[u8] = include_bytes!("../deflate/test-data/paper-100k.pdf");

        let neon = adler32_neon(42, DEFAULT);
        let rust = crate::adler32::generic::adler32_rust(42, DEFAULT);

        assert_eq!(neon, rust);
    }

    // Regression test for a bug where adler32 would not modulo by the base early enough,
    // specifically it ignored the `before` slice in when to mod.
    #[test]
    fn carry_in_with_unaligned_before_no_overflow() {
        let backing = vec![0xffu8; 5568];
        let buf: &[u8] = &backing[1..1 + 5567];
        let start: u32 = 0xa4c1_fb51;
        let neon = adler32_neon(start, buf);
        let rust = crate::adler32::generic::adler32_rust(start, buf);
        assert_eq!(neon, rust);
    }
}
