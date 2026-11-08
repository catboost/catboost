//! # Safety
//!
//! The functions in this module should only be executed on loongarch machines with the LSX
//! extension.
use core::arch::loongarch64::{
    lsx_vadd_h, lsx_vadd_w, lsx_vaddwev_h_bu, lsx_vaddwev_w_hu, lsx_vaddwod_h_bu, lsx_vaddwod_w_hu,
    lsx_vbsll_v, lsx_vbsrl_v, lsx_vexth_hu_bu, lsx_vexth_wu_hu, lsx_vinsgr2vr_w, lsx_vld, lsx_vldi,
    lsx_vmadd_w, lsx_vpickve2gr_w, lsx_vslli_w, lsx_vsllwil_wu_hu, m128i,
};

use crate::adler32::{
    generic::{adler32_len_1, adler32_len_16},
    BASE, NMAX,
};

const TAPS: [m128i; 16] = unsafe {
    core::mem::transmute::<[u32; 64], [m128i; 16]>([
        64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
        41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
        18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    ])
};

pub fn adler32_lsx(adler: u32, src: &[u8]) -> u32 {
    assert!(crate::cpu_features::is_enabled_lsx());
    // SAFETY: the assertion above ensures this code is not executed unless the CPU has LSX.
    unsafe { adler32_lsx_internal(adler, src) }
}

#[target_feature(enable = "lsx")]
unsafe fn adler32_lsx_internal(mut adler: u32, buf: &[u8]) -> u32 {
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

    const _: () = assert!(core::mem::align_of::<m128i>() == 16);
    let (before, middle, after) = unsafe { buf.align_to::<m128i>() };

    pair = handle_tail(pair, before);

    // NOTE: zlib-ng adjusts the chunk size to account for already having consumed some input. We
    // instead just mod the pair by BASE and then use the full chunk width below. Failing to do so
    // can result in overflow.
    pair.0 %= BASE;
    pair.1 %= BASE;

    for chunk in middle.chunks(NMAX as usize / core::mem::size_of::<m128i>()) {
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

#[target_feature(enable = "lsx")]
unsafe fn accum32(s: (u32, u32), buf: &[m128i]) -> (u32, u32) {
    let mut adacc = lsx_vldi::<0>();
    let mut s2acc = lsx_vldi::<0>();

    adacc = lsx_vinsgr2vr_w::<0>(adacc, s.0 as i32);
    s2acc = lsx_vinsgr2vr_w::<0>(s2acc, s.1 as i32);

    let mut s3acc = lsx_vldi::<0>();
    let mut adacc_prev = adacc;

    let mut s2_0 = lsx_vldi::<0>();
    let mut s2_1 = lsx_vldi::<0>();
    let mut s2_2 = lsx_vldi::<0>();
    let mut s2_3 = lsx_vldi::<0>();

    let mut s2_4 = lsx_vldi::<0>();
    let mut s2_5 = lsx_vldi::<0>();
    let mut s2_6 = lsx_vldi::<0>();
    let mut s2_7 = lsx_vldi::<0>();

    let mut it = buf.chunks_exact(4);

    for chunk in &mut it {
        // SAFETY: the chunks_exact iterator ensures chunk always references a 16x4 block within buf.
        let d0 = unsafe { lsx_vld::<0>(chunk.as_ptr() as *const i8) };
        let d1 = unsafe { lsx_vld::<16>(chunk.as_ptr() as *const i8) };
        let d2 = unsafe { lsx_vld::<32>(chunk.as_ptr() as *const i8) };
        let d3 = unsafe { lsx_vld::<48>(chunk.as_ptr() as *const i8) };

        // Unfortunately it doesn't look like there's a direct sum 8 bit to 32
        // bit instruction, we'll have to make due summing to 16 bits first
        let e0 = lsx_vaddwod_h_bu(d0, d2);
        let e1 = lsx_vaddwev_h_bu(d0, d2);
        let e2 = lsx_vaddwod_h_bu(d1, d3);
        let e3 = lsx_vaddwev_h_bu(d1, d3);

        let f0 = lsx_vadd_h(e0, e1);
        let f1 = lsx_vadd_h(e2, e3);

        let g0 = lsx_vaddwod_w_hu(f0, f1);
        let g1 = lsx_vaddwev_w_hu(f0, f1);

        adacc = lsx_vadd_w(adacc, g0);
        adacc = lsx_vadd_w(adacc, g1);
        s3acc = lsx_vadd_w(s3acc, adacc_prev);

        // If we do straight widening additions to the 16 bit values, we don't incur
        // the usual penalties of a pairwise add. We can defer the multiplications
        // until the very end. These will not overflow because we are incurring at
        // most 408 loop iterations (NMAX / 64), and a given lane is only going to be
        // summed into once. This means for the maximum input size, the largest value
        // we will see is 255 * 102 = 26010, safely under uint16 max

        s2_0 = lsx_vadd_h(s2_0, lsx_vexth_hu_bu(lsx_vbsll_v::<8>(d0)));
        s2_1 = lsx_vadd_h(s2_1, lsx_vexth_hu_bu(d0));
        s2_2 = lsx_vadd_h(s2_2, lsx_vexth_hu_bu(lsx_vbsll_v::<8>(d1)));
        s2_3 = lsx_vadd_h(s2_3, lsx_vexth_hu_bu(d1));
        s2_4 = lsx_vadd_h(s2_4, lsx_vexth_hu_bu(lsx_vbsll_v::<8>(d2)));
        s2_5 = lsx_vadd_h(s2_5, lsx_vexth_hu_bu(d2));
        s2_6 = lsx_vadd_h(s2_6, lsx_vexth_hu_bu(lsx_vbsll_v::<8>(d3)));
        s2_7 = lsx_vadd_h(s2_7, lsx_vexth_hu_bu(d3));

        adacc_prev = adacc;
    }

    s3acc = lsx_vslli_w::<6>(s3acc);

    let remainder = it.remainder();

    if !remainder.is_empty() {
        let mut s3acc_0 = lsx_vldi::<0>();
        let zero = lsx_vldi::<0>();
        for d0 in remainder.iter().copied() {
            let adler = lsx_vadd_h(lsx_vaddwev_h_bu(d0, zero), lsx_vaddwod_h_bu(d0, zero));
            s2_6 = lsx_vadd_w(s2_6, lsx_vexth_hu_bu(lsx_vbsll_v::<8>(d0)));
            s2_7 = lsx_vadd_w(s2_7, lsx_vexth_hu_bu(d0));
            adacc = lsx_vadd_w(
                adacc,
                lsx_vadd_w(lsx_vaddwev_w_hu(adler, zero), lsx_vaddwod_w_hu(adler, zero)),
            );
            s3acc_0 = lsx_vadd_w(s3acc_0, adacc_prev);
            adacc_prev = adacc;
        }

        s3acc_0 = lsx_vslli_w::<4>(s3acc_0);
        s3acc = lsx_vadd_w(s3acc_0, s3acc);
    }

    let mut s2acc_0 = lsx_vldi::<0>();
    let mut s2acc_1 = lsx_vldi::<0>();
    let mut s2acc_2 = lsx_vldi::<0>();

    s2acc = lsx_vmadd_w(s2acc, TAPS[1], lsx_vexth_wu_hu(s2_0));
    s2acc_0 = lsx_vmadd_w(s2acc_0, TAPS[0], lsx_vsllwil_wu_hu::<0>(s2_0));
    s2acc_1 = lsx_vmadd_w(s2acc_1, TAPS[3], lsx_vexth_wu_hu(s2_1));
    s2acc_2 = lsx_vmadd_w(s2acc_2, TAPS[2], lsx_vsllwil_wu_hu::<0>(s2_1));

    s2acc = lsx_vmadd_w(s2acc, TAPS[5], lsx_vexth_wu_hu(s2_2));
    s2acc_0 = lsx_vmadd_w(s2acc_0, TAPS[4], lsx_vsllwil_wu_hu::<0>(s2_2));
    s2acc_1 = lsx_vmadd_w(s2acc_1, TAPS[7], lsx_vexth_wu_hu(s2_3));
    s2acc_2 = lsx_vmadd_w(s2acc_2, TAPS[6], lsx_vsllwil_wu_hu::<0>(s2_3));

    s2acc = lsx_vmadd_w(s2acc, TAPS[9], lsx_vexth_wu_hu(s2_4));
    s2acc_0 = lsx_vmadd_w(s2acc_0, TAPS[8], lsx_vsllwil_wu_hu::<0>(s2_4));
    s2acc_1 = lsx_vmadd_w(s2acc_1, TAPS[11], lsx_vexth_wu_hu(s2_5));
    s2acc_2 = lsx_vmadd_w(s2acc_2, TAPS[10], lsx_vsllwil_wu_hu::<0>(s2_5));

    s2acc = lsx_vmadd_w(s2acc, TAPS[13], lsx_vexth_wu_hu(s2_6));
    s2acc_0 = lsx_vmadd_w(s2acc_0, TAPS[12], lsx_vsllwil_wu_hu::<0>(s2_6));
    s2acc_1 = lsx_vmadd_w(s2acc_1, TAPS[15], lsx_vexth_wu_hu(s2_7));
    s2acc_2 = lsx_vmadd_w(s2acc_2, TAPS[14], lsx_vsllwil_wu_hu::<0>(s2_7));

    s2acc = lsx_vadd_w(s2acc_0, s2acc);
    s2acc_2 = lsx_vadd_w(s2acc_1, s2acc_2);
    s2acc = lsx_vadd_w(s2acc, s2acc_2);

    let s2acc = lsx_vadd_w(s2acc, s3acc);

    let mut adacc2 = lsx_vadd_w(adacc, lsx_vbsrl_v::<8>(adacc));
    adacc2 = lsx_vadd_w(adacc2, lsx_vbsrl_v::<4>(adacc2));

    let mut s2acc2 = lsx_vadd_w(s2acc, lsx_vbsrl_v::<8>(s2acc));
    s2acc2 = lsx_vadd_w(s2acc2, lsx_vbsrl_v::<4>(s2acc2));

    (
        lsx_vpickve2gr_w::<0>(adacc2) as u32,
        lsx_vpickve2gr_w::<0>(s2acc2) as u32,
    )
}

#[cfg(all(test, feature = "std", any(miri, target_feature = "lsx")))]
mod tests {
    use super::*;

    quickcheck::quickcheck! {
        fn adler32_lsx_is_adler32_rust(v: Vec<u8>, start: u32) -> bool {
            let lsx = adler32_lsx(start, &v);
            let rust = crate::adler32::generic::adler32_rust(start, &v);

            rust == lsx
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
                let lsx = adler32_lsx(start, &INPUT[i..]);
                let rust = crate::adler32::generic::adler32_rust(start, &INPUT[i..]);

                assert_eq!(lsx, rust, "offset = {i}, start = {start}");
            }
        }
    }

    #[test]
    fn large_input() {
        const DEFAULT: &[u8] = include_bytes!("../deflate/test-data/paper-100k.pdf");

        let lsx = adler32_lsx(42, DEFAULT);
        let rust = crate::adler32::generic::adler32_rust(42, DEFAULT);

        assert_eq!(lsx, rust);
    }

    // Regression test for a bug where adler32 would not modulo by the base early enough,
    // specifically it ignored the `before` slice in when to mod.
    #[test]
    fn carry_in_with_unaligned_before_no_overflow() {
        let backing = vec![0xffu8; 5568];
        let buf: &[u8] = &backing[1..1 + 5567];
        let start: u32 = 0xa4c1_fb51;
        let lsx = adler32_lsx(start, buf);
        let rust = crate::adler32::generic::adler32_rust(start, buf);
        assert_eq!(lsx, rust);
    }
}
