use crate::Stage1Parse;
use std::arch::aarch64::{
    int8x16_t, int32x4_t, uint8x16_t, vaddq_s32, vandq_u8, vceqq_u8, vcleq_u8, vdupq_n_s8,
    vgetq_lane_u64, vld1q_u8, vmovq_n_u8, vpaddq_u8, vqtbl1q_u8, vreinterpretq_u8_s8,
    vreinterpretq_u64_u8, vshrq_n_u8, vtstq_u8,
};
use std::mem;

// NEON-SPECIFIC
#[cfg_attr(not(feature = "no-inline"), inline)]
pub(crate) unsafe fn bit_mask() -> uint8x16_t {
    unsafe {
        std::mem::transmute([
            0x01u8, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40,
            0x80,
        ])
    }
}

#[cfg_attr(not(feature = "no-inline"), inline)]
pub unsafe fn neon_movemask_bulk(
    p0: uint8x16_t,
    p1: uint8x16_t,
    p2: uint8x16_t,
    p3: uint8x16_t,
) -> u64 {
    unsafe {
        let bit_mask = bit_mask();

        let t0 = vandq_u8(p0, bit_mask);
        let t1 = vandq_u8(p1, bit_mask);
        let t2 = vandq_u8(p2, bit_mask);
        let t3 = vandq_u8(p3, bit_mask);
        let sum0 = vpaddq_u8(t0, t1);
        let sum1 = vpaddq_u8(t2, t3);
        let sum0 = vpaddq_u8(sum0, sum1);
        let sum0 = vpaddq_u8(sum0, sum0);

        vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0)
    }
}

// /NEON-SPECIFIC

//pub const SIMDJSON_PADDING: usize = mem::size_of::<uint8x16_t>() * 4;
//pub const SIMDINPUT_LENGTH: usize = 64;

#[derive(Debug)]
pub(crate) struct SimdInput {
    v0: uint8x16_t,
    v1: uint8x16_t,
    v2: uint8x16_t,
    v3: uint8x16_t,
}

impl Stage1Parse for SimdInput {
    type Utf8Validator = simdutf8::basic::imp::aarch64::neon::ChunkedUtf8ValidatorImp;
    type SimdRepresentation = int8x16_t;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn new(ptr: &[u8]) -> Self {
        unsafe {
            Self {
                v0: vld1q_u8(ptr.as_ptr().cast::<u8>()),
                v1: vld1q_u8(ptr.as_ptr().add(16).cast::<u8>()),
                v2: vld1q_u8(ptr.as_ptr().add(32).cast::<u8>()),
                v3: vld1q_u8(ptr.as_ptr().add(48).cast::<u8>()),
            }
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn compute_quote_mask(mut quote_bits: u64) -> u64 {
        quote_bits ^= quote_bits << 1;
        quote_bits ^= quote_bits << 2;
        quote_bits ^= quote_bits << 4;
        quote_bits ^= quote_bits << 8;
        quote_bits ^= quote_bits << 16;
        quote_bits ^= quote_bits << 32;
        quote_bits
    }

    /// a straightforward comparison of a mask against input
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn cmp_mask_against_input(&self, m: u8) -> u64 {
        unsafe {
            let mask: uint8x16_t = vmovq_n_u8(m);
            let cmp_res_0: uint8x16_t = vceqq_u8(self.v0, mask);
            let cmp_res_1: uint8x16_t = vceqq_u8(self.v1, mask);
            let cmp_res_2: uint8x16_t = vceqq_u8(self.v2, mask);
            let cmp_res_3: uint8x16_t = vceqq_u8(self.v3, mask);

            neon_movemask_bulk(cmp_res_0, cmp_res_1, cmp_res_2, cmp_res_3)
        }
    }

    // find all values less than or equal than the content of maxval (using unsigned arithmetic)
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn unsigned_lteq_against_input(&self, maxval: int8x16_t) -> u64 {
        unsafe {
            let maxval = vreinterpretq_u8_s8(maxval);
            let cmp_res_0: uint8x16_t = vcleq_u8(self.v0, maxval);
            let cmp_res_1: uint8x16_t = vcleq_u8(self.v1, maxval);
            let cmp_res_2: uint8x16_t = vcleq_u8(self.v2, maxval);
            let cmp_res_3: uint8x16_t = vcleq_u8(self.v3, maxval);
            neon_movemask_bulk(cmp_res_0, cmp_res_1, cmp_res_2, cmp_res_3)
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    unsafe fn find_whitespace_and_structurals(&self, whitespace: &mut u64, structurals: &mut u64) {
        unsafe {
            // do a 'shufti' to detect structural JSON characters
            // they are
            // * `{` 0x7b
            // * `}` 0x7d
            // * `:` 0x3a
            // * `[` 0x5b
            // * `]` 0x5d
            // * `,` 0x2c
            // these go into the first 3 buckets of the comparison (1/2/4)

            // we are also interested in the four whitespace characters:
            // * space 0x20
            // * linefeed 0x0a
            // * horizontal tab 0x09
            // * carriage return 0x0d
            // these go into the next 2 buckets of the comparison (8/16)

            const LOW_NIBBLE_MASK: uint8x16_t = unsafe {
                std::mem::transmute([16u8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0])
            };

            const HIGH_NIBBLE_MASK: uint8x16_t =
                unsafe { std::mem::transmute([8u8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0]) };

            let structural_shufti_mask: uint8x16_t = vmovq_n_u8(0x7);
            let whitespace_shufti_mask: uint8x16_t = vmovq_n_u8(0x18);
            let low_nib_and_mask: uint8x16_t = vmovq_n_u8(0xf);

            let nib_0_lo: uint8x16_t = vandq_u8(self.v0, low_nib_and_mask);
            let nib_0_hi: uint8x16_t = vshrq_n_u8(self.v0, 4);
            let shuf_0_lo: uint8x16_t = vqtbl1q_u8(LOW_NIBBLE_MASK, nib_0_lo);
            let shuf_0_hi: uint8x16_t = vqtbl1q_u8(HIGH_NIBBLE_MASK, nib_0_hi);
            let v_0: uint8x16_t = vandq_u8(shuf_0_lo, shuf_0_hi);

            let nib_1_lo: uint8x16_t = vandq_u8(self.v1, low_nib_and_mask);
            let nib_1_hi: uint8x16_t = vshrq_n_u8(self.v1, 4);
            let shuf_1_lo: uint8x16_t = vqtbl1q_u8(LOW_NIBBLE_MASK, nib_1_lo);
            let shuf_1_hi: uint8x16_t = vqtbl1q_u8(HIGH_NIBBLE_MASK, nib_1_hi);
            let v_1: uint8x16_t = vandq_u8(shuf_1_lo, shuf_1_hi);

            let nib_2_lo: uint8x16_t = vandq_u8(self.v2, low_nib_and_mask);
            let nib_2_hi: uint8x16_t = vshrq_n_u8(self.v2, 4);
            let shuf_2_lo: uint8x16_t = vqtbl1q_u8(LOW_NIBBLE_MASK, nib_2_lo);
            let shuf_2_hi: uint8x16_t = vqtbl1q_u8(HIGH_NIBBLE_MASK, nib_2_hi);
            let v_2: uint8x16_t = vandq_u8(shuf_2_lo, shuf_2_hi);

            let nib_3_lo: uint8x16_t = vandq_u8(self.v3, low_nib_and_mask);
            let nib_3_hi: uint8x16_t = vshrq_n_u8(self.v3, 4);
            let shuf_3_lo: uint8x16_t = vqtbl1q_u8(LOW_NIBBLE_MASK, nib_3_lo);
            let shuf_3_hi: uint8x16_t = vqtbl1q_u8(HIGH_NIBBLE_MASK, nib_3_hi);
            let v_3: uint8x16_t = vandq_u8(shuf_3_lo, shuf_3_hi);

            let tmp_0: uint8x16_t = vtstq_u8(v_0, structural_shufti_mask);
            let tmp_1: uint8x16_t = vtstq_u8(v_1, structural_shufti_mask);
            let tmp_2: uint8x16_t = vtstq_u8(v_2, structural_shufti_mask);
            let tmp_3: uint8x16_t = vtstq_u8(v_3, structural_shufti_mask);
            *structurals = neon_movemask_bulk(tmp_0, tmp_1, tmp_2, tmp_3);

            let tmp_ws_v0: uint8x16_t = vtstq_u8(v_0, whitespace_shufti_mask);
            let tmp_ws_v1: uint8x16_t = vtstq_u8(v_1, whitespace_shufti_mask);
            let tmp_ws_v2: uint8x16_t = vtstq_u8(v_2, whitespace_shufti_mask);
            let tmp_ws_v3: uint8x16_t = vtstq_u8(v_3, whitespace_shufti_mask);
            *whitespace = neon_movemask_bulk(tmp_ws_v0, tmp_ws_v1, tmp_ws_v2, tmp_ws_v3);
        }
    }

    // flatten out values in 'bits' assuming that they are are to have values of idx
    // plus their position in the bitvector, and store these indexes at
    // base_ptr[base] incrementing base as we go
    // will potentially store extra values beyond end of valid bits, so base_ptr
    // needs to be large enough to handle this
    //TODO: usize was u32 here does this matter?
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap, clippy::cast_ptr_alignment)]
    unsafe fn flatten_bits(base: &mut Vec<u32>, idx: u32, mut bits: u64) {
        unsafe {
            let cnt: usize = bits.count_ones() as usize;
            let mut l = base.len();
            let idx_minus_64 = idx.wrapping_sub(64);
            let idx_64_v = mem::transmute::<[i32; 4], int32x4_t>([
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
            ]);

            // We're doing some trickery here.
            // We reserve 64 extra entries, because we've at most 64 bit to set
            // then we truncate the base to the next base (that we calculated above)
            // We later indiscriminatory write over the len we set but that's OK
            // since we ensure we reserve the needed space
            base.reserve(64);
            let final_len = l + cnt;

            let is_unaligned = l % 4 != 0;
            let write_fn = if is_unaligned {
                std::ptr::write_unaligned
            } else {
                std::ptr::write
            };

            while bits != 0 {
                let v0 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v1 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v2 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);
                let v3 = bits.trailing_zeros() as i32;
                bits &= bits.wrapping_sub(1);

                let v: int32x4_t = mem::transmute([v0, v1, v2, v3]);
                let v: int32x4_t = vaddq_s32(idx_64_v, v);
                write_fn(base.as_mut_ptr().add(l).cast::<int32x4_t>(), v);
                l += 4;
            }
            // We have written all the data
            base.set_len(final_len);
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn fill_s8(n: i8) -> int8x16_t {
        unsafe { vdupq_n_s8(n) }
    }
}
