use std::simd::{ToBitMask, prelude::*};

use crate::{Stage1Parse, macros::static_cast_i32};
#[derive(Debug)]
pub(crate) struct SimdInput {
    v: u8x64,
}

impl Stage1Parse for SimdInput {
    type Utf8Validator = simdutf8::basic::imp::portable::ChunkedUtf8ValidatorImp;
    type SimdRepresentation = u8x64;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn new(ptr: &[u8]) -> Self {
        Self {
            v: u8x64::from_array(*ptr.as_ptr().cast::<[u8; 64]>()),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn compute_quote_mask(quote_bits: u64) -> u64 {
        let mut quote_mask: u64 = quote_bits ^ (quote_bits << 1);
        quote_mask = quote_mask ^ (quote_mask << 2);
        quote_mask = quote_mask ^ (quote_mask << 4);
        quote_mask = quote_mask ^ (quote_mask << 8);
        quote_mask = quote_mask ^ (quote_mask << 16);
        quote_mask = quote_mask ^ (quote_mask << 32);
        quote_mask
    }

    /// a straightforward comparison of a mask against input
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn cmp_mask_against_input(&self, m: u8) -> u64 {
        let mask = u8x64::splat(m);
        self.v.simd_eq(mask).to_bitmask()
    }

    // find all values less than or equal than the content of maxval (using unsigned arithmetic)
    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn unsigned_lteq_against_input(&self, maxval: u8x64) -> u64 {
        self.v.simd_le(maxval).to_bitmask()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn find_whitespace_and_structurals(&self, whitespace: &mut u64, structurals: &mut u64) {
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

        const LOW_NIBBLE_MASK: u8x64 = u8x64::from_array([
            16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1,
            2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
            0, 8, 12, 1, 2, 9, 0, 0,
        ]);
        const HIGH_NIBBLE_MASK: u8x64 = u8x64::from_array([
            8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3,
            2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1,
            0, 0, 0, 3, 2, 1, 0, 0,
        ]);

        let structural_shufti_mask: u8x64 = u8x64::splat(0b0000_0111); // 0x07
        let whitespace_shufti_mask: u8x64 = u8x64::splat(0b0001_1000); // 0x18

        // FIXME: do we need this dance?

        let v32 = i32x16::from_array(std::mem::transmute(*self.v.as_array()));
        let v_shifted = v32 >> i32x16::splat(4);
        let v_shifted = u8x64::from_array(std::mem::transmute(v_shifted));

        // We have to adjust the index here the reason being that while the avx instruction
        // only uses the lower 4 bits for index and the 8th bit for overflow (set to 0)
        // std::simd::swizzle uses all bits 5-8 for overflow, so we need to mask out
        // bit 6, 5 and 7 to get the correct behaviour
        let v_idx_low = self.v & u8x64::splat(0b1000_1111);
        let v_swizzle_low = LOW_NIBBLE_MASK.swizzle_dyn(v_idx_low);
        let v_idx_high = v_shifted & u8x64::splat(0b0000_1111);
        let v_swizzle_high = HIGH_NIBBLE_MASK.swizzle_dyn(v_idx_high);
        let v = v_swizzle_low & v_swizzle_high;

        let tmp: Mask<i8, 64> = (v & structural_shufti_mask).simd_eq(u8x64::splat(0));

        *structurals = !tmp.to_bitmask();

        let tmp_ws = (v & whitespace_shufti_mask).simd_eq(u8x64::splat(0));

        *whitespace = !(tmp_ws.to_bitmask());
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
        let cnt: usize = bits.count_ones() as usize;
        let mut l = base.len();
        let idx_minus_64 = static_cast_i32!(idx.wrapping_sub(64));
        let idx_64_v: i32x8 = i32x8::from_array([
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
            idx_minus_64,
        ]);

        // We're doing some trickery here.
        // We reserve 64 extra entries, because we've at most 64 bit to set
        // then we truncate the base to the next base (that we calculated above)
        // We later indiscriminatory write over the len we set but that's OK
        // since we ensure we reserve the needed space
        base.reserve(64);
        let final_len = l + cnt;

        while bits != 0 {
            let v0 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v1 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v2 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v3 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v4 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v5 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v6 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v7 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);

            let v = i32x8::from_array([v0, v1, v2, v3, v4, v5, v6, v7]);
            let v = idx_64_v + v;
            base.as_mut_ptr()
                .add(l)
                .cast::<[i32; 8]>()
                .write(*v.as_array());
            //
            l += 8;
        }
        // We have written all the data
        base.set_len(final_len);
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    unsafe fn fill_s8(n: i8) -> u8x64 {
        #[allow(clippy::cast_sign_loss)]
        u8x64::splat(n as u8)
    }
}
