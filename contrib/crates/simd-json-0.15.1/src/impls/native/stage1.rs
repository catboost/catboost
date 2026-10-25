#![allow(clippy::cast_lossless, clippy::cast_sign_loss)]

use crate::{Stage1Parse, macros::static_cast_i32};

type V128 = [u8; 16];

fn u8x16_splat(n: u8) -> V128 {
    [n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n]
}

fn v128_and(a: V128, b: V128) -> V128 {
    [
        a[0] & b[0],
        a[1] & b[1],
        a[2] & b[2],
        a[3] & b[3],
        a[4] & b[4],
        a[5] & b[5],
        a[6] & b[6],
        a[7] & b[7],
        a[8] & b[8],
        a[9] & b[9],
        a[10] & b[10],
        a[11] & b[11],
        a[12] & b[12],
        a[13] & b[13],
        a[14] & b[14],
        a[15] & b[15],
    ]
}

fn u8x16_shr(a: V128, n: i32) -> V128 {
    [
        a[0] >> n,
        a[1] >> n,
        a[2] >> n,
        a[3] >> n,
        a[4] >> n,
        a[5] >> n,
        a[6] >> n,
        a[7] >> n,
        a[8] >> n,
        a[9] >> n,
        a[10] >> n,
        a[11] >> n,
        a[12] >> n,
        a[13] >> n,
        a[14] >> n,
        a[15] >> n,
    ]
}

fn u8x16_swizzle(a: V128, s: V128) -> [u8; 16] {
    [
        if s[0] > 0x0f {
            0
        } else {
            a[(s[0] & 0x0f) as usize]
        },
        if s[1] > 0x0f {
            0
        } else {
            a[(s[1] & 0x0f) as usize]
        },
        if s[2] > 0x0f {
            0
        } else {
            a[(s[2] & 0x0f) as usize]
        },
        if s[3] > 0x0f {
            0
        } else {
            a[(s[3] & 0x0f) as usize]
        },
        if s[4] > 0x0f {
            0
        } else {
            a[(s[4] & 0x0f) as usize]
        },
        if s[5] > 0x0f {
            0
        } else {
            a[(s[5] & 0x0f) as usize]
        },
        if s[6] > 0x0f {
            0
        } else {
            a[(s[6] & 0x0f) as usize]
        },
        if s[7] > 0x0f {
            0
        } else {
            a[(s[7] & 0x0f) as usize]
        },
        if s[8] > 0x0f {
            0
        } else {
            a[(s[8] & 0x0f) as usize]
        },
        if s[9] > 0x0f {
            0
        } else {
            a[(s[9] & 0x0f) as usize]
        },
        if s[10] > 0x0f {
            0
        } else {
            a[(s[10] & 0x0f) as usize]
        },
        if s[11] > 0x0f {
            0
        } else {
            a[(s[11] & 0x0f) as usize]
        },
        if s[12] > 0x0f {
            0
        } else {
            a[(s[12] & 0x0f) as usize]
        },
        if s[13] > 0x0f {
            0
        } else {
            a[(s[13] & 0x0f) as usize]
        },
        if s[14] > 0x0f {
            0
        } else {
            a[(s[14] & 0x0f) as usize]
        },
        if s[15] > 0x0f {
            0
        } else {
            a[(s[15] & 0x0f) as usize]
        },
    ]
}

// fn vtstq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
//     [
//         u8::from((a[0] & b[0]) != 0),
//         u8::from((a[1] & b[1]) != 0),
//         u8::from((a[2] & b[2]) != 0),
//         u8::from((a[3] & b[3]) != 0),
//         u8::from((a[4] & b[4]) != 0),
//         u8::from((a[5] & b[5]) != 0),
//         u8::from((a[6] & b[6]) != 0),
//         u8::from((a[7] & b[7]) != 0),
//         u8::from((a[8] & b[8]) != 0),
//         u8::from((a[9] & b[9]) != 0),
//         u8::from((a[10] & b[10]) != 0),
//         u8::from((a[11] & b[11]) != 0),
//         u8::from((a[12] & b[12]) != 0),
//         u8::from((a[13] & b[13]) != 0),
//         u8::from((a[14] & b[14]) != 0),
//         u8::from((a[15] & b[15]) != 0),
//     ]
// }

// fn vqtbl1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
//     let mut r = [0u8; 16];
//     for i in 0..16 {
//         // if the most significant bit of b is set,
//         // then the destination byte is set to 0.
//         if b[i] & 0x80 == 0u8 {
//             r[i] = a[(b[i] % 16) as usize];
//         }
//     }
//     r
// }

// fn vpaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
//     [
//         a[0] + b[0],
//         a[1] + b[1],
//         a[2] + b[2],
//         a[3] + b[3],
//         a[4] + b[4],
//         a[5] + b[5],
//         a[6] + b[6],
//         a[7] + b[7],
//         a[8] + b[8],
//         a[9] + b[9],
//         a[10] + b[10],
//         a[11] + b[11],
//         a[12] + b[12],
//         a[13] + b[13],
//         a[14] + b[14],
//         a[15] + b[15],
//     ]
// }

fn bool_to_u8(b: bool) -> u8 {
    if b { 0xFF } else { 0x00 }
}
fn u8x16_le(a: V128, b: V128) -> V128 {
    [
        bool_to_u8(a[0] <= b[0]),
        bool_to_u8(a[1] <= b[1]),
        bool_to_u8(a[2] <= b[2]),
        bool_to_u8(a[3] <= b[3]),
        bool_to_u8(a[4] <= b[4]),
        bool_to_u8(a[5] <= b[5]),
        bool_to_u8(a[6] <= b[6]),
        bool_to_u8(a[7] <= b[7]),
        bool_to_u8(a[8] <= b[8]),
        bool_to_u8(a[9] <= b[9]),
        bool_to_u8(a[10] <= b[10]),
        bool_to_u8(a[11] <= b[11]),
        bool_to_u8(a[12] <= b[12]),
        bool_to_u8(a[13] <= b[13]),
        bool_to_u8(a[14] <= b[14]),
        bool_to_u8(a[15] <= b[15]),
    ]
}

fn u8x16_eq(a: V128, b: V128) -> V128 {
    [
        bool_to_u8(a[0] == b[0]),
        bool_to_u8(a[1] == b[1]),
        bool_to_u8(a[2] == b[2]),
        bool_to_u8(a[3] == b[3]),
        bool_to_u8(a[4] == b[4]),
        bool_to_u8(a[5] == b[5]),
        bool_to_u8(a[6] == b[6]),
        bool_to_u8(a[7] == b[7]),
        bool_to_u8(a[8] == b[8]),
        bool_to_u8(a[9] == b[9]),
        bool_to_u8(a[10] == b[10]),
        bool_to_u8(a[11] == b[11]),
        bool_to_u8(a[12] == b[12]),
        bool_to_u8(a[13] == b[13]),
        bool_to_u8(a[14] == b[14]),
        bool_to_u8(a[15] == b[15]),
    ]
}

fn u8x16_bitmask(a: V128) -> u16 {
    (a[0] & 0b1000_0000 != 0) as u16
        | (((a[1] & 0b1000_0000 != 0) as u16) << 1)
        | (((a[2] & 0b1000_0000 != 0) as u16) << 2)
        | (((a[3] & 0b1000_0000 != 0) as u16) << 3)
        | (((a[4] & 0b1000_0000 != 0) as u16) << 4)
        | (((a[5] & 0b1000_0000 != 0) as u16) << 5)
        | (((a[6] & 0b1000_0000 != 0) as u16) << 6)
        | (((a[7] & 0b1000_0000 != 0) as u16) << 7)
        | (((a[8] & 0b1000_0000 != 0) as u16) << 8)
        | (((a[9] & 0b1000_0000 != 0) as u16) << 9)
        | (((a[10] & 0b1000_0000 != 0) as u16) << 10)
        | (((a[11] & 0b1000_0000 != 0) as u16) << 11)
        | (((a[12] & 0b1000_0000 != 0) as u16) << 12)
        | (((a[13] & 0b1000_0000 != 0) as u16) << 13)
        | (((a[14] & 0b1000_0000 != 0) as u16) << 14)
        | (((a[15] & 0b1000_0000 != 0) as u16) << 15)
}
// unsafe fn vreinterpretq_u8_s8(a: int8x16_t) -> uint8x16_t {
//     unsafe { mem::transmute(a) }
// }

// #[cfg_attr(not(feature = "no-inline"), inline)]
// pub unsafe fn neon_movemask_bulk(
//     p0: uint8x16_t,
//     p1: uint8x16_t,
//     p2: uint8x16_t,
//     p3: uint8x16_t,
// ) -> u64 {
//     let bit_mask = bit_mask();

//     let t0 = vandq_u8(p0, bit_mask);
//     let t1 = vandq_u8(p1, bit_mask);
//     let t2 = vandq_u8(p2, bit_mask);
//     let t3 = vandq_u8(p3, bit_mask);
//     let sum0 = vpaddq_u8(t0, t1);
//     let sum1 = vpaddq_u8(t2, t3);
//     let sum0 = vpaddq_u8(sum0, sum1);
//     let sum0 = vpaddq_u8(sum0, sum0);

//     vreinterpretq_u64_u8(sum0)[0]
// }

// fn vreinterpretq_u64_u8(a: uint8x16_t) -> uint64x2_t {
//     unsafe { std::mem::transmute(a) }
// }

#[derive(Debug)]
pub(crate) struct SimdInput {
    v0: V128,
    v1: V128,
    v2: V128,
    v3: V128,
}

impl Stage1Parse for SimdInput {
    type Utf8Validator = super::ChunkedUtf8ValidatorImp;
    type SimdRepresentation = V128;
    unsafe fn new(ptr: &[u8]) -> Self {
        unsafe {
            SimdInput {
                v0: *(ptr.as_ptr().cast::<V128>()),
                v1: *(ptr.as_ptr().add(16).cast::<V128>()),
                v2: *(ptr.as_ptr().add(32).cast::<V128>()),
                v3: *(ptr.as_ptr().add(48).cast::<V128>()),
            }
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    unsafe fn compute_quote_mask(quote_bits: u64) -> u64 {
        let mut quote_mask: u64 = quote_bits ^ (quote_bits << 1);
        quote_mask = quote_mask ^ (quote_mask << 2);
        quote_mask = quote_mask ^ (quote_mask << 4);
        quote_mask = quote_mask ^ (quote_mask << 8);
        quote_mask = quote_mask ^ (quote_mask << 16);
        quote_mask = quote_mask ^ (quote_mask << 32);
        quote_mask
    }

    unsafe fn cmp_mask_against_input(&self, m: u8) -> u64 {
        let mask = u8x16_splat(m);
        let cmp_res_0 = u8x16_eq(self.v0, mask);
        let res_0 = u8x16_bitmask(cmp_res_0) as u64;
        let cmp_res_1 = u8x16_eq(self.v1, mask);
        let res_1 = u8x16_bitmask(cmp_res_1) as u64;
        let cmp_res_2 = u8x16_eq(self.v2, mask);
        let res_2 = u8x16_bitmask(cmp_res_2) as u64;
        let cmp_res_3 = u8x16_eq(self.v3, mask);
        let res_3 = u8x16_bitmask(cmp_res_3) as u64;
        res_0 | (res_1 << 16) | (res_2 << 32) | (res_3 << 48)
    }

    unsafe fn unsigned_lteq_against_input(&self, maxval: V128) -> u64 {
        let cmp_res_0 = u8x16_le(self.v0, maxval);
        let res_0 = u8x16_bitmask(cmp_res_0) as u64;
        let cmp_res_1 = u8x16_le(self.v1, maxval);
        let res_1 = u8x16_bitmask(cmp_res_1) as u64;
        let cmp_res_2 = u8x16_le(self.v2, maxval);
        let res_2 = u8x16_bitmask(cmp_res_2) as u64;
        let cmp_res_3 = u8x16_le(self.v3, maxval);
        let res_3 = u8x16_bitmask(cmp_res_3) as u64;
        res_0 | (res_1 << 16) | (res_2 << 32) | (res_3 << 48)
    }

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
        const LOW_NIBBLE_MASK: V128 = [16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0];
        const HIGH_NIBBLE_MASK: V128 = [8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0];

        let structural_shufti_mask = u8x16_splat(0x7);
        let whitespace_shufti_mask = u8x16_splat(0x18);
        let low_nib_and_mask = u8x16_splat(0xf);
        let high_nib_and_mask = u8x16_splat(0x7f);
        let zero_mask = u8x16_splat(0);

        let v_v0 = v128_and(
            u8x16_swizzle(LOW_NIBBLE_MASK, v128_and(self.v0, low_nib_and_mask)),
            u8x16_swizzle(
                HIGH_NIBBLE_MASK,
                v128_and(u8x16_shr(self.v0, 4), high_nib_and_mask),
            ),
        );
        let v_v1 = v128_and(
            u8x16_swizzle(LOW_NIBBLE_MASK, v128_and(self.v1, low_nib_and_mask)),
            u8x16_swizzle(
                HIGH_NIBBLE_MASK,
                v128_and(u8x16_shr(self.v1, 4), high_nib_and_mask),
            ),
        );
        let v_v2 = v128_and(
            u8x16_swizzle(LOW_NIBBLE_MASK, v128_and(self.v2, low_nib_and_mask)),
            u8x16_swizzle(
                HIGH_NIBBLE_MASK,
                v128_and(u8x16_shr(self.v2, 4), high_nib_and_mask),
            ),
        );
        let v_v3 = v128_and(
            u8x16_swizzle(LOW_NIBBLE_MASK, v128_and(self.v3, low_nib_and_mask)),
            u8x16_swizzle(
                HIGH_NIBBLE_MASK,
                v128_and(u8x16_shr(self.v3, 4), high_nib_and_mask),
            ),
        );
        let tmp_v0 = u8x16_eq(v128_and(v_v0, structural_shufti_mask), zero_mask);
        let tmp_v1 = u8x16_eq(v128_and(v_v1, structural_shufti_mask), zero_mask);
        let tmp_v2 = u8x16_eq(v128_and(v_v2, structural_shufti_mask), zero_mask);
        let tmp_v3 = u8x16_eq(v128_and(v_v3, structural_shufti_mask), zero_mask);

        let structural_res_0 = u8x16_bitmask(tmp_v0) as u64;
        let structural_res_1 = u8x16_bitmask(tmp_v1) as u64;
        let structural_res_2 = u8x16_bitmask(tmp_v2) as u64;
        let structural_res_3 = u8x16_bitmask(tmp_v3) as u64;

        *structurals = !(structural_res_0
            | (structural_res_1 << 16)
            | (structural_res_2 << 32)
            | (structural_res_3 << 48));

        let tmp_ws_v0 = u8x16_eq(v128_and(v_v0, whitespace_shufti_mask), zero_mask);
        let tmp_ws_v1 = u8x16_eq(v128_and(v_v1, whitespace_shufti_mask), zero_mask);
        let tmp_ws_v2 = u8x16_eq(v128_and(v_v2, whitespace_shufti_mask), zero_mask);
        let tmp_ws_v3 = u8x16_eq(v128_and(v_v3, whitespace_shufti_mask), zero_mask);

        let ws_res_0 = u8x16_bitmask(tmp_ws_v0) as u64;
        let ws_res_1 = u8x16_bitmask(tmp_ws_v1) as u64;
        let ws_res_2 = u8x16_bitmask(tmp_ws_v2) as u64;
        let ws_res_3 = u8x16_bitmask(tmp_ws_v3) as u64;

        *whitespace = !(ws_res_0 | (ws_res_1 << 16) | (ws_res_2 << 32) | (ws_res_3 << 48));
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap, clippy::cast_ptr_alignment)]
    unsafe fn flatten_bits(base: &mut Vec<u32>, idx: u32, mut bits: u64) {
        let cnt: usize = bits.count_ones() as usize;
        let mut l = base.len();
        let idx_minus_64 = idx.wrapping_sub(64);
        let idx_64_v: [i32; 4] = unsafe {
            [
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
            ]
        };

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

            let v: [i32; 4] = [
                idx_64_v[0] + v0,
                idx_64_v[1] + v1,
                idx_64_v[2] + v2,
                idx_64_v[3] + v3,
            ];
            unsafe { write_fn(base.as_mut_ptr().add(l).cast::<[i32; 4]>(), v) };
            l += 4;
        }
        // We have written all the data
        unsafe { base.set_len(final_len) };
    }

    unsafe fn fill_s8(n: i8) -> V128 {
        u8x16_splat(n as u8)
    }
}
