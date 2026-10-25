use crate::AtoiSimdError;
use ::core::{arch::aarch64::*, convert::TryInto};
use debug_unsafe::slice::SliceGetter;

pub(crate) const SHORT: usize = 4;

const CHAR_MAX: u8 = b'9';
const CHAR_MIN: u8 = b'0';

/* macro_rules! load_lanes {
    ($s:expr, $($i:expr),*) => {
        {
            let mut res = vdupq_n_u8(0);
            // let mut res = vcombine_u8(vcreate_u8(0),vcreate_u8(0));
            $(
                // res = vld1q_lane_u8($s.get_unchecked($i) as *const u8, res, $i);
                res = vsetq_lane_u8(*$s.get_unchecked($i), res, 15 - $i);
            )*
            res
        }
    };
} */

#[inline(always)]
unsafe fn load_8(s: &[u8]) -> uint8x8_t {
    let mut data = vdup_n_u32(0);

    match s.len() {
        8.. => vld1_u8(s.as_ptr()),
        7 => {
            data = vset_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let mut data = vreinterpret_u16_u32(data);
            data = vset_lane_u16(u16::from_le_bytes(s[4..6].try_into().unwrap()), data, 2);
            let data = vreinterpret_u8_u16(data);
            vset_lane_u8(s[6], data, 6)
        }
        6 => {
            data = vset_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let mut data = vreinterpret_u16_u32(data);
            data = vset_lane_u16(u16::from_le_bytes(s[4..6].try_into().unwrap()), data, 2);
            vreinterpret_u8_u16(data)
        }
        5 => {
            data = vset_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let data = vreinterpret_u8_u32(data);
            vset_lane_u8(s[4], data, 4)
        }
        4 => {
            data = vset_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            vreinterpret_u8_u32(data)
        }
        3 => {
            let mut data = vreinterpret_u16_u32(data);
            data = vset_lane_u16(u16::from_le_bytes(s[0..2].try_into().unwrap()), data, 0);
            let data = vreinterpret_u8_u16(data);
            vset_lane_u8(s[2], data, 2)
        }
        2 => {
            let mut data = vreinterpret_u16_u32(data);
            data = vset_lane_u16(u16::from_le_bytes(s[0..2].try_into().unwrap()), data, 0);
            vreinterpret_u8_u16(data)
        }
        1 => {
            let data = vreinterpret_u8_u32(data);
            vset_lane_u8(s[0], data, 0)
        }
        0 => vreinterpret_u8_u32(data),
        #[allow(unreachable_patterns)]
        _ => ::core::hint::unreachable_unchecked(), // unreachable since 1.75
    }
}

#[inline(always)]
unsafe fn load_16(s: &[u8]) -> uint8x16_t {
    let mut data = vdupq_n_u64(0);

    match s.len() {
        16.. => vld1q_u8(s.as_ptr()),
        15 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[8..12].try_into().unwrap()), data, 2);
            let mut data = vreinterpretq_u16_u32(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[12..14].try_into().unwrap()), data, 6);
            let data = vreinterpretq_u8_u16(data);
            vsetq_lane_u8(s[14], data, 14)
        }
        14 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[8..12].try_into().unwrap()), data, 2);
            let mut data = vreinterpretq_u16_u32(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[12..14].try_into().unwrap()), data, 6);
            vreinterpretq_u8_u16(data)
        }
        13 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[8..12].try_into().unwrap()), data, 2);
            let data = vreinterpretq_u8_u32(data);
            vsetq_lane_u8(s[12], data, 12)
        }
        12 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[8..12].try_into().unwrap()), data, 2);
            vreinterpretq_u8_u32(data)
        }
        11 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u16_u64(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[8..10].try_into().unwrap()), data, 4);
            let data = vreinterpretq_u8_u16(data);
            vsetq_lane_u8(s[10], data, 10)
        }
        10 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u16_u64(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[8..10].try_into().unwrap()), data, 4);
            vreinterpretq_u8_u16(data)
        }
        9 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            let data = vreinterpretq_u8_u64(data);
            vsetq_lane_u8(s[8], data, 8)
        }
        8 => {
            data = vsetq_lane_u64(u64::from_le_bytes(s[0..8].try_into().unwrap()), data, 0);
            vreinterpretq_u8_u64(data)
        }
        7 => {
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u16_u32(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[4..6].try_into().unwrap()), data, 2);
            let data = vreinterpretq_u8_u16(data);
            vsetq_lane_u8(s[6], data, 6)
        }
        6 => {
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let mut data = vreinterpretq_u16_u32(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[4..6].try_into().unwrap()), data, 2);
            vreinterpretq_u8_u16(data)
        }
        5 => {
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            let data = vreinterpretq_u8_u32(data);
            vsetq_lane_u8(s[4], data, 4)
        }
        4 => {
            let mut data = vreinterpretq_u32_u64(data);
            data = vsetq_lane_u32(u32::from_le_bytes(s[0..4].try_into().unwrap()), data, 0);
            vreinterpretq_u8_u32(data)
        }
        3 => {
            let mut data = vreinterpretq_u16_u64(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[0..2].try_into().unwrap()), data, 0);
            let data = vreinterpretq_u8_u16(data);
            vsetq_lane_u8(s[2], data, 2)
        }
        2 => {
            let mut data = vreinterpretq_u16_u64(data);
            data = vsetq_lane_u16(u16::from_le_bytes(s[0..2].try_into().unwrap()), data, 0);
            vreinterpretq_u8_u16(data)
        }
        1 => {
            let data = vreinterpretq_u8_u64(data);
            vsetq_lane_u8(s[0], data, 0)
        }
        0 => vreinterpretq_u8_u64(data),
        #[allow(unreachable_patterns)]
        _ => ::core::hint::unreachable_unchecked(), // unreachable since 1.75
    }
}

#[inline(always)]
unsafe fn check_len_8(mut chunk: uint8x8_t) -> u32 {
    let cmp_high = vld1_dup_u8(&CHAR_MAX);
    let cmp_low = vld1_dup_u8(&CHAR_MIN);
    let check_high = vcgt_u8(chunk, cmp_high);
    let check_low = vcgt_u8(cmp_low, chunk);

    chunk = vorr_u8(check_high, check_low);

    let chunk = vreinterpret_u64_u8(chunk);
    let res = vget_lane_u64(chunk, 0);
    res.trailing_zeros() >> 3
}

#[inline(always)]
unsafe fn check_len_16(mut chunk: uint8x16_t) -> u32 {
    let cmp_high = vld1q_dup_u8(&CHAR_MAX);
    let cmp_low = vld1q_dup_u8(&CHAR_MIN);
    let check_high = vcgtq_u8(chunk, cmp_high);
    let check_low = vcgtq_u8(cmp_low, chunk);

    chunk = vorrq_u8(check_high, check_low);

    let chunk = vreinterpretq_u16_u8(chunk);
    let chunk = vshrn_n_u16(chunk, 4);

    let chunk = vreinterpret_u64_u8(chunk);
    let res = vget_lane_u64(chunk, 0);
    res.trailing_zeros() >> 2
}

#[inline(always)]
unsafe fn parse_simd_neon(
    len: usize,
    mut chunk: uint8x16_t,
) -> Result<(u64, usize), AtoiSimdError<'static>> {
    chunk = match len {
        0 => return Err(AtoiSimdError::Empty),
        1 => return Ok((vgetq_lane_u8(chunk, 0) as u64, 1)),
        2 => vextq_u8(vdupq_n_u8(0), chunk, 2),
        3 => vextq_u8(vdupq_n_u8(0), chunk, 3),
        4 => vextq_u8(vdupq_n_u8(0), chunk, 4),
        5 => vextq_u8(vdupq_n_u8(0), chunk, 5),
        6 => vextq_u8(vdupq_n_u8(0), chunk, 6),
        7 => vextq_u8(vdupq_n_u8(0), chunk, 7),
        8 => vextq_u8(vdupq_n_u8(0), chunk, 8),
        9 => vextq_u8(vdupq_n_u8(0), chunk, 9),
        10 => vextq_u8(vdupq_n_u8(0), chunk, 10),
        11 => vextq_u8(vdupq_n_u8(0), chunk, 11),
        12 => vextq_u8(vdupq_n_u8(0), chunk, 12),
        13 => vextq_u8(vdupq_n_u8(0), chunk, 13),
        14 => vextq_u8(vdupq_n_u8(0), chunk, 14),
        15 => vextq_u8(vdupq_n_u8(0), chunk, 15),
        16 => chunk,
        _ => {
            if cfg!(debug_assertions) {
                panic!("parse_simd_neon: wrong size {}", len);
            } else {
                ::core::hint::unreachable_unchecked()
            }
        }
    };

    /* slower
    let (sum, chunk) = odd_even_8(chunk);
    let mult = vdup_n_u8(10);
    let chunk = vmla_u8(sum, chunk, mult);

    let (sum, chunk) = odd_even_small_16(chunk);
    let mult = vdup_n_u16(100);
    let chunk = vmla_u16(sum, chunk, mult);

    let (sum, chunk) = odd_even_small_32(chunk);
    let mult = vdup_n_u32(10000);
    let chunk = vmla_u32(sum, chunk, mult);*/

    chunk = vmulq_u8(
        chunk,
        vld1q_u8([10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1].as_ptr()),
    );
    let chunk = vpaddlq_u8(chunk);

    let chunk = vmulq_u16(chunk, vld1q_u16([100, 1, 100, 1, 100, 1, 100, 1].as_ptr()));
    let chunk = vpaddlq_u16(chunk);

    let chunk = vmulq_u32(chunk, vld1q_u32([10000, 1, 10000, 1].as_ptr()));
    let chunk = vpaddlq_u32(chunk);

    let res = vgetq_lane_u64(chunk, 0) * 100_000_000 + vgetq_lane_u64(chunk, 1);

    Ok((res, len))
}

#[inline(always)]
unsafe fn simd_neon_len(s: &[u8]) -> Result<(usize, uint8x16_t), AtoiSimdError> {
    let mut chunk = load_16(s);
    let len = check_len_16(chunk) as usize;

    chunk = vandq_u8(chunk, vdupq_n_u8(0xF));

    Ok((len, chunk))
}

#[inline(always)]
pub(crate) fn parse_simd_16(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
    unsafe {
        let (len, chunk) = simd_neon_len(s)?;
        parse_simd_neon(len, chunk)
    }
}

#[inline(always)]
unsafe fn odd_even_8(chunk: uint8x16_t) -> (uint8x8_t, uint8x8_t) {
    let chunk = vreinterpretq_u16_u8(chunk);
    let sum = vshrn_n_u16::<8>(chunk);
    let chunk = vmovn_u16(chunk);

    (sum, chunk)
}

#[inline(always)]
unsafe fn odd_even_16(chunk: uint8x16_t) -> (uint16x8_t, uint8x8_t) {
    let chunk = vreinterpretq_u16_u8(chunk);
    let sum = vshrq_n_u16::<8>(chunk);
    let chunk = vmovn_u16(chunk);

    (sum, chunk)
}

#[inline(always)]
unsafe fn odd_even_32(chunk: uint16x8_t) -> (uint32x4_t, uint16x4_t) {
    let chunk = vreinterpretq_u32_u16(chunk);
    let sum = vshrq_n_u32::<16>(chunk);
    let chunk = vmovn_u32(chunk);

    (sum, chunk)
}

#[inline(always)]
unsafe fn odd_even_64(chunk: int32x4_t) -> (uint64x2_t, uint32x2_t) {
    let chunk = vreinterpretq_u64_s32(chunk);
    let sum = vshrq_n_u64::<32>(chunk);
    let chunk = vmovn_u64(chunk);

    (sum, chunk)
}

/* #[inline(always)]
unsafe fn odd_even_small_16(chunk: uint8x8_t) -> (uint16x4_t, uint16x4_t) {
    let chunk = vreinterpret_u16_u8(chunk);
    let sum = vshr_n_u16::<8>(chunk);
    let chunk = vand_u16(chunk, vdup_n_u16(0xFF));

    (sum, chunk)
}

#[inline(always)]
unsafe fn odd_even_small_32(chunk: uint16x4_t) -> (uint32x2_t, uint32x2_t) {
    let chunk = vreinterpret_u32_u16(chunk);
    let sum = vshr_n_u32::<16>(chunk);
    let chunk = vand_u32(chunk, vdup_n_u32(0xFFFF));

    (sum, chunk)
} */

#[inline(always)]
unsafe fn parse_simd_extra<'a>(
    s: &'a [u8],
    chunk1: &mut uint8x16_t,
    chunk2: &mut uint8x16_t,
) -> Result<(u128, usize), AtoiSimdError<'a>> {
    let mut chunk3 = load_8(s.get_safe_unchecked(32..));
    let mut len = check_len_8(chunk3) as usize;
    chunk3 = vand_u8(chunk3, vdup_n_u8(0xF));
    let mut chunk3_16 = vcombine_u8(chunk3, vdup_n_u8(0));
    chunk3_16 = match len {
        0 => vdupq_n_u8(0), //return Ok((0, 16)), is slower
        1 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 9);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 1);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 1);
            tmp
        }
        2 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 10);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 2);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 2);
            tmp
        }
        3 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 11);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 3);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 3);
            tmp
        }
        4 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 12);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 4);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 4);
            tmp
        }
        5 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 13);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 5);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 5);
            tmp
        }
        6 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 14);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 6);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 6);
            tmp
        }
        7 => {
            let tmp = vextq_u8(vdupq_n_u8(0), *chunk1, 15);
            *chunk1 = vextq_u8(*chunk1, *chunk2, 7);
            *chunk2 = vextq_u8(*chunk2, chunk3_16, 7);
            tmp
        }
        s_len => return Err(AtoiSimdError::Size(s_len, s)),
    };
    chunk3 = vget_low_u8(chunk3_16);
    len += 16;

    let chunk3 = vmull_u8(chunk3, vld1_u8([0, 1, 10, 1, 10, 1, 10, 1].as_ptr()));
    let chunk3 = vpaddlq_u16(chunk3);

    let chunk3 = vmulq_u32(chunk3, vld1q_u32([1_000_000, 10_000, 100, 1].as_ptr()));

    let res = (vaddlvq_u32(chunk3) as u128)
        .checked_mul(100_000_000_000_000_000_000_000_000_000_000)
        .ok_or(AtoiSimdError::Overflow(s))?;

    Ok((res, len))
}

#[inline(always)]
pub(crate) fn parse_simd_u128(s: &[u8]) -> Result<(u128, usize), AtoiSimdError> {
    unsafe {
        let mut chunk1 = load_16(s);
        let mut len = check_len_16(chunk1) as usize;

        let mult = vdupq_n_u8(0xF);
        chunk1 = vandq_u8(chunk1, mult);

        if len < 16 {
            return parse_simd_neon(len, chunk1).map(|(v, l)| (v as u128, l));
        };

        let mut chunk2 = load_16(s.get_safe_unchecked(16..));

        len = check_len_16(chunk2) as usize;
        chunk2 = vandq_u8(chunk2, mult);
        let mut extra = 0;
        match len {
            0 => return parse_simd_neon(16, chunk1).map(|(v, l)| (v as u128, l)),
            1 => {
                chunk2 = vextq_u8(chunk1, chunk2, 1);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 1);
            }
            2 => {
                chunk2 = vextq_u8(chunk1, chunk2, 2);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 2);
            }
            3 => {
                chunk2 = vextq_u8(chunk1, chunk2, 3);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 3);
            }
            4 => {
                chunk2 = vextq_u8(chunk1, chunk2, 4);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 4);
            }
            5 => {
                chunk2 = vextq_u8(chunk1, chunk2, 5);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 5);
            }
            6 => {
                chunk2 = vextq_u8(chunk1, chunk2, 6);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 6);
            }
            7 => {
                chunk2 = vextq_u8(chunk1, chunk2, 7);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 7);
            }
            8 => {
                chunk2 = vextq_u8(chunk1, chunk2, 8);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 8);
            }
            9 => {
                chunk2 = vextq_u8(chunk1, chunk2, 9);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 9);
            }
            10 => {
                chunk2 = vextq_u8(chunk1, chunk2, 10);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 10);
            }
            11 => {
                chunk2 = vextq_u8(chunk1, chunk2, 11);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 11);
            }
            12 => {
                chunk2 = vextq_u8(chunk1, chunk2, 12);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 12);
            }
            13 => {
                chunk2 = vextq_u8(chunk1, chunk2, 13);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 13);
            }
            14 => {
                chunk2 = vextq_u8(chunk1, chunk2, 14);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 14);
            }
            15 => {
                chunk2 = vextq_u8(chunk1, chunk2, 15);
                chunk1 = vextq_u8(vdupq_n_u8(0), chunk1, 15);
            }
            16 => {
                (extra, len) = parse_simd_extra(s, &mut chunk1, &mut chunk2)?;
            }
            #[allow(unreachable_patterns)]
            _ => ::core::hint::unreachable_unchecked(), // unreachable since 1.75
        };

        let (sum1, chunk1) = odd_even_8(chunk1);
        let (sum2, chunk2) = odd_even_8(chunk2);
        let mult = vdupq_n_u8(10);
        let chunk = vmlaq_u8(vcombine_u8(sum1, sum2), vcombine_u8(chunk1, chunk2), mult);

        let (sum, chunk) = odd_even_16(chunk);
        let mult = vdup_n_u8(100);
        let chunk = vmlal_u8(sum, chunk, mult);

        let (sum, chunk) = odd_even_32(chunk);
        let chunk = vqdmlal_n_s16(
            vreinterpretq_s32_u32(sum),
            vreinterpret_s16_u16(chunk),
            5000, // because it's doubling
        );

        let (sum, chunk) = odd_even_64(chunk);
        let chunk = vqdmlal_n_s32(
            vreinterpretq_s64_u64(sum),
            vreinterpret_s32_u32(chunk),
            50_000_000, // because it's doubling
        );

        let mut res = vgetq_lane_s64(chunk, 0) as u128 * 10_000_000_000_000_000
            + vgetq_lane_s64(chunk, 1) as u128;
        if extra > 0 {
            res = res.checked_add(extra).ok_or(AtoiSimdError::Overflow(s))?;
        }

        Ok((res, len + 16))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_check_len_8() {
        unsafe {
            let data = [
                ([b'1', 0, 0, 0, 0, 0, 0, 0], 1),
                ([b'1', b'2', 0, 0, 0, 0, 0, 0], 2),
                ([b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8'], 8),
            ];

            for (input, len) in data {
                assert_eq!(
                    check_len_8(vld1_u8(input.as_ptr())),
                    len,
                    "input: {:?}",
                    input
                );
            }
        }
    }
}
