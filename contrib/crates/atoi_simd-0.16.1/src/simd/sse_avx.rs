#![allow(dead_code)] // used when you don't have avx

use self::arch::{
    __m128i, __m256i, _mm256_add_epi64, _mm256_alignr_epi8, _mm256_and_si256, _mm256_cmpgt_epi8,
    _mm256_cvtsi256_si32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_movemask_epi8, _mm256_mul_epu32, _mm256_or_si256,
    _mm256_packus_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi8,
    _mm256_set_epi16, _mm256_set_epi32, _mm256_set_epi8, _mm256_set_m128i, _mm256_setzero_si256,
    _mm256_srli_epi64, _mm_add_epi64, _mm_and_si128, _mm_bslli_si128, _mm_cmpgt_epi8,
    _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8,
    _mm_mul_epu32, _mm_or_si128, _mm_packus_epi32, _mm_set1_epi8, _mm_set_epi16, _mm_set_epi32,
    _mm_set_epi8, _mm_setzero_si128, _mm_srli_epi64,
};
use crate::AtoiSimdError;
#[cfg(target_arch = "x86")]
use ::core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64 as arch;
use ::core::convert::TryInto;
use debug_unsafe::slice::SliceGetter;

pub(crate) const SHORT: usize = 4;

const CHAR_MAX: i8 = b'9' as i8;
const CHAR_MIN: i8 = b'0' as i8;

/* #[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline(always)]
unsafe fn read(s: &[u8]) -> __m128i {
    let len = s.len();
    if len < 16 {
        return _mm_maskz_loadu_epi8((1 << len) - 1, ::core::mem::transmute_copy(&s));
    }

    _mm_loadu_si128(::core::mem::transmute_copy(&s))
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vl"
))]
#[inline(always)]
unsafe fn read_avx(s: &[u8]) -> __m256i {
    let len = s.len();
    if len < 32 {
        return _mm256_maskz_loadu_epi8((1 << len) - 1, ::core::mem::transmute_copy(&s));
    }

    _mm256_loadu_si256(::core::mem::transmute_copy(&s))
} */

/* #[inline(always)]
unsafe fn read(s: &[u8]) -> __m128i {
    let len = s.len();

    match len >> 2 {
        4.. => _mm_loadu_si128(::core::mem::transmute_copy(&s)),
        2 | 3 => {
            let hi = u64::from_le_bytes(s.get_safe_unchecked(len - 8..len).try_into().unwrap())
                .overflowing_shr(8 * (16 - len as u32))
                .0;

            _mm_set_epi64x(
                hi as i64,
                u64::from_le_bytes(s.get_safe_unchecked(0..8).try_into().unwrap()) as i64,
            )
        }
        1 => {
            let hi = u32::from_le_bytes(s.get_safe_unchecked(len - 4..len).try_into().unwrap())
                .overflowing_shr(8 * (8 - len as u32))
                .0;

            _mm_set_epi32(
                0,
                0,
                hi as i32,
                u32::from_le_bytes(s.get_safe_unchecked(0..4).try_into().unwrap()) as i32,
            )
        }
        0 => {
            if len == 0 {
                return _mm_setzero_si128();
            }

            let lo = *s.get_safe_unchecked(0) as u64;
            let mid = (*s.get_safe_unchecked(len >> 1) as u64) << (8 * (len >> 1));
            let hi = (*s.get_safe_unchecked(len - 1) as u64) << (8 * (len - 1));
            let res = lo | mid | hi;

            _mm_loadu_si64(&res as *const _ as *const u8)
        }
        // _ => _mm_setzero_si128(),
        _ => unsafe { ::core::hint::unreachable_unchecked() },
    }
} */

/// s = "1234567890123456"
#[inline]
unsafe fn load(s: &[u8]) -> __m128i {
    match s.len() {
        16.. => _mm_loadu_si128(::core::mem::transmute_copy(&s)),
        15 => _mm_set_epi32(
            i32::from_le_bytes(s[11..15].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        14 => _mm_set_epi32(
            u16::from_le_bytes(s[12..14].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        13 => _mm_set_epi32(
            s[12] as i32,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        12 => _mm_set_epi32(
            0,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        11 => _mm_set_epi32(
            0,
            i32::from_le_bytes(s[7..11].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        10 => _mm_set_epi32(
            0,
            u16::from_le_bytes(s[8..10].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        9 => _mm_set_epi32(
            0,
            s[8] as i32,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        8 => _mm_set_epi32(
            0,
            0,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        7 => _mm_set_epi32(
            0,
            0,
            i32::from_le_bytes(s[3..7].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        6 => _mm_set_epi32(
            0,
            0,
            u16::from_le_bytes(s[4..6].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        5 => _mm_set_epi32(
            0,
            0,
            s[4] as i32,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        4 => _mm_set_epi32(0, 0, 0, i32::from_le_bytes(s[0..4].try_into().unwrap())),
        3 => _mm_set_epi32(
            0,
            0,
            0,
            (u16::from_le_bytes(s[1..3].try_into().unwrap()) as i32) << 8 | s[0] as i32,
        ),
        2 => _mm_set_epi32(
            0,
            0,
            0,
            u16::from_le_bytes(s[0..2].try_into().unwrap()) as i32,
        ),
        1 => _mm_set_epi32(0, 0, 0, s[0] as i32),
        0 => _mm_setzero_si128(),
        #[allow(unreachable_patterns)]
        _ => ::core::hint::unreachable_unchecked(), // unreachable since 1.75
    }
}

#[inline]
unsafe fn load_avx(s: &[u8]) -> __m256i {
    match s.len() {
        32.. => _mm256_loadu_si256(::core::mem::transmute_copy(&s)),
        31 => _mm256_set_epi32(
            i32::from_le_bytes(s[27..31].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[24..28].try_into().unwrap()),
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        30 => _mm256_set_epi32(
            u16::from_le_bytes(s[28..30].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[24..28].try_into().unwrap()),
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        29 => _mm256_set_epi32(
            s[28] as i32,
            i32::from_le_bytes(s[24..28].try_into().unwrap()),
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        28 => _mm256_set_epi32(
            0,
            i32::from_le_bytes(s[24..28].try_into().unwrap()),
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        27 => _mm256_set_epi32(
            0,
            i32::from_le_bytes(s[23..27].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        26 => _mm256_set_epi32(
            0,
            u16::from_le_bytes(s[24..26].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        25 => _mm256_set_epi32(
            0,
            s[24] as i32,
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        24 => _mm256_set_epi32(
            0,
            0,
            i32::from_le_bytes(s[20..24].try_into().unwrap()),
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        23 => _mm256_set_epi32(
            0,
            0,
            i32::from_le_bytes(s[19..23].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        22 => _mm256_set_epi32(
            0,
            0,
            u16::from_le_bytes(s[20..22].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        21 => _mm256_set_epi32(
            0,
            0,
            s[20] as i32,
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        20 => _mm256_set_epi32(
            0,
            0,
            0,
            i32::from_le_bytes(s[16..20].try_into().unwrap()),
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        19 => _mm256_set_epi32(
            0,
            0,
            0,
            i32::from_le_bytes(s[15..19].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        18 => _mm256_set_epi32(
            0,
            0,
            0,
            u16::from_le_bytes(s[16..18].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        17 => _mm256_set_epi32(
            0,
            0,
            0,
            s[16] as i32,
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        16 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[12..16].try_into().unwrap()),
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        15 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[11..15].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        14 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            u16::from_le_bytes(s[12..14].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        13 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            s[12] as i32,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        12 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[8..12].try_into().unwrap()),
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        11 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[7..11].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        10 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            u16::from_le_bytes(s[8..10].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        9 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            s[8] as i32,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        8 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[4..8].try_into().unwrap()),
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        7 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[3..7].try_into().unwrap()) >> 8,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        6 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            u16::from_le_bytes(s[4..6].try_into().unwrap()) as i32,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        5 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            s[4] as i32,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        4 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
        ),
        3 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            (u16::from_le_bytes(s[1..3].try_into().unwrap()) as i32) << 8 | s[0] as i32,
        ),
        2 => _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            u16::from_le_bytes(s[0..2].try_into().unwrap()) as i32,
        ),
        1 => _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, s[0] as i32),
        0 => _mm256_setzero_si256(),
        #[allow(unreachable_patterns)]
        _ => ::core::hint::unreachable_unchecked(), // unreachable since 1.75
    }
}

/// converts chars  [ 0x36353433323130393837363534333231 ]
/// to numbers      [ 0x06050403020100090807060504030201 ]
#[inline(always)]
unsafe fn to_numbers(chunk: __m128i) -> __m128i {
    let mult = _mm_set1_epi8(0xF);

    _mm_and_si128(chunk, mult)
}

#[inline(always)]
unsafe fn process_gt(cmp_left: __m128i, cmp_right: __m128i) -> __m128i {
    _mm_cmpgt_epi8(cmp_left, cmp_right)
}

#[inline(always)]
unsafe fn process_avx_gt(cmp_left: __m256i, cmp_right: __m256i) -> __m256i {
    _mm256_cmpgt_epi8(cmp_left, cmp_right)
}

#[inline(always)]
unsafe fn check_len(mut chunk: __m128i) -> u32 {
    let cmp_high = _mm_set1_epi8(CHAR_MAX);
    let cmp_low = _mm_set1_epi8(CHAR_MIN);
    let check_high = process_gt(chunk, cmp_high);
    let check_low = process_gt(cmp_low, chunk);

    chunk = _mm_or_si128(check_high, check_low);
    let res = _mm_movemask_epi8(chunk) as u16;
    res.trailing_zeros()
}

#[inline(always)]
unsafe fn check_len_avx(mut chunk: __m256i) -> u32 {
    let cmp_max = _mm256_set1_epi8(CHAR_MAX);
    let cmp_min = _mm256_set1_epi8(CHAR_MIN);
    let check_high = process_avx_gt(chunk, cmp_max);
    let check_low = process_avx_gt(cmp_min, chunk);

    chunk = _mm256_or_si256(check_high, check_low);
    let res = _mm256_movemask_epi8(chunk);
    res.trailing_zeros()
}

/* #[cfg(target_arch = "x86")]
#[inline(always)]
unsafe fn to_u64(chunk: __m128i) -> u64 {
    ::core::mem::transmute_copy(&chunk)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn to_u64(chunk: __m128i) -> u64 {
    arch::_mm_cvtsi128_si64(chunk) as u64
} */

#[inline(always)]
unsafe fn to_u32x4(chunk: __m128i) -> [u32; 4] {
    ::core::mem::transmute(chunk)
}

#[inline]
unsafe fn parse_simd_sse(
    len: u32,
    mut chunk: __m128i,
) -> Result<(u64, usize), AtoiSimdError<'static>> {
    chunk = match len {
        0 => return Err(AtoiSimdError::Empty),
        1 => return Ok(((_mm_cvtsi128_si32(chunk) & 0xFF) as u64, len as usize)),
        2 => _mm_bslli_si128(chunk, 14),
        3 => _mm_bslli_si128(chunk, 13),
        4 => _mm_bslli_si128(chunk, 12),
        5 => _mm_bslli_si128(chunk, 11),
        6 => _mm_bslli_si128(chunk, 10),
        7 => _mm_bslli_si128(chunk, 9),
        8 => _mm_bslli_si128(chunk, 8),
        9 => _mm_bslli_si128(chunk, 7),
        10 => _mm_bslli_si128(chunk, 6),
        11 => _mm_bslli_si128(chunk, 5),
        12 => _mm_bslli_si128(chunk, 4),
        13 => _mm_bslli_si128(chunk, 3),
        14 => _mm_bslli_si128(chunk, 2),
        15 => _mm_bslli_si128(chunk, 1),
        16 => chunk,
        _ => {
            if cfg!(debug_assertions) {
                panic!("parse_simd_sse: wrong size {}", len);
            } else {
                ::core::hint::unreachable_unchecked()
            }
        }
    };

    // combine numbers [ 0x0038 | 0x0022 | 0x000c | 0x005a | 0x004e | 0x0038 | 0x0022 | 0x000c ( 56 | 34 | 12 | 90 | 78 | 56 | 34 | 12 ) ]
    chunk = _mm_maddubs_epi16(
        chunk,
        _mm_set_epi8(1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10),
    );

    // combine again   [ 0x0000 | 0x0d80 | 0x0000 | 0x2334 | 0x0000 | 0x162e | 0x0000 | 0x04d2 ( 0 | 3456 | 0 | 9012 | 0 | 5678 | 0 | 1234) ]
    chunk = _mm_madd_epi16(chunk, _mm_set_epi16(1, 100, 1, 100, 1, 100, 1, 100));
    // remove extra bytes [ (64 bits, same as the right ) | 0x0d80 | 0x2334 | 0x162e | 0x04d2 ( 3456 | 9012 | 5678 | 1234) ]
    chunk = _mm_packus_epi32(chunk, chunk);

    // combine again [ (64 bits, zeroes) | 0x055f2cc0 | 0x00bc614e ( 90123456 | 12345678 ) ]
    chunk = _mm_madd_epi16(chunk, _mm_set_epi16(0, 0, 0, 0, 1, 10000, 1, 10000));

    let arr = to_u32x4(chunk);
    let res = arr[0] as u64 * 100_000_000 + arr[1] as u64;

    Ok((res, len as usize))
}

#[inline]
unsafe fn simd_sse_len(s: &[u8]) -> (u32, __m128i) {
    let mut chunk = load(s);
    let len = check_len(chunk);

    chunk = to_numbers(chunk);

    (len, chunk)
}

#[inline]
pub(crate) fn parse_simd_16(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
    unsafe {
        let (len, chunk) = simd_sse_len(s);
        parse_simd_sse(len, chunk)
    }
}

/// Parses *only* digits
/// Uses AVX/AVX2 intrinsics
#[inline(always)]
unsafe fn process_avx(
    s: &[u8],
    mut chunk: __m256i,
    len: u32,
    chunk_extra: __m128i,
    len_extra: u32,
) -> Result<(u128, usize), AtoiSimdError> {
    let mut mult = _mm256_set_epi8(
        1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10,
        1, 10, 1, 10, 1, 10,
    );
    // mult 1 char
    chunk = _mm256_maddubs_epi16(chunk, mult);

    mult = _mm256_set_epi16(
        1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100,
    );
    // mult 2
    chunk = _mm256_madd_epi16(chunk, mult);
    // remove extra bytes
    chunk = _mm256_packus_epi32(chunk, chunk);

    // move by 64 bits ( unused | unused | third [191:128] | first [63:0] )
    // compiled assembly is different, and faster
    chunk = _mm256_permute4x64_epi64(chunk, 8);
    let mut chunk_sse = _mm256_extracti128_si256(chunk, 0);

    if len_extra == 0 {
        let mut mult = _mm_set_epi16(1, 10000, 1, 10000, 1, 10000, 1, 10000);
        // mult 4
        chunk_sse = _mm_madd_epi16(chunk_sse, mult);

        mult = _mm_set_epi32(0, 100_000_000, 0, 100_000_000);
        // requires avx512ifma,avx512vl and nightly only
        // chunk_sse = _mm_madd52lo_epu64(_mm_srli_epi64(chunk_sse, 32), chunk_sse, mult);

        // mult 8
        mult = _mm_mul_epu32(chunk_sse, mult);
        // add higher 32 bits of old 64 to mult
        chunk_sse = _mm_srli_epi64(chunk_sse, 32);
        chunk_sse = _mm_add_epi64(chunk_sse, mult);

        let arr = ::core::mem::transmute::<__m128i, [u64; 2]>(chunk_sse);

        // mult 16
        Ok((
            arr[0] as u128 * 10_000_000_000_000_000 + arr[1] as u128,
            len as usize,
        ))
    } else {
        let (mut chunk_extra, mult16) = match len_extra {
            1 => (_mm_bslli_si128(chunk_extra, 15), 10),
            2 => (_mm_bslli_si128(chunk_extra, 14), 100),
            3 => (_mm_bslli_si128(chunk_extra, 13), 1_000),
            4 => (_mm_bslli_si128(chunk_extra, 12), 10_000),
            5 => (_mm_bslli_si128(chunk_extra, 11), 100_000),
            6 => (_mm_bslli_si128(chunk_extra, 10), 1_000_000),
            7 => (_mm_bslli_si128(chunk_extra, 9), 10_000_000),
            s_len => return Err(AtoiSimdError::Size(s_len as usize, s)),
        };

        chunk_extra = _mm_maddubs_epi16(
            chunk_extra,
            _mm_set_epi8(1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10),
        );
        chunk_extra = _mm_madd_epi16(chunk_extra, _mm_set_epi16(1, 100, 1, 100, 1, 100, 1, 100));

        chunk_extra = _mm_packus_epi32(chunk_extra, chunk_extra);

        chunk = _mm256_set_m128i(chunk_extra, chunk_sse);

        mult = _mm256_set_epi16(
            1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000,
        );
        // mult 4
        chunk = _mm256_madd_epi16(chunk, mult);

        mult = _mm256_set_epi32(0, 0, 0, 0, 0, 100_000_000, 0, 100_000_000);
        // mult 8
        mult = _mm256_mul_epu32(chunk, mult);

        chunk = _mm256_srli_epi64(chunk, 32);
        chunk = _mm256_add_epi64(chunk, mult);

        let arr = ::core::mem::transmute::<__m256i, [u64; 4]>(chunk);

        Ok((
            (arr[0] as u128 * 10_000_000_000_000_000 + arr[1] as u128)
                .checked_mul(mult16 as u128)
                .ok_or(AtoiSimdError::Overflow(s))?
                .checked_add(arr[2] as u128)
                .ok_or(AtoiSimdError::Overflow(s))?,
            (len + len_extra) as usize,
        ))
    }
}

/// Uses AVX/AVX2 intrinsics
#[inline(always)]
pub(crate) fn parse_simd_u128(s: &[u8]) -> Result<(u128, usize), AtoiSimdError> {
    unsafe {
        let mut chunk = load_avx(s);
        let len = check_len_avx(chunk);

        // to numbers
        chunk = _mm256_and_si256(chunk, _mm256_set1_epi8(0xF));
        let chunk_sh = _mm256_permute2x128_si256(chunk, chunk, 0x28);
        let mut chunk_extra = _mm_set1_epi8(0);
        let mut len_extra = 0;
        chunk = match len {
            0 => return Err(AtoiSimdError::Empty),
            1 => return Ok(((_mm256_cvtsi256_si32(chunk) & 0xFF) as u128, len as usize)),
            17 => _mm256_alignr_epi8(chunk, chunk_sh, 1),
            18 => _mm256_alignr_epi8(chunk, chunk_sh, 2),
            19 => _mm256_alignr_epi8(chunk, chunk_sh, 3),
            20 => _mm256_alignr_epi8(chunk, chunk_sh, 4),
            21 => _mm256_alignr_epi8(chunk, chunk_sh, 5),
            22 => _mm256_alignr_epi8(chunk, chunk_sh, 6),
            23 => _mm256_alignr_epi8(chunk, chunk_sh, 7),
            24 => _mm256_alignr_epi8(chunk, chunk_sh, 8),
            25 => _mm256_alignr_epi8(chunk, chunk_sh, 9),
            26 => _mm256_alignr_epi8(chunk, chunk_sh, 10),
            27 => _mm256_alignr_epi8(chunk, chunk_sh, 11),
            28 => _mm256_alignr_epi8(chunk, chunk_sh, 12),
            29 => _mm256_alignr_epi8(chunk, chunk_sh, 13),
            30 => _mm256_alignr_epi8(chunk, chunk_sh, 14),
            31 => _mm256_alignr_epi8(chunk, chunk_sh, 15),
            32 => {
                if s.len() > 32 {
                    (len_extra, chunk_extra) = simd_sse_len(s.get_safe_unchecked(32..));
                }
                chunk
            }
            s_len => {
                return parse_simd_sse(s_len, ::core::mem::transmute_copy(&chunk))
                    .map(|(v, l)| (v as u128, l))
            }
        };

        process_avx(s, chunk, len, chunk_extra, len_extra)
    }
}
