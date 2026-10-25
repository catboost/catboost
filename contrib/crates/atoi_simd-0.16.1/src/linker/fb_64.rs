use super::*;
use crate::fallback::*;

#[cfg(target_pointer_width = "64")]
impl ParsePos for usize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<usize, AtoiSimdError> {
        parse_fb_checked_64_pos::<{ u64::MAX }, 4>(s).map(|v| v as usize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(usize, usize), AtoiSimdError> {
        parse_fb_64_pos::<{ u64::MAX }, 4>(s).map(|(v, i)| (v as usize, i))
    }
}

#[cfg(target_pointer_width = "64")]
impl ParsePos for isize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_fb_checked_64_pos::<{ i64::MAX as u64 }, 3>(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_fb_64_pos::<{ i64::MAX as u64 }, 3>(s).map(|(v, i)| (v as isize, i))
    }
}

#[cfg(target_pointer_width = "64")]
impl ParseNeg for isize {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_fb_checked_64_neg(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_fb_64_neg(s).map(|(v, i)| (v as isize, i))
    }
}

impl ParsePos for u64 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u64, AtoiSimdError> {
        parse_fb_checked_64_pos::<{ u64::MAX }, 4>(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
        parse_fb_64_pos::<{ u64::MAX }, 4>(s)
    }
}

impl ParsePos for i64 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i64, AtoiSimdError> {
        parse_fb_checked_64_pos::<{ i64::MAX as u64 }, 3>(s).map(|v| v as i64)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
        parse_fb_64_pos::<{ i64::MAX as u64 }, 3>(s).map(|(v, i)| (v as i64, i))
    }
}

impl ParseNeg for i64 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i64, AtoiSimdError> {
        parse_fb_checked_64_neg(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
        parse_fb_64_neg(s)
    }
}

impl ParsePos for u128 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u128, AtoiSimdError> {
        parse_fb_checked_128_pos::<{ u128::MAX }>(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u128, usize), AtoiSimdError> {
        parse_fb_128_pos::<{ u128::MAX }>(s)
    }
}

impl ParsePos for i128 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i128, AtoiSimdError> {
        parse_fb_checked_128_pos::<{ i128::MAX as u128 }>(s).map(|v| v as i128)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
        parse_fb_128_pos::<{ i128::MAX as u128 }>(s).map(|(v, i)| (v as i128, i))
    }
}

impl ParseNeg for i128 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i128, AtoiSimdError> {
        parse_fb_checked_128_neg(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
        parse_fb_128_neg(s)
    }
}
