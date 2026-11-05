use super::*;
use crate::simd::shared_64::*;

#[cfg(target_pointer_width = "64")]
impl ParsePos for usize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<usize, AtoiSimdError> {
        parse_simd_checked_u64(s).map(|v| v as usize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(usize, usize), AtoiSimdError> {
        parse_simd_u64(s).map(|(v, i)| (v as usize, i))
    }
}

#[cfg(target_pointer_width = "64")]
impl ParsePos for isize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_simd_checked_i64(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_simd_i64(s).map(|(v, i)| (v as isize, i))
    }
}

#[cfg(target_pointer_width = "64")]
impl ParseNeg for isize {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_simd_checked_i64_neg(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_simd_i64_neg(s).map(|(v, i)| (v as isize, i))
    }
}

impl ParsePos for u64 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u64, AtoiSimdError> {
        parse_simd_checked_u64(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
        parse_simd_u64(s)
    }
}

impl ParsePos for i64 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i64, AtoiSimdError> {
        parse_simd_checked_i64(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
        parse_simd_i64(s)
    }
}

impl ParseNeg for i64 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i64, AtoiSimdError> {
        parse_simd_checked_i64_neg(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
        parse_simd_i64_neg(s)
    }
}

impl ParsePos for u128 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u128, AtoiSimdError> {
        parse_simd_checked_u128(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u128, usize), AtoiSimdError> {
        parse_simd_u128(s)
    }
}

impl ParsePos for i128 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i128, AtoiSimdError> {
        parse_simd_checked_i128(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
        parse_simd_i128(s)
    }
}

impl ParseNeg for i128 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i128, AtoiSimdError> {
        parse_simd_checked_i128_neg(s)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
        parse_simd_i128_neg(s)
    }
}
