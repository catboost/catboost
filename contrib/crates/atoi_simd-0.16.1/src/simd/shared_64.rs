#![allow(dead_code)] // used when you don't have avx

use crate::{short::parse_short_pos, AtoiSimdError};

pub(crate) use super::parse_simd_u128;

#[inline(always)]
pub(crate) fn parse_simd_checked_u128(s: &[u8]) -> Result<u128, AtoiSimdError> {
    let len = s.len();
    let (res, len) = if len < super::SHORT {
        parse_short_pos::<{ u64::MAX }>(s).map(|(v, l)| (v as u128, l))?
    } else if len < 17 {
        super::parse_simd_16(s).map(|(v, l)| (v as u128, l))?
    } else {
        parse_simd_u128(s)?
    };
    if len != s.len() {
        return Err(AtoiSimdError::Invalid128(res, len, s));
    }
    Ok(res)
}

#[inline(always)]
pub(crate) fn parse_simd_i128(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
    let (res, len) = parse_simd_u128(s)?;
    if res > i128::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok((res as i128, len))
}

#[inline(always)]
pub(crate) fn parse_simd_checked_i128(s: &[u8]) -> Result<i128, AtoiSimdError> {
    let res = parse_simd_checked_u128(s)?;
    if res > i128::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok(res as i128)
}

#[inline(always)]
pub(crate) fn parse_simd_u64(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
    let (res, len) = parse_simd_u128(s)?;
    if len > 20 {
        return Err(AtoiSimdError::Size(len, s));
    } else if len == 20 && res > u64::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok((res as u64, len))
}

#[inline(always)]
pub(crate) fn parse_simd_checked_u64(s: &[u8]) -> Result<u64, AtoiSimdError> {
    let len = s.len();
    if len > 20 {
        return Err(AtoiSimdError::Size(len, s));
    }
    let res = parse_simd_checked_u128(s)?;
    if len == 20 && res > u64::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok(res as u64)
}

#[inline(always)]
pub(crate) fn parse_simd_i64(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
    let (res, len) = parse_simd_u128(s)?;
    if len > 19 {
        return Err(AtoiSimdError::Size(len, s));
    } else if len == 19 && res > i64::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok((res as i64, len))
}

#[inline(always)]
pub(crate) fn parse_simd_checked_i64(s: &[u8]) -> Result<i64, AtoiSimdError> {
    let len = s.len();
    if len > 19 {
        return Err(AtoiSimdError::Size(len, s));
    }
    let res = parse_simd_checked_u128(s)?;
    if len == 19 && res > i64::MAX as u128 {
        return Err(AtoiSimdError::Overflow(s));
    }
    Ok(res as i64)
}

#[inline(always)]
pub(crate) fn parse_simd_i64_neg(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
    let (res, len) = parse_simd_u128(s)?;
    if len > 19 {
        return Err(AtoiSimdError::Size(len, s));
    } else if len == 19 {
        const MAX: u128 = -(i64::MIN as i128) as u128;
        if res > MAX {
            return Err(AtoiSimdError::Overflow(s));
        } else if res == MAX {
            return Ok((i64::MIN, len));
        }
    }
    Ok((-(res as i64), len))
}

#[inline(always)]
pub(crate) fn parse_simd_checked_i64_neg(s: &[u8]) -> Result<i64, AtoiSimdError> {
    let len = s.len();
    if len > 19 {
        return Err(AtoiSimdError::Size(len, s));
    }
    let res = parse_simd_checked_u128(s)?;
    if len == 19 {
        const MAX: u128 = -(i64::MIN as i128) as u128;
        if res > MAX {
            return Err(AtoiSimdError::Overflow(s));
        } else if res == MAX {
            return Ok(i64::MIN);
        }
    }
    Ok(-(res as i64))
}

#[inline(always)]
pub(crate) fn parse_simd_i128_neg(s: &[u8]) -> Result<(i128, usize), AtoiSimdError> {
    let (res, len) = parse_simd_u128(s)?;
    if len == 39 {
        const MAX: u128 = i128::MAX as u128 + 1;
        if res > MAX {
            return Err(AtoiSimdError::Overflow(s));
        } else if res == MAX {
            return Ok((i128::MIN, len));
        }
    }
    Ok((-(res as i128), len))
}

#[inline(always)]
pub(crate) fn parse_simd_checked_i128_neg(s: &[u8]) -> Result<i128, AtoiSimdError> {
    let len = s.len();
    if len > 39 {
        return Err(AtoiSimdError::Size(len, s));
    }
    let res = parse_simd_checked_u128(s)?;
    if len == 39 {
        const MAX: u128 = i128::MAX as u128 + 1;
        if res > MAX {
            return Err(AtoiSimdError::Overflow(s));
        } else if res == MAX {
            return Ok(i128::MIN);
        }
    }
    Ok(-(res as i128))
}
