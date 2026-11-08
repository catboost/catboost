use super::*;
use crate::simd::shared_32::*;

impl ParsePos for u8 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u8, AtoiSimdError> {
        parse_simd_checked::<{ u8::MAX as u64 }>(s).map(|v| v as u8)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u8, usize), AtoiSimdError> {
        parse_simd::<{ u8::MAX as u64 }>(s).map(|(v, i)| (v as u8, i))
    }
}

impl ParsePos for i8 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i8, AtoiSimdError> {
        parse_simd_checked::<{ i8::MAX as u64 }>(s).map(|v| v as i8)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i8, usize), AtoiSimdError> {
        parse_simd::<{ i8::MAX as u64 }>(s).map(|(v, i)| (v as i8, i))
    }
}

impl ParseNeg for i8 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i8, AtoiSimdError> {
        parse_simd_checked_neg::<{ i8::MIN as i64 }>(s).map(|v| v as i8)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i8, usize), AtoiSimdError> {
        parse_simd_neg::<{ i8::MIN as i64 }>(s).map(|(v, i)| (v as i8, i))
    }
}

impl ParsePos for u16 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u16, AtoiSimdError> {
        parse_simd_checked::<{ u16::MAX as u64 }>(s).map(|v| v as u16)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u16, usize), AtoiSimdError> {
        parse_simd::<{ u16::MAX as u64 }>(s).map(|(v, i)| (v as u16, i))
    }
}

impl ParsePos for i16 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i16, AtoiSimdError> {
        parse_simd_checked::<{ i16::MAX as u64 }>(s).map(|v| v as i16)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i16, usize), AtoiSimdError> {
        parse_simd::<{ i16::MAX as u64 }>(s).map(|(v, i)| (v as i16, i))
    }
}

impl ParseNeg for i16 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i16, AtoiSimdError> {
        parse_simd_checked_neg::<{ i16::MIN as i64 }>(s).map(|v| v as i16)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i16, usize), AtoiSimdError> {
        parse_simd_neg::<{ i16::MIN as i64 }>(s).map(|(v, i)| (v as i16, i))
    }
}

impl ParsePos for u32 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<u32, AtoiSimdError> {
        parse_simd_checked::<{ u32::MAX as u64 }>(s).map(|v| v as u32)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(u32, usize), AtoiSimdError> {
        parse_simd::<{ u32::MAX as u64 }>(s).map(|(v, i)| (v as u32, i))
    }
}

impl ParsePos for i32 {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<i32, AtoiSimdError> {
        parse_simd_checked::<{ i32::MAX as u64 }>(s).map(|v| v as i32)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(i32, usize), AtoiSimdError> {
        parse_simd::<{ i32::MAX as u64 }>(s).map(|(v, i)| (v as i32, i))
    }
}

impl ParseNeg for i32 {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<i32, AtoiSimdError> {
        parse_simd_checked_neg::<{ i32::MIN as i64 }>(s).map(|v| v as i32)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(i32, usize), AtoiSimdError> {
        parse_simd_neg::<{ i32::MIN as i64 }>(s).map(|(v, i)| (v as i32, i))
    }
}

#[cfg(target_pointer_width = "32")]
impl ParsePos for usize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<usize, AtoiSimdError> {
        parse_simd_checked::<{ u32::MAX as u64 }>(s).map(|v| v as usize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(usize, usize), AtoiSimdError> {
        parse_simd::<{ u32::MAX as u64 }>(s).map(|(v, i)| (v as usize, i))
    }
}

#[cfg(target_pointer_width = "32")]
impl ParsePos for isize {
    #[inline(always)]
    fn atoi_simd_parse_pos(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_simd_checked::<{ i32::MAX as u64 }>(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_pos(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_simd::<{ i32::MAX as u64 }>(s).map(|(v, i)| (v as isize, i))
    }
}

#[cfg(target_pointer_width = "32")]
impl ParseNeg for isize {
    #[inline(always)]
    fn atoi_simd_parse_neg(s: &[u8]) -> Result<isize, AtoiSimdError> {
        parse_simd_checked_neg::<{ i32::MIN as i64 }>(s).map(|v| v as isize)
    }

    #[inline(always)]
    fn atoi_simd_parse_any_neg(s: &[u8]) -> Result<(isize, usize), AtoiSimdError> {
        parse_simd_neg::<{ i32::MIN as i64 }>(s).map(|(v, i)| (v as isize, i))
    }
}
