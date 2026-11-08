#![allow(dead_code)]

use crate::AtoiSimdError;
use debug_unsafe::slice::SliceGetter;

macro_rules! overflow {
    ($curr:ident * 10 + $more:ident, $max:expr) => {
        $curr >= $max / 10 && ($curr > $max / 10 || $more > $max % 10)
    };
}

macro_rules! overflow_neg {
    ($curr:ident * 10 - $more:ident, $max:expr) => {
        $curr <= $max / 10 && ($curr < $max / 10 || $more > -($max % 10))
    };
}

#[inline]
pub(crate) fn parse_short_pos<const MAX: u64>(s: &[u8]) -> Result<(u64, usize), AtoiSimdError> {
    let mut i = 0;
    if s.len() == i {
        return Err(AtoiSimdError::Empty);
    }
    match s.get_safe_unchecked(i) {
        c @ b'0'..=b'9' => {
            let mut res = (c & 0xF) as u64;
            i += 1;
            while s.len() > i {
                match s.get_safe_unchecked(i) {
                    c @ b'0'..=b'9' => {
                        let digit = (c & 0xF) as u64;

                        if MAX <= u32::MAX as u64 && overflow!(res * 10 + digit, MAX) {
                            return Err(AtoiSimdError::Overflow(s));
                        }

                        res = res * 10 + digit;
                        i += 1;
                    }
                    _ => return Ok((res, i)),
                }
            }
            Ok((res, i))
        }
        _ => Err(AtoiSimdError::Empty),
    }
}

#[inline]
pub(crate) fn parse_short_neg<const MIN: i64>(s: &[u8]) -> Result<(i64, usize), AtoiSimdError> {
    debug_assert!(MIN < 0);
    let mut i = 0;
    if s.len() == i {
        return Err(AtoiSimdError::Empty);
    }
    match s.get_safe_unchecked(i) {
        c @ b'0'..=b'9' => {
            let mut res = -((c & 0xF) as i64);
            i += 1;
            while s.len() > i {
                match s.get_safe_unchecked(i) {
                    c @ b'0'..=b'9' => {
                        let digit = (c & 0xF) as i64;

                        if MIN >= i32::MIN as i64 && overflow_neg!(res * 10 - digit, MIN) {
                            // can't overflow, because MIN is bigger than i64::MIN
                            return Err(AtoiSimdError::Overflow(s));
                        }

                        res = res * 10 - digit;
                        i += 1;
                    }
                    _ => return Ok((res, i)),
                }
            }
            Ok((res, i))
        }
        _ => Err(AtoiSimdError::Empty),
    }
}

/* #[inline(always)]
pub(crate) fn parse_short_checked_neg<const MIN: i64>(s: &[u8]) -> Result<i64, AtoiSimdError> {
    debug_assert!(MIN < 0);
    let (res, len) = parse_short_neg::<MIN>(s)?;
    if len != s.len() {
        return Err(AtoiSimdError::Invalid64(-res as u64, len, s));
    }

    Ok(res)
} */
