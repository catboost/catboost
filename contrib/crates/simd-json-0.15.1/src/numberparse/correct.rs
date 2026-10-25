mod numberconst;

use self::numberconst::{MANTISSA_128, POW10, POW10_COMPONENTS};
use super::{is_integer, is_not_structural_or_whitespace_or_exponent_or_decimal};

#[cfg(feature = "swar-number-parsing")]
use super::{is_made_of_eight_digits_fast, parse_eight_digits_unrolled};

use crate::StaticNode;
use crate::charutils::is_structural_or_whitespace;
use crate::error::Error;
use crate::macros::{static_cast_i64, unlikely};
use crate::safer_unchecked::GetSaferUnchecked;
use crate::{Deserializer, ErrorType, Result};

macro_rules! get {
    ($buf:ident, $idx:expr_2021) => {
        unsafe { *$buf.get_kinda_unchecked($idx as usize) }
    };
}
macro_rules! err {
    ($idx:ident, $num:expr_2021) => {
        return Err(Error::new_c($idx, $num as char, ErrorType::InvalidNumber))
    };
}

macro_rules! check_overflow {
    ($overflowed:ident, $buf:ident, $idx:ident, $start_idx:ident, $end_index:ident) => {
        if $overflowed {
            #[cfg(not(feature = "big-int-as-float"))]
            {
                err!($idx, get!($buf, $idx))
            }
            #[cfg(feature = "big-int-as-float")]
            {
                return f64_from_parts_slow(
                    unsafe { $buf.get_kinda_unchecked($start_idx..$end_index) },
                    $start_idx,
                );
            }
        }
    };
}

#[cfg_attr(not(feature = "no-inline"), inline)]
#[allow(clippy::cast_possible_truncation)]
fn multiply_as_u128(a: u64, b: u64) -> (u64, u64) {
    let res: u128 = u128::from(a) * u128::from(b);
    (res as u64, (res >> 64) as u64)
}

impl Deserializer<'_> {
    #[allow(
        unused_unsafe,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::too_many_lines
    )]
    pub(crate) fn parse_number(idx: usize, buf: &[u8], negative: bool) -> Result<StaticNode> {
        let start_idx = idx;
        let mut idx = idx;
        if negative {
            idx += 1;
            if !is_integer(get!(buf, idx)) {
                err!(idx, get!(buf, idx))
            }
        }
        let mut start = idx;
        let mut num: u64 = 0;
        if get!(buf, idx) == b'0' {
            idx += 1;
            if is_not_structural_or_whitespace_or_exponent_or_decimal(get!(buf, idx)) {
                err!(idx, get!(buf, idx))
            }
        } else {
            if !is_integer(get!(buf, idx)) {
                err!(idx, get!(buf, idx))
            }
            num = u64::from(get!(buf, idx) - b'0');
            idx += 1;
            while is_integer(get!(buf, idx)) {
                num = 10_u64
                    .wrapping_mul(num)
                    .wrapping_add(u64::from(get!(buf, idx) - b'0'));
                idx += 1;
            }
        }
        let mut exponent: i64 = 0;
        let mut is_float = false;
        if get!(buf, idx) == b'.' {
            is_float = true;
            idx += 1;
            let first_after_period = idx as i64;
            if is_integer(get!(buf, idx)) {
                num = 10_u64
                    .wrapping_mul(num)
                    .wrapping_add(u64::from(get!(buf, idx) - b'0'));
                idx += 1;
            } else {
                err!(idx, get!(buf, idx))
            }

            #[cfg(feature = "swar-number-parsing")]
            {
                // safety: we have 8 bytes
                let chars: [u8; 8] =
                    unsafe { *(buf.get_kinda_unchecked(idx..idx + 8).as_ptr().cast()) };
                if is_made_of_eight_digits_fast(chars) {
                    num = 100_000_000_u64.wrapping_mul(num).wrapping_add(u64::from(
                        parse_eight_digits_unrolled(unsafe { buf.get_kinda_unchecked(idx..) }),
                    ));
                    idx += 8;
                }
            }
            while is_integer(get!(buf, idx)) {
                num = 10_u64
                    .wrapping_mul(num)
                    .wrapping_add(u64::from(get!(buf, idx) - b'0'));
                idx += 1;
            }
            exponent = first_after_period.wrapping_sub(idx as i64);
        }
        let mut digit_count = idx - start_idx - 1;
        match get!(buf, idx) {
            b'e' | b'E' => {
                is_float = true;
                idx += 1;
                let neg_exp: bool;
                match get!(buf, idx) {
                    b'-' => {
                        neg_exp = true;
                        idx += 1;
                    }
                    b'+' => {
                        neg_exp = false;
                        idx += 1;
                    }
                    _ => {
                        neg_exp = false;
                    }
                }
                if !is_integer(get!(buf, idx)) {
                    err!(idx, get!(buf, idx))
                }
                let mut exp_number = i64::from(get!(buf, idx) - b'0');
                idx += 1;
                if is_integer(get!(buf, idx)) {
                    exp_number = 10 * exp_number + i64::from(get!(buf, idx) - b'0');
                    idx += 1;
                }
                if is_integer(get!(buf, idx)) {
                    exp_number = 10 * exp_number + i64::from(get!(buf, idx) - b'0');
                    idx += 1;
                }
                while is_integer(get!(buf, idx)) {
                    if exp_number > 0x0001_0000_0000 {
                        err!(idx, get!(buf, idx))
                    }
                    exp_number = 10 * exp_number + i64::from(get!(buf, idx) - b'0');
                    idx += 1;
                }
                exponent += if neg_exp { -exp_number } else { exp_number };
            }
            _ => {}
        }
        if is_float {
            if unlikely!(digit_count >= 19) {
                let start_digits = get!(buf, start) as usize;
                while get!(buf, start) == b'0' || get!(buf, start) == b'.' {
                    start += 1;
                }
                digit_count = digit_count.wrapping_sub(start.wrapping_sub(start_digits));
                if digit_count >= 19 {
                    return f64_from_parts_slow(
                        unsafe { buf.get_kinda_unchecked(start_idx..idx) },
                        start_idx,
                    );
                }
            }
            if is_structural_or_whitespace(get!(buf, idx)) == 0 {
                err!(idx, get!(buf, idx))
            }
            f64_from_parts(
                !negative,
                num,
                exponent as i32,
                unsafe { buf.get_kinda_unchecked(start_idx..idx) },
                start_idx,
            )
        } else if unlikely!(digit_count >= 18) {
            parse_large_integer(start_idx, buf, negative, idx)
        } else if is_structural_or_whitespace(get!(buf, idx)) == 0 {
            err!(idx, get!(buf, idx))
        } else {
            Ok(if negative {
                StaticNode::I64(unsafe { static_cast_i64!(num.wrapping_neg()) })
            // -(num as i64)
            } else {
                StaticNode::U64(num)
            })
        }
    }
}

#[cfg(not(feature = "128bit"))]
#[cold]
#[allow(clippy::cast_possible_wrap)]
fn parse_large_integer(
    start_idx: usize,
    buf: &[u8],
    negative: bool,
    #[allow(unused_variables)] end_index: usize,
) -> Result<StaticNode> {
    let mut idx = start_idx;
    if negative {
        idx += 1;
    }
    let mut num: u64 = 0;
    if get!(buf, idx) == b'0' {
        idx += 1;
    } else {
        num = u64::from(get!(buf, idx) - b'0');
        idx += 1;
        while is_integer(get!(buf, idx)) {
            let digit = u64::from(get!(buf, idx) - b'0');
            {
                let (res, overflowed) = 10_u64.overflowing_mul(num);
                check_overflow!(overflowed, buf, idx, start_idx, end_index);
                num = res;
            }
            {
                let (res, overflowed) = num.overflowing_add(digit);
                check_overflow!(overflowed, buf, idx, start_idx, end_index);
                num = res;
            }
            idx += 1;
        }
    }
    match (negative, num) {
        (true, 9_223_372_036_854_775_808) => Ok(StaticNode::I64(i64::MIN)),
        (true, 9_223_372_036_854_775_809..=u64::MAX) => err!(idx, get!(buf, idx)),
        (true, 0..=9_223_372_036_854_775_807) => Ok(StaticNode::I64(-(num as i64))),
        (false, _) => Ok(StaticNode::U64(num)),
    }
}

#[cfg(feature = "128bit")]
#[cold]
#[allow(clippy::cast_possible_wrap)]
fn parse_large_integer(
    start_idx: usize,
    buf: &[u8],
    negative: bool,
    #[allow(unused_variables)] end_index: usize,
) -> Result<StaticNode> {
    let mut idx = start_idx;
    if negative {
        idx += 1;
    }
    let mut num: u128 = 0;
    if get!(buf, idx) == b'0' {
        idx += 1;
    } else {
        num = u128::from(get!(buf, idx) - b'0');
        idx += 1;
        while is_integer(get!(buf, idx)) {
            let digit = u128::from(get!(buf, idx) - b'0');
            {
                let (res, overflowed) = 10_u128.overflowing_mul(num);
                check_overflow!(overflowed, buf, idx, start_idx, end_index);
                num = res;
            }
            {
                let (res, overflowed) = num.overflowing_add(digit);
                check_overflow!(overflowed, buf, idx, start_idx, end_index);
                num = res;
            }
            idx += 1;
        }
    }
    match (negative, num) {
        (true, 170_141_183_460_469_231_731_687_303_715_884_105_728_u128) => {
            Ok(StaticNode::I128(i128::MIN))
        }
        (true, 170_141_183_460_469_231_731_687_303_715_884_105_729_u128..=u128::MAX) => {
            err!(idx, get!(buf, idx))
        }
        (true, 0..=170_141_183_460_469_231_731_687_303_715_884_105_727_u128) => {
            if let Ok(i) = i64::try_from(-(num as i128)) {
                Ok(StaticNode::I64(i))
            } else {
                Ok(StaticNode::I128(-(num as i128)))
            }
        }
        (false, _) => {
            if let Ok(i) = u64::try_from(num) {
                Ok(StaticNode::U64(i))
            } else {
                Ok(StaticNode::U128(num))
            }
        }
    }
}

#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
fn f64_from_parts(
    positive: bool,
    significand: u64,
    exponent: i32,
    slice: &[u8],
    offset: usize,
) -> Result<StaticNode> {
    if (-22..=22).contains(&exponent) && significand <= 9_007_199_254_740_991 {
        let mut f = significand as f64;
        if exponent < 0 {
            f /= get!(POW10, -exponent);
        } else {
            f *= get!(POW10, exponent);
        }
        Ok(StaticNode::from(if positive { f } else { -f }))
    } else if significand == 0 {
        Ok(StaticNode::from(if positive { 0.0 } else { -0.0 }))
    } else if (-325..=308).contains(&exponent) {
        let (factor_mantissa, factor_exponent) = get!(POW10_COMPONENTS, exponent + 325);
        let mut leading_zeroes = u64::from(significand.leading_zeros());
        let f = significand << leading_zeroes;
        let (mut lower, mut upper) = multiply_as_u128(f, factor_mantissa);
        if upper & 0x1FF == 0x1FF && lower.wrapping_add(f) < lower {
            let factor_mantissa_low = get!(MANTISSA_128, exponent + 325);
            let (product_low, product_middle2) = multiply_as_u128(f, factor_mantissa_low);
            let product_middle1 = lower;
            let mut product_high = upper;
            let product_middle = product_middle1.wrapping_add(product_middle2);
            if product_middle < product_middle1 {
                product_high += 1;
            }
            if product_middle.wrapping_add(1) == 0
                && product_high & 0x1FF == 0x1FF
                && product_low.wrapping_add(f) < product_low
            {
                return f64_from_parts_slow(slice, offset);
            }
            upper = product_high;
            lower = product_middle;
        }
        let upperbit = upper.wrapping_shr(63);
        let mut mantissa = upper.wrapping_shr((upperbit + 9) as u32);
        leading_zeroes += 1 ^ upperbit;

        if lower == 0 && upper.trailing_zeros() >= 9 && mantissa & 3 == 1 {
            return f64_from_parts_slow(slice, offset);
        }
        mantissa += mantissa & 1;
        mantissa >>= 1;

        if mantissa >= 1 << 53 {
            mantissa = 1 << 52;
            leading_zeroes -= 1;
        }
        mantissa &= !(1 << 52);
        let real_exponent = (factor_exponent as u64).wrapping_sub(leading_zeroes);
        // we have to check that real_exponent is in range, otherwise we bail out
        if !(1..=2046).contains(&real_exponent) {
            return f64_from_parts_slow(slice, offset);
        }
        mantissa |= real_exponent.wrapping_shl(52);
        mantissa |= u64::from(!positive) << 63;
        let res = f64::from_bits(mantissa);
        if res.is_infinite() {
            err!(offset, get!(slice, offset))
        }
        Ok(StaticNode::from(res))
    } else {
        f64_from_parts_slow(slice, offset)
    }
}

#[cold]
fn f64_from_parts_slow(slice: &[u8], offset: usize) -> Result<StaticNode> {
    // we already validated the content of the slice we only need to translate
    // the slice to a string and parse it as parse is not defined for a u8 slice
    match unsafe { std::str::from_utf8_unchecked(slice).parse::<f64>() } {
        Ok(val) => {
            if val.is_infinite() {
                err!(offset, get!(slice, 0))
            }

            Ok(StaticNode::from(val))
        }
        Err(_) => err!(offset, get!(slice, offset)),
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::default_trait_access)]
    use crate::error::Error;
    use crate::value::owned::Value;
    use crate::value::owned::Value::Static;
    use crate::value::owned::to_value;
    use value_trait::StaticNode::{self, I64, U64};

    fn to_value_from_str(buf: &str) -> Result<Value, Error> {
        let mut val = String::from(buf);
        let val = unsafe { val.as_bytes_mut() };
        to_value(val)
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn float() -> Result<(), crate::Error> {
        assert_eq!(
            to_value_from_str("0.4e5").expect("40000.0"),
            Static(StaticNode::from(40000.0))
        );
        assert_eq!(
            to_value_from_str("-12345678901234.56789012")?,
            Static(StaticNode::from(-12_345_678_901_234.568))
        );
        assert_eq!(
            to_value_from_str("0.4e-001")?,
            Static(StaticNode::from(0.04))
        );
        assert_eq!(
            to_value_from_str("0.123456789e-12")?,
            Static(StaticNode::from(1.234_567_89e-13))
        );
        assert_eq!(to_value_from_str("1.234567890E+34")?, 1.234_567_89e34);
        assert_eq!(
            to_value_from_str("23456789012E66")?,
            Static(StaticNode::from(2.345_678_901_2e76))
        );
        assert_eq!(
            to_value_from_str("0.0000000000000000000000000000000000000000000000000123e50")
                .expect("1.23"),
            Static(StaticNode::from(1.23))
        );
        assert_eq!(
            to_value_from_str("0.6").expect("0.6"),
            Static(StaticNode::from(0.6))
        );
        Ok(())
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn float_precision() -> Result<(), crate::Error> {
        assert_eq!(
            to_value_from_str("31.245270191439438")?,
            31.245_270_191_439_438
        );
        assert_eq!(
            to_value_from_str("-31.245270191439438")?,
            -31.245_270_191_439_438
        );
        assert_eq!(
            to_value_from_str("121.48791951161945")?,
            121.487_919_511_619_45
        );
        assert_eq!(
            to_value_from_str("-121.48791951161945")?,
            -121.487_919_511_619_45
        );
        assert_eq!(
            to_value_from_str("100.78399658203125")?,
            100.783_996_582_031_25
        );
        assert_eq!(
            to_value_from_str("-100.78399658203125")?,
            -100.783_996_582_031_25
        );
        Ok(())
    }

    #[test]
    fn int_trailing_invalid() {
        // todo: these should fail but is not distinguished from trailing padding
        assert!(to_value_from_str("123\x00").is_ok());
        assert!(to_value_from_str("[123\x00]").is_ok());
    }

    #[test]
    fn float_trailing_invalid() {
        assert!(to_value_from_str("0.1.2").is_err());
        assert!(to_value_from_str("[0.1.2]").is_err());
        assert!(to_value_from_str("[-1.0.]").is_err());
        assert!(to_value_from_str("[1.2a-3]").is_err());
        assert!(to_value_from_str("[1.8011670033376514H-308]").is_err());
    }

    #[test]
    fn bad_dot() {
        assert!(to_value_from_str("1.").is_err());
        assert!(to_value_from_str("1.e").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000.").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000.e").is_err());
    }

    #[test]
    fn bad_e() {
        assert!(to_value_from_str("1.0e").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000.0e").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000.0ee").is_err());
    }

    #[test]
    fn infinite_literal() {
        assert!(to_value_from_str("inf").is_err());
        assert!(to_value_from_str("Inf").is_err());
        assert!(to_value_from_str("-inf").is_err());
        assert!(to_value_from_str("-Inf").is_err());
        assert!(to_value_from_str("infinity").is_err());
        assert!(to_value_from_str("Infinity").is_err());
        assert!(to_value_from_str("-infinity").is_err());
        assert!(to_value_from_str("-Infinity").is_err());
    }

    #[test]
    fn infinite_exponent() {
        assert!(to_value_from_str("1e309").is_err());
        assert!(to_value_from_str("1e1000").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000e309").is_err());
        assert!(to_value_from_str("100000000000000000000000000000000000000000000e1000").is_err());
    }

    #[test]
    fn nan() {
        assert!(to_value_from_str("NaN").is_err());
        assert!(to_value_from_str("nan").is_err());
    }

    #[test]
    fn zero_int() -> Result<(), crate::Error> {
        assert_eq!(to_value_from_str("0")?, Static(U64(0)));
        Ok(())
    }

    #[test]
    fn zero_float() -> Result<(), crate::Error> {
        assert_eq!(to_value_from_str("0e1")?, Static(StaticNode::from(0.0)));
        assert_eq!(
            to_value_from_str("0.00e-00")?,
            Static(StaticNode::from(0.0))
        );
        assert_eq!(to_value_from_str("0e-1")?, Static(StaticNode::from(-0.0)));
        assert_eq!(
            to_value_from_str("-0.00e-00")?,
            Static(StaticNode::from(-0.0))
        );
        Ok(())
    }

    #[test]
    fn int() -> Result<(), crate::Error> {
        assert!(matches!(to_value_from_str("1")?, Static(U64(1))));
        assert_eq!(to_value_from_str("-1")?, Static(I64(-1)));
        assert_eq!(to_value_from_str("257")?, Static(U64(257)));
        Ok(())
    }

    #[test]
    fn minus_309() -> Result<(), crate::Error> {
        assert_eq!(
            to_value_from_str("-5.96916642387374e-309")?,
            Static(StaticNode::from(-5.969_166_423_873_74e-309))
        );
        Ok(())
    }
    #[allow(clippy::unreadable_literal)]
    #[test]
    fn tiny_float() -> Result<(), crate::Error> {
        assert_eq!(to_value_from_str("-0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000596916642387374")?, Static(StaticNode::from(-0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000596916642387374)));
        Ok(())
    }

    #[test]
    #[cfg(feature = "big-int-as-float")]
    fn huge_int() -> Result<(), crate::Error> {
        assert_eq!(
            to_value_from_str("999999999999999999999999999999")?,
            StaticNode::from(999999999999999999999999999999f64)
        );
        Ok(())
    }
}
