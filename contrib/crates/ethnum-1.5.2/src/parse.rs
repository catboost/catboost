//! Module with common integer parsing logic.
//!
//! Most of these implementations were ported from the Rust standard library's
//! implementation for primitive integer types:
//! <https://doc.rust-lang.org/src/core/num/mod.rs.html>

use crate::U256;
use core::{
    mem,
    num::{IntErrorKind, ParseIntError},
    ops::{Add, Mul, Sub},
};

#[doc(hidden)]
pub(crate) trait FromStrRadixHelper:
    PartialOrd + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    const MIN: Self;
    fn from_u32(u: u32) -> Self;
    fn checked_mul(&self, other: u32) -> Option<Self>;
    fn checked_sub(&self, other: u32) -> Option<Self>;
    fn checked_add(&self, other: u32) -> Option<Self>;
}

#[inline(always)]
fn can_not_overflow<T>(radix: u32, is_signed_ty: bool, digits: &[u8]) -> bool {
    radix <= 16 && digits.len() <= mem::size_of::<T>() * 2 - is_signed_ty as usize
}

pub(crate) fn from_str_radix<T: FromStrRadixHelper>(
    src: &str,
    radix: u32,
    prefix: Option<&str>,
) -> Result<T, ParseIntError> {
    use self::IntErrorKind::*;
    use crate::error::pie;

    assert!(
        (2..=36).contains(&radix),
        "from_str_radix_int: must lie in the range `[2, 36]` - found {}",
        radix
    );

    if src.is_empty() {
        return Err(pie(Empty));
    }

    let is_signed_ty = T::from_u32(0) > T::MIN;

    // all valid digits are ascii, so we will just iterate over the utf8 bytes
    // and cast them to chars. .to_digit() will safely return None for anything
    // other than a valid ascii digit for the given radix, including the first-byte
    // of multi-byte sequences
    let src = src.as_bytes();

    let (is_positive, prefixed_digits) = match src[0] {
        b'+' | b'-' if src[1..].is_empty() => {
            return Err(pie(InvalidDigit));
        }
        b'+' => (true, &src[1..]),
        b'-' if is_signed_ty => (false, &src[1..]),
        _ => (true, src),
    };

    let digits = match prefix {
        Some(prefix) => prefixed_digits
            .strip_prefix(prefix.as_bytes())
            .ok_or(pie(InvalidDigit))?,
        None => prefixed_digits,
    };
    if digits.is_empty() {
        return Err(pie(InvalidDigit));
    }

    let mut result = T::from_u32(0);

    if can_not_overflow::<T>(radix, is_signed_ty, digits) {
        // If the len of the str is short compared to the range of the type
        // we are parsing into, then we can be certain that an overflow will not occur.
        // This bound is when `radix.pow(digits.len()) - 1 <= T::MAX` but the condition
        // above is a faster (conservative) approximation of this.
        //
        // Consider radix 16 as it has the highest information density per digit and will thus overflow the earliest:
        // `u8::MAX` is `ff` - any str of len 2 is guaranteed to not overflow.
        // `i8::MAX` is `7f` - only a str of len 1 is guaranteed to not overflow.
        macro_rules! run_unchecked_loop {
            ($unchecked_additive_op:expr) => {
                for &c in digits {
                    result = result * T::from_u32(radix);
                    let x = (c as char).to_digit(radix).ok_or(pie(InvalidDigit))?;
                    result = $unchecked_additive_op(result, T::from_u32(x));
                }
            };
        }
        if is_positive {
            run_unchecked_loop!(<T as core::ops::Add>::add)
        } else {
            run_unchecked_loop!(<T as core::ops::Sub>::sub)
        };
    } else {
        macro_rules! run_checked_loop {
            ($checked_additive_op:ident, $overflow_err:expr) => {
                for &c in digits {
                    // When `radix` is passed in as a literal, rather than doing a slow `imul`
                    // the compiler can use shifts if `radix` can be expressed as a
                    // sum of powers of 2 (x*10 can be written as x*8 + x*2).
                    // When the compiler can't use these optimisations,
                    // the latency of the multiplication can be hidden by issuing it
                    // before the result is needed to improve performance on
                    // modern out-of-order CPU as multiplication here is slower
                    // than the other instructions, we can get the end result faster
                    // doing multiplication first and let the CPU spends other cycles
                    // doing other computation and get multiplication result later.
                    let mul = result.checked_mul(radix);
                    let x = (c as char).to_digit(radix).ok_or(pie(InvalidDigit))?;
                    result = mul.ok_or_else($overflow_err)?;
                    result = T::$checked_additive_op(&result, x).ok_or_else($overflow_err)?;
                }
            };
        }
        if is_positive {
            run_checked_loop!(checked_add, || pie(PosOverflow))
        } else {
            run_checked_loop!(checked_sub, || pie(NegOverflow))
        };
    }
    Ok(result)
}

pub(crate) fn from_str_prefixed<T: FromStrRadixHelper>(src: &str) -> Result<T, ParseIntError> {
    from_str_radix(src, 2, Some("0b"))
        .or_else(|_| from_str_radix(src, 8, Some("0o")))
        .or_else(|_| from_str_radix(src, 16, Some("0x")))
        .or_else(|_| from_str_radix(src, 10, None))
}

pub(crate) const fn const_from_str_prefixed(bytes: &[u8], start: usize) -> U256 {
    const fn check(overflow: bool) {
        assert!(!overflow, "overflows integer type");
    }

    const fn add(a: U256, b: u8) -> U256 {
        let (hi, lo) = a.into_words();

        let (lo, carry) = lo.overflowing_add(b as _);
        let (hi, overflow) = hi.overflowing_add(carry as _);
        check(overflow);

        U256::from_words(hi, lo)
    }

    const fn mul(a: U256, r: u128) -> U256 {
        let (hi, lo) = a.into_words();
        let (lh, ll) = (lo >> 64, lo & u64::MAX as u128);

        let ll = ll * r;
        let lh = lh * r;
        let (hi, overflow) = hi.overflowing_mul(r);
        check(overflow);

        let (lo, overflow) = ll.overflowing_add(lh << 64);
        check(overflow);
        let (hi, overflow) = hi.overflowing_add(lh >> 64);
        check(overflow);

        U256::from_words(hi, lo)
    }

    assert!(bytes.len() > start, "missing number");

    let (radix, mut i) = if bytes.len() - start > 2 {
        match (bytes[start], bytes[start + 1]) {
            (b'0', b'b') => (2, start + 2),
            (b'0', b'o') => (8, start + 2),
            (b'0', b'x') => (16, start + 2),
            _ => (10, start),
        }
    } else {
        (10, start)
    };

    let mut value = U256::ZERO;

    while i < bytes.len() {
        let byte = bytes[i];
        i += 1;

        if byte == b'_' || byte.is_ascii_whitespace() {
            continue;
        }

        let next = match (byte, radix) {
            (b'0'..=b'1', 2 | 8 | 10 | 16) => byte - b'0',
            (b'2'..=b'7', 8 | 10 | 16) => byte - b'0',
            (b'8'..=b'9', 10 | 16) => byte - b'0',
            (b'a'..=b'f', 16) => byte - b'a' + 0xa,
            (b'A'..=b'F', 16) => byte - b'A' + 0xa,
            (b'_', _) => continue,
            _ => panic!("invalid digit"),
        };
        value = add(mul(value, radix), next);
    }

    value
}
