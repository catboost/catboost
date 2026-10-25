//! Module with common integer formatting logic for implementing the standard
//! library `core::fmt` traits.
//!
//! Most of these implementations were ported from the Rust standard library's
//! implementation for primitive integer types:
//! <https://doc.rust-lang.org/src/core/fmt/num.rs.html>

use crate::uint::U256;
use core::{fmt, mem::MaybeUninit, ptr, slice, str};

pub(crate) trait GenericRadix: Sized {
    const BASE: u8;
    const PREFIX: &'static str;
    fn digit(x: u8) -> u8;
    fn fmt_u256(&self, mut x: U256, is_nonnegative: bool, f: &mut fmt::Formatter) -> fmt::Result {
        // The radix can be as low as 2, so we need a buffer of at least 256
        // characters for a base 2 number.
        let zero = U256::ZERO;
        let mut buf = [MaybeUninit::<u8>::uninit(); 256];
        let mut curr = buf.len();
        let base = U256::from(Self::BASE);
        // Accumulate each digit of the number from the least significant
        // to the most significant figure.
        for byte in buf.iter_mut().rev() {
            let n = x % base; // Get the current place value.
            x /= base; // Deaccumulate the number.
            byte.write(Self::digit(n.as_u8())); // Store the digit in the buffer.
            curr -= 1;
            if x == zero {
                // No more digits left to accumulate.
                break;
            };
        }
        let buf = &buf[curr..];
        // SAFETY: The only chars in `buf` are created by `Self::digit` which are assumed to be
        // valid UTF-8
        let buf = unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                &buf[0] as *const _ as *const u8,
                buf.len(),
            ))
        };
        f.pad_integral(is_nonnegative, Self::PREFIX, buf)
    }
}

/// A binary (base 2) radix
#[derive(Clone, PartialEq)]
pub(crate) struct Binary;

/// An octal (base 8) radix
#[derive(Clone, PartialEq)]
pub(crate) struct Octal;

/// A hexadecimal (base 16) radix, formatted with lower-case characters
#[derive(Clone, PartialEq)]
pub(crate) struct LowerHex;

/// A hexadecimal (base 16) radix, formatted with upper-case characters
#[derive(Clone, PartialEq)]
pub(crate) struct UpperHex;

macro_rules! radix {
    ($T:ident, $base:expr, $prefix:expr, $($x:pat => $conv:expr),+) => {
        impl GenericRadix for $T {
            const BASE: u8 = $base;
            const PREFIX: &'static str = $prefix;
            fn digit(x: u8) -> u8 {
                match x {
                    $($x => $conv,)+
                    x => panic!("number not in the range 0..={}: {}", Self::BASE - 1, x),
                }
            }
        }
    }
}

radix! { Binary,    2, "0b", x @  0 ..=  1 => b'0' + x }
radix! { Octal,     8, "0o", x @  0 ..=  7 => b'0' + x }
radix! { LowerHex, 16, "0x", x @  0 ..=  9 => b'0' + x, x @ 10 ..= 15 => b'a' + (x - 10) }
radix! { UpperHex, 16, "0x", x @  0 ..=  9 => b'0' + x, x @ 10 ..= 15 => b'A' + (x - 10) }

const DEC_DIGITS_LUT: &[u8; 200] = b"\
    0001020304050607080910111213141516171819\
    2021222324252627282930313233343536373839\
    4041424344454647484950515253545556575859\
    6061626364656667686970717273747576777879\
    8081828384858687888990919293949596979899";

pub(crate) fn fmt_u256(mut n: U256, is_nonnegative: bool, f: &mut fmt::Formatter) -> fmt::Result {
    // 2^256 is about 1*10^78, so 79 gives an extra byte of space
    let mut buf = [MaybeUninit::<u8>::uninit(); 79];
    let mut curr = buf.len() as isize;
    let buf_ptr = &mut buf[0] as *mut _ as *mut u8;
    let lut_ptr = DEC_DIGITS_LUT.as_ptr();

    // SAFETY: Since `d1` and `d2` are always less than or equal to `198`, we
    // can copy from `lut_ptr[d1..d1 + 1]` and `lut_ptr[d2..d2 + 1]`. To show
    // that it's OK to copy into `buf_ptr`, notice that at the beginning
    // `curr == buf.len() == 39 > log(n)` since `n < 2^128 < 10^39`, and at
    // each step this is kept the same as `n` is divided. Since `n` is always
    // non-negative, this means that `curr > 0` so `buf_ptr[curr..curr + 1]`
    // is safe to access.
    unsafe {
        // eagerly decode 4 characters at a time
        while n >= 10000 {
            let (q, r) = n.div_rem(U256::new(10000));
            n = q;
            let rem = r.as_isize();

            let d1 = (rem / 100) << 1;
            let d2 = (rem % 100) << 1;
            curr -= 4;

            // We are allowed to copy to `buf_ptr[curr..curr + 3]` here since
            // otherwise `curr < 0`. But then `n` was originally at least `10000^10`
            // which is `10^40 > 2^128 > n`.
            ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d2), buf_ptr.offset(curr + 2), 2);
        }

        // if we reach here numbers are <= 9999, so at most 4 chars long
        let mut n = n.as_isize(); // possibly reduce 64bit math

        // decode 2 more chars, if > 2 chars
        if n >= 100 {
            let d1 = (n % 100) << 1;
            n /= 100;
            curr -= 2;
            ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
        }

        // decode last 1 or 2 chars
        if n < 10 {
            curr -= 1;
            *buf_ptr.offset(curr) = (n as u8) + b'0';
        } else {
            let d1 = n << 1;
            curr -= 2;
            ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
        }
    }

    // SAFETY: `curr` > 0 (since we made `buf` large enough), and all the chars are valid
    // UTF-8 since `DEC_DIGITS_LUT` is
    let buf_slice = unsafe {
        str::from_utf8_unchecked(slice::from_raw_parts(
            buf_ptr.offset(curr),
            buf.len() - curr as usize,
        ))
    };
    f.pad_integral(is_nonnegative, "", buf_slice)
}
