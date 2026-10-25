#[cfg(not(feature = "approx-number-parsing"))]
mod correct;

#[cfg(feature = "approx-number-parsing")]
mod approx;

use crate::safer_unchecked::GetSaferUnchecked;

#[cfg(all(target_arch = "x86", feature = "swar-number-parsing"))]
use std::arch::x86 as arch;

#[cfg(all(target_arch = "x86_64", feature = "swar-number-parsing"))]
use std::arch::x86_64 as arch;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "swar-number-parsing"
))]
use arch::{
    __m128i, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_packus_epi32, _mm_set1_epi8, _mm_setr_epi8, _mm_setr_epi16, _mm_sub_epi8,
};

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_integer(c: u8) -> bool {
    c.is_ascii_digit()
}

// We need to check that the character following a zero is valid. This is
// probably frequent and it is hard than it looks. We are building all of this
// just to differentiate between 0x1 (invalid), 0,1 (valid) 0e1 (valid)...
const STRUCTURAL_OR_WHITESPACE_OR_EXPONENT_OR_DECIMAL_NEGATED: [bool; 256] = [
    false, true, true, true, true, true, true, true, true, false, false, true, true, false, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, false, true, true, true, true, true, true, true, true, true, true, true, false, true,
    false, true, true, true, true, true, true, true, true, true, true, true, false, true, true,
    true, true, true, true, true, true, true, true, false, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    false, true, false, true, true, true, true, true, true, true, false, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, false, true, false, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true,
];

#[cfg_attr(not(feature = "no-inline"), inline)]
fn is_not_structural_or_whitespace_or_exponent_or_decimal(c: u8) -> bool {
    unsafe {
        *STRUCTURAL_OR_WHITESPACE_OR_EXPONENT_OR_DECIMAL_NEGATED.get_kinda_unchecked(c as usize)
    }
}

// #ifdef _MSC_VER
// check quickly whether the next 8 chars are made of digits
// at a glance, it looks better than Mula's
// http://0x80.pl/articles/swar-digits-validate.html

#[cfg(feature = "swar-number-parsing")]
#[cfg_attr(not(feature = "no-inline"), inline)]
#[allow(clippy::cast_ptr_alignment)]
fn is_made_of_eight_digits_fast(chars: [u8; 8]) -> bool {
    let val = u64::from_ne_bytes(chars);

    ((val & 0xF0F0_F0F0_F0F0_F0F0)
        | (((val.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0) >> 4))
        == 0x3333_3333_3333_3333
}

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "swar-number-parsing"
))]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment
)]
fn parse_eight_digits_unrolled(chars: &[u8]) -> u32 {
    unsafe {
        // this actually computes *16* values so we are being wasteful.
        let ascii0: __m128i = _mm_set1_epi8(b'0' as i8);
        let mul_1_10: __m128i =
            _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1);
        let mul_1_100: __m128i = _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1);
        let mul_1_10000: __m128i = _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1);
        // We know what we're doing right? :P
        let input: __m128i = _mm_sub_epi8(
            _mm_loadu_si128(
                chars
                    .get_kinda_unchecked(0..16)
                    .as_ptr()
                    .cast::<arch::__m128i>(),
            ),
            ascii0,
        );
        let t1: __m128i = _mm_maddubs_epi16(input, mul_1_10);
        let t2: __m128i = _mm_madd_epi16(t1, mul_1_100);
        let t3: __m128i = _mm_packus_epi32(t2, t2);
        let t4: __m128i = _mm_madd_epi16(t3, mul_1_10000);
        _mm_cvtsi128_si32(t4) as u32 // only captures the sum of the first 8 digits, drop the rest
    }
}

#[cfg_attr(not(feature = "no-inline"), inline)]
#[cfg(all(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    feature = "swar-number-parsing"
))]
#[allow(clippy::cast_ptr_alignment)]
fn parse_eight_digits_unrolled(chars: &[u8]) -> u32 {
    let val = unsafe { chars.as_ptr().cast::<u64>().read_unaligned() }; //    memcpy(&val, chars, sizeof(u64));
    let val = (val & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let val = (val & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;

    ((val & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32) as u32
}
