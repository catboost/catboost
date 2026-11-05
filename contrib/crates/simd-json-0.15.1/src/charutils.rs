use crate::safer_unchecked::GetSaferUnchecked;

const STRUCTURAL_OR_WHITESPACE_NEGATED: [u32; 256] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

const STRUCTURAL_OR_WHITESPACE: [u32; 256] = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_not_structural_or_whitespace(c: u8) -> u32 {
    unsafe { *STRUCTURAL_OR_WHITESPACE_NEGATED.get_kinda_unchecked(c as usize) }
}

#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn is_structural_or_whitespace(c: u8) -> u32 {
    unsafe { *STRUCTURAL_OR_WHITESPACE.get_kinda_unchecked(c as usize) }
}

const DIGITTOVAL: [i8; 256] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10,
    11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
];

// returns a value with the high 16 bits set if not valid
// otherwise returns the conversion of the 4 hex digits at src into the bottom 16 bits of the 32-bit
// return register
#[cfg_attr(not(feature = "no-inline"), inline)]
#[allow(clippy::cast_sign_loss)]
pub fn hex_to_u32_nocheck(src: &[u8]) -> u32 {
    // strictly speaking, static inline is a C-ism
    // all these will sign-extend the chars looked up, placing 1-bits into the high 28 bits of every
    // invalid value. After the shifts, this will *still* result in the outcome that the high 16 bits of any
    // value with any invalid char will be all 1's. We check for this in the caller.
    unsafe {
        let v1: i32 =
            i32::from(*DIGITTOVAL.get_kinda_unchecked(*src.get_kinda_unchecked(0) as usize));
        let v2: i32 =
            i32::from(*DIGITTOVAL.get_kinda_unchecked(*src.get_kinda_unchecked(1) as usize));
        let v3: i32 =
            i32::from(*DIGITTOVAL.get_kinda_unchecked(*src.get_kinda_unchecked(2) as usize));
        let v4: i32 =
            i32::from(*DIGITTOVAL.get_kinda_unchecked(*src.get_kinda_unchecked(3) as usize));
        ((v1 << 12) | (v2 << 8) | (v3 << 4) | v4) as u32
    }
}

// given a code point cp, writes to c
// the utf-8 code, outputting the length in
// bytes, if the length is zero, the code point
// is invalid
//
// This can possibly be made faster using pdep
// and clz and table lookups, but JSON documents
// have few escaped code points, and the following
// function looks cheap.
//
// Note: we assume that surrogates are treated separately
//
#[cfg_attr(not(feature = "no-inline"), inline)]
#[allow(clippy::cast_possible_truncation)]
pub fn codepoint_to_utf8(cp: u32, c: &mut [u8]) -> usize {
    unsafe {
        if cp <= 0x7F {
            *c.get_kinda_unchecked_mut(0) = cp as u8;
            1 // ascii
        } else if cp <= 0x7FF {
            *c.get_kinda_unchecked_mut(0) = ((cp >> 6) + 192) as u8;
            *c.get_kinda_unchecked_mut(1) = ((cp & 63) + 128) as u8;
            2
            // universal plane
            //  Surrogates are treated elsewhere...
            //} //else if (0xd800 <= cp && cp <= 0xdfff) {
            //  return 0; // surrogates // could put assert here
        } else if cp <= 0xFFFF {
            *c.get_kinda_unchecked_mut(0) = ((cp >> 12) + 224) as u8;
            *c.get_kinda_unchecked_mut(1) = (((cp >> 6) & 63) + 128) as u8;
            *c.get_kinda_unchecked_mut(2) = ((cp & 63) + 128) as u8;
            3
        } else if cp <= 0x0010_FFFF {
            // if you know you have a valid code point, this is not needed
            *c.get_kinda_unchecked_mut(0) = ((cp >> 18) + 240) as u8;
            *c.get_kinda_unchecked_mut(1) = (((cp >> 12) & 63) + 128) as u8;
            *c.get_kinda_unchecked_mut(2) = (((cp >> 6) & 63) + 128) as u8;
            *c.get_kinda_unchecked_mut(3) = ((cp & 63) + 128) as u8;
            4
        } else {
            // will return 0 when the code point was too large.
            0
        }
    }
}
