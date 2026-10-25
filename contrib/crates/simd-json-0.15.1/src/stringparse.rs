use std::ops::Range;

use crate::charutils::{codepoint_to_utf8, hex_to_u32_nocheck};
use crate::error::ErrorType;
use crate::safer_unchecked::GetSaferUnchecked;

/// begin copypasta
/// These chars yield themselves: " \ /
/// b -> backspace, f -> formfeed, n -> newline, r -> cr, t -> horizontal tab
/// u not handled in this table as it's Container
pub(crate) const ESCAPE_MAP: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x0.
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x2f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, // 0x4.
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x5c, 0, 0, 0, // 0x5.
    0, 0, 0x08, 0, 0, 0, 0x0c, 0, 0, 0, 0, 0, 0, 0, 0x0a, 0, // 0x6.
    0, 0, 0x0d, 0, 0x09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0x7.
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const HIGH_SURROGATES: Range<u32> = 0xd800..0xdc00;
const LOW_SURROGATES: Range<u32> = 0xdc00..0xe000;

/// handle a unicode codepoint
/// write appropriate values into dest
#[cfg_attr(not(feature = "no-inline"), inline)]
#[allow(dead_code)]
pub(crate) fn handle_unicode_codepoint(
    src_ptr: &[u8],
    dst_ptr: &mut [u8],
) -> Result<(usize, usize), ErrorType> {
    let (code_point, src_offset) = get_unicode_codepoint(src_ptr)?;
    let offset: usize = codepoint_to_utf8(code_point, dst_ptr);
    Ok((offset, src_offset))
}

/// handle a unicode codepoint
/// write appropriate values into dest
/// src will advance 6 bytes or 12 bytes
/// dest will advance a variable amount (return via pointer)
/// return true if the unicode codepoint was valid
/// We work in little-endian then swap at write time
#[cfg_attr(not(feature = "no-inline"), inline)]
pub(crate) fn get_unicode_codepoint(mut src_ptr: &[u8]) -> Result<(u32, usize), ErrorType> {
    // hex_to_u32_nocheck fills high 16 bits of the return value with 1s if the
    // conversion isn't valid; we defer the check for this to inside the
    // multilingual plane check
    let mut code_point: u32 = hex_to_u32_nocheck(unsafe { src_ptr.get_kinda_unchecked(2..) });
    src_ptr = unsafe { src_ptr.get_kinda_unchecked(6..) };
    let mut src_offset = 6;
    // check for low surrogate for characters outside the Basic
    // Multilingual Plane.
    if HIGH_SURROGATES.contains(&code_point) {
        if (unsafe { *src_ptr.get_kinda_unchecked(0) } != b'\\')
            || unsafe { *src_ptr.get_kinda_unchecked(1) } != b'u'
        {
            return Ok((0, src_offset));
        }

        let code_point_2: u32 = hex_to_u32_nocheck(unsafe { src_ptr.get_kinda_unchecked(2..) });

        // if the first code point is invalid we will get here, as we will go past
        // the check for being outside the Basic Multilingual plane. If we don't
        // find a \u immediately afterwards we fail out anyhow, but if we do,
        // this check catches both the case of the first code point being invalid
        // or the second code point being invalid.
        if ((code_point | code_point_2) >> 16) != 0 {
            return Ok((0, src_offset));
        }
        let Some(c1) = code_point.checked_sub(0xd800) else {
            return Err(ErrorType::InvalidUtf8);
        };
        let Some(c2) = code_point_2.checked_sub(0xdc00) else {
            return Err(ErrorType::InvalidUtf8);
        };
        code_point = ((c1 << 10) | c2) + 0x10000;
        src_offset += 6;
    } else if LOW_SURROGATES.contains(&code_point) {
        // This is a low surrogate on it's own, which is invalid.
        return Err(ErrorType::InvalidUtf8);
    }
    Ok((code_point, src_offset))
}
