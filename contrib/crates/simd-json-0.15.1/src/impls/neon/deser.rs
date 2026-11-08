use crate::Deserializer;
use crate::Result;
use crate::SillyWrapper;
use crate::error::ErrorType;
use crate::impls::neon::stage1::bit_mask;
use crate::safer_unchecked::GetSaferUnchecked;
use crate::stringparse::{ESCAPE_MAP, handle_unicode_codepoint};

use std::arch::aarch64::{
    uint8x16_t, vandq_u8, vceqq_u8, vgetq_lane_u32, vld1q_u8, vmovq_n_u8, vpaddq_u8,
    vreinterpretq_u32_u8,
};

#[cfg_attr(not(feature = "no-inline"), inline)]
fn find_bs_bits_and_quote_bits(v0: uint8x16_t, v1: uint8x16_t) -> (u32, u32) {
    unsafe {
        let quote_mask = vmovq_n_u8(b'"');
        let bs_mask = vmovq_n_u8(b'\\');
        let bit_mask = bit_mask();

        let cmp_bs_0: uint8x16_t = vceqq_u8(v0, bs_mask);
        let cmp_bs_1: uint8x16_t = vceqq_u8(v1, bs_mask);
        let cmp_qt_0: uint8x16_t = vceqq_u8(v0, quote_mask);
        let cmp_qt_1: uint8x16_t = vceqq_u8(v1, quote_mask);

        let cmp_bs_0 = vandq_u8(cmp_bs_0, bit_mask);
        let cmp_bs_1 = vandq_u8(cmp_bs_1, bit_mask);
        let cmp_qt_0 = vandq_u8(cmp_qt_0, bit_mask);
        let cmp_qt_1 = vandq_u8(cmp_qt_1, bit_mask);

        let sum0: uint8x16_t = vpaddq_u8(cmp_bs_0, cmp_bs_1);
        let sum1: uint8x16_t = vpaddq_u8(cmp_qt_0, cmp_qt_1);
        let sum0 = vpaddq_u8(sum0, sum1);
        let sum0 = vpaddq_u8(sum0, sum0);

        (
            vgetq_lane_u32(vreinterpretq_u32_u8(sum0), 0),
            vgetq_lane_u32(vreinterpretq_u32_u8(sum0), 1),
        )
    }
}

#[allow(clippy::if_not_else, clippy::too_many_lines)]
#[cfg_attr(not(feature = "no-inline"), inline)]
pub(crate) fn parse_str<'invoke, 'de>(
    input: SillyWrapper<'de>,
    data: &'invoke [u8],
    buffer: &'invoke mut [u8],
    mut idx: usize,
) -> Result<&'de str> {
    use ErrorType::{InvalidEscape, InvalidUnicodeCodepoint};
    let input = input.input;

    // Add 1 to skip the initial "
    idx += 1;
    //let mut read: usize = 0;

    // we include the terminal '"' so we know where to end
    // This is safe since we check sub's length in the range access above and only
    // create sub sliced form sub to `sub.len()`.

    let src: &[u8] = unsafe { data.get_kinda_unchecked(idx..) };
    let mut src_i: usize = 0;
    let mut len = src_i;
    loop {
        let (v0, v1) = unsafe {
            (
                vld1q_u8(src.get_kinda_unchecked(src_i..src_i + 16).as_ptr()),
                vld1q_u8(src.get_kinda_unchecked(src_i + 16..src_i + 32).as_ptr()),
            )
        };

        let (bs_bits, quote_bits) = find_bs_bits_and_quote_bits(v0, v1);

        if (bs_bits.wrapping_sub(1) & quote_bits) != 0 {
            // we encountered quotes first. Move dst to point to quotes and exit
            // find out where the quote is...
            let quote_dist: u32 = quote_bits.trailing_zeros();

            ///////////////////////
            // Above, check for overflow in case someone has a crazy string (>=4GB?)
            // But only add the overflow check when the document itself exceeds 4GB
            // Currently unneeded because we refuse to parse docs larger or equal to 4GB.
            ////////////////////////

            // we advance the point, accounting for the fact that we have a NULl termination

            len += quote_dist as usize;
            unsafe {
                let v =
                    std::str::from_utf8_unchecked(std::slice::from_raw_parts(input.add(idx), len));
                return Ok(v);
            }

            // we compare the pointers since we care if they are 'at the same spot'
            // not if they are the same value
        }
        if (quote_bits.wrapping_sub(1) & bs_bits) == 0 {
            // they are the same. Since they can't co-occur, it means we encountered
            // neither.
            src_i += 32;
            len += 32;
        } else {
            // Move to the 'bad' character
            let bs_dist: u32 = bs_bits.trailing_zeros();
            len += bs_dist as usize;
            src_i += bs_dist as usize;
            break;
        }
    }

    let mut dst_i: usize = 0;

    // To be more conform with upstream
    loop {
        let (v0, v1) = unsafe {
            (
                vld1q_u8(src.get_kinda_unchecked(src_i..src_i + 16).as_ptr()),
                vld1q_u8(src.get_kinda_unchecked(src_i + 16..src_i + 32).as_ptr()),
            )
        };

        unsafe {
            buffer
                .get_kinda_unchecked_mut(dst_i..dst_i + 32)
                .copy_from_slice(src.get_kinda_unchecked(src_i..src_i + 32));
        }

        // store to dest unconditionally - we can overwrite the bits we don't like
        // later
        let (bs_bits, quote_bits) = find_bs_bits_and_quote_bits(v0, v1);

        if (bs_bits.wrapping_sub(1) & quote_bits) != 0 {
            // we encountered quotes first. Move dst to point to quotes and exit
            // find out where the quote is...
            let quote_dist: u32 = quote_bits.trailing_zeros();

            ///////////////////////
            // Above, check for overflow in case someone has a crazy string (>=4GB?)
            // But only add the overflow check when the document itself exceeds 4GB
            // Currently unneeded because we refuse to parse docs larger or equal to 4GB.
            ////////////////////////

            // we advance the point, accounting for the fact that we have a NULl termination

            dst_i += quote_dist as usize;
            unsafe {
                input
                    .add(idx + len)
                    .copy_from_nonoverlapping(buffer.as_ptr(), dst_i);
                let v = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    input.add(idx),
                    len + dst_i,
                ));
                return Ok(v);
            }

            // we compare the pointers since we care if they are 'at the same spot'
            // not if they are the same value
        }
        if (quote_bits.wrapping_sub(1) & bs_bits) != 0 {
            // find out where the backspace is
            let bs_dist: u32 = bs_bits.trailing_zeros();
            let escape_char: u8 = unsafe { *src.get_kinda_unchecked(src_i + bs_dist as usize + 1) };
            // we encountered backslash first. Handle backslash
            if escape_char == b'u' {
                // move src/dst up to the start; they will be further adjusted
                // within the unicode codepoint handling code.
                src_i += bs_dist as usize;
                dst_i += bs_dist as usize;
                let Ok((o, s)) =
                    handle_unicode_codepoint(unsafe { src.get_kinda_unchecked(src_i..) }, unsafe {
                        buffer.get_kinda_unchecked_mut(dst_i..)
                    })
                else {
                    return Err(Deserializer::error_c(src_i, 'u', InvalidUnicodeCodepoint));
                };
                if o == 0 {
                    return Err(Deserializer::error_c(src_i, 'u', InvalidUnicodeCodepoint));
                }
                // We moved o steps forward at the destination and 6 on the source
                src_i += s;
                dst_i += o;
            } else {
                // simple 1:1 conversion. Will eat bs_dist+2 characters in input and
                // write bs_dist+1 characters to output
                // note this may reach beyond the part of the buffer we've actually
                // seen. I think this is ok
                let escape_result: u8 =
                    unsafe { *ESCAPE_MAP.get_kinda_unchecked(escape_char as usize) };
                if escape_result == 0 {
                    return Err(Deserializer::error_c(
                        src_i,
                        escape_char as char,
                        InvalidEscape,
                    ));
                }
                unsafe {
                    *buffer.get_kinda_unchecked_mut(dst_i + bs_dist as usize) = escape_result;
                }
                src_i += bs_dist as usize + 2;
                dst_i += bs_dist as usize + 1;
            }
        } else {
            // they are the same. Since they can't co-occur, it means we encountered
            // neither.
            src_i += 32;
            dst_i += 32;
        }
    }
}
