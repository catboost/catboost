// Copyright Mozilla Foundation. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;
use crate::handles::*;
use crate::variant::*;

cfg_if! {
    if #[cfg(all(
        feature = "simd-accel",
        target_endian = "little",
    ))] {
        use crate::ascii::STRIDE;
        use simd_funcs::*;
        use core::simd::u8x16;
        use core::simd::u16x16;
        use core::simd::cmp::SimdPartialOrd;
        #[rustversion::since(1.95)]
        use core::simd::Select;

        #[inline(always)]
        fn shift_upper(unpacked: u16x16) -> u16x16 {
            let highest_ascii = u16x16::splat(0x7F);
            unpacked + unpacked.simd_gt(highest_ascii).select(u16x16::splat(0xF700), u16x16::splat(0))        }
    } else {
    }
}

pub struct UserDefinedDecoder;

impl UserDefinedDecoder {
    pub fn new() -> VariantDecoder {
        VariantDecoder::UserDefined(UserDefinedDecoder)
    }

    pub fn max_utf16_buffer_length(&self, byte_length: usize) -> Option<usize> {
        Some(byte_length)
    }

    pub fn max_utf8_buffer_length_without_replacement(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_mul(3)
    }

    pub fn max_utf8_buffer_length(&self, byte_length: usize) -> Option<usize> {
        byte_length.checked_mul(3)
    }

    decoder_function!(
        {},
        {},
        {},
        {
            unread_handle.commit();

            if b < 0x80 {
                // ASCII run not optimized, because binary data expected
                destination_handle.write_ascii(b);
                continue;
            }
            destination_handle.write_upper_bmp(u16::from(b) + 0xF700);
            continue;
        },
        self,
        src_consumed,
        dest,
        source,
        b,
        destination_handle,
        unread_handle,
        check_space_bmp,
        decode_to_utf8_raw,
        u8,
        Utf8Destination
    );

    #[cfg(not(all(feature = "simd-accel", target_endian = "little")))]
    pub fn decode_to_utf16_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let (pending, length) = if dst.len() < src.len() {
            (DecoderResult::OutputFull, dst.len())
        } else {
            (DecoderResult::InputEmpty, src.len())
        };
        let src_trim = &src[..length];
        let dst_trim = &mut dst[..length];
        src_trim
            .iter()
            .zip(dst_trim.iter_mut())
            .for_each(|(from, to)| {
                *to = {
                    let unit = *from;
                    if unit < 0x80 {
                        u16::from(unit)
                    } else {
                        u16::from(unit) + 0xF700
                    }
                }
            });
        (pending, length, length)
    }

    #[cfg(all(feature = "simd-accel", target_endian = "little"))]
    #[inline(always)]
    pub fn decode_to_utf16_raw(
        &mut self,
        src: &[u8],
        dst: &mut [u16],
        _last: bool,
    ) -> (DecoderResult, usize, usize) {
        let (pending, length) = if dst.len() < src.len() {
            (DecoderResult::OutputFull, dst.len())
        } else {
            (DecoderResult::InputEmpty, src.len())
        };
        let (src_strides, src_tail) = src[..length].as_chunks::<STRIDE>();
        let (dst_strides, dst_tail) = dst[..length].as_chunks_mut::<STRIDE>();
        for (src_stride, dst_stride) in src_strides.iter().zip(dst_strides.iter_mut()) {
            let src_simd: u8x16 = (*src_stride).into();
            let unpacked = simd_unpack(src_simd);
            let shifted = shift_upper(unpacked);
            *dst_stride = shifted.to_array();
        }
        src_tail
            .iter()
            .zip(dst_tail.iter_mut())
            .for_each(|(from, to)| {
                *to = {
                    let unit = *from;
                    if unit < 0x80 {
                        u16::from(unit)
                    } else {
                        u16::from(unit) + 0xF700
                    }
                }
            });
        (pending, length, length)
    }
}

pub struct UserDefinedEncoder;

impl UserDefinedEncoder {
    pub fn new(encoding: &'static Encoding) -> Encoder {
        Encoder::new(encoding, VariantEncoder::UserDefined(UserDefinedEncoder))
    }

    pub fn max_buffer_length_from_utf16_without_replacement(
        &self,
        u16_length: usize,
    ) -> Option<usize> {
        Some(u16_length)
    }

    pub fn max_buffer_length_from_utf8_without_replacement(
        &self,
        byte_length: usize,
    ) -> Option<usize> {
        Some(byte_length)
    }

    encoder_functions!(
        eof = {},
        body = {
            if c <= '\u{7F}' {
                // TODO optimize ASCII run
                destination_handle.write_one(c as u8);
                continue;
            }
            if !('\u{F780}'..='\u{F7FF}').contains(&c) {
                return (
                    EncoderResult::Unmappable(c),
                    unread_handle.consumed(),
                    destination_handle.written(),
                );
            }
            destination_handle.write_one((u32::from(c) - 0xF700) as u8);
            continue;
        },
        self = self,
        src_consumed = src_consumed,
        source = source,
        dest = dest,
        c = c,
        destination_handle = destination_handle,
        unread_handle = unread_handle,
        destination_check = check_space_one
    );
}

// Any copyright to the test code below this comment is dedicated to the
// Public Domain. http://creativecommons.org/publicdomain/zero/1.0/

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::super::testing::*;
    use super::super::*;

    fn decode_x_user_defined(bytes: &[u8], expect: &str) {
        decode(X_USER_DEFINED, bytes, expect);
    }

    fn encode_x_user_defined(string: &str, expect: &[u8]) {
        encode(X_USER_DEFINED, string, expect);
    }

    #[test]
    fn test_x_user_defined_decode() {
        // Empty
        decode_x_user_defined(b"", "");

        // ASCII
        decode_x_user_defined(b"\x61\x62", "\u{0061}\u{0062}");

        decode_x_user_defined(b"\x80\xFF", "\u{F780}\u{F7FF}");
        decode_x_user_defined(
            b"\x80\xFF\x61\x62\x80\xFF\x61\x62\x80\xFF\x61\x62\x80\xFF\x61\x62\x80\xFF\x61\x62",
            "\u{F780}\u{F7FF}\u{0061}\u{0062}\u{F780}\u{F7FF}\u{0061}\u{0062}\u{F780}\u{F7FF}\u{0061}\u{0062}\u{F780}\u{F7FF}\u{0061}\u{0062}\u{F780}\u{F7FF}\u{0061}\u{0062}",
        );
    }

    #[test]
    fn test_x_user_defined_encode() {
        // Empty
        encode_x_user_defined("", b"");

        // ASCII
        encode_x_user_defined("\u{0061}\u{0062}", b"\x61\x62");

        encode_x_user_defined("\u{F780}\u{F7FF}", b"\x80\xFF");
        encode_x_user_defined("\u{F77F}\u{F800}", b"&#63359;&#63488;");
    }

    #[test]
    fn test_x_user_defined_from_two_low_surrogates() {
        let expectation = b"&#65533;&#65533;";
        let mut output = [0u8; 40];
        let mut encoder = X_USER_DEFINED.new_encoder();
        let (result, read, written, had_errors) =
            encoder.encode_from_utf16(&[0xDC00u16, 0xDEDEu16], &mut output[..], true);
        assert_eq!(result, CoderResult::InputEmpty);
        assert_eq!(read, 2);
        assert_eq!(written, expectation.len());
        assert!(had_errors);
        assert_eq!(&output[..written], expectation);
    }
}
