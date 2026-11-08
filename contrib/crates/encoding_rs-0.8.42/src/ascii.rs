// Copyright Mozilla Foundation. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `as` truncates, so works on 32-bit, too.
// Safety invariant: `data & ASCII_MASK == 0` for valid ASCII
#[allow(dead_code)]
pub const ASCII_MASK: usize = 0x8080_8080_8080_8080u64 as usize;

cfg_if! {
    if #[cfg(all(feature = "simd-accel", target_endian = "little"))] {
        pub(crate) use crate::simd_funcs::ascii_to_ascii_stride;
        pub(crate) use crate::simd_funcs::ascii_to_basic_latin_stride;
        pub(crate) use crate::simd_funcs::basic_latin_to_ascii_stride;
        #[allow(unused_imports)]
        pub(crate) use crate::simd_funcs::validate_ascii_stride;
        #[allow(unused_imports)]
        pub(crate) use crate::simd_funcs::ascii_to_ascii_double_stride;
        #[allow(unused_imports)]
        pub(crate) use crate::simd_funcs::ascii_to_basic_latin_double_stride;
        pub(crate) use crate::simd_funcs::basic_latin_to_ascii_double_stride;
        #[allow(unused_imports)]
        pub(crate) use crate::simd_funcs::validate_ascii_double_stride;
    } else {

        // These are `inline(never)`, because the autovectorizer can vectorize the
        // full-stride operations in isolation but not when combining the ASCII/Basic Latin
        // check and the pack/unpack!

        #[inline(never)]
        pub(crate) fn is_ascii(s: &[u8; STRIDE]) -> bool {
            s.iter().all(|b| *b < 0x80)
        }

        #[inline(never)]
        pub(crate) fn is_basic_latin(s: &[u16; STRIDE]) -> bool {
            s.iter().all(|b| *b < 0x80)
        }

        #[inline(never)]
        pub(crate) fn is_utf16_latin1(s: &[u16; STRIDE]) -> bool {
            s.iter().all(|b| *b < 0x100)
        }

        #[inline(never)]
        pub(crate) fn copy_stride(src_stride: &[u8; STRIDE], dst_stride: &mut [u8; STRIDE]) {
            *dst_stride = *src_stride;
        }

        #[inline(never)]
        pub(crate) fn unpack_stride(src_stride: &[u8; STRIDE], dst_stride: &mut [u16; STRIDE]) {
            src_stride.iter().zip(dst_stride.iter_mut()).for_each(|(s, d)| *d = *s as u16);
        }

        #[inline(never)]
        pub(crate) fn pack_stride(src_stride: &[u16; STRIDE], dst_stride: &mut [u8; STRIDE]) {
            src_stride.iter().zip(dst_stride.iter_mut()).for_each(|(s, d)| *d = *s as u8);
        }

        #[inline(never)]
        fn copy_stride_tail(src_stride: &[u8; 16], dst_stride: &mut [u8; 16]) -> (u8, usize) {
            for (i, (s, d)) in src_stride.iter().zip(dst_stride.iter_mut()).enumerate() {
                let c = *s;
                if c >= 0x80 {
                    return (c, i);
                }
                *d = c;
            }
            debug_assert!(false);
            (0, 0)
        }

        #[inline(never)]
        fn unpack_stride_tail(src_stride: &[u8; 16], dst_stride: &mut [u16; 16]) -> (u8, usize) {
            for (i, (s, d)) in src_stride.iter().zip(dst_stride.iter_mut()).enumerate() {
                let c = *s;
                if c >= 0x80 {
                    return (c, i);
                }
                *d = c as u16;
            }
            debug_assert!(false);
            (0, 0)
        }

        #[inline(never)]
        fn pack_stride_tail(src_stride: &[u16; 16], dst_stride: &mut [u8; 16]) -> (u16, usize) {
            for (i, (s, d)) in src_stride.iter().zip(dst_stride.iter_mut()).enumerate() {
                let c = *s;
                if c >= 0x80 {
                    return (c, i);
                }
                *d = c as u8;
            }
            debug_assert!(false);
            (0, 0)
        }

        #[allow(dead_code)]
        #[inline(never)]
        fn validate_ascii_stride_tail(stride: &[u8; 16]) -> (u8, usize) {
            for (i, s) in stride.iter().enumerate() {
                let b = *s;
                if b >= 0x80 {
                    return (b, i);
                }
            }
            debug_assert!(false);
            (0, 0)
        }

        #[allow(dead_code)]
        #[inline(never)]
        fn validate_basic_latin_stride_tail(stride: &[u16; 16]) -> usize {
            for (i, s) in stride.iter().enumerate() {
                if *s >= 0x80 {
                    return i;
                }
            }
            debug_assert!(false);
            0
        }

        #[inline(always)]
        fn ascii_to_ascii_stride(
            src_stride: &[u8; STRIDE],
            dst_stride: &mut [u8; STRIDE],
        ) -> Option<(u8, usize)> {
            if is_ascii(src_stride) {
                copy_stride(src_stride, dst_stride);
                return None;
            }
            Some(copy_stride_tail(src_stride, dst_stride))
        }

        #[inline(always)]
        fn ascii_to_basic_latin_stride(
            src_stride: &[u8; STRIDE],
            dst_stride: &mut [u16; STRIDE],
        ) -> Option<(u8, usize)> {
            if is_ascii(src_stride) {
                unpack_stride(src_stride, dst_stride);
                return None;
            }
            Some(unpack_stride_tail(src_stride, dst_stride))
        }

        #[inline(always)]
        fn basic_latin_to_ascii_stride(
            src_stride: &[u16; STRIDE],
            dst_stride: &mut [u8; STRIDE],
        ) -> Option<(u16, usize)> {
            if is_basic_latin(src_stride) {
                pack_stride(src_stride, dst_stride);
                return None;
            }
            Some(pack_stride_tail(src_stride, dst_stride))
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn validate_ascii_stride(
            stride: &[u8; STRIDE],
        ) -> Option<(u8, usize)> {
            if is_ascii(stride) {
                return None;
            }
            Some(validate_ascii_stride_tail(stride))
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn validate_basic_latin_stride(
            stride: &[u16; STRIDE],
        ) -> Option<usize> {
            if is_basic_latin(stride) {
                return None;
            }
            Some(validate_basic_latin_stride_tail(stride))
        }
    }
}

pub(crate) const STRIDE: usize = 16;

pub(crate) const MAX_STRIDE_SIZE: usize = STRIDE;

#[allow(unused_macros)]
macro_rules! ascii_copy_impl_double {
    ($name:ident, $stride:ident, $double_stride:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub(crate) fn $name(
            src: &[$src_unit],
            dst: &mut [$dst_unit],
        ) -> Option<($src_unit, usize)> {
            // Make both the same length here to have the chunks and tail match.
            let len = core::cmp::min(src.len(), dst.len());
            let mut consumed = 0usize;
            let (src_strides, src_tail) = src[..len].as_chunks::<STRIDE>();
            let (dst_strides, dst_tail) = dst[..len].as_chunks_mut::<STRIDE>();
            if let Some((src_first_stride, src_strides_tail)) = src_strides.split_first() {
                if let Some((dst_first_stride, dst_strides_tail)) = dst_strides.split_first_mut() {
                    if let Some(pos) = $stride(src_first_stride, dst_first_stride) {
                        return Some(pos);
                    }
                    consumed = STRIDE;

                    let (src_double_strides, src_single_stride) = src_strides_tail.as_chunks::<2>();
                    let (dst_double_strides, dst_single_stride) =
                        dst_strides_tail.as_chunks_mut::<2>();
                    for (src_double_stride, dst_double_stride) in
                        src_double_strides.iter().zip(dst_double_strides.iter_mut())
                    {
                        if let Some((c, pos)) = $double_stride(src_double_stride, dst_double_stride)
                        {
                            return Some((c, consumed + pos));
                        }
                        consumed += STRIDE * 2;
                    }
                    for (src_stride, dst_stride) in
                        src_single_stride.iter().zip(dst_single_stride.iter_mut())
                    {
                        if let Some((c, pos)) = $stride(src_stride, dst_stride) {
                            return Some((c, consumed + pos));
                        }
                        consumed += STRIDE;
                    }
                } else {
                    debug_assert!(false);
                }
            }
            for (src_slot, dst_slot) in src_tail.iter().zip(dst_tail.iter_mut()) {
                let c = *src_slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                *dst_slot = c as $dst_unit;
                consumed += 1;
            }
            None
        }
    };
}

#[allow(unused_macros)]
macro_rules! ascii_copy_impl_single {
    ($name:ident, $stride:ident, $double_stride:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub fn $name(src: &[$src_unit], dst: &mut [$dst_unit]) -> Option<($src_unit, usize)> {
            // Make both the same length here to have the chunks and tail match.
            let len = core::cmp::min(src.len(), dst.len());
            let mut consumed = 0usize;
            let (src_strides, src_tail) = src[..len].as_chunks::<STRIDE>();
            let (dst_strides, dst_tail) = dst[..len].as_chunks_mut::<STRIDE>();
            for (src_stride, dst_stride) in src_strides.iter().zip(dst_strides.iter_mut()) {
                if let Some((c, pos)) = $stride(src_stride, dst_stride) {
                    return Some((c, consumed + pos));
                }
                consumed += STRIDE;
            }
            for (src_slot, dst_slot) in src_tail.iter().zip(dst_tail.iter_mut()) {
                let c = *src_slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                *dst_slot = c as $dst_unit;
                consumed += 1;
            }
            None
        }
    };
}

cfg_if! {
    if #[cfg(target_arch = "arm")] {

        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            #[allow(clippy::never_loop)]
            'outer: loop {
                let (strides, tail) = bytes.as_chunks::<{core::mem::size_of::<usize>()}>();
                if let Some((first_stride, strides_tail)) = strides.split_first() {
                    let first_usize = usize::from_ne_bytes(*first_stride);
                    if first_usize & ASCII_MASK != 0 {
                        break 'outer;
                    }
                    consumed = core::mem::size_of::<usize>();

                    let (double_strides, single_stride) = strides_tail.as_chunks::<2>();
                    for double_stride in double_strides.iter() {
                        let first_usize = usize::from_ne_bytes(double_stride[0]);
                        let second_usize = usize::from_ne_bytes(double_stride[1]);
                        if (first_usize | second_usize) & ASCII_MASK != 0 {
                            break 'outer;
                        }
                        consumed += core::mem::size_of::<usize>() * 2;
                    }
                    for stride in single_stride.iter() {
                        let last_usize = usize::from_ne_bytes(*stride);
                        if last_usize & ASCII_MASK != 0 {
                            break 'outer;
                        }
                        consumed += core::mem::size_of::<usize>();
                    }
                }
                for slot in tail.iter() {
                    let c = *slot;
                    if c >= 0x80 {
                        return Some((c, consumed));
                    }
                    consumed += 1;
                }
                return None;
            }
            let tail = &bytes[consumed..];
            for slot in tail.iter() {
                let c = *slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                consumed += 1;
            }
            debug_assert!(false);
            None
        }

    } else if #[cfg(all(feature = "simd-accel", target_endian = "little"))] {

        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            let (strides, tail) = bytes.as_chunks::<STRIDE>();
            if let Some((first_stride, strides_tail)) = strides.split_first() {
                if let Some((c, pos)) = validate_ascii_stride(first_stride) {
                    return Some((c, pos));
                }
                consumed = STRIDE;

                let (double_strides, single_stride) = strides_tail.as_chunks::<2>();
                for double_stride in double_strides.iter() {
                    if let Some((c, pos)) = validate_ascii_double_stride(double_stride) {
                        return Some((c, consumed + pos));
                    }
                    consumed += STRIDE * 2;
                }
                for stride in single_stride.iter() {
                    if let Some((c, pos)) = validate_ascii_stride(stride) {
                        return Some((c, consumed + pos));
                    }
                    consumed += STRIDE;
                }
            }
            for slot in tail.iter() {
                let c = *slot;
                if c >= 0x80 {
                    return Some((c, consumed));
                }
                consumed += 1;
            }
            None
        }

    } else {

        #[inline(always)]
        fn ascii_valid_impl(bytes: &[u8]) -> Option<(u8, usize)> {
            let mut consumed = 0usize;
            let (strides, tail) = bytes.as_chunks::<STRIDE>();
            for stride in strides.iter() {
                if let Some((b, pos)) = validate_ascii_stride(stride) {
                    return Some((b, consumed + pos));
                }
                consumed += STRIDE;
            }
            for slot in tail.iter() {
                let b = *slot;
                if b >= 0x80 {
                    return Some((b, consumed));
                }
                consumed += 1;
            }
            None
        }

    }
}

cfg_if! {
    if #[cfg(all(feature = "simd-accel", target_endian = "little", not(target_arch = "arm")))] {
        ascii_copy_impl_double!(
            ascii_to_ascii_impl,
            ascii_to_ascii_stride,
            ascii_to_ascii_double_stride,
            u8,
            u8
        );
        ascii_copy_impl_double!(
            ascii_to_basic_latin_impl,
            ascii_to_basic_latin_stride,
            ascii_to_basic_latin_double_stride,
            u8,
            u16
        );
    } else {
        ascii_copy_impl_single!(
            ascii_to_ascii_impl,
            ascii_to_ascii_stride,
            ascii_to_ascii_double_stride,
            u8,
            u8
        );
        ascii_copy_impl_single!(
            ascii_to_basic_latin_impl,
            ascii_to_basic_latin_stride,
            ascii_to_basic_latin_double_stride,
            u8,
            u16
        );
    }
}

cfg_if! {
    if #[cfg(all(feature = "simd-accel", target_endian = "little"))] {
        ascii_copy_impl_double!(
            basic_latin_to_ascii_impl,
            basic_latin_to_ascii_stride,
            basic_latin_to_ascii_double_stride,
            u16,
            u8
        );
    } else {
        ascii_copy_impl_single!(
            basic_latin_to_ascii_impl,
            basic_latin_to_ascii_stride,
            basic_latin_to_ascii_double_stride,
            u16,
            u8
        );
    }
}

macro_rules! ascii_copy {
    ($name:ident, $impl:ident, $src_unit:ty, $dst_unit:ty) => {
        #[inline(always)]
        pub(crate) fn $name(
            src: &[$src_unit],
            dst: &mut [$dst_unit],
        ) -> Option<($src_unit, usize)> {
            $impl(src, dst)
        }
    };
}

ascii_copy!(ascii_to_ascii, ascii_to_ascii_impl, u8, u8);
ascii_copy!(ascii_to_basic_latin, ascii_to_basic_latin_impl, u8, u16);
ascii_copy!(basic_latin_to_ascii, basic_latin_to_ascii_impl, u16, u8);

#[inline(always)]
pub(crate) fn validate_ascii(bytes: &[u8]) -> Option<(u8, usize)> {
    ascii_valid_impl(bytes)
}

#[inline(always)]
pub(crate) fn ascii_valid_up_to(bytes: &[u8]) -> usize {
    ascii_valid_impl(bytes)
        .map(|(_, pos)| pos)
        .unwrap_or(bytes.len())
}

pub(crate) fn iso_2022_jp_ascii_valid_up_to(bytes: &[u8]) -> usize {
    for (i, b_ref) in bytes.iter().enumerate() {
        let b = *b_ref;
        if b >= 0x80 || b == 0x1B || b == 0x0E || b == 0x0F {
            return i;
        }
    }
    bytes.len()
}

// Any copyright to the test code below this comment is dedicated to the
// Public Domain. http://creativecommons.org/publicdomain/zero/1.0/

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    macro_rules! test_ascii {
        ($test_name:ident, $fn_tested:ident, $src_unit:ty, $dst_unit:ty) => {
            #[test]
            fn $test_name() {
                let mut src: Vec<$src_unit> = Vec::with_capacity(32);
                let mut dst: Vec<$dst_unit> = Vec::with_capacity(32);
                for i in 0..32 {
                    src.clear();
                    dst.clear();
                    dst.resize(32, 0);
                    for j in 0..32 {
                        let c = if i == j { 0xAA } else { j + 0x40 };
                        src.push(c as $src_unit);
                    }
                    match { $fn_tested(&src, &mut dst) } {
                        None => unreachable!("Should always find non-ASCII"),
                        Some((non_ascii, num_ascii)) => {
                            assert_eq!(non_ascii, 0xAA);
                            assert_eq!(num_ascii, i);
                            for j in 0..i {
                                assert_eq!(dst[j], (j + 0x40) as $dst_unit);
                            }
                        }
                    }
                }
            }
        };
    }

    test_ascii!(test_ascii_to_ascii, ascii_to_ascii, u8, u8);
    test_ascii!(test_ascii_to_basic_latin, ascii_to_basic_latin, u8, u16);
    test_ascii!(test_basic_latin_to_ascii, basic_latin_to_ascii, u16, u8);
}
