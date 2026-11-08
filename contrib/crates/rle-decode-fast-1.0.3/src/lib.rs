//! # rle-decode-helper
//!
//! **THE** fastest way to implement any kind of decoding for **R**un **L**ength **E**ncoded data in Rust.
//!
//! Writing a fast decoder that is also safe can be quite challenging, so this crate is here to save you the
//! hassle of maintaining and testing your own implementation.
//!
//! # Usage
//!
//! ```rust
//! let mut decode_buffer = vec![0, 0, 1, 1, 0, 2, 3];
//! let lookbehind_length = 4;
//! let output_length = 10;
//! rle_decode_fast::rle_decode(&mut decode_buffer, lookbehind_length, output_length);
//! assert_eq!(decode_buffer, [0, 0, 1, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0]);
//! ```

#![no_std]

use core::ptr;
use core::ops;
extern crate alloc;
use alloc::vec::Vec;

/// Fast decoding of run length encoded data
///
/// Takes the last `lookbehind_length` items of the buffer and repeatedly appends them until
/// `fill_length` items have been copied.
///
/// # Panics
/// * `lookbehind_length` is 0
/// * `lookbehind_length` >= `buffer.len()`
/// * `fill_length + buffer.len()` would overflow
#[inline(always)]
pub fn rle_decode<T>(
    buffer: &mut Vec<T>,
    mut lookbehind_length: usize,
    mut fill_length: usize,
) where T: Copy {
    if lookbehind_length == 0 {
        lookbehind_length_fail();
    }

    let copy_fragment_start = buffer.len()
        .checked_sub(lookbehind_length)
        .expect("attempt to repeat fragment larger than buffer size");

    // Reserve space for *all* copies
    buffer.reserve(fill_length);

    while fill_length >= lookbehind_length {{}
        append_from_within(
            buffer,
            copy_fragment_start..(copy_fragment_start + lookbehind_length),
        );
        fill_length -= lookbehind_length;
        lookbehind_length *= 2;
    }

    // Copy the last remaining bytes
    append_from_within(
        buffer,
        copy_fragment_start..(copy_fragment_start + fill_length),
    );
}


/// Copy of `vec::append_from_within()` proposed for inclusion in stdlib,
/// see https://github.com/rust-lang/rfcs/pull/2714
/// Heavily based on the implementation of `slice::copy_within()`,
/// so we're pretty sure the implementation is sound
///
/// Note that the generic bounds were replaced by an explicit a..b range.
/// This is so that we can compile this on older toolchains (< 1.28).
#[inline(always)]
fn append_from_within<T>(seif: &mut Vec<T>, src: ops::Range<usize>) where T: Copy, {
    assert!(src.start <= src.end, "src end is before src start");
    assert!(src.end <= seif.len(), "src is out of bounds");
    let count = src.end - src.start;
    seif.reserve(count);
    let vec_len = seif.len();
    unsafe {
        // This is safe because reserve() above succeeded,
        // so `seif.len() + count` did not overflow usize
        // Derive both `src_ptr` and `dest_ptr` from the same loan
        let ptr = seif.as_mut_ptr();
        let src_ptr = ptr.add(src.start);
        let dest_ptr = ptr.add(vec_len);
        ptr::copy_nonoverlapping(src_ptr, dest_ptr, count);
        seif.set_len(vec_len + count);
    }
}

#[inline(never)]
#[cold]
fn lookbehind_length_fail() -> ! {
    panic!("attempt to repeat fragment of size 0");
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_basic() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 3, 10);
        assert_eq!(buf, &[1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3]);
    }

    #[test]
    fn test_zero_repeat() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 3, 0);
        assert_eq!(buf, &[1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic]
    fn test_zero_fragment() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 0, 10);
    }

    #[test]
    #[should_panic]
    fn test_zero_fragment_and_repeat() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 0, 0);
    }

    #[test]
    #[should_panic]
    fn test_overflow_fragment() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 10, 10);
    }

    #[test]
    #[should_panic]
    fn test_overflow_buf_size() {
        let mut buf = vec![1, 2, 3, 4, 5];
        rle_decode(&mut buf, 4, usize::max_value());
    }
}
