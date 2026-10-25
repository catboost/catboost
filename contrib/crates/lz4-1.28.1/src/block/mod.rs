//! This module provides access to the block mode functions of the lz4 C library.
//! It somehow resembles the [Python-lz4](http://python-lz4.readthedocs.io/en/stable/lz4.block.html) api,
//! but using Rust's Option type, the function parameters have been a little simplified.
//! As does python-lz4, this module supports prepending the compressed buffer with a u32 value
//! representing the size of the original, uncompressed data.
//!
//! # Examples
//! ```
//!
//! use lz4::block::{compress,decompress};
//!
//! let v = vec![0u8; 1024];
//!
//! let comp_with_prefix = compress(&v, None, true).unwrap();
//! let comp_wo_prefix = compress(&v, None, false).unwrap();
//!
//! assert_eq!(v, decompress(&comp_with_prefix, None).unwrap());
//! assert_eq!(v, decompress(&comp_wo_prefix, Some(1024)).unwrap());
//! ```

use super::c_char;
use super::liblz4::*;
use std::io::{Error, ErrorKind, Result};

/// Represents the compression mode do be used.
#[derive(Clone, Copy, Debug)]
pub enum CompressionMode {
    /// High compression with compression parameter
    HIGHCOMPRESSION(i32),
    /// Fast compression with acceleration paramet
    FAST(i32),
    /// Default compression
    DEFAULT,
}

impl Default for CompressionMode {
    fn default() -> Self {
        CompressionMode::DEFAULT
    }
}

/// Returns the size of the buffer that is guaranteed to hold the result of
/// compressing `uncompressed_size` bytes of in data. Returns std::io::Error
/// with ErrorKind::InvalidInput if input data is too long to be compressed by lz4.
pub fn compress_bound(uncompressed_size: usize) -> Result<usize> {
    // 0 iff src too large
    let compress_bound: i32 = unsafe { LZ4_compressBound(uncompressed_size as i32) };

    if uncompressed_size > (i32::max_value() as usize) || compress_bound <= 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Compression input too long.",
        ));
    }

    Ok(compress_bound as usize)
}

/// Compresses the full src buffer using the specified CompressionMode, where None and Some(Default)
/// are treated equally. If prepend_size is set, the source length will be prepended to the output
/// buffer.
///
///
/// # Errors
/// Returns std::io::Error with ErrorKind::InvalidInput if the src buffer is too long.
/// Returns std::io::Error with ErrorKind::Other if the compression failed inside the C library. If
/// this happens, the C api was not able to provide more information about the cause.
pub fn compress(src: &[u8], mode: Option<CompressionMode>, prepend_size: bool) -> Result<Vec<u8>> {
    // 0 iff src too large
    let compress_bound: i32 = unsafe { LZ4_compressBound(src.len() as i32) };

    if src.len() > (i32::max_value() as usize) || compress_bound <= 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Compression input too long.",
        ));
    }

    let mut compressed: Vec<u8> = vec![
        0;
        (if prepend_size {
            compress_bound + 4
        } else {
            compress_bound
        }) as usize
    ];

    let dec_size = compress_to_buffer(src, mode, prepend_size, &mut compressed)?;
    compressed.truncate(dec_size as usize);
    Ok(compressed)
}

/// Compresses the full `src` buffer using the specified CompressionMode, where None and Some(Default)
/// are treated equally, writing compressed bytes to `buffer`.
///
/// # Errors
/// Returns std::io::Error with ErrorKind::InvalidInput if the src buffer is too long.
/// The buffer cannot be larger than `i32::MAX`.
/// Returns std::io::Error with ErrorKind::Other if the compression data does not fit in `buffer`.
pub fn compress_to_buffer(
    src: &[u8],
    mode: Option<CompressionMode>,
    prepend_size: bool,
    buffer: &mut [u8],
) -> Result<usize> {
    // check that src isn't too big for lz4
    let max_len: i32 = unsafe { LZ4_compressBound(src.len() as i32) };

    if src.len() > (i32::max_value() as usize) || max_len <= 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Compression input too long.",
        ));
    }

    let dec_size;
    {
        let dst_buf = if prepend_size {
            let size = src.len() as u32;
            buffer[0] = size as u8;
            buffer[1] = (size >> 8) as u8;
            buffer[2] = (size >> 16) as u8;
            buffer[3] = (size >> 24) as u8;
            &mut buffer[4..]
        } else {
            buffer
        };

        let buf_len = dst_buf.len() as i32;

        dec_size = match mode {
            Some(CompressionMode::HIGHCOMPRESSION(level)) => unsafe {
                LZ4_compress_HC(
                    src.as_ptr() as *const c_char,
                    dst_buf.as_mut_ptr() as *mut c_char,
                    src.len() as i32,
                    buf_len,
                    level,
                )
            },
            Some(CompressionMode::FAST(accel)) => unsafe {
                LZ4_compress_fast(
                    src.as_ptr() as *const c_char,
                    dst_buf.as_mut_ptr() as *mut c_char,
                    src.len() as i32,
                    buf_len,
                    accel,
                )
            },
            _ => unsafe {
                LZ4_compress_default(
                    src.as_ptr() as *const c_char,
                    dst_buf.as_mut_ptr() as *mut c_char,
                    src.len() as i32,
                    buf_len,
                )
            },
        };
    }
    if dec_size <= 0 {
        return Err(Error::new(ErrorKind::Other, "Compression failed"));
    }

    let written_size = if prepend_size { dec_size + 4 } else { dec_size };

    Ok(written_size as usize)
}

fn get_decompressed_size(src: &[u8], uncompressed_size: Option<i32>) -> Result<usize> {
    let size;

    if let Some(s) = uncompressed_size {
        size = s;
    } else {
        if src.len() < 4 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Source buffer must at least contain size prefix.",
            ));
        }
        size =
            (src[0] as i32) | (src[1] as i32) << 8 | (src[2] as i32) << 16 | (src[3] as i32) << 24;
    }

    if size < 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            if uncompressed_size.is_some() {
                "Size parameter must not be negative."
            } else {
                "Parsed size prefix in buffer must not be negative."
            },
        ));
    }

    if unsafe { LZ4_compressBound(size) } <= 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Given size parameter is too big",
        ));
    }

    Ok(size as usize)
}

/// Decompresses the src buffer. If uncompressed_size is None, the source length will be read from
/// the start of the input buffer.
///
///
/// # Errors
/// Returns std::io::Error with ErrorKind::InvalidInput if the src buffer is too short, the
/// provided (or parsed) uncompressed_size is to large or negative.
/// Returns std::io::Error with ErrorKind::InvalidData if the decompression failed inside the C
/// library. This is most likely due to malformed input.
///
pub fn decompress(src: &[u8], uncompressed_size: Option<i32>) -> Result<Vec<u8>> {
    let size = get_decompressed_size(src, uncompressed_size)?;

    let mut buffer = vec![0u8; size];

    let sz = decompress_to_buffer(src, uncompressed_size, &mut buffer)?;
    buffer.truncate(sz);
    Ok(buffer)
}

pub fn decompress_to_buffer(
    mut src: &[u8],
    uncompressed_size: Option<i32>,
    buffer: &mut [u8],
) -> Result<usize> {
    let size;

    if let Some(s) = uncompressed_size {
        size = s;
    } else {
        if src.len() < 4 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Source buffer must at least contain size prefix.",
            ));
        }
        size =
            (src[0] as i32) | (src[1] as i32) << 8 | (src[2] as i32) << 16 | (src[3] as i32) << 24;

        src = &src[4..];
    }

    if size < 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            if uncompressed_size.is_some() {
                "Size parameter must not be negative."
            } else {
                "Parsed size prefix in buffer must not be negative."
            },
        ));
    }

    if unsafe { LZ4_compressBound(size) } <= 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Given size parameter is too big",
        ));
    }

    if size as usize > buffer.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "buffer isn't large enough to hold decompressed data",
        ));
    }

    let dec_bytes = unsafe {
        LZ4_decompress_safe(
            src.as_ptr() as *const c_char,
            buffer.as_mut_ptr() as *mut c_char,
            src.len() as i32,
            size,
        )
    };

    if dec_bytes < 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Decompression failed. Input invalid or too long?",
        ));
    }

    Ok(dec_bytes as usize)
}

#[cfg(test)]
mod test {
    use crate::block::{compress, decompress, decompress_to_buffer, CompressionMode};

    use super::compress_to_buffer;

    /// This test will fail unless the buffer created by decompress is correctly truncated
    #[test]
    fn decompress_truncate_test() {
        let src = "111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111".as_bytes();
        let rs_compressed = compress(src, None, false).unwrap();
        let rs_compressed_rs_uncompressed =
            decompress(&rs_compressed, Some((src.len() as i32) * 256)).unwrap();

        // compare the uncompressed result from rust
        assert_eq!(rs_compressed_rs_uncompressed, src,);
    }

    #[test]
    fn test_compression_without_prefix() {
        let size = 65536;
        let mut to_compress = Vec::with_capacity(size);
        for i in 0..size {
            to_compress.push(i as u8);
        }
        let mut v: Vec<Vec<u8>> = vec![];
        for i in 1..100 {
            v.push(compress(&to_compress, Some(CompressionMode::FAST(i)), false).unwrap());
        }

        // 12 is max high compression parameter
        for i in 1..12 {
            v.push(
                compress(
                    &to_compress,
                    Some(CompressionMode::HIGHCOMPRESSION(i)),
                    false,
                )
                .unwrap(),
            );
        }

        v.push(compress(&to_compress, None, false).unwrap());

        for val in v {
            assert_eq!(
                decompress(&val, Some(to_compress.len() as i32)).unwrap(),
                to_compress
            );
        }
    }

    #[test]
    fn test_compression_with_prefix() {
        let size = 65536;
        let mut to_compress = Vec::with_capacity(size);
        for i in 0..size {
            to_compress.push(i as u8);
        }
        let mut v: Vec<Vec<u8>> = vec![];
        for i in 1..100 {
            v.push(compress(&to_compress, Some(CompressionMode::FAST(i)), true).unwrap());
        }

        // 12 is max high compression parameter
        for i in 1..12 {
            v.push(
                compress(
                    &to_compress,
                    Some(CompressionMode::HIGHCOMPRESSION(i)),
                    true,
                )
                .unwrap(),
            );
        }

        v.push(compress(&to_compress, None, true).unwrap());

        for val in v {
            assert_eq!(decompress(&val, None).unwrap(), to_compress);
        }
    }

    #[test]
    fn test_compress_to_buffer() {
        let data = b"qfn3489fqASFqvegrwe$%344thI,..kmTMN3 g{P}wefwf2fv2443ef3443RT[]rete$7-80956GRWbefvw@fVGrwGB24tggrm%&*I@!";

        // test what happens when not enough space is available.
        let mut small_buf = vec![0; 5];
        let r = compress_to_buffer(&data[..], None, true, &mut small_buf);
        assert!(r.is_err());

        let mut big_buf = vec![0; 1000];
        let r = compress_to_buffer(&data[..], None, true, &mut big_buf).unwrap();
        assert_eq!(big_buf[r], 0);

        let mut dec_buf = vec![0u8; data.len() + 1];
        let dec_bytes = decompress_to_buffer(&big_buf[..r], None, &mut dec_buf).unwrap();
        assert_eq!(&dec_buf[..dec_bytes], &data[..]);
    }
    #[test]
    fn test_decompression_with_prefix() {
        let compressed: [u8; 250] = [
            0, 188, 0, 0, 255, 32, 116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115,
            116, 32, 115, 116, 114, 105, 110, 103, 32, 99, 111, 109, 112, 114, 101, 115, 115, 101,
            100, 32, 98, 121, 32, 112, 121, 116, 104, 111, 110, 45, 108, 122, 52, 32, 47, 0, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            117, 80, 45, 108, 122, 52, 32,
        ];

        let mut reference: String = String::new();
        for _ in 0..1024 {
            reference += "this is a test string compressed by python-lz4 ";
        }

        assert_eq!(decompress(&compressed, None).unwrap(), reference.as_bytes())
    }

    #[test]
    fn test_empty_compress() {
        use crate::block::{compress, decompress};
        let v = vec![0u8; 0];
        let comp_with_prefix = compress(&v, None, true).unwrap();
        dbg!(&comp_with_prefix);
        assert_eq!(v, decompress(&comp_with_prefix, None).unwrap());
    }
}
