//! The interface and implementations of LZ77 compression algorithm.
//!
//! LZ77 is a compression algorithm used in [DEFLATE](https://tools.ietf.org/html/rfc1951).
#![warn(missing_docs)]
#![cfg_attr(no_std, feature = "no_std")]

#[cfg(feature = "no_std")]
extern crate alloc;

pub use self::default::{DefaultLz77Encoder, DefaultLz77EncoderBuilder};
#[cfg(feature = "no_std")]
use alloc::vec::Vec;
#[cfg(feature = "no_std")]
use core2::io;
use rle_decode_fast::rle_decode;
#[cfg(not(feature = "no_std"))]
use std::io;

mod default;

/// Maximum length of sharable bytes in a pointer.
pub const MAX_LENGTH: u16 = 258;

/// Maximum backward distance of a pointer.
pub const MAX_DISTANCE: u16 = 32_768;

/// Maximum size of a sliding window.
pub const MAX_WINDOW_SIZE: u16 = MAX_DISTANCE;

/// A LZ77 encoded data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Code {
    /// Literal byte.
    Literal(u8),

    /// Backward pointer to shared data.
    Pointer {
        /// Length of the shared data.
        /// The values must be limited to [`MAX_LENGTH`].
        length: u16,

        /// Distance between current position and start position of the shared data.
        /// The values must be limited to [`MAX_DISTANCE`].
        backward_distance: u16,
    },
}

/// Compression level.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// No compression.
    None,

    /// Best speed.
    Fast,

    /// Balanced between speed and size.
    Balance,

    /// Best compression.
    Best,
}

/// The [`Sink`] trait represents a consumer of LZ77 encoded data.
pub trait Sink {
    /// Consumes a LZ77 encoded `Code`.
    fn consume(&mut self, code: Code);
}
impl<'a, T> Sink for &'a mut T
where
    T: Sink,
{
    fn consume(&mut self, code: Code) {
        (*self).consume(code);
    }
}
impl<T> Sink for Vec<T>
where
    T: From<Code>,
{
    fn consume(&mut self, code: Code) {
        self.push(T::from(code));
    }
}

/// The [`Lz77Encode`] trait defines the interface of LZ77 encoding algorithm.
pub trait Lz77Encode {
    /// Encodes a buffer and writes result LZ77 codes to `sink`.
    fn encode<S>(&mut self, buf: &[u8], sink: S)
    where
        S: Sink;

    /// Flushes the encoder, ensuring that all intermediately buffered codes are consumed by `sink`.
    fn flush<S>(&mut self, sink: S)
    where
        S: Sink;

    /// Returns the compression level of the encoder.
    ///
    /// If the implementation is omitted, [`CompressionLevel::Balance`] will be returned.
    fn compression_level(&self) -> CompressionLevel {
        CompressionLevel::Balance
    }

    /// Returns the window size of the encoder.
    ///
    /// If the implementation is omitted, [`MAX_WINDOW_SIZE`] will be returned.
    fn window_size(&self) -> u16 {
        MAX_WINDOW_SIZE
    }
}

/// A no compression implementation of [`Lz77Encode`] trait.
#[derive(Debug, Default)]
pub struct NoCompressionLz77Encoder;
impl NoCompressionLz77Encoder {
    /// Makes a new encoder instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::deflate;
    /// use libflate::lz77::{Lz77Encode, NoCompressionLz77Encoder, CompressionLevel};
    ///
    /// let lz77 = NoCompressionLz77Encoder::new();
    /// assert_eq!(lz77.compression_level(), CompressionLevel::None);
    ///
    /// let options = deflate::EncodeOptions::with_lz77(lz77);
    /// let _deflate = deflate::Encoder::with_options(Vec::new(), options);
    /// ```
    pub fn new() -> Self {
        NoCompressionLz77Encoder
    }
}
impl Lz77Encode for NoCompressionLz77Encoder {
    fn encode<S>(&mut self, buf: &[u8], mut sink: S)
    where
        S: Sink,
    {
        for c in buf.iter().cloned().map(Code::Literal) {
            sink.consume(c);
        }
    }
    #[allow(unused_variables)]
    fn flush<S>(&mut self, sink: S)
    where
        S: Sink,
    {
    }
    fn compression_level(&self) -> CompressionLevel {
        CompressionLevel::None
    }
}

/// LZ77 decoder.
#[derive(Debug, Default)]
pub struct Lz77Decoder {
    buffer: Vec<u8>,
    offset: usize,
}

impl Lz77Decoder {
    /// Makes a new [`Lz77Decoder`] instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes a [`Code`].
    ///
    /// The decoded bytes are appended to the buffer of [`Lz77Decoder`].
    #[inline]
    pub fn decode(&mut self, code: Code) -> io::Result<()> {
        match code {
            Code::Literal(b) => {
                self.buffer.push(b);
            }
            Code::Pointer {
                length,
                backward_distance,
            } => {
                if self.buffer.len() < backward_distance as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        #[cfg(not(feature = "no_std"))]
                        format!(
                            "Too long backword reference: buffer.len={}, distance={}",
                            self.buffer.len(),
                            backward_distance
                        ),
                        #[cfg(feature = "no_std")]
                        "Too long backword reference",
                    ));
                }
                rle_decode(
                    &mut self.buffer,
                    usize::from(backward_distance),
                    usize::from(length),
                );
            }
        }
        Ok(())
    }

    /// Appends the bytes read from `reader` to the buffer of [`Lz77Decoder`].
    pub fn extend_from_reader<R: io::Read>(&mut self, mut reader: R) -> io::Result<usize> {
        reader.read_to_end(&mut self.buffer)
    }

    /// Appends the given bytes to the buffer of [`Lz77Decoder`].
    pub fn extend_from_slice(&mut self, buf: &[u8]) {
        self.buffer.extend_from_slice(buf);
        self.offset += buf.len();
    }

    /// Clears the buffer of [`Lz77Decoder`].
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.offset = 0;
    }

    /// Returns the buffer of [`Lz77Decoder`].
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer[self.offset..]
    }

    fn truncate_old_buffer(&mut self) {
        if self.buffer().is_empty() && self.buffer.len() > MAX_DISTANCE as usize * 4 {
            let old_len = self.buffer.len();
            let new_len = MAX_DISTANCE as usize;
            {
                // isolation to please borrow checker
                let (dst, src) = self.buffer.split_at_mut(old_len - new_len);
                dst[..new_len].copy_from_slice(src);
            }
            self.buffer.truncate(new_len);
            self.offset = new_len;
        }
    }
}

impl io::Read for Lz77Decoder {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let copy_size = std::cmp::min(buf.len(), self.buffer.len() - self.offset);
        buf[..copy_size].copy_from_slice(&self.buffer[self.offset..][..copy_size]);
        self.offset += copy_size;
        self.truncate_old_buffer();
        Ok(copy_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "no_std")]
    use alloc::vec::Vec;
    #[cfg(feature = "no_std")]
    use core2::io::Read as _;
    #[cfg(not(feature = "no_std"))]
    use std::io::Read as _;

    #[test]
    fn encoder_and_decoder_works() {
        let mut codes = Vec::new();
        let mut encoder = DefaultLz77Encoder::new();
        encoder.encode(b"hello world!", &mut codes);
        encoder.flush(&mut codes);
        assert!(!codes.is_empty());

        let mut decoder = Lz77Decoder::new();
        for code in codes {
            decoder.decode(code).unwrap();
        }
        assert_eq!(decoder.buffer(), b"hello world!");

        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, b"hello world!");
        assert!(decoder.buffer().is_empty());
    }
}
