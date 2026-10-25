//! Adaptors for various compression algorithms.

#![cfg_attr(docsrs, feature(doc_cfg))]

use std::io::Result;

pub use compression_core as core;

#[cfg(feature = "brotli")]
pub mod brotli;
#[cfg(feature = "bzip2")]
pub mod bzip2;
#[cfg(feature = "deflate")]
pub mod deflate;
#[cfg(feature = "deflate64")]
pub mod deflate64;
#[cfg(feature = "flate2")]
pub mod flate;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lz4")]
pub mod lz4;
#[cfg(feature = "lzma")]
pub mod lzma;
#[cfg(all(feature = "brotli-mbrotli", async_compression_unstable))]
pub mod mbrotli;
#[cfg(all(feature = "zstd-mzstd", async_compression_unstable))]
pub mod mzstd;
#[cfg(feature = "xz")]
pub mod xz;
#[cfg(feature = "lzma")]
pub mod xz2;
#[cfg(feature = "zlib")]
pub mod zlib;
#[cfg(feature = "zstd")]
pub mod zstd;

use compression_core::util::{PartialBuffer, WriteBuffer};

#[cfg(feature = "brotli")]
pub use self::brotli::{BrotliDecoder, BrotliEncoder};
#[cfg(feature = "bzip2")]
pub use self::bzip2::{BzDecoder, BzEncoder};
#[cfg(feature = "deflate")]
pub use self::deflate::{DeflateDecoder, DeflateEncoder};
#[cfg(feature = "deflate64")]
pub use self::deflate64::Deflate64Decoder;
#[cfg(feature = "flate2")]
pub use self::flate::{FlateDecoder, FlateEncoder};
#[cfg(feature = "gzip")]
pub use self::gzip::{GzipDecoder, GzipEncoder};
#[cfg(feature = "lz4")]
pub use self::lz4::{Lz4Decoder, Lz4Encoder};
#[cfg(feature = "lzma")]
pub use self::lzma::{LzmaDecoder, LzmaEncoder};
#[cfg(all(feature = "brotli-mbrotli", async_compression_unstable))]
pub use self::mbrotli::{MbrotliDecoder, MbrotliEncoder};
#[cfg(all(feature = "zstd-mzstd", async_compression_unstable))]
pub use self::mzstd::{MzstdDecoder, MzstdEncoder};
#[cfg(feature = "xz")]
pub use self::xz::{XzDecoder, XzEncoder};
#[cfg(feature = "lzma")]
pub use self::xz2::{Xz2Decoder, Xz2Encoder, Xz2FileFormat};
#[cfg(feature = "zlib")]
pub use self::zlib::{ZlibDecoder, ZlibEncoder};
#[cfg(feature = "zstd")]
pub use self::zstd::{ZstdDecoder, ZstdEncoder};

fn forward_output<R>(
    output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    f: impl FnOnce(&mut WriteBuffer<'_>) -> R,
) -> R {
    let written_len = output.written_len();

    let output_buffer = output.get_mut();
    let mut write_buffer = WriteBuffer::new_initialized(output_buffer.as_mut());
    write_buffer.advance(written_len);

    let result = f(&mut write_buffer);
    let new_written_len = write_buffer.written_len();
    output.advance(new_written_len - written_len);
    result
}

fn forward_input_output<R>(
    input: &mut PartialBuffer<impl AsRef<[u8]>>,
    output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    f: impl FnOnce(&mut PartialBuffer<&[u8]>, &mut WriteBuffer<'_>) -> R,
) -> R {
    let written_len = input.written_len();

    let input_buffer = input.get_mut();
    let mut partial_buffer = PartialBuffer::new(input_buffer.as_ref());
    partial_buffer.advance(written_len);

    let result = forward_output(output, |output| f(&mut partial_buffer, output));
    let new_written_len = partial_buffer.written_len();
    input.advance(new_written_len - written_len);
    result
}

pub trait Encode {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<impl AsRef<[u8]>>,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<()>;

    /// Returns whether the internal buffers are flushed
    fn flush(&mut self, output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>)
        -> Result<bool>;

    /// Returns whether the internal buffers are flushed and the end of the stream is written
    fn finish(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool>;
}
impl<T: EncodeV2 + ?Sized> Encode for T {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<impl AsRef<[u8]>>,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<()> {
        forward_input_output(input, output, |input, output| {
            EncodeV2::encode(self, input, output)
        })
    }

    fn flush(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool> {
        forward_output(output, |output| EncodeV2::flush(self, output))
    }

    fn finish(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool> {
        forward_output(output, |output| EncodeV2::finish(self, output))
    }
}

/// version 2 of [`Encode`] that is trait object safe.
///
/// The different from [`Encode`] is that:
///  - It doesn't have any generic in it, so it is trait object safe
///  - It uses [`WriteBuffer`] for output, which will support uninitialized buffer.
pub trait EncodeV2 {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<()>;

    /// Returns whether the internal buffers are flushed
    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool>;

    /// Returns whether the internal buffers are flushed and the end of the stream is written
    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool>;
}

pub trait Decode {
    /// Reinitializes this decoder ready to decode a new member/frame of data.
    fn reinit(&mut self) -> Result<()>;

    /// Returns whether the end of the stream has been read
    fn decode(
        &mut self,
        input: &mut PartialBuffer<impl AsRef<[u8]>>,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool>;

    /// Returns whether the internal buffers are flushed
    fn flush(&mut self, output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>)
        -> Result<bool>;

    /// Returns whether the internal buffers are flushed
    fn finish(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool>;
}

impl<T: DecodeV2 + ?Sized> Decode for T {
    fn reinit(&mut self) -> Result<()> {
        DecodeV2::reinit(self)
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<impl AsRef<[u8]>>,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool> {
        forward_input_output(input, output, |input, output| {
            DecodeV2::decode(self, input, output)
        })
    }

    fn flush(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool> {
        forward_output(output, |output| DecodeV2::flush(self, output))
    }

    fn finish(
        &mut self,
        output: &mut PartialBuffer<impl AsRef<[u8]> + AsMut<[u8]>>,
    ) -> Result<bool> {
        forward_output(output, |output| DecodeV2::finish(self, output))
    }
}

/// version 2 [`Decode`] that is trait object safe.
///
/// The different from [`Decode`] is that:
///  - It doesn't have any generic in it, so it is trait object safe
///  - It uses [`WriteBuffer`] for output, which will support uninitialized buffer.
pub trait DecodeV2 {
    /// Reinitializes this decoder ready to decode a new member/frame of data.
    fn reinit(&mut self) -> Result<()>;

    /// Returns whether the end of the stream has been read
    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool>;

    /// Returns whether the internal buffers are flushed
    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool>;

    /// Returns whether the internal buffers are flushed
    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool>;
}

pub trait DecodedSize {
    /// Returns the size of the input when uncompressed.
    fn decoded_size(input: &[u8]) -> Result<u64>;
}
