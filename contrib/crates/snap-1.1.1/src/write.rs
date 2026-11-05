/*!
This module provides a `std::io::Write` implementation:

- `write::FrameEncoder` wraps another `std::io::Write` implemenation, and
  compresses data encoded using the Snappy frame format. Use this if you have
  uncompressed data source and wish to write it as compressed data.

It would also be possible to provide a `write::FrameDecoder`, which decompresses
data as it writes it, but it hasn't been implemented yet.
*/

use std::fmt;
use std::io::{self, Write};

use crate::compress::Encoder;
use crate::crc32::CheckSummer;
pub use crate::error::IntoInnerError;
use crate::frame::{
    compress_frame, CHUNK_HEADER_AND_CRC_SIZE, MAX_COMPRESS_BLOCK_SIZE,
    STREAM_IDENTIFIER,
};
use crate::MAX_BLOCK_SIZE;

/// A writer for compressing a Snappy stream.
///
/// This `FrameEncoder` wraps any other writer that implements `io::Write`.
/// Bytes written to this writer are compressed using the [Snappy frame
/// format](https://github.com/google/snappy/blob/master/framing_format.txt)
/// (file extension `sz`, MIME type `application/x-snappy-framed`).
///
/// Writes are buffered automatically, so there's no need to wrap the given
/// writer in a `std::io::BufWriter`.
///
/// The writer will be flushed automatically when it is dropped. If an error
/// occurs, it is ignored.
pub struct FrameEncoder<W: io::Write> {
    /// Our main internal state, split out for borrowck reasons (happily paid).
    ///
    /// Also, it's an `Option` so we can move out of it even though
    /// `FrameEncoder` impls `Drop`.
    inner: Option<Inner<W>>,
    /// Our buffer of uncompressed bytes. This isn't part of `inner` because
    /// we may write bytes directly from the caller if the given buffer was
    /// big enough. As a result, the main `write` implementation needs to
    /// accept either the internal buffer or the caller's bytes directly. Since
    /// `write` requires a mutable borrow, we satisfy the borrow checker by
    /// separating `src` from the rest of the state.
    src: Vec<u8>,
}

struct Inner<W> {
    /// The underlying writer.
    w: W,
    /// An encoder that we reuse that does the actual block based compression.
    enc: Encoder,
    /// A CRC32 checksummer that is configured to either use the portable
    /// fallback version or the SSE4.2 accelerated version when the right CPU
    /// features are available.
    checksummer: CheckSummer,
    /// The compressed bytes buffer. Bytes are compressed from src (usually)
    /// to dst before being written to w.
    dst: Vec<u8>,
    /// When false, the stream identifier (with magic bytes) must precede the
    /// next write.
    wrote_stream_ident: bool,
    /// Space for writing the header of a chunk before writing it to the
    /// underlying writer.
    chunk_header: [u8; 8],
}

impl<W: io::Write> FrameEncoder<W> {
    /// Create a new writer for streaming Snappy compression.
    pub fn new(wtr: W) -> FrameEncoder<W> {
        FrameEncoder {
            inner: Some(Inner {
                w: wtr,
                enc: Encoder::new(),
                checksummer: CheckSummer::new(),
                dst: vec![0; MAX_COMPRESS_BLOCK_SIZE],
                wrote_stream_ident: false,
                chunk_header: [0; CHUNK_HEADER_AND_CRC_SIZE],
            }),
            src: Vec::with_capacity(MAX_BLOCK_SIZE),
        }
    }

    /// Returns the underlying stream, consuming and flushing this writer.
    ///
    /// If flushing the writer caused an error, then an `IntoInnerError` is
    /// returned, which contains both the writer and the original writer.
    pub fn into_inner(mut self) -> Result<W, IntoInnerError<FrameEncoder<W>>> {
        match self.flush() {
            Ok(()) => Ok(self.inner.take().unwrap().w),
            Err(err) => Err(IntoInnerError::new(self, err)),
        }
    }

    /// Gets a reference to the underlying writer in this encoder.
    pub fn get_ref(&self) -> &W {
        &self.inner.as_ref().unwrap().w
    }

    /// Gets a reference to the underlying writer in this encoder.
    ///
    /// Note that mutating the output/input state of the stream may corrupt
    /// this encoder, so care must be taken when using this method.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner.as_mut().unwrap().w
    }
}

impl<W: io::Write> Drop for FrameEncoder<W> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            // Ignore errors because we can't conceivably return an error and
            // panicing in a dtor is bad juju.
            let _ = self.flush();
        }
    }
}

impl<W: io::Write> io::Write for FrameEncoder<W> {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let mut total = 0;
        // If there isn't enough room to add buf to src, then add only a piece
        // of it, flush it and mush on.
        loop {
            let free = self.src.capacity() - self.src.len();
            // n is the number of bytes extracted from buf.
            let n = if buf.len() <= free {
                break;
            } else if self.src.is_empty() {
                // If buf is bigger than our entire buffer then avoid
                // the indirection and write the buffer directly.
                self.inner.as_mut().unwrap().write(buf)?
            } else {
                self.src.extend_from_slice(&buf[0..free]);
                self.flush()?;
                free
            };
            buf = &buf[n..];
            total += n;
        }
        // We're only here if buf.len() will fit within the available space of
        // self.src.
        debug_assert!(buf.len() <= (self.src.capacity() - self.src.len()));
        self.src.extend_from_slice(buf);
        total += buf.len();
        // We should never expand or contract self.src.
        debug_assert!(self.src.capacity() == MAX_BLOCK_SIZE);
        Ok(total)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.src.is_empty() {
            return Ok(());
        }
        self.inner.as_mut().unwrap().write(&self.src)?;
        self.src.truncate(0);
        Ok(())
    }
}

impl<W: io::Write> Inner<W> {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let mut total = 0;
        if !self.wrote_stream_ident {
            self.wrote_stream_ident = true;
            self.w.write_all(STREAM_IDENTIFIER)?;
        }
        while !buf.is_empty() {
            // Advance buf and get our block.
            let mut src = buf;
            if src.len() > MAX_BLOCK_SIZE {
                src = &src[0..MAX_BLOCK_SIZE];
            }
            buf = &buf[src.len()..];

            let frame_data = compress_frame(
                &mut self.enc,
                self.checksummer,
                src,
                &mut self.chunk_header,
                &mut self.dst,
                false,
            )?;
            self.w.write_all(&self.chunk_header)?;
            self.w.write_all(frame_data)?;
            total += src.len();
        }
        Ok(total)
    }
}

impl<W: fmt::Debug + io::Write> fmt::Debug for FrameEncoder<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FrameEncoder")
            .field("inner", &self.inner)
            .field("src", &"[...]")
            .finish()
    }
}

impl<W: fmt::Debug + io::Write> fmt::Debug for Inner<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Inner")
            .field("w", &self.w)
            .field("enc", &self.enc)
            .field("checksummer", &self.checksummer)
            .field("dst", &"[...]")
            .field("wrote_stream_ident", &self.wrote_stream_ident)
            .field("chunk_header", &self.chunk_header)
            .finish()
    }
}
