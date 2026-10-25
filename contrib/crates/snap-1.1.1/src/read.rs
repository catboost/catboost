/*!
This module provides two `std::io::Read` implementations:

* [`read::FrameDecoder`](struct.FrameDecoder.html)
  wraps another `std::io::Read` implemenation, and decompresses data encoded
  using the Snappy frame format. Use this if you have a compressed data source
  and wish to read it as uncompressed data.
* [`read::FrameEncoder`](struct.FrameEncoder.html)
  wraps another `std::io::Read` implemenation, and compresses data encoded
  using the Snappy frame format. Use this if you have uncompressed data source
  and wish to read it as compressed data.

Typically, `read::FrameDecoder` is the version that you'll want.
*/

use std::cmp;
use std::fmt;
use std::io;

use crate::bytes;
use crate::compress::Encoder;
use crate::crc32::CheckSummer;
use crate::decompress::{decompress_len, Decoder};
use crate::error::Error;
use crate::frame::{
    compress_frame, ChunkType, CHUNK_HEADER_AND_CRC_SIZE,
    MAX_COMPRESS_BLOCK_SIZE, STREAM_BODY, STREAM_IDENTIFIER,
};
use crate::MAX_BLOCK_SIZE;

/// The maximum size of a compressed block, including the header and stream
/// identifier, that can be emitted by FrameEncoder.
const MAX_READ_FRAME_ENCODER_BLOCK_SIZE: usize = STREAM_IDENTIFIER.len()
    + CHUNK_HEADER_AND_CRC_SIZE
    + MAX_COMPRESS_BLOCK_SIZE;

/// A reader for decompressing a Snappy stream.
///
/// This `FrameDecoder` wraps any other reader that implements `std::io::Read`.
/// Bytes read from this reader are decompressed using the
/// [Snappy frame format](https://github.com/google/snappy/blob/master/framing_format.txt)
/// (file extension `sz`, MIME type `application/x-snappy-framed`).
///
/// This reader can potentially make many small reads from the underlying
/// stream depending on its format, therefore, passing in a buffered reader
/// may be beneficial.
pub struct FrameDecoder<R: io::Read> {
    /// The underlying reader.
    r: R,
    /// A Snappy decoder that we reuse that does the actual block based
    /// decompression.
    dec: Decoder,
    /// A CRC32 checksummer that is configured to either use the portable
    /// fallback version or the SSE4.2 accelerated version when the right CPU
    /// features are available.
    checksummer: CheckSummer,
    /// The compressed bytes buffer, taken from the underlying reader.
    src: Vec<u8>,
    /// The decompressed bytes buffer. Bytes are decompressed from src to dst
    /// before being passed back to the caller.
    dst: Vec<u8>,
    /// Index into dst: starting point of bytes not yet given back to caller.
    dsts: usize,
    /// Index into dst: ending point of bytes not yet given back to caller.
    dste: usize,
    /// Whether we've read the special stream header or not.
    read_stream_ident: bool,
}

impl<R: io::Read> FrameDecoder<R> {
    /// Create a new reader for streaming Snappy decompression.
    pub fn new(rdr: R) -> FrameDecoder<R> {
        FrameDecoder {
            r: rdr,
            dec: Decoder::new(),
            checksummer: CheckSummer::new(),
            src: vec![0; MAX_COMPRESS_BLOCK_SIZE],
            dst: vec![0; MAX_BLOCK_SIZE],
            dsts: 0,
            dste: 0,
            read_stream_ident: false,
        }
    }

    /// Gets a reference to the underlying reader in this decoder.
    pub fn get_ref(&self) -> &R {
        &self.r
    }

    /// Gets a mutable reference to the underlying reader in this decoder.
    ///
    /// Note that mutation of the stream may result in surprising results if
    /// this decoder is continued to be used.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.r
    }

    /// Gets the underlying reader of this decoder.
    pub fn into_inner(self) -> R {
        self.r
    }
}

impl<R: io::Read> io::Read for FrameDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        macro_rules! fail {
            ($err:expr) => {
                return Err(io::Error::from($err))
            };
        }
        loop {
            if self.dsts < self.dste {
                let len = cmp::min(self.dste - self.dsts, buf.len());
                let dste = self.dsts.checked_add(len).unwrap();
                buf[0..len].copy_from_slice(&self.dst[self.dsts..dste]);
                self.dsts = dste;
                return Ok(len);
            }
            if !read_exact_eof(&mut self.r, &mut self.src[0..4])? {
                return Ok(0);
            }
            let ty = ChunkType::from_u8(self.src[0]);
            if !self.read_stream_ident {
                if ty != Ok(ChunkType::Stream) {
                    fail!(Error::StreamHeader { byte: self.src[0] });
                }
                self.read_stream_ident = true;
            }
            let len64 = bytes::read_u24_le(&self.src[1..]) as u64;
            if len64 > self.src.len() as u64 {
                fail!(Error::UnsupportedChunkLength {
                    len: len64,
                    header: false,
                });
            }
            let len = len64 as usize;
            match ty {
                Err(b) if 0x02 <= b && b <= 0x7F => {
                    // Spec says that chunk types 0x02-0x7F are reserved and
                    // conformant decoders must return an error.
                    fail!(Error::UnsupportedChunkType { byte: b });
                }
                Err(b) if 0x80 <= b && b <= 0xFD => {
                    // Spec says that chunk types 0x80-0xFD are reserved but
                    // skippable.
                    self.r.read_exact(&mut self.src[0..len])?;
                }
                Err(b) => {
                    // Can never happen. 0x02-0x7F and 0x80-0xFD are handled
                    // above in the error case. That leaves 0x00, 0x01, 0xFE
                    // and 0xFF, each of which correspond to one of the four
                    // defined chunk types.
                    unreachable!("BUG: unhandled chunk type: {}", b);
                }
                Ok(ChunkType::Padding) => {
                    // Just read and move on.
                    self.r.read_exact(&mut self.src[0..len])?;
                }
                Ok(ChunkType::Stream) => {
                    if len != STREAM_BODY.len() {
                        fail!(Error::UnsupportedChunkLength {
                            len: len64,
                            header: true,
                        })
                    }
                    self.r.read_exact(&mut self.src[0..len])?;
                    if &self.src[0..len] != STREAM_BODY {
                        fail!(Error::StreamHeaderMismatch {
                            bytes: self.src[0..len].to_vec(),
                        });
                    }
                }
                Ok(ChunkType::Uncompressed) => {
                    if len < 4 {
                        fail!(Error::UnsupportedChunkLength {
                            len: len as u64,
                            header: false,
                        });
                    }
                    let expected_sum = bytes::io_read_u32_le(&mut self.r)?;
                    let n = len - 4;
                    if n > self.dst.len() {
                        fail!(Error::UnsupportedChunkLength {
                            len: n as u64,
                            header: false,
                        });
                    }
                    self.r.read_exact(&mut self.dst[0..n])?;
                    let got_sum =
                        self.checksummer.crc32c_masked(&self.dst[0..n]);
                    if expected_sum != got_sum {
                        fail!(Error::Checksum {
                            expected: expected_sum,
                            got: got_sum,
                        });
                    }
                    self.dsts = 0;
                    self.dste = n;
                }
                Ok(ChunkType::Compressed) => {
                    if len < 4 {
                        fail!(Error::UnsupportedChunkLength {
                            len: len as u64,
                            header: false,
                        });
                    }
                    let expected_sum = bytes::io_read_u32_le(&mut self.r)?;
                    let sn = len - 4;
                    if sn > self.src.len() {
                        fail!(Error::UnsupportedChunkLength {
                            len: len64,
                            header: false,
                        });
                    }
                    self.r.read_exact(&mut self.src[0..sn])?;
                    let dn = decompress_len(&self.src)?;
                    if dn > self.dst.len() {
                        fail!(Error::UnsupportedChunkLength {
                            len: dn as u64,
                            header: false,
                        });
                    }
                    self.dec
                        .decompress(&self.src[0..sn], &mut self.dst[0..dn])?;
                    let got_sum =
                        self.checksummer.crc32c_masked(&self.dst[0..dn]);
                    if expected_sum != got_sum {
                        fail!(Error::Checksum {
                            expected: expected_sum,
                            got: got_sum,
                        });
                    }
                    self.dsts = 0;
                    self.dste = dn;
                }
            }
        }
    }
}

impl<R: fmt::Debug + io::Read> fmt::Debug for FrameDecoder<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FrameDecoder")
            .field("r", &self.r)
            .field("dec", &self.dec)
            .field("checksummer", &self.checksummer)
            .field("src", &"[...]")
            .field("dst", &"[...]")
            .field("dsts", &self.dsts)
            .field("dste", &self.dste)
            .field("read_stream_ident", &self.read_stream_ident)
            .finish()
    }
}

/// A reader for compressing data using snappy as it is read.
///
/// This `FrameEncoder` wraps any other reader that implements `std::io::Read`.
/// Bytes read from this reader are compressed using the
/// [Snappy frame format](https://github.com/google/snappy/blob/master/framing_format.txt)
/// (file extension `sz`, MIME type `application/x-snappy-framed`).
///
/// Usually you'll want
/// [`read::FrameDecoder`](struct.FrameDecoder.html)
/// (for decompressing while reading) or
/// [`write::FrameEncoder`](../write/struct.FrameEncoder.html)
/// (for compressing while writing) instead.
///
/// Unlike `FrameDecoder`, this will attempt to make large reads roughly
/// equivalent to the size of a single Snappy block. Therefore, callers may not
/// benefit from using a buffered reader.
pub struct FrameEncoder<R: io::Read> {
    /// Internally, we split `FrameEncoder` in two to keep the borrow checker
    /// happy. The `inner` member contains everything that `read_frame` needs
    /// to fetch a frame's worth of data and compress it.
    inner: Inner<R>,
    /// Data that we've encoded and are ready to return to our caller.
    dst: Vec<u8>,
    /// Starting point of bytes in `dst` not yet given back to the caller.
    dsts: usize,
    /// Ending point of bytes in `dst` that we want to give to our caller.
    dste: usize,
}

struct Inner<R: io::Read> {
    /// The underlying data source.
    r: R,
    /// An encoder that we reuse that does the actual block based compression.
    enc: Encoder,
    /// A CRC32 checksummer that is configured to either use the portable
    /// fallback version or the SSE4.2 accelerated version when the right CPU
    /// features are available.
    checksummer: CheckSummer,
    /// Data taken from the underlying `r`, and not yet compressed.
    src: Vec<u8>,
    /// Have we written the standard snappy header to `dst` yet?
    wrote_stream_ident: bool,
}

impl<R: io::Read> FrameEncoder<R> {
    /// Create a new reader for streaming Snappy compression.
    pub fn new(rdr: R) -> FrameEncoder<R> {
        FrameEncoder {
            inner: Inner {
                r: rdr,
                enc: Encoder::new(),
                checksummer: CheckSummer::new(),
                src: vec![0; MAX_BLOCK_SIZE],
                wrote_stream_ident: false,
            },
            dst: vec![0; MAX_READ_FRAME_ENCODER_BLOCK_SIZE],
            dsts: 0,
            dste: 0,
        }
    }

    /// Gets a reference to the underlying reader in this decoder.
    pub fn get_ref(&self) -> &R {
        &self.inner.r
    }

    /// Gets a mutable reference to the underlying reader in this decoder.
    ///
    /// Note that mutation of the stream may result in surprising results if
    /// this encoder is continued to be used.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner.r
    }

    /// Read previously compressed data from `self.dst`, returning the number of
    /// bytes read. If `self.dst` is empty, returns 0.
    fn read_from_dst(&mut self, buf: &mut [u8]) -> usize {
        let available_bytes = self.dste - self.dsts;
        let count = cmp::min(available_bytes, buf.len());
        buf[..count].copy_from_slice(&self.dst[self.dsts..self.dsts + count]);
        self.dsts += count;
        count
    }
}

impl<R: io::Read> io::Read for FrameEncoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Try reading previously compressed bytes from our `dst` buffer, if
        // any.
        let count = self.read_from_dst(buf);

        if count > 0 {
            // We had some bytes in our `dst` buffer that we used.
            Ok(count)
        } else if buf.len() >= MAX_READ_FRAME_ENCODER_BLOCK_SIZE {
            // Our output `buf` is big enough that we can directly write into
            // it, so bypass `dst` entirely.
            self.inner.read_frame(buf)
        } else {
            // We need to refill `self.dst`, and then return some bytes from
            // that.
            let count = self.inner.read_frame(&mut self.dst)?;
            self.dsts = 0;
            self.dste = count;
            Ok(self.read_from_dst(buf))
        }
    }
}

impl<R: io::Read> Inner<R> {
    /// Read from `self.r`, and create a new frame, writing it to `dst`, which
    /// must be at least `MAX_READ_FRAME_ENCODER_BLOCK_SIZE` bytes in size.
    fn read_frame(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        debug_assert!(dst.len() >= MAX_READ_FRAME_ENCODER_BLOCK_SIZE);

        // We make one read to the underlying reader. If the underlying reader
        // doesn't fill the buffer but there are still bytes to be read, then
        // compression won't be optimal. The alternative would be to block
        // until our buffer is maximally full (or we see EOF), but this seems
        // more surprising. In general, io::Read implementations should try to
        // fill the caller's buffer as much as they can, so this seems like the
        // better choice.
        let nread = self.r.read(&mut self.src)?;
        if nread == 0 {
            return Ok(0);
        }

        // If we haven't yet written the stream header to `dst`, write it.
        let mut dst_write_start = 0;
        if !self.wrote_stream_ident {
            dst[0..STREAM_IDENTIFIER.len()].copy_from_slice(STREAM_IDENTIFIER);
            dst_write_start += STREAM_IDENTIFIER.len();
            self.wrote_stream_ident = true;
        }

        // Reserve space for our chunk header. We need to use `split_at_mut` so
        // that we can get two mutable slices pointing at non-overlapping parts
        // of `dst`.
        let (chunk_header, remaining_dst) =
            dst[dst_write_start..].split_at_mut(CHUNK_HEADER_AND_CRC_SIZE);
        dst_write_start += CHUNK_HEADER_AND_CRC_SIZE;

        // Compress our frame if possible, telling `compress_frame` to always
        // put the output in `dst`.
        let frame_data = compress_frame(
            &mut self.enc,
            self.checksummer,
            &self.src[..nread],
            chunk_header,
            remaining_dst,
            true,
        )?;
        Ok(dst_write_start + frame_data.len())
    }
}

impl<R: fmt::Debug + io::Read> fmt::Debug for FrameEncoder<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FrameEncoder")
            .field("inner", &self.inner)
            .field("dst", &"[...]")
            .field("dsts", &self.dsts)
            .field("dste", &self.dste)
            .finish()
    }
}

impl<R: fmt::Debug + io::Read> fmt::Debug for Inner<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Inner")
            .field("r", &self.r)
            .field("enc", &self.enc)
            .field("checksummer", &self.checksummer)
            .field("src", &"[...]")
            .field("wrote_stream_ident", &self.wrote_stream_ident)
            .finish()
    }
}

// read_exact_eof is like Read::read_exact, except it detects EOF
// and returns Ok(false) instead of an error.
//
// If buf was read successfully, it returns Ok(true).
fn read_exact_eof<R: io::Read>(
    rdr: &mut R,
    buf: &mut [u8],
) -> io::Result<bool> {
    match rdr.read(buf) {
        // EOF
        Ok(0) => Ok(false),
        // Read everything w/ the read call
        Ok(i) if i == buf.len() => Ok(true),
        // There's some bytes left to fill, which can be deferred to read_exact
        Ok(i) => {
            rdr.read_exact(&mut buf[i..])?;
            Ok(true)
        }
        Err(e) => Err(e),
    }
}
