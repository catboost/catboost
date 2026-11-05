use crate::bytes;
use crate::compress::{max_compress_len, Encoder};
use crate::crc32::CheckSummer;
use crate::error::Error;
use crate::MAX_BLOCK_SIZE;

/// The maximum chunk of compressed bytes that can be processed at one time.
///
/// This is computed via `max_compress_len(MAX_BLOCK_SIZE)`.
///
/// TODO(ag): Replace with const fn once they support nominal branching.
pub const MAX_COMPRESS_BLOCK_SIZE: usize = 76490;

/// The special magic string that starts any stream.
///
/// This may appear more than once in a stream in order to support easy
/// concatenation of files compressed in the Snappy frame format.
pub const STREAM_IDENTIFIER: &'static [u8] = b"\xFF\x06\x00\x00sNaPpY";

/// The body of the special stream identifier.
pub const STREAM_BODY: &'static [u8] = b"sNaPpY";

/// The length of a snappy chunk type (1 byte), packet length (3 bytes)
/// and CRC field (4 bytes). This is technically the chunk header _plus_
/// the CRC present in most chunks.
pub const CHUNK_HEADER_AND_CRC_SIZE: usize = 8;

/// An enumeration describing each of the 4 main chunk types.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ChunkType {
    Stream = 0xFF,
    Compressed = 0x00,
    Uncompressed = 0x01,
    Padding = 0xFE,
}

impl ChunkType {
    /// Converts a byte to one of the four defined chunk types represented by
    /// a single byte. If the chunk type is reserved, then it is returned as
    /// an Err.
    pub fn from_u8(b: u8) -> Result<ChunkType, u8> {
        match b {
            0xFF => Ok(ChunkType::Stream),
            0x00 => Ok(ChunkType::Compressed),
            0x01 => Ok(ChunkType::Uncompressed),
            0xFE => Ok(ChunkType::Padding),
            b => Err(b),
        }
    }
}

/// Compress a single frame (or decide to pass it through uncompressed). This
/// will output a frame header in `dst_chunk_header`, and it will return a slice
/// pointing to the data to use in the frame. The `dst_chunk_header` array must
/// always have a size of 8 bytes.
///
/// If `always_use_dst` is set to false, the return value may point into either
/// `src` (for data we couldn't compress) or into `dst` (for data we could
/// compress). If `always_use_dst` is true, the data will always be in `dst`.
/// This is a bit weird, but because of Rust's ownership rules, it's easiest
/// for a single function to always be in charge of writing to `dst`.
pub fn compress_frame<'a>(
    enc: &mut Encoder,
    checksummer: CheckSummer,
    src: &'a [u8],
    dst_chunk_header: &mut [u8],
    dst: &'a mut [u8],
    always_use_dst: bool,
) -> Result<&'a [u8], Error> {
    // This is a purely internal function, with a bunch of preconditions.
    assert!(src.len() <= MAX_BLOCK_SIZE);
    assert!(dst.len() >= max_compress_len(MAX_BLOCK_SIZE));
    assert_eq!(dst_chunk_header.len(), CHUNK_HEADER_AND_CRC_SIZE);

    // Build a checksum of our _uncompressed_ data.
    let checksum = checksummer.crc32c_masked(src);

    // Compress the buffer. If compression sucked, throw it out and
    // write uncompressed bytes instead. Since our buffer is at most
    // MAX_BLOCK_SIZE and our dst buffer has size
    // max_compress_len(MAX_BLOCK_SIZE), we have enough space.
    let compress_len = enc.compress(src, dst)?;
    let (chunk_type, chunk_len) =
        // We add 4 to the chunk_len because of the checksum.
        if compress_len >= src.len() - (src.len() / 8) {
            (ChunkType::Uncompressed, 4 + src.len())
        } else {
            (ChunkType::Compressed, 4 + compress_len)
        };

    dst_chunk_header[0] = chunk_type as u8;
    bytes::write_u24_le(chunk_len as u32, &mut dst_chunk_header[1..]);
    bytes::write_u32_le(checksum, &mut dst_chunk_header[4..]);

    // Return the data to put in our frame.
    if chunk_type == ChunkType::Compressed {
        Ok(&dst[0..compress_len])
    } else if always_use_dst {
        dst[..src.len()].copy_from_slice(src);
        Ok(&dst[..src.len()])
    } else {
        Ok(src)
    }
}
