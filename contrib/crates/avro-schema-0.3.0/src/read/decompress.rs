//! APIs to read from Avro format to arrow.
use std::io::Read;

use fallible_streaming_iterator::FallibleStreamingIterator;

use crate::error::Error;

use crate::file::Compression;
use crate::file::{Block, CompressedBlock};

use super::block::CompressedBlockStreamingIterator;

#[cfg(feature = "compression")]
const CRC_TABLE: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);

/// Decompresses a [`CompressedBlock`] into [`Block`]
/// Returns whether the buffers where swapped.
pub fn decompress_block(
    block: &mut CompressedBlock,
    decompressed: &mut Block,
    compression: Option<Compression>,
) -> Result<bool, Error> {
    decompressed.number_of_rows = block.number_of_rows;
    let block = &mut block.data;
    let decompressed = &mut decompressed.data;

    match compression {
        None => {
            std::mem::swap(block, decompressed);
            Ok(true)
        }
        #[cfg(feature = "compression")]
        Some(Compression::Deflate) => {
            decompressed.clear();
            let mut decoder = libflate::deflate::Decoder::new(&block[..]);
            decoder.read_to_end(decompressed)?;
            Ok(false)
        }
        #[cfg(feature = "compression")]
        Some(Compression::Snappy) => {
            let crc = &block[block.len() - 4..];
            let block = &block[..block.len() - 4];

            let len = snap::raw::decompress_len(block).map_err(|_| Error::OutOfSpec)?;
            decompressed.clear();
            decompressed.resize(len, 0);
            snap::raw::Decoder::new()
                .decompress(block, decompressed)
                .map_err(|_| Error::OutOfSpec)?;

            let expected_crc = u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]]);

            let actual_crc = CRC_TABLE.checksum(decompressed);
            if expected_crc != actual_crc {
                return Err(Error::OutOfSpec);
            }
            Ok(false)
        }
        #[cfg(not(feature = "compression"))]
        Some(Compression::Deflate) => Err(Error::RequiresCompression),
        #[cfg(not(feature = "compression"))]
        Some(Compression::Snappy) => Err(Error::RequiresCompression),
    }
}

/// [`FallibleStreamingIterator`] of decompressed [`Block`]
pub struct BlockStreamingIterator<R: Read> {
    blocks: CompressedBlockStreamingIterator<R>,
    compression: Option<Compression>,
    buf: Block,
    was_swapped: bool,
}

/// Returns a [`FallibleStreamingIterator`] of [`Block`].
pub fn block_iterator<R: Read>(
    reader: R,
    compression: Option<Compression>,
    marker: [u8; 16],
) -> BlockStreamingIterator<R> {
    BlockStreamingIterator::<R>::new(reader, compression, marker)
}

impl<R: Read> BlockStreamingIterator<R> {
    /// Returns a new [`BlockStreamingIterator`].
    pub fn new(reader: R, compression: Option<Compression>, marker: [u8; 16]) -> Self {
        Self {
            blocks: CompressedBlockStreamingIterator::new(reader, marker, vec![]),
            compression,
            buf: Block::new(0, vec![]),
            was_swapped: false,
        }
    }

    /// Deconstructs itself into its internal reader
    #[inline]
    pub fn into_inner(self) -> R {
        self.blocks.into_inner().0
    }
}

impl<R: Read> FallibleStreamingIterator for BlockStreamingIterator<R> {
    type Error = Error;
    type Item = Block;

    #[inline]
    fn advance(&mut self) -> Result<(), Error> {
        if self.was_swapped {
            std::mem::swap(&mut self.blocks.buffer().data, &mut self.buf.data);
        }
        self.blocks.advance()?;
        self.was_swapped = decompress_block(self.blocks.buffer(), &mut self.buf, self.compression)?;
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        if self.buf.number_of_rows > 0 {
            Some(&self.buf)
        } else {
            None
        }
    }
}
