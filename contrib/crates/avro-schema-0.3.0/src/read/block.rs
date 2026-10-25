//! APIs to read from Avro format to arrow.
use std::io::Read;

use fallible_streaming_iterator::FallibleStreamingIterator;

use crate::{error::Error, file::CompressedBlock};

use super::decode;

fn read_size<R: Read>(reader: &mut R) -> Result<(usize, usize), Error> {
    let rows = match decode::internal_zigzag_i64(reader) {
        Ok(a) => a,
        Err(error) => match error {
            decode::DecodeError::EndOfFile => return Ok((0, 0)),
            decode::DecodeError::OutOfSpec => return Err(Error::OutOfSpec),
        },
    };
    let bytes = decode::zigzag_i64(reader)?;
    Ok((rows as usize, bytes as usize))
}

/// Reads a [`CompressedBlock`] from the `reader`.
/// # Error
/// This function errors iff either the block cannot be read or the sync marker does not match
fn read_block<R: Read>(
    reader: &mut R,
    block: &mut CompressedBlock,
    marker: [u8; 16],
) -> Result<(), Error> {
    let (rows, bytes) = read_size(reader)?;
    block.number_of_rows = rows;
    if rows == 0 {
        return Ok(());
    };

    block.data.clear();
    block
        .data
        .try_reserve(bytes)
        .map_err(|_| Error::OutOfSpec)?;
    reader.take(bytes as u64).read_to_end(&mut block.data)?;

    let mut block_marker = [0u8; 16];
    reader.read_exact(&mut block_marker)?;

    if block_marker != marker {
        return Err(Error::OutOfSpec);
    }
    Ok(())
}

/// [`FallibleStreamingIterator`] of [`CompressedBlock`].
pub struct CompressedBlockStreamingIterator<R: Read> {
    buf: CompressedBlock,
    reader: R,
    marker: [u8; 16],
}

impl<R: Read> CompressedBlockStreamingIterator<R> {
    /// Creates a new [`CompressedBlockStreamingIterator`].
    pub fn new(reader: R, marker: [u8; 16], scratch: Vec<u8>) -> Self {
        Self {
            reader,
            marker,
            buf: CompressedBlock::new(0, scratch),
        }
    }

    /// The buffer of [`CompressedBlockStreamingIterator`].
    pub fn buffer(&mut self) -> &mut CompressedBlock {
        &mut self.buf
    }

    /// Deconstructs itself
    pub fn into_inner(self) -> (R, Vec<u8>) {
        (self.reader, self.buf.data)
    }
}

impl<R: Read> FallibleStreamingIterator for CompressedBlockStreamingIterator<R> {
    type Error = Error;
    type Item = CompressedBlock;

    fn advance(&mut self) -> Result<(), Error> {
        read_block(&mut self.reader, &mut self.buf, self.marker)?;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        if self.buf.number_of_rows > 0 {
            Some(&self.buf)
        } else {
            None
        }
    }
}
