use std::io::Write;

use crate::error::Error;

use crate::file::CompressedBlock;

use super::{encode::zigzag_encode, file::SYNC_NUMBER};

/// Writes a [`CompressedBlock`] to `writer`
pub fn write_block<W: Write>(writer: &mut W, block: &CompressedBlock) -> Result<(), Error> {
    // write size and rows
    zigzag_encode(block.number_of_rows as i64, writer)?;
    zigzag_encode(block.data.len() as i64, writer)?;

    writer.write_all(&block.data)?;

    writer.write_all(&SYNC_NUMBER)?;

    Ok(())
}
