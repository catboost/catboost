//! APIs to read from Avro format to arrow.
use async_stream::try_stream;
use futures::AsyncRead;
use futures::AsyncReadExt;
use futures::Stream;

use crate::error::Error;
use crate::file::CompressedBlock;
use crate::read::DecodeError;

use super::decode::zigzag_i64;

async fn read_size<R: AsyncRead + Unpin + Send>(reader: &mut R) -> Result<(usize, usize), Error> {
    let rows = match zigzag_i64(reader).await {
        Ok(a) => a,
        Err(error) => match error {
            DecodeError::EndOfFile => return Ok((0, 0)),
            DecodeError::OutOfSpec => return Err(Error::OutOfSpec),
        },
    };

    let bytes = zigzag_i64(reader).await?;
    Ok((rows as usize, bytes as usize))
}

/// Reads a [`CompressedBlock`] from the `reader`.
/// # Error
/// This function errors iff either the block cannot be read or the sync marker does not match
async fn read_block<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    block: &mut CompressedBlock,
    marker: [u8; 16],
) -> Result<(), Error> {
    let (rows, bytes) = read_size(reader).await?;
    block.number_of_rows = rows;
    if rows == 0 {
        return Ok(());
    };

    block.data.clear();
    block
        .data
        .try_reserve(bytes)
        .map_err(|_| Error::OutOfSpec)?;
    reader
        .take(bytes as u64)
        .read_to_end(&mut block.data)
        .await?;

    let mut block_marker = [0u8; 16];
    reader.read_exact(&mut block_marker).await?;

    if block_marker != marker {
        return Err(Error::OutOfSpec);
    }
    Ok(())
}

/// Returns a fallible [`Stream`] of Avro blocks bound to `reader`
pub async fn block_stream<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    marker: [u8; 16],
) -> impl Stream<Item = Result<CompressedBlock, Error>> + '_ {
    try_stream! {
        loop {
            let mut block = CompressedBlock::new(0, vec![]);
            read_block(reader, &mut block, marker).await?;
            if block.number_of_rows == 0 {
                break
            }
            yield block
        }
    }
}
