//! Functions to asynchronously write Files' metadata and blocks
use futures::{AsyncWrite, AsyncWriteExt};

use crate::{
    error::Error,
    file::{CompressedBlock, Compression},
    schema::{Record, Schema},
    write::encode::zigzag_encode,
    write::file::{write_schema, AVRO_MAGIC, SYNC_NUMBER},
};

/// Writes Avro's metadata to `writer`.
pub async fn write_metadata<W>(
    writer: &mut W,
    record: Record,
    compression: Option<Compression>,
) -> Result<(), Error>
where
    W: AsyncWrite + Unpin,
{
    writer.write_all(&AVRO_MAGIC).await?;

    // * file metadata, including the schema.
    let schema = Schema::Record(record);

    let mut scratch = vec![];
    write_schema(&mut scratch, &schema, compression)?;

    writer.write_all(&scratch).await?;

    // The 16-byte, randomly-generated sync marker for this file.
    writer.write_all(&SYNC_NUMBER).await?;

    Ok(())
}

/// Writes a [`CompressedBlock`] to `writer`
pub async fn write_block<W>(writer: &mut W, block: &CompressedBlock) -> Result<(), Error>
where
    W: AsyncWrite + Unpin,
{
    // write size and rows
    let mut scratch = Vec::with_capacity(10);
    zigzag_encode(block.number_of_rows as i64, &mut scratch)?;
    writer.write_all(&scratch).await?;
    scratch.clear();
    zigzag_encode(block.data.len() as i64, &mut scratch)?;
    writer.write_all(&scratch).await?;

    writer.write_all(&block.data).await?;

    writer.write_all(&SYNC_NUMBER).await?;

    Ok(())
}
