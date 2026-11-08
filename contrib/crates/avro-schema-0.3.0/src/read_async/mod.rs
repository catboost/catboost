//! Async Avro
use futures::AsyncRead;
use futures::AsyncReadExt;

use crate::error::Error;
use crate::file::FileMetadata;

use crate::read::read_metadata_macro;
use crate::schema::Schema;

mod block;
mod decode;
use crate::read::deserialize_header;
use decode::*;

/// Reads the avro metadata from `reader` into a [`Schema`], [`Compression`] and magic marker.
pub async fn read_metadata<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
) -> Result<FileMetadata, Error> {
    read_metadata_macro!(reader.await)
}

async fn _read_binary<R: AsyncRead + Unpin + Send>(reader: &mut R) -> Result<Vec<u8>, Error> {
    let len: usize = zigzag_i64(reader).await? as usize;
    let mut buf = vec![];
    buf.try_reserve(len).map_err(|_| Error::OutOfSpec)?;
    reader.take(len as u64).read_to_end(&mut buf).await?;
    Ok(buf)
}

pub use super::read::decompress::decompress_block;
pub use block::block_stream;
