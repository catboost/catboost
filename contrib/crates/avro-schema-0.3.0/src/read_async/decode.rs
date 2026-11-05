use std::collections::HashMap;

use futures::AsyncRead;
use futures::AsyncReadExt;

use crate::error::Error;
use crate::read::DecodeError;
use crate::read::{avro_decode, read_header};

pub async fn zigzag_i64<R: AsyncRead + Unpin + Send>(reader: &mut R) -> Result<i64, DecodeError> {
    let z = decode_variable(reader).await?;
    Ok(if z & 0x1 == 0 {
        (z >> 1) as i64
    } else {
        !(z >> 1) as i64
    })
}

async fn decode_variable<R: AsyncRead + Unpin + Send>(reader: &mut R) -> Result<u64, DecodeError> {
    avro_decode!(reader.await)
}

/// Reads the file marker asynchronously
pub(crate) async fn read_file_marker<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
) -> Result<[u8; 16], Error> {
    let mut marker = [0u8; 16];
    reader.read_exact(&mut marker).await?;
    Ok(marker)
}

async fn _read_binary<R: AsyncRead + Unpin + Send>(reader: &mut R) -> Result<Vec<u8>, Error> {
    let len: usize = zigzag_i64(reader).await? as usize;
    let mut buf = vec![];
    buf.try_reserve(len).map_err(|_| Error::OutOfSpec)?;
    reader.take(len as u64).read_to_end(&mut buf).await?;
    Ok(buf)
}

pub(crate) async fn read_header<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
) -> Result<HashMap<String, Vec<u8>>, Error> {
    read_header!(reader.await)
}
