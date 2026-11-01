use std::collections::HashMap;
use std::io::Read;

use crate::error::Error;
use crate::file::Compression;
use crate::schema::Schema;

use super::{avro_decode, read_header};

pub enum DecodeError {
    OutOfSpec,
    EndOfFile,
}

impl From<DecodeError> for Error {
    fn from(_: DecodeError) -> Self {
        Error::OutOfSpec
    }
}

pub fn internal_zigzag_i64<R: Read>(reader: &mut R) -> Result<i64, DecodeError> {
    let z = decode_variable(reader)?;
    Ok(if z & 0x1 == 0 {
        (z >> 1) as i64
    } else {
        !(z >> 1) as i64
    })
}

pub fn zigzag_i64<R: Read>(reader: &mut R) -> Result<i64, Error> {
    let z = decode_variable(reader)?;
    Ok(if z & 0x1 == 0 {
        (z >> 1) as i64
    } else {
        !(z >> 1) as i64
    })
}

#[inline]
fn decode_variable<R: Read>(reader: &mut R) -> Result<u64, DecodeError> {
    avro_decode!(reader)
}

fn _read_binary<R: Read>(reader: &mut R) -> Result<Vec<u8>, Error> {
    let len: usize = zigzag_i64(reader)? as usize;
    let mut buf = vec![];
    buf.try_reserve(len).map_err(|_| Error::OutOfSpec)?;
    reader.take(len as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

pub fn read_header<R: Read>(reader: &mut R) -> Result<HashMap<String, Vec<u8>>, Error> {
    read_header!(reader)
}

pub(crate) fn read_file_marker<R: Read>(reader: &mut R) -> Result<[u8; 16], Error> {
    let mut marker = [0u8; 16];
    reader.read_exact(&mut marker)?;
    Ok(marker)
}

/// Deserializes the Avro header into an Avro [`Schema`] and optional [`Compression`].
pub(crate) fn deserialize_header(
    header: HashMap<String, Vec<u8>>,
) -> Result<(Schema, Option<Compression>), Error> {
    let schema = header
        .get("avro.schema")
        .ok_or(Error::OutOfSpec)
        .and_then(|bytes| serde_json::from_slice(bytes.as_ref()).map_err(|_| Error::OutOfSpec))?;

    let compression = header.get("avro.codec").and_then(|bytes| {
        let bytes: &[u8] = bytes.as_ref();
        match bytes {
            b"snappy" => Some(Compression::Snappy),
            b"deflate" => Some(Compression::Deflate),
            _ => None,
        }
    });
    Ok((schema, compression))
}
