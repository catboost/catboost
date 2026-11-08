use std::collections::HashMap;

use crate::error::Error;
use crate::file::Compression;
use crate::schema::{Record, Schema};

use super::encode;

pub(crate) const SYNC_NUMBER: [u8; 16] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
// * Four bytes, ASCII 'O', 'b', 'j', followed by 1.
pub(crate) const AVRO_MAGIC: [u8; 4] = [b'O', b'b', b'j', 1u8];

/// Serializes an [`Schema`] and optional [`Compression`] into an avro header.
fn serialize_header(
    schema: &Schema,
    compression: Option<Compression>,
) -> Result<HashMap<String, Vec<u8>>, Error> {
    let schema = serde_json::to_string(schema).map_err(|_| Error::OutOfSpec)?;

    let mut header = HashMap::<String, Vec<u8>>::default();

    header.insert("avro.schema".to_string(), schema.into_bytes());
    if let Some(compression) = compression {
        let value = match compression {
            Compression::Snappy => b"snappy".to_vec(),
            Compression::Deflate => b"deflate".to_vec(),
        };
        header.insert("avro.codec".to_string(), value);
    };

    Ok(header)
}

/// Writes Avro's metadata to `writer`.
pub fn write_metadata<W: std::io::Write>(
    writer: &mut W,
    record: Record,
    compression: Option<Compression>,
) -> Result<(), Error> {
    writer.write_all(&AVRO_MAGIC)?;

    // * file metadata, including the schema.
    let schema = Schema::Record(record);

    write_schema(writer, &schema, compression)?;

    // The 16-byte, randomly-generated sync marker for this file.
    writer.write_all(&SYNC_NUMBER)?;

    Ok(())
}

pub(crate) fn write_schema<W: std::io::Write>(
    writer: &mut W,
    schema: &Schema,
    compression: Option<Compression>,
) -> Result<(), Error> {
    let header = serialize_header(schema, compression)?;

    encode::zigzag_encode(header.len() as i64, writer)?;
    for (name, item) in header {
        encode::write_binary(name.as_bytes(), writer)?;
        encode::write_binary(&item, writer)?;
    }
    writer.write_all(&[0])?;
    Ok(())
}
