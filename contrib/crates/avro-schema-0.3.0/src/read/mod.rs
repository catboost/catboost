//! Functions to read and decompress Files' metadata and blocks
mod block;
mod decode;
pub(crate) mod decompress;

use std::io::Read;

use crate::error::Error;
use crate::file::FileMetadata;
use crate::schema::Schema;

pub use fallible_streaming_iterator;

// macros that can operate in sync and async code.
macro_rules! avro_decode {
    ($reader:ident $($_await:tt)*) => {
        {
            let mut i = 0u64;
            let mut buf = [0u8; 1];
            let mut j = 0;
            loop {
                if j > 9 {
                    // if j * 7 > 64
                    return Err(DecodeError::OutOfSpec);
                }
                $reader.read_exact(&mut buf[..])$($_await)*.map_err(|_| DecodeError::EndOfFile)?;
                i |= (u64::from(buf[0] & 0x7F)) << (j * 7);
                if (buf[0] >> 7) == 0 {
                    break;
                } else {
                    j += 1;
                }
            }

            Ok(i)
        }
    }
}

macro_rules! read_header {
    ($reader:ident $($_await:tt)*) => {{
        let mut items = HashMap::new();

        loop {
            let len = zigzag_i64($reader)$($_await)*.map_err(|_| Error::OutOfSpec)? as usize;
            if len == 0 {
                break Ok(items);
            }

            items.reserve(len);
            for _ in 0..len {
                let key = _read_binary($reader)$($_await)*?;
                let key = String::from_utf8(key)
                    .map_err(|_| Error::OutOfSpec)?;
                let value = _read_binary($reader)$($_await)*?;
                items.insert(key, value);
            }
        }
    }};
}

macro_rules! read_metadata_macro {
    ($reader:ident $($_await:tt)*) => {{
        let mut magic_number = [0u8; 4];
        $reader.read_exact(&mut magic_number)$($_await)*.map_err(|_| Error::OutOfSpec)?;

        // see https://avro.apache.org/docs/current/spec.html#Object+Container+Files
        if magic_number != [b'O', b'b', b'j', 1u8] {
            return Err(Error::OutOfSpec);
        }

        let header = decode::read_header($reader)$($_await)*?;

        let (schema, compression) = deserialize_header(header)?;

        let marker = decode::read_file_marker($reader)$($_await)*?;

        let record = if let Schema::Record(record) = schema {
            record
        } else {
            return Err(Error::OutOfSpec)
        };

        Ok(FileMetadata {
            record,
            compression,
            marker,
        })
    }};
}

#[allow(unused_imports)]
pub(crate) use {
    avro_decode, decode::deserialize_header, decode::DecodeError, read_header, read_metadata_macro,
};

/// Reads the metadata from `reader` into [`FileMetadata`].
/// # Error
/// This function errors iff the header is not a valid avro file header.
pub fn read_metadata<R: Read>(reader: &mut R) -> Result<FileMetadata, Error> {
    read_metadata_macro!(reader)
}

pub use decompress::{block_iterator, BlockStreamingIterator};
