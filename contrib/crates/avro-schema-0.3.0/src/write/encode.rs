//! Functions used to encode Avro physical types
use crate::error::Error;

/// Zigzag encoding of a signed integer.
#[inline]
pub fn zigzag_encode<W: std::io::Write>(n: i64, writer: &mut W) -> Result<(), Error> {
    _zigzag_encode(((n << 1) ^ (n >> 63)) as u64, writer)
}

#[inline]
fn _zigzag_encode<W: std::io::Write>(mut z: u64, writer: &mut W) -> Result<(), Error> {
    loop {
        if z <= 0x7F {
            writer.write_all(&[(z & 0x7F) as u8])?;
            break;
        } else {
            writer.write_all(&[(0x80 | (z & 0x7F)) as u8])?;
            z >>= 7;
        }
    }
    Ok(())
}

pub(crate) fn write_binary<W: std::io::Write>(bytes: &[u8], writer: &mut W) -> Result<(), Error> {
    zigzag_encode(bytes.len() as i64, writer)?;
    writer.write_all(bytes)?;
    Ok(())
}
