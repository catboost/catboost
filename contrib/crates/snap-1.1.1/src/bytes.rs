use std::convert::TryInto;
use std::io;

/// Read a u16 in little endian format from the beginning of the given slice.
/// This panics if the slice has length less than 2.
pub fn read_u16_le(slice: &[u8]) -> u16 {
    u16::from_le_bytes(slice[..2].try_into().unwrap())
}

/// Read a u24 (returned as a u32 with the most significant 8 bits always set
/// to 0) in little endian format from the beginning of the given slice. This
/// panics if the slice has length less than 3.
pub fn read_u24_le(slice: &[u8]) -> u32 {
    slice[0] as u32 | (slice[1] as u32) << 8 | (slice[2] as u32) << 16
}

/// Read a u32 in little endian format from the beginning of the given slice.
/// This panics if the slice has length less than 4.
pub fn read_u32_le(slice: &[u8]) -> u32 {
    u32::from_le_bytes(slice[..4].try_into().unwrap())
}

/// Like read_u32_le, but from an io::Read implementation. If io::Read does
/// not yield at least 4 bytes, then this returns an unexpected EOF error.
pub fn io_read_u32_le<R: io::Read>(mut rdr: R) -> io::Result<u32> {
    let mut buf = [0; 4];
    rdr.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Write a u16 in little endian format to the beginning of the given slice.
/// This panics if the slice has length less than 2.
pub fn write_u16_le(n: u16, slice: &mut [u8]) {
    assert!(slice.len() >= 2);
    let bytes = n.to_le_bytes();
    slice[0] = bytes[0];
    slice[1] = bytes[1];
}

/// Write a u24 (given as a u32 where the most significant 8 bits are ignored)
/// in little endian format to the beginning of the given slice. This panics
/// if the slice has length less than 3.
pub fn write_u24_le(n: u32, slice: &mut [u8]) {
    slice[0] = n as u8;
    slice[1] = (n >> 8) as u8;
    slice[2] = (n >> 16) as u8;
}

/// Write a u32 in little endian format to the beginning of the given slice.
/// This panics if the slice has length less than 4.
pub fn write_u32_le(n: u32, slice: &mut [u8]) {
    assert!(slice.len() >= 4);
    let bytes = n.to_le_bytes();
    slice[0] = bytes[0];
    slice[1] = bytes[1];
    slice[2] = bytes[2];
    slice[3] = bytes[3];
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn write_varu64(data: &mut [u8], mut n: u64) -> usize {
    let mut i = 0;
    while n >= 0b1000_0000 {
        data[i] = (n as u8) | 0b1000_0000;
        n >>= 7;
        i += 1;
    }
    data[i] = n as u8;
    i + 1
}

/// https://developers.google.com/protocol-buffers/docs/encoding#varints
pub fn read_varu64(data: &[u8]) -> (u64, usize) {
    let mut n: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        if b < 0b1000_0000 {
            return match (b as u64).checked_shl(shift) {
                None => (0, 0),
                Some(b) => (n | b, i + 1),
            };
        }
        match ((b as u64) & 0b0111_1111).checked_shl(shift) {
            None => return (0, 0),
            Some(b) => n |= b,
        }
        shift += 7;
    }
    (0, 0)
}

/// Does an unaligned load of a little endian encoded u32.
///
/// This is unsafe because `data` must point to some memory of size at least 4.
pub unsafe fn loadu_u32_le(data: *const u8) -> u32 {
    loadu_u32_ne(data).to_le()
}

/// Does an unaligned load of a native endian encoded u32.
///
/// This is unsafe because `data` must point to some memory of size at least 4.
pub unsafe fn loadu_u32_ne(data: *const u8) -> u32 {
    (data as *const u32).read_unaligned()
}

/// Does an unaligned load of a little endian encoded u64.
///
/// This is unsafe because `data` must point to some memory of size at least 8.
pub unsafe fn loadu_u64_le(data: *const u8) -> u64 {
    loadu_u64_ne(data).to_le()
}

/// Does an unaligned load of a native endian encoded u64.
///
/// This is unsafe because `data` must point to some memory of size at least 8.
pub unsafe fn loadu_u64_ne(data: *const u8) -> u64 {
    (data as *const u64).read_unaligned()
}
