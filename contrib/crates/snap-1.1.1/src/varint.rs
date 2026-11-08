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
