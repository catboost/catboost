use crate::util::*;

pub(crate) const fn crc8_table(poly: u8, reflect: bool) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc8(poly, reflect, i as u8);
        i += 1;
    }
    table
}
pub(crate) const fn crc16_table(poly: u16, reflect: bool) -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc16(poly, reflect, i as u8);
        i += 1;
    }
    table
}
pub(crate) const fn crc32_table(poly: u32, reflect: bool) -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc32(poly, reflect, i as u8);
        i += 1;
    }
    table
}
pub(crate) const fn crc64_table(poly: u64, reflect: bool) -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc64(poly, reflect, i as u8);
        i += 1;
    }
    table
}
