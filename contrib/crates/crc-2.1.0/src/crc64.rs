use super::{Algorithm, Crc, Digest};
use crate::table::crc64_table;

impl Crc<u64> {
    pub const fn new(algorithm: &'static Algorithm<u64>) -> Self {
        let table = crc64_table(algorithm.poly, algorithm.refin);
        Self { algorithm, table }
    }

    pub const fn checksum(&self, bytes: &[u8]) -> u64 {
        let mut crc = self.init();
        crc = self.update(crc, bytes);
        self.finalize(crc)
    }

    const fn init(&self) -> u64 {
        if self.algorithm.refin {
            self.algorithm.init.reverse_bits()
        } else {
            self.algorithm.init
        }
    }

    const fn table_entry(&self, index: u64) -> u64 {
        self.table[(index & 0xFF) as usize]
    }

    const fn update(&self, mut crc: u64, bytes: &[u8]) -> u64 {
        let mut i = 0;
        if self.algorithm.refin {
            while i < bytes.len() {
                crc = self.table_entry(crc ^ bytes[i] as u64) ^ (crc >> 8);
                i += 1;
            }
        } else {
            while i < bytes.len() {
                crc = self.table_entry(bytes[i] as u64 ^ (crc >> 56)) ^ (crc << 8);
                i += 1;
            }
        }
        crc
    }

    const fn finalize(&self, mut crc: u64) -> u64 {
        if self.algorithm.refin ^ self.algorithm.refout {
            crc = crc.reverse_bits();
        }
        crc ^ self.algorithm.xorout
    }

    pub const fn digest(&self) -> Digest<u64> {
        Digest::new(self)
    }
}

impl<'a> Digest<'a, u64> {
    const fn new(crc: &'a Crc<u64>) -> Self {
        let value = crc.init();
        Digest { crc, value }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.value = self.crc.update(self.value, bytes);
    }

    pub const fn finalize(self) -> u64 {
        self.crc.finalize(self.value)
    }
}
