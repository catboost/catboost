use adler32::RollingAdler32;
#[cfg(feature = "no_std")]
use core::fmt;
#[cfg(not(feature = "no_std"))]
use std::fmt;

pub struct Adler32(RollingAdler32);
impl Adler32 {
    pub fn new() -> Self {
        Adler32(RollingAdler32::new())
    }
    pub fn value(&self) -> u32 {
        self.0.hash()
    }
    pub fn update(&mut self, buf: &[u8]) {
        self.0.update_buffer(buf);
    }
}
impl fmt::Debug for Adler32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Adler32(_)")
    }
}

pub struct Crc32(crc32fast::Hasher);
impl Crc32 {
    pub fn new() -> Self {
        Crc32(crc32fast::Hasher::new())
    }
    pub fn value(&self) -> u32 {
        self.0.clone().finalize()
    }
    pub fn update(&mut self, buf: &[u8]) {
        self.0.update(buf);
    }
}
impl fmt::Debug for Crc32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Crc32(_)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_works() {
        let mut crc32 = Crc32::new();
        crc32.update(b"abcde");
        assert_eq!(crc32.value(), 0x8587D865);
    }

    #[test]
    fn adler32_works() {
        let mut adler32 = Adler32::new();
        adler32.update(b"abcde");
        assert_eq!(adler32.value(), 0x05C801F0);
    }
}
