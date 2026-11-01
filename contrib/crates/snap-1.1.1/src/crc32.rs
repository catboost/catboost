use crate::bytes;
use crate::crc32_table::{TABLE, TABLE16};

/// Provides a simple API to generate "masked" CRC32C checksums specifically
/// for use in Snappy. When available, this will make use of SSE 4.2 to compute
/// checksums. Otherwise, it falls back to only-marginally-slower "slicing by
/// 16" technique.
///
/// The main purpose of this type is to cache the CPU feature check and expose
/// a safe API.
#[derive(Clone, Copy, Debug)]
pub struct CheckSummer {
    sse42: bool,
}

impl CheckSummer {
    /// Create a new checksummer that can compute CRC32C checksums on arbitrary
    /// bytes.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn new() -> CheckSummer {
        CheckSummer { sse42: false }
    }

    /// Create a new checksummer that can compute CRC32C checksums on arbitrary
    /// bytes.
    #[cfg(target_arch = "x86_64")]
    pub fn new() -> CheckSummer {
        CheckSummer { sse42: is_x86_feature_detected!("sse4.2") }
    }

    /// Returns the "masked" CRC32 checksum of `buf` using the Castagnoli
    /// polynomial. This "masked" checksum is defined by the Snappy frame
    /// format. Masking is supposed to make the checksum robust with respect to
    /// the data that contains the checksum itself.
    pub fn crc32c_masked(&self, buf: &[u8]) -> u32 {
        let sum = self.crc32c(buf);
        (sum.wrapping_shr(15) | sum.wrapping_shl(17)).wrapping_add(0xA282EAD8)
    }

    /// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
    #[cfg(not(target_arch = "x86_64"))]
    fn crc32c(&self, buf: &[u8]) -> u32 {
        crc32c_slice16(buf)
    }

    /// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
    #[cfg(target_arch = "x86_64")]
    fn crc32c(&self, buf: &[u8]) -> u32 {
        if self.sse42 {
            // SAFETY: When sse42 is true, we are guaranteed to be running on
            // a CPU that supports SSE 4.2.
            unsafe { crc32c_sse(buf) }
        } else {
            crc32c_slice16(buf)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse(buf: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let mut crc = !0u32;
    // SAFETY: This is safe since alignment is handled by align_to (oh how I
    // love you) and since 8 adjacent u8's are guaranteed to have the same
    // in-memory representation as u64 for all possible values.
    let (prefix, u64s, suffix) = buf.align_to::<u64>();
    for &b in prefix {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u8(crc, b);
    }
    for &n in u64s {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u64(crc as u64, n) as u32;
    }
    for &b in suffix {
        // SAFETY: Safe since we have sse4.2 enabled.
        crc = _mm_crc32_u8(crc, b);
    }
    !crc
}

/// Returns the CRC32 checksum of `buf` using the Castagnoli polynomial.
fn crc32c_slice16(mut buf: &[u8]) -> u32 {
    let mut crc: u32 = !0;
    while buf.len() >= 16 {
        crc ^= bytes::read_u32_le(buf);
        crc = TABLE16[0][buf[15] as usize]
            ^ TABLE16[1][buf[14] as usize]
            ^ TABLE16[2][buf[13] as usize]
            ^ TABLE16[3][buf[12] as usize]
            ^ TABLE16[4][buf[11] as usize]
            ^ TABLE16[5][buf[10] as usize]
            ^ TABLE16[6][buf[9] as usize]
            ^ TABLE16[7][buf[8] as usize]
            ^ TABLE16[8][buf[7] as usize]
            ^ TABLE16[9][buf[6] as usize]
            ^ TABLE16[10][buf[5] as usize]
            ^ TABLE16[11][buf[4] as usize]
            ^ TABLE16[12][(crc >> 24) as u8 as usize]
            ^ TABLE16[13][(crc >> 16) as u8 as usize]
            ^ TABLE16[14][(crc >> 8) as u8 as usize]
            ^ TABLE16[15][(crc) as u8 as usize];
        buf = &buf[16..];
    }
    for &b in buf {
        crc = TABLE[((crc as u8) ^ b) as usize] ^ (crc >> 8);
    }
    !crc
}
