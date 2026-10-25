//! # Safety
//!
//! The functions in this module must only be executed on an ARM system with the CRC feature.
use super::combine::{multmodp, x2nmodp};

// Constants empirically determined to maximize speed. These values are from
// measurements on a Cortex-A57. Your mileage may vary.
const Z_BATCH: usize = 3990;
const Z_BATCH_ZEROS: u32 = x2nmodp(Z_BATCH as u64, 6);
const _: () = assert!(Z_BATCH_ZEROS == 0xa10d3d0c);
const Z_BATCH_MIN: usize = 800;

/// 3-way interleaved ARM CRC32, ported from stock zlib's ARMCRC32 path.
///
/// Runs three independent crc32x streams simultaneously to exploit the
/// ~3-cycle throughput of the instruction, then combines them with
/// polynomial multiplication.
///
/// Based on the ARMCRC32 code path in stock zlib.
#[target_feature(enable = "crc")]
pub unsafe fn crc32_acle_aarch64(crc: u32, buf: &[u8]) -> u32 {
    let mut crc = !crc;

    // SAFETY: [u8; 8] safely transmutes into u64.
    let (before, middle, after) = unsafe { buf.align_to::<u64>() };

    // SAFETY: requires the "crc" feature.
    crc = unsafe { remainder(crc, before) };

    let mut words = middle;

    while words.len() >= 3 * Z_BATCH {
        let mut crc1: u32 = 0;
        let mut crc2: u32 = 0;
        for i in 0..Z_BATCH {
            // SAFETY: requires the "crc" feature.
            unsafe {
                crc = __crc32d(crc, words[i].to_le());
                crc1 = __crc32d(crc1, words[i + Z_BATCH].to_le());
                crc2 = __crc32d(crc2, words[i + 2 * Z_BATCH].to_le());
            }
        }
        words = &words[3 * Z_BATCH..];
        crc = multmodp(Z_BATCH_ZEROS, crc) ^ crc1;
        crc = multmodp(Z_BATCH_ZEROS, crc) ^ crc2;
    }

    let last = words.len() / 3;
    if last >= Z_BATCH_MIN {
        let mut crc1: u32 = 0;
        let mut crc2: u32 = 0;
        for i in 0..last {
            // SAFETY: requires the "crc" feature.
            unsafe {
                crc = __crc32d(crc, words[i].to_le());
                crc1 = __crc32d(crc1, words[i + last].to_le());
                crc2 = __crc32d(crc2, words[i + 2 * last].to_le());
            }
        }
        words = &words[3 * last..];
        let val = x2nmodp(last as u64, 6);
        crc = multmodp(val, crc) ^ crc1;
        crc = multmodp(val, crc) ^ crc2;
    }

    for &w in words {
        // SAFETY: requires the "crc" feature.
        crc = unsafe { __crc32d(crc, w.to_le()) };
    }

    // SAFETY: requires the "crc" feature.
    crc = unsafe { remainder(crc, after) };

    !crc
}

#[inline]
#[target_feature(enable = "crc")]
unsafe fn remainder(mut c: u32, mut buf: &[u8]) -> u32 {
    if let [b0, b1, b2, b3, rest @ ..] = buf {
        c = unsafe { __crc32w(c, u32::from_le_bytes([*b0, *b1, *b2, *b3])) };
        buf = rest;
    }

    if let [b0, b1, rest @ ..] = buf {
        c = unsafe { __crc32h(c, u16::from_le_bytes([*b0, *b1])) };
        buf = rest;
    }

    if let [b0, rest @ ..] = buf {
        c = unsafe { __crc32b(c, *b0) };
        buf = rest;
    }

    debug_assert!(buf.is_empty());

    c
}

crate::cfg_select! {
    miri => {
        use core::arch::aarch64::{__crc32b, __crc32d, __crc32h, __crc32w};
    }
    _ => {
        use asm::{__crc32b, __crc32d, __crc32h, __crc32w};
    }
}

// FIXME the intrinsics below are stable since rust 1.80.0: remove these and use the standard
// library versions once our MSRV reaches that version.
mod asm {

    /// CRC32 single round checksum for bytes (8 bits).
    ///
    /// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32b)
    #[target_feature(enable = "crc")]
    #[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
    pub unsafe fn __crc32b(mut crc: u32, data: u8) -> u32 {
        unsafe {
            core::arch::asm!(
                "crc32b {crc:w}, {crc:w}, {data:w}",
                crc = inout(reg) crc,
                data = in(reg) data,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        crc
    }

    /// CRC32 single round checksum for half words (16 bits).
    ///
    /// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32h)
    #[target_feature(enable = "crc")]
    #[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
    pub unsafe fn __crc32h(mut crc: u32, data: u16) -> u32 {
        unsafe {
            core::arch::asm!(
                "crc32h {crc:w}, {crc:w}, {data:w}",
                crc = inout(reg) crc,
                data = in(reg) data,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        crc
    }

    /// CRC32 single round checksum for words (32 bits).
    ///
    /// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32w)
    #[target_feature(enable = "crc")]
    #[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
    pub unsafe fn __crc32w(mut crc: u32, data: u32) -> u32 {
        unsafe {
            core::arch::asm!(
                "crc32w {crc:w}, {crc:w}, {data:w}",
                crc = inout(reg) crc,
                data = in(reg) data,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        crc
    }

    /// CRC32 single round checksum for double words (64 bits).
    ///
    /// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/__crc32d)
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "crc")]
    pub unsafe fn __crc32d(mut crc: u32, data: u64) -> u32 {
        unsafe {
            core::arch::asm!(
                "crc32x {crc:w}, {crc:w}, {data:x}",
                crc = inout(reg) crc,
                data = in(reg) data,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        crc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck::quickcheck! {
        #[cfg(target_arch = "aarch64")]
        fn crc32_acle_aarch64_is_crc32fast(v: Vec<u8>, start: u32) -> bool {
            let mut h = crc32fast::Hasher::new_with_initial(start);
            h.update(&v);

            let a = unsafe { crc32_acle_aarch64(start, &v) };
            let b = h.finalize();

            a == b
        }
    }

    #[test]
    fn test_crc32b() {
        if !crate::cpu_features::is_enabled_crc() {
            return;
        }

        unsafe {
            assert_eq!(__crc32b(0, 0), 0);
            assert_eq!(__crc32b(0, 255), 755167117);
        }
    }

    #[test]
    fn test_crc32h() {
        if !crate::cpu_features::is_enabled_crc() {
            return;
        }

        unsafe {
            assert_eq!(__crc32h(0, 0), 0);
            assert_eq!(__crc32h(0, 16384), 1994146192);
        }
    }

    #[test]
    fn test_crc32w() {
        if !crate::cpu_features::is_enabled_crc() {
            return;
        }

        unsafe {
            assert_eq!(__crc32w(0, 0), 0);
            assert_eq!(__crc32w(0, 4294967295), 3736805603);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_crc32d() {
        if !crate::cpu_features::is_enabled_crc() {
            return;
        }

        unsafe {
            assert_eq!(__crc32d(0, 0), 0);
            assert_eq!(__crc32d(0, 18446744073709551615), 1147535477);
        }
    }
}
