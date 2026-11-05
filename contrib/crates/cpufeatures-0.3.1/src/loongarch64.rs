//! LoongArch64 CPU feature detection support.
//!
//! This implementation relies on OS-specific APIs for feature detection.

// Evaluate the given `$body` expression any of the supplied target features
// are not enabled. Otherwise returns true.
#[macro_export]
#[doc(hidden)]
macro_rules! __unless_target_features {
    ($($tf:tt),+ => $body:expr ) => {
        {
            #[cfg(not(all($(target_feature=$tf,)*)))]
            $body

            #[cfg(all($(target_feature=$tf,)*))]
            true
        }
    };
}

// Linux runtime detection of target CPU features using `getauxval`.
#[cfg(target_os = "linux")]
#[macro_export]
#[doc(hidden)]
macro_rules! __detect_target_features {
    ($($tf:tt),+) => {{
        let cpucfg1: usize;
        let cpucfg2: usize;
        let cpucfg3: usize;
        unsafe {
            std::arch::asm!(
                "cpucfg {}, {}",
                "cpucfg {}, {}",
                "cpucfg {}, {}",
                out(reg) cpucfg1, in(reg) 1,
                out(reg) cpucfg2, in(reg) 2,
                out(reg) cpucfg3, in(reg) 3,
                options(pure, nomem, preserves_flags, nostack)
            );
        }
        let hwcaps = $crate::loongarch64::getauxval_hwcap();
        $($crate::check!(cpucfg1, cpucfg2, cpucfg3, hwcaps, $tf) & )+ true
    }};
}

/// Linux helper function for calling `getauxval` to get `AT_HWCAP`.
#[cfg(target_os = "linux")]
pub fn getauxval_hwcap() -> u64 {
    unsafe { libc::getauxval(libc::AT_HWCAP) }
}

#[macro_export]
#[doc(hidden)]
macro_rules! check {
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "32s") => {
        (($cpucfg1 & 1) != 0 || ($cpucfg1 & 2) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "f") => {
        (($cpucfg2 & 2) != 0 && ($hwcaps & $crate::loongarch64::hwcaps::FPU) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "d") => {
        (($cpucfg2 & 4) != 0 && ($hwcaps & $crate::loongarch64::hwcaps::FPU) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "frecipe") => {
        (($cpucfg2 & (1 << 25)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "div32") => {
        (($cpucfg2 & (1 << 26)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lsx") => {
        (($hwcaps & $crate::loongarch64::hwcaps::LSX) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lasx") => {
        (($hwcaps & $crate::loongarch64::hwcaps::LASX) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lam-bh") => {
        (($cpucfg2 & (1 << 27)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lamcas") => {
        (($cpucfg2 & (1 << 28)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "ld-seq-sa") => {
        (($cpucfg3 & (1 << 23)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "scq") => {
        (($cpucfg2 & (1 << 30)) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lbt") => {
        (($hwcaps & $crate::loongarch64::hwcaps::LBT_X86) != 0
            && ($hwcaps & $crate::loongarch64::hwcaps::LBT_ARM) != 0
            && ($hwcaps & $crate::loongarch64::hwcaps::LBT_MIPS) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "lvz") => {
        (($hwcaps & $crate::loongarch64::hwcaps::LVZ) != 0)
    };
    ($cpucfg1:expr, $cpucfg2:expr, $cpucfg3:expr, $hwcaps:expr, "ual") => {
        (($hwcaps & $crate::loongarch64::hwcaps::UAL) != 0)
    };
}

/// Linux hardware capabilities mapped to target features.
///
/// Note that LLVM target features are coarser grained than what Linux supports
/// and imply more capabilities under each feature. This module attempts to
/// provide that mapping accordingly.
#[cfg(target_os = "linux")]
pub mod hwcaps {
    use libc::c_ulong;

    pub const UAL: c_ulong = libc::HWCAP_LOONGARCH_UAL;
    pub const FPU: c_ulong = libc::HWCAP_LOONGARCH_FPU;
    pub const LSX: c_ulong = libc::HWCAP_LOONGARCH_LSX;
    pub const LASX: c_ulong = libc::HWCAP_LOONGARCH_LASX;
    pub const LVZ: c_ulong = libc::HWCAP_LOONGARCH_LVZ;
    pub const LBT_X86: c_ulong = libc::HWCAP_LOONGARCH_LBT_X86;
    pub const LBT_ARM: c_ulong = libc::HWCAP_LOONGARCH_LBT_ARM;
    pub const LBT_MIPS: c_ulong = libc::HWCAP_LOONGARCH_LBT_MIPS;
}

// On other targets, runtime CPU feature detection is unavailable
#[cfg(not(target_os = "linux"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __detect_target_features {
    ($($tf:tt),+) => {
        false
    };
}
