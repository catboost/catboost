#![allow(dead_code)]
#![allow(unreachable_code)]

pub struct CpuFeatures;

impl CpuFeatures {
    pub const NONE: usize = 0;
    pub const AVX2: usize = 1;
}

#[inline(always)]
pub fn is_enabled_sse() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(feature = "std")]
    return std::is_x86_feature_detected!("sse");

    false
}

#[inline(always)]
pub fn is_enabled_sse42() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(feature = "std")]
    return std::is_x86_feature_detected!("sse4.2");

    false
}

#[inline(always)]
pub fn is_enabled_avx2_and_bmi2() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "bmi1",
            target_feature = "bmi2"
        ))]
        return true;

        #[cfg(feature = "std")]
        {
            use std::sync::atomic::{AtomicU32, Ordering};

            static CACHE: AtomicU32 = AtomicU32::new(2);

            return match CACHE.load(Ordering::Relaxed) {
                0 => false,
                1 => true,
                _ => {
                    let detected = std::is_x86_feature_detected!("avx2")
                        && std::is_x86_feature_detected!("bmi1")
                        && std::is_x86_feature_detected!("bmi2");
                    CACHE.store(u32::from(detected), Ordering::Relaxed);
                    detected
                }
            };
        }
    }

    false
}

#[inline(always)]
pub fn is_enabled_avx512() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[cfg(feature = "std")]
    return std::is_x86_feature_detected!("avx512f");

    false
}

#[inline(always)]
pub fn is_enabled_pclmulqdq() -> bool {
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "std")]
    return std::is_x86_feature_detected!("pclmulqdq") && std::is_x86_feature_detected!("sse4.1");

    false
}

#[inline(always)]
pub fn is_enabled_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        return true;

        #[cfg(feature = "std")]
        return std::arch::is_aarch64_feature_detected!("neon");
    }

    false
}

#[inline(always)]
pub fn is_enabled_crc() -> bool {
    #[cfg(target_arch = "aarch64")]
    #[cfg(feature = "std")]
    return std::arch::is_aarch64_feature_detected!("crc");

    false
}

// FIXME: Remove lsx feature guard once MSRV is 1.89.0 or higher
#[inline(always)]
pub fn is_enabled_lsx() -> bool {
    #[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
    #[cfg(feature = "std")]
    return std::arch::is_loongarch_feature_detected!("lsx");

    false
}

#[inline(always)]
pub fn is_enabled_simd128() -> bool {
    #[cfg(target_arch = "wasm32")]
    return cfg!(target_feature = "simd128");

    false
}
