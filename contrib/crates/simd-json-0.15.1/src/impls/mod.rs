#[cfg(all(
    any(test, not(feature = "portable")),
    not(target_arch = "aarch64"),
    not(target_feature = "simd128")
))]
/// rust native implementation
pub(crate) mod native;

#[cfg(feature = "portable")]
/// rust native implementation
pub(crate) mod portable;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx2;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod sse42;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_feature = "simd128")]
pub(crate) mod simd128;
