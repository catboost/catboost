//! SIMD implementations of the argminmax functions.

// --- Generic implementations ---
mod config;
pub use config::*;
mod generic;
pub use generic::*;
// Helper mod
mod task;

// --- SIMD implementations ---

// FLOAT
#[cfg(feature = "half")]
mod simd_f16_ignore_nan;
#[cfg(feature = "half")]
mod simd_f16_return_nan;
#[cfg(feature = "float")]
mod simd_f32_ignore_nan;
#[cfg(feature = "float")]
mod simd_f32_return_nan;
#[cfg(feature = "float")]
mod simd_f64_ignore_nan;
#[cfg(feature = "float")]
mod simd_f64_return_nan;
// SIGNED INT
mod simd_i16;
mod simd_i32;
mod simd_i64;
mod simd_i8;
// UNSIGNED INT
mod simd_u16;
mod simd_u32;
mod simd_u64;
mod simd_u8;

// Test utils
#[cfg(test)]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", feature = "float"), // is stable for f64
    feature = "nightly_simd"
))]
mod test_utils;
