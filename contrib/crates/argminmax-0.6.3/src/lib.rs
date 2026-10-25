//! A crate for finding the index of the minimum and maximum values in an array.
//!
//! These operations are optimized for speed using [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions (when available).  
//! The SIMD implementation is branchless, ensuring that there is no best case / worst case.
//! Furthermore, runtime CPU feature detection is used to choose the fastest implementation for the current CPU (with a scalar fallback).
//!
//! The SIMD implementation is enabled for the following architectures:
//! - `x86` / `x86_64`: [`SSE`](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions), [`AVX2`](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), [`AVX512`](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512)
//! - `arm` / `aarch64`: [`NEON`](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon))
//!
//! # Description
//!
//! This crate provides two traits: [`ArgMinMax`](trait.ArgMinMax.html) and [`NaNArgMinMax`](trait.NaNArgMinMax.html).
//!
//! These traits are implemented for [`slice`](https://doc.rust-lang.org/std/primitive.slice.html) and [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html).  
//! - For [`ArgMinMax`](trait.ArgMinMax.html) the supported data types are
//!   - ints: `i8`, `i16`, `i32`, `i64`
//!   - uints: `u8`, `u16`, `u32`, `u64`
//!   - floats: `f16`, `f32`, `f64` (see [Features](#features))
//! - For [`NaNArgMinMax`](trait.NaNArgMinMax.html) the supported data types are
//!   - floats: `f16`, `f32`, `f64` (see [Features](#features))
//!
//! Both traits differ in how they handle NaNs:
//! - [`ArgMinMax`](trait.ArgMinMax.html) ignores NaNs and returns the index of the minimum and maximum values in an array.
//! - [`NaNArgMinMax`](trait.NaNArgMinMax.html) returns the index of the first NaN in an array if there is one, otherwise it returns the index of the minimum and maximum values in an array.
//!
//! ### Caution
//! When dealing with floats and you are sure that there are no NaNs in the array, you should use [`ArgMinMax`](trait.ArgMinMax.html) instead of [`NaNArgMinMax`](trait.NaNArgMinMax.html) for performance reasons. The former is 5%-30% faster than the latter.
//!
//!
//! # Features
//! This crate has several features.
//!
//! - **`nightly_simd`** *(default)* - enables the use of AVX512 & (often) NEON SIMD instructions (requires a nightly compiler).
//! - **`float`** *(default)* - enables the traits for floats (`f32` and `f64`).
//! - **`half`** - enables the traits for `f16` (requires the [`half`](https://crates.io/crates/half) crate).
//! - **`ndarray`** - adds the traits to [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html) (requires the `ndarray` crate).
//! - **`arrow`** - adds the traits to [`arrow::array::PrimitiveArray`](https://docs.rs/arrow/latest/arrow/array/struct.PrimitiveArray.html) (requires the `arrow` crate).
//! - **`arrow2`** - adds the traits to [`arrow2::array::PrimitiveArray`](https://docs.rs/arrow2/latest/arrow2/array/struct.PrimitiveArray.html) (requires the `arrow2` crate).
//!
//!
//! # Examples
//!
//! Two examples are provided below.
//!
//! ## Example with integers
//! ```
//! use argminmax::ArgMinMax;
//!
//! let a: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
//! let (imin, imax) = a.argminmax();
//! assert_eq!(imin, 0);
//! assert_eq!(imax, 5);
//! ```
//!
//! ## Example with NaNs (default `float` feature)
//! ```ignore
//! use argminmax::ArgMinMax; // argminmax ignores NaNs
//! use argminmax::NaNArgMinMax; // nanargminmax returns index of first NaN
//!
//! let a: Vec<f32> = vec![f32::NAN, 1.0, f32::NAN, 3.0, 4.0, 5.0];
//! let (imin, imax) = a.argminmax(); // ArgMinMax::argminmax
//! assert_eq!(imin, 1);
//! assert_eq!(imax, 5);
//! let (imin, imax) = a.nanargminmax(); // NaNArgMinMax::nanargminmax
//! assert_eq!(imin, 0);
//! assert_eq!(imax, 0);
//!```
//!

// Enable SIMD nightly features when on nightly_simd enabled
#![cfg_attr(feature = "nightly_simd", feature(cfg_version))]
// ------- version 1.78 and above
#![cfg_attr(
    all(
        feature = "nightly_simd",
        any(target_arch = "x86_64", target_arch = "x86")
    ),
    cfg_attr(version("1.78"), feature(stdarch_x86_avx512))
)]
#![cfg_attr(
    all(feature = "nightly_simd", target_arch = "arm"),
    cfg_attr(
        version("1.78"),
        feature(stdarch_arm_neon_intrinsics),
        feature(stdarch_arm_feature_detection)
    )
)]
// ------- version 1.77 and below
#![cfg_attr(
    feature = "nightly_simd",
    cfg_attr(not(version("1.78")), feature(stdsimd))
)]
// ------- any version
#![cfg_attr(feature = "nightly_simd", feature(avx512_target_feature))]
#![cfg_attr(feature = "nightly_simd", feature(arm_target_feature))]

// It is necessary to import this at the root of the crate
// See: https://github.com/la10736/rstest/tree/master/rstest_reuse#use-rstest_resuse-at-the-top-of-your-crate
#[cfg(test)]
use rstest_reuse;

// #[macro_use]
// extern crate lazy_static;

pub mod dtype_strategy;
pub mod scalar;
pub mod simd;

pub(crate) use dtype_strategy::Int;
#[cfg(any(feature = "float", feature = "half"))]
pub(crate) use dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};
pub(crate) use scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
pub(crate) use simd::AVX512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use simd::{SIMDArgMinMax, AVX2, SSE};
#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
pub(crate) use simd::{SIMDArgMinMax, NEON};

#[cfg(feature = "half")]
use half::f16;

/// Trait for finding the minimum and maximum values in an array. For floats, NaNs are ignored.  
///
/// This trait is implemented for slices (or other array-like) of integers and floats.
///  
/// See the [feature documentation](index.html#features) for more information on the supported data types and array types.
///
pub trait ArgMinMax {
    // TODO: future work implement these other functions?
    // fn min(self) -> Self::Item;
    // fn max(self) -> Self::Item;
    // fn minmax(self) -> (T, T);

    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are ignored.  
    /// Note that this differs from numpy, where the `argmin` and `argmax` functions
    /// return the index of the first NaN (which is the behavior of our nanargminmax
    /// function).
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the array
    /// `(min_index, max_index)`.
    ///
    /// # Caution
    /// When a float array contains *only* NaNs and / or infinities unexpected behavior
    /// may occur (in which case index 0 is returned for both).
    ///
    fn argminmax(&self) -> (usize, usize);

    /// Get the index of the minimum value in the array.
    ///
    /// When dealing with floats, NaNs are ignored.
    /// Note that this differs from numpy, where the `argmin` function returns the index
    /// of the first NaN (which is the behavior of our nanargmin function).
    ///
    /// # Returns
    /// The index of the minimum value in the array.
    ///
    /// # Caution
    /// When a float array contains *only* NaNs and / or infinities unexpected behavior
    /// may occur (in which case index 0 is returned).
    ///
    fn argmin(&self) -> usize;

    /// Get the index of the maximum value in the array.
    ///
    /// When dealing with floats, NaNs are ignored.
    /// Note that this differs from numpy, where the `argmax` function returns the index
    /// of the first NaN (which is the behavior of our nanargmax function).
    ///
    /// # Returns
    /// The index of the maximum value in the array.
    ///
    /// # Caution
    /// When a float array contains *only* NaNs and / or infinities unexpected behavior
    /// may occur (in which case index 0 is returned).
    ///
    fn argmax(&self) -> usize;
}

/// Trait for finding the minimum and maximum values in an array. For floats, NaNs are propagated - index of the first NaN is returned.  
///
/// This trait is implemented for slices (or other array-like) of floats.
///  
/// See the [feature documentation](index.html#features) for more information on the supported data types and array types.
///
#[cfg(any(feature = "float", feature = "half"))]
pub trait NaNArgMinMax {
    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.  
    /// Note that this differs from numpy, where the `nanargmin` and `nanargmax`
    /// functions ignore NaNs (which is the behavior of our argminmax function).
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the array
    /// `(min_index, max_index)`.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargminmax(&self) -> (usize, usize);

    /// Get the index of the minimum value in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.
    /// Note that this differs from numpy, where the `nanargmin` function ignores
    /// NaNs (which is the behavior of our argmin function).
    ///
    /// # Returns
    /// The index of the minimum value in the array.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmin(&self) -> usize;

    /// Get the index of the maximum value in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.
    /// Note that this differs from numpy, where the `nanargmax` function ignores
    /// NaNs (which is the behavior of our argmax function).
    ///
    /// # Returns
    /// The index of the maximum value in the array.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmax(&self) -> usize;
}

// ---- Helper macros ----

trait DTypeInfo {
    const NB_BITS: usize;
}

/// Macro for implementing DTypeInfo for the passed data types (uints, ints, floats)
macro_rules! impl_nb_bits {
    // $data_type is the data type (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($data_type:ty)*) => ($(
        impl DTypeInfo for $data_type {
            const NB_BITS: usize = std::mem::size_of::<$data_type>() * 8;
        }
    )*)
}

impl_nb_bits!(i8 i16 i32 i64 u8 u16 u32 u64);
#[cfg(feature = "float")]
impl_nb_bits!(f32 f64);
#[cfg(feature = "half")]
impl_nb_bits!(f16);

// ------------------------------ &[T] ------------------------------

/// Macro for implementing ArgMinMax for signed and unsigned integers
macro_rules! impl_argminmax_int {
    // $int_type is the integer data type of the array (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($int_type:ty),*) => {
        $(
            impl ArgMinMax for &[$int_type] {
                fn argminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS == 8) {
                            // 8-bit numbers are best handled by SSE4.1
                            return unsafe { SSE::<Int>::argminmax(self) }
                        }
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$int_type>::NB_BITS <= 16) {
                                // BW (ByteWord) instructions are needed for 8 or 16-bit avx512
                                return unsafe { AVX512::<Int>::argminmax(self) }
                            }
                            else if is_x86_feature_detected!("avx512f") {  // TODO: check if avx512bw is included in avx512f
                                return unsafe { AVX512::<Int>::argminmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<Int>::argminmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        // // } else if is_x86_feature_detected!("sse4.2") & (<$int_type>::NB_BITS == 64) & (<$int_type>::IS_FLOAT == false) {
                        //     // SSE4.2 is needed for comparing 64-bit integers
                        //     return unsafe { SSE::argminmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<Int>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { NEON::<Int>::argminmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // TODO: requires v7?
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<Int>::argminmax(self) }
                        }
                    }
                    SCALAR::<Int>::argminmax(self)
                }

                fn argmin(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS == 8) {
                            // 8-bit numbers are best handled by SSE4.1
                            return unsafe { SSE::<Int>::argmin(self) }
                        }
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$int_type>::NB_BITS <= 16) {
                                // BW (ByteWord) instructions are needed for 8 or 16-bit avx512
                                return unsafe { AVX512::<Int>::argmin(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<Int>::argmin(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<Int>::argmin(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        // // } else if is_x86_feature_detected!("sse4.2") & (<$int_type>::NB_BITS == 64) & (<$int_type>::IS_FLOAT == false) {
                        //     // SSE4.2 is needed for comparing 64-bit integers
                        //     return unsafe { SSE::argmin(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<Int>::argmin(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<Int>::argmin(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<Int>::argmin(self) }
                        }
                    }
                    SCALAR::<Int>::argmin(self)
                }

                fn argmax(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS == 8) {
                            // 8-bit numbers are best handled by SSE4.1
                            return unsafe { SSE::<Int>::argmax(self) }
                        }
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$int_type>::NB_BITS <= 16) {
                                // BW (ByteWord) instructions are needed for 8 or 16-bit avx512
                                return unsafe { AVX512::<Int>::argmax(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<Int>::argmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<Int>::argmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        // // } else if is_x86_feature_detected!("sse4.2") & (<$int_type>::NB_BITS == 64) & (<$int_type>::IS_FLOAT == false) {
                        //     // SSE4.2 is needed for comparing 64-bit integers
                        //     return unsafe { SSE::argmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<Int>::argmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<Int>::argmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<Int>::argmax(self) }
                        }
                    }
                    SCALAR::<Int>::argmax(self)
                }
            }
        )*
    };
}

/// Macro for implementing ArgMinMax for floats
#[cfg(any(feature = "float", feature = "half"))]
macro_rules! impl_argminmax_float {
    // $float_type is the float data type of the array (e.g. f32)
    // you can pass multiple types (separated by commas) to this macro
    ($($float_type:ty),*) => {
        $(
            impl ArgMinMax for &[$float_type] {
                fn argminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatIgnoreNaN>::argminmax(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatIgnoreNaN>::argminmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            // f16 requires avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx") & (<$float_type>::NB_BITS > 16) {
                            // f32 and f64 do not require avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    SCALAR::<FloatIgnoreNaN>::argminmax(self)
                }

                fn argmin(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatIgnoreNaN>::argmin(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatIgnoreNaN>::argmin(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            // f16 requires avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argmin(self) }
                        } else if is_x86_feature_detected!("avx") & (<$float_type>::NB_BITS > 16) {
                            // f32 and f64 do not require avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argmin(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<FloatIgnoreNaN>::argmin(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatIgnoreNaN>::argmin(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatIgnoreNaN>::argmin(self) }
                        }
                    }
                    SCALAR::<FloatIgnoreNaN>::argmin(self)
                }

                fn argmax(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatIgnoreNaN>::argmax(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatIgnoreNaN>::argmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            // f16 requires avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argmax(self) }
                        } else if is_x86_feature_detected!("avx") & (<$float_type>::NB_BITS > 16) {
                            // f32 and f64 do not require avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<FloatIgnoreNaN>::argmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatIgnoreNaN>::argmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatIgnoreNaN>::argmax(self) }
                        }
                    }
                    SCALAR::<FloatIgnoreNaN>::argmax(self)
                }
            }

            impl NaNArgMinMax for &[$float_type] {
                fn nanargminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatReturnNaN>::argminmax(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatReturnNaN>::argminmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<FloatReturnNaN>::argminmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            // TODO: double check this (observed different things for new float implementation)
                            return unsafe { SSE::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    SCALAR::<FloatReturnNaN>::argminmax(self)
                }

                fn nanargmin(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatReturnNaN>::argmin(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatReturnNaN>::argmin(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<FloatReturnNaN>::argmin(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            // TODO: double check this (observed different things for new float implementation)
                            return unsafe { SSE::<FloatReturnNaN>::argmin(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatReturnNaN>::argmin(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatReturnNaN>::argmin(self) }
                        }
                    }
                    SCALAR::<FloatReturnNaN>::argmin(self)
                }

                fn nanargmax(&self) -> usize {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "nightly_simd")]
                        {
                            if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS == 16) {
                                // BW (ByteWord) instructions are needed for 16-bit avx512
                                return unsafe { AVX512::<FloatReturnNaN>::argmax(self) }
                            } else if is_x86_feature_detected!("avx512f") {
                                return unsafe { AVX512::<FloatReturnNaN>::argmax(self) }
                            }
                        }
                        if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<FloatReturnNaN>::argmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            // TODO: double check this (observed different things for new float implementation)
                            return unsafe { SSE::<FloatReturnNaN>::argmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return unsafe { NEON::<FloatReturnNaN>::argmax(self) }
                        }
                    }
                    #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatReturnNaN>::argmax(self) }
                        }
                    }
                    SCALAR::<FloatReturnNaN>::argmax(self)
                }
            }
        )*
    };
}

// Implement ArgMinMax for (non-optional) integer rust primitive types
impl_argminmax_int!(i8, i16, i32, i64, u8, u16, u32, u64);
// Implement for (optional) float rust primitive types
#[cfg(feature = "float")]
impl_argminmax_float!(f32, f64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
impl_argminmax_float!(f16);

// ------------------------------ [T] ------------------------------

// impl<T> ArgMinMax for [T]
// where
//     for<'a> &'a [T]: ArgMinMax,
// {
//     fn argminmax(&self) -> (usize, usize) {
//         // TODO: use the slice implementation without having stack-overflow
//     }
// }

// ------------------------------ Vec ------------------------------

impl<T> ArgMinMax for Vec<T>
where
    for<'a> &'a [T]: ArgMinMax,
{
    fn argminmax(&self) -> (usize, usize) {
        self.as_slice().argminmax()
    }

    fn argmin(&self) -> usize {
        self.as_slice().argmin()
    }

    fn argmax(&self) -> usize {
        self.as_slice().argmax()
    }
}

#[cfg(any(feature = "float", feature = "half"))]
impl<T> NaNArgMinMax for Vec<T>
where
    for<'a> &'a [T]: NaNArgMinMax,
{
    fn nanargminmax(&self) -> (usize, usize) {
        self.as_slice().nanargminmax()
    }

    fn nanargmin(&self) -> usize {
        self.as_slice().nanargmin()
    }

    fn nanargmax(&self) -> usize {
        self.as_slice().nanargmax()
    }
}

// ----------------------- (optional) ndarray ----------------------

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;
    use ndarray::{ArrayBase, Data, Ix1};

    // Use the slice implementation
    // -> implement for S where slice implementation available for S::Elem
    // ArrayBase instead of Array1 or ArrayView1 -> https://github.com/rust-ndarray/ndarray/issues/1059
    impl<S> ArgMinMax for ArrayBase<S, Ix1>
    where
        S: Data,
        for<'a> &'a [S::Elem]: ArgMinMax,
    {
        fn argminmax(&self) -> (usize, usize) {
            self.as_slice().unwrap().argminmax()
        }

        fn argmin(&self) -> usize {
            self.as_slice().unwrap().argmin()
        }

        fn argmax(&self) -> usize {
            self.as_slice().unwrap().argmax()
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    impl<S> NaNArgMinMax for ArrayBase<S, Ix1>
    where
        S: Data,
        for<'a> &'a [S::Elem]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.as_slice().unwrap().nanargminmax()
        }

        fn nanargmin(&self) -> usize {
            self.as_slice().unwrap().nanargmin()
        }

        fn nanargmax(&self) -> usize {
            self.as_slice().unwrap().nanargmax()
        }
    }
}

// ----------------------- (optional) arrow ----------------------

#[cfg(feature = "arrow")]
mod arrow_impl {
    use super::*;
    use arrow::array::PrimitiveArray;

    // Use the slice implementation
    // -> implement for T where slice implementation available for T::Native
    impl<T> ArgMinMax for PrimitiveArray<T>
    where
        T: arrow::datatypes::ArrowNumericType,
        for<'a> &'a [T::Native]: ArgMinMax,
    {
        fn argminmax(&self) -> (usize, usize) {
            self.values().as_ref().argminmax()
        }

        fn argmin(&self) -> usize {
            self.values().as_ref().argmin()
        }

        fn argmax(&self) -> usize {
            self.values().as_ref().argmax()
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    impl<T> NaNArgMinMax for PrimitiveArray<T>
    where
        T: arrow::datatypes::ArrowNumericType,
        for<'a> &'a [T::Native]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.values().as_ref().nanargminmax()
        }

        fn nanargmin(&self) -> usize {
            self.values().as_ref().nanargmin()
        }

        fn nanargmax(&self) -> usize {
            self.values().as_ref().nanargmax()
        }
    }
}

// ---------------------- (optional) arrow2 ----------------------

#[cfg(feature = "arrow2")]
mod arrow2_impl {
    use super::*;
    use arrow2::array::PrimitiveArray;

    impl<T> ArgMinMax for PrimitiveArray<T>
    where
        T: arrow2::types::NativeType,
        for<'a> &'a [T]: ArgMinMax,
    {
        fn argminmax(&self) -> (usize, usize) {
            self.values().as_ref().argminmax()
        }

        fn argmin(&self) -> usize {
            self.values().as_ref().argmin()
        }

        fn argmax(&self) -> usize {
            self.values().as_ref().argmax()
        }
    }

    #[cfg(feature = "float")]
    impl<T> NaNArgMinMax for PrimitiveArray<T>
    where
        T: arrow2::types::NativeType,
        for<'a> &'a [T]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.values().as_ref().nanargminmax()
        }

        fn nanargmin(&self) -> usize {
            self.values().as_ref().nanargmin()
        }

        fn nanargmax(&self) -> usize {
            self.values().as_ref().nanargmax()
        }
    }

    #[cfg(feature = "half")]
    #[inline(always)]
    /// Convert a PrimitiveArray<arrow2::types::f16> to a slice of half::f16
    /// To do so, the pointer to the arrow2::types::f16 slice is casted to a pointer to
    /// a slice of half::f16 (since both use u16 as their underlying type)
    fn _to_half_f16_slice(
        primitive_array_f16: &PrimitiveArray<arrow2::types::f16>,
    ) -> &[half::f16] {
        unsafe {
            std::slice::from_raw_parts(
                primitive_array_f16.values().as_ptr() as *const half::f16,
                primitive_array_f16.len(),
            )
        }
    }

    #[cfg(feature = "half")]
    impl ArgMinMax for PrimitiveArray<arrow2::types::f16> {
        fn argminmax(&self) -> (usize, usize) {
            _to_half_f16_slice(self).argminmax()
        }

        fn argmin(&self) -> usize {
            _to_half_f16_slice(self).argmin()
        }

        fn argmax(&self) -> usize {
            _to_half_f16_slice(self).argmax()
        }
    }

    #[cfg(feature = "half")]
    impl NaNArgMinMax for PrimitiveArray<arrow2::types::f16> {
        fn nanargminmax(&self) -> (usize, usize) {
            _to_half_f16_slice(self).nanargminmax()
        }

        fn nanargmin(&self) -> usize {
            _to_half_f16_slice(self).nanargmin()
        }

        fn nanargmax(&self) -> usize {
            _to_half_f16_slice(self).nanargmax()
        }
    }
}
