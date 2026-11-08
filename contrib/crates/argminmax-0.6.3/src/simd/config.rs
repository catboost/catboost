/// This module contains structs and traits that are used to configure the SIMD
/// implementation. The structs are used to store the register size and the trait is
/// used to get the lane size for a given datatype.
///
/// More info on SIMD:
/// https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features
///
use std::marker::PhantomData;

/// SIMD instruction set trait - used to store the register size and get the lane size
/// for a given datatype
pub(crate) trait SIMDInstructionSet {
    /// The size of the register in bits
    const REGISTER_SIZE: usize;

    // Set the const LANE_SIZE for each datatype (postfix is the number of bits)
    const LANE_SIZE_8: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u8>() * 8);
    const LANE_SIZE_16: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u16>() * 8);
    const LANE_SIZE_32: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u32>() * 8);
    const LANE_SIZE_64: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u64>() * 8);
}

// ----------------------------------- x86_64 / x86 ------------------------------------

/// [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) instruction set.
///
/// Type that implements the [SIMDArgMinMax](crate::SIMDArgMinMax) trait.
///
/// This struct implements the SIMDArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
// This will be implemented for all:
// - ints (see simd_i*.rs files) - Int DTypeStrategy
// - uints (see simd_u*.rs files) - Int DTypeStrategy
// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
pub struct SSE<DTypeStrategy> {
    pub(crate) _dtype_strategy: PhantomData<DTypeStrategy>,
}

impl<DTypeStrategy> SIMDInstructionSet for SSE<DTypeStrategy> {
    /// SSE register size is 128 bits
    /// https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#Registers
    const REGISTER_SIZE: usize = 128;
}

/// [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) instruction set.
///
/// Type that implements the [SIMDArgMinMax](crate::SIMDArgMinMax) trait.
///
/// This struct implements the SIMDArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
/// Note that [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions) is enough for f32 and f64, but we need AVX2 for all other data types.
///
// This will be implemented for all:
// - ints (see simd_i*.rs files) - Int DTypeStrategy
// - uints (see simd_u*.rs files) - Int DTypeStrategy
// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
//     ! important remark: AVX is enough for f32 and f64, but we need AVX2 for f16
//
pub struct AVX2<DTypeStrategy> {
    pub(crate) _dtype_strategy: PhantomData<DTypeStrategy>,
}

impl<DTypeStrategy> SIMDInstructionSet for AVX2<DTypeStrategy> {
    /// AVX(2) register size is 256 bits
    /// AVX:  https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions
    /// AVX2: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX2
    const REGISTER_SIZE: usize = 256;
}

/// [AVX512](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512) instruction set.
///
/// Type that implements the [SIMDArgMinMax](crate::SIMDArgMinMax) trait.
///
/// This struct implements the SIMDArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
// This will be implemented for all:
// - ints (see simd_i*.rs files) - Int DTypeStrategy
// - uints (see simd_u*.rs files) - Int DTypeStrategy
// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
#[cfg(feature = "nightly_simd")]
pub struct AVX512<DTypeStrategy> {
    pub(crate) _dtype_strategy: PhantomData<DTypeStrategy>,
}

#[cfg(feature = "nightly_simd")]
impl<DTypeStrategy> SIMDInstructionSet for AVX512<DTypeStrategy> {
    /// AVX512 register size is 512 bits
    /// https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512
    const REGISTER_SIZE: usize = 512;
}

// ----------------------------------- aarch64 / arm -----------------------------------

/// [NEON](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon)) instruction set.
///
/// Type that implements the [SIMDArgMinMax](crate::SIMDArgMinMax) trait.
///
/// This struct implements the SIMDArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
/// Note: there are no NEON instructions for 64-bit numbers, so for 64-bit numbers we
/// fall back to the scalar implementation.
///
// This will be implemented for all:
// - ints (see simd_i*.rs files) - Int DTypeStrategy
// - uints (see simd_u*.rs files) - Int DTypeStrategy
// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
pub struct NEON<DTypeStrategy> {
    pub(crate) _dtype_strategy: PhantomData<DTypeStrategy>,
}

impl<DTypeStrategy> SIMDInstructionSet for NEON<DTypeStrategy> {
    /// NEON register size is 128 bits
    /// https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon)
    const REGISTER_SIZE: usize = 128;
}

// ======================================= TESTS =======================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype_strategy::*;

    use rstest::rstest;
    use rstest_reuse::{self, *};

    #[cfg(feature = "half")]
    use half::f16;

    // The DTypeStrategy should not influence the lane size
    #[cfg(any(feature = "float", feature = "half"))]
    #[template]
    #[rstest]
    #[case::int(Int)]
    #[case::float_return_nan(FloatIgnoreNaN)]
    #[case::float_ignore_nan(FloatReturnNaN)]
    fn dtype_strategies<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {}

    #[cfg(not(any(feature = "float", feature = "half")))]
    #[template]
    #[rstest]
    #[case::int(Int)]
    fn dtype_strategies<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {}

    #[apply(dtype_strategies)]
    fn test_lane_size_8bit_dtype<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::LANE_SIZE_8, 16);
        assert_eq!(AVX2::<DTypeStrategy>::LANE_SIZE_8, 32);
        #[cfg(feature = "nightly_simd")]
        assert_eq!(AVX512::<DTypeStrategy>::LANE_SIZE_8, 64);
        assert_eq!(NEON::<DTypeStrategy>::LANE_SIZE_8, 16);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_16bit_dtype<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::LANE_SIZE_16, 8);
        assert_eq!(AVX2::<DTypeStrategy>::LANE_SIZE_16, 16);
        #[cfg(feature = "nightly_simd")]
        assert_eq!(AVX512::<DTypeStrategy>::LANE_SIZE_16, 32);
        assert_eq!(NEON::<DTypeStrategy>::LANE_SIZE_16, 8);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_32bit_dtype<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::LANE_SIZE_32, 4);
        assert_eq!(AVX2::<DTypeStrategy>::LANE_SIZE_32, 8);
        #[cfg(feature = "nightly_simd")]
        assert_eq!(AVX512::<DTypeStrategy>::LANE_SIZE_32, 16);
        assert_eq!(NEON::<DTypeStrategy>::LANE_SIZE_32, 4);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_64bit_dtype<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::LANE_SIZE_64, 2);
        assert_eq!(AVX2::<DTypeStrategy>::LANE_SIZE_64, 4);
        #[cfg(feature = "nightly_simd")]
        assert_eq!(AVX512::<DTypeStrategy>::LANE_SIZE_64, 8);
        assert_eq!(NEON::<DTypeStrategy>::LANE_SIZE_64, 2);
    }
}
