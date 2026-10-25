//! The strategy that is used to handle the data type.

/// Strategy for signed and unsigned integers.
///
/// There is only one (default) strategy for signed and unsigned integers.
pub struct Int;

/// Strategy for floating point numbers - ignoring NaNs.
///
/// This strategy is available when the `float` or `half` feature is enabled.
///
/// Note that this strategy is the strategy for floats in the
/// [`ArgMinMax`](crate::ArgMinMax) trait.
#[cfg(any(feature = "float", feature = "half"))]
pub struct FloatIgnoreNaN;

/// Strategy for floating point numbers - returning NaNs.
///
/// This strategy is available when the `float` or `half` feature is enabled.
///
/// Note that this strategy is the strategy for floats in the
/// [`NaNArgMinMax`](crate::NaNArgMinMax) trait.
#[cfg(any(feature = "float", feature = "half"))]
pub struct FloatReturnNaN;
