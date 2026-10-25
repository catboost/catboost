//! ChaCha variant-specific configurations.

mod sealed {
    pub trait Sealed {}
}

/// A trait that distinguishes some ChaCha variants. Contains configurations
/// for "Legacy" DJB variant and the IETF variant.
pub trait Variant: sealed::Sealed {
    /// The counter's type.
    #[cfg(not(feature = "cipher"))]
    type Counter: Copy;

    /// The counter's type.
    #[cfg(feature = "cipher")]
    type Counter: cipher::StreamCipherCounter;

    /// Takes a slice of `state[12..NONCE_INDEX]` to convert it into
    /// `Self::Counter`.
    fn get_block_pos(row: &[u32]) -> Self::Counter;

    /// Breaks down the `Self::Counter` type into a u32 array for setting the
    /// block pos.
    fn set_block_pos(row: &mut [u32], pos: Self::Counter);

    /// A helper method for calculating the remaining blocks using these types
    fn remaining_blocks(block_pos: Self::Counter) -> Option<usize>;
}

/// IETF ChaCha configuration to use a 32-bit counter and 96-bit nonce.
#[derive(Clone, Copy, Debug)]
pub enum Ietf {}

impl sealed::Sealed for Ietf {}

impl Variant for Ietf {
    type Counter = u32;

    #[inline(always)]
    fn get_block_pos(row: &[u32]) -> u32 {
        row[0]
    }

    #[inline(always)]
    fn set_block_pos(row: &mut [u32], pos: u32) {
        row[0] = pos;
    }

    #[inline(always)]
    fn remaining_blocks(block_pos: u32) -> Option<usize> {
        let remaining = u32::MAX - block_pos;
        remaining.try_into().ok()
    }
}

/// DJB variant specific features: 64-bit counter and 64-bit nonce.
#[cfg(any(feature = "legacy", feature = "rng"))]
#[derive(Clone, Copy, Debug)]
pub enum Legacy {}

#[cfg(any(feature = "legacy", feature = "rng"))]
impl sealed::Sealed for Legacy {}

#[cfg(any(feature = "legacy", feature = "rng"))]
impl Variant for Legacy {
    type Counter = u64;

    #[inline(always)]
    fn get_block_pos(row: &[u32]) -> u64 {
        (u64::from(row[1]) << 32) | u64::from(row[0])
    }

    #[inline(always)]
    fn set_block_pos(row: &mut [u32], pos: u64) {
        row[0] = (pos & 0xFFFF_FFFF) as u32;
        row[1] = (pos >> 32) as u32;
    }

    #[inline(always)]
    fn remaining_blocks(block_pos: u64) -> Option<usize> {
        let remaining = u64::MAX - block_pos;
        remaining.try_into().ok()
    }
}
