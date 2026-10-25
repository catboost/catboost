//! This module contains mzstd-specific types for async-compression.
//!
//! The parameter types mirror `zstd::params` where the two backends overlap, so switching between
//! them only needs a type change.

use compression_core::Level;
use mzstd::{
    CompressionParams, DecodeLimits, DecoderConfig, EncoderConfig, FrameMode, LdmParams,
    LongDistanceMatching,
};
use std::convert::TryFrom;

/// A compression parameter for the `mzstd` backend. This is a stable wrapper around mzstd's own
/// encoder configuration, to abstract over different versions of the mzstd library.
///
/// As with the `zstd` backend, a value of `0` selects the default for the parameter, and the last
/// value given for a parameter wins.
///
/// See the [zstd documentation](https://facebook.github.io/zstd/zstd_manual.html) for more
/// information on these parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CParameter(CParameterInner);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CParameterInner {
    WindowLog(u32),
    HashLog(u32),
    ChainLog(u32),
    SearchLog(u32),
    MinMatch(u32),
    TargetLength(u32),
    EnableLongDistanceMatching(bool),
    LdmHashLog(u32),
    LdmMinMatch(u32),
    LdmBucketSizeLog(u32),
    LdmHashRateLog(u32),
    ContentSizeFlag(bool),
    ChecksumFlag(bool),
}

impl CParameter {
    pub fn quality(level: Level) -> i32 {
        // NOTE: zstd's "fastest" level is -131072 which can create outputs larger than inputs.
        // Like the `zstd` backend, this chooses a "fastest" level which has a more-or-less
        // equivalent compression ratio to gzip's fastest mode. Precise levels may still go
        // negative.
        const OUR_FASTEST: i32 = 1;

        match level {
            Level::Fastest => OUR_FASTEST,
            Level::Best => mzstd::Level::MAX.get(),
            Level::Precise(quality) => {
                quality.clamp(mzstd::Level::MIN.get(), mzstd::Level::MAX.get())
            }
            _ => mzstd::Level::DEFAULT.get(),
        }
    }

    /// Window size in bytes (as a power of two)
    pub fn window_log(value: u32) -> Self {
        Self(CParameterInner::WindowLog(value))
    }

    /// Size of the initial probe table in 4-byte entries (as a power of two)
    pub fn hash_log(value: u32) -> Self {
        Self(CParameterInner::HashLog(value))
    }

    /// Size of the multi-probe table in 4-byte entries (as a power of two)
    pub fn chain_log(value: u32) -> Self {
        Self(CParameterInner::ChainLog(value))
    }

    /// Number of search attempts (as a power of two)
    pub fn search_log(value: u32) -> Self {
        Self(CParameterInner::SearchLog(value))
    }

    /// Minimum size of matches searched for
    pub fn min_match(value: u32) -> Self {
        Self(CParameterInner::MinMatch(value))
    }

    /// Strategy-dependent length modifier
    pub fn target_length(value: u32) -> Self {
        Self(CParameterInner::TargetLength(value))
    }

    /// Enable long-distance matching mode to look for and emit long-distance references.
    ///
    /// This increases the default window size.
    pub fn enable_long_distance_matching(value: bool) -> Self {
        Self(CParameterInner::EnableLongDistanceMatching(value))
    }

    /// Size of the long-distance matching table (as a power of two)
    ///
    /// Only used when long-distance matching is explicitly enabled.
    pub fn ldm_hash_log(value: u32) -> Self {
        Self(CParameterInner::LdmHashLog(value))
    }

    /// Minimum size of long-distance matches searched for
    ///
    /// Only used when long-distance matching is explicitly enabled.
    pub fn ldm_min_match(value: u32) -> Self {
        Self(CParameterInner::LdmMinMatch(value))
    }

    /// Size of each bucket in the LDM hash table for collision resolution (as a power of two)
    ///
    /// Only used when long-distance matching is explicitly enabled.
    pub fn ldm_bucket_size_log(value: u32) -> Self {
        Self(CParameterInner::LdmBucketSizeLog(value))
    }

    /// Frequency of using the LDM hash table (as a power of two)
    ///
    /// Only used when long-distance matching is explicitly enabled.
    pub fn ldm_hash_rate_log(value: u32) -> Self {
        Self(CParameterInner::LdmHashRateLog(value))
    }

    /// Emit the size of the content (default: true).
    pub fn content_size_flag(value: bool) -> Self {
        Self(CParameterInner::ContentSizeFlag(value))
    }

    /// Emit a checksum (default: false).
    pub fn checksum_flag(value: bool) -> Self {
        Self(CParameterInner::ChecksumFlag(value))
    }
}

/// Builds the encoder configuration for `level` and `params`.
///
/// Levels outside the supported range are clamped, as zstd does. Parameter values are passed on
/// unchecked, so out-of-range values are reported when the compressor is built. As in zstd, mzstd
/// stores `0` as unset, which restores the parameter's default.
pub(crate) fn encoder_config(level: i32, params: &[CParameter]) -> EncoderConfig {
    let level = level.clamp(mzstd::Level::MIN.get(), mzstd::Level::MAX.get());
    // The clamp above keeps `level` inside the range `Level` accepts.
    let mut config = EncoderConfig::new().with_level(mzstd::Level::try_from(level).unwrap());
    let mut compression = CompressionParams::default();
    let mut ldm = LdmParams::default();
    let mut ldm_enabled = None;

    for param in params {
        match param.0 {
            CParameterInner::WindowLog(value) => compression = compression.with_window_log(value),
            CParameterInner::HashLog(value) => compression = compression.with_hash_log(value),
            CParameterInner::ChainLog(value) => compression = compression.with_chain_log(value),
            CParameterInner::SearchLog(value) => compression = compression.with_search_log(value),
            CParameterInner::MinMatch(value) => compression = compression.with_min_match(value),
            CParameterInner::TargetLength(value) => {
                compression = compression.with_target_length(value)
            }
            CParameterInner::EnableLongDistanceMatching(value) => ldm_enabled = Some(value),
            CParameterInner::LdmHashLog(value) => ldm = ldm.with_hash_log(value),
            CParameterInner::LdmMinMatch(value) => ldm = ldm.with_min_match(value),
            CParameterInner::LdmBucketSizeLog(value) => ldm = ldm.with_bucket_size_log(value),
            CParameterInner::LdmHashRateLog(value) => ldm = ldm.with_hash_rate_log(value),
            CParameterInner::ContentSizeFlag(value) => config = config.with_content_size(value),
            CParameterInner::ChecksumFlag(value) => config = config.with_checksum(value),
        }
    }

    // The LDM parameters only apply when long-distance matching is explicitly enabled, which may
    // come after them.
    let long_distance = match ldm_enabled {
        None => LongDistanceMatching::Auto,
        Some(false) => LongDistanceMatching::Disabled,
        Some(true) => LongDistanceMatching::Enabled(ldm),
    };

    config
        .with_params(compression)
        .with_long_distance_matching(long_distance)
}

/// A decompression parameter for the `mzstd` backend. This is a stable wrapper around mzstd's own
/// decoder configuration, to abstract over different versions of the mzstd library.
///
/// See the [zstd documentation](https://facebook.github.io/zstd/zstd_manual.html) for more
/// information on these parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DParameter(DParameterInner);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DParameterInner {
    WindowLogMax(u32),
}

impl DParameter {
    /// Maximum window size in bytes (as a power of two)
    pub fn window_log_max(value: u32) -> Self {
        Self(DParameterInner::WindowLogMax(value))
    }
}

/// Builds the decoder configuration for `params`.
///
/// Decoding stops after one frame, as the `zstd` backend does; further frames are read as
/// separate members.
///
/// # Panics
///
/// Panics if a parameter is out of range, as the `zstd` backend does.
pub(crate) fn decoder_config(params: &[DParameter]) -> DecoderConfig {
    let mut limits = DecodeLimits::DEFAULT;
    for param in params {
        match param.0 {
            DParameterInner::WindowLogMax(0) => limits = DecodeLimits::DEFAULT,
            DParameterInner::WindowLogMax(value) => {
                let log = u8::try_from(value).unwrap_or(u8::MAX);
                limits = limits.with_max_window_log(log).unwrap();
            }
        }
    }

    DecoderConfig::DEFAULT
        .with_frame_mode(FrameMode::Single)
        .with_limits(limits)
}
