//! This module contains mbrotli-specific types for async-compression.

use compression_core::Level;
use mbrotli::{
    BlockBits, BlockSize, CompressionMode, EncoderConfig, InputSize, Quality, StreamConfig, Window,
};
use std::convert::TryFrom;

/// Brotli compression parameters builder for the `mbrotli` backend. This is a stable wrapper
/// around mbrotli's own encoder configuration, to abstract over different versions of the mbrotli
/// library.
///
/// The builder methods mirror `brotli::params::EncoderParams` where the two backends overlap, so
/// switching between them only needs a type change.
///
/// See the [Brotli documentation](https://www.brotli.org/encode.html#a9a8) for more information on
/// these parameters.
///
/// # Examples
///
/// ```
/// use compression_codecs::mbrotli;
///
/// let params = mbrotli::params::EncoderParams::default()
///     .window_size(12)
///     .text_mode();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderParams {
    config: EncoderConfig,
    input_size: InputSize,
}

impl EncoderParams {
    pub(crate) fn config(&self) -> EncoderConfig {
        self.config
    }

    pub(crate) fn stream(&self) -> StreamConfig {
        StreamConfig::from(self.input_size)
    }

    pub fn quality(mut self, level: Level) -> Self {
        let quality = match level {
            Level::Fastest => Quality::MIN,
            Level::Best => Quality::MAX,
            Level::Precise(quality) => {
                let quality = quality.clamp(Quality::MIN.get().into(), Quality::MAX.get().into());
                // The clamp above keeps `quality` inside `0..=11`, which both conversions accept.
                Quality::try_from(quality as u8).unwrap()
            }
            _ => Quality::default(),
        };
        self.config = self.config.with_quality(quality);
        self
    }

    /// Sets window size in bytes (as a power of two).
    ///
    /// Used as Brotli's `lgwin` parameter.
    ///
    /// `window_size` is clamped to `10 <= window_size <= 24`, except `0` which selects the default
    /// window, as the `brotli` backend does.
    pub fn window_size(mut self, window_size: i32) -> Self {
        let window = if window_size == 0 {
            Window::DEFAULT
        } else {
            let bits = window_size.clamp(Window::MIN_BITS.into(), Window::MAX_STANDARD_BITS.into());
            // The clamp above keeps `bits` inside the standard window range.
            Window::standard(bits as u8).unwrap()
        };
        self.config = self.config.with_window(window);
        self
    }

    /// Sets input block size in bytes (as a power of two).
    ///
    /// Used as Brotli's `lgblock` parameter.
    ///
    /// `block_size` is clamped to `16 <= block_size <= 24`.
    pub fn block_size(mut self, block_size: i32) -> Self {
        let bits = block_size.clamp(BlockBits::MIN.get().into(), BlockBits::MAX.get().into());
        // The clamp above keeps `bits` inside the range `BlockBits` accepts.
        let bits = BlockBits::try_from(bits as u8).unwrap();
        self.config = self.config.with_block_size(BlockSize::Bits(bits));
        self
    }

    /// Sets hint for size of data to be compressed.
    pub fn size_hint(mut self, size_hint: usize) -> Self {
        self.input_size = InputSize::Exact(size_hint as u64);
        self
    }

    /// Sets encoder to text mode.
    ///
    /// If input data is known to be UTF-8 text, this allows the compressor to make assumptions and
    /// optimizations.
    ///
    /// Used as Brotli's `mode` parameter.
    pub fn text_mode(mut self) -> Self {
        self.config = self.config.with_mode(CompressionMode::Text);
        self
    }

    /// Sets encoder to font mode.
    ///
    /// Tunes the compressor for WOFF 2.0 font data.
    ///
    /// Used as Brotli's `mode` parameter.
    pub fn font_mode(mut self) -> Self {
        self.config = self.config.with_mode(CompressionMode::Font);
        self
    }
}
