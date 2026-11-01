//! Zstandard codecs backed by the pure-Rust [`mzstd`](https://docs.rs/mzstd) implementation.
//!
//! These produce and consume the same Zstandard format as the `zstd` module, and are an
//! alternative backend for it.
//!
//! This module is unstable: it requires both the `zstd-mzstd` feature and building with
//! `RUSTFLAGS="--cfg async_compression_unstable"`, and it may change in any release. `mzstd`
//! requires Rust 1.89 or newer.

mod decoder;
mod encoder;
pub mod params;

pub use self::{decoder::MzstdDecoder, encoder::MzstdEncoder};
