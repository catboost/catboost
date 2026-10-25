//! Brotli codecs backed by the pure-Rust [`mbrotli`](https://docs.rs/mbrotli) implementation.
//!
//! These produce and consume the same Brotli format as the `brotli` module, and are an
//! alternative backend for it.
//!
//! This module is unstable: it requires both the `brotli-mbrotli` feature and building with
//! `RUSTFLAGS="--cfg async_compression_unstable"`, and it may change in any release. `mbrotli`
//! requires Rust 1.89 or newer.

mod decoder;
mod encoder;
pub mod params;

pub use self::{decoder::MbrotliDecoder, encoder::MbrotliEncoder};
