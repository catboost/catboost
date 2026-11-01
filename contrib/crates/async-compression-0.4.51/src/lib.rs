//! Adaptors between compression crates and Rust's modern asynchronous IO types.
//!

//! # Feature Organization
//!
//! This crate is divided up along two axes, which can each be individually selected via Cargo
//! features.
//!
//! All features are disabled by default, you should enable just the ones you need from the lists
//! below.
//!
//! If you want to pull in everything there are three group features defined:
//!

//!  Feature | Does
//! ---------|------
//!  `all`   | Activates all implementations and algorithms.
//!  `all-implementations` | Activates all implementations, needs to be paired with a selection of algorithms
//!  `all-algorithms` | Activates all algorithms, needs to be paired with a selection of implementations
//!

//! ## IO implementation
//!
//! The first division is which underlying asynchronous IO trait will be wrapped, these are
//! available as separate features that have corresponding top-level modules:
//!

//!  Feature | Type
//! ---------|------
// TODO: Kill rustfmt on this section, `#![rustfmt::skip::attributes(cfg_attr)]` should do it, but
// that's unstable
#![allow(unexpected_cfgs)]
#![cfg_attr(
    feature = "futures-io",
    doc = "[`futures-io`](futures_io) | [`futures::io::AsyncBufRead`](futures_io::AsyncBufRead), [`futures::io::AsyncWrite`](futures_io::AsyncWrite)"
)]
#![cfg_attr(
    not(feature = "futures-io"),
    doc = "`futures-io` (*inactive*) | `futures::io::AsyncBufRead`, `futures::io::AsyncWrite`"
)]
#![cfg_attr(
    feature = "tokio",
    doc = "[`tokio`] | [`tokio::io::AsyncBufRead`](::tokio::io::AsyncBufRead), [`tokio::io::AsyncWrite`](::tokio::io::AsyncWrite)"
)]
#![cfg_attr(
    not(feature = "tokio"),
    doc = "`tokio` (*inactive*) | `tokio::io::AsyncBufRead`, `tokio::io::AsyncWrite`"
)]
//!

//! ## Compression algorithm
//!
//! The second division is which compression schemes to support, there are currently a few
//! available choices, these determine which types will be available inside the above modules:
//!

//!  Feature | Types
//! ---------|------
#![cfg_attr(
    feature = "brotli",
    doc = "`brotli` | [`BrotliEncoder`](?search=BrotliEncoder), [`BrotliDecoder`](?search=BrotliDecoder)"
)]
#![cfg_attr(
    not(feature = "brotli"),
    doc = "`brotli` (*inactive*) | `BrotliEncoder`, `BrotliDecoder`"
)]
#![cfg_attr(
    all(feature = "brotli-mbrotli", async_compression_unstable),
    doc = "`brotli-mbrotli` (*unstable*) | [`MbrotliEncoder`](?search=MbrotliEncoder), [`MbrotliDecoder`](?search=MbrotliDecoder)"
)]
#![cfg_attr(
    not(all(feature = "brotli-mbrotli", async_compression_unstable)),
    doc = "`brotli-mbrotli` (*unstable*, *inactive*) | `MbrotliEncoder`, `MbrotliDecoder`"
)]
#![cfg_attr(
    feature = "bzip2",
    doc = "`bzip2` | [`BzEncoder`](?search=BzEncoder), [`BzDecoder`](?search=BzDecoder)"
)]
#![cfg_attr(
    not(feature = "bzip2"),
    doc = "`bzip2` (*inactive*) | `BzEncoder`, `BzDecoder`"
)]
#![cfg_attr(
    feature = "deflate",
    doc = "`deflate` | [`DeflateEncoder`](?search=DeflateEncoder), [`DeflateDecoder`](?search=DeflateDecoder)"
)]
#![cfg_attr(
    not(feature = "deflate"),
    doc = "`deflate` (*inactive*) | `DeflateEncoder`, `DeflateDecoder`"
)]
#![cfg_attr(
    feature = "gzip",
    doc = "`gzip` | [`GzipEncoder`](?search=GzipEncoder), [`GzipDecoder`](?search=GzipDecoder)"
)]
#![cfg_attr(
    not(feature = "gzip"),
    doc = "`gzip` (*inactive*) | `GzipEncoder`, `GzipDecoder`"
)]
#![cfg_attr(
    feature = "lz4",
    doc = "`lz4` | [`Lz4Encoder`](?search=Lz4Encoder), [`Lz4Decoder`](?search=Lz4Decoder)"
)]
#![cfg_attr(
    not(feature = "lz4"),
    doc = "`lz4` (*inactive*) | `Lz4Encoder`, `Lz4Decoder`"
)]
#![cfg_attr(
    feature = "lzma",
    doc = "`lzma` | [`LzmaEncoder`](?search=LzmaEncoder), [`LzmaDecoder`](?search=LzmaDecoder)"
)]
#![cfg_attr(
    not(feature = "lzma"),
    doc = "`lzma` (*inactive*) | `LzmaEncoder`, `LzmaDecoder`"
)]
#![cfg_attr(
    feature = "xz",
    doc = "`xz` | [`XzEncoder`](?search=XzEncoder), [`XzDecoder`](?search=XzDecoder)"
)]
#![cfg_attr(
    not(feature = "xz"),
    doc = "`xz` (*inactive*) | `XzEncoder`, `XzDecoder`"
)]
#![cfg_attr(
    feature = "zlib",
    doc = "`zlib` | [`ZlibEncoder`](?search=ZlibEncoder), [`ZlibDecoder`](?search=ZlibDecoder)"
)]
#![cfg_attr(
    not(feature = "zlib"),
    doc = "`zlib` (*inactive*) | `ZlibEncoder`, `ZlibDecoder`"
)]
#![cfg_attr(
    feature = "zstd",
    doc = "`zstd` | [`ZstdEncoder`](?search=ZstdEncoder), [`ZstdDecoder`](?search=ZstdDecoder)"
)]
#![cfg_attr(
    not(feature = "zstd"),
    doc = "`zstd` (*inactive*) | `ZstdEncoder`, `ZstdDecoder`"
)]
#![cfg_attr(
    all(feature = "zstd-mzstd", async_compression_unstable),
    doc = "`zstd-mzstd` (*unstable*) | [`MzstdEncoder`](?search=MzstdEncoder), [`MzstdDecoder`](?search=MzstdDecoder)"
)]
#![cfg_attr(
    not(all(feature = "zstd-mzstd", async_compression_unstable)),
    doc = "`zstd-mzstd` (*unstable*, *inactive*) | `MzstdEncoder`, `MzstdDecoder`"
)]
#![cfg_attr(
    feature = "deflate64",
    doc = "`deflate64` | (encoder not implemented), [`Deflate64Decoder`](?search=Deflate64Decoder)"
)]
#![cfg_attr(
    not(feature = "deflate64"),
    doc = "`deflate64` (*inactive*) | (encoder not implemented), `Deflate64Decoder`"
)]
//!

//! ## Unstable features
//!
//! Some features are not yet covered by this crate's stability guarantees. They only take effect
//! when the crate is also built with `--cfg async_compression_unstable`, for example through
//! `RUSTFLAGS="--cfg async_compression_unstable"`, and they may change or be removed in any
//! release:
//!
//! - `brotli-mbrotli` adds `MbrotliEncoder` and `MbrotliDecoder`, a Brotli implementation backed by
//!   the pure-Rust [`mbrotli`](https://docs.rs/mbrotli) crate. It is an alternative to the `brotli`
//!   feature and produces the same format. `mbrotli` requires Rust 1.89 or newer.
//! - `zstd-mzstd` adds `MzstdEncoder` and `MzstdDecoder`, a Zstandard implementation backed by the
//!   pure-Rust [`mzstd`](https://docs.rs/mzstd) crate. It is an alternative to the `zstd` feature
//!   and produces the same format. `mzstd` requires Rust 1.89 or newer.
//!

//! ## Multi-thread support
//! The `xz` compression algorithm supports multi-threaded compression and decompression.
//! Enable the `xz-parallel` feature to enable multi-threading support.
//!

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(all), allow(unused))]

#[macro_use]
mod macros;

/// Generic, async runtime agonistc implementation of en/decoders
mod generic;

#[cfg(feature = "futures-io")]
pub mod futures;
#[cfg(feature = "tokio")]
pub mod tokio;

pub use compression_codecs as codecs;
pub use compression_core as core;

pub use core::Level;

#[cfg(feature = "zstd")]
pub use codecs::zstd::params as zstd;

#[cfg(feature = "lz4")]
pub use codecs::lz4::params as lz4;

#[cfg(feature = "brotli")]
pub use codecs::brotli::params as brotli;

#[cfg(all(feature = "brotli-mbrotli", async_compression_unstable))]
pub use codecs::mbrotli::params as mbrotli;

#[cfg(all(feature = "zstd-mzstd", async_compression_unstable))]
pub use codecs::mzstd::params as mzstd;
