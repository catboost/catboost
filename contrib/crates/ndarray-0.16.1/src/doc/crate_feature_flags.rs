//! Crate Feature Flags
//!
//! The following crate feature flags are available. They are configured in your
//! `Cargo.toml` where the dependency on `ndarray` is defined.
//!
//! ## `std`
//!   - Rust standard library (enabled by default)
//!   - This crate can be used without the standard library by disabling the
//!     default `std` feature. To do so, use `default-features = false` in
//!     your `Cargo.toml`.
//!   - The `geomspace` `linspace` `logspace` `range` `std` `var` `var_axis`
//!     and `std_axis` methods are only available when `std` is enabled.
//!
//! ## `serde`
//!   - Enables serialization support for serde 1.x
//!
//! ## `rayon`
//!   - Enables parallel iterators, parallelized methods, the [`parallel`] module and [`par_azip!`].
//!   - Implies std
//!
//! ## `approx`
//!   - Enables implementations of traits of the [`approx`] crate.
//!
//! ## `blas`
//!   - Enable transparent BLAS support for matrix multiplication.
//!     Uses ``blas-src`` for pluggable backend, which needs to be configured
//!     separately (see the README).
//!
//! ## `matrixmultiply-threading`
//!   - Enable the ``threading`` feature in the matrixmultiply package
//!
//! [`parallel`]: crate::parallel

#[cfg(doc)]
use crate::parallel::par_azip;
