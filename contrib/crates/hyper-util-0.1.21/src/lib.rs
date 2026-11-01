#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Utilities for working with hyper.
//!
//! This crate is less-stable than [`hyper`](https://docs.rs/hyper). However,
//! does respect Rust's semantic version regarding breaking changes.
//!
//! # Feature flags
//!
//! `hyper-util` relies on feature flags to allow dependents to opt into
//! building particular components. None of these feature flags are enabled by
//! default. You may use the `full` feature flag to enable all of the standard
//! features.
//!
//! Note that the `full` feature flag may introduce superfluous dependencies.
//! You may improve compilation times by enabling the individual features that
//! you need.
//!
//! **Client features**
//!
//! * `client`: Enable [`client`] interfaces.
//! * `client-legacy`: Enable the legacy
//!   [`Client`][crate::client::legacy::Client] implementation.
//! * `client-pool`: Enable [`client::pool`]. This submodule contains
//!   interfaces for constructing connection pools.
//! * `client-proxy`: Enable [`client::proxy`].
//! * `client-proxy-system`: Enable platform-specific system proxy support.
//!
//! **Server features**
//!
//! * `server`: Enable [`server`] interfaces.
//! * `server-auto`: Enable automatic HTTP version.
//! * `server-graceful`: Enable [`server::graceful`] interfaces for gracefully
//!   shutting down a server. See the module-level documentation of
//!   [`server::graceful`] for more information.
//!
//! **Other features**
//!
//! * `service`: Enable [`service`] interfaces for compatibility with tower's
//!   [`Service`][tower_service::Service]. See the module-level documentation
//!   of [`service`] for more information.
//! * `http1`: Enable HTTP/1 support.
//! * `http2`: Enable HTTP/2 support.
//! * `tokio`: Enable [`rt::tokio`] runtime components to integrate with
//!   [`tokio`]. See the module-level documentation of [`rt::tokio`] for more
//!   information.
//! * `tracing`: Enable [`rt::tracing`] runtime components to integrate with
//!   [`tracing`]. See the module-level documentation of [`rt::tracing`] for
//!   more information.
//! * `rt-tracing-exec-force`: A temporary compatibility opt-in feature to
//!   facilitate migrating telemetry from v0.1.20. If set,
//!   [`rt::tokio::TokioExecutor<I>`] will continue to propagate the current
//!   [`tracing::Span`] to spawned tasks. This may be removed in a future
//!   release. Note that this feature is **not** included in the `full`
//!   feature set, and must be explicitly enabled. See
//!   [#322](https://github.com/hyperium/hyper-util/pull/322) and
//!   [#323](https://github.com/hyperium/hyper-util/pull/323) for more
//!   information.

#[cfg(feature = "client")]
pub mod client;
mod common;
pub mod rt;
#[cfg(feature = "server")]
pub mod server;
#[cfg(any(feature = "service", feature = "client-legacy"))]
pub mod service;
