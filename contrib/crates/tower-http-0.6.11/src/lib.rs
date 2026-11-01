//! `async fn(HttpRequest) -> Result<HttpResponse, Error>`
//!
//! # Overview
//!
//! tower-http is a library that provides HTTP-specific middleware and utilities built on top of
//! [tower].
//!
//! All middleware uses the [http] and [http-body] crates as the HTTP abstractions. That means
//! they're compatible with any library or framework that also uses those crates, such as
//! [hyper], [tonic], and [warp].
//!
//! # Example server
//!
//! This example shows how to apply middleware from `tower-http` to a [`Service`] and then run
//! that service using [hyper] and [hyper-util].
//!
//! ```rust,no_run
//! use tower_http::{
//!     compression::CompressionLayer,
//!     trace::TraceLayer,
//! };
//! use tower::{ServiceBuilder, service_fn, BoxError};
//! use http::{Request, Response};
//! use std::net::{SocketAddr, Ipv6Addr};
//! use bytes::Bytes;
//! use http_body_util::Full;
//! use hyper_util::rt::{TokioIo, TokioExecutor};
//! use hyper_util::service::TowerToHyperService;
//!
//! # async fn handler(_: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, BoxError> {
//! #     Ok(Response::new(Full::new(Bytes::from("hello"))))
//! # }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), BoxError> {
//!     let addr = SocketAddr::from((Ipv6Addr::LOCALHOST, 8000));
//!     let listener = tokio::net::TcpListener::bind(addr).await?;
//!     println!("Listening on http://{}", addr);
//!
//!     loop {
//!         let (socket, _) = listener.accept().await?;
//!
//!         tokio::spawn(async move {
//!             let socket = TokioIo::new(socket);
//!
//!             // Build our middleware stack
//!             let service = ServiceBuilder::new()
//!                 .layer(TraceLayer::new_for_http())
//!                 .layer(CompressionLayer::new())
//!                 .service_fn(handler);
//!
//!             // Wrap it for hyper
//!             let hyper_service = TowerToHyperService::new(service);
//!
//!             if let Err(err) = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
//!                 .serve_connection(socket, hyper_service)
//!                 .await
//!             {
//!                 eprintln!("failed to serve connection: {err:#}");
//!             }
//!         });
//!     }
//! }
//! ```
//!
//! Keep in mind that while this example uses [hyper], tower-http supports any HTTP
//! client/server implementation that uses the [http] and [http-body] crates.
//!
//! # Example client
//!
//! tower-http middleware can also be applied to HTTP clients:
//!
//! ```rust,no_run
//! use tower_http::{
//!     decompression::DecompressionLayer,
//!     set_header::SetRequestHeaderLayer,
//!     trace::TraceLayer,
//!     classify::StatusInRangeAsFailures,
//! };
//! use tower::{ServiceBuilder, Service, ServiceExt};
//! use hyper_util::{rt::TokioExecutor, client::legacy::Client};
//! use http_body_util::Full;
//! use bytes::Bytes;
//! use http::{Request, HeaderValue, header::USER_AGENT};
//!
//! #[tokio::main]
//! async fn main() {
//! let client = Client::builder(TokioExecutor::new()).build_http();
//!     let mut client = ServiceBuilder::new()
//!         // Add tracing and consider server errors and client
//!         // errors as failures.
//!         .layer(TraceLayer::new(
//!             StatusInRangeAsFailures::new(400..=599).into_make_classifier()
//!         ))
//!         // Set a `User-Agent` header on all requests.
//!         .layer(SetRequestHeaderLayer::overriding(
//!             USER_AGENT,
//!             HeaderValue::from_static("tower-http demo")
//!         ))
//!         // Decompress response bodies
//!         .layer(DecompressionLayer::new())
//!         // Wrap a `Client` in our middleware stack.
//!         // This is possible because `Client` implements
//!         // `tower::Service`.
//!         .service(client);
//!
//!     // Make a request
//!     let request = Request::builder()
//!         .uri("http://example.com")
//!         .body(Full::<Bytes>::default())
//!         .unwrap();
//!
//!     let response = client
//!         .ready()
//!         .await
//!         .unwrap()
//!         .call(request)
//!         .await
//!         .unwrap();
//! }
//! ```
//!
//! # Feature Flags
//!
//! All middleware are disabled by default and can be enabled using [cargo features].
//!
//! For example, to enable the [`Trace`] middleware, add the "trace" feature flag in
//! your `Cargo.toml`:
//!
//! ```toml
//! tower-http = { version = "0.1", features = ["trace"] }
//! ```
//!
//! You can use `"full"` to enable everything:
//!
//! ```toml
//! tower-http = { version = "0.1", features = ["full"] }
//! ```
//!
//! # Getting Help
//!
//! If you're new to tower its [guides] might help. In the tower-http repo we also have a [number
//! of examples][examples] showing how to put everything together. You're also welcome to ask in
//! the [`#tower` Discord channel][chat] or open an [issue] with your question.
//!
//! [tower]: https://crates.io/crates/tower
//! [http]: https://crates.io/crates/http
//! [http-body]: https://crates.io/crates/http-body
//! [hyper]: https://crates.io/crates/hyper
//! [hyper-util]: https://crates.io/crates/hyper-util
//! [guides]: https://github.com/tower-rs/tower/tree/master/guides
//! [tonic]: https://crates.io/crates/tonic
//! [warp]: https://crates.io/crates/warp
//! [cargo features]: https://doc.rust-lang.org/cargo/reference/features.html
//! [`AddExtension`]: crate::add_extension::AddExtension
//! [`Service`]: https://docs.rs/tower/latest/tower/trait.Service.html
//! [chat]: https://discord.gg/tokio
//! [issue]: https://github.com/tower-rs/tower-http/issues/new
//! [`Trace`]: crate::trace::Trace
//! [examples]: https://github.com/tower-rs/tower-http/tree/master/examples

#![warn(
    clippy::all,
    clippy::dbg_macro,
    clippy::todo,
    clippy::empty_enums,
    clippy::enum_glob_use,
    clippy::mem_forget,
    clippy::unused_self,
    clippy::filter_map_next,
    clippy::needless_continue,
    clippy::needless_borrow,
    clippy::match_wildcard_for_single_variants,
    clippy::if_let_mutex,
    clippy::await_holding_lock,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::lossy_float_literal,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::fn_params_excessive_bools,
    clippy::exit,
    clippy::inefficient_to_string,
    clippy::linkedlist,
    clippy::macro_use_imports,
    clippy::option_option,
    clippy::verbose_file_reads,
    clippy::unnested_or_patterns,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style,
    missing_docs
)]
#![deny(unreachable_pub)]
#![allow(
    elided_lifetimes_in_paths,
    // TODO: Remove this once the MSRV bumps to 1.42.0 or above.
    clippy::match_like_matches_macro,
    clippy::type_complexity
)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(test, allow(clippy::float_cmp))]

#[macro_use]
pub(crate) mod macros;

#[cfg(test)]
mod test_helpers;

#[cfg(feature = "auth")]
pub mod auth;

#[cfg(feature = "set-header")]
pub mod set_header;

#[cfg(feature = "propagate-header")]
pub mod propagate_header;

#[cfg(any(
    feature = "compression-br",
    feature = "compression-deflate",
    feature = "compression-gzip",
    feature = "compression-zstd",
))]
pub mod compression;

#[cfg(feature = "add-extension")]
pub mod add_extension;

#[cfg(feature = "sensitive-headers")]
pub mod sensitive_headers;

#[cfg(any(
    feature = "decompression-br",
    feature = "decompression-deflate",
    feature = "decompression-gzip",
    feature = "decompression-zstd",
))]
pub mod decompression;

#[cfg(any(
    feature = "compression-br",
    feature = "compression-deflate",
    feature = "compression-gzip",
    feature = "compression-zstd",
    feature = "decompression-br",
    feature = "decompression-deflate",
    feature = "decompression-gzip",
    feature = "decompression-zstd",
    feature = "fs" // Used for serving precompressed static files as well
))]
mod content_encoding;

#[cfg(any(
    feature = "compression-br",
    feature = "compression-deflate",
    feature = "compression-gzip",
    feature = "compression-zstd",
    feature = "decompression-br",
    feature = "decompression-deflate",
    feature = "decompression-gzip",
    feature = "decompression-zstd",
))]
mod compression_utils;

#[cfg(any(
    feature = "compression-br",
    feature = "compression-deflate",
    feature = "compression-gzip",
    feature = "compression-zstd",
    feature = "decompression-br",
    feature = "decompression-deflate",
    feature = "decompression-gzip",
    feature = "decompression-zstd",
))]
pub use compression_utils::CompressionLevel;

#[cfg(feature = "map-response-body")]
pub mod map_response_body;

#[cfg(feature = "map-request-body")]
pub mod map_request_body;

#[cfg(feature = "trace")]
pub mod trace;

#[cfg(feature = "follow-redirect")]
pub mod follow_redirect;

#[cfg(feature = "limit")]
pub mod limit;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "cors")]
pub mod cors;

#[cfg(feature = "request-id")]
pub mod request_id;

#[cfg(feature = "catch-panic")]
pub mod catch_panic;

#[cfg(feature = "set-status")]
pub mod set_status;

#[cfg(feature = "timeout")]
pub mod timeout;

#[cfg(feature = "normalize-path")]
pub mod normalize_path;

#[cfg(feature = "on-early-drop")]
pub mod on_early_drop;

pub mod classify;
pub mod services;

#[cfg(feature = "util")]
mod builder;
#[cfg(feature = "util")]
mod service_ext;

#[cfg(feature = "util")]
#[doc(inline)]
pub use self::{builder::ServiceBuilderExt, service_ext::ServiceExt};

#[cfg(feature = "validate-request")]
pub mod validate_request;

#[cfg(any(
    feature = "catch-panic",
    feature = "decompression-br",
    feature = "decompression-deflate",
    feature = "decompression-gzip",
    feature = "decompression-zstd",
    feature = "fs",
    feature = "limit",
))]
pub mod body;

/// The latency unit used to report latencies by middleware.
#[non_exhaustive]
#[derive(Copy, Clone, Debug)]
pub enum LatencyUnit {
    /// Use seconds.
    Seconds,
    /// Use milliseconds.
    Millis,
    /// Use microseconds.
    Micros,
    /// Use nanoseconds.
    Nanos,
}

/// Alias for a type-erased error type.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
