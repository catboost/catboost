//! Middleware that detects when a response future or response body is
//! dropped before completion.
//!
//! HTTP services typically learn nothing when a client disconnects
//! mid-request. This middleware installs drop guards so that premature
//! termination is observable: logs, metrics, cleanup.
//!
//! Two events are distinguished:
//!
//! * **Future drop**: the response future was dropped before the inner
//!   service produced any response.
//! * **Body drop**: the response body was dropped before reaching
//!   [`is_end_stream`](http_body::Body::is_end_stream).
//!
//! # Example: bridge to [`trace::OnFailure`](crate::trace::OnFailure)
//!
//! With the `trace` feature, [`EarlyDropsAsFailures`] wraps any
//! [`OnFailure<DroppedFailure>`](crate::trace::OnFailure) and routes both
//! events through it. Place this layer inside [`TraceLayer`](crate::trace::TraceLayer)
//! so that the emitted events inherit the request span.
//!
//! ```
//! # #[cfg(feature = "trace")] {
//! use tower_http::on_early_drop::{OnEarlyDropLayer, EarlyDropsAsFailures};
//! use tower_http::trace::DefaultOnFailure;
//!
//! let layer = OnEarlyDropLayer::new(
//!     EarlyDropsAsFailures::new(DefaultOnFailure::default()),
//! );
//! # }
//! ```
//!
//! Use [`OnEarlyDropLayer::builder`] to place [`EarlyDropsAsFailures`] in a
//! single slot (track only body or only future drops) or to install plain
//! closures.
//!
//! # Example: builder with direct callbacks
//!
//! Use [`OnEarlyDropLayer::builder`] to install closures in either or both
//! slots. The future-drop closure is a factory: outer closure runs at
//! request time, inner closure fires on drop. The body-drop closure uses a
//! three-level chain via [`OnBodyDropFn`]: outer at request time, middle at
//! response-ready time, inner on drop.
//!
//! ```
//! use http::Request;
//! use tower_http::on_early_drop::{OnBodyDropFn, OnEarlyDropLayer};
//!
//! let layer = OnEarlyDropLayer::builder()
//!     .on_future_drop(|req: &Request<()>| {
//!         let uri = req.uri().clone();
//!         move || eprintln!("future dropped for {}", uri)
//!     })
//!     .on_body_drop(OnBodyDropFn::new(|req: &Request<()>| {
//!         let uri = req.uri().clone();
//!         move |parts: &http::response::Parts| {
//!             let status = parts.status;
//!             move || eprintln!("body dropped for {} status {}", uri, status)
//!         }
//!     }));
//! ```
//!
//! Chain just one of the two methods to hook only that event; the other
//! slot stays no-op.
//!
//! # Panics in callbacks
//!
//! Callbacks fire from [`Drop`]. Panicking during a drop that occurs while
//! another panic is unwinding aborts the process. Closures and custom
//! [`OnDropCallback`] implementations must not panic.
//!
//! # Standalone guard
//!
//! [`OnEarlyDropGuard`] is usable on its own to detect early drop of
//! arbitrary scopes.
//!
//! ```
//! use tower_http::on_early_drop::OnEarlyDropGuard;
//!
//! let mut guard = OnEarlyDropGuard::new(|| {
//!     eprintln!("scope exited early");
//! });
//! // ... work that might return early ...
//! guard.completed();
//! ```

mod body;
mod failure;
mod future;
mod guard;
mod layer;
mod service;
mod traits;

#[cfg(feature = "trace")]
mod early_drops_as_failures;

pub use self::{
    body::OnEarlyDropBody,
    failure::{BodyDropped, DroppedFailure, FutureDropped},
    future::OnEarlyDropFuture,
    guard::OnEarlyDropGuard,
    layer::OnEarlyDropLayer,
    service::OnEarlyDropService,
    traits::{NoopDropCallback, OnBodyDrop, OnBodyDropFn, OnDropCallback, OnFutureDrop},
};

#[cfg(feature = "trace")]
pub use self::early_drops_as_failures::{
    BodyDropFailureCallback, EarlyDropsAsFailures, FutureDropFailureCallback,
    PreResponseBodyDropCallback,
};
