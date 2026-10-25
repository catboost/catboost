//! Middleware for setting headers on HTTP requests.
//!
//! This module provides middleware for setting one or more headers on HTTP requests, either with fixed values or values determined dynamically from the request.
//!
//! # Single Header
//!
//! Use [`SetRequestHeaderLayer`] and [`SetRequestHeader`] to set a single header. The header value can be a fixed value or computed dynamically using a closure. See [`crate::set_header::MakeHeaderValue`] for details.
//!
//! ## Example: Fixed Value
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::SetRequestHeaderLayer;
//! use http_body_util::Full;
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let http_client = tower::service_fn(|_: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(Full::<Bytes>::default()))
//! # });
//! #
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         // Layer that sets `User-Agent: my very cool app` on requests.
//!         //
//!         // `if_not_present` will only insert the header if it does not already
//!         // have a value.
//!         SetRequestHeaderLayer::if_not_present(
//!             header::USER_AGENT,
//!             HeaderValue::from_static("my very cool app"),
//!         )
//!     )
//!     .service(http_client);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//! #
//! # Ok(())
//! # }
//! ```
//!
//! Setting a header based on a value determined dynamically from the request:
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::SetRequestHeaderLayer;
//! use bytes::Bytes;
//! use http_body_util::Full;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let http_client = tower::service_fn(|_: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(Full::<Bytes>::default()))
//! # });
//! fn date_header_value() -> HeaderValue {
//!     // ...
//!     # HeaderValue::from_static("now")
//! }
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         // Layer that sets `Date` to the current date and time.
//!         //
//!         // `overriding` will insert the header and override any previous values it
//!         // may have.
//!         SetRequestHeaderLayer::overriding(
//!             header::DATE,
//!             |request: &Request<Full<Bytes>>| {
//!                 Some(date_header_value())
//!             }
//!         )
//!     )
//!     .service(http_client);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//! #
//! # Ok(())
//! # }
//! ```
//!
//! # Multiple Headers
//!
//! Use [`SetMultipleRequestHeadersLayer`] and [`SetMultipleRequestHeader`] to set multiple headers at once. Each header can have a fixed value or be computed dynamically.
//!
//! Note: this layer uses boxing (allocation + dynamic dispatch) to support mixed producer
//! types in a single `vec`. Stacking multiple [`SetRequestHeaderLayer`] instances avoids this at the
//! cost of a more complex composed service type.
//!
//! ## Example: Multiple Dynamic Values
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::{HeaderMetadata, request::{SetMultipleRequestHeadersLayer}};
//! use bytes::Bytes;
//! use http_body_util::Full;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let http_client = tower::service_fn(|_: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(Full::<Bytes>::default()))
//! # });
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         SetMultipleRequestHeadersLayer::overriding(vec![
//!             (header::DATE, |_: &Request<Full<Bytes>>| {
//!                 Some(HeaderValue::from_static("now"))
//!             }).into(),
//!         ])
//!     )
//!     .service(tower::service_fn(|req: Request<Full<Bytes>>| async move {
//!         assert_eq!(req.headers()["date"], "now");
//!         Ok::<_, std::convert::Infallible>(Response::new(Full::<Bytes>::default()))
//!     }));
//!
//! let request = Request::new(Full::default());
//!
//! let _response = svc.ready().await?.call(request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Modes
//!
//! - `overriding`: If a previous value exists for the same header, it is removed and replaced with the new value.
//! - `appending`: The new header is always added, preserving any existing values. If previous values exist, the header will have multiple values.
//! - `if_not_present`: If a previous value exists for the header, the new value is not inserted.
//!
//! See [`SetRequestHeaderLayer`], [`SetRequestHeader`], [`SetMultipleRequestHeadersLayer`], and [`SetMultipleRequestHeader`] for more details.

mod multiple_headers;
mod single_header;

pub use multiple_headers::{SetMultipleRequestHeader, SetMultipleRequestHeadersLayer};
pub use single_header::{SetRequestHeader, SetRequestHeaderLayer};
