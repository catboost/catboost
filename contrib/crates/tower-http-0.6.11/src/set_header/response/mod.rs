//! Middleware for setting headers on HTTP responses.
//!
//! This module provides middleware for setting one or more headers on HTTP responses, either with fixed values or values determined dynamically from the response.
//!
//! # Single Header
//!
//! Use [`SetResponseHeaderLayer`] and [`SetResponseHeader`] to set a single header. The header value can be a fixed value or computed dynamically using a closure. See [`crate::set_header::MakeHeaderValue`] for details.
//!
//! ## Example: Fixed Value
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::SetResponseHeaderLayer;
//! use http_body_util::Full;
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let render_html = tower::service_fn(|request: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(request.into_body()))
//! # });
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         SetResponseHeaderLayer::if_not_present(
//!             header::CONTENT_TYPE,
//!             HeaderValue::from_static("text/html"),
//!         )
//!     )
//!     .service(render_html);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//!
//! assert_eq!(response.headers()["content-type"], "text/html");
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Dynamic Value
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::SetResponseHeaderLayer;
//! use bytes::Bytes;
//! use http_body_util::Full;
//! use http_body::Body as _; // for `Body::size_hint`
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let render_html = tower::service_fn(|request: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(Full::from("1234567890")))
//! # });
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         SetResponseHeaderLayer::overriding(
//!             header::CONTENT_LENGTH,
//!             |response: &Response<Full<Bytes>>| {
//!                 if let Some(size) = response.body().size_hint().exact() {
//!                     Some(HeaderValue::from_str(&size.to_string()).unwrap())
//!                 } else {
//!                     None
//!                 }
//!             }
//!         )
//!     )
//!     .service(render_html);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//!
//! assert_eq!(response.headers()["content-length"], "10");
//! # Ok(())
//! # }
//! ```
//!
//! # Multiple Headers
//!
//! Use [`SetMultipleResponseHeadersLayer`] and [`SetMultipleResponseHeader`] to set multiple headers at once. Each header can have a fixed value or be computed dynamically.
//!
//! Note: this layer uses boxing (allocation + dynamic dispatch) to support mixed producer
//! types in a single vec. Stacking multiple [`SetResponseHeaderLayer`] avoids this at the
//! cost of a more complex composed service type.
//!
//! ## Example: Multiple Fixed Values
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::{HeaderMetadata, response::{SetMultipleResponseHeadersLayer}};
//! use http_body_util::Full;
//! use bytes::Bytes;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let render_html = tower::service_fn(|request: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(request.into_body()))
//! # });
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         SetMultipleResponseHeadersLayer::overriding(vec![
//!             (header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into(),
//!             (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")).into(),
//!         ])
//!     )
//!     .service(render_html);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//!
//! assert_eq!(response.headers()["content-type"], "text/html");
//! assert_eq!(response.headers()["cache-control"], "no-cache");
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Multiple Dynamic Values
//!
//! ```
//! use http::{Request, Response, header::{self, HeaderValue}};
//! use tower::{Service, ServiceExt, ServiceBuilder};
//! use tower_http::set_header::{HeaderMetadata, response::{SetMultipleResponseHeadersLayer}};
//! use bytes::Bytes;
//! use http_body_util::Full;
//! use http_body::Body as _; // for `Body::size_hint`
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let render_html = tower::service_fn(|request: Request<Full<Bytes>>| async move {
//! #     Ok::<_, std::convert::Infallible>(Response::new(Full::from("1234567890")))
//! # });
//!
//! let mut svc = ServiceBuilder::new()
//!     .layer(
//!         SetMultipleResponseHeadersLayer::overriding(vec![
//!             (header::CONTENT_LENGTH, |response: &Response<Full<Bytes>>| {
//!                 if let Some(size) = response.body().size_hint().exact() {
//!                     Some(HeaderValue::from_str(&size.to_string()).unwrap())
//!                 } else {
//!                     None
//!                 }
//!             }).into(),
//!         ])
//!     )
//!     .service(render_html);
//!
//! let request = Request::new(Full::default());
//!
//! let response = svc.ready().await?.call(request).await?;
//!
//! assert_eq!(response.headers()["content-length"], "10");
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
//! See [`SetResponseHeaderLayer`], [`SetResponseHeader`], [`SetMultipleResponseHeadersLayer`], and [`SetMultipleResponseHeader`] for more details.

mod multiple_headers;
mod single_header;

pub use multiple_headers::{SetMultipleResponseHeader, SetMultipleResponseHeadersLayer};
pub use single_header::{SetResponseHeader, SetResponseHeaderLayer};
