//! Set a single header on the request.
//!
//! See the root [`crate::set_header::request`] module for full documentation and usage examples.
//!
use http::{header::HeaderName, Request, Response};
use std::{
    fmt,
    task::{Context, Poll},
};
use tower_layer::Layer;
use tower_service::Service;

use crate::set_header::{InsertHeaderMode, MakeHeaderValue};

/// Layer that applies [`SetRequestHeader`] which adds a request header.
///
/// See [`SetRequestHeader`] for more details.
pub struct SetRequestHeaderLayer<M> {
    header_name: HeaderName,
    make: M,
    mode: InsertHeaderMode,
}

impl<M> fmt::Debug for SetRequestHeaderLayer<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetRequestHeaderLayer")
            .field("header_name", &self.header_name)
            .field("mode", &self.mode)
            .field("make", &std::any::type_name::<M>())
            .finish()
    }
}

impl<M> SetRequestHeaderLayer<M> {
    /// Create a new [`SetRequestHeaderLayer`].
    ///
    /// If a previous value exists for the same header, it is removed and replaced with the new
    /// header value.
    pub fn overriding(header_name: HeaderName, make: M) -> Self {
        Self::new(header_name, make, InsertHeaderMode::Override)
    }

    /// Create a new [`SetRequestHeaderLayer`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(header_name: HeaderName, make: M) -> Self {
        Self::new(header_name, make, InsertHeaderMode::Append)
    }

    /// Create a new [`SetRequestHeaderLayer`].
    ///
    /// If a previous value exists for the header, the new value is not inserted.
    pub fn if_not_present(header_name: HeaderName, make: M) -> Self {
        Self::new(header_name, make, InsertHeaderMode::IfNotPresent)
    }

    fn new(header_name: HeaderName, make: M, mode: InsertHeaderMode) -> Self {
        Self {
            make,
            header_name,
            mode,
        }
    }
}

impl<S, M> Layer<S> for SetRequestHeaderLayer<M>
where
    M: Clone,
{
    type Service = SetRequestHeader<S, M>;

    fn layer(&self, inner: S) -> Self::Service {
        SetRequestHeader {
            inner,
            header_name: self.header_name.clone(),
            make: self.make.clone(),
            mode: self.mode,
        }
    }
}

impl<M> Clone for SetRequestHeaderLayer<M>
where
    M: Clone,
{
    fn clone(&self) -> Self {
        Self {
            make: self.make.clone(),
            header_name: self.header_name.clone(),
            mode: self.mode,
        }
    }
}

/// Middleware that sets a header on the request.
#[derive(Clone)]
pub struct SetRequestHeader<S, M> {
    inner: S,
    header_name: HeaderName,
    make: M,
    mode: InsertHeaderMode,
}

impl<S, M> SetRequestHeader<S, M> {
    /// Create a new [`SetRequestHeader`].
    ///
    /// If a previous value exists for the same header, it is removed and replaced with the new
    /// header value.
    pub fn overriding(inner: S, header_name: HeaderName, make: M) -> Self {
        Self::new(inner, header_name, make, InsertHeaderMode::Override)
    }

    /// Create a new [`SetRequestHeader`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(inner: S, header_name: HeaderName, make: M) -> Self {
        Self::new(inner, header_name, make, InsertHeaderMode::Append)
    }

    /// Create a new [`SetRequestHeader`].
    ///
    /// If a previous value exists for the header, the new value is not inserted.
    pub fn if_not_present(inner: S, header_name: HeaderName, make: M) -> Self {
        Self::new(inner, header_name, make, InsertHeaderMode::IfNotPresent)
    }

    fn new(inner: S, header_name: HeaderName, make: M, mode: InsertHeaderMode) -> Self {
        Self {
            inner,
            header_name,
            make,
            mode,
        }
    }

    define_inner_service_accessors!();
}

impl<S, M> fmt::Debug for SetRequestHeader<S, M>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetRequestHeader")
            .field("inner", &self.inner)
            .field("header_name", &self.header_name)
            .field("mode", &self.mode)
            .field("make", &std::any::type_name::<M>())
            .finish()
    }
}

impl<ReqBody, ResBody, S, M> Service<Request<ReqBody>> for SetRequestHeader<S, M>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    M: MakeHeaderValue<Request<ReqBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        self.mode.apply(&self.header_name, &mut req, &mut self.make);
        self.inner.call(req)
    }
}
