//! Set a single header on the response.
//!
//! See the root [`crate::set_header::response`] module for full documentation and usage examples.
//!
use http::{header::HeaderName, Request, Response};
use pin_project_lite::pin_project;
use std::{
    fmt,
    future::Future,
    pin::Pin,
    task::{ready, Context, Poll},
};
use tower_layer::Layer;
use tower_service::Service;

use crate::set_header::{InsertHeaderMode, MakeHeaderValue};

/// Layer that applies [`SetResponseHeader`] which adds a response header.
///
/// See [`SetResponseHeader`] for more details.
pub struct SetResponseHeaderLayer<M> {
    header_name: HeaderName,
    make: M,
    mode: InsertHeaderMode,
}

impl<M> fmt::Debug for SetResponseHeaderLayer<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetResponseHeaderLayer")
            .field("header_name", &self.header_name)
            .field("mode", &self.mode)
            .field("make", &std::any::type_name::<M>())
            .finish()
    }
}

impl<M> SetResponseHeaderLayer<M> {
    /// Create a new [`SetResponseHeaderLayer`].
    ///
    /// If a previous value exists for the same header, it is removed and replaced with the new
    /// header value.
    pub fn overriding(header_name: HeaderName, make: M) -> Self {
        Self::new(header_name, make, InsertHeaderMode::Override)
    }

    /// Create a new [`SetResponseHeaderLayer`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(header_name: HeaderName, make: M) -> Self {
        Self::new(header_name, make, InsertHeaderMode::Append)
    }

    /// Create a new [`SetResponseHeaderLayer`].
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

impl<S, M> Layer<S> for SetResponseHeaderLayer<M>
where
    M: Clone,
{
    type Service = SetResponseHeader<S, M>;

    fn layer(&self, inner: S) -> Self::Service {
        SetResponseHeader {
            inner,
            header_name: self.header_name.clone(),
            make: self.make.clone(),
            mode: self.mode,
        }
    }
}

impl<M> Clone for SetResponseHeaderLayer<M>
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

/// Middleware that sets a header on the response.
#[derive(Clone)]
pub struct SetResponseHeader<S, M> {
    inner: S,
    header_name: HeaderName,
    make: M,
    mode: InsertHeaderMode,
}

impl<S, M> SetResponseHeader<S, M> {
    /// Create a new [`SetResponseHeader`].
    ///
    /// If a previous value exists for the same header, it is removed and replaced with the new
    /// header value.
    pub fn overriding(inner: S, header_name: HeaderName, make: M) -> Self {
        Self::new(inner, header_name, make, InsertHeaderMode::Override)
    }

    /// Create a new [`SetResponseHeader`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(inner: S, header_name: HeaderName, make: M) -> Self {
        Self::new(inner, header_name, make, InsertHeaderMode::Append)
    }

    /// Create a new [`SetResponseHeader`].
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

impl<S, M> fmt::Debug for SetResponseHeader<S, M>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetResponseHeader")
            .field("inner", &self.inner)
            .field("header_name", &self.header_name)
            .field("mode", &self.mode)
            .field("make", &std::any::type_name::<M>())
            .finish()
    }
}

impl<ReqBody, ResBody, S, M> Service<Request<ReqBody>> for SetResponseHeader<S, M>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    M: MakeHeaderValue<Response<ResBody>> + Clone,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = ResponseFuture<S::Future, M>;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        ResponseFuture {
            future: self.inner.call(req),
            header_name: self.header_name.clone(),
            make: self.make.clone(),
            mode: self.mode,
        }
    }
}

pin_project! {
    /// Response future for [`SetResponseHeader`].
    #[derive(Debug)]
    pub struct ResponseFuture<F, M> {
        #[pin]
        future: F,
        header_name: HeaderName,
        make: M,
        mode: InsertHeaderMode,
    }
}

impl<F, ResBody, E, M> Future for ResponseFuture<F, M>
where
    F: Future<Output = Result<Response<ResBody>, E>>,
    M: MakeHeaderValue<Response<ResBody>>,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let mut res = ready!(this.future.poll(cx)?);

        this.mode.apply(this.header_name, &mut res, &mut *this.make);

        Poll::Ready(Ok(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::Body;
    use http::{header, HeaderValue};
    use std::convert::Infallible;
    use tower::{service_fn, ServiceExt};

    #[tokio::test]
    async fn test_override_mode() {
        let svc = SetResponseHeader::overriding(
            service_fn(|_req: Request<Body>| async {
                let res = Response::builder()
                    .header(header::CONTENT_TYPE, "good-content")
                    .body(Body::empty())
                    .unwrap();
                Ok::<_, Infallible>(res)
            }),
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html"),
        );

        let res = svc.oneshot(Request::new(Body::empty())).await.unwrap();

        let mut values = res.headers().get_all(header::CONTENT_TYPE).iter();
        assert_eq!(values.next().unwrap(), "text/html");
        assert_eq!(values.next(), None);
    }

    #[tokio::test]
    async fn test_append_mode() {
        let svc = SetResponseHeader::appending(
            service_fn(|_req: Request<Body>| async {
                let res = Response::builder()
                    .header(header::CONTENT_TYPE, "good-content")
                    .body(Body::empty())
                    .unwrap();
                Ok::<_, Infallible>(res)
            }),
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html"),
        );

        let res = svc.oneshot(Request::new(Body::empty())).await.unwrap();

        let mut values = res.headers().get_all(header::CONTENT_TYPE).iter();
        assert_eq!(values.next().unwrap(), "good-content");
        assert_eq!(values.next().unwrap(), "text/html");
        assert_eq!(values.next(), None);
    }

    #[tokio::test]
    async fn test_skip_if_present_mode() {
        let svc = SetResponseHeader::if_not_present(
            service_fn(|_req: Request<Body>| async {
                let res = Response::builder()
                    .header(header::CONTENT_TYPE, "good-content")
                    .body(Body::empty())
                    .unwrap();
                Ok::<_, Infallible>(res)
            }),
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html"),
        );

        let res = svc.oneshot(Request::new(Body::empty())).await.unwrap();

        let mut values = res.headers().get_all(header::CONTENT_TYPE).iter();
        assert_eq!(values.next().unwrap(), "good-content");
        assert_eq!(values.next(), None);
    }

    #[tokio::test]
    async fn test_skip_if_present_mode_when_not_present() {
        let svc = SetResponseHeader::if_not_present(
            service_fn(|_req: Request<Body>| async {
                let res = Response::builder().body(Body::empty()).unwrap();
                Ok::<_, Infallible>(res)
            }),
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html"),
        );

        let res = svc.oneshot(Request::new(Body::empty())).await.unwrap();

        let mut values = res.headers().get_all(header::CONTENT_TYPE).iter();
        assert_eq!(values.next().unwrap(), "text/html");
        assert_eq!(values.next(), None);
    }
}
