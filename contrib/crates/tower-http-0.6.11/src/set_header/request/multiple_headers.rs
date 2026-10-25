//! Set multiple headers on the request.
//!
//! See the root [`crate::set_header::request`] module for full documentation and usage examples.
//!
use http::{Request, Response};
use std::{
    fmt,
    task::{Context, Poll},
};
use tower_layer::Layer;
use tower_service::Service;

use crate::set_header::{HeaderInsertionConfig, HeaderMetadata, InsertHeaderMode};

/// Layer that applies [`SetMultipleRequestHeader`] which adds multiple request headers.
///
/// See [`SetMultipleRequestHeader`] for more details.
#[derive(Clone)]
pub struct SetMultipleRequestHeadersLayer<M> {
    headers: Vec<HeaderInsertionConfig<M>>,
}

impl<M> fmt::Debug for SetMultipleRequestHeadersLayer<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetMultipleRequestHeadersLayer")
            .field("headers", &self.headers)
            .finish()
    }
}

impl<M> SetMultipleRequestHeadersLayer<M> {
    /// Create a new [`SetMultipleRequestHeadersLayer`].
    ///
    /// If any previous value exists for the same header, it is removed and replaced with the new matching header value.
    pub fn overriding(metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::Override))
            .collect();

        Self::new(headers)
    }

    /// Create a new [`SetMultipleRequestHeadersLayer`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::Append))
            .collect();

        Self::new(headers)
    }

    /// Create a new [`SetMultipleRequestHeadersLayer`].
    ///
    /// If a previous value exists for the header, the new value is not inserted.
    pub fn if_not_present(metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::IfNotPresent))
            .collect();

        Self::new(headers)
    }

    /// Internal constructor for a new [`SetMultipleRequestHeadersLayer`] from a list of headers.
    fn new(headers: Vec<HeaderInsertionConfig<M>>) -> Self {
        Self { headers }
    }
}

impl<S, M> Layer<S> for SetMultipleRequestHeadersLayer<M> {
    type Service = SetMultipleRequestHeader<S, M>;

    fn layer(&self, inner: S) -> Self::Service {
        SetMultipleRequestHeader {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Middleware that sets multiple headers on the request.
#[derive(Clone)]
pub struct SetMultipleRequestHeader<S, M> {
    inner: S,
    headers: Vec<HeaderInsertionConfig<M>>,
}

impl<S, M> SetMultipleRequestHeader<S, M> {
    /// Create a new [`SetMultipleRequestHeader`].
    ///
    /// If a previous value exists for the same header, it is removed and replaced with the new
    /// header value.
    pub fn overriding(inner: S, metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::Override))
            .collect();

        Self::new(inner, headers)
    }

    /// Create a new [`SetMultipleRequestHeader`].
    ///
    /// The new header is always added, preserving any existing values. If previous values exist,
    /// the header will have multiple values.
    pub fn appending(inner: S, metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::Append))
            .collect();

        Self::new(inner, headers)
    }

    /// Create a new [`SetMultipleRequestHeader`].
    ///
    /// If a previous value exists for the header, the new value is not inserted.
    pub fn if_not_present(inner: S, metadata: Vec<HeaderMetadata<M>>) -> Self {
        let headers: Vec<HeaderInsertionConfig<M>> = metadata
            .into_iter()
            .map(|m| m.build_config(InsertHeaderMode::IfNotPresent))
            .collect();

        Self::new(inner, headers)
    }

    /// Internal constructor for a new [`SetMultipleRequestHeader`] from an inner service and a list of headers.
    fn new(inner: S, headers: Vec<HeaderInsertionConfig<M>>) -> Self {
        Self { inner, headers }
    }

    define_inner_service_accessors!();
}

impl<S, M> fmt::Debug for SetMultipleRequestHeader<S, M>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SetMultipleRequestHeader")
            .field("inner", &self.inner)
            .field("headers", &self.headers)
            .finish()
    }
}

impl<ReqBody, ResBody, S> Service<Request<ReqBody>>
    for SetMultipleRequestHeader<S, Request<ReqBody>>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        for header in &mut self.headers {
            header
                .mode
                .apply(&header.header_name, &mut req, &mut header.make);
        }

        self.inner.call(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::Body;
    use http::{header, HeaderValue, Request, Response};
    use std::convert::Infallible;
    use tower::{service_fn, ServiceExt};

    #[tokio::test]
    async fn test_override_mode() {
        let svc = SetMultipleRequestHeader::overriding(
            service_fn(|req: Request<Body>| async move {
                assert_eq!(req.headers()["content-type"], "text/html");
                Ok::<_, Infallible>(Response::new(Body::empty()))
            }),
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into()],
        );

        let mut req = Request::new(Body::empty());

        // Add an initial CONTENT_TYPE header to the request
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("good-content"),
        );

        let _ = svc.oneshot(req).await.unwrap();
    }

    #[tokio::test]
    async fn test_append_mode() {
        let svc = SetMultipleRequestHeader::appending(
            service_fn(|req: Request<Body>| async move {
                let mut values = req.headers().get_all("content-type").iter();
                assert_eq!(values.next().unwrap(), "good-content");
                assert_eq!(values.next().unwrap(), "text/html");
                assert_eq!(values.next(), None);

                Ok::<_, Infallible>(Response::new(Body::empty()))
            }),
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into()],
        );

        // Add an initial CONTENT_TYPE header to the request
        let mut req = Request::new(Body::empty());
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("good-content"),
        );

        _ = svc.oneshot(req).await.unwrap();
    }

    #[tokio::test]
    async fn test_skip_if_present_mode() {
        let svc = SetMultipleRequestHeader::if_not_present(
            service_fn(|req: Request<Body>| async move {
                let mut values = req.headers().get_all("content-type").iter();
                assert_eq!(values.next().unwrap(), "good-content");
                assert_eq!(values.next(), None);

                Ok::<_, Infallible>(Response::new(Body::empty()))
            }),
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into()],
        );

        // Add an initial CONTENT_TYPE header to the request
        let mut req = Request::new(Body::empty());
        req.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("good-content"),
        );

        let _ = svc.oneshot(req).await.unwrap();
    }

    #[tokio::test]
    async fn test_skip_if_present_mode_when_not_present() {
        let svc = SetMultipleRequestHeader::if_not_present(
            service_fn(|req: Request<Body>| async move {
                let mut values = req.headers().get_all("content-type").iter();
                assert_eq!(values.next().unwrap(), "text/html");
                assert_eq!(values.next(), None);
                Ok::<_, Infallible>(Response::new(Body::empty()))
            }),
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into()],
        );

        // No header to the request
        let req = Request::new(Body::empty());

        _ = svc.oneshot(req).await.unwrap();
    }

    #[test]
    fn test_debug_impls() {
        let meta: HeaderMetadata<HeaderValue> =
            (header::CONTENT_TYPE, HeaderValue::from_static("bar")).into();
        let rh = meta
            .clone()
            .build_config(crate::set_header::InsertHeaderMode::Override);
        let layer = SetMultipleRequestHeadersLayer::overriding(vec![meta]);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("SetMultipleRequestHeadersLayer"));
        let debug_rh = format!("{:?}", rh);
        assert!(debug_rh.contains("HeaderInsertionConfig"));

        let svc = SetMultipleRequestHeader::overriding(
            tower::service_fn(|_req: Request<Body>| async {
                Ok::<_, std::convert::Infallible>(Response::new(Body::empty()))
            }),
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("foo")).into()]
                as Vec<HeaderMetadata<HeaderValue>>,
        );
        let debug_svc = format!("{:?}", svc);
        assert!(debug_svc.contains("SetMultipleRequestHeader"));
    }

    #[tokio::test]
    async fn test_layer_construction_and_multiple_headers() {
        // Multiple different headers in the same vec
        let svc = tower::ServiceBuilder::new()
            .layer(SetMultipleRequestHeadersLayer::overriding(vec![
                (header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into(),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")).into(),
            ]))
            .service(service_fn(|req: Request<Body>| async move {
                assert_eq!(req.headers()["content-type"], "text/html");
                assert_eq!(req.headers()["cache-control"], "no-cache");

                Ok::<_, Infallible>(Response::new(Body::empty()))
            }));

        _ = svc.oneshot(Request::new(Body::empty())).await.unwrap();
    }

    #[tokio::test]
    async fn test_layer_with_empty_vec() {
        let header_metadatas: Vec<HeaderMetadata<Request<Body>>> = vec![];
        let svc = tower::ServiceBuilder::new()
            .layer(SetMultipleRequestHeadersLayer::<Request<Body>>::overriding(
                header_metadatas,
            ))
            .service(service_fn(|req: Request<Body>| async move {
                assert_eq!(req.headers().len(), 0);
                Ok::<_, Infallible>(Response::new(Body::empty()))
            }));

        _ = svc.oneshot(Request::new(Body::empty())).await.unwrap();
    }

    #[tokio::test]
    async fn test_layer_with_static_and_closure_headers_fixed() {
        // Wrap the static value
        let static_meta: HeaderMetadata<Request<Body>> =
            (header::CONTENT_TYPE, HeaderValue::from_static("text/html")).into();

        // Wrap the closure
        let closure_meta: HeaderMetadata<Request<Body>> =
            (header::X_FRAME_OPTIONS, |_req: &Request<Body>| {
                Some(HeaderValue::from_static("DENY"))
            })
                .into();

        let svc = tower::ServiceBuilder::new()
            .layer(SetMultipleRequestHeadersLayer::overriding(vec![
                static_meta,
                closure_meta,
            ]))
            .service(service_fn(|req: Request<Body>| async move {
                assert_eq!(req.headers()["content-type"], "text/html");
                assert_eq!(req.headers()["x-frame-options"], "DENY");

                Ok::<_, Infallible>(Response::new(Body::empty()))
            }));

        _ = svc.oneshot(Request::new(Body::empty())).await.unwrap();
    }

    #[test]
    fn test_debug_layer_and_service() {
        let meta: HeaderMetadata<HeaderValue> =
            (header::CONTENT_TYPE, HeaderValue::from_static("foo")).into();
        let layer = SetMultipleRequestHeadersLayer::overriding(vec![meta]);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("SetMultipleRequestHeadersLayer"));
    }
}
