use crate::timeout::body::TimeoutBody;
use http::{Request, Response, StatusCode};
use pin_project_lite::pin_project;
use std::{
    future::Future,
    pin::Pin,
    task::{ready, Context, Poll},
    time::Duration,
};
use tokio::time::Sleep;
use tower_layer::Layer;
use tower_service::Service;

/// Layer that applies the [`Timeout`] middleware which apply a timeout to requests.
///
/// See the [module docs](super) for an example.
#[derive(Debug, Clone, Copy)]
pub struct TimeoutLayer {
    timeout: Duration,
    status_code: StatusCode,
}

impl TimeoutLayer {
    /// Creates a new [`TimeoutLayer`].
    ///
    /// By default, it will return a `408 Request Timeout` response if the request does not complete within the specified timeout.
    /// To customize the response status code, use the `with_status_code` method.
    #[deprecated(since = "0.6.7", note = "Use `TimeoutLayer::with_status_code` instead")]
    pub fn new(timeout: Duration) -> Self {
        Self::with_status_code(StatusCode::REQUEST_TIMEOUT, timeout)
    }

    /// Creates a new [`TimeoutLayer`] with the specified status code for the timeout response.
    pub fn with_status_code(status_code: StatusCode, timeout: Duration) -> Self {
        Self {
            timeout,
            status_code,
        }
    }
}

impl<S> Layer<S> for TimeoutLayer {
    type Service = Timeout<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Timeout::with_status_code(inner, self.status_code, self.timeout)
    }
}

/// Middleware which apply a timeout to requests.
///
/// See the [module docs](super) for an example.
#[derive(Debug, Clone, Copy)]
pub struct Timeout<S> {
    inner: S,
    timeout: Duration,
    status_code: StatusCode,
}

impl<S> Timeout<S> {
    /// Creates a new [`Timeout`].
    ///
    /// By default, it will return a `408 Request Timeout` response if the request does not complete within the specified timeout.
    /// To customize the response status code, use the `with_status_code` method.
    #[deprecated(since = "0.6.7", note = "Use `Timeout::with_status_code` instead")]
    pub fn new(inner: S, timeout: Duration) -> Self {
        Self::with_status_code(inner, StatusCode::REQUEST_TIMEOUT, timeout)
    }

    /// Creates a new [`Timeout`] with the specified status code for the timeout response.
    pub fn with_status_code(inner: S, status_code: StatusCode, timeout: Duration) -> Self {
        Self {
            inner,
            timeout,
            status_code,
        }
    }

    define_inner_service_accessors!();

    /// Returns a new [`Layer`] that wraps services with a `Timeout` middleware.
    ///
    /// [`Layer`]: tower_layer::Layer
    #[deprecated(
        since = "0.6.7",
        note = "Use `Timeout::layer_with_status_code` instead"
    )]
    pub fn layer(timeout: Duration) -> TimeoutLayer {
        TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, timeout)
    }

    /// Returns a new [`Layer`] that wraps services with a `Timeout` middleware with the specified status code.
    pub fn layer_with_status_code(status_code: StatusCode, timeout: Duration) -> TimeoutLayer {
        TimeoutLayer::with_status_code(status_code, timeout)
    }
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for Timeout<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    ResBody: Default,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = ResponseFuture<S::Future>;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let sleep = tokio::time::sleep(self.timeout);
        ResponseFuture {
            inner: self.inner.call(req),
            sleep,
            status_code: self.status_code,
        }
    }
}

pin_project! {
    /// Response future for [`Timeout`].
    pub struct ResponseFuture<F> {
        #[pin]
        inner: F,
        #[pin]
        sleep: Sleep,
        status_code: StatusCode,
    }
}

impl<F, B, E> Future for ResponseFuture<F>
where
    F: Future<Output = Result<Response<B>, E>>,
    B: Default,
{
    type Output = Result<Response<B>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        if this.sleep.poll(cx).is_ready() {
            let mut res = Response::new(B::default());
            *res.status_mut() = *this.status_code;
            return Poll::Ready(Ok(res));
        }

        this.inner.poll(cx)
    }
}

/// Applies a [`TimeoutBody`] to the request body.
#[derive(Clone, Debug)]
pub struct RequestBodyTimeoutLayer {
    timeout: Duration,
}

impl RequestBodyTimeoutLayer {
    /// Creates a new [`RequestBodyTimeoutLayer`].
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl<S> Layer<S> for RequestBodyTimeoutLayer {
    type Service = RequestBodyTimeout<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestBodyTimeout::new(inner, self.timeout)
    }
}

/// Applies a [`TimeoutBody`] to the request body.
#[derive(Clone, Debug)]
pub struct RequestBodyTimeout<S> {
    inner: S,
    timeout: Duration,
}

impl<S> RequestBodyTimeout<S> {
    /// Creates a new [`RequestBodyTimeout`].
    pub fn new(service: S, timeout: Duration) -> Self {
        Self {
            inner: service,
            timeout,
        }
    }

    /// Returns a new [`Layer`] that wraps services with a [`RequestBodyTimeoutLayer`] middleware.
    ///
    /// [`Layer`]: tower_layer::Layer
    pub fn layer(timeout: Duration) -> RequestBodyTimeoutLayer {
        RequestBodyTimeoutLayer::new(timeout)
    }

    define_inner_service_accessors!();
}

impl<S, ReqBody> Service<Request<ReqBody>> for RequestBodyTimeout<S>
where
    S: Service<Request<TimeoutBody<ReqBody>>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let req = req.map(|body| TimeoutBody::new(self.timeout, body));
        self.inner.call(req)
    }
}

/// Applies a [`TimeoutBody`] to the response body.
#[derive(Clone)]
pub struct ResponseBodyTimeoutLayer {
    timeout: Duration,
}

impl ResponseBodyTimeoutLayer {
    /// Creates a new [`ResponseBodyTimeoutLayer`].
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl<S> Layer<S> for ResponseBodyTimeoutLayer {
    type Service = ResponseBodyTimeout<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ResponseBodyTimeout::new(inner, self.timeout)
    }
}

/// Applies a [`TimeoutBody`] to the response body.
#[derive(Clone)]
pub struct ResponseBodyTimeout<S> {
    inner: S,
    timeout: Duration,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for ResponseBodyTimeout<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = Response<TimeoutBody<ResBody>>;
    type Error = S::Error;
    type Future = ResponseBodyTimeoutFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        ResponseBodyTimeoutFuture {
            inner: self.inner.call(req),
            timeout: self.timeout,
        }
    }
}

impl<S> ResponseBodyTimeout<S> {
    /// Creates a new [`ResponseBodyTimeout`].
    pub fn new(service: S, timeout: Duration) -> Self {
        Self {
            inner: service,
            timeout,
        }
    }

    /// Returns a new [`Layer`] that wraps services with a [`ResponseBodyTimeoutLayer`] middleware.
    ///
    /// [`Layer`]: tower_layer::Layer
    pub fn layer(timeout: Duration) -> ResponseBodyTimeoutLayer {
        ResponseBodyTimeoutLayer::new(timeout)
    }

    define_inner_service_accessors!();
}

pin_project! {
    /// Response future for [`ResponseBodyTimeout`].
    pub struct ResponseBodyTimeoutFuture<Fut> {
        #[pin]
        inner: Fut,
        timeout: Duration,
    }
}

impl<Fut, ResBody, E> Future for ResponseBodyTimeoutFuture<Fut>
where
    Fut: Future<Output = Result<Response<ResBody>, E>>,
{
    type Output = Result<Response<TimeoutBody<ResBody>>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let timeout = self.timeout;
        let this = self.project();
        let res = ready!(this.inner.poll(cx))?;
        Poll::Ready(Ok(res.map(|body| TimeoutBody::new(timeout, body))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::Body;
    use http::{Request, Response, StatusCode};
    use std::time::Duration;
    use tower::{BoxError, ServiceBuilder, ServiceExt};

    #[tokio::test]
    async fn request_completes_within_timeout() {
        let mut service = ServiceBuilder::new()
            .layer(TimeoutLayer::with_status_code(
                StatusCode::GATEWAY_TIMEOUT,
                Duration::from_secs(1),
            ))
            .service_fn(fast_handler);

        let request = Request::get("/").body(Body::empty()).unwrap();
        let res = service.ready().await.unwrap().call(request).await.unwrap();

        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn timeout_middleware_with_custom_status_code() {
        let timeout_service = Timeout::with_status_code(
            tower::service_fn(slow_handler),
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_millis(10),
        );

        let mut service = ServiceBuilder::new().service(timeout_service);

        let request = Request::get("/").body(Body::empty()).unwrap();
        let res = service.ready().await.unwrap().call(request).await.unwrap();

        assert_eq!(res.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn timeout_response_has_empty_body() {
        let mut service = ServiceBuilder::new()
            .layer(TimeoutLayer::with_status_code(
                StatusCode::GATEWAY_TIMEOUT,
                Duration::from_millis(10),
            ))
            .service_fn(slow_handler);

        let request = Request::get("/").body(Body::empty()).unwrap();
        let res = service.ready().await.unwrap().call(request).await.unwrap();

        assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);

        // Verify the body is empty (default)
        use http_body_util::BodyExt;
        let body = res.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        assert!(bytes.is_empty());
    }

    #[tokio::test]
    async fn deprecated_new_method_compatibility() {
        #[allow(deprecated)]
        let layer = TimeoutLayer::new(Duration::from_millis(10));

        let mut service = ServiceBuilder::new().layer(layer).service_fn(slow_handler);

        let request = Request::get("/").body(Body::empty()).unwrap();
        let res = service.ready().await.unwrap().call(request).await.unwrap();

        // Should use default 408 status code
        assert_eq!(res.status(), StatusCode::REQUEST_TIMEOUT);
    }

    async fn slow_handler(_req: Request<Body>) -> Result<Response<Body>, BoxError> {
        tokio::time::sleep(Duration::from_secs(10)).await;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap())
    }

    async fn fast_handler(_req: Request<Body>) -> Result<Response<Body>, BoxError> {
        Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap())
    }
}
