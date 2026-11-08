//! Service implementation for the on-early-drop middleware.

use crate::on_early_drop::body::OnEarlyDropBody;
use crate::on_early_drop::future::OnEarlyDropFuture;
use crate::on_early_drop::traits::{OnBodyDrop, OnFutureDrop};
use http::{Request, Response};
use std::task::{Context, Poll};
use tower_service::Service;

/// [`Service`] produced by [`OnEarlyDropLayer`].
///
/// See the [module docs](super) for details and examples.
///
/// [`OnEarlyDropLayer`]: super::OnEarlyDropLayer
pub struct OnEarlyDropService<S, OFD, OBD> {
    pub(crate) inner: S,
    pub(crate) on_future_drop: OFD,
    pub(crate) on_body_drop: OBD,
}

impl<S, OFD, OBD> std::fmt::Debug for OnEarlyDropService<S, OFD, OBD>
where
    S: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnEarlyDropService")
            .field("inner", &self.inner)
            .field("on_future_drop", &format_args!(".."))
            .field("on_body_drop", &format_args!(".."))
            .finish()
    }
}

impl<S, OFD, OBD> Clone for OnEarlyDropService<S, OFD, OBD>
where
    S: Clone,
    OFD: Clone,
    OBD: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            on_future_drop: self.on_future_drop.clone(),
            on_body_drop: self.on_body_drop.clone(),
        }
    }
}

impl<S, OFD, OBD> OnEarlyDropService<S, OFD, OBD> {
    /// Construct a new service directly. Most uses go through
    /// [`OnEarlyDropLayer`](super::OnEarlyDropLayer).
    pub fn new(inner: S, on_future_drop: OFD, on_body_drop: OBD) -> Self {
        Self {
            inner,
            on_future_drop,
            on_body_drop,
        }
    }

    define_inner_service_accessors!();
}

impl<S, OFD, OBD, ReqB, ResB> Service<Request<ReqB>> for OnEarlyDropService<S, OFD, OBD>
where
    S: Service<Request<ReqB>, Response = Response<ResB>>,
    OFD: OnFutureDrop<ReqB>,
    OBD: OnBodyDrop<ReqB> + Clone,
    ResB: http_body::Body,
{
    type Response = Response<OnEarlyDropBody<ResB, OBD::Callback>>;
    type Error = S::Error;
    type Future = OnEarlyDropFuture<S::Future, OBD, ReqB, OFD::Callback, OBD::Callback>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqB>) -> Self::Future {
        let future_callback = self.on_future_drop.make(&req);
        let intermediate = self.on_body_drop.make_at_call(&req);
        let inner = self.inner.call(req);
        OnEarlyDropFuture::new(
            inner,
            future_callback,
            self.on_body_drop.clone(),
            intermediate,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::on_early_drop::{OnBodyDropFn, OnEarlyDropLayer};
    use bytes::Bytes;
    use http::{Request, Response, StatusCode};
    use http_body_util::{BodyExt, Full};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};
    use tower::{service_fn, Layer, ServiceExt};

    fn ok_service() -> impl Service<
        Request<()>,
        Response = Response<Full<Bytes>>,
        Error = std::convert::Infallible,
        Future = impl std::future::Future<
            Output = Result<Response<Full<Bytes>>, std::convert::Infallible>,
        > + Send,
    > + Clone {
        service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from_static(b"hello")))
                    .unwrap(),
            )
        })
    }

    fn request() -> Request<()> {
        Request::builder().uri("http://example/").body(()).unwrap()
    }

    #[tokio::test]
    async fn forwards_response() {
        let layer = OnEarlyDropLayer::builder();
        let service = layer.layer(ok_service());
        let response = service.oneshot(request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body, "hello");
    }

    #[tokio::test]
    async fn future_drop_fires_callback() {
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = fired.clone();

        let slow_service = service_fn(|_req: Request<()>| async move {
            sleep(Duration::from_secs(60)).await;
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::new()))
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::builder().on_future_drop(move |_req: &Request<()>| {
            let fired = fired_clone.clone();
            move || {
                fired.fetch_add(1, Ordering::Relaxed);
            }
        });
        let service = layer.layer(slow_service);
        let _ = timeout(Duration::from_millis(50), service.oneshot(request())).await;

        sleep(Duration::from_millis(10)).await;
        assert_eq!(fired.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn future_drop_suppressed_on_completion() {
        let fired = Arc::new(AtomicUsize::new(0));
        let fired_clone = fired.clone();

        let layer = OnEarlyDropLayer::builder().on_future_drop(move |_req: &Request<()>| {
            let fired = fired_clone.clone();
            move || {
                fired.fetch_add(1, Ordering::Relaxed);
            }
        });
        let service = layer.layer(ok_service());
        let _ = service.oneshot(request()).await.unwrap();

        assert_eq!(fired.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn body_drop_fires_callback_with_status() {
        let observed_status = Arc::new(std::sync::Mutex::new(None));
        let observed_clone = observed_status.clone();

        // Body that never reaches end-of-stream.
        struct PendingBody;
        impl http_body::Body for PendingBody {
            type Data = Bytes;
            type Error = std::convert::Infallible;
            fn poll_frame(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>>
            {
                std::task::Poll::Pending
            }
            fn is_end_stream(&self) -> bool {
                false
            }
        }

        let pending_service = service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::CREATED)
                    .body(PendingBody)
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::builder().on_body_drop(OnBodyDropFn::new(
            move |_req: &Request<()>| {
                let observed = observed_clone.clone();
                move |parts: &http::response::Parts| {
                    let status = parts.status;
                    move || {
                        *observed.lock().unwrap() = Some(status);
                    }
                }
            },
        ));
        let service = layer.layer(pending_service);
        let response = service.oneshot(request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        drop(response);

        assert_eq!(
            *observed_status.lock().unwrap(),
            Some(StatusCode::CREATED),
            "body-drop callback should observe the response status",
        );
    }

    #[tokio::test]
    async fn body_drop_suppressed_when_body_consumed() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();

        let layer = OnEarlyDropLayer::builder().on_body_drop(OnBodyDropFn::new(
            move |_req: &Request<()>| {
                let fired = fired_clone.clone();
                move |_parts: &http::response::Parts| {
                    let fired = fired.clone();
                    move || {
                        fired.store(true, Ordering::Relaxed);
                    }
                }
            },
        ));
        let service = layer.layer(ok_service());
        let response = service.oneshot(request()).await.unwrap();
        let _body = response.into_body().collect().await.unwrap();

        assert!(!fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn inner_error_does_not_fire() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();

        let err_service = service_fn(|_req: Request<()>| async move {
            Err::<Response<Full<Bytes>>, _>(std::io::Error::other("boom"))
        });

        let layer = OnEarlyDropLayer::builder().on_future_drop(move |_req: &Request<()>| {
            let fired = fired_clone.clone();
            move || {
                fired.store(true, Ordering::Relaxed);
            }
        });
        let service = layer.layer(err_service);
        let _ = service.oneshot(request()).await;

        assert!(!fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn body_error_frame_does_not_fire() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();

        // Body that returns Err once, then is dropped.
        struct ErrBody {
            yielded: bool,
        }
        impl http_body::Body for ErrBody {
            type Data = Bytes;
            type Error = std::io::Error;
            fn poll_frame(
                mut self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>>
            {
                if self.yielded {
                    std::task::Poll::Ready(None)
                } else {
                    self.yielded = true;
                    std::task::Poll::Ready(Some(Err(std::io::Error::other("frame err"))))
                }
            }
            fn is_end_stream(&self) -> bool {
                false
            }
        }

        let err_body_service = service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::OK)
                    .body(ErrBody { yielded: false })
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::builder().on_body_drop(OnBodyDropFn::new(
            move |_req: &Request<()>| {
                let fired = fired_clone.clone();
                move |_parts: &http::response::Parts| {
                    let fired = fired.clone();
                    move || {
                        fired.store(true, Ordering::Relaxed);
                    }
                }
            },
        ));
        let service = layer.layer(err_body_service);
        let response = service.oneshot(request()).await.unwrap();
        // Poll the body until it surfaces the Err frame, then drop.
        let mut body = response.into_body();
        use http_body::Body as _;
        let frame = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)).await;
        assert!(matches!(frame, Some(Err(_))));
        drop(body);

        assert!(
            !fired.load(Ordering::Relaxed),
            "body-level error must not be reported as a body drop",
        );
    }

    // The service's trait bounds must not require hook types to be `Debug`;
    // non-Debug closures must produce a service that still compiles.
    #[allow(dead_code)]
    fn static_property_hooks_without_debug() {
        fn hook_without_debug<F>(f: F) -> F {
            f
        }
        let _layer = OnEarlyDropLayer::builder()
            .on_future_drop(hook_without_debug(|_req: &Request<()>| || {}))
            .on_body_drop(OnBodyDropFn::new(hook_without_debug(
                |_req: &Request<()>| |_parts: &http::response::Parts| || {},
            )));
    }

    // The service must be Send + Sync + Clone whenever the underlying
    // hooks and inner service are.
    #[allow(dead_code)]
    fn static_property_service_is_send_sync() {
        fn assert_send<T: Send>(_: &T) {}
        fn assert_sync<T: Sync>(_: &T) {}
        fn assert_clone<T: Clone>(_: &T) {}

        let layer = OnEarlyDropLayer::builder();
        let service = layer.layer(ok_service());
        assert_send(&service);
        assert_sync(&service);
        assert_clone(&service);
    }

    #[tokio::test]
    async fn body_drop_suppressed_when_is_end_stream_at_construction() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();

        // Body already at end-of-stream at construction (HEAD response,
        // 204 No Content, etc).
        let empty_service = service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::NO_CONTENT)
                    .body(http_body_util::Empty::<Bytes>::new())
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::builder().on_body_drop(OnBodyDropFn::new(
            move |_req: &Request<()>| {
                let fired = fired_clone.clone();
                move |_parts: &http::response::Parts| {
                    let fired = fired.clone();
                    move || {
                        fired.store(true, Ordering::Relaxed);
                    }
                }
            },
        ));
        let service = layer.layer(empty_service);
        let response = service.oneshot(request()).await.unwrap();
        // Drop immediately without polling the body.
        drop(response);

        assert!(
            !fired.load(Ordering::Relaxed),
            "body already at end-of-stream at construction must not fire the callback",
        );
    }

    #[tokio::test]
    async fn body_drop_does_not_fire_on_inner_error() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();

        let err_service = service_fn(|_req: Request<()>| async move {
            Err::<Response<Full<Bytes>>, _>(std::io::Error::other("boom"))
        });

        let layer = OnEarlyDropLayer::builder().on_body_drop(OnBodyDropFn::new(
            move |_req: &Request<()>| {
                let fired = fired_clone.clone();
                move |_parts: &http::response::Parts| {
                    let fired = fired.clone();
                    move || {
                        fired.store(true, Ordering::Relaxed);
                    }
                }
            },
        ));
        let service = layer.layer(err_service);
        let _ = service.oneshot(request()).await;

        assert!(!fired.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn noop_slots_do_not_fire() {
        // Builder with default () slots: no hook is installed. Even on a
        // dropped pending future and a dropped incomplete body, nothing
        // should be observable.
        let layer = OnEarlyDropLayer::builder();
        let service = layer.layer(ok_service());
        let response = service.oneshot(request()).await.unwrap();
        // Dropping without consuming the body.
        drop(response);
        // Nothing to assert; reaching here without panic confirms the
        // no-op slots do not panic or invoke any user code.
    }
}
