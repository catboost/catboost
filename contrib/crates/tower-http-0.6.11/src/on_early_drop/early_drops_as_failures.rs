//! Adapter that bridges early-drop events to [`trace::OnFailure`].
//!
//! [`trace::OnFailure`]: crate::trace::OnFailure

use crate::on_early_drop::failure::{BodyDropped, DroppedFailure, FutureDropped};
use crate::on_early_drop::traits::{OnBodyDrop, OnDropCallback, OnFutureDrop};
use crate::trace::OnFailure;
use http::{response, Request, StatusCode};
use std::time::Instant;
use tracing::Span;

/// Bridges early-drop events to [`trace::OnFailure`](crate::trace::OnFailure).
///
/// Each event is reported by invoking the wrapped hook with a
/// [`DroppedFailure`]: `Future` for future drops, `Body` for body drops
/// (carrying the emitted response status).
///
/// Latency is computed from the moment the hook is produced (either at
/// `Service::call` or at response-ready time). The captured span is
/// [`Span::current()`] at that same moment. To report events against the
/// request span used by [`TraceLayer`](crate::trace::TraceLayer), place
/// [`OnEarlyDropLayer`] inside `TraceLayer`.
///
/// See the [module docs](super) for the example.
///
/// [`OnEarlyDropLayer`]: super::OnEarlyDropLayer
#[derive(Debug, Clone, Copy)]
pub struct EarlyDropsAsFailures<F> {
    on_failure: F,
}

impl<F> EarlyDropsAsFailures<F> {
    /// Wrap an [`OnFailure`] implementation.
    pub fn new(on_failure: F) -> Self {
        Self { on_failure }
    }
}

/// Future-drop callback produced by [`EarlyDropsAsFailures`].
pub struct FutureDropFailureCallback<F> {
    start: Instant,
    on_failure: F,
    span: Span,
}

impl<F> OnDropCallback for FutureDropFailureCallback<F>
where
    F: OnFailure<DroppedFailure> + Send + 'static,
{
    fn on_drop(mut self) {
        let latency = self.start.elapsed();
        let _entered = self.span.enter();
        self.on_failure
            .on_failure(DroppedFailure::Future(FutureDropped), latency, &self.span);
    }
}

/// Intermediate produced by [`OnBodyDrop::make_at_call`] for
/// [`EarlyDropsAsFailures`], carrying state forward to
/// [`OnBodyDrop::make_at_response`].
pub struct PreResponseBodyDropCallback<F> {
    start: Instant,
    on_failure: F,
    span: Span,
}

/// Body-drop callback produced by [`EarlyDropsAsFailures`].
pub struct BodyDropFailureCallback<F> {
    start: Instant,
    on_failure: F,
    span: Span,
    status: StatusCode,
}

impl<F> OnDropCallback for BodyDropFailureCallback<F>
where
    F: OnFailure<DroppedFailure> + Send + 'static,
{
    fn on_drop(mut self) {
        let latency = self.start.elapsed();
        let _entered = self.span.enter();
        self.on_failure.on_failure(
            DroppedFailure::Body(BodyDropped {
                status: self.status,
            }),
            latency,
            &self.span,
        );
    }
}

impl<F, ReqB> OnFutureDrop<ReqB> for EarlyDropsAsFailures<F>
where
    F: OnFailure<DroppedFailure> + Clone + Send + 'static,
{
    type Callback = FutureDropFailureCallback<F>;

    fn make(&mut self, _request: &Request<ReqB>) -> Self::Callback {
        FutureDropFailureCallback {
            start: Instant::now(),
            on_failure: self.on_failure.clone(),
            span: Span::current(),
        }
    }
}

impl<F, ReqB> OnBodyDrop<ReqB> for EarlyDropsAsFailures<F>
where
    F: OnFailure<DroppedFailure> + Clone + Send + 'static,
{
    type Intermediate = PreResponseBodyDropCallback<F>;
    type Callback = BodyDropFailureCallback<F>;

    fn make_at_call(&mut self, _request: &Request<ReqB>) -> Self::Intermediate {
        PreResponseBodyDropCallback {
            start: Instant::now(),
            on_failure: self.on_failure.clone(),
            span: Span::current(),
        }
    }

    fn make_at_response(
        &mut self,
        intermediate: Self::Intermediate,
        response_parts: &response::Parts,
    ) -> Self::Callback {
        BodyDropFailureCallback {
            start: intermediate.start,
            on_failure: intermediate.on_failure,
            span: intermediate.span,
            status: response_parts.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::on_early_drop::OnEarlyDropLayer;
    use bytes::Bytes;
    use http::{Request, Response, StatusCode};
    use http_body_util::{BodyExt, Full};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::time::{sleep, timeout};
    use tower::{service_fn, Layer, ServiceExt};
    use tracing::Span;

    #[derive(Clone, Default)]
    struct RecordingOnFailure {
        events: Arc<Mutex<Vec<DroppedFailure>>>,
    }

    impl OnFailure<DroppedFailure> for RecordingOnFailure {
        fn on_failure(&mut self, class: DroppedFailure, _latency: Duration, _span: &Span) {
            self.events.lock().unwrap().push(class);
        }
    }

    #[tokio::test]
    async fn future_drop_reports_future_failure() {
        let recorder = RecordingOnFailure::default();
        let events = recorder.events.clone();

        let slow_service = service_fn(|_req: Request<()>| async move {
            sleep(Duration::from_secs(60)).await;
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::new()))
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::new(EarlyDropsAsFailures::new(recorder));
        let service = layer.layer(slow_service);
        let _ = timeout(
            Duration::from_millis(50),
            service.oneshot(Request::builder().uri("/").body(()).unwrap()),
        )
        .await;

        sleep(Duration::from_millis(10)).await;
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!(matches!(captured[0], DroppedFailure::Future(_)));
    }

    #[tokio::test]
    async fn body_drop_reports_body_failure_with_status() {
        let recorder = RecordingOnFailure::default();
        let events = recorder.events.clone();

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

        let service = service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::CREATED)
                    .body(PendingBody)
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::new(EarlyDropsAsFailures::new(recorder));
        let service = layer.layer(service);
        let response = service
            .oneshot(Request::builder().uri("/").body(()).unwrap())
            .await
            .unwrap();
        drop(response);

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        match &captured[0] {
            DroppedFailure::Body(body) => assert_eq!(body.status, StatusCode::CREATED),
            other => panic!("expected Body failure, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn completion_suppresses_both() {
        let recorder = RecordingOnFailure::default();
        let events = recorder.events.clone();

        let ok_service = service_fn(|_req: Request<()>| async move {
            Ok::<_, std::convert::Infallible>(
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from_static(b"hi")))
                    .unwrap(),
            )
        });

        let layer = OnEarlyDropLayer::new(EarlyDropsAsFailures::new(recorder));
        let service = layer.layer(ok_service);
        let response = service
            .oneshot(Request::builder().uri("/").body(()).unwrap())
            .await
            .unwrap();
        let _body = response.into_body().collect().await.unwrap();

        assert!(events.lock().unwrap().is_empty());
    }
}
