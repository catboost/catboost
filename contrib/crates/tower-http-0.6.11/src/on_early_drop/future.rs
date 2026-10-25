//! Response future for [`OnEarlyDropService`].
//!
//! [`OnEarlyDropService`]: super::OnEarlyDropService

use crate::on_early_drop::body::OnEarlyDropBody;
use crate::on_early_drop::guard::OnEarlyDropGuard;
use crate::on_early_drop::traits::{OnBodyDrop, OnDropCallback};
use http::Response;
use pin_project_lite::pin_project;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    /// Response future for [`OnEarlyDropService`].
    ///
    /// [`OnEarlyDropService`]: super::OnEarlyDropService
    pub struct OnEarlyDropFuture<F, OBD, ReqB, FC, BC>
    where
        OBD: OnBodyDrop<ReqB, Callback = BC>,
        FC: OnDropCallback,
        BC: OnDropCallback,
    {
        #[pin]
        inner: F,
        // `Some` while the inner future is pending.
        future_guard: Option<OnEarlyDropGuard<FC>>,
        // `Some` between call-time and response-ready time.
        intermediate: Option<OBD::Intermediate>,
        // Retained for `make_at_response`.
        on_body_drop: Option<OBD>,
        // `fn(ReqB)` keeps Send/Sync and other auto-traits independent of ReqB.
        _phantom: std::marker::PhantomData<fn(ReqB)>,
    }
}

impl<F, OBD, ReqB, FC, BC> OnEarlyDropFuture<F, OBD, ReqB, FC, BC>
where
    OBD: OnBodyDrop<ReqB, Callback = BC>,
    FC: OnDropCallback,
    BC: OnDropCallback,
{
    pub(crate) fn new(
        inner: F,
        future_callback: FC,
        on_body_drop: OBD,
        intermediate: OBD::Intermediate,
    ) -> Self {
        Self {
            inner,
            future_guard: Some(OnEarlyDropGuard::new(future_callback)),
            intermediate: Some(intermediate),
            on_body_drop: Some(on_body_drop),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, OBD, B, E, ReqB, FC, BC> Future for OnEarlyDropFuture<F, OBD, ReqB, FC, BC>
where
    F: Future<Output = Result<Response<B>, E>>,
    OBD: OnBodyDrop<ReqB, Callback = BC>,
    FC: OnDropCallback,
    BC: OnDropCallback,
    B: http_body::Body,
{
    type Output = Result<Response<OnEarlyDropBody<B, BC>>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        let result = match this.inner.poll(cx) {
            Poll::Ready(result) => result,
            Poll::Pending => return Poll::Pending,
        };

        // Inner resolved: suppress the future-drop guard for both Ok and Err.
        if let Some(guard) = this.future_guard.as_mut() {
            guard.completed();
        }
        *this.future_guard = None;

        match result {
            Ok(response) => {
                let (parts, body) = response.into_parts();
                let intermediate = this
                    .intermediate
                    .take()
                    .expect("intermediate already consumed; OnEarlyDropFuture polled after Ready");
                let mut hook = this
                    .on_body_drop
                    .take()
                    .expect("on_body_drop already consumed; OnEarlyDropFuture polled after Ready");
                let callback = hook.make_at_response(intermediate, &parts);
                let wrapped_body = OnEarlyDropBody::new(body, callback);
                Poll::Ready(Ok(Response::from_parts(parts, wrapped_body)))
            }
            Err(err) => Poll::Ready(Err(err)),
        }
    }
}
