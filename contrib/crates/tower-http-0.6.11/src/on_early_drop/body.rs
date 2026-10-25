//! Response body wrapper for [`OnEarlyDropService`].
//!
//! [`OnEarlyDropService`]: super::OnEarlyDropService

use crate::on_early_drop::guard::OnEarlyDropGuard;
use crate::on_early_drop::traits::OnDropCallback;
use http_body::{Body, Frame};
use pin_project_lite::pin_project;
use std::{
    pin::Pin,
    task::{ready, Context, Poll},
};

pin_project! {
    /// Response body for [`OnEarlyDropService`]. Fires its callback if
    /// dropped before reaching end-of-stream.
    ///
    /// Bodies that already report [`Body::is_end_stream`] at construction
    /// (HEAD requests, 204 responses, etc.) never fire.
    ///
    /// [`OnEarlyDropService`]: super::OnEarlyDropService
    pub struct OnEarlyDropBody<B, Callback>
    where
        Callback: OnDropCallback,
    {
        #[pin]
        inner: B,
        guard: OnEarlyDropGuard<Callback>,
    }
}

impl<B, Callback> OnEarlyDropBody<B, Callback>
where
    Callback: OnDropCallback,
{
    /// Wrap `body` with a drop callback.
    pub(crate) fn new(body: B, callback: Callback) -> Self
    where
        B: Body,
    {
        let mut guard = OnEarlyDropGuard::new(callback);
        if body.is_end_stream() {
            guard.completed();
        }
        Self { inner: body, guard }
    }
}

impl<B, Callback> Body for OnEarlyDropBody<B, Callback>
where
    B: Body,
    Callback: OnDropCallback,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        let result = ready!(this.inner.poll_frame(cx));
        // End-of-stream (Ready(None)) or body-level error (Ready(Some(Err)))
        // both mean the body will not yield more frames. Suppress the guard
        // in either case; service-level errors are out of scope for this
        // middleware.
        if matches!(result, None | Some(Err(_))) {
            this.guard.completed();
        }
        Poll::Ready(result)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}
