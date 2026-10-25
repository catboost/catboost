//! Hook traits invoked when a response future or response body is dropped
//! Hook traits invoked when a response future or response body is dropped
//! before completion.

use http::{response, Request};

/// Callback fired exactly once when an early-drop event is observed.
///
/// `FnOnce() + Send + 'static` closures implement this via a blanket impl.
///
/// # Panics
///
/// Implementations must not panic. Callbacks fire from [`Drop`]; panicking
/// during a drop that occurs while another panic is unwinding aborts the
/// process. The crate's built-in callbacks are panic-free.
pub trait OnDropCallback: Send + 'static {
    /// Fire the callback. Consumes `self`.
    fn on_drop(self);
}

impl<F> OnDropCallback for F
where
    F: FnOnce() + Send + 'static,
{
    fn on_drop(self) {
        self()
    }
}

/// No-op callback used as the default for empty hook slots.
#[derive(Debug, Clone, Copy)]
pub struct NoopDropCallback;

impl OnDropCallback for NoopDropCallback {
    fn on_drop(self) {}
}

/// Hook fired when the response future is dropped before producing a result.
///
/// See the [module docs](super) for the example.
pub trait OnFutureDrop<ReqB> {
    /// The callback produced for each request.
    type Callback: OnDropCallback;

    /// Produce a callback for the given request.
    fn make(&mut self, request: &Request<ReqB>) -> Self::Callback;
}

impl<F, C, ReqB> OnFutureDrop<ReqB> for F
where
    F: FnMut(&Request<ReqB>) -> C,
    C: OnDropCallback,
{
    type Callback = C;

    fn make(&mut self, request: &Request<ReqB>) -> Self::Callback {
        (self)(request)
    }
}

impl<ReqB> OnFutureDrop<ReqB> for () {
    type Callback = NoopDropCallback;

    fn make(&mut self, _request: &Request<ReqB>) -> Self::Callback {
        NoopDropCallback
    }
}

/// Hook fired when the response body is dropped before reaching
/// end-of-stream.
///
/// See the [module docs](super) for the example and phase breakdown.
pub trait OnBodyDrop<ReqB> {
    /// State carried from [`make_at_call`](Self::make_at_call) to
    /// [`make_at_response`](Self::make_at_response).
    type Intermediate: Send + 'static;

    /// The final callback produced for each response.
    type Callback: OnDropCallback;

    /// Capture request-time context.
    fn make_at_call(&mut self, request: &Request<ReqB>) -> Self::Intermediate;

    /// Capture response-time context and produce the final callback.
    fn make_at_response(
        &mut self,
        intermediate: Self::Intermediate,
        response_parts: &response::Parts,
    ) -> Self::Callback;
}

impl<ReqB> OnBodyDrop<ReqB> for () {
    type Intermediate = ();
    type Callback = NoopDropCallback;

    fn make_at_call(&mut self, _request: &Request<ReqB>) -> Self::Intermediate {}

    fn make_at_response(
        &mut self,
        _intermediate: Self::Intermediate,
        _response_parts: &response::Parts,
    ) -> Self::Callback {
        NoopDropCallback
    }
}

/// Adapter making `FnMut(&Request) -> FnOnce(&Parts) -> FnOnce()` closure
/// chains implement [`OnBodyDrop`].
///
/// See the [module docs](super) for the example.
#[derive(Clone, Copy)]
pub struct OnBodyDropFn<F>(F);

impl<F> OnBodyDropFn<F> {
    /// Wrap a closure chain as an [`OnBodyDrop`] hook.
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> std::fmt::Debug for OnBodyDropFn<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnBodyDropFn").finish_non_exhaustive()
    }
}

impl<F, G, C, ReqB> OnBodyDrop<ReqB> for OnBodyDropFn<F>
where
    F: FnMut(&Request<ReqB>) -> G,
    G: FnOnce(&response::Parts) -> C + Send + 'static,
    C: OnDropCallback,
{
    type Intermediate = G;
    type Callback = C;

    fn make_at_call(&mut self, request: &Request<ReqB>) -> Self::Intermediate {
        (self.0)(request)
    }

    fn make_at_response(
        &mut self,
        intermediate: Self::Intermediate,
        response_parts: &response::Parts,
    ) -> Self::Callback {
        (intermediate)(response_parts)
    }
}
