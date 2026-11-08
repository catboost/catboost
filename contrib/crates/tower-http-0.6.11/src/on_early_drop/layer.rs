//! Tower [`Layer`] for the on-early-drop middleware.

use crate::on_early_drop::service::OnEarlyDropService;
use tower_layer::Layer;

/// [`Layer`] that applies [`OnEarlyDropService`].
///
/// See the [module docs](super) for details and examples.
#[derive(Clone, Copy)]
pub struct OnEarlyDropLayer<OFD, OBD> {
    on_future_drop: OFD,
    on_body_drop: OBD,
}

impl<OFD, OBD> std::fmt::Debug for OnEarlyDropLayer<OFD, OBD> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnEarlyDropLayer")
            .field("on_future_drop", &format_args!(".."))
            .field("on_body_drop", &format_args!(".."))
            .finish()
    }
}

impl OnEarlyDropLayer<(), ()> {
    /// Start with both slots set to no-op. Chain
    /// [`on_future_drop`](Self::on_future_drop) and
    /// [`on_body_drop`](Self::on_body_drop) to install hooks.
    pub fn builder() -> Self {
        Self {
            on_future_drop: (),
            on_body_drop: (),
        }
    }
}

impl<H: Clone> OnEarlyDropLayer<H, H> {
    /// Install the same hook in both slots.
    ///
    /// Typical choice is
    /// [`EarlyDropsAsFailures`](crate::on_early_drop::EarlyDropsAsFailures);
    /// see the [module docs](super) for the example.
    pub fn new(hook: H) -> Self {
        Self {
            on_future_drop: hook.clone(),
            on_body_drop: hook,
        }
    }
}

impl<OFD, OBD> OnEarlyDropLayer<OFD, OBD> {
    /// Replace the future-drop slot.
    pub fn on_future_drop<T>(self, hook: T) -> OnEarlyDropLayer<T, OBD> {
        OnEarlyDropLayer {
            on_future_drop: hook,
            on_body_drop: self.on_body_drop,
        }
    }

    /// Replace the body-drop slot.
    pub fn on_body_drop<T>(self, hook: T) -> OnEarlyDropLayer<OFD, T> {
        OnEarlyDropLayer {
            on_future_drop: self.on_future_drop,
            on_body_drop: hook,
        }
    }
}

impl<S, OFD, OBD> Layer<S> for OnEarlyDropLayer<OFD, OBD>
where
    OFD: Clone,
    OBD: Clone,
{
    type Service = OnEarlyDropService<S, OFD, OBD>;

    fn layer(&self, inner: S) -> Self::Service {
        OnEarlyDropService::new(
            inner,
            self.on_future_drop.clone(),
            self.on_body_drop.clone(),
        )
    }
}
