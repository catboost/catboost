//! Drop guard that fires a callback when dropped unless marked completed.

use crate::on_early_drop::traits::OnDropCallback;

/// Runs a callback on drop unless [`completed`](Self::completed) is called
/// first.
///
/// # Examples
///
/// ```
/// use tower_http::on_early_drop::OnEarlyDropGuard;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::sync::Arc;
///
/// let count = Arc::new(AtomicUsize::new(0));
/// let count_for_guard = count.clone();
/// {
///     let _guard = OnEarlyDropGuard::new(move || {
///         count_for_guard.fetch_add(1, Ordering::Relaxed);
///     });
/// }
/// assert_eq!(count.load(Ordering::Relaxed), 1);
/// ```
///
/// Marking the guard completed suppresses the callback:
///
/// ```
/// use tower_http::on_early_drop::OnEarlyDropGuard;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::sync::Arc;
///
/// let count = Arc::new(AtomicUsize::new(0));
/// let count_for_guard = count.clone();
/// {
///     let mut guard = OnEarlyDropGuard::new(move || {
///         count_for_guard.fetch_add(1, Ordering::Relaxed);
///     });
///     guard.completed();
/// }
/// assert_eq!(count.load(Ordering::Relaxed), 0);
/// ```
///
/// [`OnEarlyDropLayer`]: super::OnEarlyDropLayer
#[derive(Debug)]
pub struct OnEarlyDropGuard<Callback: OnDropCallback> {
    callback: Option<Callback>,
}

impl<Callback: OnDropCallback> OnEarlyDropGuard<Callback> {
    /// Create a guard that will fire `callback` on drop.
    pub fn new(callback: Callback) -> Self {
        Self {
            callback: Some(callback),
        }
    }

    /// Mark the guard completed and drop the callback without firing it.
    ///
    /// Any resources captured by the callback are released immediately
    /// rather than at guard drop time.
    pub fn completed(&mut self) {
        self.callback = None;
    }
}

impl<Callback: OnDropCallback> Drop for OnEarlyDropGuard<Callback> {
    fn drop(&mut self) {
        // Panicking in Drop aborts the process if we are already unwinding,
        // so avoid `expect` here.
        if let Some(callback) = self.callback.take() {
            callback.on_drop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn fires_on_drop() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_guard = fired.clone();
        {
            let _guard = OnEarlyDropGuard::new(move || {
                fired_for_guard.store(true, Ordering::Relaxed);
            });
        }
        assert!(fired.load(Ordering::Relaxed));
    }

    #[test]
    fn suppresses_when_completed() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_guard = fired.clone();
        {
            let mut guard = OnEarlyDropGuard::new(move || {
                fired_for_guard.store(true, Ordering::Relaxed);
            });
            guard.completed();
        }
        assert!(!fired.load(Ordering::Relaxed));
    }

    #[test]
    fn accepts_custom_on_drop_callback_impl() {
        // Verify a named struct implementing OnDropCallback works through
        // the guard, not just closures via the blanket impl.
        struct Counter(Arc<AtomicBool>);
        impl super::super::traits::OnDropCallback for Counter {
            fn on_drop(self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }

        let fired = Arc::new(AtomicBool::new(false));
        {
            let _guard = OnEarlyDropGuard::new(Counter(fired.clone()));
        }
        assert!(fired.load(Ordering::Relaxed));
    }

    // The guard must be Send and Sync whenever its callback is.
    #[allow(dead_code)]
    fn static_property_guard_is_send_sync() {
        fn assert_send<T: Send>(_: &T) {}
        fn assert_sync<T: Sync>(_: &T) {}

        let guard = OnEarlyDropGuard::new(|| {});
        assert_send(&guard);
        assert_sync(&guard);
    }
}
