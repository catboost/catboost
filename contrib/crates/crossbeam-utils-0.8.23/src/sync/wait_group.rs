use crate::primitive::sync::atomic::{AtomicUsize, Ordering};
use crate::primitive::sync::{Arc, Condvar, Mutex};
use core::mem::ManuallyDrop;
use std::fmt;

/// Enables threads to synchronize the beginning or end of some computation.
///
/// # Wait groups vs barriers
///
/// `WaitGroup` is very similar to [`Barrier`], but there are a few differences:
///
/// * [`Barrier`] needs to know the number of threads at construction, while `WaitGroup` is cloned to
///   register more threads.
///
/// * A [`Barrier`] can be reused even after all threads have synchronized, while a `WaitGroup`
///   synchronizes threads only once.
///
/// * All threads wait for others to reach the [`Barrier`]. With `WaitGroup`, each thread can choose
///   to either wait for other threads or to continue without blocking.
///
/// # Examples
///
/// ```
/// use crossbeam_utils::sync::WaitGroup;
/// use std::thread;
///
/// // Create a new wait group.
/// let wg = WaitGroup::new();
///
/// for _ in 0..4 {
///     // Create another reference to the wait group.
///     let wg = wg.clone();
///
///     thread::spawn(move || {
///         // Do some work.
///
///         // Drop the reference to the wait group.
///         drop(wg);
///     });
/// }
///
/// // Block until all threads have finished their work.
/// wg.wait();
/// # std::thread::sleep(std::time::Duration::from_millis(500)); // wait for background threads closed: https://github.com/rust-lang/miri/issues/1371
/// ```
///
/// [`Barrier`]: std::sync::Barrier
pub struct WaitGroup {
    inner: Arc<Inner>,
}

/// Inner state of a `WaitGroup`.
struct Inner {
    cvar: Condvar,
    lock: Mutex<()>,
    count: AtomicUsize,
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self {
            inner: Arc::new(Inner {
                cvar: Condvar::new(),
                lock: Mutex::new(()),
                count: AtomicUsize::new(1),
            }),
        }
    }
}

impl WaitGroup {
    /// Creates a new wait group and returns the single reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::sync::WaitGroup;
    ///
    /// let wg = WaitGroup::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops this reference and waits until all other references are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::sync::WaitGroup;
    /// use std::thread;
    ///
    /// let wg = WaitGroup::new();
    ///
    /// thread::spawn({
    ///     let wg = wg.clone();
    ///     move || {
    ///         // Block until both threads have reached `wait()`.
    ///         wg.wait();
    ///     }
    /// });
    ///
    /// // Block until both threads have reached `wait()`.
    /// wg.wait();
    /// # std::thread::sleep(std::time::Duration::from_millis(500)); // wait for background threads closed: https://github.com/rust-lang/miri/issues/1371
    /// ```
    pub fn wait(self) {
        // SAFETY: this is equivalent to let Self { inner } = self, without calling our Drop.
        let inner = unsafe {
            let slf = ManuallyDrop::new(self);
            core::ptr::read(&slf.inner)
        };

        if inner.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Acquire lock after updating count, see below.
            drop(inner.lock.lock().unwrap());
            inner.cvar.notify_all();
            return;
        }

        // We check the counter while holding the lock, and notifiers acquire
        // the lock between updating the counter and notifying, ensuring we
        // can not miss the notification.
        let mut guard = inner.lock.lock().unwrap();
        while inner.count.load(Ordering::Acquire) != 0 {
            guard = inner.cvar.wait(guard).unwrap();
        }
    }
}

impl Drop for WaitGroup {
    fn drop(&mut self) {
        if self.inner.count.fetch_sub(1, Ordering::Release) == 1 {
            // Acquire lock after updating count, see wait().
            drop(self.inner.lock.lock().unwrap());
            self.inner.cvar.notify_all();
        }
    }
}

impl Clone for WaitGroup {
    fn clone(&self) -> Self {
        self.inner.count.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for WaitGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.inner.count.load(Ordering::Relaxed);
        f.debug_struct("WaitGroup").field("count", &count).finish()
    }
}
