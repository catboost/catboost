//! A cache of services
//!
//! The cache is a single list of cached services, bundled with a `MakeService`.
//! Calling the cache returns either an existing service, or makes a new one.
//! The returned `impl Service` can be used to send requests, and when dropped,
//! it will try to be returned back to the cache.

pub use self::internal::builder;

#[cfg(docsrs)]
pub use self::internal::Builder;
#[cfg(docsrs)]
pub use self::internal::Cache;
#[cfg(docsrs)]
pub use self::internal::Cached;

// For now, nothing else in this module is nameable. We can always make things
// more public, but we can't change type shapes (generics) once things are
// public.
mod internal {
    use std::collections::VecDeque;
    use std::fmt;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex, Weak};
    use std::task::{self, Poll, Waker, ready};

    use tower_service::Service;

    use super::events;

    /// Start a builder to construct a `Cache` pool.
    pub fn builder() -> Builder<events::Ignore> {
        Builder {
            events: events::Ignore,
        }
    }

    /// A cache pool of services from the inner make service.
    ///
    /// Created with [`builder()`].
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    #[derive(Debug)]
    pub struct Cache<M, Dst, Ev>
    where
        M: Service<Dst>,
    {
        connector: M,
        shared: Arc<Mutex<Shared<M::Response>>>,
        events: Ev,
        ready: Ready<M::Response>,
        ready_waiter: Option<WaiterId>,
    }

    /// A builder to configure a `Cache`.
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    #[derive(Debug)]
    pub struct Builder<Ev> {
        events: Ev,
    }

    /// A cached service returned from a [`Cache`].
    ///
    /// Implements `Service` by delegating to the inner service. Once dropped,
    /// tries to reinsert into the `Cache`.
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    pub struct Cached<S> {
        is_closed: bool,
        inner: Option<S>,
        shared: Weak<Mutex<Shared<S>>>,
        // todo: on_idle
    }

    #[derive(Debug)]
    enum Ready<S> {
        None,
        Cached(S),
    }

    pub enum CacheFuture<M, Dst, Ev>
    where
        M: Service<Dst>,
    {
        Racing {
            shared: Arc<Mutex<Shared<M::Response>>>,
            waiter: WaiterId,
            future: Option<M::Future>,
            events: Ev,
        },
        Cached {
            svc: Option<Cached<M::Response>>,
        },
    }

    // shouldn't be pub
    #[derive(Debug)]
    pub struct Shared<S> {
        services: Vec<S>,
        waiters: VecDeque<Waiter>,
        reservations: Vec<(WaiterId, S)>,
        next_waiter: usize,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WaiterId(usize);

    #[derive(Debug)]
    struct Waiter {
        id: WaiterId,
        waker: Option<Waker>,
    }

    // impl Builder

    impl<Ev> Builder<Ev> {
        /// Provide a `Future` executor to be used by the `Cache`.
        ///
        /// The executor is used handle some optional background tasks that
        /// can improve the behavior of the cache, such as reducing connection
        /// thrashing when a race is won. If not configured with an executor,
        /// the default behavior is to ignore any of these optional background
        /// tasks.
        ///
        /// The executor should implmenent [`hyper::rt::Executor`].
        ///
        /// # Example
        ///
        /// ```rust
        /// # #[cfg(feature = "tokio")]
        /// # fn run() {
        /// let builder = hyper_util::client::pool::cache::builder()
        ///     .executor(hyper_util::rt::TokioExecutor::new());
        /// # }
        /// ```
        pub fn executor<E>(self, exec: E) -> Builder<events::WithExecutor<E>> {
            Builder {
                events: events::WithExecutor(exec),
            }
        }

        /// Build a `Cache` pool around the `connector`.
        pub fn build<M, Dst>(self, connector: M) -> Cache<M, Dst, Ev>
        where
            M: Service<Dst>,
        {
            Cache {
                connector,
                events: self.events,
                ready: Ready::None,
                ready_waiter: None,
                shared: Arc::new(Mutex::new(Shared {
                    services: Vec::new(),
                    waiters: VecDeque::new(),
                    reservations: Vec::new(),
                    next_waiter: 0,
                })),
            }
        }
    }

    // impl Cache

    impl<M, Dst, Ev> Cache<M, Dst, Ev>
    where
        M: Service<Dst>,
    {
        /// Retain all cached services indicated by the predicate.
        pub fn retain<F>(&mut self, predicate: F)
        where
            F: FnMut(&mut M::Response) -> bool,
        {
            let mut predicate = predicate;
            if let Ready::Cached(svc) = &mut self.ready {
                if !predicate(svc) {
                    self.ready = Ready::None;
                }
            }

            self.shared.lock().unwrap().services.retain_mut(predicate);
        }

        /// Check whether this cache has no cached services.
        pub fn is_empty(&self) -> bool {
            matches!(self.ready, Ready::None) && self.shared.lock().unwrap().services.is_empty()
        }
    }

    impl<M, Dst, Ev> Service<Dst> for Cache<M, Dst, Ev>
    where
        M: Service<Dst>,
        M::Future: Unpin,
        M::Response: Unpin,
        Ev: events::Events<BackgroundConnect<M::Future, M::Response>> + Clone + Unpin,
    {
        type Response = Cached<M::Response>;
        type Error = M::Error;
        type Future = CacheFuture<M, Dst, Ev>;

        fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            match self.ready {
                Ready::Cached(_) => return Poll::Ready(Ok(())),
                Ready::None => {}
            }

            {
                let mut shared = self.shared.lock().unwrap();
                if let Some(id) = self.ready_waiter {
                    if let Some(svc) = shared.take_reserved(id) {
                        self.ready_waiter = None;
                        self.ready = Ready::Cached(svc);
                        return Poll::Ready(Ok(()));
                    }
                } else if let Some(svc) = shared.take_available() {
                    self.ready = Ready::Cached(svc);
                    return Poll::Ready(Ok(()));
                }

                let id = *self
                    .ready_waiter
                    .get_or_insert_with(|| shared.push_waiter());
                shared.store_waker(id, cx.waker());
            }

            match self.connector.poll_ready(cx) {
                Poll::Ready(result) => {
                    if let Some(id) = self.ready_waiter.take() {
                        self.shared.lock().unwrap().cancel_waiter(id);
                    }
                    Poll::Ready(result)
                }
                Poll::Pending => Poll::Pending,
            }
        }

        fn call(&mut self, target: Dst) -> Self::Future {
            // 1. If already cached, easy!
            match std::mem::replace(&mut self.ready, Ready::None) {
                Ready::Cached(svc) => {
                    return CacheFuture::Cached {
                        svc: Some(Cached::new(svc, Arc::downgrade(&self.shared))),
                    };
                }
                Ready::None => {
                    if let Some(id) = self.ready_waiter.take() {
                        let mut shared = self.shared.lock().unwrap();
                        if let Some(svc) = shared.take_reserved(id) {
                            return CacheFuture::Cached {
                                svc: Some(Cached::new(svc, Arc::downgrade(&self.shared))),
                            };
                        }
                        shared.cancel_waiter(id);
                    }
                    if let Some(svc) = self.shared.lock().unwrap().take_available() {
                        return CacheFuture::Cached {
                            svc: Some(Cached::new(svc, Arc::downgrade(&self.shared))),
                        };
                    }
                }
            }

            let waiter = {
                let mut locked = self.shared.lock().unwrap();
                locked.push_waiter()
            };

            // 2. Otherwise, we start a new connect, and also listen for
            //    any newly idle.
            CacheFuture::Racing {
                shared: self.shared.clone(),
                waiter,
                future: Some(self.connector.call(target)),
                events: self.events.clone(),
            }
        }
    }

    impl<M, Dst, Ev> Clone for Cache<M, Dst, Ev>
    where
        M: Service<Dst> + Clone,
        Ev: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                connector: self.connector.clone(),
                events: self.events.clone(),
                shared: self.shared.clone(),
                ready: Ready::None,
                ready_waiter: None,
            }
        }
    }

    impl<M, Dst, Ev> Drop for Cache<M, Dst, Ev>
    where
        M: Service<Dst>,
    {
        fn drop(&mut self) {
            if let Ready::Cached(svc) = std::mem::replace(&mut self.ready, Ready::None) {
                if let Ok(mut shared) = self.shared.lock() {
                    shared.put(svc);
                }
            }
            if let Some(id) = self.ready_waiter.take() {
                if let Ok(mut shared) = self.shared.lock() {
                    shared.cancel_waiter(id);
                }
            }
        }
    }

    impl<M, Dst, Ev> Drop for CacheFuture<M, Dst, Ev>
    where
        M: Service<Dst>,
    {
        fn drop(&mut self) {
            if let CacheFuture::Racing { shared, waiter, .. } = self {
                if let Ok(mut shared) = shared.lock() {
                    shared.cancel_waiter(*waiter);
                }
            }
        }
    }

    impl<M, Dst, Ev> Future for CacheFuture<M, Dst, Ev>
    where
        M: Service<Dst>,
        M::Future: Unpin,
        M::Response: Unpin,
        Ev: events::Events<BackgroundConnect<M::Future, M::Response>> + Unpin,
    {
        type Output = Result<Cached<M::Response>, M::Error>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
            match &mut *self.as_mut() {
                CacheFuture::Racing {
                    shared,
                    waiter,
                    future,
                    events,
                } => {
                    {
                        let mut locked = shared.lock().unwrap();
                        if let Some(pool_got) = locked.take_reserved(*waiter) {
                            events.on_race_lost(BackgroundConnect {
                                future: future.take().expect("racing future polled after done"),
                                shared: Arc::downgrade(&shared),
                            });
                            return Poll::Ready(Ok(Cached::new(pool_got, Arc::downgrade(&shared))));
                        }
                        locked.store_waker(*waiter, cx.waker());
                    }

                    let connected = match ready!(
                        Pin::new(future.as_mut().expect("racing future polled after done"))
                            .poll(cx)
                    ) {
                        Ok(inner) => inner,
                        Err(err) => {
                            shared.lock().unwrap().cancel_waiter(*waiter);
                            return Poll::Ready(Err(err));
                        }
                    };

                    shared.lock().unwrap().cancel_waiter(*waiter);
                    Poll::Ready(Ok(Cached::new(connected, Arc::downgrade(&shared))))
                }
                CacheFuture::Cached { svc } => Poll::Ready(Ok(svc.take().unwrap())),
            }
        }
    }

    // impl Cached

    impl<S> Cached<S> {
        fn new(inner: S, shared: Weak<Mutex<Shared<S>>>) -> Self {
            Cached {
                is_closed: false,
                inner: Some(inner),
                shared,
            }
        }

        // TODO: inner()? looks like `tower` likes `get_ref()` and `get_mut()`.

        /// Get a reference to the inner service.
        pub fn inner(&self) -> &S {
            self.inner.as_ref().expect("inner only taken in drop")
        }

        /// Get a mutable reference to the inner service.
        pub fn inner_mut(&mut self) -> &mut S {
            self.inner.as_mut().expect("inner only taken in drop")
        }
    }

    impl<S, Req> Service<Req> for Cached<S>
    where
        S: Service<Req>,
    {
        type Response = S::Response;
        type Error = S::Error;
        type Future = S::Future;

        fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.inner.as_mut().unwrap().poll_ready(cx).map_err(|err| {
                self.is_closed = true;
                err
            })
        }

        fn call(&mut self, req: Req) -> Self::Future {
            self.inner.as_mut().unwrap().call(req)
        }
    }

    impl<S> Drop for Cached<S> {
        fn drop(&mut self) {
            if self.is_closed {
                return;
            }
            if let Some(value) = self.inner.take() {
                if let Some(shared) = self.shared.upgrade() {
                    if let Ok(mut shared) = shared.lock() {
                        shared.put(value);
                    }
                }
            }
        }
    }

    impl<S: fmt::Debug> fmt::Debug for Cached<S> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("Cached")
                .field(self.inner.as_ref().unwrap())
                .finish()
        }
    }

    // impl Shared

    impl<V> Shared<V> {
        fn put(&mut self, val: V) {
            if let Some(mut waiter) = self.waiters.pop_front() {
                self.reservations.push((waiter.id, val));
                if let Some(waker) = waiter.waker.take() {
                    waker.wake();
                }
                return;
            }

            self.services.push(val);
        }

        fn take_available(&mut self) -> Option<V> {
            if self.waiters.is_empty() {
                self.services.pop()
            } else {
                None
            }
        }

        fn push_waiter(&mut self) -> WaiterId {
            let id = WaiterId(self.next_waiter);
            self.next_waiter = self.next_waiter.wrapping_add(1);
            self.waiters.push_back(Waiter { id, waker: None });
            id
        }

        fn store_waker(&mut self, id: WaiterId, waker: &Waker) {
            if let Some(waiter) = self.waiters.iter_mut().find(|waiter| waiter.id == id) {
                if waiter
                    .waker
                    .as_ref()
                    .is_none_or(|current| !current.will_wake(waker))
                {
                    waiter.waker = Some(waker.clone());
                }
            }
        }

        fn take_reserved(&mut self, id: WaiterId) -> Option<V> {
            let index = self
                .reservations
                .iter()
                .position(|(reserved_id, _)| *reserved_id == id)?;
            Some(self.reservations.remove(index).1)
        }

        fn cancel_waiter(&mut self, id: WaiterId) {
            if let Some(index) = self.waiters.iter().position(|waiter| waiter.id == id) {
                self.waiters.remove(index);
                return;
            }

            if let Some(svc) = self.take_reserved(id) {
                self.put(svc);
            }
        }
    }

    pub struct BackgroundConnect<CF, S> {
        future: CF,
        shared: Weak<Mutex<Shared<S>>>,
    }

    impl<CF, S, E> Future for BackgroundConnect<CF, S>
    where
        CF: Future<Output = Result<S, E>> + Unpin,
    {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
            match ready!(Pin::new(&mut self.future).poll(cx)) {
                Ok(svc) => {
                    if let Some(shared) = self.shared.upgrade() {
                        if let Ok(mut locked) = shared.lock() {
                            locked.put(svc);
                        }
                    }
                    Poll::Ready(())
                }
                Err(_e) => Poll::Ready(()),
            }
        }
    }
}

mod events {
    #[derive(Clone, Debug)]
    #[non_exhaustive]
    pub struct Ignore;

    #[derive(Clone, Debug)]
    pub struct WithExecutor<E>(pub(super) E);

    pub trait Events<CF> {
        fn on_race_lost(&self, fut: CF);
    }

    impl<CF> Events<CF> for Ignore {
        fn on_race_lost(&self, _fut: CF) {}
    }

    impl<E, CF> Events<CF> for WithExecutor<E>
    where
        E: hyper::rt::Executor<CF>,
    {
        fn on_race_lost(&self, fut: CF) {
            self.0.execute(fut);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use std::task::{self, Poll};

    use futures_util::future;
    use tower_service::Service;
    use tower_test::assert_request_eq;

    #[tokio::test]
    async fn test_makes_svc_when_empty() {
        let (mock, mut handle) = tower_test::mock::pair();
        let mut cache = super::builder().build(mock);
        handle.allow(1);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();

        let f = cache.call(1);

        future::join(f, async move {
            assert_request_eq!(handle, 1).send_response("one");
        })
        .await
        .0
        .expect("call");
    }

    #[tokio::test]
    async fn test_reuses_after_idle() {
        let (mock, mut handle) = tower_test::mock::pair();
        let mut cache = super::builder().build(mock);

        // only 1 connection should ever be made
        handle.allow(1);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let f = cache.call(1);
        let cached = future::join(f, async {
            assert_request_eq!(handle, 1).send_response("one");
        })
        .await
        .0
        .expect("call");
        drop(cached);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let f = cache.call(1);
        let cached = f.await.expect("call");
        drop(cached);
    }

    // A returned connection is handed to waiters in the order they parked
    // (FIFO), so a waiter cannot be starved by later arrivals.
    #[tokio::test]
    async fn test_waiters_woken_in_fifo_order() {
        use std::task::{Context, Poll, Waker};

        let (mock, mut handle) = tower_test::mock::pair::<u32, &'static str>();
        let mut cache = super::builder().build(mock);
        handle.allow(16);

        // Establish one connection and hold it, so the next checkouts find no
        // idle service and park.
        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let held = future::join(cache.call(0), async {
            assert_request_eq!(handle, 0).send_response("conn");
        })
        .await
        .0
        .expect("call");

        // Park three checkouts in order. Each misses and starts a connect, but
        // the connect is never completed, so each parks on its waiter.
        let mut cx = Context::from_waker(Waker::noop());
        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let mut first = Box::pin(cache.call(1));
        assert!(first.as_mut().poll(&mut cx).is_pending());
        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let mut second = Box::pin(cache.call(2));
        assert!(second.as_mut().poll(&mut cx).is_pending());
        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let mut third = Box::pin(cache.call(3));
        assert!(third.as_mut().poll(&mut cx).is_pending());

        // Returning the connection wakes the oldest waiter first.
        drop(held);
        let first = match first.as_mut().poll(&mut cx) {
            Poll::Ready(r) => r.expect("first"),
            Poll::Pending => panic!("oldest waiter was not woken first"),
        };
        assert!(second.as_mut().poll(&mut cx).is_pending());
        assert!(third.as_mut().poll(&mut cx).is_pending());

        // Returning it again wakes the next-oldest, then the last.
        drop(first);
        let second = match second.as_mut().poll(&mut cx) {
            Poll::Ready(r) => r.expect("second"),
            Poll::Pending => panic!("second waiter was not woken next"),
        };
        assert!(third.as_mut().poll(&mut cx).is_pending());

        drop(second);
        match third.as_mut().poll(&mut cx) {
            Poll::Ready(r) => {
                r.expect("third");
            }
            Poll::Pending => panic!("last waiter was not woken"),
        }
    }

    #[tokio::test]
    async fn dropped_racing_future_cancels_waiter() {
        use std::task::{Context, Poll, Waker};

        let (mock, mut handle) = tower_test::mock::pair::<u32, &'static str>();
        let mut cache = super::builder().build(mock);
        handle.allow(16);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let held = future::join(cache.call(0), async {
            assert_request_eq!(handle, 0).send_response("conn");
        })
        .await
        .0
        .expect("call");

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let mut dropped = Box::pin(cache.call(1));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(dropped.as_mut().poll(&mut cx).is_pending());
        drop(dropped);

        drop(held);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let mut reused = Box::pin(cache.call(2));
        match reused.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(cached)) => {
                assert_eq!(*cached.inner(), "conn");
            }
            Poll::Ready(Err(err)) => panic!("unexpected error: {err}"),
            Poll::Pending => panic!("dropped waiter blocked idle reuse"),
        }
    }

    #[tokio::test]
    async fn clone_readiness_reserves_idle_service() {
        let connector = StrictConnector::default();
        let poll_ready_count = connector.poll_ready_count.clone();
        let calls = connector.calls.clone();
        let mut cache = super::builder().build(connector);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let cached = cache.call(1).await.unwrap();
        assert_eq!(*cached.inner(), 0);
        drop(cached);

        let mut a = cache.clone();
        let mut b = cache.clone();

        std::future::poll_fn(|cx| a.poll_ready(cx)).await.unwrap();
        assert_eq!(poll_ready_count.load(Ordering::SeqCst), 1);
        assert!(!a.is_empty());

        std::future::poll_fn(|cx| b.poll_ready(cx)).await.unwrap();
        assert_eq!(poll_ready_count.load(Ordering::SeqCst), 2);

        let a_cached = a.call(10).await.unwrap();
        assert_eq!(*a_cached.inner(), 0);

        let b_cached = b.call(20).await.unwrap();
        assert_eq!(*b_cached.inner(), 1);

        assert_eq!(*calls.lock().unwrap(), vec![1, 20]);
    }

    #[tokio::test]
    async fn dropped_ready_slot_returns_idle_service() {
        let connector = StrictConnector::default();
        let poll_ready_count = connector.poll_ready_count.clone();
        let mut cache = super::builder().build(connector);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let cached = cache.call(1).await.unwrap();
        drop(cached);

        let mut clone = cache.clone();
        std::future::poll_fn(|cx| clone.poll_ready(cx))
            .await
            .unwrap();
        drop(clone);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        assert_eq!(poll_ready_count.load(Ordering::SeqCst), 1);

        let cached = cache.call(2).await.unwrap();
        assert_eq!(*cached.inner(), 0);
    }

    #[tokio::test]
    async fn retain_checks_ready_slot() {
        let connector = StrictConnector::default();
        let poll_ready_count = connector.poll_ready_count.clone();
        let mut cache = super::builder().build(connector);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let cached = cache.call(1).await.unwrap();
        drop(cached);

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        assert!(!cache.is_empty());

        cache.retain(|svc| *svc != 0);
        assert!(cache.is_empty());

        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        assert_eq!(poll_ready_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn idle_return_wakes_pending_poll_ready() {
        use std::sync::atomic::AtomicBool;
        use std::task::{Context, Waker};

        let connector = PendingConnector::default();
        let allow_ready = connector.allow_ready.clone();
        let mut cache = super::builder().build(connector);

        allow_ready.store(true, Ordering::SeqCst);
        std::future::poll_fn(|cx| cache.poll_ready(cx))
            .await
            .unwrap();
        let held = cache.call(1).await.unwrap();
        assert_eq!(*held.inner(), 0);

        let mut ready = Box::pin(std::future::poll_fn(|cx| cache.poll_ready(cx)));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(ready.as_mut().poll(&mut cx).is_pending());

        drop(held);

        match ready.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(err)) => match err {},
            Poll::Pending => panic!("idle return did not wake pending poll_ready"),
        }
        drop(ready);

        let cached = cache.call(2).await.unwrap();
        assert_eq!(*cached.inner(), 0);

        #[derive(Default)]
        struct PendingConnector {
            allow_ready: Arc<AtomicBool>,
            next: Arc<AtomicUsize>,
            ready: bool,
        }

        impl Service<usize> for PendingConnector {
            type Response = usize;
            type Error = Infallible;
            type Future = std::future::Ready<Result<usize, Infallible>>;

            fn poll_ready(&mut self, _cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
                if self.allow_ready.swap(false, Ordering::SeqCst) {
                    self.ready = true;
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            }

            fn call(&mut self, _target: usize) -> Self::Future {
                assert!(self.ready, "connector called without poll_ready");
                self.ready = false;
                let id = self.next.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Ok(id))
            }
        }
    }

    #[derive(Default)]
    struct StrictConnector {
        poll_ready_count: Arc<AtomicUsize>,
        next: Arc<AtomicUsize>,
        calls: Arc<Mutex<Vec<usize>>>,
        ready: bool,
    }

    impl Clone for StrictConnector {
        fn clone(&self) -> Self {
            StrictConnector {
                poll_ready_count: self.poll_ready_count.clone(),
                next: self.next.clone(),
                calls: self.calls.clone(),
                ready: false,
            }
        }
    }

    impl Service<usize> for StrictConnector {
        type Response = usize;
        type Error = Infallible;
        type Future = std::future::Ready<Result<usize, Infallible>>;

        fn poll_ready(&mut self, _cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.ready = true;
            self.poll_ready_count.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, target: usize) -> Self::Future {
            assert!(self.ready, "connector called without poll_ready");
            self.ready = false;
            self.calls.lock().unwrap().push(target);
            let id = self.next.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(id))
        }
    }
}
