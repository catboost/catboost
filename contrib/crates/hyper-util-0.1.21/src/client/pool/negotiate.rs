//! Negotiate a pool of services
//!
//! The negotiate pool allows for a service that can decide between two service
//! types based on an intermediate return value. It differs from typical
//! routing since it doesn't depend on the request, but the response.
//!
//! The original use case is support ALPN upgrades to HTTP/2, with a fallback
//! to HTTP/1.
//!
//! # Example
//!
//! ```rust,ignore
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # struct Conn;
//! # impl Conn { fn negotiated_protocol(&self) -> &[u8] { b"h2" } }
//! # let some_tls_connector = tower::service::service_fn(|_| async move {
//! #     Ok::<_, std::convert::Infallible>(Conn)
//! # });
//! # let http1_layer = tower::layer::layer_fn(|s| s);
//! # let http2_layer = tower::layer::layer_fn(|s| s);
//! let mut pool = hyper_util::client::pool::negotiate::builder()
//!     .connect(some_tls_connector)
//!     .inspect(|c| c.negotiated_protocol() == b"h2")
//!     .fallback(http1_layer)
//!     .upgrade(http2_layer)
//!     .build();
//!
//! // connect
//! let mut svc = pool.call(http::Uri::from_static("https://hyper.rs")).await?;
//! svc.ready().await;
//!
//! // http1 or http2 is now set up
//! # let some_http_req = http::Request::new(());
//! let resp = svc.call(some_http_req).await?;
//! # Ok(())
//! # }
//! ```

pub use self::internal::builder;

#[cfg(docsrs)]
pub use self::internal::Builder;
#[cfg(docsrs)]
pub use self::internal::Negotiate;
#[cfg(docsrs)]
pub use self::internal::Negotiated;

mod internal {
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{self, Poll, ready};

    use pin_project_lite::pin_project;
    use tower_layer::Layer;
    use tower_service::Service;

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    /// A negotiating pool over an inner make service.
    ///
    /// Created with [`builder()`].
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    #[derive(Clone)]
    pub struct Negotiate<L, R> {
        left: L,
        right: R,
    }

    /// A negotiated service returned by [`Negotiate`].
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    #[derive(Clone, Debug)]
    pub enum Negotiated<L, R> {
        #[doc(hidden)]
        Fallback(L),
        #[doc(hidden)]
        Upgraded(R),
    }

    pin_project! {
        pub struct Negotiating<Dst, L, R>
        where
            L: Service<Dst>,
            R: Service<()>,
        {
            #[pin]
            state: State<Dst, L::Future, R::Future>,
            left: L,
            right: R,
        }
    }

    pin_project! {
        #[project = StateProj]
        enum State<Dst, FL, FR> {
            Eager {
                #[pin]
                future: FR,
                dst: Option<Dst>,
            },
            Fallback {
                #[pin]
                future: FL,
            },
            Upgrade {
                #[pin]
                future: FR,
            }
        }
    }

    pin_project! {
        #[project = NegotiatedProj]
        pub enum NegotiatedFuture<L, R> {
            Fallback {
                #[pin]
                future: L
            },
            Upgraded {
                #[pin]
                future: R
            },
        }
    }

    /// A builder to configure a `Negotiate`.
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    #[derive(Debug)]
    pub struct Builder<C, I, L, R> {
        connect: C,
        inspect: I,
        fallback: L,
        upgrade: R,
    }

    #[derive(Debug)]
    pub struct WantsConnect;
    #[derive(Debug)]
    pub struct WantsInspect;
    #[derive(Debug)]
    pub struct WantsFallback;
    #[derive(Debug)]
    pub struct WantsUpgrade;

    /// Start a builder to construct a `Negotiate` pool.
    pub fn builder() -> Builder<WantsConnect, WantsInspect, WantsFallback, WantsUpgrade> {
        Builder {
            connect: WantsConnect,
            inspect: WantsInspect,
            fallback: WantsFallback,
            upgrade: WantsUpgrade,
        }
    }

    impl<C, I, L, R> Builder<C, I, L, R> {
        /// Provide the initial connector.
        pub fn connect<CC>(self, connect: CC) -> Builder<CC, I, L, R> {
            Builder {
                connect,
                inspect: self.inspect,
                fallback: self.fallback,
                upgrade: self.upgrade,
            }
        }

        /// Provide the inspector that determines the result of the negotiation.
        pub fn inspect<II>(self, inspect: II) -> Builder<C, II, L, R> {
            Builder {
                connect: self.connect,
                inspect,
                fallback: self.fallback,
                upgrade: self.upgrade,
            }
        }

        /// Provide the layer to fallback to if negotiation fails.
        pub fn fallback<LL>(self, fallback: LL) -> Builder<C, I, LL, R> {
            Builder {
                connect: self.connect,
                inspect: self.inspect,
                fallback,
                upgrade: self.upgrade,
            }
        }

        /// Provide the layer to upgrade to if negotiation succeeds.
        pub fn upgrade<RR>(self, upgrade: RR) -> Builder<C, I, L, RR> {
            Builder {
                connect: self.connect,
                inspect: self.inspect,
                fallback: self.fallback,
                upgrade,
            }
        }

        /// Build the `Negotiate` pool.
        pub fn build<Dst>(self) -> Negotiate<L::Service, R::Service>
        where
            C: Service<Dst>,
            C::Error: Into<BoxError>,
            L: Layer<Inspector<C, C::Response, I>>,
            L::Service: Service<Dst> + Clone,
            <L::Service as Service<Dst>>::Error: Into<BoxError>,
            R: Layer<Inspected<C::Response>>,
            R::Service: Service<()> + Clone,
            <R::Service as Service<()>>::Error: Into<BoxError>,
            I: Fn(&C::Response) -> bool + Clone,
        {
            let Builder {
                connect,
                inspect,
                fallback,
                upgrade,
            } = self;

            let slot = Arc::new(Mutex::new(None));
            let wrapped = Inspector {
                svc: connect,
                inspect,
                slot: slot.clone(),
            };
            let left = fallback.layer(wrapped);

            let right = upgrade.layer(Inspected { slot });

            Negotiate { left, right }
        }
    }

    impl<L, R> Negotiate<L, R> {
        /// Get a mutable reference to the fallback service.
        pub fn fallback_mut(&mut self) -> &mut L {
            &mut self.left
        }

        /// Get a mutable reference to the upgrade service.
        pub fn upgrade_mut(&mut self) -> &mut R {
            &mut self.right
        }
    }

    impl<L, R, Target> Service<Target> for Negotiate<L, R>
    where
        L: Service<Target> + Clone,
        L::Error: Into<BoxError>,
        R: Service<()> + Clone,
        R::Error: Into<BoxError>,
    {
        type Response = Negotiated<L::Response, R::Response>;
        type Error = BoxError;
        type Future = Negotiating<Target, L, R>;

        fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.left.poll_ready(cx).map_err(Into::into)
        }

        fn call(&mut self, dst: Target) -> Self::Future {
            let left = self.left.clone();
            Negotiating {
                state: State::Eager {
                    future: self.right.call(()),
                    dst: Some(dst),
                },
                // place clone, take original that we already polled-ready.
                left: std::mem::replace(&mut self.left, left),
                right: self.right.clone(),
            }
        }
    }

    impl<Dst, L, R> Future for Negotiating<Dst, L, R>
    where
        L: Service<Dst>,
        L::Error: Into<BoxError>,
        R: Service<()>,
        R::Error: Into<BoxError>,
    {
        type Output = Result<Negotiated<L::Response, R::Response>, BoxError>;

        fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
            // States:
            // - `Eager`: try the "right" path first; on `UseOther` sentinel, fall back to left.
            // - `Fallback`: try the left path; on `UseOther` sentinel, upgrade back to right.
            // - `Upgrade`: retry the right path after a fallback.
            // If all fail, give up.
            let mut me = self.project();
            loop {
                match me.state.as_mut().project() {
                    StateProj::Eager { future, dst } => match ready!(future.poll(cx)) {
                        Ok(out) => return Poll::Ready(Ok(Negotiated::Upgraded(out))),
                        Err(err) => {
                            let err = err.into();
                            if UseOther::is(&*err) {
                                let dst = dst.take().unwrap();
                                let f = me.left.call(dst);
                                me.state.set(State::Fallback { future: f });
                                continue;
                            } else {
                                return Poll::Ready(Err(err));
                            }
                        }
                    },
                    StateProj::Fallback { future } => match ready!(future.poll(cx)) {
                        Ok(out) => return Poll::Ready(Ok(Negotiated::Fallback(out))),
                        Err(err) => {
                            let err = err.into();
                            if UseOther::is(&*err) {
                                let f = me.right.call(());
                                me.state.set(State::Upgrade { future: f });
                                continue;
                            } else {
                                return Poll::Ready(Err(err));
                            }
                        }
                    },
                    StateProj::Upgrade { future } => match ready!(future.poll(cx)) {
                        Ok(out) => return Poll::Ready(Ok(Negotiated::Upgraded(out))),
                        Err(err) => return Poll::Ready(Err(err.into())),
                    },
                }
            }
        }
    }

    impl<L, R> Negotiated<L, R> {
        // Could be useful?
        #[cfg(test)]
        pub(super) fn is_fallback(&self) -> bool {
            matches!(self, Negotiated::Fallback(_))
        }

        #[cfg(test)]
        pub(super) fn is_upgraded(&self) -> bool {
            matches!(self, Negotiated::Upgraded(_))
        }

        // TODO: are these the correct methods? Or .as_ref().fallback(), etc?

        /// Get a reference to the fallback service if this is it.
        pub fn fallback_ref(&self) -> Option<&L> {
            if let Negotiated::Fallback(left) = self {
                Some(left)
            } else {
                None
            }
        }

        /// Get a mutable reference to the fallback service if this is it.
        pub fn fallback_mut(&mut self) -> Option<&mut L> {
            if let Negotiated::Fallback(left) = self {
                Some(left)
            } else {
                None
            }
        }

        /// Get a reference to the upgraded service if this is it.
        pub fn upgraded_ref(&self) -> Option<&R> {
            if let Negotiated::Upgraded(right) = self {
                Some(right)
            } else {
                None
            }
        }

        /// Get a mutable reference to the upgraded service if this is it.
        pub fn upgraded_mut(&mut self) -> Option<&mut R> {
            if let Negotiated::Upgraded(right) = self {
                Some(right)
            } else {
                None
            }
        }
    }

    impl<L, R, Req, Res, E> Service<Req> for Negotiated<L, R>
    where
        L: Service<Req, Response = Res, Error = E>,
        R: Service<Req, Response = Res, Error = E>,
    {
        type Response = Res;
        type Error = E;
        type Future = NegotiatedFuture<L::Future, R::Future>;

        fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            match self {
                Negotiated::Fallback(s) => s.poll_ready(cx),
                Negotiated::Upgraded(s) => s.poll_ready(cx),
            }
        }

        fn call(&mut self, req: Req) -> Self::Future {
            match self {
                Negotiated::Fallback(s) => NegotiatedFuture::Fallback {
                    future: s.call(req),
                },
                Negotiated::Upgraded(s) => NegotiatedFuture::Upgraded {
                    future: s.call(req),
                },
            }
        }
    }

    impl<L, R, Out> Future for NegotiatedFuture<L, R>
    where
        L: Future<Output = Out>,
        R: Future<Output = Out>,
    {
        type Output = Out;

        fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
            match self.project() {
                NegotiatedProj::Fallback { future } => future.poll(cx),
                NegotiatedProj::Upgraded { future } => future.poll(cx),
            }
        }
    }

    // ===== internal =====

    pub struct Inspector<M, S, I> {
        svc: M,
        inspect: I,
        slot: Arc<Mutex<Option<S>>>,
    }

    pin_project! {
        pub struct InspectFuture<F, S, I> {
            #[pin]
            future: F,
            inspect: I,
            slot: Arc<Mutex<Option<S>>>,
        }
    }

    impl<M: Clone, S, I: Clone> Clone for Inspector<M, S, I> {
        fn clone(&self) -> Self {
            Self {
                svc: self.svc.clone(),
                inspect: self.inspect.clone(),
                slot: self.slot.clone(),
            }
        }
    }

    impl<M, S, I, Target> Service<Target> for Inspector<M, S, I>
    where
        M: Service<Target, Response = S>,
        M::Error: Into<BoxError>,
        I: Clone + Fn(&S) -> bool,
    {
        type Response = M::Response;
        type Error = BoxError;
        type Future = InspectFuture<M::Future, S, I>;

        fn poll_ready(&mut self, cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.svc.poll_ready(cx).map_err(Into::into)
        }

        fn call(&mut self, dst: Target) -> Self::Future {
            InspectFuture {
                future: self.svc.call(dst),
                inspect: self.inspect.clone(),
                slot: self.slot.clone(),
            }
        }
    }

    impl<F, I, S, E> Future for InspectFuture<F, S, I>
    where
        F: Future<Output = Result<S, E>>,
        E: Into<BoxError>,
        I: Fn(&S) -> bool,
    {
        type Output = Result<S, BoxError>;

        fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
            let me = self.project();
            let s = ready!(me.future.poll(cx)).map_err(Into::into)?;
            Poll::Ready(if (me.inspect)(&s) {
                *me.slot.lock().unwrap() = Some(s);
                Err(UseOther.into())
            } else {
                Ok(s)
            })
        }
    }

    pub struct Inspected<S> {
        slot: Arc<Mutex<Option<S>>>,
    }

    impl<S, Target> Service<Target> for Inspected<S> {
        type Response = S;
        type Error = BoxError;
        type Future = std::future::Ready<Result<S, BoxError>>;

        fn poll_ready(&mut self, _cx: &mut task::Context<'_>) -> Poll<Result<(), Self::Error>> {
            if self.slot.lock().unwrap().is_some() {
                Poll::Ready(Ok(()))
            } else {
                Poll::Ready(Err(UseOther.into()))
            }
        }

        fn call(&mut self, _dst: Target) -> Self::Future {
            let s = self
                .slot
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| UseOther.into());
            std::future::ready(s)
        }
    }

    impl<S> Clone for Inspected<S> {
        fn clone(&self) -> Inspected<S> {
            Inspected {
                slot: self.slot.clone(),
            }
        }
    }

    #[derive(Debug)]
    struct UseOther;

    impl std::fmt::Display for UseOther {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("sentinel error; using other")
        }
    }

    impl std::error::Error for UseOther {}

    impl UseOther {
        fn is(err: &(dyn std::error::Error + 'static)) -> bool {
            let mut source = Some(err);
            while let Some(err) = source {
                if err.is::<UseOther>() {
                    return true;
                }
                source = err.source();
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::future;
    use tower_service::Service;
    use tower_test::assert_request_eq;

    #[tokio::test]
    async fn not_negotiated_falls_back_to_left() {
        let (mock_svc, mut handle) = tower_test::mock::pair::<(), &'static str>();

        let mut negotiate = super::builder()
            .connect(mock_svc)
            .inspect(|_: &&str| false)
            .fallback(layer_fn(|s| s))
            .upgrade(layer_fn(|s| s))
            .build();

        std::future::poll_fn(|cx| negotiate.poll_ready(cx))
            .await
            .unwrap();

        let fut = negotiate.call(());
        let nsvc = future::join(fut, async move {
            assert_request_eq!(handle, ()).send_response("one");
        })
        .await
        .0
        .expect("call");
        assert!(nsvc.is_fallback());
    }

    #[tokio::test]
    async fn negotiated_uses_right() {
        let (mock_svc, mut handle) = tower_test::mock::pair::<(), &'static str>();

        let mut negotiate = super::builder()
            .connect(mock_svc)
            .inspect(|_: &&str| true)
            .fallback(layer_fn(|s| s))
            .upgrade(layer_fn(|s| s))
            .build();

        std::future::poll_fn(|cx| negotiate.poll_ready(cx))
            .await
            .unwrap();

        let fut = negotiate.call(());
        let nsvc = future::join(fut, async move {
            assert_request_eq!(handle, ()).send_response("one");
        })
        .await
        .0
        .expect("call");

        assert!(nsvc.is_upgraded());
    }

    fn layer_fn<F>(f: F) -> LayerFn<F> {
        LayerFn(f)
    }

    #[derive(Clone)]
    struct LayerFn<F>(F);

    impl<F, S, Out> tower_layer::Layer<S> for LayerFn<F>
    where
        F: Fn(S) -> Out,
    {
        type Service = Out;
        fn layer(&self, inner: S) -> Self::Service {
            (self.0)(inner)
        }
    }
}
