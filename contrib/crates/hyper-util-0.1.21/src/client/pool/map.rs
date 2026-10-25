//! Map pool utilities
//!
//! The map isn't a typical `Service`, but rather stand-alone type that can map
//! requests to a key and  service factory. This is because the service is more
//! of a router, and cannot determine which inner service to check for
//! backpressure since it's not know until the request is made.
//!
//! The map implementation allows customization of extracting a key, and how to
//! construct a MakeService for that key.
//!
//! # Example
//!
//! ```rust,ignore
//! # async fn run() {
//! # use hyper_util::client::pool;
//! # let req = http::Request::new(());
//! # let some_http1_connector = || {
//! #     tower::service::service_fn(|_req| async { Ok::<_, &'static str>(()) })
//! # };
//! let mut map = pool::map::Map::builder()
//!     .keys(|uri| (uri.scheme().clone(), uri.authority().clone()))
//!     .values(|_uri| {
//!         some_http1_connector()
//!     })
//!     .build();
//!
//! let resp = map.service(req.uri()).call(req).await;
//! # }
//! ```

use std::collections::HashMap;

// expose the documentation
#[cfg(docsrs)]
pub use self::builder::Builder;

/// A map caching `MakeService`s per key.
///
/// Create one with the [`Map::builder()`].
pub struct Map<T, Req>
where
    T: target::Target<Req>,
{
    map: HashMap<T::Key, T::Service>,
    targeter: T,
}

// impl Map

impl Map<builder::StartHere, builder::StartHere> {
    #[allow(
        rustdoc::broken_intra_doc_links,
        reason = "this link is intended for docs.rs builds"
    )]
    /// Create a [`Builder`] to configure a new `Map`.
    pub fn builder<Dst>() -> builder::Builder<Dst, builder::WantsKeyer, builder::WantsServiceMaker>
    {
        builder::Builder::new()
    }
}

impl<T, Req> Map<T, Req>
where
    T: target::Target<Req>,
{
    fn new(targeter: T) -> Self {
        Map {
            map: HashMap::new(),
            targeter,
        }
    }
}

impl<T, Req> Map<T, Req>
where
    T: target::Target<Req>,
    T::Key: Eq + std::hash::Hash,
{
    /// Get a service after extracting the key from `req`.
    pub fn service(&mut self, req: &Req) -> &mut T::Service {
        let key = self.targeter.key(req);
        self.map
            .entry(key)
            .or_insert_with(|| self.targeter.service(req))
    }

    /// Retains only the services specified by the predicate.
    pub fn retain<F>(&mut self, predicate: F)
    where
        F: FnMut(&T::Key, &mut T::Service) -> bool,
    {
        self.map.retain(predicate);
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

// sealed and unnameable for now
mod target {
    pub trait Target<Dst> {
        type Key;
        type Service;

        fn key(&self, dst: &Dst) -> Self::Key;
        fn service(&self, dst: &Dst) -> Self::Service;
    }
}

// sealed and unnameable for now
mod builder {
    use std::marker::PhantomData;

    /// A builder to configure a `Map`.
    ///
    /// # Unnameable
    ///
    /// This type is normally unnameable, forbidding naming of the type within
    /// code. The type is exposed in the documentation to show which methods
    /// can be publicly called.
    pub struct Builder<Dst, K, S> {
        _dst: PhantomData<fn(Dst)>,
        keys: K,
        svcs: S,
    }

    pub struct WantsKeyer;
    pub struct WantsServiceMaker;

    pub enum StartHere {}

    pub struct Built<K, S> {
        keys: K,
        svcs: S,
    }

    impl<Dst> Builder<Dst, WantsKeyer, WantsServiceMaker> {
        pub(super) fn new() -> Self {
            Builder {
                _dst: PhantomData,
                keys: WantsKeyer,
                svcs: WantsServiceMaker,
            }
        }
    }

    impl<Dst, S> Builder<Dst, WantsKeyer, S> {
        /// Provide a closure that extracts a pool key for the destination.
        pub fn keys<K, KK>(self, keyer: K) -> Builder<Dst, K, S>
        where
            K: Fn(&Dst) -> KK,
        {
            Builder {
                _dst: PhantomData,
                keys: keyer,
                svcs: self.svcs,
            }
        }
    }

    impl<Dst, K> Builder<Dst, K, WantsServiceMaker> {
        /// Provide a closure to create a new `MakeService` for the destination.
        pub fn values<S, SS>(self, svcs: S) -> Builder<Dst, K, S>
        where
            S: Fn(&Dst) -> SS,
        {
            Builder {
                _dst: PhantomData,
                keys: self.keys,
                svcs,
            }
        }
    }

    impl<Dst, K, S> Builder<Dst, K, S>
    where
        Built<K, S>: super::target::Target<Dst>,
        <Built<K, S> as super::target::Target<Dst>>::Key: Eq + std::hash::Hash,
    {
        /// Build the `Map` pool.
        pub fn build(self) -> super::Map<Built<K, S>, Dst> {
            super::Map::new(Built {
                keys: self.keys,
                svcs: self.svcs,
            })
        }
    }

    impl super::target::Target<StartHere> for StartHere {
        type Key = StartHere;
        type Service = StartHere;

        fn key(&self, _: &StartHere) -> Self::Key {
            match *self {}
        }

        fn service(&self, _: &StartHere) -> Self::Service {
            match *self {}
        }
    }

    impl<K, KK, S, SS, Dst> super::target::Target<Dst> for Built<K, S>
    where
        K: Fn(&Dst) -> KK,
        S: Fn(&Dst) -> SS,
        KK: Eq + std::hash::Hash,
    {
        type Key = KK;
        type Service = SS;

        fn key(&self, dst: &Dst) -> Self::Key {
            (self.keys)(dst)
        }

        fn service(&self, dst: &Dst) -> Self::Service {
            (self.svcs)(dst)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        let mut pool = super::Map::builder().keys(|_| "a").values(|_| "b").build();
        pool.service(&"hello");
    }
}
