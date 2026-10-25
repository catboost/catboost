// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Map object URLs to [`ObjectStore`]

use crate::path::{InvalidPart, Path, PathPart};
use crate::{parse_url_opts, ObjectStore};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use url::Url;

/// [`ObjectStoreRegistry`] maps a URL to an [`ObjectStore`] instance
pub trait ObjectStoreRegistry: Send + Sync + std::fmt::Debug + 'static {
    /// Register a new store for the provided store URL
    ///
    /// If a store with the same URL existed before, it is replaced and returned
    fn register(&self, url: Url, store: Arc<dyn ObjectStore>) -> Option<Arc<dyn ObjectStore>>;

    /// Resolve an object URL
    ///
    /// If [`ObjectStoreRegistry::register`] has been called with a URL with the same
    /// scheme, and authority as the object URL, and a path that is a prefix of the object
    /// URL's, it should be returned along with the trailing path. Paths should be matched
    /// on a path segment basis, and in the event of multiple possibilities the longest
    /// path match should be returned.
    ///
    /// If a store hasn't been registered, an [`ObjectStoreRegistry`] may lazily create
    /// one if the URL is understood
    ///
    /// For example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use url::Url;
    /// # use object_store::memory::InMemory;
    /// # use object_store::ObjectStore;
    /// # use object_store::prefix::PrefixStore;
    /// # use object_store::registry::{DefaultObjectStoreRegistry, ObjectStoreRegistry};
    /// #
    /// let registry = DefaultObjectStoreRegistry::new();
    ///
    /// let bucket1 = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    /// let base = Url::parse("s3://bucket1/").unwrap();
    /// registry.register(base, bucket1.clone());
    ///
    /// let url = Url::parse("s3://bucket1/path/to/object").unwrap();
    /// let (ret, path) = registry.resolve(&url).unwrap();
    /// assert_eq!(path.as_ref(), "path/to/object");
    /// assert!(Arc::ptr_eq(&ret, &bucket1));
    ///
    /// let bucket2 = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    /// let base = Url::parse("https://s3.region.amazonaws.com/bucket").unwrap();
    /// registry.register(base, bucket2.clone());
    ///
    /// let url = Url::parse("https://s3.region.amazonaws.com/bucket/path/to/object").unwrap();
    /// let (ret, path) = registry.resolve(&url).unwrap();
    /// assert_eq!(path.as_ref(), "path/to/object");
    /// assert!(Arc::ptr_eq(&ret, &bucket2));
    ///
    /// let bucket3 = Arc::new(PrefixStore::new(InMemory::new(), "path")) as Arc<dyn ObjectStore>;
    /// let base = Url::parse("https://s3.region.amazonaws.com/bucket/path").unwrap();
    /// registry.register(base, bucket3.clone());
    ///
    /// let url = Url::parse("https://s3.region.amazonaws.com/bucket/path/to/object").unwrap();
    /// let (ret, path) = registry.resolve(&url).unwrap();
    /// assert_eq!(path.as_ref(), "to/object");
    /// assert!(Arc::ptr_eq(&ret, &bucket3));
    /// ```
    fn resolve(&self, url: &Url) -> crate::Result<(Arc<dyn ObjectStore>, Path)>;
}

/// Error type for [`DefaultObjectStoreRegistry`]
///
/// Crate private/opaque type to make the error handling code more ergonomic.
/// Always converted into `crate::Error` when reported externally.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
enum Error {
    #[error("ObjectStore not found")]
    NotFound,

    #[error("Error parsing URL path segment")]
    InvalidPart(#[from] InvalidPart),
}

impl From<Error> for crate::Error {
    fn from(value: Error) -> Self {
        Self::Generic {
            store: "ObjectStoreRegistry",
            source: Box::new(value),
        }
    }
}

/// An [`ObjectStoreRegistry`] that uses [`parse_url_opts`] to create stores based on the environment
#[derive(Debug, Default)]
pub struct DefaultObjectStoreRegistry {
    /// Mapping from [`url_key`] to [`PathEntry`]
    map: RwLock<HashMap<String, PathEntry>>,
}

/// [`PathEntry`] construct a tree of path segments starting from the root
///
/// For example the following paths
///
/// * `/` => store1
/// * `/foo/bar` => store2
///
/// Would be represented by
///
/// ```yaml
/// store: Some(store1)
/// children:
///   foo:
///     store: None
///     children:
///       bar:
///         store: Some(store2)
/// ```
///
#[derive(Debug, Default)]
struct PathEntry {
    /// Store, if defined at this path
    store: Option<Arc<dyn ObjectStore>>,
    /// Child [`PathEntry`], keyed by the next path segment in their path
    children: HashMap<String, Self>,
}

impl PathEntry {
    /// Lookup a store based on URL path
    ///
    /// Returns the store and its path segment depth
    fn lookup(&self, to_resolve: &Url) -> Option<(&Arc<dyn ObjectStore>, usize)> {
        let mut current = self;
        let mut ret = self.store.as_ref().map(|store| (store, 0));
        let mut depth = 0;
        // Traverse the PathEntry tree to find the longest match
        for segment in path_segments(to_resolve.path()) {
            match current.children.get(segment) {
                Some(e) => {
                    current = e;
                    depth += 1;
                    if let Some(store) = &current.store {
                        ret = Some((store, depth))
                    }
                }
                None => break,
            }
        }
        ret
    }
}

impl DefaultObjectStoreRegistry {
    /// Create a new [`DefaultObjectStoreRegistry`]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ObjectStoreRegistry for DefaultObjectStoreRegistry {
    fn register(&self, url: Url, store: Arc<dyn ObjectStore>) -> Option<Arc<dyn ObjectStore>> {
        let mut map = self.map.write();
        let key = url_key(&url);
        let mut entry = map.entry(key.to_string()).or_default();

        for segment in path_segments(url.path()) {
            entry = entry.children.entry(segment.to_string()).or_default();
        }
        entry.store.replace(store)
    }

    fn resolve(&self, to_resolve: &Url) -> crate::Result<(Arc<dyn ObjectStore>, Path)> {
        let key = url_key(to_resolve);
        {
            let map = self.map.read();

            if let Some((store, depth)) = map.get(key).and_then(|entry| entry.lookup(to_resolve)) {
                let path = path_suffix(to_resolve, depth)?;
                return Ok((Arc::clone(store), path));
            }
        }

        if let Ok((store, path)) = parse_url_opts(to_resolve, std::env::vars()) {
            let depth = num_segments(to_resolve.path()) - num_segments(path.as_ref());

            let mut map = self.map.write();
            let mut entry = map.entry(key.to_string()).or_default();
            for segment in path_segments(to_resolve.path()).take(depth) {
                entry = entry.children.entry(segment.to_string()).or_default();
            }
            let store = Arc::clone(match &entry.store {
                None => entry.store.insert(Arc::from(store)),
                Some(x) => x, // Racing creation - use existing
            });

            let path = path_suffix(to_resolve, depth)?;
            return Ok((store, path));
        }

        Err(Error::NotFound.into())
    }
}

/// Extracts the scheme and authority of a URL (components before the Path)
fn url_key(url: &Url) -> &str {
    &url[..url::Position::AfterPort]
}

/// Returns the non-empty segments of a path
///
/// Note: We don't use [`Url::path_segments`] as we only want non-empty paths
fn path_segments(s: &str) -> impl Iterator<Item = &str> {
    s.split('/').filter(|x| !x.is_empty())
}

/// Returns the number of non-empty path segments in a path
fn num_segments(s: &str) -> usize {
    path_segments(s).count()
}

/// Returns the path of `url` skipping the first `depth` segments
fn path_suffix(url: &Url, depth: usize) -> Result<Path, Error> {
    let segments = path_segments(url.path()).skip(depth);
    let path = segments.map(PathPart::parse).collect::<Result<_, _>>()?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemory;
    use crate::prefix::PrefixStore;

    #[test]
    fn test_num_segments() {
        assert_eq!(num_segments(""), 0);
        assert_eq!(num_segments("/"), 0);
        assert_eq!(num_segments("/banana"), 1);
        assert_eq!(num_segments("banana"), 1);
        assert_eq!(num_segments("/banana/crumble"), 2);
        assert_eq!(num_segments("banana/crumble"), 2);
    }

    #[test]
    fn test_default_registry() {
        let registry = DefaultObjectStoreRegistry::new();

        // Should automatically register in memory store
        let banana_url = Url::parse("memory:///banana").unwrap();
        let (resolved, path) = registry.resolve(&banana_url).unwrap();
        assert_eq!(path.as_ref(), "banana");

        // Should replace store
        let url = Url::parse("memory:///").unwrap();
        let root = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let replaced = registry.register(url, Arc::clone(&root)).unwrap();
        assert!(Arc::ptr_eq(&resolved, &replaced));

        // Should not replace store
        let banana = Arc::new(PrefixStore::new(InMemory::new(), "banana")) as Arc<dyn ObjectStore>;
        assert!(registry
            .register(banana_url.clone(), Arc::clone(&banana))
            .is_none());

        // Should resolve to banana store
        let (resolved, path) = registry.resolve(&banana_url).unwrap();
        assert_eq!(path.as_ref(), "");
        assert!(Arc::ptr_eq(&resolved, &banana));

        // If we register another store it still resolves banana
        let apples_url = Url::parse("memory:///apples").unwrap();
        let apples = Arc::new(PrefixStore::new(InMemory::new(), "apples")) as Arc<dyn ObjectStore>;
        assert!(registry.register(apples_url, Arc::clone(&apples)).is_none());

        // Should still resolve to banana store
        let (resolved, path) = registry.resolve(&banana_url).unwrap();
        assert_eq!(path.as_ref(), "");
        assert!(Arc::ptr_eq(&resolved, &banana));

        // Should be path segment based
        let banana_muffins_url = Url::parse("memory:///banana_muffins").unwrap();
        let (resolved, path) = registry.resolve(&banana_muffins_url).unwrap();
        assert_eq!(path.as_ref(), "banana_muffins");
        assert!(Arc::ptr_eq(&resolved, &root));

        // Should resolve to root even though path contains prefix of valid store
        let to_resolve = Url::parse("memory:///foo/banana").unwrap();
        let (resolved, path) = registry.resolve(&to_resolve).unwrap();
        assert_eq!(path.as_ref(), "foo/banana");
        assert!(Arc::ptr_eq(&resolved, &root));

        let nested_url = Url::parse("memory:///apples/bananas").unwrap();
        let nested =
            Arc::new(PrefixStore::new(InMemory::new(), "apples/bananas")) as Arc<dyn ObjectStore>;
        assert!(registry.register(nested_url, Arc::clone(&nested)).is_none());

        let to_resolve = Url::parse("memory:///apples/bananas/muffins/cupcakes").unwrap();
        let (resolved, path) = registry.resolve(&to_resolve).unwrap();
        assert_eq!(path.as_ref(), "muffins/cupcakes");
        assert!(Arc::ptr_eq(&resolved, &nested));

        let nested_url2 = Url::parse("memory:///1/2/3").unwrap();
        let nested2 = Arc::new(PrefixStore::new(InMemory::new(), "1/2/3")) as Arc<dyn ObjectStore>;
        assert!(registry
            .register(nested_url2, Arc::clone(&nested2))
            .is_none());

        let to_resolve = Url::parse("memory:///1/2/3/4/5/6").unwrap();
        let (resolved, path) = registry.resolve(&to_resolve).unwrap();
        assert_eq!(path.as_ref(), "4/5/6");
        assert!(Arc::ptr_eq(&resolved, &nested2));

        let custom_scheme_url = Url::parse("custom:///").unwrap();
        let custom_scheme = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        assert!(registry
            .register(custom_scheme_url, Arc::clone(&custom_scheme))
            .is_none());

        let to_resolve = Url::parse("custom:///6/7").unwrap();
        let (resolved, path) = registry.resolve(&to_resolve).unwrap();
        assert_eq!(path.as_ref(), "6/7");
        assert!(Arc::ptr_eq(&resolved, &custom_scheme));
    }
}
