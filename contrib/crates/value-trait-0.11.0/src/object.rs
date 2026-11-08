#[cfg(feature = "halfbrown")]
use halfbrown::HashMap as Halfbrown;
#[cfg(feature = "hashbrown")]
use hashbrown::HashMap as Hashbrown;
use std::collections::HashMap;
use std::hash::Hash;
use std::{borrow::Borrow, hash::BuildHasher};

/// A trait for the minimal common functionality of a vale object
pub trait Object {
    /// The key in the objects
    type Key: ?Sized;
    /// The values in the object
    type Element;

    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    #[must_use]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Iterates over the key value paris
    #[must_use]
    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::Key, &Self::Element)> + '_>;

    /// Iterates over the keys
    #[must_use]
    fn keys(&self) -> Box<dyn Iterator<Item = &Self::Key> + '_>;

    /// Iterates over the values
    #[must_use]
    fn values(&self) -> Box<dyn Iterator<Item = &Self::Element> + '_>;

    /// Number of key/value pairs
    #[must_use]
    fn len(&self) -> usize;

    /// Returns if the array is empty
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A mutable value Object
pub trait ObjectMut {
    /// The key in the objects
    type Key: ?Sized;
    /// The values in the object
    type Element;

    /// Gets the value of a key as a mutable reference.
    #[must_use]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Inserts a value
    #[must_use]
    fn insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Element>
    where
        Self::Key: From<K> + Hash + Eq,
        V: Into<Self::Element>;

    /// Removes a value from the object
    #[must_use]
    fn remove<Q>(&mut self, k: &Q) -> Option<Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

#[cfg(feature = "halfbrown")]
impl<MapK, MapE, S> Object for Halfbrown<MapK, MapE, S>
where
    MapK: Hash + Eq,
    S: BuildHasher + Default,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Halfbrown::get(self, k)
    }

    #[inline]
    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::Key, &Self::Element)> + '_> {
        Box::new(Halfbrown::iter(self))
    }

    #[inline]
    fn keys(&self) -> Box<dyn Iterator<Item = &Self::Key> + '_> {
        Box::new(Halfbrown::keys(self))
    }

    #[inline]
    fn values(&self) -> Box<dyn Iterator<Item = &Self::Element> + '_> {
        Box::new(Halfbrown::values(self))
    }

    #[inline]
    fn len(&self) -> usize {
        Halfbrown::len(self)
    }
}

#[cfg(feature = "halfbrown")]
impl<MapK, MapE, S> ObjectMut for Halfbrown<MapK, MapE, S>
where
    MapK: Hash + Eq,
    S: BuildHasher + Default,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Halfbrown::get_mut(self, k)
    }

    #[inline]
    fn insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Element>
    where
        K: Into<Self::Key>,
        V: Into<Self::Element>,
        Self::Key: Hash + Eq,
    {
        Halfbrown::insert(self, k.into(), v.into())
    }

    #[inline]
    fn remove<Q>(&mut self, k: &Q) -> Option<Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Halfbrown::remove(self, k)
    }
}

impl<MapK, MapE, S: BuildHasher> Object for HashMap<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        HashMap::get(self, k)
    }

    #[inline]
    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::Key, &Self::Element)> + '_> {
        Box::new(HashMap::iter(self))
    }

    #[inline]
    fn keys(&self) -> Box<dyn Iterator<Item = &Self::Key> + '_> {
        Box::new(HashMap::keys(self))
    }

    #[inline]
    fn values(&self) -> Box<dyn Iterator<Item = &Self::Element> + '_> {
        Box::new(HashMap::values(self))
    }

    #[inline]
    fn len(&self) -> usize {
        HashMap::len(self)
    }
}

impl<MapK, MapE, S: BuildHasher> ObjectMut for HashMap<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        HashMap::get_mut(self, k)
    }

    #[inline]
    fn insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Element>
    where
        K: Into<Self::Key>,
        V: Into<Self::Element>,
        Self::Key: Hash + Eq,
    {
        HashMap::insert(self, k.into(), v.into())
    }

    #[inline]
    fn remove<Q>(&mut self, k: &Q) -> Option<Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        HashMap::remove(self, k)
    }
}

#[cfg(feature = "hashbrown")]
impl<MapK, MapE, S: BuildHasher> Object for Hashbrown<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Hashbrown::get(self, k)
    }

    #[inline]
    fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = (&Self::Key, &Self::Element)> + 'i> {
        Box::new(Hashbrown::iter(self))
    }

    #[inline]
    fn keys<'i>(&'i self) -> Box<dyn Iterator<Item = &Self::Key> + 'i> {
        Box::new(Hashbrown::keys(self))
    }

    #[inline]
    fn values<'i>(&'i self) -> Box<dyn Iterator<Item = &Self::Element> + 'i> {
        Box::new(Hashbrown::values(self))
    }

    #[inline]
    fn len(&self) -> usize {
        Hashbrown::len(self)
    }
}
#[cfg(feature = "hashbrown")]
impl<MapK, MapE, S: BuildHasher> ObjectMut for Hashbrown<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Hashbrown::get_mut(self, k)
    }

    #[inline]
    fn insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Element>
    where
        K: Into<Self::Key>,
        V: Into<Self::Element>,
        Self::Key: Hash + Eq,
    {
        Hashbrown::insert(self, k.into(), v.into())
    }

    #[inline]
    fn remove<Q>(&mut self, k: &Q) -> Option<Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Hashbrown::remove(self, k)
    }
}

#[cfg(feature = "c-abi")]
impl<MapK, MapE, S: ::std::hash::BuildHasher> Object
    for abi_stable::std_types::RHashMap<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        abi_stable::std_types::RHashMap::get(self, k)
    }

    #[inline]
    fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = (&Self::Key, &Self::Element)> + 'i> {
        Box::new(abi_stable::std_types::RHashMap::iter(self).map(Into::into))
    }

    #[inline]
    fn keys<'i>(&'i self) -> Box<dyn Iterator<Item = &Self::Key> + 'i> {
        Box::new(abi_stable::std_types::RHashMap::keys(self))
    }

    #[inline]
    fn values<'i>(&'i self) -> Box<dyn Iterator<Item = &Self::Element> + 'i> {
        Box::new(abi_stable::std_types::RHashMap::values(self))
    }

    #[inline]
    fn len(&self) -> usize {
        abi_stable::std_types::RHashMap::len(self)
    }
}

#[cfg(feature = "c-abi")]
impl<MapK, MapE, S: ::std::hash::BuildHasher> ObjectMut
    for abi_stable::std_types::RHashMap<MapK, MapE, S>
where
    MapK: Hash + Eq,
{
    type Key = MapK;
    type Element = MapE;

    #[inline]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        abi_stable::std_types::RHashMap::get_mut(self, k)
    }

    #[inline]
    fn insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Element>
    where
        K: Into<Self::Key>,
        V: Into<Self::Element>,
        Self::Key: Hash + Eq,
    {
        abi_stable::std_types::RHashMap::insert(self, k.into(), v.into()).into()
    }

    #[inline]
    fn remove<Q>(&mut self, k: &Q) -> Option<Self::Element>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        abi_stable::std_types::RHashMap::remove(self, k).into()
    }
}
