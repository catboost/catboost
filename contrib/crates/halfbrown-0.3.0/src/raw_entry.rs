// based on / take from <https://github.com/rust-lang/hashbrown/blob/62a1ae24d4678fcbf777bef6b205fadeecb781d9/src/map.rs>

use super::{fmt, hashbrown, Borrow, BuildHasher, Debug, Hash};
use crate::vecmap;
use hashbrown::hash_map;
/*
use std::fmt::{self, Debug};
use std::mem;
*/

/// A builder for computing where in a [`HashMap`] a key-value pair would be stored.
///
/// See the [`HashMap::raw_entry_mut`] docs for usage examples.
///
/// [`HashMap::raw_entry_mut`]: struct.HashMap.html#method.raw_entry_mut
pub struct RawEntryBuilderMut<'map, K, V, const N: usize, S>(
    RawEntryBuilderMutInt<'map, K, V, N, S>,
);

impl<'map, K, V, const N: usize, S> From<hash_map::RawEntryBuilderMut<'map, K, V, S>>
    for RawEntryBuilderMut<'map, K, V, N, S>
{
    fn from(m: hash_map::RawEntryBuilderMut<'map, K, V, S>) -> Self {
        Self(RawEntryBuilderMutInt::Map(m))
    }
}

impl<'map, K, V, const N: usize, S> From<vecmap::RawEntryBuilderMut<'map, K, V, N, S>>
    for RawEntryBuilderMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    fn from(m: vecmap::RawEntryBuilderMut<'map, K, V, N, S>) -> Self {
        Self(RawEntryBuilderMutInt::Vec(m))
    }
}
enum RawEntryBuilderMutInt<'map, K, V, const N: usize, S> {
    Vec(vecmap::RawEntryBuilderMut<'map, K, V, N, S>),
    Map(hash_map::RawEntryBuilderMut<'map, K, V, S>),
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This is a lower-level version of [`Entry`].
///
/// This `enum` is constructed through the [`raw_entry_mut`] method on [`HashMap`],
/// then calling one of the methods of that [`RawEntryBuilderMut`].
///
/// [`HashMap`]: struct.HashMap.html
/// [`Entry`]: enum.Entry.html
/// [`raw_entry_mut`]: struct.HashMap.html#method.raw_entry_mut
/// [`RawEntryBuilderMut`]: struct.RawEntryBuilderMut.html
pub enum RawEntryMut<'map, K, V, const N: usize, S> {
    /// An occupied entry.
    Occupied(RawOccupiedEntryMut<'map, K, V, N, S>),
    /// A vacant entry.
    Vacant(RawVacantEntryMut<'map, K, V, N, S>),
}

impl<'map, K, V, const N: usize, S> From<vecmap::RawEntryMut<'map, K, V, N, S>>
    for RawEntryMut<'map, K, V, N, S>
{
    fn from(e: vecmap::RawEntryMut<'map, K, V, N, S>) -> Self {
        match e {
            vecmap::RawEntryMut::Occupied(o) => Self::Occupied(o.into()),
            vecmap::RawEntryMut::Vacant(v) => Self::Vacant(v.into()),
        }
    }
}

impl<'map, K, V, const N: usize, S> From<hash_map::RawEntryMut<'map, K, V, S>>
    for RawEntryMut<'map, K, V, N, S>
{
    fn from(e: hash_map::RawEntryMut<'map, K, V, S>) -> Self {
        match e {
            hash_map::RawEntryMut::Occupied(o) => Self::Occupied(o.into()),
            hash_map::RawEntryMut::Vacant(v) => Self::Vacant(v.into()),
        }
    }
}

/// A view into an occupied entry in a `HashMap`.
/// It is part of the [`RawEntryMut`] enum.
///
/// [`RawEntryMut`]: enum.RawEntryMut.html
pub struct RawOccupiedEntryMut<'map, K, V, const N: usize, S>(
    RawOccupiedEntryMutInt<'map, K, V, N, S>,
);

impl<'map, K, V, const N: usize, S> From<vecmap::RawOccupiedEntryMut<'map, K, V, N, S>>
    for RawOccupiedEntryMut<'map, K, V, N, S>
{
    fn from(m: vecmap::RawOccupiedEntryMut<'map, K, V, N, S>) -> Self {
        Self(RawOccupiedEntryMutInt::Vec(m))
    }
}

impl<'map, K, V, const N: usize, S> From<hash_map::RawOccupiedEntryMut<'map, K, V, S>>
    for RawOccupiedEntryMut<'map, K, V, N, S>
{
    fn from(m: hash_map::RawOccupiedEntryMut<'map, K, V, S>) -> Self {
        Self(RawOccupiedEntryMutInt::Map(m))
    }
}

unsafe impl<K, V, const N: usize, S> Send for RawOccupiedEntryMut<'_, K, V, N, S>
where
    K: Send,
    V: Send,
    S: Send,
{
}

unsafe impl<K, V, const N: usize, S> Sync for RawOccupiedEntryMut<'_, K, V, N, S>
where
    K: Sync,
    V: Sync,
    S: Sync,
{
}

enum RawOccupiedEntryMutInt<'map, K, V, const N: usize, S> {
    Vec(vecmap::RawOccupiedEntryMut<'map, K, V, N, S>),
    Map(hash_map::RawOccupiedEntryMut<'map, K, V, S>),
}

unsafe impl<K, V, const N: usize, S> Send for RawOccupiedEntryMutInt<'_, K, V, N, S>
where
    K: Send,
    V: Send,
    S: Send,
{
}
unsafe impl<K, V, const N: usize, S> Sync for RawOccupiedEntryMutInt<'_, K, V, N, S>
where
    K: Sync,
    V: Sync,
    S: Sync,
{
}

/// A view into a vacant entry in a `HashMap`.
/// It is part of the [`RawEntryMut`] enum.
///
/// [`RawEntryMut`]: enum.RawEntryMut.html
pub struct RawVacantEntryMut<'map, K, V, const N: usize, S>(RawVacantEntryMutInt<'map, K, V, N, S>);

impl<'map, K, V, const N: usize, S> From<vecmap::RawVacantEntryMut<'map, K, V, N, S>>
    for RawVacantEntryMut<'map, K, V, N, S>
{
    fn from(m: vecmap::RawVacantEntryMut<'map, K, V, N, S>) -> Self {
        Self(RawVacantEntryMutInt::Vec(m))
    }
}

impl<'map, K, V, const N: usize, S> From<hash_map::RawVacantEntryMut<'map, K, V, S>>
    for RawVacantEntryMut<'map, K, V, N, S>
{
    fn from(m: hash_map::RawVacantEntryMut<'map, K, V, S>) -> Self {
        Self(RawVacantEntryMutInt::Map(m))
    }
}

enum RawVacantEntryMutInt<'map, K, V, const N: usize, S> {
    Vec(vecmap::RawVacantEntryMut<'map, K, V, N, S>),
    Map(hash_map::RawVacantEntryMut<'map, K, V, S>),
}

/// A builder for computing where in a [`HashMap`] a key-value pair would be stored.
///
/// See the [`HashMap::raw_entry`] docs for usage examples.
///
/// [`HashMap::raw_entry`]: struct.HashMap.html#method.raw_entry
///
pub struct RawEntryBuilder<'map, K, V, const N: usize, S>(RawEntryBuilderInt<'map, K, V, N, S>);

impl<'map, K, V, const N: usize, S> From<hash_map::RawEntryBuilder<'map, K, V, S>>
    for RawEntryBuilder<'map, K, V, N, S>
{
    fn from(m: hash_map::RawEntryBuilder<'map, K, V, S>) -> Self {
        Self(RawEntryBuilderInt::Map(m))
    }
}

impl<'map, K, V, const N: usize, S> From<vecmap::RawEntryBuilder<'map, K, V, N, S>>
    for RawEntryBuilder<'map, K, V, N, S>
{
    fn from(m: vecmap::RawEntryBuilder<'map, K, V, N, S>) -> Self {
        Self(RawEntryBuilderInt::Vec(m))
    }
}

enum RawEntryBuilderInt<'map, K, V, const N: usize, S> {
    Vec(vecmap::RawEntryBuilder<'map, K, V, N, S>),
    Map(hash_map::RawEntryBuilder<'map, K, V, S>),
}

impl<'map, K, V, const N: usize, S> RawEntryBuilderMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Creates a `RawEntryMut` from the given key.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key<Q>(self, k: &Q) -> RawEntryMut<'map, K, V, N, S>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.0 {
            RawEntryBuilderMutInt::Vec(m) => m.from_key(k).into(),
            RawEntryBuilderMutInt::Map(m) => m.from_key(k).into(),
        }
    }

    /// Creates a `RawEntryMut` from the given key and its hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck<Q>(self, hash: u64, k: &Q) -> RawEntryMut<'map, K, V, N, S>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        match self.0 {
            RawEntryBuilderMutInt::Vec(m) => m.from_key_hashed_nocheck(hash, k).into(),
            RawEntryBuilderMutInt::Map(m) => m.from_key_hashed_nocheck(hash, k).into(),
        }
    }
}

impl<'map, K, V, const N: usize, S> RawEntryBuilderMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Creates a `RawEntryMut` from the given hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> RawEntryMut<'map, K, V, N, S>
    where
        for<'b> F: FnMut(&'b K) -> bool,
    {
        match self.0 {
            RawEntryBuilderMutInt::Vec(m) => m.from_hash(hash, is_match).into(),
            RawEntryBuilderMutInt::Map(m) => m.from_hash(hash, is_match).into(),
        }
    }
}

impl<'map, K, V, const N: usize, S> RawEntryBuilder<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Access an entry by key.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key<Q>(self, k: &Q) -> Option<(&'map K, &'map V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.0 {
            RawEntryBuilderInt::Vec(m) => m.from_key(k),
            RawEntryBuilderInt::Map(m) => m.from_key(k),
        }
    }

    /// Access an entry by a key and its hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck<Q>(self, hash: u64, k: &Q) -> Option<(&'map K, &'map V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.0 {
            RawEntryBuilderInt::Vec(m) => m.from_key_hashed_nocheck(hash, k),
            RawEntryBuilderInt::Map(m) => m.from_key_hashed_nocheck(hash, k),
        }
    }

    /// Access an entry by hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> Option<(&'map K, &'map V)>
    where
        F: FnMut(&K) -> bool,
    {
        match self.0 {
            RawEntryBuilderInt::Vec(m) => m.from_hash(hash, is_match),
            RawEntryBuilderInt::Map(m) => m.from_hash(hash, is_match),
        }
    }
}

impl<'map, K, V, const N: usize, S> RawEntryMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Sets the value of the entry, and returns a `RawOccupiedEntryMut`.
    ///
    /// # Examples
    ///
    /// ```
    /// use halfbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// let entry = map.raw_entry_mut().from_key("horseyland").insert("horseyland", 37);
    ///
    /// assert_eq!(entry.remove_entry(), ("horseyland", 37));
    /// ```
    #[inline]
    pub fn insert(self, key: K, value: V) -> RawOccupiedEntryMut<'map, K, V, N, S>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            RawEntryMut::Vacant(entry) => {
                //entry.insert_entry(key, value)
                match entry.0 {
                    RawVacantEntryMutInt::Vec(e) => {
                        vecmap::RawEntryMut::Vacant(e).insert(key, value).into()
                    }
                    RawVacantEntryMutInt::Map(e) => {
                        hash_map::RawEntryMut::Vacant(e).insert(key, value).into()
                    }
                }
            }
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// mutable references to the key and value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use halfbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.raw_entry_mut().from_key("poneyland").or_insert("poneyland", 3);
    /// assert_eq!(map["poneyland"], 3);
    ///
    /// *map.raw_entry_mut().from_key("poneyland").or_insert("poneyland", 10).1 *= 2;
    /// assert_eq!(map["poneyland"], 6);
    /// ```
    #[inline]
    pub fn or_insert(self, default_key: K, default_val: V) -> (&'map mut K, &'map mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => entry.insert(default_key, default_val),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns mutable references to the key and value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use halfbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, String> = HashMap::new();
    ///
    /// map.raw_entry_mut().from_key("poneyland").or_insert_with(|| {
    ///     ("poneyland", "hoho".to_string())
    /// });
    ///
    /// assert_eq!(map["poneyland"], "hoho".to_string());
    /// ```
    #[inline]
    pub fn or_insert_with<F>(self, default: F) -> (&'map mut K, &'map mut V)
    where
        F: FnOnce() -> (K, V),
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => {
                let (k, v) = default();
                entry.insert(k, v)
            }
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use halfbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.raw_entry_mut()
    ///    .from_key("poneyland")
    ///    .and_modify(|_k, v| { *v += 1 })
    ///    .or_insert("poneyland", 42);
    /// assert_eq!(map["poneyland"], 42);
    ///
    /// map.raw_entry_mut()
    ///    .from_key("poneyland")
    ///    .and_modify(|_k, v| { *v += 1 })
    ///    .or_insert("poneyland", 0);
    /// assert_eq!(map["poneyland"], 43);
    /// ```
    #[inline]
    #[must_use]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut K, &mut V),
    {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                {
                    let (k, v) = entry.get_key_value_mut();
                    f(k, v);
                }
                RawEntryMut::Occupied(entry)
            }
            RawEntryMut::Vacant(entry) => RawEntryMut::Vacant(entry),
        }
    }
}

impl<'map, K, V, const N: usize, S> RawOccupiedEntryMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Gets a reference to the key in the entry.
    #[inline]
    #[must_use]
    pub fn key(&self) -> &K {
        match &self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.key(),
            RawOccupiedEntryMutInt::Map(e) => e.key(),
        }
    }

    /// Gets a mutable reference to the key in the entry.
    #[inline]
    pub fn key_mut(&mut self) -> &mut K {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.key_mut(),
            RawOccupiedEntryMutInt::Map(e) => e.key_mut(),
        }
    }

    /// Converts the entry into a mutable reference to the key in the entry
    /// with a lifetime bound to the map itself.
    #[inline]
    #[must_use]
    pub fn into_key(self) -> &'map mut K {
        match self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.into_key(),
            RawOccupiedEntryMutInt::Map(e) => e.into_key(),
        }
    }

    /// Gets a reference to the value in the entry.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &V {
        match &self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.get(),
            RawOccupiedEntryMutInt::Map(e) => e.get(),
        }
    }

    /// Converts the `OccupiedEntry` into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    #[inline]
    #[must_use]
    pub fn into_mut(self) -> &'map mut V {
        match self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.into_mut(),
            RawOccupiedEntryMutInt::Map(e) => e.into_mut(),
        }
    }

    /// Gets a mutable reference to the value in the entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.get_mut(),
            RawOccupiedEntryMutInt::Map(e) => e.get_mut(),
        }
    }

    /// Gets a reference to the key and value in the entry.
    #[inline]
    pub fn get_key_value(&mut self) -> (&K, &V) {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.get_key_value(),
            RawOccupiedEntryMutInt::Map(e) => e.get_key_value(),
        }
    }

    /// Gets a mutable reference to the key and value in the entry.
    #[inline]
    pub fn get_key_value_mut(&mut self) -> (&mut K, &mut V) {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.get_key_value_mut(),
            RawOccupiedEntryMutInt::Map(e) => e.get_key_value_mut(),
        }
    }

    /// Converts the `OccupiedEntry` into a mutable reference to the key and value in the entry
    /// with a lifetime bound to the map itself.
    #[inline]
    #[must_use]
    pub fn into_key_value(self) -> (&'map mut K, &'map mut V) {
        match self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.into_key_value(),
            RawOccupiedEntryMutInt::Map(e) => e.into_key_value(),
        }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.insert(value),
            RawOccupiedEntryMutInt::Map(e) => e.insert(value),
        }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[inline]
    pub fn insert_key(&mut self, key: K) -> K {
        match &mut self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.insert_key(key),
            RawOccupiedEntryMutInt::Map(e) => e.insert_key(key),
        }
    }

    /// Takes the value out of the entry, and returns it.
    #[inline]
    #[must_use]
    pub fn remove(self) -> V {
        match self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.remove(),
            RawOccupiedEntryMutInt::Map(e) => e.remove(),
        }
    }

    /// Take the ownership of the key and value from the map.
    #[inline]
    #[must_use]
    pub fn remove_entry(self) -> (K, V) {
        match self.0 {
            RawOccupiedEntryMutInt::Vec(e) => e.remove_entry(),
            RawOccupiedEntryMutInt::Map(e) => e.remove_entry(),
        }
    }
}

impl<'map, K, V, const N: usize, S> RawVacantEntryMut<'map, K, V, N, S>
where
    S: BuildHasher,
{
    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    #[inline]
    pub fn insert(self, key: K, value: V) -> (&'map mut K, &'map mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        match self.0 {
            RawVacantEntryMutInt::Vec(e) => e.insert(key, value),
            RawVacantEntryMutInt::Map(e) => e.insert(key, value),
        }
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    #[inline]
    #[allow(clippy::shadow_unrelated)]
    pub fn insert_hashed_nocheck(self, hash: u64, key: K, value: V) -> (&'map mut K, &'map mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        match self.0 {
            RawVacantEntryMutInt::Vec(e) => e.insert_hashed_nocheck(hash, key, value),
            RawVacantEntryMutInt::Map(e) => e.insert_hashed_nocheck(hash, key, value),
        }
    }

    /// Set the value of an entry with a custom hasher function.
    #[inline]
    pub fn insert_with_hasher<H>(
        self,
        hash: u64,
        key: K,
        value: V,
        hasher: H,
    ) -> (&'map mut K, &'map mut V)
    where
        S: BuildHasher,
        H: Fn(&K) -> u64,
    {
        match self.0 {
            RawVacantEntryMutInt::Vec(e) => e.insert_with_hasher(hash, key, value, hasher),
            RawVacantEntryMutInt::Map(e) => e.insert_with_hasher(hash, key, value, hasher),
        }
    }
}

impl<K, V, const N: usize, S> Debug for RawEntryBuilderMut<'_, K, V, N, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawEntryBuilder").finish()
    }
}

impl<K: Debug, V: Debug, const N: usize, S> Debug for RawEntryMut<'_, K, V, N, S>
where
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            RawEntryMut::Vacant(ref v) => f.debug_tuple("RawEntry").field(v).finish(),
            RawEntryMut::Occupied(ref o) => f.debug_tuple("RawEntry").field(o).finish(),
        }
    }
}

impl<K: Debug, V: Debug, const N: usize, S> Debug for RawOccupiedEntryMut<'_, K, V, N, S>
where
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawOccupiedEntryMut")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<K, V, const N: usize, S> Debug for RawVacantEntryMut<'_, K, V, N, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawVacantEntryMut").finish()
    }
}

impl<K, V, const N: usize, S> Debug for RawEntryBuilder<'_, K, V, N, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawEntryBuilder").finish()
    }
}
