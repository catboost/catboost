use crate::BorrowedValue as Value;
use crate::cow::Cow;
use crate::prelude::*;
use halfbrown::RawEntryMut;
use std::fmt;
use std::hash::BuildHasher;

use ahash::{AHasher, RandomState};
use once_cell::sync::OnceCell;
static NOT_RANDOM: OnceCell<RandomState> = OnceCell::new();

/// `AHash` `BuildHasher` that uses a startup initialized random state for known keys
#[derive(Clone)]
pub struct NotSoRandomState(RandomState);

impl Default for NotSoRandomState {
    fn default() -> Self {
        Self(NOT_RANDOM.get_or_init(RandomState::new).clone())
    }
}

impl BuildHasher for NotSoRandomState {
    type Hasher = AHasher;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn build_hasher(&self) -> AHasher {
        self.0.build_hasher()
    }
}

/// Well known key that can be looked up in a `Value` faster.
/// It achieves this by memorizing the hash.
#[derive(Debug, Clone, PartialEq)]
pub struct KnownKey<'key> {
    key: Cow<'key, str>,
    hash: u64,
}

/// Error for known keys
#[derive(Debug, PartialEq, Clone)]
pub enum Error {
    /// The target passed wasn't an object
    NotAnObject(ValueType),
}

#[cfg(not(tarpaulin_include))]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NotAnObject(t) => write!(f, "Expected object but got {t:?}"),
        }
    }
}
impl std::error::Error for Error {}

impl<'key, S> From<S> for KnownKey<'key>
where
    Cow<'key, str>: From<S>,
{
    fn from(key: S) -> Self {
        let key = Cow::from(key);
        let hash_builder = NotSoRandomState::default();
        Self {
            hash: hash_builder.hash_one(&key),
            key,
        }
    }
}

impl<'key> KnownKey<'key> {
    /// The known key
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn key(&self) -> &Cow<'key, str> {
        &self.key
    }

    /// Looks up this key in a `Value`, returns None if the
    /// key wasn't present or `target` isn't an object
    ///
    /// # Errors
    ///  * If target is not an object
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let object = json!({
    ///   "answer": 42,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    /// assert_eq!(known_key.lookup(&object).unwrap(), &42);
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn lookup<'target, 'value>(
        &self,
        target: &'target Value<'value>,
    ) -> Option<&'target Value<'value>>
    where
        'key: 'value,
        'value: 'target,
    {
        target.as_object().and_then(|m| self.map_lookup(m))
    }

    /// Looks up this key in a `Object<Cow<'value>` the inner representation of an object `Value`, returns None if the
    /// key wasn't present.
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let object: BorrowedValue = json!({
    ///   "answer": 42,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    /// if let Some(inner) = object.as_object() {
    ///   assert_eq!(known_key.map_lookup(inner).unwrap(), &42);
    /// }
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn map_lookup<'target, 'value>(
        &self,
        map: &'target super::borrowed::Object<'value>,
    ) -> Option<&'target Value<'value>>
    where
        'key: 'value,
        'value: 'target,
    {
        map.raw_entry()
            .from_key_hashed_nocheck(self.hash, &self.key)
            .map(|kv| kv.1)
    }

    /// Looks up this key in a `Value`, returns None if the
    /// key wasn't present or `target` isn't an object
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// if let Some(answer) = known_key.lookup_mut(&mut object) {
    ///   *answer = BorrowedValue::from(42);
    /// }
    ///
    /// assert_eq!(object["answer"], 42);
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn lookup_mut<'target, 'value>(
        &self,
        target: &'target mut Value<'value>,
    ) -> Option<&'target mut Value<'value>>
    where
        'key: 'value,
        'value: 'target,
    {
        target.as_object_mut().and_then(|m| self.map_lookup_mut(m))
    }

    /// Looks up this key in a `Object<'value>`, the inner representation of an object value.
    /// returns None if the key wasn't present.
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// let known_key = KnownKey::from("answer");
    /// if let Some(inner) = object.as_object_mut() {
    ///   if let Some(answer) = known_key.map_lookup_mut(inner) {
    ///     *answer = BorrowedValue::from(42);
    ///   }
    /// }
    /// assert_eq!(object["answer"], 42);
    ///
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn map_lookup_mut<'target, 'value>(
        &self,
        map: &'target mut super::borrowed::Object<'value>,
    ) -> Option<&'target mut Value<'value>>
    where
        'key: 'value,
        'value: 'target,
    {
        match map
            .raw_entry_mut()
            .from_key_hashed_nocheck(self.hash, &self.key)
        {
            RawEntryMut::Occupied(e) => Some(e.into_mut()),
            RawEntryMut::Vacant(_e) => None,
        }
    }

    /// Looks up this key in a `Value`, inserts `with` when the key
    ///  when wasn't present returns None if the `target` isn't an object
    /// # Errors
    /// * if target is not a record
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// if let Ok(answer) = known_key.lookup_or_insert_mut(&mut object, || 17.into()) {
    ///   assert_eq!(*answer, 23);
    ///   *answer = BorrowedValue::from(42);
    /// }
    ///
    /// assert_eq!(object["answer"], 42);
    ///
    /// let known_key2 = KnownKey::from("also the answer");
    /// if let Ok(answer) = known_key2.lookup_or_insert_mut(&mut object, || 8.into()) {
    ///   assert_eq!(*answer, 8);
    ///   *answer = BorrowedValue::from(42);
    /// }
    ///
    /// assert_eq!(object["also the answer"], 42);
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn lookup_or_insert_mut<'target, 'value, F>(
        &self,
        target: &'target mut Value<'value>,
        with: F,
    ) -> Result<&'target mut Value<'value>, Error>
    where
        'key: 'value,
        'value: 'target,
        F: FnOnce() -> Value<'value>,
    {
        match target {
            Value::Object(inner) => Ok(self.map_lookup_or_insert_mut(inner, with)),
            other => Err(Error::NotAnObject(other.value_type())),
        }
    }

    /// Looks up this key in a `Object<'value>`, the inner representation of an object `Value`.
    /// Inserts `with` when the key when wasn't present.
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// if let Some(inner) = object.as_object_mut() {
    ///   let answer = known_key.map_lookup_or_insert_mut(inner, || 17.into());
    ///   assert_eq!(*answer, 23);
    ///   *answer = BorrowedValue::from(42);
    /// }
    ///
    /// assert_eq!(object["answer"], 42);
    ///
    /// let known_key2 = KnownKey::from("also the answer");
    /// if let Some(inner) = object.as_object_mut() {
    ///   let answer = known_key2.map_lookup_or_insert_mut(inner, || 8.into());
    ///   assert_eq!(*answer, 8);
    ///   *answer = BorrowedValue::from(42);
    /// }
    ///
    /// assert_eq!(object["also the answer"], 42);
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn map_lookup_or_insert_mut<'target, 'value, F>(
        &self,
        map: &'target mut super::borrowed::Object<'value>,
        with: F,
    ) -> &'target mut Value<'value>
    where
        'key: 'value,
        'value: 'target,
        F: FnOnce() -> Value<'value>,
    {
        map.raw_entry_mut()
            .from_key_hashed_nocheck(self.hash, &self.key)
            .or_insert_with(|| (self.key.clone(), with()))
            .1
    }

    /// Inserts a value key into  `Value`, returns None if the
    /// key wasn't present otherwise Some(`old value`).
    /// # Errors
    ///   * if `target` isn't an object
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// assert!(known_key.insert(&mut object, BorrowedValue::from(42)).is_ok());
    ///
    /// assert_eq!(object["answer"], 42);
    ///
    /// let known_key2 = KnownKey::from("also the answer");
    ///
    /// assert!(known_key2.insert(&mut object, BorrowedValue::from(42)).is_ok());
    ///
    /// assert_eq!(object["also the answer"], 42);
    ///
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn insert<'target, 'value>(
        &self,
        target: &'target mut Value<'value>,
        value: Value<'value>,
    ) -> Result<Option<Value<'value>>, Error>
    where
        'key: 'value,
        'value: 'target,
    {
        target
            .as_object_mut()
            .map(|m| self.map_insert(m, value))
            .ok_or_else(|| Error::NotAnObject(target.value_type()))
    }

    /// Inserts a value key into `map`, returns None if the
    /// key wasn't present otherwise Some(`old value`).
    ///
    /// ```rust
    /// use simd_json::prelude::*;
    /// use simd_json::*;
    ///
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// let known_key = KnownKey::from("answer");
    ///
    /// assert_eq!(object["answer"], 23);
    ///
    /// if let Some(inner) = object.as_object_mut() {
    ///   assert!(known_key.map_insert(inner.into(), BorrowedValue::from(42)).is_some());
    /// }
    ///
    /// assert_eq!(object["answer"], 42);
    ///
    /// let known_key2 = KnownKey::from("also the answer");
    ///
    /// if let Some(inner) = object.as_object_mut() {
    ///   assert!(known_key2.map_insert(inner.into(), BorrowedValue::from(42)).is_none());
    /// }
    ///
    /// assert_eq!(object["also the answer"], 42);
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn map_insert<'target, 'value>(
        &self,
        map: &'target mut super::borrowed::Object<'value>,
        value: Value<'value>,
    ) -> Option<Value<'value>>
    where
        'key: 'value,
        'value: 'target,
    {
        match map
            .raw_entry_mut()
            .from_key_hashed_nocheck(self.hash, &self.key)
        {
            RawEntryMut::Occupied(mut e) => Some(e.insert(value)),
            RawEntryMut::Vacant(e) => {
                e.insert_hashed_nocheck(self.hash, self.key.clone(), value);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unnecessary_operation, clippy::non_ascii_literal)]
    use super::*;

    #[test]
    fn known_key() {
        use crate::cow::Cow;
        let mut v = Value::object();
        v.try_insert("key", 1);
        let key1 = KnownKey::from(Cow::from("key"));
        let key2 = KnownKey::from(Cow::from("cake"));

        assert!(key1.lookup(&Value::null()).is_none());
        assert!(key2.lookup(&Value::null()).is_none());
        assert!(key1.lookup(&v).is_some());
        assert!(key2.lookup(&v).is_none());
        assert!(key1.lookup_mut(&mut v).is_some());
        assert!(key2.lookup_mut(&mut v).is_none());
    }

    #[test]
    fn known_key_insert() {
        use crate::cow::Cow;
        let mut v = Value::object();
        let _: Option<_> = v.try_insert("key", 1);
        let key1 = KnownKey::from(Cow::from("key"));
        let key2 = KnownKey::from(Cow::from("cake"));

        let mut v1 = Value::null();
        assert!(key1.insert(&mut v1, 2.into()).is_err());
        assert!(key2.insert(&mut v1, 2.into()).is_err());
        assert_eq!(
            key1.insert(&mut v, 2.into()).expect("failed to insert"),
            Some(1.into())
        );
        assert_eq!(
            key2.insert(&mut v, 3.into()).expect("failed to insert"),
            None
        );
        assert_eq!(v["key"], 2);
        assert_eq!(v["cake"], 3);
    }

    #[test]
    fn lookup_or_insert_mut() {
        use crate::cow::Cow;
        let mut v = Value::object();
        let _: Option<_> = v.try_insert("key", 1);
        let key1 = KnownKey::from(Cow::from("key"));
        let key2 = KnownKey::from(Cow::from("cake"));

        let mut v1 = Value::null();
        assert!(key1.lookup_or_insert_mut(&mut v1, || 2.into()).is_err());
        assert!(key2.lookup_or_insert_mut(&mut v1, || 2.into()).is_err());

        {
            let r1 = key1
                .lookup_or_insert_mut(&mut v, || 2.into())
                .expect("failed to insert");
            assert_eq!(r1.as_u8(), Some(1));
        }
        {
            let r2 = key2
                .lookup_or_insert_mut(&mut v, || 3.into())
                .expect("failed to insert");
            assert_eq!(r2.as_u8(), Some(3));
        }
    }
    #[test]
    fn known_key_map() {
        use crate::cow::Cow;
        let mut v = Value::object_with_capacity(128);
        v.insert("key", 1).expect("failed to insert");
        let key1 = KnownKey::from(Cow::from("key"));
        let key2 = KnownKey::from(Cow::from("cake"));

        assert!(key1.lookup(&Value::null()).is_none());
        assert!(key2.lookup(&Value::null()).is_none());
        assert!(key1.lookup(&v).is_some());
        assert!(key2.lookup(&v).is_none());
    }

    #[test]
    fn known_key_insert_map() {
        use crate::cow::Cow;
        let mut v = Value::object_with_capacity(128);
        v.insert("key", 1).expect("failed to insert");
        let key1 = KnownKey::from(Cow::from("key"));
        let key2 = KnownKey::from(Cow::from("cake"));

        let mut v1 = Value::null();

        assert!(key1.insert(&mut v1, 2.into()).is_err());
        assert!(key2.insert(&mut v1, 2.into()).is_err());
        assert_eq!(
            key1.insert(&mut v, 2.into()).expect("failed to insert"),
            Some(1.into())
        );
        assert_eq!(
            key2.insert(&mut v, 3.into()).expect("failed to insert"),
            None
        );
        assert_eq!(v["key"], 2);
        assert_eq!(v["cake"], 3);
    }

    #[test]
    fn known_key_get_key() {
        let key1 = KnownKey::from("snot");

        assert_eq!(key1.key(), "snot");
    }
}
