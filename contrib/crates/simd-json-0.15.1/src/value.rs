/// This module holds the two dom implementations we use.
///
/// We distinguish between owned and borrowed. The difference being is that
/// the borrowed value will use `&str` as its string type, referencing the input,
/// while owned will allocate a new String for each value.
///
/// Note that since JSON strings allow for escape sequences the borrowed
/// value does not implement zero copy parsing, it does however not allocate
/// new memory for strings.
///
/// This differs notably from Serde's zero copy implementation as, unlike Serde,
/// we do not require prior knowledge about string content to take advantage
/// of it.
///
/// ## Usage
/// The value trait is meant to simplify interacting with DOM values, for both
/// creation as well as mutation and inspection.
///
/// Objects can be treated as hashmap's for the most part
/// ```rust
/// use simd_json::{OwnedValue as Value, prelude::*};
/// let mut v = Value::object();
/// v.insert("key", 42);
/// assert_eq!(v.get("key").unwrap(), &42);
/// assert_eq!(v["key"], &42);
/// assert_eq!(v.remove("key").unwrap().unwrap(), 42);
/// assert_eq!(v.get("key"), None);
/// ```
///
/// Arrays can be treated as vectors for the most part
///
/// ```rust
/// use simd_json::{OwnedValue as Value, prelude::*};
/// let mut v = Value::array();
/// v.push("zero");
/// v.push(1);
/// assert_eq!(v[0], &"zero");
/// assert_eq!(v.get_idx(1).unwrap(), &1);
/// assert_eq!(v.pop().unwrap().unwrap(), 1);
/// assert_eq!(v.pop().unwrap().unwrap(), "zero");
/// assert_eq!(v.pop().unwrap(), None);
/// ```
///
/// Nested changes are also possible:
/// ```rust
/// use simd_json::{OwnedValue as Value, prelude::*};
/// let mut o = Value::object();
/// o.insert("key", Value::array());
/// o["key"].push(Value::object());
/// o["key"][0].insert("other", "value");
/// assert_eq!(o.encode(), r#"{"key":[{"other":"value"}]}"#);
/// ```
/// Borrowed values, using Cow's for strings using in situ parsing strategies wherever possible
pub mod borrowed;
/// Owned, lifetimeless version of the value for times when lifetimes are to be avoided
pub mod owned;
/// Tape implementation
pub mod tape;

pub mod lazy;

pub use self::borrowed::{
    Value as BorrowedValue, to_value as to_borrowed_value,
    to_value_with_buffers as to_borrowed_value_with_buffers,
};
pub use self::owned::{
    Value as OwnedValue, to_value as to_owned_value,
    to_value_with_buffers as to_owned_value_with_buffers,
};
use crate::{Buffers, Deserializer, Result};
use halfbrown::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use tape::Node;
pub use value_trait::*;

/// Hasher used for objects
#[cfg(feature = "known-key")]
pub type ObjectHasher = crate::known_key::NotSoRandomState;
/// Hasher used for objects
#[cfg(not(feature = "known-key"))]
pub type ObjectHasher = halfbrown::DefaultHashBuilder;

/// Parses a slice of bytes into a Value dom.
///
/// This function will rewrite the slice to de-escape strings.
/// As we reference parts of the input slice the resulting dom
/// has the same lifetime as the slice it was created from.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn deserialize<'de, Value, Key>(s: &'de mut [u8]) -> Result<Value>
where
    Value: ValueBuilder<'de> + From<Vec<Value>> + From<HashMap<Key, Value, ObjectHasher>> + 'de,
    Key: Hash + Eq + From<&'de str>,
{
    match Deserializer::from_slice(s) {
        Ok(de) => Ok(ValueDeserializer::from_deserializer(de).parse()),
        Err(e) => Err(e),
    }
}

/// Parses a slice of bytes into a Value dom.
///
/// This function will rewrite the slice to de-escape strings.
/// As we reference parts of the input slice the resulting dom
/// has the same lifetime as the slice it was created from.
///
/// Passes in reusable buffers to reduce allocations.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn deserialize_with_buffers<'de, Value, Key>(
    s: &'de mut [u8],
    buffers: &mut Buffers,
) -> Result<Value>
where
    Value: ValueBuilder<'de> + From<Vec<Value>> + From<HashMap<Key, Value, ObjectHasher>> + 'de,
    Key: Hash + Eq + From<&'de str>,
{
    match Deserializer::from_slice_with_buffers(s, buffers) {
        Ok(de) => Ok(ValueDeserializer::from_deserializer(de).parse()),
        Err(e) => Err(e),
    }
}

struct ValueDeserializer<'de, Value, Key>
where
    Value: ValueBuilder<'de> + From<Vec<Value>> + From<HashMap<Key, Value, ObjectHasher>> + 'de,
    Key: Hash + Eq + From<&'de str>,
{
    de: Deserializer<'de>,
    _marker: PhantomData<(Value, Key)>,
}

impl<'de, Value, Key> ValueDeserializer<'de, Value, Key>
where
    Value: ValueBuilder<'de>
        + From<&'de str>
        + From<Vec<Value>>
        + From<HashMap<Key, Value, ObjectHasher>>
        + 'de,
    Key: Hash + Eq + From<&'de str>,
{
    pub fn from_deserializer(de: Deserializer<'de>) -> Self {
        Self {
            de,
            _marker: PhantomData,
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn parse(&mut self) -> Value {
        match unsafe { self.de.next_() } {
            Node::Static(s) => Value::from(s),
            Node::String(s) => Value::from(s),
            Node::Array { len, count: _ } => self.parse_array(len),
            Node::Object { len, count: _ } => self.parse_map(len),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_array(&mut self, len: usize) -> Value {
        // Rust doesn't optimize the normal loop away here
        // so we write our own avoiding the length
        // checks during push
        let mut res: Vec<Value> = Vec::with_capacity(len);
        let res_ptr = res.as_mut_ptr();
        unsafe {
            for i in 0..len {
                res_ptr.add(i).write(self.parse());
            }
            res.set_len(len);
        }
        Value::from(res)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_map(&mut self, len: usize) -> Value {
        let mut res: HashMap<Key, Value, ObjectHasher> =
            HashMap::with_capacity_and_hasher(len, ObjectHasher::default());

        // Since we checked if it's empty we know that we at least have one
        // element so we eat this
        for _ in 0..len {
            if let Node::String(key) = unsafe { self.de.next_() } {
                #[cfg(not(feature = "value-no-dup-keys"))]
                unsafe {
                    res.insert_nocheck(key.into(), self.parse());
                };
                #[cfg(feature = "value-no-dup-keys")]
                res.insert(key.into(), self.parse());
            } else {
                unreachable!("parse_map: key needs to be a string");
            }
        }
        Value::from(res)
    }
}
