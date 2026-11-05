/// A dom object that references the raw input data to avoid allocations
/// it trades having lifetimes for a gain in performance.
///
/// Access via array indexes is possible:
/// ```rust
/// use simd_json::{BorrowedValue, json};
/// use simd_json::prelude::*;
/// let mut a = json!([1, 2, 3]);
/// assert_eq!(a[1], 2);
/// a[1] = 42.into();
/// assert_eq!(a[1], 42);
/// ```
///
/// Access via object keys is possible as well:
/// ```rust
/// use simd_json::{BorrowedValue, json};
/// use simd_json::prelude::*;
/// let mut a = json!({"key": "not the value"});
/// assert_eq!(a["key"], "not the value");
/// a["key"] = "value".into();
/// assert_eq!(a["key"], "value");
/// ```
mod cmp;
mod from;
mod serialize;

use super::ObjectHasher;
use crate::{Buffers, prelude::*};
use crate::{Deserializer, Node, Result};
use crate::{cow::Cow, safer_unchecked::GetSaferUnchecked as _};
use halfbrown::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Representation of a JSON object
pub type Object<'value> = HashMap<Cow<'value, str>, Value<'value>, ObjectHasher>;
/// Representation of a JSON array
pub type Array<'value> = Vec<Value<'value>>;

/// Parses a slice of bytes into a Value dom.
///
/// This function will rewrite the slice to de-escape strings.
/// As we reference parts of the input slice the resulting dom
/// has the same lifetime as the slice it was created from.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn to_value(s: &mut [u8]) -> Result<Value> {
    match Deserializer::from_slice(s) {
        Ok(de) => Ok(BorrowDeserializer::from_deserializer(de).parse()),
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
///  # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn to_value_with_buffers<'value>(
    s: &'value mut [u8],
    buffers: &mut Buffers,
) -> Result<Value<'value>> {
    match Deserializer::from_slice_with_buffers(s, buffers) {
        Ok(de) => Ok(BorrowDeserializer::from_deserializer(de).parse()),
        Err(e) => Err(e),
    }
}

/// Borrowed JSON-DOM Value, consider using the `ValueTrait`
/// to access its content
#[derive(Debug, Clone)]
#[cfg_attr(feature = "ordered-float", derive(Eq))]
pub enum Value<'value> {
    /// Static values
    Static(StaticNode),
    /// string type
    String(Cow<'value, str>),
    /// array type
    Array(Box<Vec<Value<'value>>>),
    /// object type
    Object(Box<Object<'value>>),
}

impl<'value> Value<'value> {
    fn as_static(&self) -> Option<StaticNode> {
        match self {
            Self::Static(s) => Some(*s),
            _ => None,
        }
    }

    /// Enforces static lifetime on a borrowed value, this will
    /// force all strings to become owned COW's, the same applies for
    /// Object keys.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn into_static(self) -> Value<'static> {
        match self {
            // Ensure strings are static by turing the cow into a 'static
            // This cow has static lifetime as it's owned, this information however is lost
            // by the borrow checker so we need to transmute it to static.
            // This invariant is guaranteed by the implementation of the cow, cloning an owned
            // value will produce a owned value again see:
            // https://docs.rs/beef/0.4.4/src/beef/generic.rs.html#379-391
            Self::String(s) => unsafe {
                std::mem::transmute::<Value<'value>, Value<'static>>(Self::String(Cow::from(
                    s.into_owned(),
                )))
            },
            // For an array we turn every value into a static
            Self::Array(arr) => arr.into_iter().map(Value::into_static).collect(),
            // For an object, we turn all keys into owned Cows and all values into 'static Values
            Self::Object(obj) => obj
                .into_iter()
                .map(|(k, v)| (Cow::from(k.into_owned()), v.into_static()))
                .collect(),

            // Static nodes are always static
            Value::Static(s) => Value::Static(s),
        }
    }

    /// Clones the current value and enforces a static lifetime, it works the same
    /// as `into_static` but includes cloning logic
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn clone_static(&self) -> Value<'static> {
        match self {
            // Ensure strings are static by turing the cow into a 'static
            // This cow has static lifetime as it's owned, this information however is lost
            // by the borrow checker so we need to transmute it to static.
            // This invariant is guaranteed by the implementation of the cow, cloning an owned
            // value will produce a owned value again see:
            // https://docs.rs/beef/0.4.4/src/beef/generic.rs.html#379-391
            Self::String(s) => unsafe {
                std::mem::transmute::<Value<'value>, Value<'static>>(Self::String(Cow::from(
                    s.to_string(),
                )))
            },
            // For an array we turn every value into a static
            Self::Array(arr) => arr.iter().cloned().map(Value::into_static).collect(),
            // For an object, we turn all keys into owned Cows and all values into 'static Values
            Self::Object(obj) => obj
                .iter()
                .map(|(k, v)| (Cow::from(k.to_string()), v.clone_static()))
                .collect(),

            // Static nodes are always static
            Value::Static(s) => Value::Static(*s),
        }
    }
}

impl<'value> ValueBuilder<'value> for Value<'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn null() -> Self {
        Self::Static(StaticNode::Null)
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn array_with_capacity(capacity: usize) -> Self {
        Self::Array(Box::new(Vec::with_capacity(capacity)))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn object_with_capacity(capacity: usize) -> Self {
        Self::Object(Box::new(Object::with_capacity_and_hasher(
            capacity,
            ObjectHasher::default(),
        )))
    }
}

impl<'value> ValueAsMutArray for Value<'value> {
    type Array = Array<'value>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_array_mut(&mut self) -> Option<&mut Vec<Value<'value>>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}
impl<'value> ValueAsMutObject for Value<'value> {
    type Object = Object<'value>;
    /// Get mutable access to a map.
    ///
    /// ```rust
    /// use simd_json::*;
    /// use value_trait::prelude::*;
    ///
    /// let mut object: BorrowedValue = json!({
    ///   "answer": 23,
    ///   "key": 7
    /// }).into();
    /// assert_eq!(object["answer"], 23);
    ///
    /// if let Some(inner) = object.as_object_mut() {
    ///   inner.insert("value".into(), BorrowedValue::from(json!({"nested": 42})));
    /// }
    /// assert_eq!(object["value"], json!({"nested": 42}));
    ///
    /// ```
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_object_mut(&mut self) -> Option<&mut Object<'value>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }
}

impl TypedValue for Value<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn value_type(&self) -> ValueType {
        match self {
            Self::Static(s) => s.value_type(),
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
            Self::Object(_) => ValueType::Object,
        }
    }
}
impl ValueAsScalar for Value<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_null(&self) -> Option<()> {
        self.as_static()?.as_null()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_bool(&self) -> Option<bool> {
        self.as_static()?.as_bool()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_i64(&self) -> Option<i64> {
        self.as_static()?.as_i64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_i128(&self) -> Option<i128> {
        self.as_static()?.as_i128()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_u64(&self) -> Option<u64> {
        self.as_static()?.as_u64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_u128(&self) -> Option<u128> {
        self.as_static()?.as_u128()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_f64(&self) -> Option<f64> {
        self.as_static()?.as_f64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn cast_f64(&self) -> Option<f64> {
        self.as_static()?.cast_f64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_str(&self) -> Option<&str> {
        use std::borrow::Borrow;
        match self {
            Self::String(s) => Some(s.borrow()),
            _ => None,
        }
    }
}
impl<'value> ValueAsArray for Value<'value> {
    type Array = Array<'value>;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_array(&self) -> Option<&Vec<Value<'value>>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl<'value> ValueAsObject for Value<'value> {
    type Object = Object<'value>;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_object(&self) -> Option<&Object<'value>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }
}

impl<'value> ValueIntoString for Value<'value> {
    type String = Cow<'value, str>;

    fn into_string(self) -> Option<<Self as ValueIntoString>::String> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

impl<'value> ValueIntoArray for Value<'value> {
    type Array = Array<'value>;

    fn into_array(self) -> Option<<Self as ValueIntoArray>::Array> {
        match self {
            Self::Array(a) => Some(*a),
            _ => None,
        }
    }
}

impl<'value> ValueIntoObject for Value<'value> {
    type Object = Object<'value>;

    fn into_object(self) -> Option<<Self as ValueIntoObject>::Object> {
        match self {
            Self::Object(a) => Some(*a),
            _ => None,
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(s) => write!(f, "{s}"),
            Self::String(s) => write!(f, "{s}"),
            Self::Array(a) => write!(f, "{a:?}"),
            Self::Object(o) => write!(f, "{o:?}"),
        }
    }
}

impl<'value> Index<&str> for Value<'value> {
    type Output = Value<'value>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index(&self, index: &str) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'value> Index<usize> for Value<'value> {
    type Output = Value<'value>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get_idx(index).expect("index out of bounds")
    }
}

impl IndexMut<&str> for Value<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl IndexMut<usize> for Value<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_idx_mut(index).expect("index out of bounds")
    }
}

impl Default for Value<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn default() -> Self {
        Self::Static(StaticNode::Null)
    }
}

pub(super) struct BorrowDeserializer<'de>(Deserializer<'de>);

impl<'de> BorrowDeserializer<'de> {
    pub fn from_deserializer(de: Deserializer<'de>) -> Self {
        Self(de)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn parse(&mut self) -> Value<'de> {
        match unsafe { self.0.next_() } {
            Node::Static(s) => Value::Static(s),
            Node::String(s) => Value::from(s),
            Node::Array { len, count: _ } => self.parse_array(len),
            Node::Object { len, count: _ } => self.parse_map(len),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_array(&mut self, len: usize) -> Value<'de> {
        // Rust doesn't optimize the normal loop away here
        // so we write our own avoiding the length
        // checks during push
        let mut res: Vec<Value<'de>> = Vec::with_capacity(len);
        let res_ptr = res.as_mut_ptr();
        unsafe {
            for i in 0..len {
                res_ptr.add(i).write(self.parse());
            }
            res.set_len(len);
        }
        Value::Array(Box::new(res))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_map(&mut self, len: usize) -> Value<'de> {
        let mut res = Object::with_capacity_and_hasher(len, ObjectHasher::default());

        // Since we checked if it's empty we know that we at least have one
        // element so we eat this
        for _ in 0..len {
            if let Node::String(key) = unsafe { self.0.next_() } {
                #[cfg(not(feature = "value-no-dup-keys"))]
                unsafe {
                    res.insert_nocheck(key.into(), self.parse());
                };
                #[cfg(feature = "value-no-dup-keys")]
                res.insert(key.into(), self.parse());
            } else {
                unreachable!("parse_map: key not a string");
            }
        }
        Value::from(res)
    }
}
pub(super) struct BorrowSliceDeserializer<'tape, 'de> {
    tape: &'tape [Node<'de>],
    idx: usize,
}
impl<'tape, 'de> BorrowSliceDeserializer<'tape, 'de> {
    pub fn from_tape(de: &'tape [Node<'de>]) -> Self {
        Self { tape: de, idx: 0 }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub unsafe fn next_(&mut self) -> Node<'de> {
        let r = unsafe { *self.tape.get_kinda_unchecked(self.idx) };
        self.idx += 1;
        r
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn parse(&mut self) -> Value<'de> {
        match unsafe { self.next_() } {
            Node::Static(s) => Value::Static(s),
            Node::String(s) => Value::from(s),
            Node::Array { len, count: _ } => self.parse_array(len),
            Node::Object { len, count: _ } => self.parse_map(len),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_array(&mut self, len: usize) -> Value<'de> {
        // Rust doesn't optimize the normal loop away here
        // so we write our own avoiding the length
        // checks during push
        let mut res: Vec<Value<'de>> = Vec::with_capacity(len);
        let res_ptr = res.as_mut_ptr();
        unsafe {
            for i in 0..len {
                res_ptr.add(i).write(self.parse());
            }
            res.set_len(len);
        }
        Value::Array(Box::new(res))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_map(&mut self, len: usize) -> Value<'de> {
        let mut res = Object::with_capacity_and_hasher(len, ObjectHasher::default());

        // Since we checked if it's empty we know that we at least have one
        // element so we eat this
        for _ in 0..len {
            if let Node::String(key) = unsafe { self.next_() } {
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

#[cfg(test)]
mod test {
    #![allow(clippy::ignored_unit_patterns)]
    #![allow(clippy::cognitive_complexity)]
    use super::*;

    #[test]
    fn object_access() {
        let mut v = Value::null();
        assert_eq!(v.insert("key", ()), Err(AccessError::NotAnObject));
        assert_eq!(v.remove("key"), Err(AccessError::NotAnObject));
        let mut v = Value::object();
        assert_eq!(v.insert("key", 1), Ok(None));
        assert_eq!(v["key"], 1);
        assert_eq!(v.insert("key", 2), Ok(Some(Value::from(1))));
        v["key"] = 3.into();
        assert_eq!(v.remove("key"), Ok(Some(Value::from(3))));
    }

    #[test]
    fn array_access() {
        let mut v = Value::null();
        assert_eq!(v.push("key"), Err(AccessError::NotAnArray));
        assert_eq!(v.pop(), Err(AccessError::NotAnArray));
        let mut v = Value::array();
        assert_eq!(v.push(1), Ok(()));
        assert_eq!(v.push(2), Ok(()));
        assert_eq!(v[0], 1);
        v[0] = 0.into();
        v[1] = 1.into();
        assert_eq!(v.pop(), Ok(Some(Value::from(1))));
        assert_eq!(v.pop(), Ok(Some(Value::from(0))));
        assert_eq!(v.pop(), Ok(None));
    }

    #[cfg(feature = "128bit")]
    #[test]
    fn conversions_i128() {
        let v = Value::from(i128::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i128::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i64() {
        let v = Value::from(i64::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i64::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i32() {
        let v = Value::from(i32::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i32::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i16() {
        let v = Value::from(i16::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i16::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i8() {
        let v = Value::from(i8::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i8::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(v.is_i16());
        assert!(!v.is_u16());
        assert!(v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_usize() {
        let v = Value::from(usize::MIN as u64);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_usize());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[cfg(feature = "128bit")]
    #[test]
    fn conversions_u128() {
        let v = Value::from(u128::MIN);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u64() {
        let v = Value::from(u64::MIN);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u32() {
        let v = Value::from(u32::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(!v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u16() {
        let v = Value::from(u16::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u8() {
        let v = Value::from(u8::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_f64() {
        let v = Value::from(f64::MAX);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(f64::MIN);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from("not a f64");
        assert!(!v.is_f64_castable());
    }

    #[test]
    fn conversions_f32() {
        let v = Value::from(f32::MAX);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(f32::MIN);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_array() {
        let v = Value::from(vec![true]);
        assert!(v.is_array());
        assert_eq!(v.value_type(), ValueType::Array);
        let v = Value::from("no array");
        assert!(!v.is_array());
    }

    #[test]
    fn conversions_bool() {
        let v = Value::from(true);
        assert!(v.is_bool());
        assert_eq!(v.value_type(), ValueType::Bool);
        let v = Value::from("no bool");
        assert!(!v.is_bool());
    }

    #[test]
    fn conversions_float() {
        let v = Value::from(42.0);
        assert!(v.is_f64());
        assert_eq!(v.value_type(), ValueType::F64);
        let v = Value::from("no float");
        assert!(!v.is_f64());
    }

    #[test]
    fn conversions_int() {
        let v = Value::from(-42);
        assert!(v.is_i64());
        assert_eq!(v.value_type(), ValueType::I64);
        #[cfg(feature = "128bit")]
        {
            let v = Value::from(-42_i128);
            assert!(v.is_i64());
            assert!(v.is_i128());
            assert_eq!(v.value_type(), ValueType::I128);
        }
        let v = Value::from("no i64");
        assert!(!v.is_i64());
        assert!(!v.is_i128());
    }

    #[test]
    fn conversions_uint() {
        let v = Value::from(42_u64);
        assert!(v.is_u64());
        assert_eq!(v.value_type(), ValueType::U64);
        #[cfg(feature = "128bit")]
        {
            let v = Value::from(42_u128);
            assert!(v.is_u64());
            assert!(v.is_u128());
            assert_eq!(v.value_type(), ValueType::U128);
        }
        let v = Value::from("no u64");
        assert!(!v.is_u64());
        #[cfg(feature = "128bit")]
        assert!(!v.is_u128());
    }

    #[test]
    fn conversions_null() {
        let v = Value::from(());
        assert!(v.is_null());
        assert_eq!(v.value_type(), ValueType::Null);
        let v = Value::from("no null");
        assert!(!v.is_null());
    }

    #[test]
    fn conversions_object() {
        let v = Value::from(Object::with_capacity_and_hasher(1, ObjectHasher::default()));
        assert!(v.is_object());
        assert_eq!(v.value_type(), ValueType::Object);
        let v = Value::from("no object");
        assert!(!v.is_object());
    }

    #[test]
    fn conversions_str() {
        let v = Value::from("bla");
        assert!(v.is_str());
        assert_eq!(v.value_type(), ValueType::String);
        let v = Value::from(42);
        assert!(!v.is_str());
    }

    #[test]
    fn default() {
        assert_eq!(Value::default(), Value::null());
    }

    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    #[cfg(not(target_arch = "wasm32"))]
    fn arb_value() -> BoxedStrategy<Value<'static>> {
        let leaf = prop_oneof![
            Just(Value::Static(StaticNode::Null)),
            any::<bool>()
                .prop_map(StaticNode::Bool)
                .prop_map(Value::Static),
            any::<i64>()
                .prop_map(StaticNode::I64)
                .prop_map(Value::Static),
            any::<u64>()
                .prop_map(StaticNode::U64)
                .prop_map(Value::Static),
            any::<f64>()
                .prop_map(StaticNode::from)
                .prop_map(Value::Static),
            ".*".prop_map(Value::from),
        ];
        leaf.prop_recursive(
            8,   // 8 levels deep
            256, // Shoot for maximum size of 256 nodes
            10,  // We put up to 10 items per collection
            |inner| {
                prop_oneof![
                    // Take the inner strategy and make the two recursive cases.
                    prop::collection::vec(inner.clone(), 0..10).prop_map(Value::from),
                    prop::collection::hash_map(".*".prop_map(Cow::from), inner, 0..10)
                        .prop_map(|m| m.into_iter().collect()),
                ]
            },
        )
        .boxed()
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #![proptest_config(ProptestConfig {
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_to_owned(borrowed in arb_value()) {
            use crate::OwnedValue;
            let owned: OwnedValue = borrowed.clone().into();
            prop_assert_eq!(borrowed, owned);
        }
        #[test]
        fn prop_into_static(borrowed in arb_value()) {
            let static_borrowed = borrowed.clone().into_static();
            assert_eq!(borrowed, static_borrowed);
        }
        #[test]
        fn prop_clone_static(borrowed in arb_value()) {
            let static_borrowed = borrowed.clone_static();
            assert_eq!(borrowed, static_borrowed);
        }
        #[test]
        fn prop_serialize_deserialize(borrowed in arb_value()) {
            let mut string = borrowed.encode();
            let bytes = unsafe{ string.as_bytes_mut()};
            let decoded = to_value(bytes).expect("Failed to decode");
            prop_assert_eq!(borrowed, decoded);
        }
        #[test]
        #[allow(clippy::float_cmp)]
        fn prop_f64_cmp(f in proptest::num::f64::NORMAL) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);

        }

        #[test]
        #[allow(clippy::float_cmp)]
        fn prop_f32_cmp(f in proptest::num::f32::NORMAL) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);

        }
        #[test]
        fn prop_i64_cmp(f in proptest::num::i64::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_i32_cmp(f in proptest::num::i32::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_i16_cmp(f in proptest::num::i16::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_i8_cmp(f in proptest::num::i8::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_u64_cmp(f in proptest::num::u64::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }

        #[test]
        fn prop_usize_cmp(f in proptest::num::usize::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
         #[test]
        fn prop_u32_cmp(f in proptest::num::u32::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_u16_cmp(f in proptest::num::u16::ANY) {
            let v: Value = f.into();
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_u8_cmp(f in proptest::num::u8::ANY) {
            let v: Value = f.into();
            assert_eq!(v, &f);
            prop_assert_eq!(v, f);
        }
        #[test]
        fn prop_string_cmp(f in ".*") {
            let v: Value = f.clone().into();
            prop_assert_eq!(v.clone(), f.as_str());
            prop_assert_eq!(v, f);
        }

    }
    #[test]
    fn test_union_cmp() {
        let v: Value = ().into();
        assert_eq!(v, ());
    }
    #[test]
    #[allow(clippy::bool_assert_comparison)]
    fn test_bool_cmp() {
        let v: Value = true.into();
        assert_eq!(v, true);
        let v: Value = false.into();
        assert_eq!(v, false);
    }
    #[test]
    fn test_slice_cmp() {
        use std::iter::FromIterator;
        let v: Value = Value::from_iter(vec!["a", "b"]);
        assert_eq!(v, &["a", "b"][..]);
    }
    #[test]
    fn test_hashmap_cmp() {
        use std::iter::FromIterator;
        let v: Value = Value::from_iter(vec![("a", 1)]);
        assert_eq!(
            v,
            [("a", 1)]
                .iter()
                .copied()
                .collect::<std::collections::HashMap<&str, i32>>()
        );
    }

    #[test]
    fn test_option_from() {
        let v: Option<u8> = None;
        let v: Value = v.into();
        assert_eq!(v, ());
        let v: Option<u8> = Some(42);
        let v: Value = v.into();
        assert_eq!(v, 42);
    }

    // #[test]
    // fn size() {
    //     assert_eq!(std::mem::size_of::<Value>(), 24);
    // }
}
