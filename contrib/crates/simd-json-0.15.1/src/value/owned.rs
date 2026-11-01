/// A lifetime less DOM implementation. It uses strings to make te
/// structure fully owned, avoiding lifetimes at the cost of performance.
/// Access via array indexes is possible:
/// ```rust
/// use simd_json::{OwnedValue, json};
/// use simd_json::prelude::*;
/// let mut a = json!([1, 2, 3]);
/// assert_eq!(a[1], 2);
/// a[1] = 42.into();
/// assert_eq!(a[1], 42);
/// ```
///
/// Access via object keys is possible as well:
/// ```rust
/// use simd_json::{OwnedValue, json};
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
use halfbrown::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Representation of a JSON object
pub type Object = HashMap<String, Value, ObjectHasher>;

/// Parses a slice of bytes into a Value dom.
///
/// This function will rewrite the slice to de-escape strings.
/// We do not keep any references to the raw data but re-allocate
/// owned memory wherever required thus returning a value without
/// a lifetime.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn to_value(s: &mut [u8]) -> Result<Value> {
    match Deserializer::from_slice(s) {
        Ok(de) => Ok(OwnedDeserializer::from_deserializer(de).parse()),
        Err(e) => Err(e),
    }
}

/// Parses a slice of bytes into a Value dom.
///
/// This function will rewrite the slice to de-escape strings.
/// We do not keep any references to the raw data but re-allocate
/// owned memory wherever required thus returning a value without
/// a lifetime.
///
/// Passes in reusable buffers to reduce allocations.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
pub fn to_value_with_buffers(s: &mut [u8], buffers: &mut Buffers) -> Result<Value> {
    match Deserializer::from_slice_with_buffers(s, buffers) {
        Ok(de) => Ok(OwnedDeserializer::from_deserializer(de).parse()),
        Err(e) => Err(e),
    }
}

/// Owned JSON-DOM Value, consider using the `ValueTrait`
/// to access it's content.
/// This is slower then the `BorrowedValue` as a tradeoff
/// for getting rid of lifetimes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "ordered-float", derive(Eq))]
pub enum Value {
    /// Static values
    Static(StaticNode),
    /// string type
    String(String),
    /// array type
    Array(Box<Vec<Value>>),
    /// object type
    Object(Box<Object>),
}

impl Value {
    fn as_static(&self) -> Option<StaticNode> {
        match self {
            Value::Static(s) => Some(*s),
            _ => None,
        }
    }
}

impl ValueBuilder<'_> for Value {
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

impl ValueAsMutArray for Value {
    type Array = Vec<Self>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}
impl ValueAsMutObject for Value {
    type Object = Object;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }
}

impl TypedValue for Value {
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
impl ValueAsScalar for Value {
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
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}
impl ValueAsArray for Value {
    type Array = Vec<Self>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}
impl ValueAsObject for Value {
    type Object = Object;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }
}

impl ValueIntoString for Value {
    type String = String;

    fn into_string(self) -> Option<<Value as ValueIntoString>::String> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}
impl ValueIntoArray for Value {
    type Array = Vec<Self>;

    fn into_array(self) -> Option<<Value as ValueIntoArray>::Array> {
        match self {
            Self::Array(a) => Some(*a),
            _ => None,
        }
    }
}
impl ValueIntoObject for Value {
    type Object = Object;

    fn into_object(self) -> Option<<Value as ValueIntoObject>::Object> {
        match self {
            Self::Object(a) => Some(*a),
            _ => None,
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(s) => s.fmt(f),
            Self::String(s) => write!(f, "{s}"),
            Self::Array(a) => write!(f, "{a:?}"),
            Self::Object(o) => write!(f, "{o:?}"),
        }
    }
}

impl Index<&str> for Value {
    type Output = Self;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index(&self, index: &str) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl Index<usize> for Value {
    type Output = Self;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get_idx(index).expect("index out of bounds")
    }
}

impl IndexMut<&str> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl IndexMut<usize> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_idx_mut(index).expect("index out of bounds")
    }
}

impl Default for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn default() -> Self {
        Self::Static(StaticNode::Null)
    }
}

struct OwnedDeserializer<'de> {
    de: Deserializer<'de>,
}

impl<'de> OwnedDeserializer<'de> {
    pub fn from_deserializer(de: Deserializer<'de>) -> Self {
        Self { de }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn parse(&mut self) -> Value {
        match unsafe { self.de.next_() } {
            Node::Static(s) => Value::Static(s),
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
        Value::Array(Box::new(res))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn parse_map(&mut self, len: usize) -> Value {
        let mut res = Object::with_capacity_and_hasher(len, ObjectHasher::default());

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

#[cfg(test)]
mod test {
    #![allow(clippy::cognitive_complexity, clippy::ignored_unit_patterns)]
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
        #[cfg(feature = "128bit")]
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
        let v = Value::from(Object::with_capacity_and_hasher(0, ObjectHasher::default()));
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
    fn arb_value() -> BoxedStrategy<Value> {
        let leaf = prop_oneof![
            Just(Value::Static(StaticNode::Null)),
            any::<bool>()
                .prop_map(StaticNode::Bool)
                .prop_map(Value::Static),
            any::<i64>()
                .prop_map(StaticNode::I64)
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
                    prop::collection::hash_map(".*", inner, 0..10)
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
        fn prop_to_owned(owned in arb_value()) {
            use crate::BorrowedValue;
            let borrowed: BorrowedValue = owned.clone().into();
            prop_assert_eq!(owned, borrowed);
        }

        #[test]
        fn prop_serialize_deserialize(owned in arb_value()) {
            let mut string = owned.encode();
            let bytes = unsafe{ string.as_bytes_mut()};
            let decoded = to_value(bytes).expect("Failed to decode");
            prop_assert_eq!(owned, decoded);
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
            prop_assert_eq!(v.clone(), &f);
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
}
