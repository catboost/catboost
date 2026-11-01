use super::{Object, Value};
use crate::{BorrowedValue, StaticNode};

impl From<crate::BorrowedValue<'_>> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(b: BorrowedValue<'_>) -> Self {
        match b {
            BorrowedValue::Static(s) => Self::from(s),
            BorrowedValue::String(s) => Self::from(s.to_string()),
            BorrowedValue::Array(a) => a.into_iter().collect(),
            BorrowedValue::Object(m) => m.into_iter().collect(),
        }
    }
}

impl<T> From<Option<T>> for Value
where
    Value: From<T>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(s: Option<T>) -> Self {
        s.map_or(Value::Static(StaticNode::Null), Value::from)
    }
}

impl From<StaticNode> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(s: StaticNode) -> Self {
        Self::Static(s)
    }
}
/********* str_ **********/

impl From<&str> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl<'value> From<std::borrow::Cow<'value, str>> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(c: std::borrow::Cow<'value, str>) -> Self {
        Self::String(c.to_string())
    }
}

#[cfg(feature = "beef")]
impl<'value> From<beef::lean::Cow<'value, str>> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(c: beef::lean::Cow<'value, str>) -> Self {
        Self::String(c.to_string())
    }
}

impl From<String> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&String> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(s: &String) -> Self {
        Self::String(s.clone())
    }
}

/********* atoms **********/

impl From<bool> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(b: bool) -> Self {
        Self::Static(StaticNode::Bool(b))
    }
}

impl From<()> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(_b: ()) -> Self {
        Self::Static(StaticNode::Null)
    }
}

/********* i_ **********/
impl From<i8> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: i8) -> Self {
        Self::Static(StaticNode::I64(i64::from(i)))
    }
}

impl From<i16> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: i16) -> Self {
        Self::Static(StaticNode::I64(i64::from(i)))
    }
}

impl From<i32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: i32) -> Self {
        Self::Static(StaticNode::I64(i64::from(i)))
    }
}

impl From<i64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: i64) -> Self {
        Self::Static(StaticNode::I64(i))
    }
}
#[cfg(feature = "128bit")]
impl From<i128> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: i128) -> Self {
        Self::Static(StaticNode::I128(i))
    }
}

/********* u_ **********/
impl From<u8> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: u8) -> Self {
        Self::Static(StaticNode::U64(u64::from(i)))
    }
}

impl From<u16> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: u16) -> Self {
        Self::Static(StaticNode::U64(u64::from(i)))
    }
}

impl From<u32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: u32) -> Self {
        Self::Static(StaticNode::U64(u64::from(i)))
    }
}

impl From<u64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap)]
    fn from(i: u64) -> Self {
        Self::Static(StaticNode::U64(i))
    }
}

#[cfg(feature = "128bit")]
impl From<u128> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: u128) -> Self {
        Self::Static(StaticNode::U128(i))
    }
}

impl From<usize> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(i: usize) -> Self {
        Self::Static(StaticNode::U64(i as u64))
    }
}

/********* f_ **********/
impl From<f32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(f: f32) -> Self {
        Self::Static(StaticNode::from(f64::from(f)))
    }
}

impl From<f64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(f: f64) -> Self {
        Self::Static(StaticNode::from(f))
    }
}

impl<S> From<Vec<S>> for Value
where
    Value: From<S>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: Vec<S>) -> Self {
        v.into_iter().collect()
    }
}

impl<V: Into<Value>> FromIterator<V> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self::Array(Box::new(iter.into_iter().map(Into::into).collect()))
    }
}

impl<K: ToString, V: Into<Value>> FromIterator<(K, V)> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self::Object(Box::new(
            iter.into_iter()
                .map(|(k, v)| (k.to_string(), Into::into(v)))
                .collect(),
        ))
    }
}

impl From<Object> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: Object) -> Self {
        Self::Object(Box::new(v))
    }
}

impl From<std::collections::HashMap<String, Value>> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: std::collections::HashMap<String, Self>) -> Self {
        Self::from(v.into_iter().collect::<Object>())
    }
}
