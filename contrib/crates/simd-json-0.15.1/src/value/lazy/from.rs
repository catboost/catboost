use super::Value;
use crate::StaticNode;
use crate::{borrowed, cow::Cow};
use std::borrow::Cow as StdCow;

impl<'value> From<borrowed::Value<'value>> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: borrowed::Value<'value>) -> Self {
        Value::Value(StdCow::Owned(v))
    }
}

impl From<StaticNode> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: StaticNode) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl<'value, T> From<Option<T>> for Value<'_, '_, 'value>
where
    borrowed::Value<'value>: From<Option<T>>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: Option<T>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}
/********* str_ **********/
impl<'value> From<&'value str> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: &'value str) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

#[cfg(feature = "beef")]
impl<'value> From<std::borrow::Cow<'value, str>> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: std::borrow::Cow<'value, str>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

#[cfg(not(feature = "beef"))]
impl<'value> From<std::borrow::Cow<'value, str>> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: std::borrow::Cow<'value, str>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

#[cfg(feature = "beef")]
impl<'value> From<beef::lean::Cow<'value, str>> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: beef::lean::Cow<'value, str>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<String> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: String) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

/********* atoms **********/
impl From<bool> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: bool) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}
impl From<()> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: ()) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

/********* i_ **********/
impl From<i8> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: i8) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<i16> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: i16) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<i32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: i32) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<i64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: i64) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

#[cfg(feature = "128bit")]
impl From<i128> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: i128) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

/********* u_ **********/
impl From<u8> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: u8) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<u16> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: u16) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<u32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: u32) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<u64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: u64) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

#[cfg(feature = "128bit")]
impl From<u128> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: u128) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<usize> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: usize) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

/********* f_ **********/
impl From<f32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: f32) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl From<f64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: f64) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl<'value, S> From<Vec<S>> for Value<'_, '_, 'value>
where
    borrowed::Value<'value>: From<Vec<S>>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: Vec<S>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}

impl<'value, V: Into<borrowed::Value<'value>>> FromIterator<V> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from_iter<I: IntoIterator<Item = V>>(v: I) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from_iter(v)))
    }
}

impl<'value, K: Into<Cow<'value, str>>, V: Into<borrowed::Value<'value>>> FromIterator<(K, V)>
    for Value<'_, '_, 'value>
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from_iter<I: IntoIterator<Item = (K, V)>>(v: I) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from_iter(v)))
    }
}

impl<'value> From<crate::borrowed::Object<'value>> for Value<'_, '_, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(v: crate::borrowed::Object<'value>) -> Self {
        Value::Value(StdCow::Owned(borrowed::Value::from(v)))
    }
}
