use std::borrow::Borrow;
use value_trait::{base::ValueAsScalar, derived::TypedScalarValue};

use super::Value;

impl PartialEq<()> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, _other: &()) -> bool {
        self.is_null()
    }
}

impl PartialEq<bool> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &bool) -> bool {
        self.as_bool().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<str> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &str) -> bool {
        self.as_str().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<&str> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl PartialEq<String> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &String) -> bool {
        self.as_str().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i8> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i8) -> bool {
        self.as_i8().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i16> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i16) -> bool {
        self.as_i16().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i32) -> bool {
        self.as_i32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i64) -> bool {
        self.as_i64().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i128> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i128) -> bool {
        self.as_i128().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u8> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u8) -> bool {
        self.as_u8().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u16> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u16) -> bool {
        self.as_u16().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u32) -> bool {
        self.as_u32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u64) -> bool {
        self.as_u64().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<usize> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &usize) -> bool {
        self.as_usize().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u128> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u128) -> bool {
        self.as_u128().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<f32> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &f32) -> bool {
        self.as_f32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<f64> for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &f64) -> bool {
        self.as_f64().is_some_and(|t| t.eq(other))
    }
}

impl<K, T, S> PartialEq<std::collections::HashMap<K, T, S>> for Value<'_, '_, '_>
where
    K: Borrow<str> + std::hash::Hash + Eq,
    for<'b, 't, 'i> T: PartialEq<Value<'b, 't, 'i>>,
    S: std::hash::BuildHasher,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &std::collections::HashMap<K, T, S>) -> bool {
        let Some(object) = self.as_object() else {
            return false;
        };
        if object.len() != other.len() {
            return false;
        }
        for (key, value) in object.iter() {
            if !other.get(key).is_some_and(|v| v == &value) {
                return false;
            }
        }
        true
    }
}
