use super::Value;
use crate::{BorrowedValue, prelude::*};

#[allow(clippy::cast_sign_loss, clippy::default_trait_access)]
impl PartialEq<BorrowedValue<'_>> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &BorrowedValue<'_>) -> bool {
        match (self, other) {
            (Self::Static(s1), BorrowedValue::Static(s2)) => s1 == s2,
            (Self::String(v1), BorrowedValue::String(v2)) => v1.eq(v2),
            (Self::Array(v1), BorrowedValue::Array(v2)) => v1.as_ref().eq(v2.as_ref()),
            (Self::Object(v1), BorrowedValue::Object(v2)) => {
                if v1.len() != v2.len() {
                    return false;
                }
                v1.iter()
                    .all(|(key, value)| v2.get(key.as_str()).is_some_and(|v| value == v))
            }
            _ => false,
        }
    }
}

#[allow(clippy::cast_sign_loss, clippy::default_trait_access)]
impl PartialEq for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Static(s1), Self::Static(s2)) => s1.eq(s2),
            (Self::String(v1), Self::String(v2)) => v1.eq(v2),
            (Self::Array(v1), Self::Array(v2)) => v1.eq(v2),
            (Self::Object(v1), Self::Object(v2)) => v1.eq(v2),
            _ => false,
        }
    }
}

impl<T> PartialEq<&T> for Value
where
    Value: PartialEq<T>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &&T) -> bool {
        self == *other
    }
}

impl PartialEq<()> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, _other: &()) -> bool {
        self.is_null()
    }
}

impl PartialEq<bool> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &bool) -> bool {
        self.as_bool().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<str> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &str) -> bool {
        self.as_str().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<&str> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl PartialEq<String> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &String) -> bool {
        self.as_str().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i8> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i8) -> bool {
        self.as_i8().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i16> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i16) -> bool {
        self.as_i16().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i32) -> bool {
        self.as_i32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i64) -> bool {
        self.as_i64().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<i128> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &i128) -> bool {
        self.as_i128().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u8> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u8) -> bool {
        self.as_u8().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u16> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u16) -> bool {
        self.as_u16().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u32) -> bool {
        self.as_u32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u64) -> bool {
        self.as_u64().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<usize> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &usize) -> bool {
        self.as_usize().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<u128> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &u128) -> bool {
        self.as_u128().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<f32> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &f32) -> bool {
        self.as_f32().is_some_and(|t| t.eq(other))
    }
}

impl PartialEq<f64> for Value {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &f64) -> bool {
        self.as_f64().is_some_and(|t| t.eq(other))
    }
}

impl<T> PartialEq<&[T]> for Value
where
    Value: PartialEq<T>,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &&[T]) -> bool {
        self.as_array().is_some_and(|t| t.eq(other))
    }
}
impl<K, T, S> PartialEq<std::collections::HashMap<K, T, S>> for Value
where
    K: AsRef<str> + std::hash::Hash + Eq,
    Value: PartialEq<T>,
    S: std::hash::BuildHasher,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn eq(&self, other: &std::collections::HashMap<K, T, S>) -> bool {
        self.as_object().is_some_and(|object| {
            object.len() == other.len()
                && other
                    .iter()
                    .all(|(key, value)| object.get(key.as_ref()).is_some_and(|v| *v == *value))
        })
    }
}
