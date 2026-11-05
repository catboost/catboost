// A lot of this logic is a re-implementation or copy of serde_json::Value
use crate::ErrorType;
use crate::{Error, ObjectHasher};
use crate::{
    prelude::*,
    serde::value::shared::MapKeyDeserializer,
    value::owned::{Object, Value},
};
use serde_ext::{
    de::{
        self, Deserialize, DeserializeSeed, Deserializer, EnumAccess, IntoDeserializer, MapAccess,
        SeqAccess, VariantAccess, Visitor,
    },
    forward_to_deserialize_any,
};
use std::fmt;

impl<'de> de::Deserializer<'de> for Value {
    type Error = Error;

    // Look at the input data to decide what Serde data model type to
    // deserialize as. Not all data formats are able to support this operation.
    // Formats that support `deserialize_any` are known as self-describing.
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self {
            Value::Static(StaticNode::Null) => visitor.visit_unit(),
            Value::Static(StaticNode::Bool(b)) => visitor.visit_bool(b),
            Value::Static(StaticNode::I64(n)) => visitor.visit_i64(n),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::I128(n)) => visitor.visit_i128(n),
            Value::Static(StaticNode::U64(n)) => visitor.visit_u64(n),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::U128(n)) => visitor.visit_u128(n),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Value::Static(StaticNode::F64(n)) => visitor.visit_f64(n.into()),
            Value::String(s) => visitor.visit_string(s),
            Value::Array(a) => visitor.visit_seq(Array(a.into_iter())),
            Value::Object(o) => visitor.visit_map(ObjectAccess {
                i: o.into_iter(),
                v: None,
            }),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self == Value::Static(StaticNode::Null) {
            visitor.visit_unit()
        } else {
            visitor.visit_some(self)
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_enum<V>(
        self,
        _name: &str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let (variant, value) = match self {
            Value::Object(value) => {
                let mut iter = value.into_iter();
                let Some((variant, value)) = iter.next() else {
                    return Err(crate::Deserializer::error(ErrorType::Eof));
                };
                // enums are encoded in json as maps with a single key:value pair
                if iter.next().is_some() {
                    return Err(crate::Deserializer::error(ErrorType::TrailingData));
                }
                (variant, Some(value))
            }
            Value::String(variant) => (variant, None),
            other => {
                return Err(crate::Deserializer::error(ErrorType::Unexpected(
                    Some(ValueType::Object),
                    Some(other.value_type()),
                )));
            }
        };

        visitor.visit_enum(EnumDeserializer { variant, value })
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self {
            // Give the visitor access to each element of the sequence.
            Value::Array(a) => visitor.visit_seq(Array(a.into_iter())),
            Value::Object(o) => visitor.visit_map(ObjectAccess::new(o.into_iter())),
            other => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                Some(other.value_type()),
            ))),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf unit unit_struct seq tuple
            tuple_struct map identifier ignored_any
    }
}

struct Array(std::vec::IntoIter<Value>);

// `SeqAccess` is provided to the `Visitor` to give it the ability to iterate
// through elements of the sequence.
impl<'de> SeqAccess<'de> for Array {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        self.0
            .next()
            .map_or(Ok(None), |v| seed.deserialize(v).map(Some))
    }
}

struct ArrayRef<'de>(std::slice::Iter<'de, Value>);

// `SeqAccess` is provided to the `Visitor` to give it the ability to iterate
// through elements of the sequence.
impl<'de> SeqAccess<'de> for ArrayRef<'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        self.0
            .next()
            .map_or(Ok(None), |v| seed.deserialize(v).map(Some))
    }
}

struct ObjectAccess<const N: usize = 32> {
    i: halfbrown::IntoIter<String, Value, N>,
    v: Option<Value>,
}

impl<const N: usize> ObjectAccess<N> {
    fn new(i: halfbrown::IntoIter<String, Value, N>) -> Self {
        Self { i, v: None }
    }
}
// `MapAccess` is provided to the `Visitor` to give it the ability to iterate
// through entries of the map.
impl<'de> MapAccess<'de> for ObjectAccess {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: DeserializeSeed<'de>,
    {
        match self.i.next() {
            Some((k, v)) => {
                self.v = Some(v);
                seed.deserialize(Value::String(k)).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        match self.v.take() {
            Some(v) => seed.deserialize(v),
            None => Err(crate::Deserializer::error(ErrorType::Eof)),
        }
    }
}

struct ObjectRefAccess<'de> {
    i: halfbrown::Iter<'de, String, Value>,
    v: Option<&'de Value>,
}

impl<'de> ObjectRefAccess<'de> {
    fn new(i: halfbrown::Iter<'de, String, Value>) -> Self {
        Self { i, v: None }
    }
}

// `MapAccess` is provided to the `Visitor` to give it the ability to iterate
// through entries of the map.
impl<'de> MapAccess<'de> for ObjectRefAccess<'de> {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: DeserializeSeed<'de>,
    {
        if let Some((k, v)) = self.i.next() {
            self.v = Some(v);
            seed.deserialize(MapKeyDeserializer::borrowed(k)).map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        match self.v.take() {
            Some(v) => seed.deserialize(v),
            None => Err(crate::Deserializer::error(ErrorType::Eof)),
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSONesque value")
    }

    /****************** unit ******************/
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Static(StaticNode::Null))
    }

    /****************** bool ******************/
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Static(StaticNode::Bool(value)))
    }

    /****************** Option ******************/
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Static(StaticNode::Null))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }

    /****************** enum ******************/
    /*
    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error> where
        A: EnumAccess<'de>,
    {
    }
     */

    /****************** i64 ******************/
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_i8<E>(self, value: i8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::I64(i64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::I64(i64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::I64(i64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::I64(value)))
    }

    #[cfg(feature = "128bit")]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::I128(value)))
    }

    /****************** u64 ******************/

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::U64(u64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::U64(u64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::U64(u64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::U64(value)))
    }

    #[cfg(feature = "128bit")]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::U128(value)))
    }
    /****************** f64 ******************/

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::from(f64::from(value))))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Static(StaticNode::from(value)))
    }

    /****************** stringy stuff ******************/
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_char<E>(self, value: char) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::from(value.to_string()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::from(value))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value.to_owned()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value))
    }

    /****************** byte stuff ******************/

    /*
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_borrowed_bytes<E>(self, value: &'de [u8]) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_str<E>(self, value: &[u8]) -> Result<Self::Value, E>
    where
    'a: 'de
        E: de::Error,
    {
      Ok(Value::String(value))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_string<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
      Ok(Value::String(&value))
    }
     */
    /****************** nexted stuff ******************/

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let size = map.size_hint().unwrap_or_default();

        let mut m = Object::with_capacity_and_hasher(size, ObjectHasher::default());
        while let Some(k) = map.next_key()? {
            let v = map.next_value()?;
            m.insert(k, v);
        }
        Ok(Value::from(m))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let size = seq.size_hint().unwrap_or_default();

        let mut v = Vec::with_capacity(size);
        while let Some(e) = seq.next_element()? {
            v.push(e);
        }
        Ok(Value::Array(Box::new(v)))
    }
}

struct EnumDeserializer {
    variant: String,
    value: Option<Value>,
}

impl<'de> EnumAccess<'de> for EnumDeserializer {
    type Error = Error;
    type Variant = VariantDeserializer;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, VariantDeserializer), Error>
    where
        V: DeserializeSeed<'de>,
    {
        let variant = self.variant.into_deserializer();
        let visitor = VariantDeserializer { value: self.value };
        seed.deserialize(variant).map(|v| (v, visitor))
    }
}

impl IntoDeserializer<'_, Error> for Value {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self::Deserializer {
        self
    }
}

struct VariantDeserializer {
    value: Option<Value>,
}

impl<'de> VariantAccess<'de> for VariantDeserializer {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        match self.value {
            Some(value) => Deserialize::deserialize(value),
            None => Ok(()),
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.value {
            Some(value) => seed.deserialize(value),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                None,
            ))),
        }
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Value::Array(v)) => {
                if v.is_empty() {
                    visitor.visit_unit()
                } else {
                    visitor.visit_seq(Array(v.into_iter()))
                }
            }
            Some(other) => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Array),
                Some(other.value_type()),
            ))),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Array),
                None,
            ))),
        }
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Value::Object(o)) => visitor.visit_map(ObjectAccess::new(o.into_iter())),
            Some(other) => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                Some(other.value_type()),
            ))),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                None,
            ))),
        }
    }
}

impl<'de> de::Deserializer<'de> for &'de Value {
    type Error = Error;

    // Look at the input data to decide what Serde data model type to
    // deserialize as. Not all data formats are able to support this operation.
    // Formats that support `deserialize_any` are known as self-describing.
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self {
            Value::Static(StaticNode::Null) => visitor.visit_unit(),
            Value::Static(StaticNode::Bool(b)) => visitor.visit_bool(*b),
            Value::Static(StaticNode::I64(n)) => visitor.visit_i64(*n),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::I128(n)) => visitor.visit_i128(*n),
            Value::Static(StaticNode::U64(n)) => visitor.visit_u64(*n),
            #[cfg(feature = "128bit")]
            Value::Static(StaticNode::U128(n)) => visitor.visit_u128(*n),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Value::Static(StaticNode::F64(n)) => visitor.visit_f64((*n).into()),
            Value::String(s) => visitor.visit_borrowed_str(s),
            Value::Array(a) => visitor.visit_seq(ArrayRef(a.as_slice().iter())),
            Value::Object(o) => visitor.visit_map(ObjectRefAccess::new(o.iter())),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self == &Value::Static(StaticNode::Null) {
            visitor.visit_unit()
        } else {
            visitor.visit_some(self)
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self {
            // Give the visitor access to each element of the sequence.
            Value::Array(a) => visitor.visit_seq(ArrayRef(a.as_slice().iter())),
            Value::Object(o) => visitor.visit_map(ObjectRefAccess::new(o.iter())),
            other => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                Some(other.value_type()),
            ))),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_enum<V>(
        self,
        _name: &str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let (variant, value) = match self {
            Value::Object(value) => {
                let mut iter = value.iter();
                let Some((variant, value)) = iter.next() else {
                    return Err(crate::Deserializer::error(ErrorType::Eof));
                };
                // enums are encoded in json as maps with a single key:value pair
                if iter.next().is_some() {
                    return Err(crate::Deserializer::error(ErrorType::TrailingData));
                }
                (variant, Some(value))
            }
            Value::String(variant) => (variant, None),
            other => {
                return Err(crate::Deserializer::error(ErrorType::Unexpected(
                    Some(ValueType::Object),
                    Some(other.value_type()),
                )));
            }
        };

        visitor.visit_enum(EnumRefDeserializer { variant, value })
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf unit unit_struct seq tuple
            tuple_struct map identifier ignored_any
    }
}

struct EnumRefDeserializer<'de> {
    variant: &'de str,
    value: Option<&'de Value>,
}

impl<'de> EnumAccess<'de> for EnumRefDeserializer<'de> {
    type Error = Error;
    type Variant = VariantRefDeserializer<'de>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Error>
    where
        V: DeserializeSeed<'de>,
    {
        let variant = self.variant.into_deserializer();
        let visitor = VariantRefDeserializer { value: self.value };
        seed.deserialize(variant).map(|v| (v, visitor))
    }
}
struct VariantRefDeserializer<'de> {
    value: Option<&'de Value>,
}

impl<'de> VariantAccess<'de> for VariantRefDeserializer<'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        match self.value {
            Some(value) => Deserialize::deserialize(value),
            None => Ok(()),
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.value {
            Some(value) => seed.deserialize(value),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                None,
            ))),
        }
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Value::Array(v)) => {
                if v.is_empty() {
                    visitor.visit_unit()
                } else {
                    visitor.visit_seq(ArrayRef(v.as_slice().iter()))
                }
            }
            Some(other) => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Array),
                Some(other.value_type()),
            ))),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Array),
                None,
            ))),
        }
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Value::Object(o)) => visitor.visit_map(ObjectRefAccess::new(o.iter())),
            Some(other) => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                Some(other.value_type()),
            ))),
            None => Err(crate::Deserializer::error(ErrorType::Unexpected(
                Some(ValueType::Object),
                None,
            ))),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{json, owned, prelude::*};
    use serde::Deserialize;

    #[test]
    fn option_field_absent_owned() {
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct Person {
            pub name: String,
            pub middle_name: Option<String>,
            pub friends: Vec<String>,
        }
        let mut raw_json = r#"{"name":"bob","friends":[]}"#.to_string();
        let result: Result<Person, _> = crate::to_owned_value(unsafe { raw_json.as_bytes_mut() })
            .and_then(super::super::from_value);
        assert_eq!(
            result,
            Ok(Person {
                name: "bob".to_string(),
                middle_name: None,
                friends: vec![]
            })
        );
    }
    #[test]
    fn option_field_present_owned() {
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct Point {
            pub x: u64,
            pub y: u64,
        }
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct Person {
            pub name: String,
            pub middle_name: Option<String>,
            pub friends: Vec<String>,
            pub pos: Point,
        }
        let mut raw_json =
            r#"{"name":"bob","middle_name": "frank", "friends":[], "pos": [1,2]}"#.to_string();
        let result: Result<Person, _> = crate::to_owned_value(unsafe { raw_json.as_bytes_mut() })
            .and_then(super::super::from_value);
        assert_eq!(
            result,
            Ok(Person {
                name: "bob".to_string(),
                middle_name: Some("frank".to_string()),
                friends: vec![],
                pos: Point { x: 1, y: 2 }
            })
        );
    }

    #[test]
    fn deserialize() {
        use halfbrown::{HashMap, hashmap};
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        #[serde(rename_all = "lowercase")]
        pub enum Rotate {
            Left,
            Right,
            Up,
            Down,
        }
        #[derive(serde::Deserialize, Debug, PartialEq)]
        pub struct Point {
            pub x: i64,
            pub y: i64,
            pub z: f64,
            pub rotate: Rotate,
        }
        #[derive(serde::Deserialize, Debug, PartialEq)]
        pub struct Person {
            pub name: String,
            pub middle_name: Option<String>,
            pub friends: Vec<String>,
            pub pos: Point,
            pub age: u64,
        }
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct TestStruct {
            pub key: HashMap<String, String>,
            pub vec: Vec<Vec<Option<u8>>>,
        }

        let mut raw_json =
            r#"{"name":"bob","middle_name": "frank", "friends":[], "pos": [-1, 2, -3.25, "up"], "age": 123}"#.to_string();
        let serde_result: Person = serde_json::from_str(&raw_json).expect("serde_json::from_str");
        let value =
            crate::to_owned_value(unsafe { raw_json.as_bytes_mut() }).expect("to_owned_value");
        let result: Person = super::super::from_refvalue(&value).expect("from_refvalue");
        let expected = Person {
            name: "bob".to_string(),
            middle_name: Some("frank".to_string()),
            friends: Vec::new(),
            pos: Point {
                x: -1,
                y: 2,
                z: -3.25_f64,
                rotate: Rotate::Up,
            },
            age: 123,
        };
        assert_eq!(result, expected);
        assert_eq!(result, serde_result);

        let mut raw_json = r#"{"key":{"subkey": "value"}, "vec":[[null], [1]]}"#.to_string();
        let value =
            crate::to_owned_value(unsafe { raw_json.as_bytes_mut() }).expect("to_owned_value");
        let result: TestStruct = super::super::from_refvalue(&value).expect("from_refvalue");
        let expected = TestStruct {
            key: hashmap!("subkey".to_string() => "value".to_string()),
            vec: vec![vec![None], vec![Some(1)]],
        };
        assert_eq!(result, expected);
    }

    #[cfg(feature = "128bit")]
    #[test]
    fn deserialize_128bit() {
        let value = i64::MIN as i128 - 1;
        let int128 = crate::OwnedValue::Static(crate::StaticNode::I128(value));
        let res: i128 = super::super::from_refvalue(&int128).expect("from_refvalue");
        assert_eq!(value, res);

        let value = u64::MAX as u128;
        let int128 = crate::OwnedValue::Static(crate::StaticNode::U128(value));
        let res: u128 = super::super::from_refvalue(&int128).expect("from_refvalue");
        assert_eq!(value, res);
    }
    #[test]
    fn variant() {
        struct NameAndConfig {
            name: String,
            config: Option<owned::Value>,
        }
        impl<'v> serde::Deserialize<'v> for NameAndConfig {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: serde::Deserializer<'v>,
            {
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum Variants {
                    Name(String),
                    NameAndConfig {
                        name: String,
                        config: Option<owned::Value>,
                    },
                }

                let var = Variants::deserialize(deserializer)?;

                match var {
                    Variants::Name(name) => Ok(NameAndConfig { name, config: None }),
                    Variants::NameAndConfig { name, config } => Ok(NameAndConfig { name, config }),
                }
            }
        }

        let v = json!({"name": "name", "config": 42});
        let nac = NameAndConfig::deserialize(v).expect("could structurize two element struct");
        assert_eq!(nac.name, "name");
        assert_eq!(nac.config.as_u8(), Some(42));
        let v = json!({"name": "name"});
        let nac = NameAndConfig::deserialize(v).expect("could structurize one element struct");
        assert_eq!(nac.name, "name");
        assert_eq!(nac.config, None);
        let v = json!("name");
        let nac = NameAndConfig::deserialize(v).expect("could structurize string");
        assert_eq!(nac.name, "name");
        assert_eq!(nac.config, None);
    }
}
