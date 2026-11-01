use crate::serde_ext::de::IntoDeserializer;
use crate::{Deserializer, Error, ErrorType, Node, Result, StaticNode, macros::stry};
use serde_ext::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_ext::forward_to_deserialize_any;
use std::str;

impl<'a, 'de> de::Deserializer<'de> for &'a mut Deserializer<'de>
where
    'de: 'a,
{
    type Error = Error;

    // Look at the input data to decide what Serde data model type to
    // deserialize as. Not all data formats are able to support this operation.
    // Formats that support `deserialize_any` are known as self-describing.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match stry!(self.next()) {
            Node::String(s) => visitor.visit_borrowed_str(s),
            Node::Static(StaticNode::Null) => visitor.visit_unit(),
            Node::Static(StaticNode::Bool(b)) => visitor.visit_bool(b),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Node::Static(StaticNode::F64(n)) => visitor.visit_f64(n.into()),
            Node::Static(StaticNode::I64(n)) => visitor.visit_i64(n),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::I128(n)) => visitor.visit_i128(n),
            Node::Static(StaticNode::U64(n)) => visitor.visit_u64(n),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::U128(n)) => visitor.visit_u128(n),
            Node::Array { len, count: _ } => visitor.visit_seq(CommaSeparated::new(self, len)),
            Node::Object { len, count: _ } => visitor.visit_map(CommaSeparated::new(self, len)),
        }
    }

    // Uses the `parse_bool` parsing function defined above to read the JSON
    // identifier `true` or `false` from the input.
    //
    // Parsing refers to looking at the input and deciding that it contains the
    // JSON value `true` or `false`.
    //
    // Deserialization refers to mapping that JSON value into Serde's data
    // model by invoking one of the `Visitor` methods. In the case of JSON and
    // bool that mapping is straightforward so the distinction may seem silly,
    // but in other cases Deserializers sometimes perform non-obvious mappings.
    // For example the TOML format has a Datetime type and Serde's data model
    // does not. In the `toml` crate, a Datetime in the input is deserialized by
    // mapping it to a Serde data model "struct" type with a special name and a
    // single field containing the Datetime represented as a string.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match stry!(self.next()) {
            Node::Static(StaticNode::Bool(b)) => visitor.visit_bool(b),
            _c => Err(Deserializer::error(ErrorType::ExpectedBoolean)),
        }
    }

    // Refer to the "Understanding deserializer lifetimes" page for information
    // about the three deserialization flavors of strings in Serde.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Ok(Node::String(s)) = self.next() {
            visitor.visit_borrowed_str(s)
        } else {
            Err(Deserializer::error(ErrorType::ExpectedString))
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Ok(Node::String(s)) = self.next() {
            visitor.visit_str(s)
        } else {
            Err(Deserializer::error(ErrorType::ExpectedString))
        }
    }

    // The `parse_signed` function is generic over the integer type `T` so here
    // it is invoked with `T=i8`. The next 8 methods are similar.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i8(stry!(self.parse_i8()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(stry!(self.parse_i16()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(stry!(self.parse_i32()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(stry!(self.parse_i64()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i128(stry!(self.parse_i128()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(stry!(self.parse_u8()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(stry!(self.parse_u16()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u32(stry!(self.parse_u32()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(stry!(self.parse_u64()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u128(stry!(self.parse_u128()))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let v: f64 = stry!(self.parse_double());
        visitor.visit_f32(v as f32)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(stry!(self.parse_double()))
    }

    // An absent optional is represented as the JSON `null` and a present
    // optional is represented as just the contained value.
    //
    // As commented in `Serializer` implementation, this is a lossy
    // representation. For example the values `Some(())` and `None` both
    // serialize as just `null`. Unfortunately this is typically what people
    // expect when working with JSON. Other formats are encouraged to behave
    // more intelligently if possible.

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if stry!(self.peek()) == Node::Static(StaticNode::Null) {
            self.skip();
            visitor.visit_unit()
        } else {
            visitor.visit_some(self)
        }
    }

    // In Serde, unit means an anonymous value containing no data.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if stry!(self.next()) != Node::Static(StaticNode::Null) {
            return Err(Deserializer::error(ErrorType::ExpectedNull));
        }
        visitor.visit_unit()
    }

    // Deserialization of compound types like sequences and maps happens by
    // passing the visitor an "Access" object that gives it the ability to
    // iterate through the data contained in the sequence.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // Parse the opening bracket of the sequence.
        if let Ok(Node::Array { len, count: _ }) = self.next() {
            // Give the visitor access to each element of the sequence.
            visitor.visit_seq(CommaSeparated::new(self, len))
        } else {
            Err(Deserializer::error(ErrorType::ExpectedArray))
        }
    }

    // Tuples look just like sequences in JSON. Some formats may be able to
    // represent tuples more efficiently.
    //
    // As indicated by the length parameter, the `Deserialize` implementation
    // for a tuple in the Serde data model is required to know the length of the
    // tuple before even looking at the input data.

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
        // tuples have a known length damn you serde ...
        //self.skip();
        // r
    }

    // Tuple structs look just like sequences in JSON.
    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    // Unit struct means a named value containing no data.
    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    // As is done here, serializers are encouraged to treat newtype structs as
    // insignificant wrappers around the data they contain. That means not
    // parsing anything other than the contained value.
    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // Parse the opening bracket of the sequence.
        if let Ok(Node::Object { len, count: _ }) = self.next() {
            // Give the visitor access to each element of the sequence.
            visitor.visit_map(CommaSeparated::new(self, len))
        } else {
            Err(Deserializer::error(ErrorType::ExpectedMap))
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.next() {
            // Give the visitor access to each element of the sequence.
            Ok(Node::Object { len, count: _ }) => visitor.visit_map(CommaSeparated::new(self, len)),
            Ok(Node::Array { len, count: _ }) => visitor.visit_seq(CommaSeparated::new(self, len)),
            _ => Err(Deserializer::error(ErrorType::ExpectedMap)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        // Parse the opening bracket of the sequence.
        match self.next() {
            Ok(Node::Object { len: 1, .. }) => {
                // Give the visitor access to each element of the sequence.
                // let value = ri!(visitor.visit_enum(VariantAccess::new(self)));
                visitor.visit_enum(VariantAccess::new(self))
            }
            Ok(Node::String(s)) => visitor.visit_enum(s.into_deserializer()),
            _ => Err(Deserializer::error(ErrorType::ExpectedEnum)),
        }
    }

    forward_to_deserialize_any! {
            char
            bytes byte_buf
            identifier ignored_any
    }
}

// From  https://github.com/serde-rs/json/blob/2d81cbd11302bd246db248dfb335110d1827e893/src/de.rs
struct VariantAccess<'a, 'de> {
    de: &'a mut Deserializer<'de>,
}

impl<'a, 'de> VariantAccess<'a, 'de> {
    fn new(de: &'a mut Deserializer<'de>) -> Self {
        VariantAccess { de }
    }
}

impl<'de> de::EnumAccess<'de> for VariantAccess<'_, 'de> {
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self)>
    where
        V: de::DeserializeSeed<'de>,
    {
        let val = stry!(seed.deserialize(&mut *self.de));
        Ok((val, self))
    }
}

impl<'de> de::VariantAccess<'de> for VariantAccess<'_, 'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        de::Deserialize::deserialize(self.de)
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: de::DeserializeSeed<'de>,
    {
        seed.deserialize(self.de)
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        de::Deserializer::deserialize_seq(self.de, visitor)
    }

    fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        de::Deserializer::deserialize_struct(self.de, "", fields, visitor)
    }
}

// In order to handle commas correctly when deserializing a JSON array or map,
// we need to track whether we are on the first element or past the first
// element.
struct CommaSeparated<'a, 'de: 'a> {
    de: &'a mut Deserializer<'de>,
    len: usize,
}
impl<'a, 'de> CommaSeparated<'a, 'de> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn new(de: &'a mut Deserializer<'de>, len: usize) -> Self {
        CommaSeparated { de, len }
    }
}

// `SeqAccess` is provided to the `Visitor` to give it the ability to iterate
// through elements of the sequence.
impl<'de> SeqAccess<'de> for CommaSeparated<'_, 'de> {
    type Error = Error;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        if self.len == 0 {
            Ok(None)
        } else {
            self.len -= 1;
            seed.deserialize(&mut *self.de).map(Some)
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn size_hint(&self) -> Option<usize> {
        Some(self.len)
    }
}

// `MapAccess` is provided to the `Visitor` to give it the ability to iterate
// through entries of the map.
impl<'de> MapAccess<'de> for CommaSeparated<'_, 'de> {
    type Error = Error;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: DeserializeSeed<'de>,
    {
        if self.len == 0 {
            Ok(None)
        } else {
            self.len -= 1;
            seed.deserialize(MapKey { de: &mut *self.de }).map(Some)
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: DeserializeSeed<'de>,
    {
        // read the value
        seed.deserialize(&mut *self.de)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn size_hint(&self) -> Option<usize> {
        Some(self.len)
    }
}

// `MapKey` is provided to the `Visitor` to give it the ability to parse integers
// from string as JSON keys are always string
struct MapKey<'de: 'a, 'a> {
    de: &'a mut Deserializer<'de>,
}

macro_rules! deserialize_integer_key {
    ($method:ident => $visit:ident; $type:ty) => {
        fn $method<V>(self, visitor: V) -> Result<V::Value>
        where
            V: de::Visitor<'de>,
        {
            visitor.$visit(stry!(match unsafe { self.de.next_() } {
                Node::String(s) => s
                    .parse::<$type>()
                    .map_err(|_| Deserializer::error(ErrorType::InvalidNumber)),
                _ => Err(Deserializer::error(ErrorType::ExpectedString)),
            }))
        }
    };
}

impl<'de> de::Deserializer<'de> for MapKey<'de, '_> {
    type Error = Error;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        match stry!(self.de.next()) {
            Node::String(s) => visitor.visit_borrowed_str(s),
            _ => Err(Deserializer::error(ErrorType::ExpectedString)),
        }
    }

    deserialize_integer_key!(deserialize_i8 => visit_i8; i8);
    deserialize_integer_key!(deserialize_i16 => visit_i16; i16);
    deserialize_integer_key!(deserialize_i32 => visit_i32; i32);
    deserialize_integer_key!(deserialize_i64 => visit_i64; i64);
    deserialize_integer_key!(deserialize_u8 => visit_u8; u8);
    deserialize_integer_key!(deserialize_u16 => visit_u16; u16);
    deserialize_integer_key!(deserialize_u32 => visit_u32; u32);
    deserialize_integer_key!(deserialize_u64 => visit_u64; u64);

    #[cfg(feature = "128bit")]
    deserialize_integer_key!(deserialize_i128 => visit_i128; i128);
    #[cfg(feature = "128bit")]
    deserialize_integer_key!(deserialize_u128 => visit_u128; u128);

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        // Map keys cannot be null.
        visitor.visit_some(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        self.de.deserialize_enum(name, variants, visitor)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        self.de.deserialize_bytes(visitor)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value>
    where
        V: de::Visitor<'de>,
    {
        self.de.deserialize_bytes(visitor)
    }

    forward_to_deserialize_any! {
        bool f32 f64 char str string unit unit_struct seq tuple tuple_struct map
        struct identifier ignored_any
    }
}

#[cfg(test)]
mod tests;
