/// Hacky serializer that only allows `u8`

use std::fmt;
use serde::ser::Impossible;
use serde::Serialize;

pub(crate) struct OnlyBytes;
pub(crate) struct Nope;

impl std::error::Error for Nope {
}

impl std::fmt::Display for Nope {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl std::fmt::Debug for Nope {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl serde::ser::Error for Nope {
    fn custom<T: fmt::Display>(_: T) -> Self {
        Self
    }
}

impl serde::de::Error for Nope {
    fn custom<T: fmt::Display>(_: T) -> Self {
        Self
    }
}

impl serde::Serializer for OnlyBytes {
    type Ok = u8;
    type Error = Nope;
    type SerializeSeq = Impossible<u8, Nope>;
    type SerializeTuple = Impossible<u8, Nope>;
    type SerializeTupleStruct = Impossible<u8, Nope>;
    type SerializeTupleVariant = Impossible<u8, Nope>;
    type SerializeMap = Impossible<u8, Nope>;
    type SerializeStruct = Impossible<u8, Nope>;
    type SerializeStructVariant = Impossible<u8, Nope>;

    fn serialize_u8(self, val: u8) -> Result<u8, Nope> {
        Ok(val)
    }

    fn serialize_bool(self, _: bool) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_i8(self, _: i8) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_i16(self, _: i16) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_i32(self, _: i32) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_i64(self, _: i64) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_u16(self, _: u16) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_u32(self, _: u32) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_u64(self, _: u64) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_f32(self, _: f32) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_f64(self, _: f64) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_char(self, _: char) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_str(self, _: &str) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_none(self) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_some<T: ?Sized>(self, _: &T) -> Result<u8, Nope> where T: Serialize {
        Err(Nope)
    }

    fn serialize_unit(self) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> Result<u8, Nope> {
        Err(Nope)
    }

    fn serialize_newtype_struct<T: ?Sized>(self, _: &'static str, _: &T) -> Result<u8, Nope> where T: Serialize {
        Err(Nope)
    }

    fn serialize_newtype_variant<T: ?Sized>(self, _: &'static str, _: u32, _: &'static str, _: &T) -> Result<u8, Nope> where T: Serialize {
        Err(Nope)
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Nope> {
        Err(Nope)
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Nope> {
        Err(Nope)
    }

    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeTupleStruct, Nope> {
        Err(Nope)
    }

    fn serialize_tuple_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self::SerializeTupleVariant, Nope> {
        Err(Nope)
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Nope> {
        Err(Nope)
    }

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self::SerializeStruct, Nope> {
        Err(Nope)
    }

    fn serialize_struct_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self::SerializeStructVariant, Nope> {
        Err(Nope)
    }

    fn collect_seq<I>(self, _: I) -> Result<u8, Nope> where I: IntoIterator, <I as IntoIterator>::Item: Serialize {
        Err(Nope)
    }

    fn collect_map<K, V, I>(self, _: I) -> Result<u8, Nope> where K: Serialize, V: Serialize, I: IntoIterator<Item = (K, V)> {
        Err(Nope)
    }

    fn collect_str<T: ?Sized>(self, _: &T) -> Result<u8, Nope> where T: fmt::Display {
        Err(Nope)
    }
}
