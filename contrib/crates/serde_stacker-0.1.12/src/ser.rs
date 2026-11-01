use crate::param::Param;
use serde::ser;
use std::fmt::Display;

/// Serializer adapter that avoids stack overflows by dynamically growing the
/// stack.
///
/// At each level of nested serialization, the adapter will check whether it is
/// within `red_zone` bytes of the end of the stack. If so, it will allocate a
/// new stack of size `stack_size` on which to continue deserialization.
pub struct Serializer<S> {
    pub ser: S,
    pub red_zone: usize,
    pub stack_size: usize,
}

impl<S> Serializer<S> {
    /// Build a serializer adapter with reasonable default `red_zone` (64 KB)
    /// and `stack_size` (2 MB).
    pub fn new(serializer: S) -> Self {
        let default_param = Param::default();
        Serializer {
            ser: serializer,
            red_zone: default_param.red_zone,
            stack_size: default_param.stack_size,
        }
    }
}

impl<S> ser::Serializer for Serializer<S>
where
    S: ser::Serializer,
{
    type Ok = S::Ok;
    type Error = S::Error;

    type SerializeSeq = SerializeSeq<S::SerializeSeq>;
    type SerializeTuple = SerializeTuple<S::SerializeTuple>;
    type SerializeTupleStruct = SerializeTupleStruct<S::SerializeTupleStruct>;
    type SerializeTupleVariant = SerializeTupleVariant<S::SerializeTupleVariant>;
    type SerializeMap = SerializeMap<S::SerializeMap>;
    type SerializeStruct = SerializeStruct<S::SerializeStruct>;
    type SerializeStructVariant = SerializeStructVariant<S::SerializeStructVariant>;

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_bool(v)
    }

    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_i8(v)
    }

    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_i16(v)
    }

    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_i32(v)
    }

    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_i64(v)
    }

    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_i128(v)
    }

    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_u8(v)
    }

    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_u16(v)
    }

    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_u32(v)
    }

    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_u64(v)
    }

    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_u128(v)
    }

    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_f32(v)
    }

    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_f64(v)
    }

    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_char(v)
    }

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_str(v)
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_bytes(v)
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_none()
    }

    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser.serialize_some(&Serialize::new(value, param))
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_unit()
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.ser.serialize_unit_struct(name)
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.ser
            .serialize_unit_variant(name, variant_index, variant)
    }

    fn serialize_newtype_struct<T>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_newtype_struct(name, &Serialize::new(value, param))
    }

    fn serialize_newtype_variant<T>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser.serialize_newtype_variant(
            name,
            variant_index,
            variant,
            &Serialize::new(value, param),
        )
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_seq(len)
            .map(|ser| SerializeSeq::new(ser, param))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_tuple(len)
            .map(|ser| SerializeTuple::new(ser, param))
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_tuple_struct(name, len)
            .map(|ser| SerializeTupleStruct::new(ser, param))
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_tuple_variant(name, variant_index, variant, len)
            .map(|ser| SerializeTupleVariant::new(ser, param))
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_map(len)
            .map(|ser| SerializeMap::new(ser, param))
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_struct(name, len)
            .map(|ser| SerializeStruct::new(ser, param))
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        let param = Param::new(self.red_zone, self.stack_size);
        self.ser
            .serialize_struct_variant(name, variant_index, variant, len)
            .map(|ser| SerializeStructVariant::new(ser, param))
    }

    fn collect_seq<I>(self, iter: I) -> Result<Self::Ok, Self::Error>
    where
        I: IntoIterator,
        I::Item: ser::Serialize,
    {
        let param = Param::new(self.red_zone, self.stack_size);
        let iter = iter
            .into_iter()
            .map(|item| SerializeSized::new(item, param));
        self.ser.collect_seq(iter)
    }

    fn collect_map<K, V, I>(self, iter: I) -> Result<Self::Ok, Self::Error>
    where
        K: ser::Serialize,
        V: ser::Serialize,
        I: IntoIterator<Item = (K, V)>,
    {
        let param = Param::new(self.red_zone, self.stack_size);
        let iter = iter
            .into_iter()
            .map(|(k, v)| (SerializeSized::new(k, param), SerializeSized::new(v, param)));
        self.ser.collect_map(iter)
    }

    fn collect_str<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Display,
    {
        self.ser.collect_str(value)
    }

    fn is_human_readable(&self) -> bool {
        self.ser.is_human_readable()
    }
}

pub struct SerializeSeq<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeSeq<S> {
    fn new(serialize_seq: S, param: Param) -> Self {
        SerializeSeq {
            ser: serialize_seq,
            param,
        }
    }
}

impl<S> ser::SerializeSeq for SerializeSeq<S>
where
    S: ser::SerializeSeq,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser
            .serialize_element(&Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }
}

pub struct SerializeTuple<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeTuple<S> {
    fn new(serialize_tuple: S, param: Param) -> Self {
        SerializeTuple {
            ser: serialize_tuple,
            param,
        }
    }
}

impl<S> ser::SerializeTuple for SerializeTuple<S>
where
    S: ser::SerializeTuple,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser
            .serialize_element(&Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }
}

pub struct SerializeTupleStruct<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeTupleStruct<S> {
    fn new(serialize_tuple_struct: S, param: Param) -> Self {
        SerializeTupleStruct {
            ser: serialize_tuple_struct,
            param,
        }
    }
}

impl<S> ser::SerializeTupleStruct for SerializeTupleStruct<S>
where
    S: ser::SerializeTupleStruct,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser.serialize_field(&Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }
}

pub struct SerializeTupleVariant<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeTupleVariant<S> {
    fn new(serialize_tuple_variant: S, param: Param) -> Self {
        SerializeTupleVariant {
            ser: serialize_tuple_variant,
            param,
        }
    }
}

impl<S> ser::SerializeTupleVariant for SerializeTupleVariant<S>
where
    S: ser::SerializeTupleVariant,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser.serialize_field(&Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }
}

pub struct SerializeMap<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeMap<S> {
    fn new(serialize_map: S, param: Param) -> Self {
        SerializeMap {
            ser: serialize_map,
            param,
        }
    }
}

impl<S> ser::SerializeMap for SerializeMap<S>
where
    S: ser::SerializeMap,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser.serialize_key(&Serialize::new(key, self.param))
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser.serialize_value(&Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }

    fn serialize_entry<K, V>(&mut self, key: &K, value: &V) -> Result<(), Self::Error>
    where
        K: ?Sized + ser::Serialize,
        V: ?Sized + ser::Serialize,
    {
        self.ser.serialize_entry(
            &Serialize::new(key, self.param),
            &Serialize::new(value, self.param),
        )
    }
}

pub struct SerializeStruct<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeStruct<S> {
    fn new(serialize_struct: S, param: Param) -> Self {
        SerializeStruct {
            ser: serialize_struct,
            param,
        }
    }
}

impl<S> ser::SerializeStruct for SerializeStruct<S>
where
    S: ser::SerializeStruct,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser
            .serialize_field(key, &Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }

    fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
        self.ser.skip_field(key)
    }
}

pub struct SerializeStructVariant<S> {
    ser: S,
    param: Param,
}

impl<S> SerializeStructVariant<S> {
    fn new(serialize_struct_variant: S, param: Param) -> Self {
        SerializeStructVariant {
            ser: serialize_struct_variant,
            param,
        }
    }
}

impl<S> ser::SerializeStructVariant for SerializeStructVariant<S>
where
    S: ser::SerializeStructVariant,
{
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + ser::Serialize,
    {
        self.ser
            .serialize_field(key, &Serialize::new(value, self.param))
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.ser.end()
    }

    fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
        self.ser.skip_field(key)
    }
}

struct Serialize<'a, T: ?Sized> {
    value: &'a T,
    param: Param,
}

impl<'a, T: ?Sized> Serialize<'a, T> {
    fn new(value: &'a T, param: Param) -> Self {
        Serialize { value, param }
    }
}

impl<'a, T> ser::Serialize for Serialize<'a, T>
where
    T: ?Sized + ser::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        ser::Serialize::serialize(
            self.value,
            Serializer {
                ser: serializer,
                red_zone: self.param.red_zone,
                stack_size: self.param.stack_size,
            },
        )
    }
}

struct SerializeSized<T> {
    value: T,
    param: Param,
}

impl<T> SerializeSized<T> {
    fn new(value: T, param: Param) -> Self {
        SerializeSized { value, param }
    }
}

impl<T> ser::Serialize for SerializeSized<T>
where
    T: ser::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        stacker::maybe_grow(self.param.red_zone, self.param.stack_size, || {
            ser::Serialize::serialize(
                &self.value,
                Serializer {
                    ser: serializer,
                    red_zone: self.param.red_zone,
                    stack_size: self.param.stack_size,
                },
            )
        })
    }
}
