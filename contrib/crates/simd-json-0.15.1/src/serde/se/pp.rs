use crate::{Error, ErrorType, macros::stry};
use serde_ext::ser;
use std::io::Write;
use std::str;
use value_trait::generator::BaseGenerator;

use super::key_must_be_a_string;

macro_rules! iomap {
    ($e:expr_2021) => {
        ($e).map_err(|err| Error::generic(ErrorType::Io(err)))
    };
}

/// Write a value to a vector
/// # Errors
/// when the data can not be written
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn to_vec_pretty<T>(to: &T) -> crate::Result<Vec<u8>>
where
    T: ser::Serialize + ?Sized,
{
    let v = Vec::with_capacity(512);
    let mut s = PrettySerializer::new(v);
    to.serialize(&mut s).map(|()| s.writer)
}

/// Write a value to a string
///
/// # Errors
/// when the data can not be written
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn to_string_pretty<T>(to: &T) -> crate::Result<String>
where
    T: ser::Serialize + ?Sized,
{
    to_vec_pretty(to).map(|v| unsafe { String::from_utf8_unchecked(v) })
}

/// Write a value to a string
/// # Errors
/// when the data can not be written
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn to_writer_pretty<T, W>(writer: W, to: &T) -> crate::Result<()>
where
    T: ser::Serialize + ?Sized,
    W: Write,
{
    let mut s = PrettySerializer::new(writer);
    to.serialize(&mut s)
}
struct PrettySerializer<W: Write> {
    writer: W,
    dent: u32,
}
impl<W: Write> PrettySerializer<W> {
    fn new(writer: W) -> Self {
        Self { writer, dent: 0 }
    }
}

impl<W> BaseGenerator for PrettySerializer<W>
where
    W: Write,
{
    type T = W;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_writer(&mut self) -> &mut Self::T {
        &mut self.writer
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_min(&mut self, _slice: &[u8], min: u8) -> std::io::Result<()> {
        self.writer.write_all(&[min])
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn new_line(&mut self) -> std::io::Result<()> {
        self.write_char(b'\n').and_then(|()| match self.dent {
            0 => Ok(()),
            1 => self.get_writer().write_all(b"  "),
            2 => self.get_writer().write_all(b"    "),
            3 => self.get_writer().write_all(b"      "),
            4 => self.get_writer().write_all(b"        "),
            5 => self.get_writer().write_all(b"          "),
            6 => self.get_writer().write_all(b"            "),
            7 => self.get_writer().write_all(b"              "),
            8 => self.get_writer().write_all(b"                "),
            9 => self.get_writer().write_all(b"                  "),
            _ => {
                for _ in 0..(self.dent * 2) {
                    stry!(self.get_writer().write_all(b" "));
                }
                Ok(())
            }
        })
    }

    fn indent(&mut self) {
        self.dent += 1;
    }

    fn dedent(&mut self) {
        self.dent -= 1;
    }
}
struct SerializeSeq<'serializer, W: Write + 'serializer> {
    s: &'serializer mut PrettySerializer<W>,
    first: bool,
}
impl<W> ser::SerializeSeq for SerializeSeq<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeSeq {
            ref mut s,
            ref mut first,
            ..
        } = *self;
        if *first {
            *first = false;
            iomap!(s.new_line()).and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(s.write(b",").and_then(|()| s.new_line()))
                .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"]")))
        }
    }
}

impl<W> ser::SerializeTuple for SerializeSeq<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeSeq {
            ref mut s,
            ref mut first,
        } = *self;
        if *first {
            *first = false;
            iomap!(s.new_line()).and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(s.write(b",").and_then(|()| s.new_line()))
                .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"]")))
        }
    }
}

impl<W> ser::SerializeTupleStruct for SerializeSeq<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeSeq {
            ref mut s,
            ref mut first,
        } = *self;
        if *first {
            *first = false;
            iomap!(s.new_line()).and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(s.write(b",").and_then(|()| s.new_line()))
                .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"]")))
        }
    }
}

impl<W> ser::SerializeTupleVariant for SerializeSeq<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeSeq {
            ref mut s,
            ref mut first,
        } = *self;
        if *first {
            *first = false;
            iomap!(s.new_line()).and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(s.write(b",").and_then(|()| s.new_line()))
                .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"}")))
        }
    }
}

struct MapKeySerializer<'serializer, W: Write + 'serializer> {
    s: &'serializer mut PrettySerializer<W>,
}

impl<W> ser::Serializer for MapKeySerializer<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_str(self, value: &str) -> Result<(), Self::Error> {
        self.s.serialize_str(value)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<(), Self::Error> {
        self.s.serialize_str(variant)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        value.serialize(self)
    }

    type SerializeSeq = ser::Impossible<(), Error>;
    type SerializeTuple = ser::Impossible<(), Error>;
    type SerializeTupleStruct = ser::Impossible<(), Error>;
    type SerializeTupleVariant = ser::Impossible<(), Error>;
    type SerializeMap = ser::Impossible<(), Error>;
    type SerializeStruct = ser::Impossible<(), Error>;
    type SerializeStructVariant = ser::Impossible<(), Error>;

    fn serialize_bool(self, _value: bool) -> Result<(), Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        iomap!(
            self.s
                .write_char(b'"')
                .and_then(|()| self.s.write_int(v))
                .and_then(|()| self.s.write_char(b'"'))
        )
    }

    fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        self.s.serialize_str(&v.to_string())
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(key_must_be_a_string())
    }
}

struct SerializeMap<'serializer, W: Write + 'serializer> {
    s: &'serializer mut PrettySerializer<W>,
    first: bool,
}

impl<W> ser::SerializeMap for SerializeMap<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_key<T>(&mut self, key: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeMap {
            ref mut s,
            ref mut first,
            ..
        } = *self;

        if *first {
            *first = false;
            iomap!(s.new_line())
                .and_then(|()| key.serialize(MapKeySerializer { s: &mut **s }))
                .and_then(|()| iomap!(s.write(b": ")))
        } else {
            iomap!(s.write(b",").and_then(|()| s.new_line()))
                .and_then(|()| key.serialize(MapKeySerializer { s: &mut **s }))
                .and_then(|()| iomap!(s.write(b": ")))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_value<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeMap { ref mut s, .. } = *self;
        value.serialize(&mut **s)
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"}")))
        }
    }
}

impl<W> ser::SerializeStruct for SerializeMap<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeMap {
            ref mut s,
            ref mut first,
            ..
        } = *self;
        if *first {
            *first = false;
            iomap!(
                s.new_line()
                    .and_then(|()| s.write_simple_string(key))
                    .and_then(|()| s.write(b": "))
            )
            .and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(
                s.write(b",")
                    .and_then(|()| s.write_simple_string(key))
                    .and_then(|()| s.write(b": "))
            )
            .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.first {
            Ok(())
        } else {
            self.s.dedent();
            iomap!(self.s.new_line().and_then(|()| self.s.write(b"}")))
        }
    }
}

struct SerializeStructVariant<'serializer, W: Write + 'serializer> {
    s: &'serializer mut PrettySerializer<W>,
    first: bool,
}

impl<W> ser::SerializeStructVariant for SerializeStructVariant<'_, W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        let SerializeStructVariant {
            ref mut s,
            ref mut first,
            ..
        } = *self;
        if *first {
            *first = false;
            s.indent();
            iomap!(
                s.new_line()
                    .and_then(|()| s.write_simple_string(key))
                    .and_then(|()| s.write(b": "))
            )
            .and_then(|()| value.serialize(&mut **s))
        } else {
            iomap!(
                s.write(b",")
                    .and_then(|()| s.write_simple_string(key))
                    .and_then(|()| s.write(b": "))
            )
            .and_then(|()| value.serialize(&mut **s))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.s.dedent();
        iomap!(self.s.new_line().and_then(|()| self.s.write(b"}"))).and_then(move |()| {
            if self.first {
                Ok(())
            } else {
                self.s.dedent();

                iomap!(self.s.new_line().and_then(|()| self.s.write(b"}")))
            }
        })
    }
}

impl<'writer, W> ser::Serializer for &'writer mut PrettySerializer<W>
where
    W: Write,
{
    type Ok = ();
    type Error = Error;
    type SerializeSeq = SerializeSeq<'writer, W>;
    type SerializeTuple = SerializeSeq<'writer, W>;
    type SerializeTupleStruct = SerializeSeq<'writer, W>;
    type SerializeTupleVariant = SerializeSeq<'writer, W>;
    type SerializeMap = SerializeMap<'writer, W>;
    type SerializeStruct = SerializeMap<'writer, W>;
    type SerializeStructVariant = SerializeStructVariant<'writer, W>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        if v {
            iomap!(self.write(b"true"))
        } else {
            iomap!(self.write(b"false"))
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_int(v))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_float(f64::from(v)))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_float(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        // A char encoded as UTF-8 takes 4 bytes at most.
        // taken from: https://docs.serde.rs/src/serde_json/ser.rs.html#213
        let mut buf = [0; 4];
        iomap!(self.write_simple_string(v.encode_utf8(&mut buf)))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_string(v))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write(b"[").and_then(|()| {
            if let Some((first, rest)) = v.split_first() {
                self.indent();
                self.new_line().and_then(|()| {
                    self.write_int(*first).and_then(|()| {
                        for v in rest {
                            self.write(b",").and_then(|()| self.write_int(*v))?;
                        }
                        self.dedent();
                        self.new_line().and_then(|()| self.write(b"]"))
                    })
                })
            } else {
                self.write(b"]")
            }
        }))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        value.serialize(self)
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write(b"null"))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        iomap!(self.write_simple_string(variant))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        value.serialize(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + serde_ext::Serialize,
    {
        iomap!(
            self.write(b"{")
                .and_then(|()| self.write_simple_string(variant))
                .and_then(|()| self.write(b": "))
        )
        .and_then(|()| value.serialize(&mut *self))
        .and_then(|()| iomap!(self.write(b"}")))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        if len == Some(0) {
            iomap!(self.write(b"[]"))
        } else {
            self.indent();
            iomap!(self.write(b"["))
        }
        .map(move |()| SerializeSeq {
            s: self,
            first: true,
        })
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.serialize_seq(Some(len))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.serialize_seq(Some(len))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.indent();
        iomap!(
            self.write(b"{")
                .and_then(|()| self.new_line())
                .and_then(|()| self.write_simple_string(variant))
                .and_then(|()| self.write(b": "))
        )
        .and_then(move |()| self.serialize_seq(Some(len)))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        if len == Some(0) {
            iomap!(self.write(b"{}"))
        } else {
            self.indent();
            iomap!(self.write(b"{"))
        }
        .map(move |()| SerializeMap {
            s: self,
            first: true,
        })
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.serialize_map(Some(len))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.indent();
        iomap!(
            self.write(b"{")
                .and_then(|()| self.new_line())
                .and_then(|()| self.write_simple_string(variant))
                .and_then(|()| self.write(b": "))
        )
        .and_then(move |()| {
            if len == 0 {
                iomap!(self.write(b"{}"))
            } else {
                iomap!(self.write(b"{"))
            }
            .map(move |()| SerializeStructVariant {
                s: self,
                first: true,
            })
        })
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::ignored_unit_patterns, unused_imports)]
    use crate::OwnedValue as Value;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::StaticNode;
    use crate::from_slice;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;
    #[test]
    fn pretty_print_serde() {
        #[derive(Clone, Debug, PartialEq, serde::Serialize)]
        enum Segment {
            Id { mid: usize },
        }

        assert_eq!(
            "{\n  \"Id\": {\n    \"mid\": 0\n  }\n}",
            crate::to_string_pretty(&Segment::Id { mid: 0 }).expect("to_string_pretty")
        );
    }

    #[test]
    fn numerical_map_serde() {
        use std::collections::HashMap;

        #[derive(Clone, Debug, PartialEq, serde::Serialize)]
        struct Foo {
            pub bar: HashMap<i32, i32>,
        }

        let mut foo = Foo {
            bar: HashMap::new(),
        };

        foo.bar.insert(1337, 1337);

        assert_eq!(
            r#"{
  "bar": {
    "1337": 1337
  }
}"#,
            crate::to_string_pretty(&foo).expect("to_string_pretty")
        );
    }

    #[cfg(not(feature = "128bit"))]
    #[cfg(not(target_arch = "wasm32"))]
    fn arb_json_value() -> BoxedStrategy<Value> {
        let leaf = prop_oneof![
            Just(Value::Static(StaticNode::Null)),
            any::<bool>().prop_map(Value::from),
            //(-1.0e306f64..1.0e306f64).prop_map(Value::from), // damn you float!
            any::<i8>().prop_map(Value::from),
            any::<i16>().prop_map(Value::from),
            any::<i32>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            any::<u8>().prop_map(Value::from),
            any::<u16>().prop_map(Value::from),
            any::<u32>().prop_map(Value::from),
            any::<u64>().prop_map(Value::from),
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
                    prop::collection::hash_map(".*", inner, 0..10).prop_map(Value::from),
                ]
            },
        )
        .boxed()
    }

    #[cfg(feature = "128bit")]
    #[cfg(not(target_arch = "wasm32"))]
    fn arb_json_value() -> BoxedStrategy<Value> {
        let leaf = prop_oneof![
            Just(Value::Static(StaticNode::Null)),
            any::<bool>().prop_map(Value::from),
            //(-1.0e306f64..1.0e306f64).prop_map(Value::from), // damn you float!
            any::<i8>().prop_map(Value::from),
            any::<i16>().prop_map(Value::from),
            any::<i32>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            any::<u8>().prop_map(Value::from),
            any::<u16>().prop_map(Value::from),
            any::<u32>().prop_map(Value::from),
            any::<u64>().prop_map(Value::from),
            any::<i128>().prop_map(Value::from),
            any::<u128>().prop_map(Value::from),
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
                    prop::collection::hash_map(".*", inner, 0..10).prop_map(Value::from),
                ]
            },
        )
        .boxed()
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #![proptest_config(ProptestConfig {
            // Setting both fork and timeout is redundant since timeout implies
            // fork, but both are shown for clarity.
            // Disabled for code coverage, enable to track bugs
            // fork: true,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_json_encode_decode(val in arb_json_value()) {
            let mut encoded = crate::to_vec_pretty(&val).expect("to_vec_pretty");
            println!("{}", String::from_utf8_lossy(&encoded.clone()));
            let res: Value = crate::from_slice(encoded.as_mut_slice()).expect("can't convert");
            assert_eq!(val, res);
        }
        #[test]
        fn prop_serd_compat(val in arb_json_value()) {
            let simd = crate::to_string_pretty(&val).expect("to_string_pretty");
            let serde = serde_json::to_string_pretty(&val).expect("to_string_pretty");
            assert_eq!(simd, serde);
        }
    }

    #[test]
    fn prettyfy() {
        let v = crate::json!({"key1":{}, "key2":[], "key3":[1,{"key4":null}]});
        let mut res = match crate::to_vec_pretty(&v) {
            Ok(res) => res,
            Err(e) => {
                println!("prettify: {e}");
                assert_eq!(v, "snot");
                vec![]
            }
        };
        let s = unsafe { String::from_utf8_unchecked(res.clone()) };
        println!("{s}");
        let v2: Value = from_slice(&mut res).expect("generated bad json");
        assert_eq!(v, v2);
    }
}
