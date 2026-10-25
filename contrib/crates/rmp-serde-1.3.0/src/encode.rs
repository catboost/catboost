//! Serialize a Rust data structure into MessagePack data.

use crate::bytes::OnlyBytes;
use crate::config::BytesMode;
use std::error;
use std::fmt::{self, Display};
use std::io::Write;
use std::marker::PhantomData;

use serde;
use serde::ser::{
    SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant,
};
use serde::Serialize;

use rmp::encode::ValueWriteError;
use rmp::{encode, Marker};

use crate::config::{
    BinaryConfig, DefaultConfig, HumanReadableConfig, RuntimeConfig, SerializerConfig, StructMapConfig, StructTupleConfig
};
use crate::MSGPACK_EXT_STRUCT_NAME;

/// This type represents all possible errors that can occur when serializing or
/// deserializing MessagePack data.
#[derive(Debug)]
pub enum Error {
    /// Failed to write a MessagePack value.
    InvalidValueWrite(ValueWriteError),
    //TODO: This can be removed at some point
    /// Failed to serialize struct, sequence or map, because its length is unknown.
    UnknownLength,
    /// Invalid Data model, i.e. Serialize trait is not implmented correctly
    InvalidDataModel(&'static str),
    /// Depth limit exceeded
    DepthLimitExceeded,
    /// Catchall for syntax error messages.
    Syntax(String),
}

impl error::Error for Error {
    #[cold]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::InvalidValueWrite(ref err) => Some(err),
            Error::UnknownLength => None,
            Error::InvalidDataModel(_) => None,
            Error::DepthLimitExceeded => None,
            Error::Syntax(..) => None,
        }
    }
}

impl Display for Error {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Error::InvalidValueWrite(ref err) => write!(f, "invalid value write: {err}"),
            Error::UnknownLength => {
                f.write_str("attempt to serialize struct, sequence or map with unknown length")
            }
            Error::InvalidDataModel(r) => write!(f, "serialize data model is invalid: {r}"),
            Error::DepthLimitExceeded => f.write_str("depth limit exceeded"),
            Error::Syntax(ref msg) => f.write_str(msg),
        }
    }
}

impl From<ValueWriteError> for Error {
    #[cold]
    fn from(err: ValueWriteError) -> Error {
        Error::InvalidValueWrite(err)
    }
}

impl serde::ser::Error for Error {
    /// Raised when there is general error when deserializing a type.
    #[cold]
    fn custom<T: Display>(msg: T) -> Error {
        Error::Syntax(msg.to_string())
    }
}

/// Obtain the underlying writer.
pub trait UnderlyingWrite {
    /// Underlying writer type.
    type Write: Write;

    /// Gets a reference to the underlying writer.
    fn get_ref(&self) -> &Self::Write;

    /// Gets a mutable reference to the underlying writer.
    ///
    /// It is inadvisable to directly write to the underlying writer.
    fn get_mut(&mut self) -> &mut Self::Write;

    /// Unwraps this `Serializer`, returning the underlying writer.
    fn into_inner(self) -> Self::Write;
}

/// Represents MessagePack serialization implementation.
///
/// # Note
///
/// MessagePack has no specification about how to encode enum types. Thus we are free to do
/// whatever we want, so the given choice may be not ideal for you.
///
/// An enum value is represented as a single-entry map whose key is the variant
/// id and whose value is a sequence containing all associated data. If the enum
/// does not have associated data, the sequence is empty.
///
/// All instances of `ErrorKind::Interrupted` are handled by this function and the underlying
/// operation is retried.
// TODO: Docs. Examples.
#[derive(Debug)]
pub struct Serializer<W, C = DefaultConfig> {
    wr: W,
    depth: u16,
    config: RuntimeConfig,
    _back_compat_config: PhantomData<C>,
}

impl<W: Write, C> Serializer<W, C> {
    /// Gets a reference to the underlying writer.
    #[inline(always)]
    pub fn get_ref(&self) -> &W {
        &self.wr
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// It is inadvisable to directly write to the underlying writer.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.wr
    }

    /// Unwraps this `Serializer`, returning the underlying writer.
    #[inline(always)]
    pub fn into_inner(self) -> W {
        self.wr
    }

    /// Changes the maximum nesting depth that is allowed.
    ///
    /// Currently unused.
    #[doc(hidden)]
    #[inline]
    pub fn unstable_set_max_depth(&mut self, depth: usize) {
        self.depth = depth.min(u16::MAX as _) as u16;
    }
}

impl<W: Write> Serializer<W, DefaultConfig> {
    /// Constructs a new `MessagePack` serializer whose output will be written to the writer
    /// specified.
    ///
    /// # Note
    ///
    /// This is the default constructor, which returns a serializer that will serialize structs
    /// and enums using the most compact representation.
    #[inline]
    pub fn new(wr: W) -> Self {
        Serializer {
            wr,
            depth: 1024,
            config: RuntimeConfig::new(DefaultConfig),
            _back_compat_config: PhantomData,
        }
    }
}

impl<'a, W: Write + 'a, C> Serializer<W, C> {
    #[inline]
    fn compound(&'a mut self) -> Result<Compound<'a, W, C>, Error> {
        Ok(Compound { se: self })
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> Serializer<W, C> {
    #[inline]
    fn maybe_unknown_len_compound<F>(&'a mut self, len: Option<u32>, f: F) -> Result<MaybeUnknownLengthCompound<'a, W, C>, Error>
    where F: Fn(&mut W, u32) -> Result<Marker, ValueWriteError>
    {
        Ok(MaybeUnknownLengthCompound {
            compound: match len {
                Some(len) => {
                    f(&mut self.wr, len)?;
                    None
                }
                None => Some(UnknownLengthCompound::from(&*self)),
            },
            se: self,
        })
    }
}

impl<W: Write, C> Serializer<W, C> {
    /// Consumes this serializer returning the new one, which will serialize structs as a map.
    ///
    /// This is used, when the default struct serialization as a tuple does not fit your
    /// requirements.
    #[inline]
    pub fn with_struct_map(self) -> Serializer<W, StructMapConfig<C>> {
        let Serializer { wr, depth, config, _back_compat_config: _ } = self;
        Serializer {
            wr,
            depth,
            config: RuntimeConfig::new(StructMapConfig::new(config)),
            _back_compat_config: PhantomData,
        }
    }

    /// Consumes this serializer returning the new one, which will serialize structs as a tuple
    /// without field names.
    ///
    /// This is the default MessagePack serialization mechanism, emitting the most compact
    /// representation.
    #[inline]
    pub fn with_struct_tuple(self) -> Serializer<W, StructTupleConfig<C>> {
        let Serializer { wr, depth, config, _back_compat_config: _ } = self;
        Serializer {
            wr,
            depth,
            config: RuntimeConfig::new(StructTupleConfig::new(config)),
            _back_compat_config: PhantomData,
        }
    }

    /// Consumes this serializer returning the new one, which will serialize some types in
    /// human-readable representations (`Serializer::is_human_readable` will return `true`). Note
    /// that the overall representation is still binary, but some types such as IP addresses will
    /// be saved as human-readable strings.
    ///
    /// This is primarily useful if you need to interoperate with serializations produced by older
    /// versions of `rmp-serde`.
    #[inline]
    pub fn with_human_readable(self) -> Serializer<W, HumanReadableConfig<C>> {
        let Serializer { wr, depth, config, _back_compat_config: _ } = self;
        Serializer {
            wr,
            depth,
            config: RuntimeConfig::new(HumanReadableConfig::new(config)),
            _back_compat_config: PhantomData,
        }
    }

    /// Consumes this serializer returning the new one, which will serialize types as binary
    /// (`Serializer::is_human_readable` will return `false`).
    ///
    /// This is the default MessagePack serialization mechanism, emitting the most compact
    /// representation.
    #[inline]
    pub fn with_binary(self) -> Serializer<W, BinaryConfig<C>> {
        let Serializer { wr, depth, config, _back_compat_config: _ } = self;
        Serializer {
            wr,
            depth,
            config: RuntimeConfig::new(BinaryConfig::new(config)),
            _back_compat_config: PhantomData,
        }
    }

    /// Prefer encoding sequences of `u8` as bytes, rather than
    /// as a sequence of variable-size integers.
    ///
    /// This reduces overhead of binary data, but it may break
    /// decodnig of some Serde types that happen to contain `[u8]`s,
    /// but don't implement Serde's `visit_bytes`.
    ///
    /// ```rust
    /// use serde::ser::Serialize;
    /// let mut msgpack_data = Vec::new();
    /// let mut serializer = rmp_serde::Serializer::new(&mut msgpack_data)
    ///     .with_bytes(rmp_serde::config::BytesMode::ForceAll);
    /// vec![255u8; 100].serialize(&mut serializer).unwrap();
    /// ```
    #[inline]
    pub fn with_bytes(mut self, mode: BytesMode) -> Serializer<W, C> {
        self.config.bytes = mode;
        self
    }
}

impl<W: Write, C> UnderlyingWrite for Serializer<W, C> {
    type Write = W;

    #[inline(always)]
    fn get_ref(&self) -> &Self::Write {
        &self.wr
    }

    #[inline(always)]
    fn get_mut(&mut self) -> &mut Self::Write {
        &mut self.wr
    }

    #[inline(always)]
    fn into_inner(self) -> Self::Write {
        self.wr
    }
}

/// Hack to store fixed-size arrays (which serde says are tuples)
#[derive(Debug)]
#[doc(hidden)]
pub struct Tuple<'a, W, C> {
    len: u32,
    // can't know if all elements are u8 until the end ;(
    buf: Option<Vec<u8>>,
    se: &'a mut Serializer<W, C>,
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeTuple for Tuple<'a, W, C> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        if let Some(buf) = &mut self.buf {
            if let Ok(byte) = value.serialize(OnlyBytes) {
                buf.push(byte);
                return Ok(());
            } else {
                encode::write_array_len(&mut self.se.wr, self.len)?;
                for b in buf {
                    b.serialize(&mut *self.se)?;
                }
                self.buf = None;
            }
        }
        value.serialize(&mut *self.se)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if let Some(buf) = self.buf {
            if self.len < 16 && buf.iter().all(|&b| b < 128) {
                encode::write_array_len(&mut self.se.wr, self.len)?;
            } else {
                encode::write_bin_len(&mut self.se.wr, self.len)?;
            }
            self.se.wr.write_all(&buf)
                .map_err(ValueWriteError::InvalidDataWrite)?;
        }
        Ok(())
    }
}

/// Part of serde serialization API.
#[derive(Debug)]
#[doc(hidden)]
pub struct Compound<'a, W, C> {
    se: &'a mut Serializer<W, C>,
}

#[derive(Debug)]
#[allow(missing_docs)]
pub struct ExtFieldSerializer<'a, W> {
    wr: &'a mut W,
    tag: Option<i8>,
    finish: bool,
}

/// Represents MessagePack serialization implementation for Ext.
#[derive(Debug)]
pub struct ExtSerializer<'a, W> {
    fields_se: ExtFieldSerializer<'a, W>,
    tuple_received: bool,
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeSeq for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeTuple for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeTupleStruct for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeStruct for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) ->
        Result<(), Self::Error>
    {
        if self.se.config.is_named {
            encode::write_str(self.se.get_mut(), key)?;
        }
        value.serialize(&mut *self.se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeTupleVariant for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeStructVariant for Compound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) ->
        Result<(), Self::Error>
    {
        if self.se.config.is_named {
            encode::write_str(self.se.get_mut(), key)?;
            value.serialize(&mut *self.se)
        } else {
            value.serialize(&mut *self.se)
        }
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

/// Contains a `Serializer` for sequences and maps whose length is not yet known
/// and a counter for the number of elements that are encoded by the `Serializer`.
#[derive(Debug)]
struct UnknownLengthCompound {
    se: Serializer<Vec<u8>, DefaultConfig>,
    elem_count: u32,
}

impl<W, C: SerializerConfig> From<&Serializer<W, C>> for UnknownLengthCompound {
    fn from(se: &Serializer<W, C>) -> Self {
        Self {
            se: Serializer {
                wr: Vec::with_capacity(128),
                config: RuntimeConfig::new(se.config),
                depth: se.depth,
                _back_compat_config: PhantomData,
            },
            elem_count: 0
        }
    }
}

/// Contains a `Serializer` for encoding elements of sequences and maps.
///
/// # Note
///
/// If , for example, a field inside a struct is tagged with `#serde(flatten)` the total number of
/// fields of this struct will be unknown to serde because flattened fields may have name clashes
/// and then will be overwritten. So, serde wants to serialize the struct as a map with an unknown
/// length.
///
/// For the described case a `UnknownLengthCompound` is used to encode the elements. On `end()`
/// the counted length and the encoded elements will be written to the `Serializer`. A caveat is,
/// that structs that contain flattened fields arem always written as a map, even when compact
/// representaion is desired.
///
/// Otherwise, if the length is known, the elements will be encoded directly by the `Serializer`.
#[derive(Debug)]
#[doc(hidden)]
pub struct MaybeUnknownLengthCompound<'a, W, C> {
    se: &'a mut Serializer<W, C>,
    compound: Option<UnknownLengthCompound>,
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeSeq for MaybeUnknownLengthCompound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        match self.compound.as_mut() {
            None => value.serialize(&mut *self.se),
            Some(buf) => {
                value.serialize(&mut buf.se)?;
                buf.elem_count += 1;
                Ok(())
            }
        }
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if let Some(compound) = self.compound {
            encode::write_array_len(&mut self.se.wr, compound.elem_count)?;
            self.se.wr.write_all(&compound.se.into_inner())
                .map_err(ValueWriteError::InvalidDataWrite)?;
        }
        Ok(())
    }
}

impl<'a, W: Write + 'a, C: SerializerConfig> SerializeMap for MaybeUnknownLengthCompound<'a, W, C> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::serialize_element(self, key)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::serialize_element(self, value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if let Some(compound) = self.compound {
            encode::write_map_len(&mut self.se.wr, compound.elem_count / 2)?;
            self.se.wr.write_all(&compound.se.into_inner())
                .map_err(ValueWriteError::InvalidDataWrite)?;
        }
        Ok(())
    }
}

impl<'a, W, C> serde::Serializer for &'a mut Serializer<W, C>
where
    W: Write,
    C: SerializerConfig,
{
    type Ok = ();
    type Error = Error;

    type SerializeSeq = MaybeUnknownLengthCompound<'a, W, C>;
    type SerializeTuple = Tuple<'a, W, C>;
    type SerializeTupleStruct = Compound<'a, W, C>;
    type SerializeTupleVariant = Compound<'a, W, C>;
    type SerializeMap = MaybeUnknownLengthCompound<'a, W, C>;
    type SerializeStruct = Compound<'a, W, C>;
    type SerializeStructVariant = Compound<'a, W, C>;

    #[inline]
    fn is_human_readable(&self) -> bool {
        self.config.is_human_readable
    }

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        encode::write_bool(&mut self.wr, v)
            .map_err(|err| Error::InvalidValueWrite(ValueWriteError::InvalidMarkerWrite(err)))
    }

    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        self.serialize_i64(i64::from(v))
    }

    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        encode::write_sint(&mut self.wr, v)?;
        Ok(())
    }

    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        self.serialize_bytes(&v.to_be_bytes())
    }

    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.serialize_u64(u64::from(v))
    }

    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        encode::write_uint(&mut self.wr, v)?;
        Ok(())
    }

    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        self.serialize_bytes(&v.to_be_bytes())
    }

    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        encode::write_f32(&mut self.wr, v)?;
        Ok(())
    }

    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        encode::write_f64(&mut self.wr, v)?;
        Ok(())
    }

    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        // A char encoded as UTF-8 takes 4 bytes at most.
        let mut buf = [0; 4];
        self.serialize_str(v.encode_utf8(&mut buf))
    }

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        encode::write_str(&mut self.wr, v)?;
        Ok(())
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(encode::write_bin(&mut self.wr, value)?)
    }

    fn serialize_none(self) -> Result<(), Self::Error> {
        self.serialize_unit()
    }

    fn serialize_some<T: ?Sized + serde::Serialize>(self, v: &T) -> Result<(), Self::Error> {
        v.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        encode::write_nil(&mut self.wr)
            .map_err(|err| Error::InvalidValueWrite(ValueWriteError::InvalidMarkerWrite(err)))
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        encode::write_array_len(&mut self.wr, 0)?;
        Ok(())
    }

    fn serialize_unit_variant(self, _name: &str, _: u32, variant: &'static str) ->
        Result<Self::Ok, Self::Error>
    {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(self, name: &'static str, value: &T) -> Result<(), Self::Error> {
        if name == MSGPACK_EXT_STRUCT_NAME {
            let mut ext_se = ExtSerializer::new(self);
            value.serialize(&mut ext_se)?;

            return ext_se.end();
        }

        // Encode as if it's inner type.
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + serde::Serialize>(self, _name: &'static str, _: u32, variant: &'static str, value: &T) -> Result<Self::Ok, Self::Error> {
        // encode as a map from variant idx to its attributed data, like: {idx => value}
        encode::write_map_len(&mut self.wr, 1)?;
        self.serialize_str(variant)?;
        value.serialize(self)
    }

    #[inline]
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Error> {
        self.maybe_unknown_len_compound(len.map(|len| len as u32), |wr, len| encode::write_array_len(wr, len))
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(Tuple {
            buf: if self.config.bytes == BytesMode::ForceAll && len > 0 {
                Some(Vec::new())
            } else {
                encode::write_array_len(&mut self.wr, len as u32)?;
                None
            },
            len: len as u32,
            se: self,
        })
    }

    fn serialize_tuple_struct(self, _name: &'static str, len: usize) ->
        Result<Self::SerializeTupleStruct, Self::Error>
    {
        encode::write_array_len(&mut self.wr, len as u32)?;

        self.compound()
    }

    fn serialize_tuple_variant(self, _name: &'static str, _: u32, variant: &'static str, len: usize) ->
        Result<Self::SerializeTupleVariant, Error>
    {
        // encode as a map from variant idx to a sequence of its attributed data, like: {idx => [v1,...,vN]}
        encode::write_map_len(&mut self.wr, 1)?;
        self.serialize_str(variant)?;
        encode::write_array_len(&mut self.wr, len as u32)?;
        self.compound()
    }

    #[inline]
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        self.maybe_unknown_len_compound(len.map(|len| len as u32), |wr, len| encode::write_map_len(wr, len))
    }

    fn serialize_struct(self, _name: &'static str, len: usize) ->
        Result<Self::SerializeStruct, Self::Error>
    {
        if self.config.is_named {
            encode::write_map_len(self.get_mut(), len as u32)?;
        } else {
            encode::write_array_len(self.get_mut(), len as u32)?;
        }
        self.compound()
    }

    fn serialize_struct_variant(self, name: &'static str, _: u32, variant: &'static str, len: usize) ->
        Result<Self::SerializeStructVariant, Error>
    {
        // encode as a map from variant idx to a sequence of its attributed data, like: {idx => [v1,...,vN]}
        encode::write_map_len(&mut self.wr, 1)?;
        self.serialize_str(variant)?;
        self.serialize_struct(name, len)
    }

    fn collect_seq<I>(self, iter: I) -> Result<Self::Ok, Self::Error> where I: IntoIterator, I::Item: Serialize {
        let iter = iter.into_iter();
        let len = match iter.size_hint() {
            (lo, Some(hi)) if lo == hi && lo <= u32::MAX as usize => Some(lo as u32),
            _ => None,
        };

        const MAX_ITER_SIZE: usize = std::mem::size_of::<<&[u8] as IntoIterator>::IntoIter>();
        const ITEM_PTR_SIZE: usize = std::mem::size_of::<&u8>();

        // Estimate whether the input is `&[u8]` or similar (hacky, because Rust lacks proper specialization)
        let might_be_a_bytes_iter = (std::mem::size_of::<I::Item>() == 1 || std::mem::size_of::<I::Item>() == ITEM_PTR_SIZE)
            // Complex types like HashSet<u8> don't support reading bytes.
            // The simplest iterator is ptr+len.
            && std::mem::size_of::<I::IntoIter>() <= MAX_ITER_SIZE;

        let mut iter = iter.peekable();
        if might_be_a_bytes_iter && self.config.bytes != BytesMode::Normal {
            if let Some(len) = len {
                // The `OnlyBytes` serializer emits `Err` for everything except `u8`
                if iter.peek().map_or(false, |item| item.serialize(OnlyBytes).is_ok()) {
                    return self.bytes_from_iter(iter, len);
                }
            }
        }

        let mut serializer = self.serialize_seq(len.map(|len| len as usize))?;
        iter.try_for_each(|item| serializer.serialize_element(&item))?;
        SerializeSeq::end(serializer)
    }
}

impl<W: Write, C: SerializerConfig> Serializer<W, C> {
    fn bytes_from_iter<I>(&mut self, mut iter: I, len: u32) -> Result<(), <&mut Self as serde::Serializer>::Error> where I: Iterator, I::Item: Serialize {
        encode::write_bin_len(&mut self.wr, len)?;
        iter.try_for_each(|item| {
            self.wr.write(std::slice::from_ref(&item.serialize(OnlyBytes)
                .map_err(|_| Error::InvalidDataModel("BytesMode"))?))
                .map_err(ValueWriteError::InvalidDataWrite)?;
             Ok(())
        })
    }
}

impl<'a, W: Write + 'a> serde::Serializer for &mut ExtFieldSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = serde::ser::Impossible<(), Error>;
    type SerializeTuple = serde::ser::Impossible<(), Error>;
    type SerializeTupleStruct = serde::ser::Impossible<(), Error>;
    type SerializeTupleVariant = serde::ser::Impossible<(), Error>;
    type SerializeMap = serde::ser::Impossible<(), Error>;
    type SerializeStruct = serde::ser::Impossible<(), Error>;
    type SerializeStructVariant = serde::ser::Impossible<(), Error>;

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Self::Error> {
        if self.tag.is_none() {
            self.tag.replace(value);
            Ok(())
        } else {
            Err(Error::InvalidDataModel("expected i8 and bytes"))
        }
    }

    #[inline]
    fn serialize_bytes(self, val: &[u8]) -> Result<Self::Ok, Self::Error> {
        if let Some(tag) = self.tag.take() {
            encode::write_ext_meta(self.wr, val.len() as u32, tag)?;
            self.wr
                .write_all(val)
                .map_err(|err| Error::InvalidValueWrite(ValueWriteError::InvalidDataWrite(err)))?;

            self.finish = true;

            Ok(())
        } else {
            Err(Error::InvalidDataModel("expected i8 and bytes"))
        }
    }

    #[inline]
    fn serialize_bool(self, _val: bool) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_i16(self, _val: i16) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_i32(self, _val: i32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_i64(self, _val: i64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_u8(self, _val: u8) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_u16(self, _val: u16) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_u32(self, _val: u32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_u64(self, _val: u64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_f32(self, _val: f32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_f64(self, _val: f64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_char(self, _val: char) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_str(self, _val: &str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_unit_variant(self, _name: &'static str, _idx: u32, _variant: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_newtype_struct<T: ?Sized>(self, _name: &'static str, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    fn serialize_newtype_variant<T: ?Sized>(self, _name: &'static str, _idx: u32, _variant: &'static str, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_some<T: ?Sized>(self, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_tuple_variant(self, _name: &'static str, _idx: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeTupleVariant, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }

    #[inline]
    fn serialize_struct_variant(self, _name: &'static str, _idx: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeStructVariant, Error> {
        Err(Error::InvalidDataModel("expected i8 and bytes"))
    }
}

impl<'a, W: Write + 'a> serde::ser::Serializer for &mut ExtSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = serde::ser::Impossible<(), Error>;
    type SerializeTuple = Self;
    type SerializeTupleStruct = serde::ser::Impossible<(), Error>;
    type SerializeTupleVariant = serde::ser::Impossible<(), Error>;
    type SerializeMap = serde::ser::Impossible<(), Error>;
    type SerializeStruct = serde::ser::Impossible<(), Error>;
    type SerializeStructVariant = serde::ser::Impossible<(), Error>;

    #[inline]
    fn serialize_bytes(self, _val: &[u8]) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_bool(self, _val: bool) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_i8(self, _value: i8) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_i16(self, _val: i16) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_i32(self, _val: i32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_i64(self, _val: i64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_u8(self, _val: u8) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_u16(self, _val: u16) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_u32(self, _val: u32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_u64(self, _val: u64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_f32(self, _val: f32) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_f64(self, _val: f64) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_char(self, _val: char) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_str(self, _val: &str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_unit_variant(self, _name: &'static str, _idx: u32, _variant: &'static str) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_newtype_struct<T: ?Sized>(self, _name: &'static str, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_newtype_variant<T: ?Sized>(self, _name: &'static str, _idx: u32, _variant: &'static str, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_some<T: ?Sized>(self, _value: &T) -> Result<Self::Ok, Self::Error>
        where T: Serialize
    {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Error> {
        // FIXME check len
        self.tuple_received = true;

        Ok(self)
    }

    #[inline]
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct, Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_tuple_variant(self, _name: &'static str, _idx: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeTupleVariant, Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct, Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }

    #[inline]
    fn serialize_struct_variant(self, _name: &'static str, _idx: u32, _variant: &'static str, _len: usize) -> Result<Self::SerializeStructVariant, Error> {
        Err(Error::InvalidDataModel("expected tuple"))
    }
}

impl<'a, W: Write + 'a> SerializeTuple for &mut ExtSerializer<'a, W> {
    type Ok = ();
    type Error = Error;

    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut self.fields_se)
    }

    #[inline(always)]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, W: Write + 'a> ExtSerializer<'a, W> {
    #[inline]
    fn new<C>(ser: &'a mut Serializer<W, C>) -> Self {
        Self {
            fields_se: ExtFieldSerializer::new(ser),
            tuple_received: false,
        }
    }

    #[inline]
    fn end(self) -> Result<(), Error> {
        if !self.tuple_received {
            Err(Error::InvalidDataModel("expected tuple"))
        } else {
            self.fields_se.end()
        }
    }
}

impl<'a, W: Write + 'a> ExtFieldSerializer<'a, W> {
    #[inline]
    fn new<C>(ser: &'a mut Serializer<W, C>) -> Self {
        Self {
            wr: UnderlyingWrite::get_mut(ser),
            tag: None,
            finish: false,
        }
    }

    #[inline]
    fn end(self) -> Result<(), Error> {
        if self.finish {
            Ok(())
        } else {
            Err(Error::InvalidDataModel("expected i8 and bytes"))
        }
    }
}

/// Serialize the given data structure as MessagePack into the I/O stream.
/// This function uses compact representation - structures as arrays
///
/// Serialization can fail if `T`'s implementation of `Serialize` decides to fail.
#[inline]
pub fn write<W, T>(wr: &mut W, val: &T) -> Result<(), Error>
where
    W: Write + ?Sized,
    T: Serialize + ?Sized,
{
    val.serialize(&mut Serializer::new(wr))
}

/// Serialize the given data structure as MessagePack into the I/O stream.
/// This function serializes structures as maps
///
/// Serialization can fail if `T`'s implementation of `Serialize` decides to fail.
pub fn write_named<W, T>(wr: &mut W, val: &T) -> Result<(), Error>
where
    W: Write + ?Sized,
    T: Serialize + ?Sized,
{
    let mut se = Serializer::new(wr);
    // Avoids another monomorphisation of `StructMapConfig`
    se.config = RuntimeConfig::new(StructMapConfig::new(se.config));
    val.serialize(&mut se)
}

/// Serialize the given data structure as a MessagePack byte vector.
/// This method uses compact representation, structs are serialized as arrays
///
/// Serialization can fail if `T`'s implementation of `Serialize` decides to fail.
#[inline]
pub fn to_vec<T>(val: &T) -> Result<Vec<u8>, Error>
where
    T: Serialize + ?Sized,
{
    let mut wr = FallibleWriter(Vec::new());
    write(&mut wr, val)?;
    Ok(wr.0)
}

/// Serializes data structure into byte vector as a map
/// Resulting MessagePack message will contain field names
///
/// # Errors
///
/// Serialization can fail if `T`'s implementation of `Serialize` decides to fail.
#[inline]
pub fn to_vec_named<T>(val: &T) -> Result<Vec<u8>, Error>
where
    T: Serialize + ?Sized,
{
    let mut wr = FallibleWriter(Vec::new());
    write_named(&mut wr, val)?;
    Ok(wr.0)
}

#[repr(transparent)]
struct FallibleWriter(Vec<u8>);

impl Write for FallibleWriter {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.0.try_reserve(buf.len()).map_err(|_| std::io::ErrorKind::OutOfMemory)?;
        self.0.extend_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
