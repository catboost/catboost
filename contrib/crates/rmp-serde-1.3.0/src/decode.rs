//! Generic MessagePack deserialization.

use crate::config::sealed::SerializerConfig as _;
use std::convert::TryInto;
use std::error;
use std::fmt::{self, Display, Formatter};
use std::io::{self, Cursor, ErrorKind, Read};
use std::marker::PhantomData;
use std::num::TryFromIntError;
use std::str::{self, Utf8Error};

use byteorder::{self, ReadBytesExt};

use serde;
use serde::de::value::SeqDeserializer;
use serde::de::{self, Deserialize, DeserializeOwned, DeserializeSeed, Unexpected, Visitor};
use serde::forward_to_deserialize_any;

use rmp;
use rmp::decode::{self, DecodeStringError, MarkerReadError, NumValueReadError, RmpRead, ValueReadError};
use rmp::Marker;

use crate::config::{BinaryConfig, DefaultConfig, HumanReadableConfig, SerializerConfig};
use crate::MSGPACK_EXT_STRUCT_NAME;

/// Enum representing errors that can occur while decoding MessagePack data.
#[derive(Debug)]
pub enum Error {
    /// The enclosed I/O error occurred while trying to read a MessagePack
    /// marker.
    InvalidMarkerRead(io::Error),
    /// The enclosed I/O error occurred while trying to read the encoded
    /// MessagePack data.
    InvalidDataRead(io::Error),
    /// A mismatch occurred between the decoded and expected value types.
    TypeMismatch(Marker),
    /// A numeric cast failed due to an out-of-range error.
    OutOfRange,
    /// A decoded array did not have the enclosed expected length.
    LengthMismatch(u32),
    /// An otherwise uncategorized error occurred. See the enclosed `String` for
    /// details.
    Uncategorized(String),
    /// A general error occurred while deserializing the expected type. See the
    /// enclosed `String` for details.
    Syntax(String),
    /// An encoded string could not be parsed as UTF-8.
    Utf8Error(Utf8Error),
    /// The depth limit was exceeded.
    DepthLimitExceeded,
}

macro_rules! depth_count(
    ( $counter:expr, $expr:expr ) => {
        {
            $counter -= 1;
            if $counter == 0 {
                return Err(Error::DepthLimitExceeded)
            }
            let res = $expr;
            $counter += 1;
            res
        }
    }
);

impl error::Error for Error {
    #[cold]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::TypeMismatch(..) => None,
            Error::InvalidMarkerRead(ref err) => Some(err),
            Error::InvalidDataRead(ref err) => Some(err),
            Error::LengthMismatch(..) => None,
            Error::OutOfRange => None,
            Error::Uncategorized(..) => None,
            Error::Syntax(..) => None,
            Error::Utf8Error(ref err) => Some(err),
            Error::DepthLimitExceeded => None,
        }
    }
}

impl de::Error for Error {
    #[cold]
    fn custom<T: Display>(msg: T) -> Self {
        Error::Syntax(msg.to_string())
    }
}

impl Display for Error {
    #[cold]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Error::InvalidMarkerRead(ref err) => write!(fmt, "IO error while reading marker: {err}"),
            Error::InvalidDataRead(ref err) => write!(fmt, "IO error while reading data: {err}"),
            Error::TypeMismatch(ref actual_marker) => {
                write!(fmt, "wrong msgpack marker {actual_marker:?}")
            }
            Error::OutOfRange => fmt.write_str("numeric cast found out of range"),
            Error::LengthMismatch(expected_length) => write!(
                fmt,
                "array had incorrect length, expected {expected_length}"
            ),
            Error::Uncategorized(ref msg) => write!(fmt, "uncategorized error: {msg}"),
            Error::Syntax(ref msg) => fmt.write_str(msg),
            Error::Utf8Error(ref err) => write!(fmt, "string found to be invalid utf8: {err}"),
            Error::DepthLimitExceeded => fmt.write_str("depth limit exceeded"),
        }
    }
}

impl From<MarkerReadError> for Error {
    #[cold]
    fn from(err: MarkerReadError) -> Error {
        match err {
            MarkerReadError(err) => Error::InvalidMarkerRead(err),
        }
    }
}

impl From<Utf8Error> for Error {
    #[cold]
    fn from(err: Utf8Error) -> Error {
        Error::Utf8Error(err)
    }
}

impl From<ValueReadError> for Error {
    #[cold]
    fn from(err: ValueReadError) -> Error {
        match err {
            ValueReadError::TypeMismatch(marker) => Error::TypeMismatch(marker),
            ValueReadError::InvalidMarkerRead(err) => Error::InvalidMarkerRead(err),
            ValueReadError::InvalidDataRead(err) => Error::InvalidDataRead(err),
        }
    }
}

impl From<NumValueReadError> for Error {
    #[cold]
    fn from(err: NumValueReadError) -> Error {
        match err {
            NumValueReadError::TypeMismatch(marker) => Error::TypeMismatch(marker),
            NumValueReadError::InvalidMarkerRead(err) => Error::InvalidMarkerRead(err),
            NumValueReadError::InvalidDataRead(err) => Error::InvalidDataRead(err),
            NumValueReadError::OutOfRange => Error::OutOfRange,
        }
    }
}

impl<'a> From<DecodeStringError<'a>> for Error {
    #[cold]
    fn from(err: DecodeStringError<'_>) -> Error {
        match err {
            DecodeStringError::InvalidMarkerRead(err) => Error::InvalidMarkerRead(err),
            DecodeStringError::InvalidDataRead(err) => Error::InvalidDataRead(err),
            DecodeStringError::TypeMismatch(marker) => Error::TypeMismatch(marker),
            DecodeStringError::BufferSizeTooSmall(..) => Error::Uncategorized("BufferSizeTooSmall".to_string()),
            DecodeStringError::InvalidUtf8(..) => Error::Uncategorized("InvalidUtf8".to_string()),
        }
    }
}

impl From<TryFromIntError> for Error {
    #[cold]
    fn from(_: TryFromIntError) -> Self {
        Error::OutOfRange
    }
}

/// A Deserializer that reads bytes from a buffer.
///
/// # Note
///
/// All instances of `ErrorKind::Interrupted` are handled by this function and the underlying
/// operation is retried.
#[derive(Debug)]
pub struct Deserializer<R, C = DefaultConfig> {
    rd: R,
    _config: PhantomData<C>,
    is_human_readable: bool,
    marker: Option<Marker>,
    depth: u16,
}

impl<R: Read, C> Deserializer<R, C> {
    #[inline]
    fn take_or_read_marker(&mut self) -> Result<Marker, MarkerReadError> {
        self.marker
            .take()
            .map_or_else(|| rmp::decode::read_marker(&mut self.rd), Ok)
    }

    #[inline]
    fn peek_or_read_marker(&mut self) -> Result<Marker, MarkerReadError> {
        if let Some(m) = self.marker {
            Ok(m)
        } else {
            let m = rmp::decode::read_marker(&mut self.rd)?;
            Ok(self.marker.insert(m).to_owned())
        }
    }
}

impl<R: Read> Deserializer<ReadReader<R>, DefaultConfig> {
    /// Constructs a new `Deserializer` by consuming the given reader.
    #[inline]
    pub fn new(rd: R) -> Self {
        Self {
            rd: ReadReader::new(rd),
            _config: PhantomData,
            is_human_readable: DefaultConfig.is_human_readable(),
            // Cached marker in case of deserializing optional values.
            marker: None,
            depth: 1024,
        }
    }
}

impl<R: Read, C> Deserializer<ReadReader<R>, C> {
    /// Gets a reference to the underlying reader in this decoder.
    #[inline(always)]
    pub fn get_ref(&self) -> &R {
        &self.rd.rd
    }

    /// Gets a mutable reference to the underlying reader in this decoder.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.rd.rd
    }

    /// Consumes this deserializer returning the underlying reader.
    #[inline]
    pub fn into_inner(self) -> R {
        self.rd.rd
    }
}

impl<R: Read, C: SerializerConfig> Deserializer<R, C> {
    /// Consumes this deserializer and returns a new one, which will deserialize types with
    /// human-readable representations (`Deserializer::is_human_readable` will return `true`).
    ///
    /// This is primarily useful if you need to interoperate with serializations produced by older
    /// versions of `rmp-serde`.
    #[inline]
    pub fn with_human_readable(self) -> Deserializer<R, HumanReadableConfig<C>> {
        let Deserializer { rd, _config: _, is_human_readable: _, marker, depth } = self;
        Deserializer {
            rd,
            is_human_readable: true,
            _config: PhantomData,
            marker,
            depth,
        }
    }

    /// Consumes this deserializer and returns a new one, which will deserialize types with
    /// binary representations (`Deserializer::is_human_readable` will return `false`).
    ///
    /// This is the default MessagePack deserialization mechanism, consuming the most compact
    /// representation.
    #[inline]
    pub fn with_binary(self) -> Deserializer<R, BinaryConfig<C>> {
        let Deserializer { rd, _config: _, is_human_readable: _, marker, depth } = self;
        Deserializer {
            rd,
            is_human_readable: false,
            _config: PhantomData,
            marker,
            depth,
        }
    }
}

impl<R: AsRef<[u8]>> Deserializer<ReadReader<Cursor<R>>> {
    /// Returns the current position of this deserializer, i.e. how many bytes were read.
    #[inline(always)]
    pub fn position(&self) -> u64 {
        self.rd.rd.position()
    }
}

impl<'de, R> Deserializer<ReadRefReader<'de, R>>
where
    R: AsRef<[u8]> + ?Sized,
{
    /// Constructs a new `Deserializer` from the given byte slice.
    #[inline(always)]
    pub fn from_read_ref(rd: &'de R) -> Self {
        Deserializer {
            rd: ReadRefReader::new(rd),
            is_human_readable: DefaultConfig.is_human_readable(),
            _config: PhantomData,
            marker: None,
            depth: 1024,
        }
    }

    /// Gets a reference to the underlying reader in this decoder.
    #[inline(always)]
    #[must_use]
    pub fn get_ref(&self) -> &R {
        self.rd.whole_slice
    }
}

impl<'de, R: ReadSlice<'de>, C: SerializerConfig> Deserializer<R, C> {
    /// Changes the maximum nesting depth that is allowed
    #[inline(always)]
    pub fn set_max_depth(&mut self, depth: usize) {
        self.depth = depth.min(u16::MAX as _) as u16;
    }
}

#[inline(never)]
fn read_i128_marker<'de, R: ReadSlice<'de>>(marker: Marker, rd: &mut R) -> Result<i128, Error> {
    Ok(match marker {
        Marker::FixPos(val) => val.into(),
        Marker::FixNeg(val) => val.into(),
        Marker::U8 => rd.read_data_u8()?.into(),
        Marker::U16 => rd.read_data_u16()?.into(),
        Marker::U32 => rd.read_data_u32()?.into(),
        Marker::U64 => rd.read_data_u64()?.into(),
        Marker::I8 => rd.read_data_i8()?.into(),
        Marker::I16 => rd.read_data_i16()?.into(),
        Marker::I32 => rd.read_data_i32()?.into(),
        Marker::I64 => rd.read_data_i64()?.into(),
        Marker::Bin8 => {
            let len = read_u8(&mut *rd)?;
            read_128_buf(rd, len)?
        },
        Marker::FixArray(len) => {
            read_128_buf(rd, len)?
        },
        marker => return Err(Error::TypeMismatch(marker)),
    })
}

fn read_128_buf<'de, R: ReadSlice<'de>>(rd: &mut R, len: u8) -> Result<i128, Error> {
    if len != 16 {
        return Err(Error::LengthMismatch(16));
    }
    let buf = match read_bin_data(rd, 16)? {
        Reference::Borrowed(buf) => buf,
        Reference::Copied(buf) => buf,
    };
    Ok(i128::from_be_bytes(buf.try_into().map_err(|_| Error::LengthMismatch(16))?))
}

fn read_str_data<'de, V, R>(rd: &mut R, len: u32, visitor: V) -> Result<V::Value, Error>
    where V: Visitor<'de>, R: ReadSlice<'de>
{
    match read_bin_data(rd, len)? {
        Reference::Borrowed(buf) => {
            match str::from_utf8(buf) {
                Ok(s) => visitor.visit_borrowed_str(s),
                Err(err) => {
                    // Allow to unpack invalid UTF-8 bytes into a byte array.
                    match visitor.visit_borrowed_bytes::<Error>(buf) {
                        Ok(buf) => Ok(buf),
                        Err(..) => Err(Error::Utf8Error(err)),
                    }
                }
            }
        }
        Reference::Copied(buf) => {
            match str::from_utf8(buf) {
                Ok(s) => visitor.visit_str(s),
                Err(err) => {
                    // Allow to unpack invalid UTF-8 bytes into a byte array.
                    match visitor.visit_bytes::<Error>(buf) {
                        Ok(buf) => Ok(buf),
                        Err(..) => Err(Error::Utf8Error(err)),
                    }
                }
            }
        }
    }
}

fn read_bin_data<'a, 'de, R: ReadSlice<'de>>(rd: &'a mut R, len: u32) -> Result<Reference<'de, 'a, [u8]>, Error> {
    rd.read_slice(len as usize).map_err(Error::InvalidDataRead)
}

fn read_u8<R: Read>(rd: &mut R) -> Result<u8, Error> {
    byteorder::ReadBytesExt::read_u8(rd).map_err(Error::InvalidDataRead)
}

fn read_u16<R: Read>(rd: &mut R) -> Result<u16, Error> {
    rd.read_u16::<byteorder::BigEndian>()
        .map_err(Error::InvalidDataRead)
}

fn read_u32<R: Read>(rd: &mut R) -> Result<u32, Error> {
    rd.read_u32::<byteorder::BigEndian>()
        .map_err(Error::InvalidDataRead)
}

fn ext_len<R: Read>(rd: &mut R, marker: Marker) -> Result<u32, Error> {
    Ok(match marker {
        Marker::FixExt1 => 1,
        Marker::FixExt2 => 2,
        Marker::FixExt4 => 4,
        Marker::FixExt8 => 8,
        Marker::FixExt16 => 16,
        Marker::Ext8 => u32::from(read_u8(rd)?),
        Marker::Ext16 => u32::from(read_u16(rd)?),
        Marker::Ext32 => read_u32(rd)?,
        _ => return Err(Error::TypeMismatch(marker)),
    })
}

#[derive(Debug)]
enum ExtDeserializerState {
    New,
    ReadTag,
    ReadBinary,
}

#[derive(Debug)]
struct ExtDeserializer<'a, R, C> {
    rd: &'a mut R,
    _config: PhantomData<C>,
    len: u32,
    state: ExtDeserializerState,
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> ExtDeserializer<'a, R, C> {
    fn new(d: &'a mut Deserializer<R, C>, len: u32) -> Self {
        ExtDeserializer {
            rd: &mut d.rd,
            _config: d._config,
            len,
            state: ExtDeserializerState::New,
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::Deserializer<'de> for ExtDeserializer<'a, R, C> {
    type Error = Error;

    #[inline(always)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        visitor.visit_seq(self)
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        struct identifier tuple enum ignored_any tuple_struct
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::SeqAccess<'de> for ExtDeserializer<'a, R, C> {
    type Error = Error;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.state {
            ExtDeserializerState::New | ExtDeserializerState::ReadTag => Ok(Some(seed.deserialize(self)?)),
            ExtDeserializerState::ReadBinary => Ok(None),
        }
    }
}

/// Deserializer for Ext `SeqAccess`
impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::Deserializer<'de> for &mut ExtDeserializer<'a, R, C> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        match self.state {
            ExtDeserializerState::New => {
                let tag = self.rd.read_data_i8()?;
                self.state = ExtDeserializerState::ReadTag;
                visitor.visit_i8(tag)
            }
            ExtDeserializerState::ReadTag => {
                let data = self.rd.read_slice(self.len as usize).map_err(Error::InvalidDataRead)?;
                self.state = ExtDeserializerState::ReadBinary;
                match data {
                    Reference::Borrowed(bytes) => visitor.visit_borrowed_bytes(bytes),
                    Reference::Copied(bytes) => visitor.visit_bytes(bytes),
                }
            }
            ExtDeserializerState::ReadBinary => {
                debug_assert!(false);
                Err(Error::TypeMismatch(Marker::Reserved))
            },
        }
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        tuple_struct struct identifier tuple enum ignored_any
    }
}

#[inline(never)]
fn any_num<'de, R: ReadSlice<'de>, V: Visitor<'de>>(rd: &mut R, visitor: V, marker: Marker) -> Result<V::Value, Error> {
    match marker {
        Marker::Null => visitor.visit_unit(),
        Marker::True |
        Marker::False => visitor.visit_bool(marker == Marker::True),
        Marker::FixPos(val) => visitor.visit_u8(val),
        Marker::FixNeg(val) => visitor.visit_i8(val),
        Marker::U8 => visitor.visit_u8(rd.read_data_u8()?),
        Marker::U16 => visitor.visit_u16(rd.read_data_u16()?),
        Marker::U32 => visitor.visit_u32(rd.read_data_u32()?),
        Marker::U64 => visitor.visit_u64(rd.read_data_u64()?),
        Marker::I8 => visitor.visit_i8(rd.read_data_i8()?),
        Marker::I16 => visitor.visit_i16(rd.read_data_i16()?),
        Marker::I32 => visitor.visit_i32(rd.read_data_i32()?),
        Marker::I64 => visitor.visit_i64(rd.read_data_i64()?),
        Marker::F32 => visitor.visit_f32(rd.read_data_f32()?),
        Marker::F64 => visitor.visit_f64(rd.read_data_f64()?),
        other_marker => Err(Error::TypeMismatch(other_marker)),
    }
}

impl<'de, R: ReadSlice<'de>, C: SerializerConfig> Deserializer<R, C> {
    fn any_inner<V: Visitor<'de>>(&mut self, visitor: V, allow_bytes: bool) -> Result<V::Value, Error> {
        let marker = self.take_or_read_marker()?;
        match marker {
            Marker::Null |
            Marker::True |
            Marker::False |
            Marker::FixPos(_) |
            Marker::FixNeg(_) |
            Marker::U8 |
            Marker::U16 |
            Marker::U32 |
            Marker::U64 |
            Marker::I8 |
            Marker::I16 |
            Marker::I32 |
            Marker::I64 |
            Marker::F32 |
            Marker::F64 => any_num(&mut self.rd, visitor, marker),
            Marker::FixStr(_) | Marker::Str8 | Marker::Str16 | Marker::Str32 => {
                let len = match marker {
                    Marker::FixStr(len) => Ok(len.into()),
                    Marker::Str8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Str16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Str32 => read_u32(&mut self.rd).map(u32::from),
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                }?;
                read_str_data(&mut self.rd, len, visitor)
            }
            Marker::FixArray(_) |
            Marker::Array16 |
            Marker::Array32 => {
                let len = match marker {
                    Marker::FixArray(len) => len.into(),
                    Marker::Array16 => read_u16(&mut self.rd)?.into(),
                    Marker::Array32 => read_u32(&mut self.rd)?,
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                };

                depth_count!(self.depth, {
                    let mut seq = SeqAccess::new(self, len);
                    let res = visitor.visit_seq(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            }
            Marker::FixMap(_) |
            Marker::Map16 |
            Marker::Map32 => {
                let len = match marker {
                    Marker::FixMap(len) => len.into(),
                    Marker::Map16 => read_u16(&mut self.rd)?.into(),
                    Marker::Map32 => read_u32(&mut self.rd)?,
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                };

                depth_count!(self.depth, {
                    let mut seq = MapAccess::new(self, len);
                    let res = visitor.visit_map(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            }
            Marker::Bin8 | Marker::Bin16 | Marker::Bin32 => {
                let len = match marker {
                    Marker::Bin8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Bin16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Bin32 => read_u32(&mut self.rd).map(u32::from),
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                }?;
                match read_bin_data(&mut self.rd, len)? {
                    Reference::Borrowed(buf) if allow_bytes => visitor.visit_borrowed_bytes(buf),
                    Reference::Copied(buf) if allow_bytes => visitor.visit_bytes(buf),
                    Reference::Borrowed(buf) | Reference::Copied(buf) => {
                        visitor.visit_seq(SeqDeserializer::new(buf.iter().copied()))
                    },
                }
            }
            Marker::FixExt1 |
            Marker::FixExt2 |
            Marker::FixExt4 |
            Marker::FixExt8 |
            Marker::FixExt16 |
            Marker::Ext8 |
            Marker::Ext16 |
            Marker::Ext32 => {
                let len = ext_len(&mut self.rd, marker)?;
                depth_count!(self.depth, visitor.visit_newtype_struct(ExtDeserializer::new(self, len)))
            }
            Marker::Reserved => Err(Error::TypeMismatch(Marker::Reserved)),
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> serde::Deserializer<'de> for &'a mut Deserializer<R, C> {
    type Error = Error;

    #[inline(always)]
    fn is_human_readable(&self) -> bool {
        self.is_human_readable
    }

    #[inline(always)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        self.any_inner(visitor, true)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // # Important
        //
        // If a nested Option `o ∈ { Option<Opion<t>>, Option<Option<Option<t>>>, ..., Option<Option<...Option<t>...> }`
        // is visited for the first time, the marker (read from the underlying Reader) will determine
        // `o`'s innermost type `t`.
        // For subsequent visits of `o` the marker will not be re-read again but kept until type `t`
        // is visited.
        //
        // # Note
        //
        // Round trips of Options where `Option<t> = None` such as `Some(None)` will fail because
        // they are just seriialized as `nil`. The serialization format has probably to be changed
        // to solve this. But as serde_json behaves the same, I think it's not worth doing this.
        let marker = self.take_or_read_marker()?;

        if marker == Marker::Null {
            visitor.visit_none()
        } else {
            // Keep the marker until `o`'s innermost type `t` is visited.
            self.marker = Some(marker);
            visitor.visit_some(self)
        }
    }

    fn deserialize_enum<V>(self, _name: &str, _variants: &[&str], visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        let marker = self.peek_or_read_marker()?;
        match rmp::decode::marker_to_len(&mut self.rd, marker) {
            Ok(len) => match len {
                // Enums are either encoded as maps with a single K/V pair
                // where the K = the variant & V = associated data
                // or as just the variant
                1 => {
                    self.marker = None;
                    visitor.visit_enum(VariantAccess::new(self))
                }
                n => Err(Error::LengthMismatch(n)),
            },
            // TODO: Check this is a string
            Err(_) => visitor.visit_enum(UnitVariantAccess::new(self)),
        }
    }

    fn deserialize_newtype_struct<V>(self, name: &'static str, visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        if name == MSGPACK_EXT_STRUCT_NAME {
            let marker = self.take_or_read_marker()?;

            let len = ext_len(&mut self.rd, marker)?;
            let ext_de = ExtDeserializer::new(self, len);
            return visitor.visit_newtype_struct(ext_de);
        }

        visitor.visit_newtype_struct(self)
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // We need to special case this so that [] is treated as a unit struct when asked for,
        // but as a sequence otherwise. This is because we serialize unit structs as [] rather
        // than as 'nil'.
        match self.take_or_read_marker()? {
            Marker::Null | Marker::FixArray(0) => visitor.visit_unit(),
            marker => {
                self.marker = Some(marker);
                self.deserialize_any(visitor)
            }
        }
    }

    #[inline]
    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i128(read_i128_marker(self.take_or_read_marker()?, &mut self.rd)?)
    }

    #[inline]
    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u128(read_i128_marker(self.take_or_read_marker()?, &mut self.rd)? as u128)
    }

    #[inline]
    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_struct<V>(self, _: &'static str, _: &'static [&'static str], visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_tuple_struct<V>(self, _: &'static str, _: usize, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    forward_to_deserialize_any! {
        bytes byte_buf unit
        map identifier str string char
        ignored_any
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }
}

struct SeqAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
    left: u32,
}

impl<'a, R: 'a, C> SeqAccess<'a, R, C> {
    #[inline]
    fn new(de: &'a mut Deserializer<R, C>, len: u32) -> Self {
        SeqAccess { de, left: len }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::SeqAccess<'de> for SeqAccess<'a, R, C> {
    type Error = Error;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where T: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            Ok(Some(seed.deserialize(&mut *self.de)?))
        } else {
            Ok(None)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct MapAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
    left: u32,
}

impl<'a, R: 'a, C> MapAccess<'a, R, C> {
    #[inline]
    fn new(de: &'a mut Deserializer<R, C>, len: u32) -> Self {
        MapAccess { de, left: len }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::MapAccess<'de> for MapAccess<'a, R, C> {
    type Error = Error;

    #[inline]
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
        where K: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            seed.deserialize(&mut *self.de).map(Some)
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
        where V: DeserializeSeed<'de>
    {
        seed.deserialize(&mut *self.de)
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct UnitVariantAccess<'a, R: 'a, C> {
    de: &'a mut Deserializer<R, C>,
}

impl<'a, R: 'a, C> UnitVariantAccess<'a, R, C> {
    pub fn new(de: &'a mut Deserializer<R, C>) -> Self {
        UnitVariantAccess { de }
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::EnumAccess<'de>
    for UnitVariantAccess<'a, R, C>
{
    type Error = Error;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let variant = seed.deserialize(&mut *self.de)?;
        Ok((variant, self))
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a, C: SerializerConfig> de::VariantAccess<'de>
    for UnitVariantAccess<'a, R, C>
{
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, _seed: T) -> Result<T::Value, Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"struct variant",
        ))
    }
}

struct VariantAccess<'a, R, C> {
    de: &'a mut Deserializer<R, C>,
}

impl<'a, R: 'a, C> VariantAccess<'a, R, C> {
    pub fn new(de: &'a mut Deserializer<R, C>) -> Self {
        VariantAccess { de }
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::EnumAccess<'de> for VariantAccess<'a, R, C> {
    type Error = Error;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Error>
        where V: de::DeserializeSeed<'de>,
    {
        Ok((seed.deserialize(&mut *self.de)?, self))
    }
}

impl<'de, 'a, R: ReadSlice<'de>, C: SerializerConfig> de::VariantAccess<'de> for VariantAccess<'a, R, C> {
    type Error = Error;

    #[inline]
    fn unit_variant(self) -> Result<(), Error> {
        decode::read_nil(&mut self.de.rd)?;
        Ok(())
    }

    #[inline]
    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
        where T: DeserializeSeed<'de>
    {
        seed.deserialize(self.de)
    }

    #[inline]
    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, len, visitor)
    }

    #[inline]
    fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, fields.len(), visitor)
    }
}

/// Unification of both borrowed and non-borrowed reference types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference<'b, 'c, T: ?Sized + 'static> {
    /// The reference is pointed at data that was borrowed.
    Borrowed(&'b T),
    /// The reference is pointed at data that was copied.
    Copied(&'c T),
}

/// Extends the `Read` trait by allowing to read slices directly by borrowing bytes.
///
/// Used to allow zero-copy reading.
pub trait ReadSlice<'de>: Read {
    /// Reads the exact number of bytes from the underlying byte-array.
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, io::Error>;
}

/// Owned reader wrapper.
#[derive(Debug)]
pub struct ReadReader<R: Read> {
    rd: R,
    buf: Vec<u8>,
}

impl<R: Read> ReadReader<R> {
    #[inline]
    fn new(rd: R) -> Self {
        ReadReader {
            rd,
            buf: Vec::with_capacity(128),
        }
    }
}

impl<'de, R: Read> ReadSlice<'de> for ReadReader<R> {
    #[inline]
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, io::Error> {
        self.buf.clear();
        let read = self.rd.by_ref().take(len as u64).read_to_end(&mut self.buf)?;
        if read != len {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }

        Ok(Reference::Copied(&self.buf[..]))
    }
}

impl<R: Read> Read for ReadReader<R> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.rd.read(buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.rd.read_exact(buf)
    }
}

/// Borrowed reader wrapper.
#[derive(Debug)]
pub struct ReadRefReader<'a, R: ?Sized> {
    whole_slice: &'a R,
    buf: &'a [u8],
}

impl<'a, T> ReadRefReader<'a, T> {
    /// Returns the part that hasn't been consumed yet
    #[must_use]
    pub fn remaining_slice(&self) -> &'a [u8] {
        self.buf
    }
}

impl<'a, T: AsRef<[u8]> + ?Sized> ReadRefReader<'a, T> {
    #[inline]
    fn new(rd: &'a T) -> Self {
        Self {
            whole_slice: rd,
            buf: rd.as_ref(),
        }
    }
}

impl<'a, T: AsRef<[u8]> + ?Sized> Read for ReadRefReader<'a, T> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        self.buf.read(buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), io::Error> {
        self.buf.read_exact(buf)
    }
}

impl<'de, T: AsRef<[u8]> + ?Sized> ReadSlice<'de> for ReadRefReader<'de, T> {
    #[inline]
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, io::Error> {
        if len > self.buf.len() {
            return Err(ErrorKind::UnexpectedEof.into());
        }
        let (a, b) = self.buf.split_at(len);
        self.buf = b;
        Ok(Reference::Borrowed(a))
    }
}

#[test]
fn test_as_ref_reader() {
    let buf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut rd = ReadRefReader::new(&buf);

    assert_eq!(rd.read_slice(1).unwrap(), Reference::Borrowed(&[0][..]));
    assert_eq!(rd.read_slice(6).unwrap(), Reference::Borrowed(&[1, 2, 3, 4, 5, 6][..]));
    assert!(rd.read_slice(5).is_err());
    assert_eq!(rd.read_slice(4).unwrap(), Reference::Borrowed(&[7, 8, 9, 10][..]));
}

/// Deserialize an instance of type `T` from an I/O stream of MessagePack.
///
/// # Errors
///
/// This conversion can fail if the structure of the Value does not match the structure expected
/// by `T`. It can also fail if the structure is correct but `T`'s implementation of `Deserialize`
/// decides that something is wrong with the data, for example required struct fields are missing.
#[inline]
pub fn from_read<R, T>(rd: R) -> Result<T, Error>
where R: Read,
      T: DeserializeOwned
{
    Deserialize::deserialize(&mut Deserializer::new(rd))
}

/// Deserialize a temporary scope-bound instance of type `T` from a slice, with zero-copy if possible.
///
/// Deserialization will be performed in zero-copy manner whenever it is possible, borrowing the
/// data from the slice itself. For example, strings and byte-arrays won't copied.
///
/// # Errors
///
/// This conversion can fail if the structure of the Value does not match the structure expected
/// by `T`. It can also fail if the structure is correct but `T`'s implementation of `Deserialize`
/// decides that something is wrong with the data, for example required struct fields are missing.
///
/// # Examples
///
/// ```
/// use serde::Deserialize;
///
/// // Encoded `["Bobby", 8]`.
/// let buf = [0x92, 0xa5, 0x42, 0x6f, 0x62, 0x62, 0x79, 0x8];
///
/// #[derive(Debug, Deserialize, PartialEq)]
/// struct Dog<'a> {
///    name: &'a str,
///    age: u8,
/// }
///
/// assert_eq!(Dog { name: "Bobby", age: 8 }, rmp_serde::from_slice(&buf).unwrap());
/// ```
#[inline(always)]
#[allow(deprecated)]
pub fn from_slice<'a, T>(input: &'a [u8]) -> Result<T, Error>
where
    T: Deserialize<'a>,
{
    from_read_ref(input)
}

#[inline]
#[doc(hidden)]
#[deprecated(note = "use from_slice")]
pub fn from_read_ref<'a, R, T>(rd: &'a R) -> Result<T, Error>
where
    R: AsRef<[u8]> + ?Sized,
    T: Deserialize<'a>,
{
    let mut de = Deserializer::from_read_ref(rd);
    Deserialize::deserialize(&mut de)
}
