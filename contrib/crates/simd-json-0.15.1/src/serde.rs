/// simd-json integrates with serde, this module holds this integration.
/// note that when parsing to a dom you should use the functions in
/// `to_owned_value` or `to_borrowed_value` as they provide much
/// better performance.
///
/// However if have to use serde for other reasons or are parsing
/// directly to structs this is the place to go.
///
mod de;
mod se;
mod value;
pub use self::se::*;
pub use self::value::*;
use crate::{BorrowedValue, OwnedValue};
use crate::{Buffers, Deserializer, Error, ErrorType, Node, Result, macros::stry};
use serde::de::DeserializeOwned;
use serde_ext::Deserialize;
use std::fmt;
use std::io;
use value_trait::prelude::*;
type ConvertResult<T> = std::result::Result<T, SerdeConversionError>;

/// Error while converting from or to serde values
#[derive(Debug)]
pub enum SerdeConversionError {
    /// Serde can not reflect NAN or Infinity
    NanOrInfinity,
    /// The number is out of the 64 bit bound
    NumberOutOfBounds,
    /// Something horrible went wrong, please open a ticket at <https://simd-json.rs>
    Oops,
}
impl std::fmt::Display for SerdeConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SerdeConversionError::{NanOrInfinity, NumberOutOfBounds, Oops};
        match self {
            NanOrInfinity => write!(f, "JSON can not represent NAN or Infinity values"),
            NumberOutOfBounds => write!(f, "Serde can not represent 128 bit values"),
            Oops => write!(
                f,
                "Unreachable code is reachable, oops - please open a bug with simd-json"
            ),
        }
    }
}

impl std::error::Error for SerdeConversionError {}

/// parses a byte slice using a serde deserializer.
/// note that the slice will be rewritten in the process.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn from_slice<'a, T>(s: &'a mut [u8]) -> Result<T>
where
    T: Deserialize<'a>,
{
    let mut deserializer = stry!(Deserializer::from_slice(s));
    T::deserialize(&mut deserializer)
}

/// Parses a byte slice using a serde deserializer.
/// note that the slice will be rewritten in the process.
///
/// Passes in reusable buffers to reduce allocations
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn from_slice_with_buffers<'a, T>(s: &'a mut [u8], buffers: &mut Buffers) -> Result<T>
where
    T: Deserialize<'a>,
{
    let mut deserializer = stry!(Deserializer::from_slice_with_buffers(s, buffers));
    T::deserialize(&mut deserializer)
}

/// Parses a str using a serde deserializer.
/// note that the slice will be rewritten in the process and
/// might not remain a valid utf8 string in its entirety.
///
/// It is recommended to use `from_slice` instead.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
///
/// # Safety
///
/// This function mutates the string passed into it, it's a convenience wrapper around `from_slice`,
/// holding the same guarantees as `str::as_bytes_mut` in that after the call &str might include
/// invalid utf8 bytes.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub unsafe fn from_str<'a, T>(s: &'a mut str) -> Result<T>
where
    T: Deserialize<'a>,
{
    let mut deserializer = stry!(Deserializer::from_slice(unsafe { s.as_bytes_mut() }));

    T::deserialize(&mut deserializer)
}

/// Parses a str using a serde deserializer.
/// note that the slice will be rewritten in the process and
/// might not remain a valid utf8 string in its entirety.
///
/// It is recommended to use `from_slice_with_buffers` instead.
///
/// Passes in reusable buffers to reduce allocations.
///
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
///
/// # Safety
///
/// This function mutates the string passed into it, it's a convinience wrapper around `from_slice`,
/// holding the same guarantees as `str::as_bytes_mut` in that after the call &str might include
/// invalid utf8 bytes.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub unsafe fn from_str_with_buffers<'a, T>(s: &'a mut str, buffers: &mut Buffers) -> Result<T>
where
    T: Deserialize<'a>,
{
    let mut deserializer = stry!(Deserializer::from_slice_with_buffers(
        unsafe { s.as_bytes_mut() },
        buffers
    ));

    T::deserialize(&mut deserializer)
}

/// parses a Reader using a serde deserializer.
///
/// # Warning
///     
/// Since simd-json does not support streaming and requires mutability of the data, this function
/// will read the entire reader into memory before parsing it.
///
/// You might do better using `from_slice` and managing the read buffer yourself.
///
/// # Errors
///
/// Will return `Err` if an IO error is encountered while reading
/// rdr or if the readers content is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn from_reader<R, T>(mut rdr: R) -> Result<T>
where
    R: io::Read,
    T: DeserializeOwned,
{
    let mut data = Vec::new();
    if let Err(e) = rdr.read_to_end(&mut data) {
        return Err(Error::generic(ErrorType::Io(e)));
    }
    let mut deserializer = stry!(Deserializer::from_slice(&mut data));
    T::deserialize(&mut deserializer)
}

/// Parses a Reader using a serde deserializer.
///
/// Passes in reusable buffers to reduce allocations.
///
/// # Errors
///
/// Will return `Err` if an IO error is encountered while reading
/// rdr or if the readers content is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn from_reader_with_buffers<R, T>(mut rdr: R, buffers: &mut Buffers) -> Result<T>
where
    R: io::Read,
    T: DeserializeOwned,
{
    let mut data = Vec::new();
    if let Err(e) = rdr.read_to_end(&mut data) {
        return Err(Error::generic(ErrorType::Io(e)));
    }
    let mut deserializer = stry!(Deserializer::from_slice_with_buffers(&mut data, buffers));
    T::deserialize(&mut deserializer)
}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::generic(ErrorType::Serde(msg.to_string()))
    }
}

impl serde_ext::ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::generic(ErrorType::Serde(msg.to_string()))
    }
}

// Functions purely used by serde
impl<'de> Deserializer<'de> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn next(&mut self) -> Result<Node<'de>> {
        let r = self
            .tape
            .get(self.idx)
            .copied()
            .ok_or_else(|| Self::error(ErrorType::Syntax));
        self.idx += 1;
        r
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn peek(&self) -> Result<Node> {
        self.tape
            .get(self.idx)
            .copied()
            .ok_or_else(|| Self::error(ErrorType::Eof))
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_u8(&mut self) -> Result<u8> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_u8()
                .ok_or_else(|| Self::error(ErrorType::ExpectedUnsigned)),
            _ => Err(Self::error(ErrorType::ExpectedUnsigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_u16(&mut self) -> Result<u16> {
        let next = unsafe { self.next_() };
        match next {
            Node::Static(s) => s
                .as_u16()
                .ok_or_else(|| Self::error(ErrorType::ExpectedUnsigned)),
            _ => Err(Self::error(ErrorType::ExpectedUnsigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_u32(&mut self) -> Result<u32> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_u32()
                .ok_or_else(|| Self::error(ErrorType::ExpectedUnsigned)),
            _ => Err(Self::error(ErrorType::ExpectedUnsigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_u64(&mut self) -> Result<u64> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_u64()
                .ok_or_else(|| Self::error(ErrorType::ExpectedUnsigned)),
            _ => Err(Self::error(ErrorType::ExpectedUnsigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_u128(&mut self) -> Result<u128> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_u128()
                .ok_or_else(|| Self::error(ErrorType::ExpectedUnsigned)),
            _ => Err(Self::error(ErrorType::ExpectedUnsigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_i8(&mut self) -> Result<i8> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_i8()
                .ok_or_else(|| Self::error(ErrorType::ExpectedSigned)),
            _ => Err(Self::error(ErrorType::ExpectedSigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_i16(&mut self) -> Result<i16> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_i16()
                .ok_or_else(|| Self::error(ErrorType::ExpectedSigned)),
            _ => Err(Self::error(ErrorType::ExpectedSigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_i32(&mut self) -> Result<i32> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_i32()
                .ok_or_else(|| Self::error(ErrorType::ExpectedSigned)),
            _ => Err(Self::error(ErrorType::ExpectedSigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_i64(&mut self) -> Result<i64> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_i64()
                .ok_or_else(|| Self::error(ErrorType::ExpectedSigned)),
            _ => Err(Self::error(ErrorType::ExpectedSigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_sign_loss)]
    fn parse_i128(&mut self) -> Result<i128> {
        match unsafe { self.next_() } {
            Node::Static(s) => s
                .as_i128()
                .ok_or_else(|| Self::error(ErrorType::ExpectedSigned)),
            _ => Err(Self::error(ErrorType::ExpectedSigned)),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
    fn parse_double(&mut self) -> Result<f64> {
        match unsafe { self.next_() } {
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Node::Static(StaticNode::F64(n)) => Ok(n.into()),
            Node::Static(StaticNode::I64(n)) => Ok(n as f64),
            Node::Static(StaticNode::U64(n)) => Ok(n as f64),
            _ => Err(Self::error(ErrorType::ExpectedFloat)),
        }
    }
}

impl TryFrom<serde_json::Value> for OwnedValue {
    type Error = SerdeConversionError;
    fn try_from(item: serde_json::Value) -> ConvertResult<Self> {
        use serde_json::Value;
        Ok(match item {
            Value::Null => Self::Static(StaticNode::Null),
            Value::Bool(b) => Self::Static(StaticNode::Bool(b)),
            Value::Number(b) => {
                if let Some(n) = b.as_i64() {
                    Self::Static(StaticNode::I64(n))
                } else if let Some(n) = b.as_u64() {
                    Self::Static(StaticNode::U64(n))
                } else if let Some(n) = b.as_f64() {
                    Self::Static(StaticNode::from(n))
                } else {
                    return Err(SerdeConversionError::Oops);
                }
            }
            Value::String(b) => Self::String(b),
            Value::Array(a) => a
                .into_iter()
                .map(Self::try_from)
                .collect::<ConvertResult<Self>>()?,
            Value::Object(o) => o
                .into_iter()
                .map(|(k, v)| Ok((k, Self::try_from(v)?)))
                .collect::<ConvertResult<Self>>()?,
        })
    }
}

impl TryInto<serde_json::Value> for OwnedValue {
    type Error = SerdeConversionError;
    fn try_into(self) -> ConvertResult<serde_json::Value> {
        use serde_json::Value;
        Ok(match self {
            Self::Static(StaticNode::Null) => Value::Null,
            Self::Static(StaticNode::Bool(b)) => Value::Bool(b),
            Self::Static(StaticNode::I64(n)) => Value::Number(n.into()),
            #[cfg(feature = "128bit")] // FIXME error for too large numbers
            Self::Static(StaticNode::I128(n)) => Value::Number(
                i64::try_from(n)
                    .map_err(|_| SerdeConversionError::NumberOutOfBounds)?
                    .into(),
            ),
            Self::Static(StaticNode::U64(n)) => Value::Number(n.into()),
            #[cfg(feature = "128bit")] // FIXME error for too large numbers
            Self::Static(StaticNode::U128(n)) => Value::Number(
                u64::try_from(n)
                    .map_err(|_| SerdeConversionError::NumberOutOfBounds)?
                    .into(),
            ),
            Self::Static(StaticNode::F64(n)) => {
                #[allow(clippy::useless_conversion)] // .into() required by ordered-float
                if let Some(n) = serde_json::Number::from_f64(n.into()) {
                    Value::Number(n)
                } else {
                    return Err(SerdeConversionError::NanOrInfinity);
                }
            }
            Self::String(b) => Value::String(b),
            Self::Array(a) => Value::Array(
                a.into_iter()
                    .map(TryInto::try_into)
                    .collect::<ConvertResult<Vec<Value>>>()?,
            ),
            Self::Object(o) => Value::Object(
                o.into_iter()
                    .map(|(k, v)| Ok((k, v.try_into()?)))
                    .collect::<ConvertResult<serde_json::map::Map<String, Value>>>()?,
            ),
        })
    }
}

impl TryFrom<serde_json::Value> for BorrowedValue<'_> {
    type Error = SerdeConversionError;
    fn try_from(item: serde_json::Value) -> ConvertResult<Self> {
        use serde_json::Value;
        match item {
            Value::Null => Ok(BorrowedValue::from(())),
            Value::Bool(b) => Ok(BorrowedValue::from(b)),
            Value::Number(b) => match (b.as_i64(), b.as_u64(), b.as_f64()) {
                (Some(n), _, _) => Ok(Self::from(n)),
                (_, Some(n), _) => Ok(Self::from(n)),
                (_, _, Some(n)) => Ok(Self::from(n)),
                _ => Err(SerdeConversionError::Oops),
            },
            Value::String(b) => Ok(Self::String(b.into())),
            Value::Array(a) => a.into_iter().map(Self::try_from).collect(),
            Value::Object(o) => o
                .into_iter()
                .map(|(k, v)| Ok((k, Self::try_from(v)?)))
                .collect(),
        }
    }
}

impl TryInto<serde_json::Value> for BorrowedValue<'_> {
    type Error = SerdeConversionError;
    fn try_into(self) -> ConvertResult<serde_json::Value> {
        use serde_json::Value;
        Ok(match self {
            BorrowedValue::Static(StaticNode::Null) => Value::Null,
            BorrowedValue::Static(StaticNode::Bool(b)) => Value::Bool(b),
            BorrowedValue::Static(StaticNode::I64(n)) => Value::Number(n.into()),
            #[cfg(feature = "128bit")] // FIXME error for too large numbers
            BorrowedValue::Static(StaticNode::I128(n)) => Value::Number(
                i64::try_from(n)
                    .map_err(|_| SerdeConversionError::NumberOutOfBounds)?
                    .into(),
            ),
            BorrowedValue::Static(StaticNode::U64(n)) => Value::Number(n.into()),
            #[cfg(feature = "128bit")] // FIXME error for too large numbers
            BorrowedValue::Static(StaticNode::U128(n)) => Value::Number(
                u64::try_from(n)
                    .map_err(|_| SerdeConversionError::NumberOutOfBounds)?
                    .into(),
            ),
            BorrowedValue::Static(StaticNode::F64(n)) => {
                #[allow(clippy::useless_conversion)] // .into() required by ordered-float
                if let Some(n) = serde_json::Number::from_f64(n.into()) {
                    Value::Number(n)
                } else {
                    return Err(SerdeConversionError::NanOrInfinity);
                }
            }
            BorrowedValue::String(b) => Value::String(b.to_string()),
            BorrowedValue::Array(a) => Value::Array(
                a.into_iter()
                    .map(TryInto::try_into)
                    .collect::<ConvertResult<Vec<Value>>>()?,
            ),
            BorrowedValue::Object(o) => Value::Object(
                o.into_iter()
                    .map(|(k, v)| Ok((k.to_string(), v.try_into()?)))
                    .collect::<ConvertResult<serde_json::map::Map<String, Value>>>()?,
            ),
        })
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::unwrap_used)]
    use crate::{
        BorrowedValue, Deserializer as SimdDeserializer, ErrorType, OwnedValue, error::Error, json,
    };
    use float_cmp::assert_approx_eq;
    use halfbrown::{HashMap, hashmap};
    use serde::{Deserialize, Serialize};
    use serde_json::{Value as SerdeValue, json as sjson, to_string as sto_string};
    use std::collections::BTreeMap;

    #[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct UnitStruct;
    #[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct NewTypeStruct(u8);
    #[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TupleStruct(u8, u8);
    #[derive(Debug, Serialize, Deserialize)]
    struct TestStruct {
        value: String,
    }
    #[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestStruct2 {
        value: u8,
    }
    #[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    enum E {
        NewTypeVariant(u8),
        UnitVariant,
        StructVariant { r: u8, g: u8, b: u8 },
        StructVariant2 { r: u8, g: u8, b: u8 },
        TupleVariant(u8, u8, u8),
    }
    #[derive(Debug, Serialize, Deserialize)]
    struct TestPoint(f64, f64);

    #[test]
    fn convert_owned_value() {
        let v: OwnedValue = json!({
            "int": 42,
            "int2": i64::MAX as u64 + 1,
            "float": 7.2,
            "neg-int": -23,
            "string": "string",
            "bytes": b"bytes",
            "bool": true,
            "null": null,
            "array": [42, 7, -23, false, null, {"key": "value"}],
            "object": {
            "array": [42, 7, -23, false, null, {"key": "value"}],
            },
            "tuple": (122, -14, true, 13_i8, -14_i16, 'c', 22_u8, 23_u16, 24_u32, 25_u64, (), None as Option<i32>, Some(3.25_f32), b"bytes"),
            "struct": TestStruct{value: "value".to_string()},
            "test_struct": TestStruct2{value: 3},
            "point": TestPoint(3., 4.),
            "unit_variant": E::UnitVariant,
            "new_type_variant": E::NewTypeVariant(3),
            "struct_variant": E::StructVariant{r:0, g:0, b:0},
            "tuple_variant": E::TupleVariant(3, 4, 5),
        });

        let s: SerdeValue = sjson!({
            "int": 42,
            "int2": i64::MAX as u64 + 1,
            "float": 7.2,
            "neg-int": -23,
            "string": "string",
            "bytes": b"bytes",
            "bool": true,
            "null": null,
            "array": [42, 7, -23, false, null, {"key": "value"}],
            "object": {
            "array": [42, 7, -23, false, null, {"key": "value"}],
            },
            "tuple": (122, -14, true, 13_i8, -14_i16, 'c', 22_u8, 23_u16, 24_u32, 25_u64, (), None as Option<i32>, Some(3.25_f32), b"bytes"),
            "struct": TestStruct{value: "value".to_string()},
            "test_struct": TestStruct2{value: 3},
            "point": TestPoint(3., 4.),
            "unit_variant": E::UnitVariant,
            "new_type_variant": E::NewTypeVariant(3),
            "struct_variant": E::StructVariant{r:0, g:0, b:0},
            "tuple_variant": E::TupleVariant(3, 4, 5),
        });
        let s_c: SerdeValue = v.clone().try_into().unwrap();
        assert_eq!(s, s_c);
        let v_c: OwnedValue = s.try_into().unwrap();
        assert_eq!(v, v_c);

        let mut v_ser = crate::serde::to_string(&v).unwrap();
        let s_ser = serde_json::to_string(&v).unwrap();
        assert_eq!(s_ser, v_ser);

        let s_deser: OwnedValue = serde_json::from_str(&v_ser).unwrap();
        assert_eq!(v, s_deser);

        let v_deser: OwnedValue = unsafe { crate::serde::from_str(&mut v_ser).unwrap() };
        assert_eq!(v, v_deser);
    }

    #[test]
    fn convert_borrowed_value() {
        let v: BorrowedValue = json!({
            "int": 42,
            "int2": i64::MAX as u64 + 1,
            "float": 7.2,
            "neg-int": -23,
            "string": "string",
            "bool": true,
            "null": null,
            "object": {
            "array": [42, 7, -23, false, null, {"key": "value"}],
            },
            "tuple": (122, -14, true, 13_i8, -14_i16, 'c', 22_u8, 23_u16, 24_u32, 25_u64, (), None as Option<i32>, Some(3.25_f32), b"bytes"),
            "unit_struct": UnitStruct,
            "new_type_struct": NewTypeStruct(3),
            "tuple_struct": TupleStruct(3, 4),
            "struct": TestStruct{value: "value".to_string()},
            "test_struct": TestStruct2{value: 3},
            "point": TestPoint(3., 4.),
            "unit_variant": E::UnitVariant,
            "new_type_variant": E::NewTypeVariant(3),
            "struct_variant": E::StructVariant{r:0, g:0, b:0},
            "tuple_variant": E::TupleVariant(3, 4, 5),
        })
        .into();

        let s: SerdeValue = sjson!({
            "int": 42,
            "int2": i64::MAX as u64 + 1,
            "float": 7.2,
            "neg-int": -23,
            "string": "string",
            "bool": true,
            "null": null,
            "object": {
            "array": [42, 7, -23, false, null, {"key": "value"}],
            },
            "tuple": (122, -14, true, 13_i8, -14_i16, 'c', 22_u8, 23_u16, 24_u32, 25_u64, (), None as Option<i32>, Some(3.25_f32), b"bytes"),
            "unit_struct": UnitStruct,
            "new_type_struct": NewTypeStruct(3),
            "tuple_struct": TupleStruct(3, 4),
            "struct": TestStruct{value: "value".to_string()},
            "test_struct": TestStruct2{value: 3},
            "point": TestPoint(3., 4.),
            "unit_variant": E::UnitVariant,
            "new_type_variant": E::NewTypeVariant(3),
            "struct_variant": E::StructVariant{r:0, g:0, b:0},
            "tuple_variant": E::TupleVariant(3, 4, 5),
        });
        let s_c: SerdeValue = v.clone().try_into().unwrap();
        assert_eq!(s, s_c);
        let v_c: BorrowedValue = s.try_into().unwrap();
        assert_eq!(v, v_c);
    }

    #[test]
    fn option_field_absent() {
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct Person {
            pub name: String,
            pub middle_name: Option<String>,
            pub friends: Vec<String>,
        }
        let mut raw_json = r#"{"name":"bob","friends":[]}"#.to_string();
        let result: Result<Person, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(
            result,
            Ok(Person {
                name: "bob".to_string(),
                middle_name: None,
                friends: vec![],
            })
        );
    }
    #[test]
    fn option_field_present() {
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        pub struct Person {
            pub name: String,
            pub middle_name: Option<String>,
            pub friends: Vec<String>,
        }
        let mut raw_json = r#"{"name":"bob","middle_name": "frank", "friends":[]}"#.to_string();
        let result: Result<Person, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(
            result,
            Ok(Person {
                name: "bob".to_string(),
                middle_name: Some("frank".to_string()),
                friends: vec![],
            })
        );
    }

    #[test]
    fn convert_enum() {
        #[allow(dead_code)]
        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        #[serde(tag = "type")]
        enum Message {
            Request { id: usize, method: String },
            Response { id: String, result: String },
        }

        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        #[serde(tag = "type", content = "v")]
        pub enum Color {
            Red(String), // TODO: If `content` flag is present, `Red` works and `Green` doesn't
            Green { v: bool },
            Blue,
        }

        #[derive(serde::Deserialize, Debug, PartialEq, Eq)]
        #[serde(tag = "type")]
        pub enum Color1 {
            Red(String),
            Green { v: bool }, // TODO: If `content` flag is absent, `Green` works and `Red` doesn't
            Blue,
        }

        let mut raw_json = r#"{"type": "Request", "id": 1, "method": "..."}"#.to_string();
        let result: Result<Message, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(
            result,
            Ok(Message::Request {
                id: 1,
                method: "...".to_string()
            })
        );

        let mut raw_json = r#"{"type": "Response", "id": "1", "result": "..."}"#.to_string();
        let result: Result<Message, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(
            result,
            Ok(Message::Response {
                id: "1".to_string(),
                result: "...".to_string()
            })
        );

        let mut raw_json = r#"{"type": "Red", "v": "1"}"#.to_string();
        let result: Result<Color, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(result, Ok(Color::Red("1".to_string())));

        let mut raw_json = r#"{"type": "Blue"}"#.to_string();
        let result: Result<Color, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(result, Ok(Color::Blue));

        let mut raw_json = r#"{"type": "Green", "v": false}"#.to_string();
        let result: Result<Color1, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(result, Ok(Color1::Green { v: false }));

        let mut raw_json = r#"{"type": "Blue"}"#.to_string();
        let result: Result<Color1, _> = super::from_slice(unsafe { raw_json.as_bytes_mut() });
        assert_eq!(result, Ok(Color1::Blue));
    }

    #[derive(serde_ext::Deserialize)]
    pub struct Foo {
        #[allow(unused)]
        bar: Bar,
    }

    #[derive(serde_ext::Deserialize)]
    pub enum Bar {
        A,
    }

    #[test]
    fn object_simd_json() {
        let mut json = br#"{"bar":"A"}"#.to_vec();

        crate::from_slice::<Foo>(&mut json).unwrap();
    }

    #[test]
    fn simple_simd_json() {
        let mut json = br#""A""#.to_vec();

        assert!(crate::from_slice::<Bar>(&mut json).is_ok());
    }

    #[test]
    fn array_as_struct() {
        #[derive(serde_ext::Deserialize)]
        struct Point {
            x: u64,
            y: u64,
        }

        let mut json = b"[1,2]".to_vec();

        let p: Point = serde_json::from_slice(&json).unwrap();
        assert_eq!(p.x, 1);
        assert_eq!(p.y, 2);

        let p: Point = crate::from_slice(&mut json).unwrap();
        assert_eq!(p.x, 1);
        assert_eq!(p.y, 2);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn floats() {
        #[derive(serde_ext::Deserialize)]
        struct Point {
            x: f64,
            y: f64,
        }

        let mut json = br#"{"x":1.0,"y":2.0}"#.to_vec();

        let p: Point = crate::from_slice(&mut json).unwrap();
        assert_approx_eq!(f64, p.x, 1_f64);
        assert_approx_eq!(f64, p.y, 2_f64);

        let json = json!({"x":-1,"y":i64::MAX as u64 + 1});

        let p: Point = unsafe { crate::from_str(&mut crate::to_string(&json).unwrap()).unwrap() };
        assert_approx_eq!(f64, p.x, -1_f64);
        assert_approx_eq!(f64, p.y, i64::MAX as f64 + 1.0);
    }

    #[test]
    fn vectors() {
        let input: Vec<UnitStruct> = vec![UnitStruct];
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<UnitStruct>>(&mut v_str).unwrap()
        });
        let input: Vec<()> = Vec::new();
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<()>>(&mut v_str).unwrap()
        });
        let input: Vec<Option<u8>> = vec![None, Some(3_u8)];
        let mut v_str = crate::to_string(&input).unwrap();
        dbg!(&v_str);
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<Option<u8>>>(&mut v_str).unwrap()
        });
        let input: Vec<(i32, f32)> = vec![(3, 3.)];
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<(i32, f32)>>(&mut v_str).unwrap()
        });
        let input = vec![vec![3_u8]];
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<Vec<u8>>>(&mut v_str).unwrap()
        });
        let input: Vec<NewTypeStruct> = vec![NewTypeStruct(3_u8)];
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<NewTypeStruct>>(&mut v_str).unwrap()
        });
        let input: Vec<TupleStruct> = Vec::new();
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<TupleStruct>>(&mut v_str).unwrap()
        });
        let input = vec![TupleStruct(3, 3)];
        let mut v_str = crate::to_string(&input).unwrap();
        assert_eq!(input, unsafe {
            crate::from_str::<Vec<TupleStruct>>(&mut v_str).unwrap()
        });
        let input = vec![E::NewTypeVariant(3)];
        let mut _v_str = crate::to_string(&input).unwrap();
        // Enums are not handled yet
        // assert_eq!(input, crate::from_str::<Vec<E>>(&mut v_str).unwrap());
        let input = vec![E::UnitVariant, E::UnitVariant];
        let mut _v_str = crate::to_string(&input).unwrap();
        // Enums are not handled yet
        // assert_eq!(input, crate::from_str::<Vec<E>>(&mut v_str).unwrap());
        let input = vec![
            E::StructVariant { r: 0, g: 0, b: 0 },
            E::StructVariant { r: 0, g: 0, b: 1 },
        ];
        let mut _v_str = crate::to_string(&input).unwrap();
        // Enums are not handled yet
        // assert_eq!(input, crate::from_str::<Vec<E>>(&mut v_str).unwrap());
        let input = vec![E::TupleVariant(0, 0, 0), E::TupleVariant(1, 1, 1)];
        let mut _v_str = crate::to_string(&input).unwrap();
        // Enums are not handled yet
        // assert_eq!(input, crate::from_str::<Vec<E>>(&mut v_str).unwrap());
    }

    macro_rules! parsing_error {
        ($input:expr_2021; $type:ty => $err:ident) => {{
            let mut json_str = $input.to_string();
            assert_eq!(
                unsafe { crate::from_str::<$type>(&mut json_str) },
                Err(SimdDeserializer::error(ErrorType::$err))
            );
        }};
    }

    #[test]
    fn test_parsing_errors() {
        parsing_error!(r#""3""#; i8 => ExpectedSigned);
        parsing_error!(r#""3""#; i16 => ExpectedSigned);
        parsing_error!(r#""3""#; i32 => ExpectedSigned);
        parsing_error!(r#""3""#; i64 => ExpectedSigned);
        parsing_error!(r#""3""#; u8 => ExpectedUnsigned);
        parsing_error!(r#""3""#; u16 => ExpectedUnsigned);
        parsing_error!(r#""3""#; u32 => ExpectedUnsigned);
        parsing_error!(r#""3""#; u64 => ExpectedUnsigned);

        parsing_error!("null"; i8 => ExpectedSigned);
        parsing_error!("null"; i16 => ExpectedSigned);
        parsing_error!("null"; i32 => ExpectedSigned);
        parsing_error!("null"; i64 => ExpectedSigned);
        parsing_error!("-3"; u8 => ExpectedUnsigned);
        parsing_error!("-3"; u16 => ExpectedUnsigned);
        parsing_error!("-3"; u32 => ExpectedUnsigned);
        parsing_error!("-3"; u64 => ExpectedUnsigned);
        parsing_error!("-3"; String => ExpectedString);

        #[cfg(feature = "128bit")]
        {
            parsing_error!(r#""3""#; i128 => ExpectedSigned);
            parsing_error!(r#""3""#; u128 => ExpectedUnsigned);
            parsing_error!("null"; i128 => ExpectedSigned);
            parsing_error!("-3"; u128 => ExpectedUnsigned);
        }

        parsing_error!("null"; f64 => ExpectedFloat);
    }

    macro_rules! ser_deser_map {
        ($key:expr_2021 => $value:expr_2021, $type:ty) => {
            let input = hashmap! {$key => $value};
            let mut m_str = crate::to_string(&input).unwrap();
            assert_eq!(m_str, sto_string(&input).unwrap());
            assert_eq!(input, unsafe {
                crate::from_str::<$type>(&mut m_str).unwrap()
            });
        };
    }

    #[test]
    fn maps() {
        let key_error = Err(Error::generic(ErrorType::KeyMustBeAString));
        assert_eq!(crate::to_string(&hashmap! {b"1234" => 3_i8}), key_error);
        assert_eq!(crate::to_string(&hashmap! {true => 3_i8}), key_error);
        assert_eq!(
            crate::to_string(&hashmap! {[3_u8, 4_u8] => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {None as Option<u8> => 3_i8}),
            key_error
        );
        assert_eq!(crate::to_string(&hashmap! {Some(3_u8) => 3_i8}), key_error);
        assert_eq!(crate::to_string(&hashmap! {() => 3_i8}), key_error);
        assert_eq!(crate::to_string(&hashmap! {(3, 4) => 3_i8}), key_error);
        assert_eq!(crate::to_string(&hashmap! {[3, 4] => 3_i8}), key_error);
        assert_eq!(crate::to_string(&hashmap! {UnitStruct => 3_i8}), key_error);
        assert_eq!(
            crate::to_string(&hashmap! {TupleStruct(3, 3) => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {TestStruct2{value:3} => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {E::NewTypeVariant(0) => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {E::StructVariant{r:0, g:0, b:0} => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {E::StructVariant2{r:0, g:0, b:0} => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {E::TupleVariant(0, 0, 0) => 3_i8}),
            key_error
        );
        assert_eq!(
            crate::to_string(&hashmap! {vec![0, 0, 0] => 3_i8}),
            key_error
        );
        let mut m = BTreeMap::new();
        m.insert("value", 3_u8);
        assert_eq!(crate::to_string(&hashmap! {m => 3_i8}), key_error);

        // f32 and f64 do not implement std::cmp:Eq nor Hash traits
        // assert_eq!(crate::to_string(&hashmap! {3f32 => 3i8}), key_error);
        // assert_eq!(crate::to_string(&hashmap! {3f64 => 3i8}), key_error);

        let mut input = std::collections::HashMap::new();
        input.insert(128_u8, "3");
        let mut input_str = crate::to_string(&input).unwrap();
        assert_eq!(input_str, sto_string(&input).unwrap());
        assert_eq!(
            unsafe { crate::from_str::<std::collections::HashMap<u8, i8>>(&mut input_str) },
            Err(Error::new(0, None, ErrorType::ExpectedSigned))
        );
        assert_eq!(
            unsafe { crate::from_str::<std::collections::HashMap<i8, String>>(&mut input_str) },
            Err(Error::new(0, None, ErrorType::InvalidNumber))
        );
        assert_eq!(
            unsafe { crate::from_str::<HashMap<Option<u8>, String>>(&mut input_str) },
            Ok(hashmap! {Some(128_u8) => "3".to_string()})
        );

        ser_deser_map!('c' => 3_i8, HashMap<char, i8>);
        ser_deser_map!(3_i8 => 3_i8, HashMap<i8, i8>);
        ser_deser_map!(3_i16 => 3_i8, HashMap<i16, i8>);
        ser_deser_map!(3_i32 => 3_i8, HashMap<i32, i8>);
        ser_deser_map!(3_i64 => 3_i8, HashMap<i64, i8>);
        ser_deser_map!(3_u8 => 3_i8, HashMap<u8, i8>);
        ser_deser_map!(3_u16 => 3_i8, HashMap<u16, i8>);
        ser_deser_map!(3_u32 => 3_i8, HashMap<u32, i8>);
        ser_deser_map!(3_u64 => 3_i8, HashMap<u64, i8>);
        #[cfg(feature = "128bit")]
        {
            ser_deser_map!(3_i128 => 3_i8, HashMap<i128, i8>);
            ser_deser_map!(3_u128 => 3_i8, HashMap<u128, i8>);
        }
        ser_deser_map!(NewTypeStruct(1) => 3_i8, HashMap<NewTypeStruct, i8>);
        ser_deser_map!(E::UnitVariant => 3_i8, HashMap<E, i8>);
    }
}
