//! Lazy value, uses a tape until mutated.
//!
//! If it is mutated it is upgraded to a borrowed value.
//! This allows for cheap parsing and data access while still maintaining mutability.
//!
//! # Example
//!
//! ```rust
//! use simd_json::{prelude::*, value::lazy::Value};
//!
//! let mut json = br#"{"key": "value", "snot": 42}"#.to_vec();
//! let tape = simd_json::to_tape( json.as_mut_slice()).unwrap();
//! let value = tape.as_value();
//! let mut lazy = Value::from_tape(value);
//!
//! assert_eq!(lazy.get("key").unwrap(), "value");
//!
//! assert!(lazy.is_tape());
//! lazy.insert("new", 42);
//! assert!(lazy.is_value());
//! assert_eq!(lazy.get("key").unwrap(), "value");
//! assert_eq!(lazy.get("new").unwrap(), 42);
//! ```

use crate::{borrowed, tape};
use std::borrow::Cow;
use std::fmt;

/// Lazy implemntation of the array trait and associated functionality
pub mod array;
mod cmp;
mod from;
/// Lazy implementation of the object trait and associated functionality
pub mod object;
mod trait_impls;

pub use array::Array;
pub use object::Object;

/// A lazy value, this gets initialized with a tape and as long as only non mutating operations are
/// performed it will stay a tape. If a mutating operation is performed it will upgrade to a borrowed
/// value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<'borrow, 'tape, 'input> {
    /// tape variant
    Tape(tape::Value<'tape, 'input>),
    /// borrowed variant
    Value(Cow<'borrow, borrowed::Value<'input>>),
}

impl Default for Value<'static, 'static, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn default() -> Self {
        Value::Tape(tape::Value::null())
    }
}

impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// turns the lazy value into a borrowed value
    #[must_use]
    pub fn into_value(self) -> borrowed::Value<'input> {
        match self {
            Value::Tape(tape) => {
                super::borrowed::BorrowSliceDeserializer::from_tape(tape.0).parse()
            }
            Value::Value(value) => value.into_owned(),
        }
    }
    /// extends the Value COW is owned
    #[must_use]
    pub fn into_owned<'snot>(self) -> Value<'snot, 'tape, 'input> {
        match self {
            Value::Tape(tape) => Value::Tape(tape),
            Value::Value(Cow::Owned(value)) => Value::Value(Cow::Owned(value)),
            Value::Value(Cow::Borrowed(value)) => Value::Value(Cow::Owned(value.clone())),
        }
    }
    /// returns true when the current representation is a tape
    #[must_use]
    pub fn is_tape(&self) -> bool {
        match self {
            Value::Tape(_) => true,
            Value::Value(_) => false,
        }
    }
    /// returns true when the current representation is a borrowed value
    /// this is the opposite of `is_tape`
    #[must_use]
    pub fn is_value(&self) -> bool {
        !self.is_tape()
    }
    /// Creates a new lazy Value from a tape
    #[must_use]
    pub fn from_tape(tape: tape::Value<'tape, 'input>) -> Self {
        Value::Tape(tape)
    }
    unsafe fn into_tape(self) -> tape::Value<'tape, 'input> {
        match self {
            Value::Tape(tape) => tape,
            Value::Value(_) => unreachable!("we know we are not a value"),
        }
    }

    fn upgrade(&mut self) {
        if let Value::Value(_) = &self {
            return;
        }
        let mut dummy = Value::Tape(tape::Value::null());
        std::mem::swap(self, &mut dummy);
        let tape = unsafe { dummy.into_tape() };

        let value = super::borrowed::BorrowSliceDeserializer::from_tape(tape.0).parse();

        *self = Value::Value(Cow::Owned(value));
    }

    fn as_mut(&mut self) -> &mut borrowed::Value<'input> {
        if self.is_tape() {
            self.upgrade();
        }

        if let Value::Value(value) = self {
            value.to_mut()
        } else {
            unreachable!("can't mut a tape");
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl fmt::Display for Value<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Value::Tape(tape) => write!(f, "{tape:?}"),
            Value::Value(value) => write!(f, "{value}"),
        }
    }
}

// impl<'value> Index<&str> for Value<'value> {
//     type Output = Value<'value>;
//     #[cfg_attr(not(feature = "no-inline"), inline)]
//     #[must_use]
//     fn index(&self, index: &str) -> &Self::Output {
//         self.get(index).expect("index out of bounds")
//     }
// }

// impl<'value> Index<usize> for Value<'value> {
//     type Output = Value<'value>;
//     #[cfg_attr(not(feature = "no-inline"), inline)]
//     #[must_use]
//     fn index(&self, index: usize) -> &Self::Output {
//         self.get_idx(index).expect("index out of bounds")
//     }
// }

// impl<'value> IndexMut<&str> for Value<'value> {
//     #[cfg_attr(not(feature = "no-inline"), inline)]
//     #[must_use]
//     fn index_mut(&mut self, index: &str) -> &mut Self::Output {
//         self.get_mut(index).expect("index out of bounds")
//     }
// }

// impl<'value> IndexMut<usize> for Value<'value> {
//     #[cfg_attr(not(feature = "no-inline"), inline)]
//     #[must_use]
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         self.get_idx_mut(index).expect("index out of bounds")
//     }
// }

#[cfg(test)]
mod test {
    #![allow(clippy::cognitive_complexity)]
    use value_trait::prelude::*;

    use super::Value;

    #[test]
    #[should_panic = "Not supported"]
    #[allow(unused_variables, clippy::no_effect)]
    fn object_index() {
        let v = StaticNode::Null;
        v["test"];
    }

    #[test]
    #[should_panic = "Not supported"]
    fn mut_object_index() {
        let mut v = StaticNode::Null;
        v["test"] = ();
    }

    #[test]
    #[should_panic = "Not supported"]
    #[allow(unused_variables, clippy::no_effect)]
    fn array_index() {
        let v = StaticNode::Null;
        v[0];
    }

    #[test]
    #[should_panic = "Not supported"]
    fn mut_array_index() {
        let mut v = StaticNode::Null;
        v[0] = ();
    }

    #[test]
    fn conversion_str() {
        let v = StaticNode::Null;
        assert!(!v.is_str());
    }
    #[cfg(feature = "128bit")]
    #[test]
    fn conversions_i128() {
        let v = Value::from(i128::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i128::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i64() {
        let v = Value::from(i64::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i64::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(!v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i32() {
        let v = Value::from(i32::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i32::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i16() {
        let v = Value::from(i16::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i16::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_i8() {
        let v = Value::from(i8::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(i8::MIN);
        assert!(v.is_i128());
        assert!(!v.is_u128());
        assert!(v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_i32());
        assert!(!v.is_u32());
        assert!(v.is_i16());
        assert!(!v.is_u16());
        assert!(v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_usize() {
        let v = Value::from(usize::MIN as u64);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_usize());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[cfg(feature = "128bit")]
    #[test]
    fn conversions_u128() {
        let v = Value::from(u128::MIN);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u64() {
        let v = Value::from(u64::MIN);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u32() {
        let v = Value::from(u32::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(!v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(!v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u16() {
        let v = Value::from(u16::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(!v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(!v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_u8() {
        let v = Value::from(u8::MAX);
        assert!(v.is_i128());
        assert!(v.is_u128());
        assert!(v.is_i64());
        assert!(v.is_u64());
        assert!(v.is_i32());
        assert!(v.is_u32());
        assert!(v.is_i16());
        assert!(v.is_u16());
        assert!(!v.is_i8());
        assert!(v.is_u8());
        assert!(!v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_f64() {
        let v = Value::from(f64::MAX);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(f64::MIN);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(!v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(());
        assert!(!v.is_f64_castable());
    }

    #[test]
    fn conversions_f32() {
        let v = Value::from(f32::MAX);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(v.is_f32());
        assert!(v.is_f64_castable());
        let v = Value::from(f32::MIN);
        assert!(!v.is_i64());
        assert!(!v.is_u64());
        assert!(v.is_f64());
        assert!(v.is_f32());
        assert!(v.is_f64_castable());
    }

    #[test]
    fn conversions_bool() {
        let v = Value::from(true);
        assert!(v.is_bool());
        assert_eq!(v.value_type(), ValueType::Bool);
        let v = Value::from(());
        assert!(!v.is_bool());
    }

    #[test]
    fn conversions_float() {
        let v = Value::from(42.0);
        assert!(v.is_f64());
        assert_eq!(v.value_type(), ValueType::F64);
        let v = Value::from(());
        assert!(!v.is_f64());
    }

    #[test]
    fn conversions_int() {
        let v = Value::from(-42);
        assert!(v.is_i64());
        assert_eq!(v.value_type(), ValueType::I64);
        #[cfg(feature = "128bit")]
        {
            let v = Value::from(-42_i128);
            assert!(v.is_i64());
            assert!(v.is_i128());
            assert_eq!(v.value_type(), ValueType::I128);
        }
        let v = Value::from(());
        assert!(!v.is_i64());
        assert!(!v.is_i128());
    }

    #[test]
    fn conversions_uint() {
        let v = Value::from(42_u64);
        assert!(v.is_u64());
        assert_eq!(v.value_type(), ValueType::U64);
        #[cfg(feature = "128bit")]
        {
            let v = Value::from(42_u128);
            assert!(v.is_u64());
            assert!(v.is_u128());
            assert_eq!(v.value_type(), ValueType::U128);
        }
        let v = Value::from(());
        assert!(!v.is_u64());
        assert!(!v.is_u128());
    }

    #[test]
    fn conversions_null() {
        let v = Value::from(());
        assert!(v.is_null());
        assert_eq!(v.value_type(), ValueType::Null);
        let v = Value::from(1);
        assert!(!v.is_null());
    }

    #[test]
    fn default() {
        assert_eq!(Value::default(), Value::null());
    }

    #[test]
    fn mixed_int_cmp() {
        assert_eq!(Value::from(1_u64), Value::from(1_i64));
        assert_eq!(Value::from(1_i64), Value::from(1_u64));
    }

    #[test]
    #[cfg(feature = "128bit")]
    fn mixed_int_cmp_128() {
        assert_eq!(Value::from(1_u64), Value::from(1_u128));
        assert_eq!(Value::from(1_u64), Value::from(1_i128));
        assert_eq!(Value::from(1_i64), Value::from(1_u128));
        assert_eq!(Value::from(1_i64), Value::from(1_i128));

        assert_eq!(Value::from(1_u128), Value::from(1_u128));
        assert_eq!(Value::from(1_u128), Value::from(1_i128));
        assert_eq!(Value::from(1_u128), Value::from(1_u64));
        assert_eq!(Value::from(1_u128), Value::from(1_i64));

        assert_eq!(Value::from(1_i128), Value::from(1_u128));
        assert_eq!(Value::from(1_i128), Value::from(1_i128));
        assert_eq!(Value::from(1_i128), Value::from(1_u64));
        assert_eq!(Value::from(1_i128), Value::from(1_i64));
    }

    #[test]
    fn test_union_cmp() {
        let v: Value = ().into();
        assert_eq!(v, ());
    }
    #[test]
    #[allow(clippy::bool_assert_comparison)]
    fn test_bool_cmp() {
        let v: Value = true.into();
        assert_eq!(v, true);
        let v: Value = false.into();
        assert_eq!(v, false);
    }
}
