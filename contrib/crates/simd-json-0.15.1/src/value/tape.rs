/// A tape of a parsed json, all values are extracted and validated and
/// can be used without further computation.
use value_trait::{StaticNode, TryTypeError, ValueType, base::TypedValue as _};

pub(super) mod array;
mod cmp;
pub(super) mod object;
mod trait_impls;
#[derive(Debug)]
/// `Tape`
pub struct Tape<'input>(pub Vec<Node<'input>>);
pub use array::Array;
pub use object::Object;
impl<'input> Tape<'input> {
    /// Turns the tape into a `Value` that can be used like a `value_trait::Value`
    #[must_use]
    pub fn as_value(&self) -> Value<'_, 'input> {
        // Skip initial zero
        Value(&self.0)
    }
    /// Creates an empty tape with a null element in it
    #[must_use]
    pub fn null() -> Self {
        Self(vec![Node::Static(StaticNode::Null)])
    }

    /// Clears the tape and returns it with a new lifetime to allow re-using the already
    /// allocated buffer.
    #[must_use]
    pub fn reset<'new>(mut self) -> Tape<'new> {
        self.0.clear();
        // SAFETY: At this point the tape is empty, so no data in there has a lifetime associated with it,
        // so we can safely change the lifetime of the tape to 'new
        unsafe { std::mem::transmute(self) }
    }

    /// Deserializes the tape into a type that implements `serde::Deserialize`
    /// # Errors
    /// Returns an error if the deserialization fails
    #[cfg(feature = "serde")]
    pub fn deserialize<T>(self) -> crate::Result<T>
    where
        T: serde::Deserialize<'input>,
    {
        use crate::Deserializer;

        let mut deserializer = Deserializer {
            tape: self.0,
            idx: 0,
        };

        T::deserialize(&mut deserializer)
    }
}

/// Wrapper around the tape that allows interaction via a `Value`-like API.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Value<'tape, 'input>(pub(crate) &'tape [Node<'input>])
where
    'input: 'tape;

impl Value<'static, 'static> {
    const NULL_TAPE: [Node<'static>; 1] = [Node::Static(StaticNode::Null)];
    /// A static null value
    pub const NULL: Value<'static, 'static> = Value(&Self::NULL_TAPE);
    /// Creates tape value representing a null value
    #[must_use]
    pub const fn null() -> Self {
        Self::NULL
    }
}

#[allow(clippy::derive_partial_eq_without_eq)]
/// Tape `Node`
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Node<'input> {
    /// A string, located inside the input slice
    String(&'input str),
    /// An `Object` with the given `size` starts here.
    /// the following values are keys and values, alternating
    /// however values can be nested and have a length themselves.
    Object {
        /// The number of keys in the object
        len: usize,
        /// The total number of nodes in the object, including subelements.
        count: usize,
    },
    /// An array with a given size starts here. The next `size`
    /// elements belong to it - values can be nested and have a
    /// `size` of their own.
    Array {
        /// The number of elements in the array
        len: usize,
        /// The total number of nodes in the array, including subelements.
        count: usize,
    },
    /// A static value that is interned into the tape, it can
    /// be directly taken and isn't nested.
    Static(StaticNode),
}

impl<'input> Node<'input> {
    fn as_str(&self) -> Option<&'input str> {
        if let Node::String(s) = self {
            Some(*s)
        } else {
            None
        }
    }
    /// Returns the type of the node
    #[must_use]
    pub fn value_type(&self) -> ValueType {
        match self {
            Node::String(_) => ValueType::String,
            Node::Object { .. } => ValueType::Object,
            Node::Array { .. } => ValueType::Array,
            Node::Static(v) => v.value_type(),
        }
    }

    // returns the count of elements in an array
    fn array_count(&self) -> Result<usize, TryTypeError> {
        if let Node::Array { count, .. } = self {
            Ok(*count)
        } else {
            Err(TryTypeError {
                expected: ValueType::Array,
                got: self.value_type(),
            })
        }
    }

    // // returns the length of an array
    // fn array_len(&self) -> Result<usize, TryTypeError> {
    //     if let Node::Array { len, .. } = self {
    //         Ok(*len)
    //     } else {
    //         Err(TryTypeError {
    //             expected: ValueType::Array,
    //             got: self.value_type(),
    //         })
    //     }
    // }

    // returns the count of nodes in an object
    fn object_count(&self) -> Result<usize, TryTypeError> {
        if let Node::Object { count, .. } = self {
            Ok(*count)
        } else {
            Err(TryTypeError {
                expected: ValueType::Object,
                got: self.value_type(),
            })
        }
    }

    // returns the count of elements in an array
    fn object_len(&self) -> Result<usize, TryTypeError> {
        if let Node::Object { len, .. } = self {
            Ok(*len)
        } else {
            Err(TryTypeError {
                expected: ValueType::Object,
                got: self.value_type(),
            })
        }
    }

    // returns the count of elements in this node, including the node itself (n for nested, 1 for the rest)
    fn count(&self) -> usize {
        match self {
            // We add 1 as we need to include the header itself
            Node::Object { count, .. } | Node::Array { count, .. } => *count + 1,
            _ => 1,
        }
    }
    //     // Returns the lenght of nested elements
    //     fn as_len(&self) -> Option<usize> {
    //         match self {
    //             Node::Object { len, .. } | Node::Array { len, .. } => Some(*len),
    //             _ => None,
    //         }
    //     }

    // fn as_len_and_count(&self) -> Option<(usize, usize)> {
    //     match self {
    //         Node::Object { len, count } | Node::Array { len, count } => Some((*len, *count)),
    //         _ => None,
    //     }
    // }
}

#[cfg(test)]
mod test {
    #![allow(clippy::cognitive_complexity)]
    use super::StaticNode as Value;
    use super::*;
    use crate::prelude::*;

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
        assert_eq!(Value::default(), Value::Null);
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
