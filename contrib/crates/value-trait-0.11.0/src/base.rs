use std::io::{self, Write};

use crate::{TryTypeError, ValueType, array::Array, object::Object};

/// Type information on a value
pub trait TypedValue {
    /// Gets the type of the current value
    #[must_use]
    fn value_type(&self) -> ValueType;
}

/// Type checks for custom values on a value
pub trait TypedCustomValue {
    #[cfg(feature = "custom-types")]
    /// returns if a type is a custom type
    fn is_custom(&self) -> bool {
        false
    }
}

/// Access to scalar value types
pub trait ValueAsScalar {
    /// Tries to represent the value as a 'null';
    #[must_use]
    fn as_null(&self) -> Option<()>;

    /// Tries to represent the value as a bool
    #[must_use]
    fn as_bool(&self) -> Option<bool>;

    /// Tries to represent the value as an i128
    #[inline]
    #[must_use]
    fn as_i128(&self) -> Option<i128> {
        self.as_i64().map(Into::into)
    }

    /// Tries to represent the value as an i64
    #[must_use]
    fn as_i64(&self) -> Option<i64>;

    /// Tries to represent the value as an i32
    #[inline]
    #[must_use]
    fn as_i32(&self) -> Option<i32> {
        self.as_i64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as an i16
    #[inline]
    #[must_use]
    fn as_i16(&self) -> Option<i16> {
        self.as_i64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as an i8
    #[inline]
    #[must_use]
    fn as_i8(&self) -> Option<i8> {
        self.as_i64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as an u128
    #[inline]
    #[must_use]
    fn as_u128(&self) -> Option<u128> {
        self.as_u64().map(Into::into)
    }

    /// Tries to represent the value as an u64
    #[must_use]
    fn as_u64(&self) -> Option<u64>;

    /// Tries to represent the value as an usize
    #[inline]
    #[must_use]
    fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as an u32
    #[inline]
    #[must_use]
    fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|u| u.try_into().ok())
    }
    /// Tries to represent the value as an u16
    #[inline]
    #[must_use]
    fn as_u16(&self) -> Option<u16> {
        self.as_u64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as an u8
    #[inline]
    #[must_use]
    fn as_u8(&self) -> Option<u8> {
        self.as_u64().and_then(|u| u.try_into().ok())
    }

    /// Tries to represent the value as a f64
    #[must_use]
    fn as_f64(&self) -> Option<f64>;

    /// Tries to represent the value as a f32
    #[inline]
    #[must_use]
    fn as_f32(&self) -> Option<f32> {
        self.as_f64().and_then(|u| {
            if u <= f64::from(f32::MAX) && u >= f64::from(f32::MIN) {
                // Since we check above
                #[allow(clippy::cast_possible_truncation)]
                Some(u as f32)
            } else {
                None
            }
        })
    }

    /// Casts the current value to a f64 if possible, this will turn integer
    /// values into floats.
    #[must_use]
    #[inline]
    #[allow(clippy::cast_precision_loss, clippy::option_if_let_else)]
    fn cast_f64(&self) -> Option<f64> {
        if let Some(f) = self.as_f64() {
            Some(f)
        } else if let Some(u) = self.as_u128() {
            Some(u as f64)
        } else {
            self.as_i128().map(|i| i as f64)
        }
    }

    /// Tries to represent the value as a &str
    #[must_use]
    fn as_str(&self) -> Option<&str>;

    /// Tries to represent the value as a Char
    #[inline]
    #[must_use]
    fn as_char(&self) -> Option<char> {
        self.as_str().and_then(|s| s.chars().next())
    }
}

/// Trait to allow accessing data inside a Value
pub trait ValueAsArray {
    /// The array structure
    type Array: Array;

    /// Tries to represent the value as an array and returns a reference to it
    #[must_use]
    fn as_array(&self) -> Option<&Self::Array>;
}

/// Trait to allow Value as an object
pub trait ValueAsObject {
    /// The object structure
    type Object: Object;

    /// Tries to represent the value as an object and returns a reference to it
    #[must_use]
    fn as_object(&self) -> Option<&Self::Object>;
}

/// Mutatability for Array values
pub trait ValueAsMutArray {
    /// The type for Arrays
    type Array;
    /// Tries to represent the value as an array and returns a mutable reference to it
    fn as_array_mut(&mut self) -> Option<&mut Self::Array>;
}

/// `try_as_array_mut` access to array value types
pub trait ValueTryAsArrayMut {
    /// The array structure
    type Array: Array;

    /// Tries to represent the value as an array and returns a mutable reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_array_mut(&mut self) -> Result<&mut Self::Array, TryTypeError>;
}

/// Mutatability for Object values
pub trait ValueAsMutObject {
    /// The type for Objects
    type Object;
    /// Tries to represent the value as an object and returns a mutable reference to it
    fn as_object_mut(&mut self) -> Option<&mut Self::Object>;
}
/// Mutatability for Object values
pub trait ValueTryAsMutObject {
    /// The type for Objects
    type Object;
    /// Tries to represent the value as an object and returns a mutable reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_object_mut(&mut self) -> Result<&mut Self::Object, TryTypeError>;
}

/// A trait that specifies how to turn the Value `into` it's sub types
pub trait ValueIntoString {
    /// The type for Strings
    type String;

    /// Tries to turn the value into it's string representation
    #[must_use]
    fn into_string(self) -> Option<Self::String>;
}

/// A trait that specifies how to turn the Value `into` it's sub types
pub trait ValueIntoArray {
    /// The type for Arrays
    type Array;

    /// Tries to turn the value into it's array representation
    #[must_use]
    fn into_array(self) -> Option<Self::Array>;
}
/// A trait that specifies how to turn the Value `into` it's sub types
pub trait ValueIntoObject {
    /// The type for Objects
    type Object;

    /// Tries to turn the value into it's object representation
    #[must_use]
    fn into_object(self) -> Option<Self::Object>;
}

/// A Value that can be serialized and written
pub trait Writable {
    /// Encodes the value into it's JSON representation as a string
    #[must_use]
    fn encode(&self) -> String;

    /// Encodes the value into it's JSON representation as a string (pretty printed)
    #[must_use]
    fn encode_pp(&self) -> String;

    /// Encodes the value into it's JSON representation into a Writer
    ///
    /// # Errors
    ///
    /// Will return `Err` if an IO error is encountered
    fn write<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write;

    /// Encodes the value into it's JSON representation into a Writer, pretty printed
    ///
    /// # Errors
    ///
    /// Will return `Err` if an IO error is encountered.
    fn write_pp<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write;
}
