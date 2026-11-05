use std::{
    borrow::{Borrow, Cow},
    hash::Hash,
    io::{self, Write},
};

use value_trait::{
    TryTypeError, ValueBuilder, ValueType,
    base::{
        TypedValue, ValueAsArray as _, ValueAsMutArray, ValueAsMutObject, ValueAsObject as _,
        ValueAsScalar, ValueIntoString, Writable,
    },
    derived::{
        ValueArrayTryAccess, ValueObjectAccessAsArray as _, ValueObjectAccessAsObject as _,
        ValueObjectAccessAsScalar, ValueObjectAccessTryAsArray as _,
        ValueObjectAccessTryAsObject as _, ValueObjectAccessTryAsScalar, ValueObjectTryAccess,
        ValueTryAsScalar,
    },
};

use crate::{borrowed, tape};

use super::{Array, Object, Value};

impl<'value> ValueBuilder<'value> for Value<'static, 'static, 'value> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn null() -> Self {
        Value::Tape(tape::Value::null())
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn array_with_capacity(capacity: usize) -> Self {
        Value::Value(Cow::Owned(borrowed::Value::array_with_capacity(capacity)))
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn object_with_capacity(capacity: usize) -> Self {
        Value::Value(Cow::Owned(borrowed::Value::object_with_capacity(capacity)))
    }
}

// TypedContainerValue
impl<'tape, 'input> Value<'_, 'tape, 'input>
where
    'input: 'tape,
{
    /// returns true if the current value can be represented as an array
    #[must_use]
    pub fn is_array(&self) -> bool {
        self.as_array().is_some()
    }

    /// returns true if the current value can be represented as an object
    #[must_use]
    pub fn is_object(&self) -> bool {
        self.as_object().is_some()
    }
}

impl ValueAsScalar for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_null(&self) -> Option<()> {
        match &self {
            Value::Tape(tape) => tape.as_null(),
            Value::Value(value) => value.as_null(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_bool(&self) -> Option<bool> {
        match &self {
            Value::Tape(tape) => tape.as_bool(),
            Value::Value(value) => value.as_bool(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_i64(&self) -> Option<i64> {
        match &self {
            Value::Tape(tape) => tape.as_i64(),
            Value::Value(value) => value.as_i64(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_i128(&self) -> Option<i128> {
        match &self {
            Value::Tape(tape) => tape.as_i128(),
            Value::Value(value) => value.as_i128(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_u64(&self) -> Option<u64> {
        match &self {
            Value::Tape(tape) => tape.as_u64(),
            Value::Value(value) => value.as_u64(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_u128(&self) -> Option<u128> {
        match &self {
            Value::Tape(tape) => tape.as_u128(),
            Value::Value(value) => value.as_u128(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_f64(&self) -> Option<f64> {
        match &self {
            Value::Tape(tape) => tape.as_f64(),
            Value::Value(value) => value.as_f64(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn cast_f64(&self) -> Option<f64> {
        match &self {
            Value::Tape(tape) => tape.cast_f64(),
            Value::Value(value) => value.cast_f64(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_str(&self) -> Option<&str> {
        match &self {
            Value::Tape(tape) => tape.as_str(),
            Value::Value(value) => value.as_str(),
        }
    }
}

impl<'value> ValueIntoString for Value<'_, '_, 'value> {
    type String = Cow<'value, str>;

    fn into_string(self) -> Option<<Self as ValueIntoString>::String> {
        match self {
            Value::Tape(tape) => tape.into_string().map(Cow::Borrowed),
            // This is a bit complex but it allows us to avoid cloning
            Value::Value(value) => match value {
                Cow::Borrowed(value) => match value {
                    #[cfg(feature = "beef")]
                    borrowed::Value::String(s) => Some(s.clone().into()),
                    #[cfg(not(feature = "beef"))]
                    borrowed::Value::String(s) => Some(s.clone()),
                    _ => None,
                },
                Cow::Owned(value) => match value {
                    #[cfg(feature = "beef")]
                    borrowed::Value::String(s) => Some(s.into()),
                    #[cfg(not(feature = "beef"))]
                    borrowed::Value::String(s) => Some(s),
                    _ => None,
                },
            },
        }
    }
}

// impl<'tape, 'input> ValueIntoContainer for Value<'tape, 'input> {
//     type Array = array::Array<'tape, 'input>;
//     type Object = Object<'tape, 'input>;
//     fn into_array(self) -> Option<Self::Array> {
//         if let Some(Node::Array { count, .. }) = self.0.first() {
//             // we add one element as we want to keep the array header
//             let count = *count + 1;
//             Some(Array(&self.0[..count]))
//         } else {
//             None
//         }
//     }

//     fn into_object(self) -> Option<Self::Object> {
//         if let Some(Node::Object { count, .. }) = self.0.first() {
//             // we add one element as we want to keep the object header
//             let count = *count + 1;
//             Some(Object(&self.0[..count]))
//         } else {
//             None
//         }
//     }
// }

impl TypedValue for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn value_type(&self) -> ValueType {
        match &self {
            Value::Tape(tape) => tape.value_type(),
            Value::Value(value) => value.value_type(),
        }
    }
}

// TryValueObjectAccess
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    // type Key = &str;
    // type Target = Value<'tape, 'input>;

    /// Tries to get a value based on a key, returns a `TryTypeError` if the
    /// current Value isn't an Object, returns `None` if the key isn't in the object
    /// # Errors
    /// if the value is not an object
    pub fn try_get<Q>(&self, k: &Q) -> Result<Option<Value<'_, 'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        for<'b> crate::cow::Cow<'b, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => Ok(tape.try_get(k)?.map(Value::Tape)),
            Value::Value(value) => Ok(value.try_get(k)?.map(Cow::Borrowed).map(Value::Value)),
        }
    }
}

//TryValueArrayAccess
impl<'tape, 'input> Value<'_, 'tape, 'input>
where
    'input: 'tape,
{
    /// Tries to get a value based on n index, returns a type error if the
    /// current value isn't an Array, returns `None` if the index is out of bounds
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_idx(&self, i: usize) -> Result<Option<Value<'_, 'tape, 'input>>, TryTypeError> {
        match self {
            Value::Tape(tape) => Ok(tape.try_get_idx(i)?.map(Value::Tape)),
            Value::Value(value) => Ok(value.try_get_idx(i)?.map(Cow::Borrowed).map(Value::Value)),
        }
    }
}

// impl<'tape, 'value> ValueAsContainer for Value<'tape, 'value> {
impl<'tape, 'input> Value<'_, 'tape, 'input>
where
    'input: 'tape,
{
    // type Array = array::Array<'tape, 'value>;
    // type Object = Object<'tape, 'value>;

    /// Tries to represent the value as an array and returns a reference to it
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn as_array(&self) -> Option<Array<'_, 'tape, 'input>> {
        match self {
            Value::Tape(tape) => tape.as_array().map(Array::Tape),
            Value::Value(value) => value.as_array().map(Array::Value),
        }
    }

    /// Tries to represent the value as an array and returns a reference to it
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn as_object(&self) -> Option<Object<'_, 'tape, 'input>> {
        match self {
            Value::Tape(tape) => tape.as_object().map(Object::Tape),
            Value::Value(value) => value.as_object().map(Object::Value),
        }
    }
}

impl<'input> ValueAsMutArray for Value<'_, '_, 'input> {
    type Array = Vec<borrowed::Value<'input>>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_array_mut(&mut self) -> Option<&mut Vec<borrowed::Value<'input>>> {
        self.as_mut().as_array_mut()
    }
}
impl<'input> ValueAsMutObject for Value<'_, '_, 'input> {
    type Object = super::borrowed::Object<'input>;
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_object_mut(&mut self) -> Option<&mut super::borrowed::Object<'input>> {
        self.as_mut().as_object_mut()
    }
}

// ContainerValueTryAs (needed as we don't have ValueAsContainer)
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// Tries to represent the value as an array and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    pub fn try_as_array(&self) -> Result<Array<'_, 'tape, 'input>, TryTypeError> {
        self.as_array().ok_or(TryTypeError {
            expected: ValueType::Array,
            got: self.value_type(),
        })
    }

    /// Tries to represent the value as an object and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    pub fn try_as_object(&self) -> Result<Object<'_, 'tape, 'input>, TryTypeError> {
        self.as_object().ok_or(TryTypeError {
            expected: ValueType::Object,
            got: self.value_type(),
        })
    }
}
// ValueObjectAccess (needed as we don't have ValueAsContainer ) and can't return references
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    #[must_use]
    pub fn get<'k, Q>(&self, k: &'k Q) -> Option<Value<'_, 'tape, 'input>>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => tape.get(k).map(Value::Tape),
            Value::Value(value) => value
                .as_object()?
                .get(k)
                .map(Cow::Borrowed)
                .map(Value::Value),
        }
    }

    /// Checks if a Value contains a given key. This will return
    /// flase if Value isn't an object  
    #[must_use]
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let Some(o) = self.as_object() else {
            return false;
        };

        o.get(k).is_some()
    }
}

// ValueArrayAccess (needed as we don't have ValueAsContainer)
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    pub fn get_idx(&self, i: usize) -> Option<Value<'_, 'tape, 'input>> {
        match self {
            Value::Tape(tape) => tape.get_idx(i).map(Value::Tape),
            Value::Value(value) => value
                .as_array()?
                .get(i)
                .map(Cow::Borrowed)
                .map(Value::Value),
        }
    }
}

// impl<'tape, 'input> ValueObjectAccessAsScalar for Value<'tape, 'input>
impl<'tape, 'input> Value<'_, 'tape, 'input>
where
    'input: 'tape,
{
    /// Tries to get an element of an object as a bool
    pub fn get_bool<Q>(&self, k: &Q) -> Option<bool>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_bool()
    }

    /// Tries to get an element of an object as a i128
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_i128<Q>(&self, k: &Q) -> Option<i128>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i128()
    }

    /// Tries to get an element of an object as a i64
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_i64<Q>(&self, k: &Q) -> Option<i64>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i64()
    }

    /// Tries to get an element of an object as a i32
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_i32<Q>(&self, k: &Q) -> Option<i32>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i32()
    }

    /// Tries to get an element of an object as a i16
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_i16<Q>(&self, k: &Q) -> Option<i16>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i16()
    }

    /// Tries to get an element of an object as a i8
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_i8<Q>(&self, k: &Q) -> Option<i8>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i8()
    }

    /// Tries to get an element of an object as a u128
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_u128<Q>(&self, k: &Q) -> Option<u128>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_u128()
    }

    /// Tries to get an element of an object as a u64
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_u64<Q>(&self, k: &Q) -> Option<u64>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u64())
    }

    /// Tries to get an element of an object as a usize
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_usize<Q>(&self, k: &Q) -> Option<usize>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_usize())
    }

    /// Tries to get an element of an object as a u32
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_u32<Q>(&self, k: &Q) -> Option<u32>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u32())
    }

    /// Tries to get an element of an object as a u16
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_u16<Q>(&self, k: &Q) -> Option<u16>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u16())
    }

    /// Tries to get an element of an object as a u8
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_u8<Q>(&self, k: &Q) -> Option<u8>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u8())
    }

    /// Tries to get an element of an object as a f64
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_f64<Q>(&self, k: &Q) -> Option<f64>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_f64())
    }

    /// Tries to get an element of an object as a f32
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_f32<Q>(&self, k: &Q) -> Option<f32>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_f32())
    }
    /// Tries to get an element of an object as a str
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn get_str<Q>(&self, k: &Q) -> Option<&str>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => tape.get_str(k),
            Value::Value(value) => value.get_str(k),
        }
    }
}

// ValueObjectContainerAccess
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// Tries to get an element of an object as a array
    #[must_use]
    pub fn get_array<Q>(&self, k: &Q) -> Option<Array<'_, 'tape, 'input>>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => tape.get_array(k).map(Array::Tape),
            Value::Value(value) => value.get_array(k).map(Array::Value),
        }
    }

    /// Tries to get an element of an object as a object
    #[must_use]
    pub fn get_object<Q>(&self, k: &Q) -> Option<Object<'_, 'tape, 'input>>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => tape.get_object(k).map(Object::Tape),
            Value::Value(value) => value.get_object(k).map(Object::Value),
        }
    }
}
// TryValueObjectContainerAccess
impl<'tape, 'input> Value<'_, 'tape, 'input> {
    /// Tries to get an element of an object as an array, returns
    /// an error if it isn't a array
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_array<Q>(&self, k: &Q) -> Result<Option<Array<'_, 'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => Ok(tape.try_get_array(k)?.map(Array::Tape)),
            Value::Value(value) => Ok(value.try_get_array(k)?.map(Array::Value)),
        }
    }

    /// Tries to get an element of an object as an object, returns
    /// an error if it isn't an object
    ///
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_object<Q>(
        &self,
        k: &Q,
    ) -> Result<Option<Object<'_, 'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => Ok(tape.try_get_object(k)?.map(Object::Tape)),
            Value::Value(value) => Ok(value.try_get_object(k)?.map(Object::Value)),
        }
    }
}

// impl<'tape, 'input> ValueObjectAccessTryAsScalar for Value<'tape, 'input> {
impl Value<'_, '_, '_> {
    /// Tries to get an element of an object as a bool, returns
    /// an error if it isn't bool
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_bool<Q>(&self, k: &Q) -> Result<Option<bool>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_bool()).transpose()
    }

    /// Tries to get an element of an object as a i128, returns
    /// an error if it isn't i128
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_i128<Q>(&self, k: &Q) -> Result<Option<i128>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i128()).transpose()
    }

    /// Tries to get an element of an object as a i64, returns
    /// an error if it isn't a i64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_i64<Q>(&self, k: &Q) -> Result<Option<i64>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i64()).transpose()
    }

    /// Tries to get an element of an object as a i32, returns
    /// an error if it isn't a i32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_i32<Q>(&self, k: &Q) -> Result<Option<i32>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i32()).transpose()
    }

    /// Tries to get an element of an object as a i16, returns
    /// an error if it isn't a i16
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_i16<Q>(&self, k: &Q) -> Result<Option<i16>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i16()).transpose()
    }

    /// Tries to get an element of an object as a u128, returns
    /// an error if it isn't a u128
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_i8<Q>(&self, k: &Q) -> Result<Option<i8>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i8()).transpose()
    }

    /// Tries to get an element of an object as a u64, returns
    /// an error if it isn't a u64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_u128<Q>(&self, k: &Q) -> Result<Option<u128>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u128()).transpose()
    }

    /// Tries to get an element of an object as a usize, returns
    /// an error if it isn't a usize
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_u64<Q>(&self, k: &Q) -> Result<Option<u64>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u64()).transpose()
    }

    /// Tries to get an element of an object as a u32, returns
    /// an error if it isn't a u32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_usize<Q>(&self, k: &Q) -> Result<Option<usize>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_usize()).transpose()
    }

    /// Tries to get an element of an object as a u16, returns
    /// an error if it isn't a u16
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_u32<Q>(&self, k: &Q) -> Result<Option<u32>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u32()).transpose()
    }

    /// Tries to get an element of an object as a u8, returns
    /// an error if it isn't a u8
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_u16<Q>(&self, k: &Q) -> Result<Option<u16>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u16()).transpose()
    }

    /// Tries to get an element of an object as a u8, returns
    /// an error if it isn't a u8
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_u8<Q>(&self, k: &Q) -> Result<Option<u8>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u8()).transpose()
    }

    /// Tries to get an element of an object as a f64, returns
    /// an error if it isn't a f64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_f64<Q>(&self, k: &Q) -> Result<Option<f64>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_f64()).transpose()
    }

    /// Tries to get an element of an object as a f32, returns
    /// an error if it isn't a f32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_f32<Q>(&self, k: &Q) -> Result<Option<f32>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_f32()).transpose()
    }

    /// Tries to get an element of an object as a str, returns
    /// an error if it isn't a str
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub fn try_get_str<Q>(&self, k: &Q) -> Result<Option<&'_ str>, TryTypeError>
    where
        str: Borrow<Q>,
        for<'a> crate::cow::Cow<'a, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Value::Tape(tape) => tape.try_get_str(k),
            Value::Value(value) => value.try_get_str(k),
        }
    }
}

impl Writable for Value<'_, '_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode(&self) -> String {
        match self {
            Value::Tape(tape) => tape.encode(),
            Value::Value(value) => value.encode(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode_pp(&self) -> String {
        match self {
            Value::Tape(tape) => tape.encode_pp(),
            Value::Value(value) => value.encode_pp(),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        match self {
            Value::Tape(tape) => tape.write(w),
            Value::Value(value) => value.write(w),
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_pp<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        match self {
            Value::Tape(tape) => tape.write_pp(w),
            Value::Value(value) => value.write_pp(w),
        }
    }
}
