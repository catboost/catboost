use std::{
    borrow::Borrow,
    hash::Hash,
    io::{self, Write},
};

use value_trait::{
    StaticNode, TryTypeError, ValueType,
    base::{TypedValue, ValueAsScalar, ValueIntoArray, ValueIntoObject, ValueIntoString, Writable},
    derived::{
        ValueObjectAccessAsScalar, ValueObjectAccessTryAsScalar, ValueTryAsScalar,
        ValueTryIntoString,
    },
    generator::{
        BaseGenerator, DumpGenerator, PrettyGenerator, PrettyWriterGenerator, WriterGenerator,
    },
};

use crate::Node;

use super::{Array, Object, Value};

// Custom functions
impl Value<'_, '_> {
    fn as_static(&self) -> Option<StaticNode> {
        match self.0.first()? {
            Node::Static(s) => Some(*s),
            _ => None,
        }
    }
}

// TypedContainerValue
impl Value<'_, '_> {
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

impl<'tape, 'input> ValueAsScalar for Value<'tape, 'input>
where
    'input: 'tape,
{
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_null(&self) -> Option<()> {
        self.as_static()?.as_null()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_bool(&self) -> Option<bool> {
        self.as_static()?.as_bool()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_i64(&self) -> Option<i64> {
        self.as_static()?.as_i64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_u64(&self) -> Option<u64> {
        self.as_static()?.as_u64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_f64(&self) -> Option<f64> {
        self.as_static()?.as_f64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn as_str(&self) -> Option<&'input str> {
        self.into_string()
    }
}

impl<'input> ValueIntoString for Value<'_, 'input> {
    type String = &'input str;

    fn into_string(self) -> Option<&'input str> {
        if let Some(Node::String(v)) = self.0.first() {
            Some(v)
        } else {
            None
        }
    }
}

impl<'tape, 'input> ValueIntoArray for Value<'tape, 'input> {
    type Array = Array<'tape, 'input>;
    fn into_array(self) -> Option<Self::Array> {
        if let Some(Node::Array { count, .. }) = self.0.first() {
            // we add one element as we want to keep the array header
            let count = *count + 1;
            Some(Array(&self.0[..count]))
        } else {
            None
        }
    }
}
impl<'tape, 'input> ValueIntoObject for Value<'tape, 'input> {
    type Object = Object<'tape, 'input>;
    fn into_object(self) -> Option<Self::Object> {
        if let Some(Node::Object { count, .. }) = self.0.first() {
            // we add one element as we want to keep the object header
            let count = *count + 1;
            Some(Object(&self.0[..count]))
        } else {
            None
        }
    }
}

impl TypedValue for Value<'_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn value_type(&self) -> ValueType {
        match self.0.first().expect("invalid tape value") {
            Node::Static(StaticNode::Null) => ValueType::Null,
            Node::Static(StaticNode::Bool(_)) => ValueType::Bool,
            Node::Static(StaticNode::I64(_)) => ValueType::I64,
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::I128(_)) => ValueType::I128,
            Node::Static(StaticNode::U64(_)) => ValueType::U64,
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::U128(_)) => ValueType::U128,
            Node::Static(StaticNode::F64(_)) => ValueType::F64,
            Node::String(_) => ValueType::String,
            Node::Array { .. } => ValueType::Array,
            Node::Object { .. } => ValueType::Object,
        }
    }
}

// TryValueObjectAccess
impl<'tape, 'input> Value<'tape, 'input> {
    // type Key = &str;
    // type Target = Value<'tape, 'input>;

    /// Tries to get a value based on a key, returns a `TryTypeError` if the
    /// current Value isn't an Object, returns `None` if the key isn't in the object
    /// # Errors
    /// if the value is not an object
    pub fn try_get<Q>(&self, k: &Q) -> Result<Option<Value<'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Ok(self.try_as_object()?.get(k))
    }
}

//TryValueArrayAccess
impl<'tape, 'input> Value<'tape, 'input>
where
    'input: 'tape,
{
    /// Tries to get a value based on n index, returns a type error if the
    /// current value isn't an Array, returns `None` if the index is out of bounds
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_idx(&self, i: usize) -> Result<Option<Value<'tape, 'input>>, TryTypeError> {
        Ok(self.try_as_array()?.get(i))
    }
}

//ValueAsContainer
impl<'tape, 'input> Value<'tape, 'input>
where
    'input: 'tape,
{
    /// Tries to represent the value as an array and returns a reference to it
    #[must_use]
    pub fn as_array(&self) -> Option<Array<'tape, 'input>> {
        if let Some(Node::Array { count, .. }) = self.0.first() {
            // we add one element as we want to keep the array header
            let count = *count + 1;
            Some(Array(&self.0[..count]))
        } else {
            None
        }
    }

    /// Tries to represent the value as an array and returns a reference to it
    #[must_use]
    pub fn as_object(&self) -> Option<Object<'tape, 'input>> {
        if let Some(Node::Object { count, .. }) = self.0.first() {
            // we add one element as we want to keep the object header
            let count = *count + 1;
            Some(Object(&self.0[..count]))
        } else {
            None
        }
    }
}

// ContainerValueTryAs (needed as we don't have ValueAsContainer)
impl<'tape, 'input> Value<'tape, 'input> {
    /// Tries to represent the value as an array and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    pub fn try_as_array(&self) -> Result<Array<'tape, 'input>, TryTypeError> {
        self.as_array().ok_or(TryTypeError {
            expected: ValueType::Array,
            got: self.value_type(),
        })
    }

    /// Tries to represent the value as an object and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    pub fn try_as_object(&self) -> Result<Object<'tape, 'input>, TryTypeError> {
        self.as_object().ok_or(TryTypeError {
            expected: ValueType::Object,
            got: self.value_type(),
        })
    }
}
// ValueObjectAccess (needed as we don't have ValueAsContainer ) and can't return references
impl<'tape, 'input> Value<'tape, 'input> {
    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    pub fn get<Q>(&self, k: &Q) -> Option<Value<'tape, 'input>>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.as_object().and_then(|a| a.get(k))
    }

    /// Checks if a Value contains a given key. This will return
    /// flase if Value isn't an object  
    #[must_use]
    pub fn contains_key(&self, k: &str) -> bool {
        self.as_object().and_then(|a| a.get(k)).is_some()
    }
}

// ValueArrayAccess (needed as we don't have ValueAsContainer)
impl<'tape, 'input> Value<'tape, 'input> {
    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    pub fn get_idx(&self, i: usize) -> Option<Value<'tape, 'input>> {
        self.as_array().and_then(|a| a.get(i))
    }
}

impl<'tape, 'input> ValueObjectAccessAsScalar for Value<'tape, 'input>
where
    'input: 'tape,
{
    type Key = str;

    fn get_bool<Q>(&self, k: &Q) -> Option<bool>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_bool()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_i128<Q>(&self, k: &Q) -> Option<i128>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i128()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_i64<Q>(&self, k: &Q) -> Option<i64>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i64()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_i32<Q>(&self, k: &Q) -> Option<i32>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i32()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_i16<Q>(&self, k: &Q) -> Option<i16>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i16()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_i8<Q>(&self, k: &Q) -> Option<i8>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_i8()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_u128<Q>(&self, k: &Q) -> Option<u128>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k)?.as_u128()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_u64<Q>(&self, k: &Q) -> Option<u64>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u64())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_usize<Q>(&self, k: &Q) -> Option<usize>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_usize())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_u32<Q>(&self, k: &Q) -> Option<u32>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u32())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_u16<Q>(&self, k: &Q) -> Option<u16>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u16())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_u8<Q>(&self, k: &Q) -> Option<u8>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_u8())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_f64<Q>(&self, k: &Q) -> Option<f64>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_f64())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_f32<Q>(&self, k: &Q) -> Option<f32>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(|v| v.as_f32())
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn get_str<Q>(&self, k: &Q) -> Option<&'input str>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        if let Some(Node::String(v)) = self.get(k)?.0.first() {
            Some(v)
        } else {
            None
        }
    }
}

// ValueObjectContainerAccess
impl<'tape, 'input> Value<'tape, 'input> {
    /// Tries to get an element of an object as a array
    #[must_use]
    pub fn get_array<Q>(&self, k: &Q) -> Option<Array<'tape, 'input>>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let mut len = self.0[0].object_len().ok()?;
        let mut idx = 1;
        while len > 0 {
            let Some(s) = self.0[idx].as_str() else {
                unreachable!(
                    "get_array: object len not reached but no key found {} {}",
                    len, idx
                );
            };
            idx += 1;
            len -= 1;
            let count = self.0[idx].count();
            let s: &Q = s.borrow();
            if s == k {
                let count: usize = self.0[idx].array_count().ok()?;
                return Some(Array(&self.0[idx..idx + count]));
            }
            idx += count;
        }
        None
    }

    /// Tries to get an element of an object as a object
    #[must_use]
    pub fn get_object<Q>(&self, k: &Q) -> Option<Object<'tape, 'input>>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let mut len = self.0[0].object_len().ok()?;
        let mut idx = 1;
        while len > 0 {
            let Some(s) = self.0[idx].as_str() else {
                unreachable!(
                    "get_object: object len not reached but no key found {} {}",
                    len, idx
                );
            };
            idx += 1;
            len -= 1;
            let count = self.0[idx].count();
            let s: &Q = s.borrow();
            if s == k {
                let count: usize = self.0[idx].object_count().ok()?;
                return Some(Object(&self.0[idx..idx + count]));
            }
            idx += count;
        }
        None
    }
}
// TryValueObjectContainerAccess
impl<'tape, 'input> Value<'tape, 'input> {
    /// Tries to get an element of an object as an array, returns
    /// an error if it isn't a array
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_array<Q>(&self, k: &Q) -> Result<Option<Array<'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let mut len = self.0[0].object_len()?;
        let mut idx = 1;
        while len > 0 {
            let Some(s) = self.0[idx].as_str() else {
                unreachable!(
                    "try_get_array: object len not reached but no key found {} {}",
                    len, idx
                );
            };
            idx += 1;
            len -= 1;
            let count = self.0[idx].count();
            let s: &Q = s.borrow();
            if s == k {
                let count: usize = self.0[idx].array_count()?;
                return Ok(Some(Array(&self.0[idx..idx + count])));
            }
            idx += count;
        }
        Ok(None)
    }

    /// Tries to get an element of an object as an object, returns
    /// an error if it isn't an object
    ///
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    pub fn try_get_object<Q>(&self, k: &Q) -> Result<Option<Object<'tape, 'input>>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let mut len = self.0[0].object_len()?;
        let mut idx = 1;
        while len > 0 {
            let Some(s) = self.0[idx].as_str() else {
                unreachable!(
                    "try_get_object: object len not reached but no key found {} {}",
                    len, idx
                );
            };
            idx += 1;
            len -= 1;
            let count = self.0[idx].count();
            let s: &Q = s.borrow();
            if s == k {
                let count: usize = self.0[idx].object_count()?;
                return Ok(Some(Object(&self.0[idx..idx + count])));
            }
            idx += count;
        }
        Ok(None)
    }
}

impl ValueObjectAccessTryAsScalar for Value<'_, '_> {
    type Key = str;
    fn try_get_bool<Q>(&self, k: &Q) -> Result<Option<bool>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_bool()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_i128<Q>(&self, k: &Q) -> Result<Option<i128>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i128()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_i64<Q>(&self, k: &Q) -> Result<Option<i64>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i64()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_i32<Q>(&self, k: &Q) -> Result<Option<i32>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i32()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_i16<Q>(&self, k: &Q) -> Result<Option<i16>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i16()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_i8<Q>(&self, k: &Q) -> Result<Option<i8>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_i8()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_u128<Q>(&self, k: &Q) -> Result<Option<u128>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u128()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_u64<Q>(&self, k: &Q) -> Result<Option<u64>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u64()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_usize<Q>(&self, k: &Q) -> Result<Option<usize>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_usize()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_u32<Q>(&self, k: &Q) -> Result<Option<u32>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u32()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_u16<Q>(&self, k: &Q) -> Result<Option<u16>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u16()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_u8<Q>(&self, k: &Q) -> Result<Option<u8>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_u8()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_f64<Q>(&self, k: &Q) -> Result<Option<f64>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_f64()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_f32<Q>(&self, k: &Q) -> Result<Option<f32>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?.map(|v| v.try_as_f32()).transpose()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn try_get_str<Q>(&self, k: &Q) -> Result<Option<&str>, TryTypeError>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_as_object()?
            .get(k)
            .map(ValueTryIntoString::try_into_string)
            .transpose()
    }
}

impl Writable for Value<'_, '_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode(&self) -> String {
        let mut g = DumpGenerator::new();
        let _r = g.write_json(self);
        g.consume()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn encode_pp(&self) -> String {
        let mut g = PrettyGenerator::new(2);
        let _r = g.write_json(self);
        g.consume()
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        let mut g = WriterGenerator::new(w);
        g.write_json(self)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_pp<'writer, W>(&self, w: &mut W) -> io::Result<()>
    where
        W: 'writer + Write,
    {
        let mut g = PrettyWriterGenerator::new(w, 2);
        g.write_json(self)
    }
}

trait Generator: BaseGenerator {
    type T: Write;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_object(&mut self, object: &Object) -> io::Result<()> {
        if object.is_empty() {
            self.write(b"{}")
        } else {
            let mut iter = object.iter();
            stry!(self.write(b"{"));

            // We know this exists since it's not empty
            let Some((key, value)) = iter.next() else {
                // We check against size
                unreachable!("object is not empty but has no next");
            };
            self.indent();
            stry!(self.new_line());
            stry!(self.write_simple_string(key));
            stry!(self.write_min(b": ", b':'));
            stry!(self.write_json(&value));

            for (key, value) in iter {
                stry!(self.write(b","));
                stry!(self.new_line());
                stry!(self.write_simple_string(key));
                stry!(self.write_min(b": ", b':'));
                stry!(self.write_json(&value));
            }
            self.dedent();
            stry!(self.new_line());
            self.write(b"}")
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_json(&mut self, json: &Value) -> io::Result<()> {
        //FIXME no expect
        match *json.0.first().expect("invalid JSON") {
            Node::Static(StaticNode::Null) => self.write(b"null"),
            Node::Static(StaticNode::I64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::I128(number)) => self.write_int(number),
            Node::Static(StaticNode::U64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::U128(number)) => self.write_int(number),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Node::Static(StaticNode::F64(number)) => self.write_float(number.into()),
            Node::Static(StaticNode::Bool(true)) => self.write(b"true"),
            Node::Static(StaticNode::Bool(false)) => self.write(b"false"),
            Node::String(string) => self.write_string(string),
            Node::Array { len, count } => {
                if len == 0 {
                    self.write(b"[]")
                } else {
                    let array = Array(&json.0[..=count]);
                    let mut iter = array.iter();
                    // We know we have one item

                    let Some(item) = iter.next() else {
                        // We check against size
                        unreachable!("array is not empty but has no next");
                    };
                    stry!(self.write(b"["));
                    self.indent();

                    stry!(self.new_line());
                    stry!(self.write_json(&item));

                    for item in iter {
                        stry!(self.write(b","));
                        stry!(self.new_line());
                        stry!(self.write_json(&item));
                    }
                    self.dedent();
                    stry!(self.new_line());
                    self.write(b"]")
                }
            }
            Node::Object { count, .. } => self.write_object(&Object(&json.0[..=count])),
        }
    }
}

trait FastGenerator: BaseGenerator {
    type T: Write;

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_object(&mut self, object: &Object) -> io::Result<()> {
        if object.is_empty() {
            self.write(b"{}")
        } else {
            let mut iter = object.iter();
            stry!(self.write(b"{\""));

            // We know this exists since it's not empty
            let Some((key, value)) = iter.next() else {
                // We check against size
                unreachable!("object is not empty but has no next");
            };
            stry!(self.write_simple_str_content(key));
            stry!(self.write(b"\":"));
            stry!(self.write_json(&value));

            for (key, value) in iter {
                stry!(self.write(b",\""));
                stry!(self.write_simple_str_content(key));
                stry!(self.write(b"\":"));
                stry!(self.write_json(&value));
            }
            self.write(b"}")
        }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn write_json(&mut self, json: &Value) -> io::Result<()> {
        match *json.0.first().expect("invalid JSON") {
            Node::Static(StaticNode::Null) => self.write(b"null"),
            Node::Static(StaticNode::I64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::I128(number)) => self.write_int(number),
            Node::Static(StaticNode::U64(number)) => self.write_int(number),
            #[cfg(feature = "128bit")]
            Node::Static(StaticNode::U128(number)) => self.write_int(number),
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Node::Static(StaticNode::F64(number)) => self.write_float(number.into()),
            Node::Static(StaticNode::Bool(true)) => self.write(b"true"),
            Node::Static(StaticNode::Bool(false)) => self.write(b"false"),
            Node::String(string) => self.write_string(string),
            Node::Array { len, count } => {
                if len == 0 {
                    self.write(b"[]")
                } else {
                    let array = Array(&json.0[..=count]);
                    let mut iter = array.iter();
                    // We know we have one item
                    let Some(item) = iter.next() else {
                        // We check against size
                        unreachable!("array is not empty but has no next");
                    };

                    stry!(self.write(b"["));
                    stry!(self.write_json(&item));

                    for item in iter {
                        stry!(self.write(b","));
                        stry!(self.write_json(&item));
                    }
                    self.write(b"]")
                }
            }
            Node::Object { count, .. } => self.write_object(&Object(&json.0[..=count])),
        }
    }
}

impl FastGenerator for DumpGenerator {
    type T = Vec<u8>;
}

impl Generator for PrettyGenerator {
    type T = Vec<u8>;
}

impl<W> FastGenerator for WriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;
}

impl<W> Generator for PrettyWriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;
}
