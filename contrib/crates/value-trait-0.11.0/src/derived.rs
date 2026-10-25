use std::{borrow::Borrow, hash::Hash};

use crate::{AccessError, TryTypeError, array::Array, object::Object};

/// `try_as_*` access to scalar value types
pub trait ValueTryAsScalar {
    /// Tries to represent the value as a bool
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_bool(&self) -> Result<bool, TryTypeError>;

    /// Tries to represent the value as a i128
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_i128(&self) -> Result<i128, TryTypeError>;

    /// Tries to represent the value as an i64
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_i64(&self) -> Result<i64, TryTypeError>;
    /// Tries to represent the value as an i32
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_i32(&self) -> Result<i32, TryTypeError>;

    /// Tries to represent the value as an i16
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_i16(&self) -> Result<i16, TryTypeError>;
    /// Tries to represent the value as an i8
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_i8(&self) -> Result<i8, TryTypeError>;

    /// Tries to represent the value as an u128
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_u128(&self) -> Result<u128, TryTypeError>;

    /// Tries to represent the value as an u64
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_u64(&self) -> Result<u64, TryTypeError>;

    /// Tries to represent the value as an usize
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_usize(&self) -> Result<usize, TryTypeError>;

    /// Tries to represent the value as an u32
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_u32(&self) -> Result<u32, TryTypeError>;

    /// Tries to represent the value as an u16
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_u16(&self) -> Result<u16, TryTypeError>;

    /// Tries to represent the value as an u8
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_u8(&self) -> Result<u8, TryTypeError>;

    /// Tries to represent the value as a f64
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_f64(&self) -> Result<f64, TryTypeError>;

    /// Tries to Casts the current value to a f64 if possible, this will turn integer
    /// values into floats and error if it isn't possible
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_cast_f64(&self) -> Result<f64, TryTypeError>;

    /// Tries to represent the value as a f32
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_f32(&self) -> Result<f32, TryTypeError>;

    /// Tries to represent the value as a &str
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_str(&self) -> Result<&str, TryTypeError>;

    /// Tries to represent the value as a Char
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_char(&self) -> Result<char, TryTypeError>;
}

/// Type checks for scalar values on a value
pub trait TypedScalarValue {
    /// returns true if the current value is null
    #[must_use]
    fn is_null(&self) -> bool;

    /// returns true if the current value a floatingpoint number
    #[must_use]
    fn is_float(&self) -> bool;

    /// returns true if the current value a integer number
    #[must_use]
    fn is_integer(&self) -> bool;

    /// returns true if the current value a number either float or intege    
    #[must_use]
    fn is_number(&self) -> bool;

    /// returns true if the current value a bool
    #[must_use]
    fn is_bool(&self) -> bool;

    /// returns true if the current value can be represented as a i128
    #[must_use]
    fn is_i128(&self) -> bool;

    /// returns true if the current value can be represented as a i64
    #[must_use]
    fn is_i64(&self) -> bool;

    /// returns true if the current value can be represented as a i32
    #[must_use]
    fn is_i32(&self) -> bool;

    /// returns true if the current value can be represented as a i16
    #[must_use]
    fn is_i16(&self) -> bool;

    /// returns true if the current value can be represented as a i8
    #[must_use]
    fn is_i8(&self) -> bool;

    /// returns true if the current value can be represented as a u128
    #[must_use]
    fn is_u128(&self) -> bool;

    /// returns true if the current value can be represented as a u64
    #[must_use]
    fn is_u64(&self) -> bool;

    /// returns true if the current value can be represented as a usize
    #[must_use]
    fn is_usize(&self) -> bool;

    /// returns true if the current value can be represented as a u32    
    #[must_use]
    fn is_u32(&self) -> bool;

    /// returns true if the current value can be represented as a u16
    #[must_use]
    fn is_u16(&self) -> bool;

    /// returns true if the current value can be represented as a u8
    #[must_use]
    fn is_u8(&self) -> bool;

    /// returns true if the current value can be represented as a f64
    #[must_use]
    fn is_f64(&self) -> bool;

    /// returns true if the current value can be cast into a f64
    #[must_use]
    fn is_f64_castable(&self) -> bool;

    /// returns true if the current value can be represented as a f64
    #[must_use]
    fn is_f32(&self) -> bool;
    /// returns true if the current value can be represented as a str
    #[must_use]
    fn is_str(&self) -> bool;

    /// returns true if the current value can be represented as a char
    #[must_use]
    fn is_char(&self) -> bool;
}

/// Type checks for array values on a value
pub trait TypedArrayValue {
    /// returns true if the current value can be represented as an array
    #[must_use]
    fn is_array(&self) -> bool;
}

/// Type checks for object values on a value
pub trait TypedObjectValue {
    /// returns true if the current value can be represented as an object
    #[must_use]
    fn is_object(&self) -> bool;
}

/// `try_as_*` access to array value types
pub trait ValueTryAsArray {
    /// The array structure
    type Array: Array;

    /// Tries to represent the value as an array and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_array(&self) -> Result<&Self::Array, TryTypeError>;
}
/// `try_as_*` access to object value types
pub trait ValueTryAsObject {
    /// The object structure
    type Object: Object;

    /// Tries to represent the value as an object and returns a reference to it
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_as_object(&self) -> Result<&Self::Object, TryTypeError>;
}

/// Access to a value as an object
pub trait ValueObjectAccess {
    /// The type for Objects
    type Key: ?Sized;
    /// The target for nested lookups
    type Target;

    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    fn get<Q>(&self, k: &Q) -> Option<&Self::Target>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Checks if a Value contains a given key. This will return
    /// flase if Value isn't an object  
    fn contains_key<Q>(&self, k: &Q) -> bool
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

/// Access to a value as an array
pub trait ValueArrayAccess<I> {
    /// The target for nested lookups
    type Target: ?Sized;
    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    fn get_idx(&self, i: I) -> Option<&Self::Target>;
}

/// Access to scalar values in an object
pub trait ValueObjectAccessAsScalar {
    /// The type for Objects
    type Key: ?Sized;
    /// Tries to get an element of an object as a bool
    fn get_bool<Q>(&self, k: &Q) -> Option<bool>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i128
    fn get_i128<Q>(&self, k: &Q) -> Option<i128>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i64
    fn get_i64<Q>(&self, k: &Q) -> Option<i64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i32
    fn get_i32<Q>(&self, k: &Q) -> Option<i32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i16
    fn get_i16<Q>(&self, k: &Q) -> Option<i16>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i8
    fn get_i8<Q>(&self, k: &Q) -> Option<i8>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u128
    fn get_u128<Q>(&self, k: &Q) -> Option<u128>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u64
    fn get_u64<Q>(&self, k: &Q) -> Option<u64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a usize
    fn get_usize<Q>(&self, k: &Q) -> Option<usize>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u32
    fn get_u32<Q>(&self, k: &Q) -> Option<u32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u16
    fn get_u16<Q>(&self, k: &Q) -> Option<u16>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u8
    fn get_u8<Q>(&self, k: &Q) -> Option<u8>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a f64
    fn get_f64<Q>(&self, k: &Q) -> Option<f64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a f32
    fn get_f32<Q>(&self, k: &Q) -> Option<f32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a str
    fn get_str<Q>(&self, k: &Q) -> Option<&str>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}
/// `try_as_*` access to scalar values in an object
pub trait ValueObjectAccessTryAsScalar {
    /// The type for Objects
    type Key: ?Sized;

    /// Tries to get an element of an object as a bool, returns
    /// an error if it isn't bool
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_bool<Q>(&self, k: &Q) -> Result<Option<bool>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i128, returns
    /// an error if it isn't i128
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_i128<Q>(&self, k: &Q) -> Result<Option<i128>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i64, returns
    /// an error if it isn't a i64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_i64<Q>(&self, k: &Q) -> Result<Option<i64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i32, returns
    /// an error if it isn't a i32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_i32<Q>(&self, k: &Q) -> Result<Option<i32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i16, returns
    /// an error if it isn't a i16
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_i16<Q>(&self, k: &Q) -> Result<Option<i16>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a i8, returns
    /// an error if it isn't a i8
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_i8<Q>(&self, k: &Q) -> Result<Option<i8>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u128, returns
    /// an error if it isn't a u128
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_u128<Q>(&self, k: &Q) -> Result<Option<u128>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u64, returns
    /// an error if it isn't a u64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_u64<Q>(&self, k: &Q) -> Result<Option<u64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a usize, returns
    /// an error if it isn't a usize
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_usize<Q>(&self, k: &Q) -> Result<Option<usize>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u32, returns
    /// an error if it isn't a u32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_u32<Q>(&self, k: &Q) -> Result<Option<u32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u16, returns
    /// an error if it isn't a u16
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_u16<Q>(&self, k: &Q) -> Result<Option<u16>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a u8, returns
    /// an error if it isn't a u8
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_u8<Q>(&self, k: &Q) -> Result<Option<u8>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a f64, returns
    /// an error if it isn't a f64
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_f64<Q>(&self, k: &Q) -> Result<Option<f64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a f32, returns
    /// an error if it isn't a f32
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_f32<Q>(&self, k: &Q) -> Result<Option<f32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to get an element of an object as a str, returns
    /// an error if it isn't a str
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_str<Q>(&self, k: &Q) -> Result<Option<&str>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

/// Access to array values in an object
pub trait ValueObjectAccessAsArray {
    /// The type for Objects
    type Key: ?Sized;
    /// The array structure
    type Array: Array;

    /// Tries to get an element of an object as a array
    fn get_array<Q>(&self, k: &Q) -> Option<&Self::Array>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

/// Access to object values in an object
pub trait ValueObjectAccessAsObject {
    /// The type for Objects
    type Key: ?Sized;
    /// The object structure
    type Object: Object;

    /// Tries to get an element of an object as a object
    fn get_object<Q>(&self, k: &Q) -> Option<&Self::Object>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

/// `try_get_array` access to object values in an object
pub trait ValueObjectAccessTryAsArray {
    /// The type for Objects
    type Key: ?Sized;
    /// The array structure
    type Array: Array;

    /// Tries to get an element of an object as an array, returns
    /// an error if it isn't a array
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_array<Q>(&self, k: &Q) -> Result<Option<&Self::Array>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}
/// `try_get_object` access to object values in an object
pub trait ValueObjectAccessTryAsObject {
    /// The type for Objects
    type Key: ?Sized;
    /// The object structure
    type Object: Object;

    /// Tries to get an element of an object as an object, returns
    /// an error if it isn't an object
    ///
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_object<Q>(&self, k: &Q) -> Result<Option<&Self::Object>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}
/// Mutatability for object like values
pub trait MutableObject {
    /// The type for Object Keys
    type Key: ?Sized;
    /// The type for Object Values
    type Target;
    /// The type for Objects
    type Object;
    /// Insert into this `Value` as an `Object`.
    /// Will return an `AccessError::NotAnObject` if called
    /// on a `Value` that isn't an object - otherwise will
    /// behave the same as `HashMap::insert`
    /// # Errors
    ///
    /// Will return `Err` if `self` is not an object.
    fn insert<K, V>(
        &mut self,
        k: K,
        v: V,
    ) -> std::result::Result<Option<Self::Target>, AccessError>
    where
        Self::Key: From<K> + Hash + Eq,
        V: Into<Self::Target>;

    /// Tries to insert into this `Value` as an `Object`.
    /// If the `Value` isn't an object this opoeration will
    /// return `None` and have no effect.
    #[inline]
    fn try_insert<K, V>(&mut self, k: K, v: V) -> Option<Self::Target>
    where
        Self::Key: From<K> + Hash + Eq,
        V: Into<Self::Target>,
    {
        self.insert(k, v).ok().flatten()
    }

    /// Remove from this `Value` as an `Object`.
    /// Will return an `AccessError::NotAnObject` if called
    /// on a `Value` that isn't an object - otherwise will
    /// behave the same as `HashMap::remove`
    /// # Errors
    ///
    /// Will return `Err` if `self` is not an Object.
    fn remove<Q>(&mut self, k: &Q) -> std::result::Result<Option<Self::Target>, AccessError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;

    /// Tries to remove from this `Value` as an `Object`.
    /// If the `Value` isn't an object this opoeration will
    /// return `None` and have no effect.
    #[inline]
    fn try_remove<Q>(&mut self, k: &Q) -> Option<Self::Target>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.remove(k).ok().flatten()
    }

    /// Same as `get` but returns a mutable ref instead
    //    fn get_amut(&mut self, k: &str) -> Option<&mut Self>;
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Target>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}
/// `try_as_*` access to a value as an object
pub trait ValueObjectTryAccess {
    /// The type for Objects
    type Key: ?Sized;
    /// The target for nested lookups
    type Target;
    /// Tries to get a value based on a key, returns a `TryTypeError` if the
    /// current Value isn't an Object, returns `None` if the key isn't in the object
    /// # Errors
    /// if the value is not an object
    fn try_get<Q>(&self, k: &Q) -> Result<Option<&Self::Target>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord;
}

/// Mutatability for array like values
pub trait MutableArray {
    /// The type for Array Values
    type Target;
    /// Pushes to this `Value` as an `Array`.
    /// Will return an `AccessError::NotAnArray` if called
    /// on a `Value` that isn't an `Array` - otherwise will
    /// behave the same as `Vec::push`
    /// # Errors
    ///
    /// Will return `Err` if `self` is not an array.
    fn push<V>(&mut self, v: V) -> std::result::Result<(), AccessError>
    where
        V: Into<Self::Target>;

    /// Tries to push to a `Value` if as an `Array`.
    /// This function will have no effect if `Value` is of
    /// a different type
    #[inline]
    fn try_push<V>(&mut self, v: V)
    where
        V: Into<Self::Target>,
    {
        let _: Result<_, _> = self.push(v);
    }

    /// Pops from this `Value` as an `Array`.
    /// Will return an `AccessError::NotAnArray` if called
    /// on a `Value` that isn't an `Array` - otherwise will
    /// behave the same as `Vec::pop`
    /// # Errors
    ///
    /// Will return `Err` if `self` is not an array.
    fn pop(&mut self) -> std::result::Result<Option<Self::Target>, AccessError>;

    /// Tries to pop from a `Value` as an `Array`.
    /// if the `Value` is any other type `None` will
    /// always be returned
    #[inline]
    fn try_pop(&mut self) -> Option<Self::Target> {
        self.pop().ok().flatten()
    }
}
/// Mutatability for array like values
pub trait MutableValueArrayAccess<I> {
    /// The type for Array Values
    type Target: ?Sized;

    /// Same as `get_idx` but returns a mutable ref instead
    fn get_idx_mut(&mut self, i: I) -> Option<&mut Self::Target>;
}

/// Access to a value as an array with error handling
pub trait ValueArrayTryAccess {
    /// The target for nested lookups
    type Target: ?Sized;
    /// Tries to get a value based on n index, returns a type error if the
    /// current value isn't an Array, returns `None` if the index is out of bounds
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    fn try_get_idx(&self, i: usize) -> Result<Option<&Self::Target>, TryTypeError>;
}

/// A trait that allows destructively turning a value into it's string representation
pub trait ValueTryIntoString {
    /// The type for Strings
    type String;
    /// Tries to turn the value into it's string representation
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_into_string(self) -> Result<Self::String, TryTypeError>;
}

/// A trait that specifies how to turn the Value `into` it's sub types with error handling
pub trait ValueTryIntoArray {
    /// The type for Arrays
    type Array;

    /// Tries to turn the value into it's array representation
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_into_array(self) -> Result<Self::Array, TryTypeError>;
}
/// A trait that specifies how to turn the Value `into` it's sub types with error handling
pub trait ValueTryIntoObject {
    /// The type for Objects
    type Object;

    /// Tries to turn the value into it's object representation
    /// # Errors
    /// if the requested type doesn't match the actual type
    fn try_into_object(self) -> Result<Self::Object, TryTypeError>;
}
