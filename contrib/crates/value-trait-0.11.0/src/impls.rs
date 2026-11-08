use std::{borrow::Borrow, hash::Hash};

use crate::{
    AccessError, ExtendedValueType, TryTypeError, ValueType,
    array::{self, ArrayMut, Indexed, IndexedMut},
    base::{TypedValue, ValueAsScalar, ValueIntoString, ValueTryAsArrayMut, ValueTryAsMutObject},
    derived::{
        MutableArray, MutableObject, MutableValueArrayAccess, TypedArrayValue, TypedObjectValue,
        TypedScalarValue, ValueArrayAccess, ValueArrayTryAccess, ValueObjectAccess,
        ValueObjectAccessAsArray, ValueObjectAccessAsObject, ValueObjectAccessAsScalar,
        ValueObjectAccessTryAsArray, ValueObjectAccessTryAsObject, ValueObjectAccessTryAsScalar,
        ValueObjectTryAccess, ValueTryAsArray, ValueTryAsObject, ValueTryAsScalar,
        ValueTryIntoArray, ValueTryIntoObject, ValueTryIntoString,
    },
    object::{Object, ObjectMut},
    prelude::{
        ValueAsArray, ValueAsMutArray, ValueAsMutObject, ValueAsObject, ValueIntoArray,
        ValueIntoObject,
    },
};

impl<T> ValueTryIntoString for T
where
    T: ValueIntoString + TypedValue,
{
    type String = T::String;

    #[inline]
    fn try_into_string(self) -> Result<Self::String, TryTypeError> {
        let vt = self.value_type();
        self.into_string().ok_or(TryTypeError {
            expected: ValueType::String,
            got: vt,
        })
    }
}

impl<T> ValueTryIntoArray for T
where
    T: ValueIntoArray + TypedValue,
{
    type Array = T::Array;
    /// Tries to turn the value into it's array representation
    /// # Errors
    /// if the requested type doesn't match the actual type
    #[inline]
    fn try_into_array(self) -> Result<Self::Array, TryTypeError> {
        let vt = self.value_type();
        self.into_array().ok_or(TryTypeError {
            expected: ValueType::Array,
            got: vt,
        })
    }
}
impl<T> ValueTryIntoObject for T
where
    T: ValueIntoObject + TypedValue,
{
    type Object = T::Object;

    /// Tries to turn the value into it's object representation
    /// # Errors
    /// if the requested type doesn't match the actual type
    #[inline]
    fn try_into_object(self) -> Result<Self::Object, TryTypeError> {
        let vt = self.value_type();
        self.into_object().ok_or(TryTypeError {
            expected: ValueType::Object,
            got: vt,
        })
    }
}

impl<T> ValueTryAsScalar for T
where
    T: ValueAsScalar + TypedValue,
{
    #[inline]
    fn try_as_bool(&self) -> Result<bool, TryTypeError> {
        self.as_bool().ok_or(TryTypeError {
            expected: ValueType::Bool,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_i128(&self) -> Result<i128, TryTypeError> {
        self.as_i128().ok_or(TryTypeError {
            expected: ValueType::I128,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_i64(&self) -> Result<i64, TryTypeError> {
        self.as_i64().ok_or(TryTypeError {
            expected: ValueType::I64,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_i32(&self) -> Result<i32, TryTypeError> {
        self.as_i32().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::I32),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_i16(&self) -> Result<i16, TryTypeError> {
        self.as_i16().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::I16),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_i8(&self) -> Result<i8, TryTypeError> {
        self.as_i8().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::I8),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_u128(&self) -> Result<u128, TryTypeError> {
        self.as_u128().ok_or(TryTypeError {
            expected: ValueType::U128,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_u64(&self) -> Result<u64, TryTypeError> {
        self.as_u64().ok_or(TryTypeError {
            expected: ValueType::U64,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_usize(&self) -> Result<usize, TryTypeError> {
        self.as_usize().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::Usize),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_u32(&self) -> Result<u32, TryTypeError> {
        self.as_u32().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::U32),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_u16(&self) -> Result<u16, TryTypeError> {
        self.as_u16().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::U16),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_u8(&self) -> Result<u8, TryTypeError> {
        self.as_u8().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::U8),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_f64(&self) -> Result<f64, TryTypeError> {
        self.as_f64().ok_or(TryTypeError {
            expected: ValueType::F64,
            got: self.value_type(),
        })
    }

    #[inline]
    #[allow(clippy::cast_precision_loss, clippy::option_if_let_else)]
    fn try_cast_f64(&self) -> Result<f64, TryTypeError> {
        if let Some(f) = self.as_f64() {
            Ok(f)
        } else if let Some(u) = self.as_u128() {
            Ok(u as f64)
        } else {
            self.try_as_i128().map(|i| i as f64)
        }
    }

    #[inline]
    fn try_as_f32(&self) -> Result<f32, TryTypeError> {
        self.as_f32().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::F32),
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_str(&self) -> Result<&str, TryTypeError> {
        self.as_str().ok_or(TryTypeError {
            expected: ValueType::String,
            got: self.value_type(),
        })
    }

    #[inline]
    fn try_as_char(&self) -> Result<char, TryTypeError> {
        self.as_char().ok_or(TryTypeError {
            expected: ValueType::Extended(ExtendedValueType::Char),
            got: self.value_type(),
        })
    }
}

impl<T> ValueTryAsArray for T
where
    T: ValueAsArray + TypedValue,
{
    type Array = T::Array;
    #[inline]
    fn try_as_array(&self) -> Result<&Self::Array, TryTypeError> {
        self.as_array().ok_or(TryTypeError {
            expected: ValueType::Array,
            got: self.value_type(),
        })
    }
}
impl<T> ValueTryAsArrayMut for T
where
    T: ValueAsMutArray + TypedValue,
    <T as ValueAsMutArray>::Array: array::Array,
{
    type Array = T::Array;
    #[inline]
    fn try_as_array_mut(&mut self) -> Result<&mut Self::Array, TryTypeError> {
        let got = self.value_type();
        self.as_array_mut().ok_or(TryTypeError {
            expected: ValueType::Array,
            got,
        })
    }
}

impl<T> ValueTryAsObject for T
where
    T: ValueAsObject + TypedValue,
{
    type Object = T::Object;

    #[inline]
    fn try_as_object(&self) -> Result<&Self::Object, TryTypeError> {
        self.as_object().ok_or(TryTypeError {
            expected: ValueType::Object,
            got: self.value_type(),
        })
    }
}
impl<T> ValueTryAsMutObject for T
where
    T: ValueAsMutObject + TypedValue,
{
    type Object = T::Object;

    #[inline]
    fn try_as_object_mut(&mut self) -> Result<&mut Self::Object, TryTypeError> {
        let got = self.value_type();
        self.as_object_mut().ok_or(TryTypeError {
            expected: ValueType::Object,
            got,
        })
    }
}

impl<T> ValueObjectAccess for T
where
    T: ValueAsObject,
{
    type Key = <T::Object as Object>::Key;
    type Target = <T::Object as Object>::Element;
    #[inline]
    #[must_use]
    fn get<Q>(&self, k: &Q) -> Option<&Self::Target>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.as_object().and_then(|a| a.get(k))
    }

    #[inline]
    #[must_use]
    fn contains_key<Q>(&self, k: &Q) -> bool
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.as_object().and_then(|a| a.get(k)).is_some()
    }
}

impl<T> ValueObjectTryAccess for T
where
    T: ValueTryAsObject,
{
    type Key = <T::Object as Object>::Key;
    type Target = <T::Object as Object>::Element;

    #[inline]
    fn try_get<Q>(&self, k: &Q) -> Result<Option<&Self::Target>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        Ok(self.try_as_object()?.get(k))
    }
}

impl<I, T> ValueArrayAccess<I> for T
where
    T: ValueAsArray,
    <T as ValueAsArray>::Array: Indexed<I>,
{
    type Target = <<T as ValueAsArray>::Array as Indexed<I>>::Element;
    #[inline]
    #[must_use]
    fn get_idx(&self, i: I) -> Option<&Self::Target> {
        self.as_array().and_then(|a| a.get(i))
    }
}

impl<T> ValueArrayTryAccess for T
where
    T: ValueTryAsArray,
    <T as ValueTryAsArray>::Array: Indexed<usize>,
{
    type Target = <<T as ValueTryAsArray>::Array as Indexed<usize>>::Element;

    /// Tries to get a value based on n index, returns a type error if the
    /// current value isn't an Array, returns `None` if the index is out of bounds
    /// # Errors
    /// if the requested type doesn't match the actual type or the value is not an object
    #[inline]
    fn try_get_idx(&self, i: usize) -> Result<Option<&Self::Target>, TryTypeError> {
        Ok(self.try_as_array()?.get(i))
    }
}

impl<T> ValueObjectAccessAsScalar for T
where
    T: ValueObjectAccess,
    <T as ValueObjectAccess>::Target: ValueAsScalar,
{
    type Key = T::Key;
    #[inline]
    #[must_use]
    fn get_bool<Q>(&self, k: &Q) -> Option<bool>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_bool)
    }

    #[inline]
    #[must_use]
    fn get_i128<Q>(&self, k: &Q) -> Option<i128>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_i128)
    }

    #[inline]
    #[must_use]
    fn get_i64<Q>(&self, k: &Q) -> Option<i64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_i64)
    }

    #[inline]
    #[must_use]
    fn get_i32<Q>(&self, k: &Q) -> Option<i32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_i32)
    }

    #[inline]
    #[must_use]
    fn get_i16<Q>(&self, k: &Q) -> Option<i16>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_i16)
    }

    #[inline]
    #[must_use]
    fn get_i8<Q>(&self, k: &Q) -> Option<i8>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_i8)
    }

    #[inline]
    #[must_use]
    fn get_u128<Q>(&self, k: &Q) -> Option<u128>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_u128)
    }

    #[inline]
    #[must_use]
    fn get_u64<Q>(&self, k: &Q) -> Option<u64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_u64)
    }

    #[inline]
    #[must_use]
    fn get_usize<Q>(&self, k: &Q) -> Option<usize>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_usize)
    }

    #[inline]
    #[must_use]
    fn get_u32<Q>(&self, k: &Q) -> Option<u32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_u32)
    }

    #[inline]
    #[must_use]
    fn get_u16<Q>(&self, k: &Q) -> Option<u16>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_u16)
    }

    #[inline]
    #[must_use]
    fn get_u8<Q>(&self, k: &Q) -> Option<u8>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_u8)
    }

    #[inline]
    #[must_use]
    fn get_f64<Q>(&self, k: &Q) -> Option<f64>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_f64)
    }

    #[inline]
    #[must_use]
    fn get_f32<Q>(&self, k: &Q) -> Option<f32>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_f32)
    }

    #[inline]
    #[must_use]
    fn get_str<Q>(&self, k: &Q) -> Option<&str>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsScalar::as_str)
    }
}

impl<T> ValueObjectAccessAsArray for T
where
    T: ValueObjectAccess,
    T::Target: ValueAsArray,
{
    type Key = T::Key;

    type Array = <T::Target as ValueAsArray>::Array;

    #[inline]
    #[must_use]
    fn get_array<Q>(&self, k: &Q) -> Option<&Self::Array>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsArray::as_array)
    }
}

impl<T> ValueObjectAccessAsObject for T
where
    T: ValueObjectAccess,
    T::Target: ValueAsObject,
{
    type Key = T::Key;
    type Object = <T::Target as ValueAsObject>::Object;

    #[inline]
    #[must_use]
    fn get_object<Q>(&self, k: &Q) -> Option<&Self::Object>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.get(k).and_then(ValueAsObject::as_object)
    }
}

impl<T> ValueObjectAccessTryAsArray for T
where
    T: ValueObjectTryAccess + TypedValue,
    T::Target: ValueTryAsArray,
{
    type Key = T::Key;

    type Array = <T::Target as ValueTryAsArray>::Array;

    #[inline]
    fn try_get_array<Q>(&self, k: &Q) -> Result<Option<&Self::Array>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)
            .and_then(|s| s.map(ValueTryAsArray::try_as_array).transpose())
    }
}
impl<T> ValueObjectAccessTryAsObject for T
where
    T: ValueObjectTryAccess + TypedValue,
    T::Target: ValueTryAsObject,
{
    type Key = T::Key;

    type Object = <T::Target as ValueTryAsObject>::Object;

    #[inline]
    fn try_get_object<Q>(&self, k: &Q) -> Result<Option<&Self::Object>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)
            .and_then(|s| s.map(ValueTryAsObject::try_as_object).transpose())
    }
}

impl<T> ValueObjectAccessTryAsScalar for T
where
    T: ValueObjectTryAccess + TypedValue,
    T::Target: ValueTryAsScalar,
{
    type Key = T::Key;

    fn try_get_bool<Q>(&self, k: &Q) -> Result<Option<bool>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_bool)
            .transpose()
    }

    fn try_get_i128<Q>(&self, k: &Q) -> Result<Option<i128>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_i128)
            .transpose()
    }

    fn try_get_i64<Q>(&self, k: &Q) -> Result<Option<i64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_i64)
            .transpose()
    }

    fn try_get_i32<Q>(&self, k: &Q) -> Result<Option<i32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_i32)
            .transpose()
    }

    fn try_get_i16<Q>(&self, k: &Q) -> Result<Option<i16>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_i16)
            .transpose()
    }

    fn try_get_i8<Q>(&self, k: &Q) -> Result<Option<i8>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_i8)
            .transpose()
    }

    fn try_get_u128<Q>(&self, k: &Q) -> Result<Option<u128>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_u128)
            .transpose()
    }

    fn try_get_u64<Q>(&self, k: &Q) -> Result<Option<u64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_u64)
            .transpose()
    }

    fn try_get_usize<Q>(&self, k: &Q) -> Result<Option<usize>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_usize)
            .transpose()
    }

    fn try_get_u32<Q>(&self, k: &Q) -> Result<Option<u32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_u32)
            .transpose()
    }

    fn try_get_u16<Q>(&self, k: &Q) -> Result<Option<u16>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_u16)
            .transpose()
    }

    fn try_get_u8<Q>(&self, k: &Q) -> Result<Option<u8>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_u8)
            .transpose()
    }

    fn try_get_f64<Q>(&self, k: &Q) -> Result<Option<f64>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_f64)
            .transpose()
    }

    fn try_get_f32<Q>(&self, k: &Q) -> Result<Option<f32>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_f32)
            .transpose()
    }

    fn try_get_str<Q>(&self, k: &Q) -> Result<Option<&str>, TryTypeError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.try_get(k)?
            .map(ValueTryAsScalar::try_as_str)
            .transpose()
    }
}

impl<T> TypedScalarValue for T
where
    T: ValueAsScalar,
{
    #[inline]
    #[must_use]
    fn is_null(&self) -> bool {
        self.as_null().is_some()
    }

    #[inline]
    #[must_use]
    fn is_float(&self) -> bool {
        self.is_f64()
    }

    #[inline]
    #[must_use]
    fn is_integer(&self) -> bool {
        self.is_i128() || self.is_u128()
    }

    #[inline]
    #[must_use]
    fn is_number(&self) -> bool {
        self.is_float() || self.is_integer()
    }

    #[inline]
    #[must_use]
    fn is_bool(&self) -> bool {
        self.as_bool().is_some()
    }

    #[inline]
    #[must_use]
    fn is_i128(&self) -> bool {
        self.as_i128().is_some()
    }

    #[inline]
    #[must_use]
    fn is_i64(&self) -> bool {
        self.as_i64().is_some()
    }

    #[inline]
    #[must_use]
    fn is_i32(&self) -> bool {
        self.as_i32().is_some()
    }

    #[inline]
    #[must_use]
    fn is_i16(&self) -> bool {
        self.as_i16().is_some()
    }

    #[inline]
    #[must_use]
    fn is_i8(&self) -> bool {
        self.as_i8().is_some()
    }

    #[inline]
    #[must_use]
    fn is_u128(&self) -> bool {
        self.as_u128().is_some()
    }

    #[inline]
    #[must_use]
    fn is_u64(&self) -> bool {
        self.as_u64().is_some()
    }

    #[inline]
    #[must_use]
    fn is_usize(&self) -> bool {
        self.as_usize().is_some()
    }

    #[inline]
    #[must_use]
    fn is_u32(&self) -> bool {
        self.as_u32().is_some()
    }

    #[inline]
    #[must_use]
    fn is_u16(&self) -> bool {
        self.as_u16().is_some()
    }

    #[inline]
    #[must_use]
    fn is_u8(&self) -> bool {
        self.as_u8().is_some()
    }

    #[inline]
    #[must_use]
    fn is_f64(&self) -> bool {
        self.as_f64().is_some()
    }

    #[inline]
    #[must_use]
    fn is_f64_castable(&self) -> bool {
        self.cast_f64().is_some()
    }

    #[inline]
    #[must_use]
    fn is_f32(&self) -> bool {
        self.as_f32().is_some()
    }

    #[inline]
    #[must_use]
    fn is_str(&self) -> bool {
        self.as_str().is_some()
    }

    #[inline]
    #[must_use]
    fn is_char(&self) -> bool {
        self.as_char().is_some()
    }
}

impl<T> TypedArrayValue for T
where
    T: ValueAsArray,
{
    #[inline]
    #[must_use]
    fn is_array(&self) -> bool {
        self.as_array().is_some()
    }
}
impl<T> TypedObjectValue for T
where
    T: ValueAsObject,
{
    #[inline]
    #[must_use]
    fn is_object(&self) -> bool {
        self.as_object().is_some()
    }
}
impl<T> MutableObject for T
where
    T: ValueAsMutObject,
    T::Object: ObjectMut,
{
    type Key = <T::Object as ObjectMut>::Key;
    type Target = <T::Object as ObjectMut>::Element;
    type Object = T::Object;

    #[inline]
    fn insert<K, V>(&mut self, k: K, v: V) -> std::result::Result<Option<Self::Target>, AccessError>
    where
        Self::Key: From<K> + Hash + Eq,
        V: Into<Self::Target>,
    {
        self.as_object_mut()
            .ok_or(AccessError::NotAnObject)
            .map(|o| o.insert(k, v))
    }

    #[inline]
    fn remove<Q>(&mut self, k: &Q) -> std::result::Result<Option<Self::Target>, AccessError>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.as_object_mut()
            .ok_or(AccessError::NotAnObject)
            .map(|o| o.remove(k))
    }
    #[inline]
    fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut Self::Target>
    where
        Self::Key: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        self.as_object_mut().and_then(|m| m.get_mut(k))
    }
}

impl<T> MutableArray for T
where
    T: ValueAsMutArray,
    T::Array: ArrayMut,
{
    type Target = <T::Array as ArrayMut>::Element;

    #[inline]
    fn push<V>(&mut self, v: V) -> std::result::Result<(), AccessError>
    where
        V: Into<Self::Target>,
    {
        self.as_array_mut()
            .ok_or(AccessError::NotAnArray)
            .map(|o| o.push(v.into()))
    }

    /// Pops from this `Value` as an `Array`.
    /// Will return an `AccessError::NotAnArray` if called
    /// on a `Value` that isn't an `Array` - otherwise will
    /// behave the same as `Vec::pop`
    /// # Errors
    ///
    /// Will return `Err` if `self` is not an array.
    #[inline]
    fn pop(&mut self) -> std::result::Result<Option<Self::Target>, AccessError> {
        self.as_array_mut()
            .ok_or(AccessError::NotAnArray)
            .map(ArrayMut::pop)
    }
}

impl<T, I> MutableValueArrayAccess<I> for T
where
    T: ValueAsMutArray,
    T::Array: IndexedMut<I>,
{
    type Target = <T::Array as IndexedMut<I>>::Element;

    /// Same as `get_idx` but returns a mutable ref instead
    #[inline]
    fn get_idx_mut(&mut self, i: I) -> Option<&mut Self::Target> {
        self.as_array_mut().and_then(|a| a.get_mut(i))
    }
}
