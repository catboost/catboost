use crate::{
    ExtendedValueType, ValueType,
    base::{TypedValue, ValueAsScalar, ValueIntoString},
    prelude::{ValueAsArray, ValueAsObject, ValueIntoArray, ValueIntoObject},
};

impl<V> ValueIntoString for Option<V>
where
    V: ValueIntoString,
{
    type String = V::String;

    fn into_string(self) -> Option<Self::String> {
        self.and_then(ValueIntoString::into_string)
    }
}

impl<V> ValueIntoArray for Option<V>
where
    V: ValueIntoArray,
{
    type Array = V::Array;
    fn into_array(self) -> Option<Self::Array> {
        self.and_then(ValueIntoArray::into_array)
    }
}
impl<V> ValueIntoObject for Option<V>
where
    V: ValueIntoObject,
{
    type Object = V::Object;
    fn into_object(self) -> Option<Self::Object> {
        self.and_then(ValueIntoObject::into_object)
    }
}

impl<V> TypedValue for Option<V>
where
    V: TypedValue,
{
    fn value_type(&self) -> ValueType {
        self.as_ref().map_or(
            ValueType::Extended(ExtendedValueType::None),
            TypedValue::value_type,
        )
    }
}

impl<V> ValueAsScalar for Option<V>
where
    V: ValueAsScalar,
{
    fn as_null(&self) -> Option<()> {
        self.as_ref().and_then(ValueAsScalar::as_null)
    }

    fn as_bool(&self) -> Option<bool> {
        self.as_ref().and_then(ValueAsScalar::as_bool)
    }

    fn as_i64(&self) -> Option<i64> {
        self.as_ref().and_then(ValueAsScalar::as_i64)
    }

    fn as_u64(&self) -> Option<u64> {
        self.as_ref().and_then(ValueAsScalar::as_u64)
    }

    fn as_f64(&self) -> Option<f64> {
        self.as_ref().and_then(ValueAsScalar::as_f64)
    }

    fn as_str(&self) -> Option<&str> {
        self.as_ref().and_then(ValueAsScalar::as_str)
    }
}

impl<V> ValueAsArray for Option<V>
where
    V: ValueAsArray,
{
    type Array = V::Array;

    fn as_array(&self) -> Option<&Self::Array> {
        self.as_ref().and_then(ValueAsArray::as_array)
    }
}
impl<V> ValueAsObject for Option<V>
where
    V: ValueAsObject,
{
    type Object = V::Object;

    fn as_object(&self) -> Option<&Self::Object> {
        self.as_ref().and_then(ValueAsObject::as_object)
    }
}

impl<V, E> ValueIntoString for Result<V, E>
where
    V: ValueIntoString,
{
    type String = V::String;

    fn into_string(self) -> Option<Self::String> {
        self.ok().and_then(ValueIntoString::into_string)
    }
}

impl<V, E> ValueIntoArray for Result<V, E>
where
    V: ValueIntoArray,
{
    type Array = V::Array;
    fn into_array(self) -> Option<Self::Array> {
        self.ok().and_then(ValueIntoArray::into_array)
    }
}
impl<V, E> ValueIntoObject for Result<V, E>
where
    V: ValueIntoObject,
{
    type Object = V::Object;
    fn into_object(self) -> Option<Self::Object> {
        self.ok().and_then(ValueIntoObject::into_object)
    }
}

impl<V, E> TypedValue for Result<V, E>
where
    V: TypedValue,
{
    fn value_type(&self) -> ValueType {
        self.as_ref().ok().map_or(
            ValueType::Extended(ExtendedValueType::None),
            TypedValue::value_type,
        )
    }
}

impl<V, E> ValueAsScalar for Result<V, E>
where
    V: ValueAsScalar,
{
    fn as_null(&self) -> Option<()> {
        self.as_ref().ok().and_then(ValueAsScalar::as_null)
    }
    fn as_bool(&self) -> Option<bool> {
        self.as_ref().ok().and_then(ValueAsScalar::as_bool)
    }

    fn as_i64(&self) -> Option<i64> {
        self.as_ref().ok().and_then(ValueAsScalar::as_i64)
    }

    fn as_u64(&self) -> Option<u64> {
        self.as_ref().ok().and_then(ValueAsScalar::as_u64)
    }

    fn as_f64(&self) -> Option<f64> {
        self.as_ref().ok().and_then(ValueAsScalar::as_f64)
    }

    fn as_str(&self) -> Option<&str> {
        self.as_ref().ok().and_then(ValueAsScalar::as_str)
    }
}
impl<V, E> ValueAsArray for Result<V, E>
where
    V: ValueAsArray,
{
    type Array = V::Array;

    fn as_array(&self) -> Option<&Self::Array> {
        self.as_ref().ok().and_then(ValueAsArray::as_array)
    }
}
impl<V, E> ValueAsObject for Result<V, E>
where
    V: ValueAsObject,
{
    type Object = V::Object;

    fn as_object(&self) -> Option<&Self::Object> {
        self.as_ref().ok().and_then(ValueAsObject::as_object)
    }
}

impl<V> TypedValue for &V
where
    V: TypedValue,
{
    fn value_type(&self) -> ValueType {
        (*self).value_type()
    }
}
impl<V> ValueAsScalar for &V
where
    V: ValueAsScalar,
{
    fn as_null(&self) -> Option<()> {
        (*self).as_null()
    }

    fn as_bool(&self) -> Option<bool> {
        (*self).as_bool()
    }

    fn as_i64(&self) -> Option<i64> {
        (*self).as_i64()
    }

    fn as_u64(&self) -> Option<u64> {
        (*self).as_u64()
    }

    fn as_f64(&self) -> Option<f64> {
        (*self).as_f64()
    }

    fn as_str(&self) -> Option<&str> {
        (*self).as_str()
    }
}

impl<V> ValueAsArray for &V
where
    V: ValueAsArray,
{
    type Array = V::Array;

    fn as_array(&self) -> Option<&Self::Array> {
        (*self).as_array()
    }
}
impl<V> ValueAsObject for &V
where
    V: ValueAsObject,
{
    type Object = V::Object;

    fn as_object(&self) -> Option<&Self::Object> {
        (*self).as_object()
    }
}
