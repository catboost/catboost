mod de;
mod se;

use crate::OwnedValue;
use crate::Result;
use serde_ext::de::DeserializeOwned;
use serde_ext::ser::Serialize;

/// Tries to convert a struct that implements serde's serialize into
/// an `OwnedValue`
///
/// # Errors
///
/// Will return `Err` if value fails to be turned into a owned value
pub fn to_value<T>(value: T) -> Result<OwnedValue>
where
    T: Serialize,
{
    value.serialize(se::Serializer::default())
}

/// Tries to convert a `OwnedValue` into a struct that implements
/// serde's Deserialize interface
///
/// # Errors
///
/// Will return `Err` if `value` fails to be deserialized
pub fn from_value<T>(value: OwnedValue) -> Result<T>
where
    T: DeserializeOwned,
{
    T::deserialize(value)
}

/// Tries to convert a `&OwnedValue` into a struct that implements
/// serde's Deserialize interface
///
/// # Errors
///
/// Will return `Err` if `value` fails to be deserialized
pub fn from_refvalue<T>(value: &OwnedValue) -> Result<T>
where
    T: DeserializeOwned,
{
    T::deserialize(value)
}
