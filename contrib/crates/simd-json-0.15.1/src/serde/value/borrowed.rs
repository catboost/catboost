mod de;
mod se;

use crate::{BorrowedValue, Result};
use serde_ext::de::Deserialize;
use serde_ext::ser::Serialize;

/// Tries to convert a struct that implements serde's serialize into
/// an `BorrowedValue`
///
/// # Errors
///
/// Will return `Err` if value fails to be turned into a borrowed value
pub fn to_value<'se, T>(value: T) -> Result<BorrowedValue<'se>>
where
    T: Serialize,
{
    value.serialize(se::Serializer::default())
}

/// Tries to convert a `BorrowedValue` into a struct that implements
/// serde's Deserialize interface
///
/// # Errors
///
/// Will return `Err` if `value` can not be deserialized
pub fn from_value<'de, T>(value: BorrowedValue<'de>) -> Result<T>
where
    T: Deserialize<'de>,
{
    T::deserialize(value)
}

/// Tries to convert a `&BorrowedValue` into a struct that implements
/// serde's Deserialize interface
///
/// # Errors
///
/// Will return `Err` if `value` fails to be deserialized
pub fn from_refvalue<'de, T>(value: &'de BorrowedValue<'de>) -> Result<T>
where
    T: Deserialize<'de>,
{
    T::deserialize(value)
}
