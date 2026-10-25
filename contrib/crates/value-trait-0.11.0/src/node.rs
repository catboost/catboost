use crate::base::{TypedValue, ValueAsScalar};

use super::{ValueType, fmt};
use std::convert::TryFrom;
use std::ops::{Index, IndexMut};

#[cfg(feature = "ordered-float")]
use ordered_float::OrderedFloat;

#[cfg(not(feature = "ordered-float"))]
use float_cmp::approx_eq;

mod cmp;
mod from;

/// Static tape node
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "ordered-float", derive(Eq))]
#[cfg_attr(feature = "c-abi", repr(C))]
#[cfg_attr(feature = "c-abi", derive(abi_stable::StableAbi))]

pub enum StaticNode {
    /// A signed 64 bit integer.
    I64(i64),
    #[cfg(feature = "128bit")]
    /// A signed 128 bit integer.
    I128(i128),
    /// An unsigned 64 bit integer.
    U64(u64),
    #[cfg(feature = "128bit")]
    /// An unsigned 128 bit integer.
    U128(u128),
    /// A floating point value
    #[cfg(not(feature = "ordered-float"))]
    F64(f64),
    #[cfg(feature = "ordered-float")]
    /// A floating point value, as an `OrderedFloat` (not allow for Eq)
    F64(OrderedFloat<f64>),
    /// A boolean value
    Bool(bool),
    /// The null value
    Null,
}

impl Index<&str> for StaticNode {
    type Output = ();
    #[inline]
    #[must_use]
    fn index(&self, _index: &str) -> &Self::Output {
        panic!("Not supported")
    }
}

impl Index<usize> for StaticNode {
    type Output = ();
    #[inline]
    #[must_use]
    fn index(&self, _index: usize) -> &Self::Output {
        panic!("Not supported")
    }
}

impl IndexMut<&str> for StaticNode {
    #[inline]
    #[must_use]
    fn index_mut(&mut self, _index: &str) -> &mut Self::Output {
        panic!("Not supported")
    }
}

impl IndexMut<usize> for StaticNode {
    #[inline]
    #[must_use]
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        panic!("Not supported")
    }
}

impl TypedValue for StaticNode {
    #[cfg(not(feature = "128bit"))]
    #[inline]
    #[must_use]
    fn value_type(&self) -> ValueType {
        match self {
            Self::Null => ValueType::Null,
            Self::Bool(_) => ValueType::Bool,
            Self::F64(_) => ValueType::F64,
            Self::I64(_) => ValueType::I64,

            Self::U64(_) => ValueType::U64,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn value_type(&self) -> ValueType {
        match self {
            Self::Null => ValueType::Null,
            Self::Bool(_) => ValueType::Bool,
            Self::F64(_) => ValueType::F64,
            Self::I128(_) => ValueType::I128,
            Self::I64(_) => ValueType::I64,
            Self::U128(_) => ValueType::U128,
            Self::U64(_) => ValueType::U64,
        }
    }
}

impl ValueAsScalar for StaticNode {
    #[inline]
    #[must_use]
    fn as_null(&self) -> Option<()> {
        match self {
            Self::Null => Some(()),
            _ => None,
        }
    }
    #[inline]
    #[must_use]
    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[cfg(not(feature = "128bit"))]
    #[inline]
    #[must_use]
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(i) => Some(*i),
            Self::U64(i) => i64::try_from(*i).ok(),
            _ => None,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(i) => Some(*i),
            Self::U64(i) => i64::try_from(*i).ok(),
            Self::I128(i) => i64::try_from(*i).ok(),
            Self::U128(i) => i64::try_from(*i).ok(),
            _ => None,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn as_i128(&self) -> Option<i128> {
        match self {
            Self::I128(i) => Some(*i),
            Self::U128(i) => i128::try_from(*i).ok(),
            Self::I64(i) => Some(i128::from(*i)),
            Self::U64(i) => Some(i128::from(*i)),
            _ => None,
        }
    }

    #[cfg(not(feature = "128bit"))]
    #[inline]
    #[must_use]
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::I64(i) => u64::try_from(*i).ok(),
            Self::U64(i) => Some(*i),
            _ => None,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::I64(i) => u64::try_from(*i).ok(),
            Self::U64(i) => Some(*i),
            Self::I128(i) => u64::try_from(*i).ok(),
            Self::U128(i) => u64::try_from(*i).ok(),
            _ => None,
        }
    }
    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn as_u128(&self) -> Option<u128> {
        match self {
            Self::U128(i) => Some(*i),
            Self::I128(i) => u128::try_from(*i).ok(),
            Self::I64(i) => u128::try_from(*i).ok(),
            Self::U64(i) => Some(u128::from(*i)),
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    fn as_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Self::F64(i) => Some((*i).into()),
            _ => None,
        }
    }

    #[cfg(not(feature = "128bit"))]
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn cast_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Self::F64(i) => Some((*i).into()),
            Self::I64(i) => Some(*i as f64),
            Self::U64(i) => Some(*i as f64),
            _ => None,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn cast_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::useless_conversion)] // .into() required by ordered-float
            Self::F64(i) => Some((*i).into()),
            Self::I64(i) => Some(*i as f64),
            Self::U64(i) => Some(*i as f64),
            Self::I128(i) => Some(*i as f64),
            Self::U128(i) => Some(*i as f64),
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    fn as_str(&self) -> Option<&str> {
        None
    }
}

impl fmt::Display for StaticNode {
    #[cfg(not(feature = "128bit"))]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::I64(n) => write!(f, "{n}"),
            Self::U64(n) => write!(f, "{n}"),
            #[cfg(not(feature = "ordered-float"))]
            Self::F64(n) => write!(f, "{n}"),
            #[cfg(feature = "ordered-float")]
            Self::F64(n) => write!(f, "{}", n.0),
        }
    }
    #[cfg(feature = "128bit")]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::I64(n) => write!(f, "{n}"),
            Self::U64(n) => write!(f, "{n}"),
            #[cfg(not(feature = "ordered-float"))]
            Self::F64(n) => write!(f, "{n}"),
            #[cfg(feature = "ordered-float")]
            Self::F64(n) => write!(f, "{}", n.0),
            Self::I128(n) => write!(f, "{n}"),
            Self::U128(n) => write!(f, "{n}"),
        }
    }
}

#[allow(clippy::cast_sign_loss, clippy::default_trait_access)]
impl PartialEq for StaticNode {
    #[cfg(not(feature = "128bit"))]
    #[inline]
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(v1), Self::Bool(v2)) => v1.eq(v2),
            #[cfg(not(feature = "ordered-float"))]
            (Self::F64(v1), Self::F64(v2)) => approx_eq!(f64, *v1, *v2),
            #[cfg(feature = "ordered-float")]
            (Self::F64(v1), Self::F64(v2)) => v1.eq(v2),
            (Self::U64(v1), Self::U64(v2)) => v1.eq(v2),
            (Self::I64(v1), Self::I64(v2)) => v1.eq(v2),
            (Self::U64(v1), Self::I64(v2)) if *v2 >= 0 => (*v2 as u64).eq(v1),
            (Self::I64(v1), Self::U64(v2)) if *v1 >= 0 => (*v1 as u64).eq(v2),
            _ => false,
        }
    }

    #[cfg(feature = "128bit")]
    #[inline]
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(v1), Self::Bool(v2)) => v1.eq(v2),
            #[cfg(not(feature = "ordered-float"))]
            (Self::F64(v1), Self::F64(v2)) => approx_eq!(f64, *v1, *v2),
            #[cfg(feature = "ordered-float")]
            (Self::F64(v1), Self::F64(v2)) => v1.eq(v2),
            (Self::U64(v1), Self::U64(v2)) => v1.eq(v2),
            (Self::U128(v1), Self::U128(v2)) => v1.eq(v2),
            (Self::I64(v1), Self::I64(v2)) => v1.eq(v2),
            (Self::I128(v1), Self::I128(v2)) => v1.eq(v2),

            (Self::U64(v1), Self::I64(v2)) if *v2 >= 0 => (*v2 as u64).eq(v1),
            (Self::U64(v1), Self::I128(v2)) if *v2 >= 0 => (*v2 as u128).eq(&u128::from(*v1)),
            (Self::U64(v1), Self::U128(v2)) => v2.eq(&u128::from(*v1)),

            (Self::I64(v1), Self::U64(v2)) if *v1 >= 0 => (*v1 as u64).eq(v2),
            (Self::I64(v1), Self::I128(v2)) => (*v2).eq(&i128::from(*v1)),
            (Self::I64(v1), Self::U128(v2)) if *v1 >= 0 => v2.eq(&(*v1 as u128)),

            (Self::U128(v1), Self::I128(v2)) if *v2 >= 0 => (*v2 as u128).eq(v1),
            (Self::U128(v1), Self::U64(v2)) => v1.eq(&u128::from(*v2)),
            (Self::U128(v1), Self::I64(v2)) if *v2 >= 0 => v1.eq(&(*v2 as u128)),

            (Self::I128(v1), Self::U128(v2)) if *v1 >= 0 => (*v1 as u128).eq(v2),
            (Self::I128(v1), Self::U64(v2)) => v1.eq(&i128::from(*v2)),
            (Self::I128(v1), Self::I64(v2)) => v1.eq(&i128::from(*v2)),
            _ => false,
        }
    }
}

impl Default for StaticNode {
    #[must_use]
    fn default() -> Self {
        Self::Null
    }
}
