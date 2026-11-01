#[cfg(feature = "ordered-float")]
use ordered_float::OrderedFloat;

use crate::StaticNode;

/********* atoms **********/

impl From<bool> for StaticNode {
    #[inline]
    #[must_use]
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<()> for StaticNode {
    #[inline]
    #[must_use]
    fn from(_b: ()) -> Self {
        Self::Null
    }
}

/********* i_ **********/
impl From<i8> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: i8) -> Self {
        Self::I64(i64::from(i))
    }
}

impl From<i16> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: i16) -> Self {
        Self::I64(i64::from(i))
    }
}

impl From<i32> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: i32) -> Self {
        Self::I64(i64::from(i))
    }
}

impl From<i64> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: i64) -> Self {
        Self::I64(i)
    }
}
#[cfg(feature = "128bit")]
impl From<i128> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: i128) -> Self {
        Self::I128(i)
    }
}

/********* u_ **********/
impl From<u8> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: u8) -> Self {
        Self::U64(u64::from(i))
    }
}

impl From<u16> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: u16) -> Self {
        Self::U64(u64::from(i))
    }
}

impl From<u32> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: u32) -> Self {
        Self::U64(u64::from(i))
    }
}

impl From<u64> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: u64) -> Self {
        Self::U64(i)
    }
}

#[cfg(feature = "128bit")]
impl From<u128> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: u128) -> Self {
        Self::U128(i)
    }
}

impl From<usize> for StaticNode {
    #[inline]
    #[must_use]
    fn from(i: usize) -> Self {
        Self::U64(i as u64)
    }
}

/********* f_ **********/
#[cfg(not(feature = "ordered-float"))]
impl From<f32> for StaticNode {
    #[inline]
    #[must_use]
    fn from(f: f32) -> Self {
        Self::F64(f64::from(f))
    }
}

#[cfg(not(feature = "ordered-float"))]
impl From<f64> for StaticNode {
    #[inline]
    #[must_use]
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

#[cfg(feature = "ordered-float")]
impl From<f32> for StaticNode {
    #[inline]
    #[must_use]
    fn from(f: f32) -> Self {
        Self::F64(OrderedFloat::from(f64::from(f)))
    }
}

#[cfg(feature = "ordered-float")]
impl From<f64> for StaticNode {
    #[inline]
    #[must_use]
    fn from(f: f64) -> Self {
        Self::F64(OrderedFloat::from(f))
    }
}
