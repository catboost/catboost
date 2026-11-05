//! Module containing macros for implementing string to integer parsing.

macro_rules! impl_from_str {
    (impl FromStr for $int:ident;) => {
        impl $crate::parse::FromStrRadixHelper for $int {
            const MIN: Self = Self::MIN;
            #[inline]
            fn from_u32(u: u32) -> Self {
                Self::from(u)
            }
            #[inline]
            fn checked_mul(&self, other: u32) -> Option<Self> {
                Self::checked_mul(*self, Self::from(other))
            }
            #[inline]
            fn checked_sub(&self, other: u32) -> Option<Self> {
                Self::checked_sub(*self, Self::from(other))
            }
            #[inline]
            fn checked_add(&self, other: u32) -> Option<Self> {
                Self::checked_add(*self, Self::from(other))
            }
        }

        impl ::core::str::FromStr for $int {
            type Err = ::core::num::ParseIntError;

            #[inline]
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                $crate::parse::from_str_radix(s, 10, None)
            }
        }
    };
}
