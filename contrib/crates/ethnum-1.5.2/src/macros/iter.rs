//! Module containing macros for implementing iterator specific traits.

macro_rules! impl_iter {
    (
        impl Iter for $int:ident;
    ) => {
        impl ::core::iter::Sum for $int {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($int::ZERO, ::core::ops::Add::add)
            }
        }

        impl ::core::iter::Product for $int {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($int::ONE, ::core::ops::Mul::mul)
            }
        }

        impl<'a> ::core::iter::Sum<&'a $int> for $int {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold($int::ZERO, ::core::ops::Add::add)
            }
        }

        impl<'a> ::core::iter::Product<&'a $int> for $int {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold($int::ONE, ::core::ops::Mul::mul)
            }
        }
    };
}
