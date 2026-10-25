// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tuple to array conversion, IntoDimension, and related things

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use num_traits::Zero;
use std::ops::{Index, IndexMut};

use crate::{Dim, Dimension, Ix, Ix1, IxDyn, IxDynImpl};

/// $m: macro callback
/// $m is called with $arg and then the indices corresponding to the size argument
macro_rules! index {
    ($m:ident $arg:tt 0) => ($m!($arg));
    ($m:ident $arg:tt 1) => ($m!($arg 0));
    ($m:ident $arg:tt 2) => ($m!($arg 0 1));
    ($m:ident $arg:tt 3) => ($m!($arg 0 1 2));
    ($m:ident $arg:tt 4) => ($m!($arg 0 1 2 3));
    ($m:ident $arg:tt 5) => ($m!($arg 0 1 2 3 4));
    ($m:ident $arg:tt 6) => ($m!($arg 0 1 2 3 4 5));
    ($m:ident $arg:tt 7) => ($m!($arg 0 1 2 3 4 5 6));
}

macro_rules! index_item {
    ($m:ident $arg:tt 0) => ();
    ($m:ident $arg:tt 1) => ($m!($arg 0););
    ($m:ident $arg:tt 2) => ($m!($arg 0 1););
    ($m:ident $arg:tt 3) => ($m!($arg 0 1 2););
    ($m:ident $arg:tt 4) => ($m!($arg 0 1 2 3););
    ($m:ident $arg:tt 5) => ($m!($arg 0 1 2 3 4););
    ($m:ident $arg:tt 6) => ($m!($arg 0 1 2 3 4 5););
    ($m:ident $arg:tt 7) => ($m!($arg 0 1 2 3 4 5 6););
}

/// Argument conversion a dimension.
pub trait IntoDimension
{
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}

impl IntoDimension for Ix
{
    type Dim = Ix1;
    #[inline(always)]
    fn into_dimension(self) -> Ix1
    {
        Ix1(self)
    }
}

impl<D> IntoDimension for D
where D: Dimension
{
    type Dim = D;
    #[inline(always)]
    fn into_dimension(self) -> Self
    {
        self
    }
}

impl IntoDimension for IxDynImpl
{
    type Dim = IxDyn;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim
    {
        Dim::new(self)
    }
}

impl IntoDimension for Vec<Ix>
{
    type Dim = IxDyn;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim
    {
        Dim::new(IxDynImpl::from(self))
    }
}

pub trait Convert
{
    type To;
    fn convert(self) -> Self::To;
}

macro_rules! sub {
    ($_x:tt $y:tt) => {
        $y
    };
}

macro_rules! tuple_type {
    ([$T:ident] $($index:tt)*) => (
        ( $(sub!($index $T), )* )
    );
}

macro_rules! tuple_expr {
    ([$self_:expr] $($index:tt)*) => (
        ( $($self_[$index], )* )
    );
}

macro_rules! array_expr {
    ([$self_:expr] $($index:tt)*) => (
        [$($self_ . $index, )*]
    );
}

macro_rules! array_zero {
    ([] $($index:tt)*) => (
        [$(sub!($index 0), )*]
    );
}

macro_rules! tuple_to_array {
    ([] $($n:tt)*) => {
        $(
        impl Convert for [Ix; $n] {
            type To = index!(tuple_type [Ix] $n);
            #[inline]
            fn convert(self) -> Self::To {
                index!(tuple_expr [self] $n)
            }
        }

        impl IntoDimension for [Ix; $n] {
            type Dim = Dim<[Ix; $n]>;
            #[inline(always)]
            fn into_dimension(self) -> Self::Dim {
                Dim::new(self)
            }
        }

        impl IntoDimension for index!(tuple_type [Ix] $n) {
            type Dim = Dim<[Ix; $n]>;
            #[inline(always)]
            fn into_dimension(self) -> Self::Dim {
                Dim::new(index!(array_expr [self] $n))
            }
        }

        impl Index<usize> for Dim<[Ix; $n]> {
            type Output = usize;
            #[inline(always)]
            fn index(&self, index: usize) -> &Self::Output {
                &self.ix()[index]
            }
        }

        impl IndexMut<usize> for Dim<[Ix; $n]> {
            #[inline(always)]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.ixm()[index]
            }
        }

        impl Zero for Dim<[Ix; $n]> {
            #[inline]
            fn zero() -> Self {
                Dim::new(index!(array_zero [] $n))
            }
            fn is_zero(&self) -> bool {
                self.slice().iter().all(|x| *x == 0)
            }
        }

        )*
    };
}

index_item!(tuple_to_array [] 7);
