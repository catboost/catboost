// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(feature = "std")]
use num_traits::Float;
use num_traits::{One, Zero};

#[cfg(feature = "std")]
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
#[cfg(feature = "std")]
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

#[cfg(feature = "std")]
use crate::ScalarOperand;

/// Elements that support linear algebra operations.
///
/// `'static` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait LinalgScalar:
    'static + Copy + Zero + One + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
}

impl<T> LinalgScalar for T where T: 'static + Copy + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{}

/// Floating-point element types `f32` and `f64`.
///
/// Trait `NdFloat` is only implemented for `f32` and `f64` but encompasses as
/// much float-relevant ndarray functionality as possible, including the traits
/// needed for linear algebra and for *right hand side* scalar
/// operations (`ScalarOperand`).
///
/// This trait can only be implemented by `f32` and `f64`.
#[cfg(feature = "std")]
pub trait NdFloat:
    Float
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + fmt::Display
    + fmt::Debug
    + fmt::LowerExp
    + fmt::UpperExp
    + ScalarOperand
    + LinalgScalar
    + Send
    + Sync
{
}

#[cfg(feature = "std")]
impl NdFloat for f32 {}
#[cfg(feature = "std")]
impl NdFloat for f64 {}
