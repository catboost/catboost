// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// An axis index.
///
/// An axis one of an array’s “dimensions”; an *n*-dimensional array has *n*
/// axes.  Axis *0* is the array’s outermost axis and *n*-1 is the innermost.
///
/// All array axis arguments use this type to make the code easier to write
/// correctly and easier to understand.
///
/// For example: in a method like `index_axis(axis, index)` the code becomes
/// self-explanatory when it's called like `.index_axis(Axis(1), i)`; it's
/// evident which integer is the axis number and which is the index.
///
/// Note: This type does **not** implement From/Into usize and similar trait
/// based conversions, because we want to preserve code readability and quality.
///
/// `Axis(1)` in itself is a very clear code style and the style that should be
/// avoided is code like `1.into()`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Axis(pub usize);

impl Axis
{
    /// Return the index of the axis.
    #[inline(always)]
    pub fn index(self) -> usize
    {
        self.0
    }
}
