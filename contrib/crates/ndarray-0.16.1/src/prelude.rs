// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ndarray prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! use ndarray::prelude::*;
//!
//! # let _ = arr0(1); // use the import
//! ```

#[doc(no_inline)]
pub use crate::{ArcArray, Array, ArrayBase, ArrayView, ArrayViewMut, CowArray, RawArrayView, RawArrayViewMut};

#[doc(no_inline)]
pub use crate::{Axis, Dim, Dimension};

#[doc(no_inline)]
pub use crate::{Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};

#[doc(no_inline)]
pub use crate::{ArrayView0, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView5, ArrayView6, ArrayViewD};

#[doc(no_inline)]
pub use crate::{
    ArrayViewMut0,
    ArrayViewMut1,
    ArrayViewMut2,
    ArrayViewMut3,
    ArrayViewMut4,
    ArrayViewMut5,
    ArrayViewMut6,
    ArrayViewMutD,
};

#[doc(no_inline)]
pub use crate::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

#[doc(no_inline)]
pub use crate::{arr0, arr1, arr2, aview0, aview1, aview2, aview_mut1};

pub use crate::{array, azip, s};

#[doc(no_inline)]
pub use crate::ShapeBuilder;

#[doc(no_inline)]
pub use crate::NewAxis;

#[doc(no_inline)]
pub use crate::AsArray;

#[doc(no_inline)]
#[cfg(feature = "std")]
pub use crate::NdFloat;
