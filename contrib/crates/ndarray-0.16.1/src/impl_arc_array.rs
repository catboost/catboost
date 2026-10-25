// Copyright 2019 ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

/// Methods specific to `ArcArray`.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A, D> ArcArray<A, D>
where D: Dimension
{
    /// Returns `true` iff the inner `Arc` is not shared.
    /// If you want to ensure the `Arc` is not concurrently cloned, you need to provide a `&mut self` to this function.
    pub fn is_unique(&self) -> bool
    {
        // Only strong pointers are used in this crate.
        Arc::strong_count(&self.data.0) == 1
    }
}
