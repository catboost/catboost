// Copyright 2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::MaybeUninit;

use crate::imp_prelude::*;
use crate::RawDataSubst;

/// Methods specific to arrays with `MaybeUninit` elements.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A, S, D> ArrayBase<S, D>
where
    S: RawDataSubst<A, Elem = MaybeUninit<A>>,
    D: Dimension,
{
    /// **Promise** that the array's elements are all fully initialized, and convert
    /// the array from element type `MaybeUninit<A>` to `A`.
    ///
    /// For example, it can convert an `Array<MaybeUninit<f64>, D>` to `Array<f64, D>`.
    ///
    /// ## Safety
    ///
    /// Safe to use if all the array's elements have been initialized.
    ///
    /// Note that for owned and shared ownership arrays, the promise must include all of the
    /// array's storage; it is for example possible to slice these in place, but that must
    /// only be done after all elements have been initialized.
    pub unsafe fn assume_init(self) -> ArrayBase<<S as RawDataSubst<A>>::Output, D>
    {
        let ArrayBase {
            data,
            ptr,
            dim,
            strides,
        } = self;

        // "transmute" from storage of MaybeUninit<A> to storage of A
        let data = S::data_subst(data);
        let ptr = ptr.cast::<A>();
        ArrayBase::from_data_ptr(data, ptr).with_strides_dim(strides, dim)
    }
}
