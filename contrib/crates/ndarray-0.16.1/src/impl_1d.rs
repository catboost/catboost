// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for one-dimensional arrays.
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::mem::MaybeUninit;

use crate::imp_prelude::*;
use crate::low_level_util::AbortIfPanic;

/// # Methods For 1-D Arrays
impl<A, S> ArrayBase<S, Ix1>
where S: RawData<Elem = A>
{
    /// Return an vector with the elements of the one-dimensional array.
    pub fn to_vec(&self) -> Vec<A>
    where
        A: Clone,
        S: Data,
    {
        if let Some(slc) = self.as_slice() {
            slc.to_vec()
        } else {
            crate::iterators::to_vec(self.iter().cloned())
        }
    }

    /// Rotate the elements of the array by 1 element towards the front;
    /// the former first element becomes the last.
    pub(crate) fn rotate1_front(&mut self)
    where S: DataMut
    {
        // use swapping to keep all elements initialized (as required by owned storage)
        let mut lane_iter = self.iter_mut();
        let mut dst = if let Some(dst) = lane_iter.next() { dst } else { return };

        // Logically we do a circular swap here, all elements in a chain
        // Using MaybeUninit to avoid unnecessary writes in the safe swap solution
        //
        //  for elt in lane_iter {
        //      std::mem::swap(dst, elt);
        //      dst = elt;
        //  }
        //
        let guard = AbortIfPanic(&"rotate1_front: temporarily moving out of owned value");
        let mut slot = MaybeUninit::<A>::uninit();
        unsafe {
            slot.as_mut_ptr().copy_from_nonoverlapping(dst, 1);
            for elt in lane_iter {
                (dst as *mut A).copy_from_nonoverlapping(elt, 1);
                dst = elt;
            }
            (dst as *mut A).copy_from_nonoverlapping(slot.as_ptr(), 1);
        }
        guard.defuse();
    }
}
