// Copyright 2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr;

/// Partial is a partially written contiguous slice of data;
/// it is the owner of the elements, but not the allocation,
/// and will drop the elements on drop.
#[must_use]
pub(crate) struct Partial<T>
{
    /// Data pointer
    ptr: *mut T,
    /// Current length
    pub(crate) len: usize,
}

impl<T> Partial<T>
{
    /// Create an empty partial for this data pointer
    ///
    /// ## Safety
    ///
    /// Unless ownership is released, the Partial acts as an owner of the slice of data (not the
    /// allocation); and will free the elements on drop; the pointer must be dereferenceable and
    /// the `len` elements following it valid.
    ///
    /// The Partial has an accessible length field which must only be modified in trusted code.
    pub(crate) unsafe fn new(ptr: *mut T) -> Self
    {
        Self { ptr, len: 0 }
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn stub() -> Self
    {
        Self {
            len: 0,
            ptr: ptr::null_mut(),
        }
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn is_stub(&self) -> bool
    {
        self.ptr.is_null()
    }

    /// Release Partial's ownership of the written elements, and return the current length
    pub(crate) fn release_ownership(mut self) -> usize
    {
        let ret = self.len;
        self.len = 0;
        ret
    }

    #[cfg(feature = "rayon")]
    /// Merge if they are in order (left to right) and contiguous.
    /// Skips merge if T does not need drop.
    pub(crate) fn try_merge(mut left: Self, right: Self) -> Self
    {
        if !std::mem::needs_drop::<T>() {
            return left;
        }
        // Merge the partial collect results; the final result will be a slice that
        // covers the whole output.
        if left.is_stub() {
            right
        } else if left.ptr.wrapping_add(left.len) == right.ptr {
            left.len += right.release_ownership();
            left
        } else {
            // failure to merge; this is a bug in collect, so we will never reach this
            debug_assert!(false, "Partial: failure to merge left and right parts");
            left
        }
    }
}

unsafe impl<T> Send for Partial<T> where T: Send {}

impl<T> Drop for Partial<T>
{
    fn drop(&mut self)
    {
        if !self.ptr.is_null() {
            unsafe {
                ptr::drop_in_place(alloc::slice::from_raw_parts_mut(self.ptr, self.len));
            }
        }
    }
}
