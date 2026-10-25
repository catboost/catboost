// Copyright 2020-2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::ptr::NonNull;

use crate::imp_prelude::*;
use crate::OwnedRepr;

use super::Baseiter;
use crate::impl_owned_array::drop_unreachable_raw;

/// By-value iterator for an array
pub struct IntoIter<A, D>
where D: Dimension
{
    array_data: OwnedRepr<A>,
    inner: Baseiter<A, D>,
    data_len: usize,
    /// first memory address of an array element
    array_head_ptr: NonNull<A>,
    // if true, the array owns elements that are not reachable by indexing
    // through all the indices of the dimension.
    has_unreachable_elements: bool,
}

impl<A, D> IntoIter<A, D>
where D: Dimension
{
    /// Create a new by-value iterator that consumes `array`
    pub(crate) fn new(array: Array<A, D>) -> Self
    {
        unsafe {
            let array_head_ptr = array.ptr;
            let mut array_data = array.data;
            let data_len = array_data.release_all_elements();
            debug_assert!(data_len >= array.dim.size());
            let has_unreachable_elements = array.dim.size() != data_len;
            let inner = Baseiter::new(array_head_ptr, array.dim, array.strides);

            IntoIter {
                array_data,
                inner,
                data_len,
                array_head_ptr,
                has_unreachable_elements,
            }
        }
    }
}

impl<A, D: Dimension> Iterator for IntoIter<A, D>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A>
    {
        self.inner.next().map(|p| unsafe { p.as_ptr().read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        self.inner.size_hint()
    }
}

impl<A, D: Dimension> ExactSizeIterator for IntoIter<A, D>
{
    fn len(&self) -> usize
    {
        self.inner.len()
    }
}

impl<A, D> Drop for IntoIter<A, D>
where D: Dimension
{
    fn drop(&mut self)
    {
        if !self.has_unreachable_elements || mem::size_of::<A>() == 0 || !mem::needs_drop::<A>() {
            return;
        }

        // iterate til the end
        while let Some(_) = self.next() {}

        unsafe {
            let data_ptr = self.array_data.as_nonnull_mut();
            let view = RawArrayViewMut::new(self.array_head_ptr, self.inner.dim.clone(), self.inner.strides.clone());
            debug_assert!(self.inner.dim.size() < self.data_len, "data_len {} and dim size {}",
                          self.data_len, self.inner.dim.size());
            drop_unreachable_raw(view, data_ptr, self.data_len);
        }
    }
}

impl<A, D> IntoIterator for Array<A, D>
where D: Dimension
{
    type Item = A;
    type IntoIter = IntoIter<A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        IntoIter::new(self)
    }
}

impl<A, D> IntoIterator for ArcArray<A, D>
where
    D: Dimension,
    A: Clone,
{
    type Item = A;
    type IntoIter = IntoIter<A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        IntoIter::new(self.into_owned())
    }
}

impl<A, D> IntoIterator for CowArray<'_, A, D>
where
    D: Dimension,
    A: Clone,
{
    type Item = A;
    type IntoIter = IntoIter<A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        IntoIter::new(self.into_owned())
    }
}
