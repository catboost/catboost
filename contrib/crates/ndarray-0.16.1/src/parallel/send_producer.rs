use crate::imp_prelude::*;
use crate::{Layout, NdProducer};
use std::ops::{Deref, DerefMut};

/// An NdProducer that is unconditionally `Send`.
#[repr(transparent)]
pub(crate) struct SendProducer<T>
{
    inner: T,
}

impl<T> SendProducer<T>
{
    /// Create an unconditionally `Send` ndproducer from the producer
    pub(crate) unsafe fn new(producer: T) -> Self
    {
        Self { inner: producer }
    }
}

unsafe impl<P> Send for SendProducer<P> {}

impl<P> Deref for SendProducer<P>
{
    type Target = P;
    fn deref(&self) -> &P
    {
        &self.inner
    }
}

impl<P> DerefMut for SendProducer<P>
{
    fn deref_mut(&mut self) -> &mut P
    {
        &mut self.inner
    }
}

impl<P: NdProducer> NdProducer for SendProducer<P>
where P: NdProducer
{
    type Item = P::Item;
    type Dim = P::Dim;
    type Ptr = P::Ptr;
    type Stride = P::Stride;

    private_impl! {}

    #[inline(always)]
    fn raw_dim(&self) -> Self::Dim
    {
        self.inner.raw_dim()
    }

    #[inline(always)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.inner.equal_dim(dim)
    }

    #[inline(always)]
    fn as_ptr(&self) -> Self::Ptr
    {
        self.inner.as_ptr()
    }

    #[inline(always)]
    fn layout(&self) -> Layout
    {
        self.inner.layout()
    }

    #[inline(always)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item
    {
        self.inner.as_ref(ptr)
    }

    #[inline(always)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr
    {
        self.inner.uget_ptr(i)
    }

    #[inline(always)]
    fn stride_of(&self, axis: Axis) -> Self::Stride
    {
        self.inner.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride
    {
        self.inner.contiguous_stride()
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        let (a, b) = self.inner.split_at(axis, index);
        (Self { inner: a }, Self { inner: b })
    }
}
