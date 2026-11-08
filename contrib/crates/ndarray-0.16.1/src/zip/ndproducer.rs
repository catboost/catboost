use crate::imp_prelude::*;
use crate::Layout;
use crate::NdIndex;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Argument conversion into a producer.
///
/// Slices and vectors can be used (equivalent to 1-dimensional array views).
///
/// This trait is like `IntoIterator` for `NdProducers` instead of iterators.
pub trait IntoNdProducer
{
    /// The element produced per iteration.
    type Item;
    /// Dimension type of the producer
    type Dim: Dimension;
    type Output: NdProducer<Dim = Self::Dim, Item = Self::Item>;
    /// Convert the value into an `NdProducer`.
    fn into_producer(self) -> Self::Output;
}

impl<P> IntoNdProducer for P
where P: NdProducer
{
    type Item = P::Item;
    type Dim = P::Dim;
    type Output = Self;
    fn into_producer(self) -> Self::Output
    {
        self
    }
}

/// A producer of an n-dimensional set of elements;
/// for example an array view, mutable array view or an iterator
/// that yields chunks.
///
/// Producers are used as a arguments to [`Zip`](crate::Zip) and
/// [`azip!()`].
///
/// # Comparison to `IntoIterator`
///
/// Most `NdProducers` are *iterable* (implement `IntoIterator`) but not directly
/// iterators. This separation is needed because the producer represents
/// a multidimensional set of items, it can be split along a particular axis for
/// parallelization, and it has no fixed correspondence to a sequence.
///
/// The natural exception is one dimensional producers, like `AxisIter`, which
/// implement `Iterator` directly
/// (`AxisIter` traverses a one dimensional sequence, along an axis, while
/// *producing* multidimensional items).
///
/// See also [`IntoNdProducer`]
pub trait NdProducer
{
    /// The element produced per iteration.
    type Item;
    // Internal use / Pointee type
    /// Dimension type
    type Dim: Dimension;

    // The pointer Ptr is used by an array view to simply point to the
    // current element. It doesn't have to be a pointer (see Indices).
    // Its main function is that it can be incremented with a particular
    // stride (= along a particular axis)
    #[doc(hidden)]
    /// Pointer or stand-in for pointer
    type Ptr: Offset<Stride = Self::Stride>;
    #[doc(hidden)]
    /// Pointer stride
    type Stride: Copy;

    #[doc(hidden)]
    fn layout(&self) -> Layout;
    /// Return the shape of the producer.
    fn raw_dim(&self) -> Self::Dim;
    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.raw_dim() == *dim
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr;
    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item;
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> <Self::Ptr as Offset>::Stride;
    #[doc(hidden)]
    fn contiguous_stride(&self) -> Self::Stride;
    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    where Self: Sized;

    private_decl! {}
}

pub trait Offset: Copy
{
    type Stride: Copy;
    unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self;
    private_decl! {}
}

impl<T> Offset for *const T
{
    type Stride = isize;
    unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self
    {
        self.offset(s * (index as isize))
    }
    private_impl! {}
}

impl<T> Offset for *mut T
{
    type Stride = isize;
    unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self
    {
        self.offset(s * (index as isize))
    }
    private_impl! {}
}

/// An array reference is an n-dimensional producer of element references
/// (like ArrayView).
impl<'a, A: 'a, S, D> IntoNdProducer for &'a ArrayBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A>,
{
    type Item = &'a A;
    type Dim = D;
    type Output = ArrayView<'a, A, D>;
    fn into_producer(self) -> Self::Output
    {
        self.view()
    }
}

/// A mutable array reference is an n-dimensional producer of mutable element
/// references (like ArrayViewMut).
impl<'a, A: 'a, S, D> IntoNdProducer for &'a mut ArrayBase<S, D>
where
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Item = &'a mut A;
    type Dim = D;
    type Output = ArrayViewMut<'a, A, D>;
    fn into_producer(self) -> Self::Output
    {
        self.view_mut()
    }
}

/// A slice is a one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a [A]
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

/// A mutable slice is a mutable one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a mut [A]
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

/// A one-dimensional array is a one-dimensional producer
impl<'a, A: 'a, const N: usize> IntoNdProducer for &'a [A; N]
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

/// A mutable one-dimensional array is a mutable one-dimensional producer
impl<'a, A: 'a, const N: usize> IntoNdProducer for &'a mut [A; N]
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

/// A Vec is a one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a Vec<A>
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

/// A mutable Vec is a mutable one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a mut Vec<A>
{
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output
    {
        <_>::from(self)
    }
}

impl<'a, A, D: Dimension> NdProducer for ArrayView<'a, A, D>
{
    type Item = &'a A;
    type Dim = D;
    type Ptr = *mut A;
    type Stride = isize;

    private_impl! {}

    fn raw_dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.dim.equal(dim)
    }

    fn as_ptr(&self) -> *mut A
    {
        self.as_ptr() as _
    }

    fn layout(&self) -> Layout
    {
        self.layout_impl()
    }

    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item
    {
        &*ptr
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A
    {
        self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize
    {
        self.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride
    {
        1
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}

impl<'a, A, D: Dimension> NdProducer for ArrayViewMut<'a, A, D>
{
    type Item = &'a mut A;
    type Dim = D;
    type Ptr = *mut A;
    type Stride = isize;

    private_impl! {}

    fn raw_dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.dim.equal(dim)
    }

    fn as_ptr(&self) -> *mut A
    {
        self.as_ptr() as _
    }

    fn layout(&self) -> Layout
    {
        self.layout_impl()
    }

    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item
    {
        &mut *ptr
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A
    {
        self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize
    {
        self.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride
    {
        1
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}

impl<A, D: Dimension> NdProducer for RawArrayView<A, D>
{
    type Item = *const A;
    type Dim = D;
    type Ptr = *const A;
    type Stride = isize;

    private_impl! {}

    fn raw_dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.dim.equal(dim)
    }

    fn as_ptr(&self) -> *const A
    {
        self.as_ptr()
    }

    fn layout(&self) -> Layout
    {
        self.layout_impl()
    }

    unsafe fn as_ref(&self, ptr: *const A) -> *const A
    {
        ptr
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *const A
    {
        self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize
    {
        self.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride
    {
        1
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}

impl<A, D: Dimension> NdProducer for RawArrayViewMut<A, D>
{
    type Item = *mut A;
    type Dim = D;
    type Ptr = *mut A;
    type Stride = isize;

    private_impl! {}

    fn raw_dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn equal_dim(&self, dim: &Self::Dim) -> bool
    {
        self.dim.equal(dim)
    }

    fn as_ptr(&self) -> *mut A
    {
        self.as_ptr() as _
    }

    fn layout(&self) -> Layout
    {
        self.layout_impl()
    }

    unsafe fn as_ref(&self, ptr: *mut A) -> *mut A
    {
        ptr
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A
    {
        self.ptr.as_ptr().offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize
    {
        self.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride
    {
        1
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self)
    {
        self.split_at(axis, index)
    }
}
