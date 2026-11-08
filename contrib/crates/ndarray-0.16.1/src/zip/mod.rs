// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use]
mod zipmacro;
mod ndproducer;

#[cfg(feature = "rayon")]
use std::mem::MaybeUninit;

use crate::imp_prelude::*;
use crate::partial::Partial;
use crate::AssignElem;
use crate::IntoDimension;
use crate::Layout;

use crate::dimension;
use crate::indexes::{indices, Indices};
use crate::split_at::{SplitAt, SplitPreference};

pub use self::ndproducer::{IntoNdProducer, NdProducer, Offset};

/// Return if the expression is a break value.
macro_rules! fold_while {
    ($e:expr) => {
        match $e {
            FoldWhile::Continue(x) => x,
            x => return x,
        }
    };
}

/// Broadcast an array so that it acts like a larger size and/or shape array.
///
/// See [broadcasting](ArrayBase#broadcasting) for more information.
trait Broadcast<E>
where E: IntoDimension
{
    type Output: NdProducer<Dim = E::Dim>;
    /// Broadcast the array to the new dimensions `shape`.
    ///
    /// ***Panics*** if broadcasting isn’t possible.
    #[track_caller]
    fn broadcast_unwrap(self, shape: E) -> Self::Output;
    private_decl! {}
}

/// Compute `Layout` hints for array shape dim, strides
fn array_layout<D: Dimension>(dim: &D, strides: &D) -> Layout
{
    let n = dim.ndim();
    if dimension::is_layout_c(dim, strides) {
        // effectively one-dimensional => C and F layout compatible
        if n <= 1 || dim.slice().iter().filter(|&&len| len > 1).count() <= 1 {
            Layout::one_dimensional()
        } else {
            Layout::c()
        }
    } else if n > 1 && dimension::is_layout_f(dim, strides) {
        Layout::f()
    } else if n > 1 {
        if dim[0] > 1 && strides[0] == 1 {
            Layout::fpref()
        } else if dim[n - 1] > 1 && strides[n - 1] == 1 {
            Layout::cpref()
        } else {
            Layout::none()
        }
    } else {
        Layout::none()
    }
}

impl<S, D> ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    pub(crate) fn layout_impl(&self) -> Layout
    {
        array_layout(&self.dim, &self.strides)
    }
}

impl<'a, A, D, E> Broadcast<E> for ArrayView<'a, A, D>
where
    E: IntoDimension,
    D: Dimension,
{
    type Output = ArrayView<'a, A, E::Dim>;
    fn broadcast_unwrap(self, shape: E) -> Self::Output
    {
        #[allow(clippy::needless_borrow)]
        let res: ArrayView<'_, A, E::Dim> = (&self).broadcast_unwrap(shape.into_dimension());
        unsafe { ArrayView::new(res.ptr, res.dim, res.strides) }
    }
    private_impl! {}
}

trait ZippableTuple: Sized
{
    type Item;
    type Ptr: OffsetTuple<Args = Self::Stride> + Copy;
    type Dim: Dimension;
    type Stride: Copy;
    fn as_ptr(&self) -> Self::Ptr;
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item;
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
    fn stride_of(&self, index: usize) -> Self::Stride;
    fn contiguous_stride(&self) -> Self::Stride;
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self);
}

/// Lock step function application across several arrays or other producers.
///
/// Zip allows matching several producers to each other elementwise and applying
/// a function over all tuples of elements (one item from each input at
/// a time).
///
/// In general, the zip uses a tuple of producers
/// ([`NdProducer`] trait) that all have to be of the
/// same shape. The NdProducer implementation defines what its item type is
/// (for example if it's a shared reference, mutable reference or an array
/// view etc).
///
/// If all the input arrays are of the same memory layout the zip performs much
/// better and the compiler can usually vectorize the loop (if applicable).
///
/// The order elements are visited is not specified. The producers don’t have to
/// have the same item type.
///
/// The `Zip` has two methods for function application: `for_each` and
/// `fold_while`. The zip object can be split, which allows parallelization.
/// A read-only zip object (no mutable producers) can be cloned.
///
/// See also the [`azip!()`] which offers a convenient shorthand
/// to common ways to use `Zip`.
///
/// ```
/// use ndarray::Zip;
/// use ndarray::Array2;
///
/// type M = Array2<f64>;
///
/// // Create four 2d arrays of the same size
/// let mut a = M::zeros((64, 32));
/// let b = M::from_elem(a.dim(), 1.);
/// let c = M::from_elem(a.dim(), 2.);
/// let d = M::from_elem(a.dim(), 3.);
///
/// // Example 1: Perform an elementwise arithmetic operation across
/// // the four arrays a, b, c, d.
///
/// Zip::from(&mut a)
///     .and(&b)
///     .and(&c)
///     .and(&d)
///     .for_each(|w, &x, &y, &z| {
///         *w += x + y * z;
///     });
///
/// // Example 2: Create a new array `totals` with one entry per row of `a`.
/// //  Use Zip to traverse the rows of `a` and assign to the corresponding
/// //  entry in `totals` with the sum across each row.
/// //  This is possible because the producer for `totals` and the row producer
/// //  for `a` have the same shape and dimensionality.
/// //  The rows producer yields one array view (`row`) per iteration.
///
/// use ndarray::{Array1, Axis};
///
/// let mut totals = Array1::zeros(a.nrows());
///
/// Zip::from(&mut totals)
///     .and(a.rows())
///     .for_each(|totals, row| *totals = row.sum());
///
/// // Check the result against the built in `.sum_axis()` along axis 1.
/// assert_eq!(totals, a.sum_axis(Axis(1)));
///
///
/// // Example 3: Recreate Example 2 using map_collect to make a new array
///
/// let totals2 = Zip::from(a.rows()).map_collect(|row| row.sum());
///
/// // Check the result against the previous example.
/// assert_eq!(totals, totals2);
/// ```
#[derive(Debug, Clone)]
#[must_use = "zipping producers is lazy and does nothing unless consumed"]
pub struct Zip<Parts, D>
{
    parts: Parts,
    dimension: D,
    layout: Layout,
    /// The sum of the layout tendencies of the parts;
    /// positive for c- and negative for f-layout preference.
    layout_tendency: i32,
}

impl<P, D> Zip<(P,), D>
where
    D: Dimension,
    P: NdProducer<Dim = D>,
{
    /// Create a new `Zip` from the input array or other producer `p`.
    ///
    /// The Zip will take the exact dimension of `p` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    pub fn from<IP>(p: IP) -> Self
    where IP: IntoNdProducer<Dim = D, Output = P, Item = P::Item>
    {
        let array = p.into_producer();
        let dim = array.raw_dim();
        let layout = array.layout();
        Zip {
            dimension: dim,
            layout,
            parts: (array,),
            layout_tendency: layout.tendency(),
        }
    }
}
impl<P, D> Zip<(Indices<D>, P), D>
where
    D: Dimension + Copy,
    P: NdProducer<Dim = D>,
{
    /// Create a new `Zip` with an index producer and the producer `p`.
    ///
    /// The Zip will take the exact dimension of `p` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    ///
    /// *Note:* Indexed zip has overhead.
    pub fn indexed<IP>(p: IP) -> Self
    where IP: IntoNdProducer<Dim = D, Output = P, Item = P::Item>
    {
        let array = p.into_producer();
        let dim = array.raw_dim();
        Zip::from(indices(dim)).and(array)
    }
}

#[inline]
fn zip_dimension_check<D, P>(dimension: &D, part: &P)
where
    D: Dimension,
    P: NdProducer<Dim = D>,
{
    ndassert!(
        part.equal_dim(dimension),
        "Zip: Producer dimension mismatch, expected: {:?}, got: {:?}",
        dimension,
        part.raw_dim()
    );
}

impl<Parts, D> Zip<Parts, D>
where D: Dimension
{
    /// Return a the number of element tuples in the Zip
    pub fn size(&self) -> usize
    {
        self.dimension.size()
    }

    /// Return the length of `axis`
    ///
    /// ***Panics*** if `axis` is out of bounds.
    #[track_caller]
    fn len_of(&self, axis: Axis) -> usize
    {
        self.dimension[axis.index()]
    }

    fn prefer_f(&self) -> bool
    {
        !self.layout.is(Layout::CORDER) && (self.layout.is(Layout::FORDER) || self.layout_tendency < 0)
    }

    /// Return an *approximation* to the max stride axis; if
    /// component arrays disagree, there may be no choice better than the
    /// others.
    fn max_stride_axis(&self) -> Axis
    {
        let i = if self.prefer_f() {
            self.dimension
                .slice()
                .iter()
                .rposition(|&len| len > 1)
                .unwrap_or(self.dimension.ndim() - 1)
        } else {
            /* corder or default */
            self.dimension
                .slice()
                .iter()
                .position(|&len| len > 1)
                .unwrap_or(0)
        };
        Axis(i)
    }
}

impl<P, D> Zip<P, D>
where D: Dimension
{
    fn for_each_core<F, Acc>(&mut self, acc: Acc, mut function: F) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple<Dim = D>,
    {
        if self.dimension.ndim() == 0 {
            function(acc, unsafe { self.parts.as_ref(self.parts.as_ptr()) })
        } else if self.layout.is(Layout::CORDER | Layout::FORDER) {
            self.for_each_core_contiguous(acc, function)
        } else {
            self.for_each_core_strided(acc, function)
        }
    }

    fn for_each_core_contiguous<F, Acc>(&mut self, acc: Acc, mut function: F) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple<Dim = D>,
    {
        debug_assert!(self.layout.is(Layout::CORDER | Layout::FORDER));
        let size = self.dimension.size();
        let ptrs = self.parts.as_ptr();
        let inner_strides = self.parts.contiguous_stride();
        unsafe { self.inner(acc, ptrs, inner_strides, size, &mut function) }
    }

    /// The innermost loop of the Zip for_each methods
    ///
    /// Run the fold while operation on a stretch of elements with constant strides
    ///
    /// `ptr`: base pointer for the first element in this stretch
    /// `strides`: strides for the elements in this stretch
    /// `len`: number of elements
    /// `function`: closure
    unsafe fn inner<F, Acc>(
        &self, mut acc: Acc, ptr: P::Ptr, strides: P::Stride, len: usize, function: &mut F,
    ) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple,
    {
        let mut i = 0;
        while i < len {
            let p = ptr.stride_offset(strides, i);
            acc = fold_while!(function(acc, self.parts.as_ref(p)));
            i += 1;
        }
        FoldWhile::Continue(acc)
    }

    fn for_each_core_strided<F, Acc>(&mut self, acc: Acc, function: F) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple<Dim = D>,
    {
        let n = self.dimension.ndim();
        if n == 0 {
            panic!("Unreachable: ndim == 0 is contiguous")
        }
        if n == 1 || self.layout_tendency >= 0 {
            self.for_each_core_strided_c(acc, function)
        } else {
            self.for_each_core_strided_f(acc, function)
        }
    }

    // Non-contiguous but preference for C - unroll over Axis(ndim - 1)
    fn for_each_core_strided_c<F, Acc>(&mut self, mut acc: Acc, mut function: F) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple<Dim = D>,
    {
        let n = self.dimension.ndim();
        let unroll_axis = n - 1;
        let inner_len = self.dimension[unroll_axis];
        self.dimension[unroll_axis] = 1;
        let mut index_ = self.dimension.first_index();
        let inner_strides = self.parts.stride_of(unroll_axis);
        // Loop unrolled over closest axis
        while let Some(index) = index_ {
            unsafe {
                let ptr = self.parts.uget_ptr(&index);
                acc = fold_while![self.inner(acc, ptr, inner_strides, inner_len, &mut function)];
            }

            index_ = self.dimension.next_for(index);
        }
        FoldWhile::Continue(acc)
    }

    // Non-contiguous but preference for F - unroll over Axis(0)
    fn for_each_core_strided_f<F, Acc>(&mut self, mut acc: Acc, mut function: F) -> FoldWhile<Acc>
    where
        F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
        P: ZippableTuple<Dim = D>,
    {
        let unroll_axis = 0;
        let inner_len = self.dimension[unroll_axis];
        self.dimension[unroll_axis] = 1;
        let index_ = self.dimension.first_index();
        let inner_strides = self.parts.stride_of(unroll_axis);
        // Loop unrolled over closest axis
        if let Some(mut index) = index_ {
            loop {
                unsafe {
                    let ptr = self.parts.uget_ptr(&index);
                    acc = fold_while![self.inner(acc, ptr, inner_strides, inner_len, &mut function)];
                }

                if !self.dimension.next_for_f(&mut index) {
                    break;
                }
            }
        }
        FoldWhile::Continue(acc)
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn uninitialized_for_current_layout<T>(&self) -> Array<MaybeUninit<T>, D>
    {
        let is_f = self.prefer_f();
        Array::uninit(self.dimension.clone().set_f(is_f))
    }
}

impl<D, P1, P2> Zip<(P1, P2), D>
where
    D: Dimension,
    P1: NdProducer<Dim = D>,
    P1: NdProducer<Dim = D>,
{
    /// Debug assert traversal order is like c (including 1D case)
    // Method placement: only used for binary Zip at the moment.
    #[inline]
    pub(crate) fn debug_assert_c_order(self) -> Self
    {
        debug_assert!(self.layout.is(Layout::CORDER) || self.layout_tendency >= 0 ||
                      self.dimension.slice().iter().filter(|&&d| d > 1).count() <= 1,
                      "Assertion failed: traversal is not c-order or 1D for \
                      layout {:?}, tendency {}, dimension {:?}",
                      self.layout, self.layout_tendency, self.dimension);
        self
    }
}

/*
trait Offset : Copy {
    unsafe fn offset(self, off: isize) -> Self;
    unsafe fn stride_offset(self, index: usize, stride: isize) -> Self {
        self.offset(index as isize * stride)
    }
}

impl<T> Offset for *mut T {
    unsafe fn offset(self, off: isize) -> Self {
        self.offset(off)
    }
}
*/

trait OffsetTuple
{
    type Args;
    unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self;
}

impl<T> OffsetTuple for *mut T
{
    type Args = isize;
    unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self
    {
        self.offset(index as isize * stride)
    }
}

macro_rules! offset_impl {
    ($([$($param:ident)*][ $($q:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<$($param: Offset),*> OffsetTuple for ($($param, )*) {
            type Args = ($($param::Stride,)*);
            unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self {
                let ($($param, )*) = self;
                let ($($q, )*) = stride;
                ($(Offset::stride_offset($param, $q, index),)*)
            }
        }
        )+
    };
}

offset_impl! {
    [A ][ a],
    [A B][ a b],
    [A B C][ a b c],
    [A B C D][ a b c d],
    [A B C D E][ a b c d e],
    [A B C D E F][ a b c d e f],
}

macro_rules! zipt_impl {
    ($([$($p:ident)*][ $($q:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> ZippableTuple for ($($p, )*) {
            type Item = ($($p::Item, )*);
            type Ptr = ($($p::Ptr, )*);
            type Dim = Dim;
            type Stride = ($($p::Stride,)* );

            fn stride_of(&self, index: usize) -> Self::Stride {
                let ($(ref $p,)*) = *self;
                ($($p.stride_of(Axis(index)), )*)
            }

            fn contiguous_stride(&self) -> Self::Stride {
                let ($(ref $p,)*) = *self;
                ($($p.contiguous_stride(), )*)
            }

            fn as_ptr(&self) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.as_ptr(), )*)
            }
            unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
                let ($(ref $q ,)*) = *self;
                let ($($p,)*) = ptr;
                ($($q.as_ref($p),)*)
            }

            unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.uget_ptr(i), )*)
            }

            fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
                let ($($p,)*) = self;
                let ($($p,)*) = (
                    $($p.split_at(axis, index), )*
                );
                (
                    ($($p.0,)*),
                    ($($p.1,)*)
                )
            }
        }
        )+
    };
}

zipt_impl! {
    [A ][ a],
    [A B][ a b],
    [A B C][ a b c],
    [A B C D][ a b c d],
    [A B C D E][ a b c d e],
    [A B C D E F][ a b c d e f],
}

macro_rules! map_impl {
    ($([$notlast:ident $($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<D, $($p),*> Zip<($($p,)*), D>
            where D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            /// Apply a function to all elements of the input arrays,
            /// visiting elements in lock step.
            pub fn for_each<F>(mut self, mut function: F)
                where F: FnMut($($p::Item),*)
            {
                self.for_each_core((), move |(), args| {
                    let ($($p,)*) = args;
                    FoldWhile::Continue(function($($p),*))
                });
            }

            /// Apply a fold function to all elements of the input arrays,
            /// visiting elements in lock step.
            ///
            /// # Example
            ///
            /// The expression `tr(AᵀB)` can be more efficiently computed as
            /// the equivalent expression `∑ᵢⱼ(A∘B)ᵢⱼ` (i.e. the sum of the
            /// elements of the entry-wise product). It would be possible to
            /// evaluate this expression by first computing the entry-wise
            /// product, `A∘B`, and then computing the elementwise sum of that
            /// product, but it's possible to do this in a single loop (and
            /// avoid an extra heap allocation if `A` and `B` can't be
            /// consumed) by using `Zip`:
            ///
            /// ```
            /// use ndarray::{array, Zip};
            ///
            /// let a = array![[1, 5], [3, 7]];
            /// let b = array![[2, 4], [8, 6]];
            ///
            /// // Without using `Zip`. This involves two loops and an extra
            /// // heap allocation for the result of `&a * &b`.
            /// let sum_prod_nonzip = (&a * &b).sum();
            /// // Using `Zip`. This is a single loop without any heap allocations.
            /// let sum_prod_zip = Zip::from(&a).and(&b).fold(0, |acc, a, b| acc + a * b);
            ///
            /// assert_eq!(sum_prod_nonzip, sum_prod_zip);
            /// ```
            pub fn fold<F, Acc>(mut self, acc: Acc, mut function: F) -> Acc
            where
                F: FnMut(Acc, $($p::Item),*) -> Acc,
            {
                self.for_each_core(acc, move |acc, args| {
                    let ($($p,)*) = args;
                    FoldWhile::Continue(function(acc, $($p),*))
                }).into_inner()
            }

            /// Apply a fold function to the input arrays while the return
            /// value is `FoldWhile::Continue`, visiting elements in lock step.
            ///
            pub fn fold_while<F, Acc>(mut self, acc: Acc, mut function: F)
                -> FoldWhile<Acc>
                where F: FnMut(Acc, $($p::Item),*) -> FoldWhile<Acc>
            {
                self.for_each_core(acc, move |acc, args| {
                    let ($($p,)*) = args;
                    function(acc, $($p),*)
                })
            }

            /// Tests if every element of the iterator matches a predicate.
            ///
            /// Returns `true` if `predicate` evaluates to `true` for all elements.
            /// Returns `true` if the input arrays are empty.
            ///
            /// Example:
            ///
            /// ```
            /// use ndarray::{array, Zip};
            /// let a = array![1, 2, 3];
            /// let b = array![1, 4, 9];
            /// assert!(Zip::from(&a).and(&b).all(|&a, &b| a * a == b));
            /// ```
            pub fn all<F>(mut self, mut predicate: F) -> bool
                where F: FnMut($($p::Item),*) -> bool
            {
                !self.for_each_core((), move |_, args| {
                    let ($($p,)*) = args;
                    if predicate($($p),*) {
                        FoldWhile::Continue(())
                    } else {
                        FoldWhile::Done(())
                    }
                }).is_done()
            }

            /// Tests if at least one element of the iterator matches a predicate.
            ///
            /// Returns `true` if `predicate` evaluates to `true` for at least one element.
            /// Returns `false` if the input arrays are empty.
            ///
            /// Example:
            ///
            /// ```
            /// use ndarray::{array, Zip};
            /// let a = array![1, 2, 3];
            /// let b = array![1, 4, 9];
            /// assert!(Zip::from(&a).and(&b).any(|&a, &b| a == b));
            /// assert!(!Zip::from(&a).and(&b).any(|&a, &b| a - 1 == b));
            /// ```
            pub fn any<F>(mut self, mut predicate: F) -> bool
                where F: FnMut($($p::Item),*) -> bool
            {
                self.for_each_core((), move |_, args| {
                    let ($($p,)*) = args;
                    if predicate($($p),*) {
                        FoldWhile::Done(())
                    } else {
                        FoldWhile::Continue(())
                    }
                }).is_done()
            }

            expand_if!(@bool [$notlast]

            /// Include the producer `p` in the Zip.
            ///
            /// ***Panics*** if `p`’s shape doesn’t match the Zip’s exactly.
            #[track_caller]
            pub fn and<P>(self, p: P) -> Zip<($($p,)* P::Output, ), D>
                where P: IntoNdProducer<Dim=D>,
            {
                let part = p.into_producer();
                zip_dimension_check(&self.dimension, &part);
                self.build_and(part)
            }

            /// Include the producer `p` in the Zip.
            ///
            /// ## Safety
            ///
            /// The caller must ensure that the producer's shape is equal to the Zip's shape.
            /// Uses assertions when debug assertions are enabled.
            #[allow(unused)]
            pub(crate) unsafe fn and_unchecked<P>(self, p: P) -> Zip<($($p,)* P::Output, ), D>
                where P: IntoNdProducer<Dim=D>,
            {
                #[cfg(debug_assertions)]
                {
                    self.and(p)
                }
                #[cfg(not(debug_assertions))]
                {
                    self.build_and(p.into_producer())
                }
            }

            /// Include the producer `p` in the Zip, broadcasting if needed.
            ///
            /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
            ///
            /// ***Panics*** if broadcasting isn’t possible.
            #[track_caller]
            pub fn and_broadcast<'a, P, D2, Elem>(self, p: P)
                -> Zip<($($p,)* ArrayView<'a, Elem, D>, ), D>
                where P: IntoNdProducer<Dim=D2, Output=ArrayView<'a, Elem, D2>, Item=&'a Elem>,
                      D2: Dimension,
            {
                let part = p.into_producer().broadcast_unwrap(self.dimension.clone());
                self.build_and(part)
            }

            fn build_and<P>(self, part: P) -> Zip<($($p,)* P, ), D>
                where P: NdProducer<Dim=D>,
            {
                let part_layout = part.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* part, ),
                    layout: self.layout.intersect(part_layout),
                    dimension: self.dimension,
                    layout_tendency: self.layout_tendency + part_layout.tendency(),
                }
            }

            /// Map and collect the results into a new array, which has the same size as the
            /// inputs.
            ///
            /// If all inputs are c- or f-order respectively, that is preserved in the output.
            pub fn map_collect<R>(self, f: impl FnMut($($p::Item,)* ) -> R) -> Array<R, D> {
                self.map_collect_owned(f)
            }

            pub(crate) fn map_collect_owned<S, R>(self, f: impl FnMut($($p::Item,)* ) -> R)
                -> ArrayBase<S, D>
                where S: DataOwned<Elem = R>
            {
                // safe because: all elements are written before the array is completed

                let shape = self.dimension.clone().set_f(self.prefer_f());
                let output = <ArrayBase<S, D>>::build_uninit(shape, |output| {
                    // Use partial to count the number of filled elements, and can drop the right
                    // number of elements on unwinding (if it happens during apply/collect).
                    unsafe {
                        let output_view = output.into_raw_view_mut().cast::<R>();
                        self.and(output_view)
                            .collect_with_partial(f)
                            .release_ownership();
                    }
                });
                unsafe {
                    output.assume_init()
                }
            }

            /// Map and assign the results into the producer `into`, which should have the same
            /// size as the other inputs.
            ///
            /// The producer should have assignable items as dictated by the `AssignElem` trait,
            /// for example `&mut R`.
            pub fn map_assign_into<R, Q>(self, into: Q, mut f: impl FnMut($($p::Item,)* ) -> R)
                where Q: IntoNdProducer<Dim=D>,
                      Q::Item: AssignElem<R>
            {
                self.and(into)
                    .for_each(move |$($p, )* output_| {
                        output_.assign_elem(f($($p ),*));
                    });
            }

            );

            /// Split the `Zip` evenly in two.
            ///
            /// It will be split in the way that best preserves element locality.
            pub fn split(self) -> (Self, Self) {
                debug_assert_ne!(self.size(), 0, "Attempt to split empty zip");
                debug_assert_ne!(self.size(), 1, "Attempt to split zip with 1 elem");
                SplitPreference::split(self)
            }
        }

        expand_if!(@bool [$notlast]
            // For collect; Last producer is a RawViewMut
            #[allow(non_snake_case)]
            impl<D, PLast, R, $($p),*> Zip<($($p,)* PLast), D>
                where D: Dimension,
                      $($p: NdProducer<Dim=D> ,)*
                      PLast: NdProducer<Dim = D, Item = *mut R, Ptr = *mut R, Stride = isize>,
            {
                /// The inner workings of map_collect and par_map_collect
                ///
                /// Apply the function and collect the results into the output (last producer)
                /// which should be a raw array view; a Partial that owns the written
                /// elements is returned.
                ///
                /// Elements will be overwritten in place (in the sense of std::ptr::write).
                ///
                /// ## Safety
                ///
                /// The last producer is a RawArrayViewMut and must be safe to write into.
                /// The producer must be c- or f-contig and have the same layout tendency
                /// as the whole Zip.
                ///
                /// The returned Partial's proxy ownership of the elements must be handled,
                /// before the array the raw view points to realizes its ownership.
                pub(crate) unsafe fn collect_with_partial<F>(self, mut f: F) -> Partial<R>
                    where F: FnMut($($p::Item,)* ) -> R
                {
                    // Get the last producer; and make a Partial that aliases its data pointer
                    let (.., ref output) = &self.parts;

                    // debug assert that the output is contiguous in the memory layout we need
                    if cfg!(debug_assertions) {
                        let out_layout = output.layout();
                        assert!(out_layout.is(Layout::CORDER | Layout::FORDER));
                        assert!(
                            (self.layout_tendency <= 0 && out_layout.tendency() <= 0) ||
                            (self.layout_tendency >= 0 && out_layout.tendency() >= 0),
                            "layout tendency violation for self layout {:?}, output layout {:?},\
                            output shape {:?}",
                            self.layout, out_layout, output.raw_dim());
                    }

                    let mut partial = Partial::new(output.as_ptr());

                    // Apply the mapping function on this zip
                    // if we panic with unwinding; Partial will drop the written elements.
                    let partial_len = &mut partial.len;
                    self.for_each(move |$($p,)* output_elem: *mut R| {
                        output_elem.write(f($($p),*));
                        if std::mem::needs_drop::<R>() {
                            *partial_len += 1;
                        }
                    });

                    partial
                }
            }
        );

        impl<D, $($p),*> SplitPreference for Zip<($($p,)*), D>
            where D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            fn can_split(&self) -> bool { self.size() > 1 }

            fn split_preference(&self) -> (Axis, usize) {
                // Always split in a way that preserves layout (if any)
                let axis = self.max_stride_axis();
                let index = self.len_of(axis) / 2;
                (axis, index)
            }
        }

        impl<D, $($p),*> SplitAt for Zip<($($p,)*), D>
            where D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
                let (p1, p2) = self.parts.split_at(axis, index);
                let (d1, d2) = self.dimension.split_at(axis, index);
                (Zip {
                    dimension: d1,
                    layout: self.layout,
                    parts: p1,
                    layout_tendency: self.layout_tendency,
                },
                Zip {
                    dimension: d2,
                    layout: self.layout,
                    parts: p2,
                    layout_tendency: self.layout_tendency,
                })
            }

        }

        )+
    };
}

map_impl! {
    [true P1],
    [true P1 P2],
    [true P1 P2 P3],
    [true P1 P2 P3 P4],
    [true P1 P2 P3 P4 P5],
    [false P1 P2 P3 P4 P5 P6],
}

/// Value controlling the execution of `.fold_while` on `Zip`.
#[derive(Debug, Copy, Clone)]
pub enum FoldWhile<T>
{
    /// Continue folding with this value
    Continue(T),
    /// Fold is complete and will return this value
    Done(T),
}

impl<T> FoldWhile<T>
{
    /// Return the inner value
    pub fn into_inner(self) -> T
    {
        match self {
            FoldWhile::Continue(x) | FoldWhile::Done(x) => x,
        }
    }

    /// Return true if it is `Done`, false if `Continue`
    pub fn is_done(&self) -> bool
    {
        match *self {
            FoldWhile::Continue(_) => false,
            FoldWhile::Done(_) => true,
        }
    }
}
