// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use crate::dimension::slices_intersect;
use crate::error::{ErrorKind, ShapeError};
#[cfg(doc)]
use crate::s;
use crate::{ArrayViewMut, DimAdd, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// A slice (range with step size).
///
/// `end` is an exclusive index. Negative `start` or `end` indexes are counted
/// from the back of the axis. If `end` is `None`, the slice extends to the end
/// of the axis.
///
/// See also the [`s![]`](s!) macro.
///
/// ## Examples
///
/// `Slice::new(0, None, 1)` is the full range of an axis. It can also be
/// created with `Slice::from(..)`. The Python equivalent is `[:]`.
///
/// `Slice::new(a, b, 2)` is every second element from `a` until `b`. It can
/// also be created with `Slice::from(a..b).step_by(2)`. The Python equivalent
/// is `[a:b:2]`.
///
/// `Slice::new(a, None, -1)` is every element, from `a` until the end, in
/// reverse order. It can also be created with `Slice::from(a..).step_by(-1)`.
/// The Python equivalent is `[a::-1]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slice
{
    /// start index; negative are counted from the back of the axis
    pub start: isize,
    /// end index; negative are counted from the back of the axis; when not present
    /// the default is the full length of the axis.
    pub end: Option<isize>,
    /// step size in elements; the default is 1, for every element.
    pub step: isize,
}

impl Slice
{
    /// Create a new `Slice` with the given extents.
    ///
    /// See also the `From` impls, converting from ranges; for example
    /// `Slice::from(i..)` or `Slice::from(j..k)`.
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Slice
    {
        debug_assert_ne!(step, 0, "Slice::new: step must be nonzero");
        Slice { start, end, step }
    }

    /// Create a new `Slice` with the given step size (multiplied with the
    /// previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: isize) -> Self
    {
        debug_assert_ne!(step, 0, "Slice::step_by: step must be nonzero");
        Slice {
            step: self.step * step,
            ..self
        }
    }
}

/// Token to represent a new axis in a slice description.
///
/// See also the [`s![]`](s!) macro.
#[derive(Clone, Copy, Debug)]
pub struct NewAxis;

/// A slice (range with step), an index, or a new axis token.
///
/// See also the [`s![]`](s!) macro for a convenient way to create a
/// `SliceInfo<[SliceInfoElem; n], Din, Dout>`.
///
/// ## Examples
///
/// `SliceInfoElem::Index(a)` is the index `a`. It can also be created with
/// `SliceInfoElem::from(a)`. The Python equivalent is `[a]`. The macro
/// equivalent is `s![a]`.
///
/// `SliceInfoElem::Slice { start: 0, end: None, step: 1 }` is the full range
/// of an axis. It can also be created with `SliceInfoElem::from(..)`. The
/// Python equivalent is `[:]`. The macro equivalent is `s![..]`.
///
/// `SliceInfoElem::Slice { start: a, end: Some(b), step: 2 }` is every second
/// element from `a` until `b`. It can also be created with
/// `SliceInfoElem::from(Slice::from(a..b).step_by(2))`. The Python equivalent
/// is `[a:b:2]`. The macro equivalent is `s![a..b;2]`.
///
/// `SliceInfoElem::Slice { start: a, end: None, step: -1 }` is every element,
/// from `a` until the end, in reverse order. It can also be created with
/// `SliceInfoElem::from(Slice::from(a..).step_by(-1))`. The Python equivalent
/// is `[a::-1]`. The macro equivalent is `s![a..;-1]`.
///
/// `SliceInfoElem::NewAxis` is a new axis of length 1. It can also be created
/// with `SliceInfoElem::from(NewAxis)`. The Python equivalent is
/// `[np.newaxis]`. The macro equivalent is `s![NewAxis]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SliceInfoElem
{
    /// A range with step size. `end` is an exclusive index. Negative `start`
    /// or `end` indexes are counted from the back of the axis. If `end` is
    /// `None`, the slice extends to the end of the axis.
    Slice
    {
        /// start index; negative are counted from the back of the axis
        start: isize,
        /// end index; negative are counted from the back of the axis; when not present
        /// the default is the full length of the axis.
        end: Option<isize>,
        /// step size in elements; the default is 1, for every element.
        step: isize,
    },
    /// A single index.
    Index(isize),
    /// A new axis of length 1.
    NewAxis,
}

copy_and_clone! {SliceInfoElem}

impl SliceInfoElem
{
    /// Returns `true` if `self` is a `Slice` value.
    pub fn is_slice(&self) -> bool
    {
        matches!(self, SliceInfoElem::Slice { .. })
    }

    /// Returns `true` if `self` is an `Index` value.
    pub fn is_index(&self) -> bool
    {
        matches!(self, SliceInfoElem::Index(_))
    }

    /// Returns `true` if `self` is a `NewAxis` value.
    pub fn is_new_axis(&self) -> bool
    {
        matches!(self, SliceInfoElem::NewAxis)
    }
}

impl fmt::Display for SliceInfoElem
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match *self {
            SliceInfoElem::Index(index) => write!(f, "{}", index)?,
            SliceInfoElem::Slice { start, end, step } => {
                if start != 0 {
                    write!(f, "{}", start)?;
                }
                write!(f, "..")?;
                if let Some(i) = end {
                    write!(f, "{}", i)?;
                }
                if step != 1 {
                    write!(f, ";{}", step)?;
                }
            }
            SliceInfoElem::NewAxis => write!(f, stringify!(NewAxis))?,
        }
        Ok(())
    }
}

macro_rules! impl_slice_variant_from_range {
    ($self:ty, $constructor:path, $index:ty) => {
        impl From<Range<$index>> for $self {
            #[inline]
            fn from(r: Range<$index>) -> $self {
                $constructor {
                    start: r.start as isize,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeInclusive<$index>) -> $self {
                let end = *r.end() as isize;
                $constructor {
                    start: *r.start() as isize,
                    end: if end == -1 { None } else { Some(end + 1) },
                    step: 1,
                }
            }
        }

        impl From<RangeFrom<$index>> for $self {
            #[inline]
            fn from(r: RangeFrom<$index>) -> $self {
                $constructor {
                    start: r.start as isize,
                    end: None,
                    step: 1,
                }
            }
        }

        impl From<RangeTo<$index>> for $self {
            #[inline]
            fn from(r: RangeTo<$index>) -> $self {
                $constructor {
                    start: 0,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeToInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeToInclusive<$index>) -> $self {
                let end = r.end as isize;
                $constructor {
                    start: 0,
                    end: if end == -1 { None } else { Some(end + 1) },
                    step: 1,
                }
            }
        }
    };
}
impl_slice_variant_from_range!(Slice, Slice, isize);
impl_slice_variant_from_range!(Slice, Slice, usize);
impl_slice_variant_from_range!(Slice, Slice, i32);
impl_slice_variant_from_range!(SliceInfoElem, SliceInfoElem::Slice, isize);
impl_slice_variant_from_range!(SliceInfoElem, SliceInfoElem::Slice, usize);
impl_slice_variant_from_range!(SliceInfoElem, SliceInfoElem::Slice, i32);

impl From<RangeFull> for Slice
{
    #[inline]
    fn from(_: RangeFull) -> Slice
    {
        Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFull> for SliceInfoElem
{
    #[inline]
    fn from(_: RangeFull) -> SliceInfoElem
    {
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<Slice> for SliceInfoElem
{
    #[inline]
    fn from(s: Slice) -> SliceInfoElem
    {
        SliceInfoElem::Slice {
            start: s.start,
            end: s.end,
            step: s.step,
        }
    }
}

macro_rules! impl_sliceinfoelem_from_index {
    ($index:ty) => {
        impl From<$index> for SliceInfoElem {
            #[inline]
            fn from(r: $index) -> SliceInfoElem {
                SliceInfoElem::Index(r as isize)
            }
        }
    };
}
impl_sliceinfoelem_from_index!(isize);
impl_sliceinfoelem_from_index!(usize);
impl_sliceinfoelem_from_index!(i32);

impl From<NewAxis> for SliceInfoElem
{
    #[inline]
    fn from(_: NewAxis) -> SliceInfoElem
    {
        SliceInfoElem::NewAxis
    }
}

/// A type that can slice an array of dimension `D`.
///
/// This trait is unsafe to implement because the implementation must ensure
/// that `D`, `Self::OutDim`, `self.in_dim()`, and `self.out_ndim()` are
/// consistent with the `&[SliceInfoElem]` returned by `self.as_ref()` and that
/// `self.as_ref()` always returns the same value when called multiple times.
#[allow(clippy::missing_safety_doc)] // not implementable downstream
pub unsafe trait SliceArg<D: Dimension>: AsRef<[SliceInfoElem]>
{
    /// Dimensionality of the output array.
    type OutDim: Dimension;

    /// Returns the number of axes in the input array.
    fn in_ndim(&self) -> usize;

    /// Returns the number of axes in the output array.
    fn out_ndim(&self) -> usize;

    private_decl! {}
}

unsafe impl<T, D> SliceArg<D> for &T
where
    T: SliceArg<D> + ?Sized,
    D: Dimension,
{
    type OutDim = T::OutDim;

    fn in_ndim(&self) -> usize
    {
        T::in_ndim(self)
    }

    fn out_ndim(&self) -> usize
    {
        T::out_ndim(self)
    }

    private_impl! {}
}

macro_rules! impl_slicearg_samedim {
    ($in_dim:ty) => {
        unsafe impl<T, Dout> SliceArg<$in_dim> for SliceInfo<T, $in_dim, Dout>
        where
            T: AsRef<[SliceInfoElem]>,
            Dout: Dimension,
        {
            type OutDim = Dout;

            fn in_ndim(&self) -> usize {
                self.in_ndim()
            }

            fn out_ndim(&self) -> usize {
                self.out_ndim()
            }

            private_impl! {}
        }
    };
}
impl_slicearg_samedim!(Ix0);
impl_slicearg_samedim!(Ix1);
impl_slicearg_samedim!(Ix2);
impl_slicearg_samedim!(Ix3);
impl_slicearg_samedim!(Ix4);
impl_slicearg_samedim!(Ix5);
impl_slicearg_samedim!(Ix6);

unsafe impl<T, Din, Dout> SliceArg<IxDyn> for SliceInfo<T, Din, Dout>
where
    T: AsRef<[SliceInfoElem]>,
    Din: Dimension,
    Dout: Dimension,
{
    type OutDim = Dout;

    fn in_ndim(&self) -> usize
    {
        self.in_ndim()
    }

    fn out_ndim(&self) -> usize
    {
        self.out_ndim()
    }

    private_impl! {}
}

unsafe impl SliceArg<IxDyn> for [SliceInfoElem]
{
    type OutDim = IxDyn;

    fn in_ndim(&self) -> usize
    {
        self.iter().filter(|s| !s.is_new_axis()).count()
    }

    fn out_ndim(&self) -> usize
    {
        self.iter().filter(|s| !s.is_index()).count()
    }

    private_impl! {}
}

/// Represents all of the necessary information to perform a slice.
///
/// The type `T` is typically `[SliceInfoElem; n]`, `&[SliceInfoElem]`, or
/// `Vec<SliceInfoElem>`. The type `Din` is the dimension of the array to be
/// sliced, and `Dout` is the output dimension after calling [`.slice()`]. Note
/// that if `Din` is a fixed dimension type (`Ix0`, `Ix1`, `Ix2`, etc.), the
/// `SliceInfo` instance can still be used to slice an array with dimension
/// `IxDyn` as long as the number of axes matches.
///
/// [`.slice()`]: crate::ArrayBase::slice
#[derive(Debug)]
pub struct SliceInfo<T, Din: Dimension, Dout: Dimension>
{
    in_dim: PhantomData<Din>,
    out_dim: PhantomData<Dout>,
    indices: T,
}

impl<T, Din, Dout> Deref for SliceInfo<T, Din, Dout>
where
    Din: Dimension,
    Dout: Dimension,
{
    type Target = T;
    fn deref(&self) -> &Self::Target
    {
        &self.indices
    }
}

fn check_dims_for_sliceinfo<Din, Dout>(indices: &[SliceInfoElem]) -> Result<(), ShapeError>
where
    Din: Dimension,
    Dout: Dimension,
{
    if let Some(in_ndim) = Din::NDIM {
        if in_ndim != indices.in_ndim() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    }
    if let Some(out_ndim) = Dout::NDIM {
        if out_ndim != indices.out_ndim() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    }
    Ok(())
}

impl<T, Din, Dout> SliceInfo<T, Din, Dout>
where
    T: AsRef<[SliceInfoElem]>,
    Din: Dimension,
    Dout: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// **Note:** only unchecked for non-debug builds of `ndarray`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `in_dim` and `out_dim` are consistent with
    /// `indices` and that `indices.as_ref()` always returns the same value
    /// when called multiple times.
    #[doc(hidden)]
    pub unsafe fn new_unchecked(
        indices: T, in_dim: PhantomData<Din>, out_dim: PhantomData<Dout>,
    ) -> SliceInfo<T, Din, Dout>
    {
        if cfg!(debug_assertions) {
            check_dims_for_sliceinfo::<Din, Dout>(indices.as_ref())
                .expect("`Din` and `Dout` must be consistent with `indices`.");
        }
        SliceInfo {
            in_dim,
            out_dim,
            indices,
        }
    }

    /// Returns a new `SliceInfo` instance.
    ///
    /// Errors if `Din` or `Dout` is not consistent with `indices`.
    ///
    /// For common types, a safe alternative is to use `TryFrom` instead.
    ///
    /// # Safety
    ///
    /// The caller must ensure `indices.as_ref()` always returns the same value
    /// when called multiple times.
    pub unsafe fn new(indices: T) -> Result<SliceInfo<T, Din, Dout>, ShapeError>
    {
        check_dims_for_sliceinfo::<Din, Dout>(indices.as_ref())?;
        Ok(SliceInfo {
            in_dim: PhantomData,
            out_dim: PhantomData,
            indices,
        })
    }

    /// Returns the number of dimensions of the input array for
    /// [`.slice()`](crate::ArrayBase::slice).
    ///
    /// If `Din` is a fixed-size dimension type, then this is equivalent to
    /// `Din::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the `SliceInfoElem` elements.
    pub fn in_ndim(&self) -> usize
    {
        if let Some(ndim) = Din::NDIM {
            ndim
        } else {
            self.indices.as_ref().in_ndim()
        }
    }

    /// Returns the number of dimensions after calling
    /// [`.slice()`](crate::ArrayBase::slice) (including taking
    /// subviews).
    ///
    /// If `Dout` is a fixed-size dimension type, then this is equivalent to
    /// `Dout::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the `SliceInfoElem` elements.
    pub fn out_ndim(&self) -> usize
    {
        if let Some(ndim) = Dout::NDIM {
            ndim
        } else {
            self.indices.as_ref().out_ndim()
        }
    }
}

impl<'a, Din, Dout> TryFrom<&'a [SliceInfoElem]> for SliceInfo<&'a [SliceInfoElem], Din, Dout>
where
    Din: Dimension,
    Dout: Dimension,
{
    type Error = ShapeError;

    fn try_from(indices: &'a [SliceInfoElem]) -> Result<SliceInfo<&'a [SliceInfoElem], Din, Dout>, ShapeError>
    {
        unsafe {
            // This is okay because `&[SliceInfoElem]` always returns the same
            // value for `.as_ref()`.
            Self::new(indices)
        }
    }
}

impl<Din, Dout> TryFrom<Vec<SliceInfoElem>> for SliceInfo<Vec<SliceInfoElem>, Din, Dout>
where
    Din: Dimension,
    Dout: Dimension,
{
    type Error = ShapeError;

    fn try_from(indices: Vec<SliceInfoElem>) -> Result<SliceInfo<Vec<SliceInfoElem>, Din, Dout>, ShapeError>
    {
        unsafe {
            // This is okay because `Vec` always returns the same value for
            // `.as_ref()`.
            Self::new(indices)
        }
    }
}

macro_rules! impl_tryfrom_array_for_sliceinfo {
    ($len:expr) => {
        impl<Din, Dout> TryFrom<[SliceInfoElem; $len]>
            for SliceInfo<[SliceInfoElem; $len], Din, Dout>
        where
            Din: Dimension,
            Dout: Dimension,
        {
            type Error = ShapeError;

            fn try_from(
                indices: [SliceInfoElem; $len],
            ) -> Result<SliceInfo<[SliceInfoElem; $len], Din, Dout>, ShapeError> {
                unsafe {
                    // This is okay because `[SliceInfoElem; N]` always returns
                    // the same value for `.as_ref()`.
                    Self::new(indices)
                }
            }
        }
    };
}
impl_tryfrom_array_for_sliceinfo!(0);
impl_tryfrom_array_for_sliceinfo!(1);
impl_tryfrom_array_for_sliceinfo!(2);
impl_tryfrom_array_for_sliceinfo!(3);
impl_tryfrom_array_for_sliceinfo!(4);
impl_tryfrom_array_for_sliceinfo!(5);
impl_tryfrom_array_for_sliceinfo!(6);
impl_tryfrom_array_for_sliceinfo!(7);
impl_tryfrom_array_for_sliceinfo!(8);

impl<T, Din, Dout> AsRef<[SliceInfoElem]> for SliceInfo<T, Din, Dout>
where
    T: AsRef<[SliceInfoElem]>,
    Din: Dimension,
    Dout: Dimension,
{
    fn as_ref(&self) -> &[SliceInfoElem]
    {
        self.indices.as_ref()
    }
}

impl<'a, T, Din, Dout> From<&'a SliceInfo<T, Din, Dout>> for SliceInfo<&'a [SliceInfoElem], Din, Dout>
where
    T: AsRef<[SliceInfoElem]>,
    Din: Dimension,
    Dout: Dimension,
{
    fn from(info: &'a SliceInfo<T, Din, Dout>) -> SliceInfo<&'a [SliceInfoElem], Din, Dout>
    {
        SliceInfo {
            in_dim: info.in_dim,
            out_dim: info.out_dim,
            indices: info.indices.as_ref(),
        }
    }
}

impl<T, Din, Dout> Copy for SliceInfo<T, Din, Dout>
where
    T: Copy,
    Din: Dimension,
    Dout: Dimension,
{
}

impl<T, Din, Dout> Clone for SliceInfo<T, Din, Dout>
where
    T: Clone,
    Din: Dimension,
    Dout: Dimension,
{
    fn clone(&self) -> Self
    {
        SliceInfo {
            in_dim: PhantomData,
            out_dim: PhantomData,
            indices: self.indices.clone(),
        }
    }
}

/// Trait for determining dimensionality of input and output for [`s!`] macro.
#[doc(hidden)]
pub trait SliceNextDim
{
    /// Number of dimensions that this slicing argument consumes in the input array.
    type InDim: Dimension;
    /// Number of dimensions that this slicing argument produces in the output array.
    type OutDim: Dimension;

    fn next_in_dim<D>(&self, _: PhantomData<D>) -> PhantomData<<D as DimAdd<Self::InDim>>::Output>
    where D: Dimension + DimAdd<Self::InDim>
    {
        PhantomData
    }

    fn next_out_dim<D>(&self, _: PhantomData<D>) -> PhantomData<<D as DimAdd<Self::OutDim>>::Output>
    where D: Dimension + DimAdd<Self::OutDim>
    {
        PhantomData
    }
}

macro_rules! impl_slicenextdim {
    (($($generics:tt)*), $self:ty, $in:ty, $out:ty) => {
        impl<$($generics)*> SliceNextDim for $self {
            type InDim = $in;
            type OutDim = $out;
        }
    };
}

impl_slicenextdim!((), isize, Ix1, Ix0);
impl_slicenextdim!((), usize, Ix1, Ix0);
impl_slicenextdim!((), i32, Ix1, Ix0);

impl_slicenextdim!((T), Range<T>, Ix1, Ix1);
impl_slicenextdim!((T), RangeInclusive<T>, Ix1, Ix1);
impl_slicenextdim!((T), RangeFrom<T>, Ix1, Ix1);
impl_slicenextdim!((T), RangeTo<T>, Ix1, Ix1);
impl_slicenextdim!((T), RangeToInclusive<T>, Ix1, Ix1);
impl_slicenextdim!((), RangeFull, Ix1, Ix1);
impl_slicenextdim!((), Slice, Ix1, Ix1);

impl_slicenextdim!((), NewAxis, Ix0, Ix1);

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices/new-axes, separated by comma,
/// with optional step sizes that are separated from the range by a semicolon.
/// It is converted into a [`SliceInfo`] instance.
///
/// Each range/slice/index uses signed indices, where a negative value is
/// counted from the end of the axis. Step sizes are also signed and may be
/// negative, but must not be zero.
///
/// The syntax is `s![` *[ elem [, elem [ , ... ] ] ]* `]`, where *elem* is any
/// of the following:
///
/// * *index*: an index to use for taking a subview with respect to that axis.
///   (The index is selected. The axis is removed except with
///   [`.slice_collapse()`].)
/// * *range*: a range with step size 1 to use for slicing that axis.
/// * *range* `;` *step*: a range with step size *step* to use for slicing that axis.
/// * *slice*: a [`Slice`] instance to use for slicing that axis.
/// * *slice* `;` *step*: a range constructed from a [`Slice`] instance,
///   multiplying the step size by *step*, to use for slicing that axis.
/// * *new-axis*: a [`NewAxis`] instance that represents the creation of a new axis.
///   (Except for [`.slice_collapse()`], which panics on [`NewAxis`] elements.)
///
/// The number of *elem*, not including *new-axis*, must match the
/// number of axes in the array. *index*, *range*, *slice*, *step*, and
/// *new-axis* can be expressions. *index* must be of type `isize`, `usize`, or
/// `i32`. *range* must be of type `Range<I>`, `RangeTo<I>`, `RangeFrom<I>`, or
/// `RangeFull` where `I` is `isize`, `usize`, or `i32`. *step* must be a type
/// that can be converted to `isize` with the `as` keyword.
///
/// For example, `s![0..4;2, 6, 1..5, NewAxis]` is a slice of the first axis
/// for 0..4 with step size 2, a subview of the second axis at index 6, a slice
/// of the third axis for 1..5 with default step size 1, and a new axis of
/// length 1 at the end of the shape. The input array must have 3 dimensions.
/// The resulting slice would have shape `[2, 4, 1]` for [`.slice()`],
/// [`.slice_mut()`], and [`.slice_move()`], while [`.slice_collapse()`] would
/// panic. Without the `NewAxis`, i.e. `s![0..4;2, 6, 1..5]`,
/// [`.slice_collapse()`] would result in an array of shape `[2, 1, 4]`.
///
/// [`.slice()`]: crate::ArrayBase::slice
/// [`.slice_mut()`]: crate::ArrayBase::slice_mut
/// [`.slice_move()`]: crate::ArrayBase::slice_move
/// [`.slice_collapse()`]: crate::ArrayBase::slice_collapse
///
/// See also [*Slicing*](crate::ArrayBase#slicing).
///
/// # Example
///
/// ```
/// use ndarray::{s, Array2, ArrayView2};
///
/// fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
///     -4. * &v.slice(s![1..-1, 1..-1])
///     + v.slice(s![ ..-2, 1..-1])
///     + v.slice(s![1..-1,  ..-2])
///     + v.slice(s![1..-1, 2..  ])
///     + v.slice(s![2..  , 1..-1])
/// }
/// # fn main() { let _ = laplacian; }
/// ```
///
/// # Negative *step*
///
/// The behavior of negative *step* arguments is most easily understood with
/// slicing as a two-step process:
///
/// 1. First, perform a slice with *range*.
///
/// 2. If *step* is positive, start with the front of the slice; if *step* is
///    negative, start with the back of the slice. Then, add *step* until
///    reaching the other end of the slice (inclusive).
///
/// An equivalent way to think about step 2 is, "If *step* is negative, reverse
/// the slice. Start at the front of the (possibly reversed) slice, and add
/// *step.abs()* until reaching the back of the slice (inclusive)."
///
/// For example,
///
/// ```
/// # use ndarray::prelude::*;
/// #
/// # fn main() {
/// let arr = array![0, 1, 2, 3];
/// assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
/// assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! s(
    // convert a..b;c into @convert(a..b, c), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextDim::next_in_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextDim::next_out_dim(&r, $out_dim);
                (
                    [$($stack)* $crate::s!(@convert r, $s)],
                    in_dim,
                    out_dim,
                )
            }
        }
    };
    // convert a..b into @convert(a..b), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextDim::next_in_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextDim::next_out_dim(&r, $out_dim);
                (
                    [$($stack)* $crate::s!(@convert r)],
                    in_dim,
                    out_dim,
                )
            }
        }
    };
    // convert a..b;c into @convert(a..b, c), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r;$s]
    };
    // convert a..b into @convert(a..b), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r]
    };
    // convert a..b;c into @convert(a..b, c)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextDim::next_in_dim(&r, $in_dim),
                   $crate::SliceNextDim::next_out_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r, $s),]
                   $($t)*
                ]
            }
        }
    };
    // convert a..b into @convert(a..b)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextDim::next_in_dim(&r, $in_dim),
                   $crate::SliceNextDim::next_out_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r),]
                   $($t)*
                ]
            }
        }
    };
    // empty call, i.e. `s![]`
    (@parse ::core::marker::PhantomData::<$crate::Ix0>, ::core::marker::PhantomData::<$crate::Ix0>, []) => {
        (
            [],
            ::core::marker::PhantomData::<$crate::Ix0>,
            ::core::marker::PhantomData::<$crate::Ix0>,
        )
    };
    // Catch-all clause for syntax errors
    (@parse $($t:tt)*) => { compile_error!("Invalid syntax in s![] call.") };
    // convert range/index/new-axis into SliceInfoElem
    (@convert $r:expr) => {
        <$crate::SliceInfoElem as ::core::convert::From<_>>::from($r)
    };
    // convert range/index/new-axis and step into SliceInfoElem
    (@convert $r:expr, $s:expr) => {
        <$crate::SliceInfoElem as ::core::convert::From<_>>::from(
            <$crate::Slice as ::core::convert::From<_>>::from($r).step_by($s as isize)
        )
    };
    ($($t:tt)*) => {
        {
            let (indices, in_dim, out_dim) = $crate::s![@parse
                ::core::marker::PhantomData::<$crate::Ix0>,
                ::core::marker::PhantomData::<$crate::Ix0>,
                []
                $($t)*
            ];
            // Safety: The `s![@parse ...]` above always constructs the correct
            // values to meet the constraints of `SliceInfo::new_unchecked`.
            #[allow(unsafe_code)]
            unsafe {
                $crate::SliceInfo::new_unchecked(indices, in_dim, out_dim)
            }
        }
    };
);

/// Slicing information describing multiple mutable, disjoint slices.
///
/// It's unfortunate that we need `'a` and `A` to be parameters of the trait,
/// but they're necessary until Rust supports generic associated types.
pub trait MultiSliceArg<'a, A, D>
where
    A: 'a,
    D: Dimension,
{
    /// The type of the slices created by `.multi_slice_move()`.
    type Output;

    /// Split the view into multiple disjoint slices.
    ///
    /// **Panics** if performing any individual slice panics or if the slices
    /// are not disjoint (i.e. if they intersect).
    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output;

    private_decl! {}
}

impl<'a, A, D> MultiSliceArg<'a, A, D> for ()
where
    A: 'a,
    D: Dimension,
{
    type Output = ();

    fn multi_slice_move(&self, _view: ArrayViewMut<'a, A, D>) -> Self::Output {}

    private_impl! {}
}

impl<'a, A, D, I0> MultiSliceArg<'a, A, D> for (I0,)
where
    A: 'a,
    D: Dimension,
    I0: SliceArg<D>,
{
    type Output = (ArrayViewMut<'a, A, I0::OutDim>,);

    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output
    {
        (view.slice_move(&self.0),)
    }

    private_impl! {}
}

macro_rules! impl_multislice_tuple {
    ([$($but_last:ident)*] $last:ident) => {
        impl_multislice_tuple!(@def_impl ($($but_last,)* $last,), [$($but_last)*] $last);
    };
    (@def_impl ($($all:ident,)*), [$($but_last:ident)*] $last:ident) => {
        impl<'a, A, D, $($all,)*> MultiSliceArg<'a, A, D> for ($($all,)*)
        where
            A: 'a,
            D: Dimension,
            $($all: SliceArg<D>,)*
        {
            type Output = ($(ArrayViewMut<'a, A, $all::OutDim>,)*);

            fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output {
                #[allow(non_snake_case)]
                let ($($all,)*) = self;

                let shape = view.raw_dim();
                assert!(!impl_multislice_tuple!(@intersects_self &shape, ($($all,)*)));

                let raw_view = view.into_raw_view_mut();
                unsafe {
                    (
                        $(raw_view.clone().slice_move($but_last).deref_into_view_mut(),)*
                        raw_view.slice_move($last).deref_into_view_mut(),
                    )
                }
            }

            private_impl! {}
        }
    };
    (@intersects_self $shape:expr, ($head:expr,)) => {
        false
    };
    (@intersects_self $shape:expr, ($head:expr, $($tail:expr,)*)) => {
        $(slices_intersect($shape, $head, $tail)) ||*
            || impl_multislice_tuple!(@intersects_self $shape, ($($tail,)*))
    };
}

impl_multislice_tuple!([I0] I1);
impl_multislice_tuple!([I0 I1] I2);
impl_multislice_tuple!([I0 I1 I2] I3);
impl_multislice_tuple!([I0 I1 I2 I3] I4);
impl_multislice_tuple!([I0 I1 I2 I3 I4] I5);

impl<'a, A, D, T> MultiSliceArg<'a, A, D> for &T
where
    A: 'a,
    D: Dimension,
    T: MultiSliceArg<'a, A, D>,
{
    type Output = T::Output;

    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output
    {
        T::multi_slice_move(self, view)
    }

    private_impl! {}
}
