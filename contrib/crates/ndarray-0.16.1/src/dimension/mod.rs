// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::error::{from_kind, ErrorKind, ShapeError};
use crate::shape_builder::Strides;
use crate::slice::SliceArg;
use crate::{Ix, Ixs, Slice, SliceInfoElem};
use num_integer::div_floor;

pub use self::axes::{Axes, AxisDescription};
pub use self::axis::Axis;
pub use self::broadcast::DimMax;
pub use self::conversion::IntoDimension;
pub use self::dim::*;
pub use self::dimension_trait::Dimension;
pub use self::dynindeximpl::IxDynImpl;
pub use self::ndindex::NdIndex;
pub use self::ops::DimAdd;
pub use self::remove_axis::RemoveAxis;

pub(crate) use self::axes::axes_of;
pub(crate) use self::reshape::reshape_dim;

use std::mem;

#[macro_use]
mod macros;
mod axes;
mod axis;
pub(crate) mod broadcast;
mod conversion;
pub mod dim;
mod dimension_trait;
mod dynindeximpl;
mod ndindex;
mod ops;
mod remove_axis;
pub(crate) mod reshape;
mod sequence;

/// Calculate offset from `Ix` stride converting sign properly
#[inline(always)]
pub fn stride_offset(n: Ix, stride: Ix) -> isize
{
    (n as isize) * (stride as Ixs)
}

/// Check whether the given `dim` and `stride` lead to overlapping indices
///
/// There is overlap if, when iterating through the dimensions in order of
/// increasing stride, the current stride is less than or equal to the maximum
/// possible offset along the preceding axes. (Axes of length ≤1 are ignored.)
pub(crate) fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool
{
    let order = strides._fastest_varying_stride_order();
    let mut sum_prev_offsets = 0;
    for &index in order.slice() {
        let d = dim[index];
        let s = (strides[index] as isize).abs();
        match d {
            0 => return false,
            1 => {}
            _ => {
                if s <= sum_prev_offsets {
                    return true;
                }
                sum_prev_offsets += (d - 1) as isize * s;
            }
        }
    }
    false
}

/// Returns the `size` of the `dim`, checking that the product of non-zero axis
/// lengths does not exceed `isize::MAX`.
///
/// If `size_of_checked_shape(dim)` returns `Ok(size)`, the data buffer is a
/// slice or `Vec` of length `size`, and `strides` are created with
/// `self.default_strides()` or `self.fortran_strides()`, then the invariants
/// are met to construct an array from the data buffer, `dim`, and `strides`.
/// (The data buffer being a slice or `Vec` guarantees that it contains no more
/// than `isize::MAX` bytes.)
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError>
{
    let size_nonzero = dim
        .slice()
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    if size_nonzero > isize::MAX as usize {
        Err(from_kind(ErrorKind::Overflow))
    } else {
        Ok(dim.size())
    }
}

/// Select how aliasing is checked
///
/// For owned or mutable data:
///
///   The strides must not allow any element to be referenced by two different indices.
///
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum CanIndexCheckMode
{
    /// Owned or mutable: No aliasing
    OwnedMutable,
    /// Aliasing
    ReadOnly,
}

/// Checks whether the given data and dimension meet the invariants of the
/// `ArrayBase` type, assuming the strides are created using
/// `dim.default_strides()` or `dim.fortran_strides()`.
///
/// To meet the invariants,
///
/// 1. The product of non-zero axis lengths must not exceed `isize::MAX`.
///
/// 2. The result of `dim.size()` (assuming no overflow) must be less than or
///    equal to the length of the slice.
///
///    (Since `dim.default_strides()` and `dim.fortran_strides()` always return
///    contiguous strides for non-empty arrays, this ensures that for non-empty
///    arrays the difference between the least address and greatest address
///    accessible by moving along all axes is < the length of the slice. Since
///    `dim.default_strides()` and `dim.fortran_strides()` always return all
///    zero strides for empty arrays, this ensures that for empty arrays the
///    difference between the least address and greatest address accessible by
///    moving along all axes is ≤ the length of the slice.)
///
/// Note that since slices cannot contain more than `isize::MAX` bytes,
/// conditions 1 and 2 are sufficient to guarantee that the offset in units of
/// `A` and in units of bytes between the least address and greatest address
/// accessible by moving along all axes does not exceed `isize::MAX`.
pub(crate) fn can_index_slice_with_strides<A, D: Dimension>(
    data: &[A], dim: &D, strides: &Strides<D>, mode: CanIndexCheckMode,
) -> Result<(), ShapeError>
{
    if let Strides::Custom(strides) = strides {
        can_index_slice(data, dim, strides, mode)
    } else {
        // contiguous shapes: never aliasing, mode does not matter
        can_index_slice_not_custom(data.len(), dim)
    }
}

pub(crate) fn can_index_slice_not_custom<D: Dimension>(data_len: usize, dim: &D) -> Result<(), ShapeError>
{
    // Condition 1.
    let len = size_of_shape_checked(dim)?;
    // Condition 2.
    if len > data_len {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    Ok(())
}

/// Returns the absolute difference in units of `A` between least and greatest
/// address accessible by moving along all axes.
///
/// Returns `Ok` only if
///
/// 1. The ndim of `dim` and `strides` is the same.
///
/// 2. The absolute difference in units of `A` and in units of bytes between
///    the least address and greatest address accessible by moving along all axes
///    does not exceed `isize::MAX`.
///
/// 3. The product of non-zero axis lengths does not exceed `isize::MAX`. (This
///    also implies that the length of any individual axis does not exceed
///    `isize::MAX`.)
pub fn max_abs_offset_check_overflow<A, D>(dim: &D, strides: &D) -> Result<usize, ShapeError>
where D: Dimension
{
    max_abs_offset_check_overflow_impl(mem::size_of::<A>(), dim, strides)
}

fn max_abs_offset_check_overflow_impl<D>(elem_size: usize, dim: &D, strides: &D) -> Result<usize, ShapeError>
where D: Dimension
{
    // Condition 1.
    if dim.ndim() != strides.ndim() {
        return Err(from_kind(ErrorKind::IncompatibleLayout));
    }

    // Condition 3.
    let _ = size_of_shape_checked(dim)?;

    // Determine absolute difference in units of `A` between least and greatest
    // address accessible by moving along all axes.
    let max_offset: usize = izip!(dim.slice(), strides.slice())
        .try_fold(0usize, |acc, (&d, &s)| {
            let s = s as isize;
            // Calculate maximum possible absolute movement along this axis.
            let off = d.saturating_sub(1).checked_mul(s.unsigned_abs())?;
            acc.checked_add(off)
        })
        .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    // Condition 2a.
    if max_offset > isize::MAX as usize {
        return Err(from_kind(ErrorKind::Overflow));
    }

    // Determine absolute difference in units of bytes between least and
    // greatest address accessible by moving along all axes
    let max_offset_bytes = max_offset
        .checked_mul(elem_size)
        .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    // Condition 2b.
    if max_offset_bytes > isize::MAX as usize {
        return Err(from_kind(ErrorKind::Overflow));
    }

    Ok(max_offset)
}

/// Checks whether the given data, dimension, and strides meet the invariants
/// of the `ArrayBase` type (except for checking ownership of the data).
///
/// To meet the invariants,
///
/// 1. The ndim of `dim` and `strides` must be the same.
///
/// 2. The product of non-zero axis lengths must not exceed `isize::MAX`.
///
/// 3. If the array will be empty (any axes are zero-length), the difference
///    between the least address and greatest address accessible by moving
///    along all axes must be ≤ `data.len()`. (It's fine in this case to move
///    one byte past the end of the slice since the pointers will be offset but
///    never dereferenced.)
///
///    If the array will not be empty, the difference between the least address
///    and greatest address accessible by moving along all axes must be <
///    `data.len()`. This and #3 ensure that all dereferenceable pointers point
///    to elements within the slice.
///
/// 4. The strides must not allow any element to be referenced by two different
///    indices.
///
/// Note that since slices cannot contain more than `isize::MAX` bytes,
/// condition 4 is sufficient to guarantee that the absolute difference in
/// units of `A` and in units of bytes between the least address and greatest
/// address accessible by moving along all axes does not exceed `isize::MAX`.
///
/// Warning: This function is sufficient to check the invariants of ArrayBase
/// only if the pointer to the first element of the array is chosen such that
/// the element with the smallest memory address is at the start of the
/// allocation. (In other words, the pointer to the first element of the array
/// must be computed using `offset_from_low_addr_ptr_to_logical_ptr` so that
/// negative strides are correctly handled.)
pub(crate) fn can_index_slice<A, D: Dimension>(
    data: &[A], dim: &D, strides: &D, mode: CanIndexCheckMode,
) -> Result<(), ShapeError>
{
    // Check conditions 1 and 2 and calculate `max_offset`.
    let max_offset = max_abs_offset_check_overflow::<A, _>(dim, strides)?;
    can_index_slice_impl(max_offset, data.len(), dim, strides, mode)
}

fn can_index_slice_impl<D: Dimension>(
    max_offset: usize, data_len: usize, dim: &D, strides: &D, mode: CanIndexCheckMode,
) -> Result<(), ShapeError>
{
    // Check condition 3.
    let is_empty = dim.slice().iter().any(|&d| d == 0);
    if is_empty && max_offset > data_len {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    if !is_empty && max_offset >= data_len {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }

    // Check condition 4.
    if !is_empty && mode != CanIndexCheckMode::ReadOnly && dim_stride_overlap(dim, strides) {
        return Err(from_kind(ErrorKind::Unsupported));
    }

    Ok(())
}

/// Stride offset checked general version (slices)
#[inline]
pub fn stride_offset_checked(dim: &[Ix], strides: &[Ix], index: &[Ix]) -> Option<isize>
{
    if index.len() != dim.len() {
        return None;
    }
    let mut offset = 0;
    for (&d, &i, &s) in izip!(dim, index, strides) {
        if i >= d {
            return None;
        }
        offset += stride_offset(i, s);
    }
    Some(offset)
}

/// Checks if strides are non-negative.
pub fn strides_non_negative<D>(strides: &D) -> Result<(), ShapeError>
where D: Dimension
{
    for &stride in strides.slice() {
        if (stride as isize) < 0 {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    }
    Ok(())
}

/// Implementation-specific extensions to `Dimension`
pub trait DimensionExt
{
    // note: many extensions go in the main trait if they need to be special-
    // cased per dimension
    /// Get the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[track_caller]
    fn axis(&self, axis: Axis) -> Ix;

    /// Set the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[track_caller]
    fn set_axis(&mut self, axis: Axis, value: Ix);
}

impl<D> DimensionExt for D
where D: Dimension
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix
    {
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix)
    {
        self[axis.index()] = value;
    }
}

impl DimensionExt for [Ix]
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix
    {
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix)
    {
        self[axis.index()] = value;
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Panics** if `index` is larger than the size of the axis
#[track_caller]
// FIXME: Move to Dimension trait
pub fn do_collapse_axis<D: Dimension>(dims: &mut D, strides: &D, axis: usize, index: usize) -> isize
{
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    ndassert!(
        index < dim,
        "collapse_axis: Index {} must be less than axis length {} for \
         array with shape {:?}",
        index,
        dim,
        *dims
    );
    dims.slice_mut()[axis] = 1;
    stride_offset(index, stride)
}

/// Compute the equivalent unsigned index given the axis length and signed index.
#[inline]
pub fn abs_index(len: Ix, index: Ixs) -> Ix
{
    if index < 0 {
        len - (-index as Ix)
    } else {
        index as Ix
    }
}

/// Determines nonnegative start and end indices, and performs sanity checks.
///
/// The return value is (start, end, step).
///
/// **Panics** if stride is 0 or if any index is out of bounds.
#[track_caller]
fn to_abs_slice(axis_len: usize, slice: Slice) -> (usize, usize, isize)
{
    let Slice { start, end, step } = slice;
    let start = abs_index(axis_len, start);
    let mut end = abs_index(axis_len, end.unwrap_or(axis_len as isize));
    if end < start {
        end = start;
    }
    ndassert!(
        start <= axis_len,
        "Slice begin {} is past end of axis of length {}",
        start,
        axis_len,
    );
    ndassert!(
        end <= axis_len,
        "Slice end {} is past end of axis of length {}",
        end,
        axis_len,
    );
    ndassert!(step != 0, "Slice stride must not be zero");
    (start, end, step)
}

/// This function computes the offset from the lowest address element to the
/// logically first element.
pub fn offset_from_low_addr_ptr_to_logical_ptr<D: Dimension>(dim: &D, strides: &D) -> usize
{
    let offset = izip!(dim.slice(), strides.slice()).fold(0, |_offset, (&d, &s)| {
        let s = s as isize;
        if s < 0 && d > 1 {
            _offset - s * (d as isize - 1)
        } else {
            _offset
        }
    });
    debug_assert!(offset >= 0);
    offset as usize
}

/// Modify dimension, stride and return data pointer offset
///
/// **Panics** if stride is 0 or if any index is out of bounds.
#[track_caller]
pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize
{
    let (start, end, step) = to_abs_slice(*dim, slice);

    let m = end - start;
    let s = (*stride) as isize;

    // Compute data pointer offset.
    let offset = if m == 0 {
        // In this case, the resulting array is empty, so we *can* avoid performing a nonzero
        // offset.
        //
        // In two special cases (which are the true reason for this `m == 0` check), we *must* avoid
        // the nonzero offset corresponding to the general case.
        //
        // * When `end == 0 && step < 0`. (These conditions imply that `m == 0` since `to_abs_slice`
        //   ensures that `0 <= start <= end`.) We cannot execute `stride_offset(end - 1, *stride)`
        //   because the `end - 1` would underflow.
        //
        // * When `start == *dim && step > 0`. (These conditions imply that `m == 0` since
        //   `to_abs_slice` ensures that `start <= end <= *dim`.) We cannot use the offset returned
        //   by `stride_offset(start, *stride)` because that would be past the end of the axis.
        0
    } else if step < 0 {
        // When the step is negative, the new first element is `end - 1`, not `start`, since the
        // direction is reversed.
        stride_offset(end - 1, *stride)
    } else {
        stride_offset(start, *stride)
    };

    // Update dimension.
    let abs_step = step.unsigned_abs();
    *dim = if abs_step == 1 {
        m
    } else {
        let d = m / abs_step;
        let r = m % abs_step;
        d + if r > 0 { 1 } else { 0 }
    };

    // Update stride. The additional check is necessary to avoid possible
    // overflow in the multiplication.
    *stride = if *dim <= 1 { 0 } else { (s * step) as usize };

    offset
}

/// Solves `a * x + b * y = gcd(a, b)` for `x`, `y`, and `gcd(a, b)`.
///
/// Returns `(g, (x, y))`, where `g` is `gcd(a, b)`, and `g` is always
/// nonnegative.
///
/// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
fn extended_gcd(a: isize, b: isize) -> (isize, (isize, isize))
{
    if a == 0 {
        (b.abs(), (0, b.signum()))
    } else if b == 0 {
        (a.abs(), (a.signum(), 0))
    } else {
        let mut r = (a, b);
        let mut s = (1, 0);
        let mut t = (0, 1);
        while r.1 != 0 {
            let q = r.0 / r.1;
            r = (r.1, r.0 - q * r.1);
            s = (s.1, s.0 - q * s.1);
            t = (t.1, t.0 - q * t.1);
        }
        if r.0 > 0 {
            (r.0, (s.0, t.0))
        } else {
            (-r.0, (-s.0, -t.0))
        }
    }
}

/// Solves `a * x + b * y = c` for `x` where `a`, `b`, `c`, `x`, and `y` are
/// integers.
///
/// If the return value is `Some((x0, xd))`, there is a solution. `xd` is
/// always positive. Solutions `x` are given by `x0 + xd * t` where `t` is any
/// integer. The value of `y` for any `x` is then `y = (c - a * x) / b`.
///
/// If the return value is `None`, no solutions exist.
///
/// **Note** `a` and `b` must be nonzero.
///
/// See https://en.wikipedia.org/wiki/Diophantine_equation#One_equation
/// and https://math.stackexchange.com/questions/1656120#1656138
fn solve_linear_diophantine_eq(a: isize, b: isize, c: isize) -> Option<(isize, isize)>
{
    debug_assert_ne!(a, 0);
    debug_assert_ne!(b, 0);
    let (g, (u, _)) = extended_gcd(a, b);
    if c % g == 0 {
        Some((c / g * u, (b / g).abs()))
    } else {
        None
    }
}

/// Returns `true` if two (finite length) arithmetic sequences intersect.
///
/// `min*` and `max*` are the (inclusive) bounds of the sequences, and they
/// must be elements in the sequences. `step*` are the steps between
/// consecutive elements (the sign is irrelevant).
///
/// **Note** `step1` and `step2` must be nonzero.
fn arith_seq_intersect((min1, max1, step1): (isize, isize, isize), (min2, max2, step2): (isize, isize, isize)) -> bool
{
    debug_assert!(max1 >= min1);
    debug_assert!(max2 >= min2);
    debug_assert_eq!((max1 - min1) % step1, 0);
    debug_assert_eq!((max2 - min2) % step2, 0);

    // Handle the easy case where we don't have to solve anything.
    if min1 > max2 || min2 > max1 {
        false
    } else {
        // The sign doesn't matter semantically, and it's mathematically convenient
        // for `step1` and `step2` to be positive.
        let step1 = step1.abs();
        let step2 = step2.abs();
        // Ignoring the min/max bounds, the sequences are
        //   a(x) = min1 + step1 * x
        //   b(y) = min2 + step2 * y
        //
        // For intersections a(x) = b(y), we have:
        //   min1 + step1 * x = min2 + step2 * y
        //   ⇒ -step1 * x + step2 * y = min1 - min2
        // which is a linear Diophantine equation.
        if let Some((x0, xd)) = solve_linear_diophantine_eq(-step1, step2, min1 - min2) {
            // Minimum of [min1, max1] ∩ [min2, max2]
            let min = ::std::cmp::max(min1, min2);
            // Maximum of [min1, max1] ∩ [min2, max2]
            let max = ::std::cmp::min(max1, max2);
            // The potential intersections are
            //   a(x) = min1 + step1 * (x0 + xd * t)
            // where `t` is any integer.
            //
            // There is an intersection in `[min, max]` if there exists an
            // integer `t` such that
            //   min ≤ a(x) ≤ max
            //   ⇒ min ≤ min1 + step1 * (x0 + xd * t) ≤ max
            //   ⇒ min ≤ min1 + step1 * x0 + step1 * xd * t ≤ max
            //   ⇒ min - min1 - step1 * x0 ≤ (step1 * xd) * t ≤ max - min1 - step1 * x0
            //
            // Therefore, the least possible intersection `a(x)` that is ≥ `min` has
            //   t = ⌈(min - min1 - step1 * x0) / (step1 * xd)⌉
            // If this `a(x) is also ≤ `max`, then there is an intersection in `[min, max]`.
            //
            // The greatest possible intersection `a(x)` that is ≤ `max` has
            //   t = ⌊(max - min1 - step1 * x0) / (step1 * xd)⌋
            // If this `a(x) is also ≥ `min`, then there is an intersection in `[min, max]`.
            min1 + step1 * (x0 - xd * div_floor(min - min1 - step1 * x0, -step1 * xd)) <= max
                || min1 + step1 * (x0 + xd * div_floor(max - min1 - step1 * x0, step1 * xd)) >= min
        } else {
            false
        }
    }
}

/// Returns the minimum and maximum values of the indices (inclusive).
///
/// If the slice is empty, then returns `None`, otherwise returns `Some((min, max))`.
fn slice_min_max(axis_len: usize, slice: Slice) -> Option<(usize, usize)>
{
    let (start, end, step) = to_abs_slice(axis_len, slice);
    if start == end {
        None
    } else if step > 0 {
        Some((start, end - 1 - (end - start - 1) % (step as usize)))
    } else {
        Some((start + (end - start - 1) % (-step as usize), end - 1))
    }
}

/// Returns `true` iff the slices intersect.
pub fn slices_intersect<D: Dimension>(dim: &D, indices1: impl SliceArg<D>, indices2: impl SliceArg<D>) -> bool
{
    debug_assert_eq!(indices1.in_ndim(), indices2.in_ndim());
    for (&axis_len, &si1, &si2) in izip!(
        dim.slice(),
        indices1.as_ref().iter().filter(|si| !si.is_new_axis()),
        indices2.as_ref().iter().filter(|si| !si.is_new_axis()),
    ) {
        // The slices do not intersect iff any pair of `SliceInfoElem` does not intersect.
        match (si1, si2) {
            (
                SliceInfoElem::Slice {
                    start: start1,
                    end: end1,
                    step: step1,
                },
                SliceInfoElem::Slice {
                    start: start2,
                    end: end2,
                    step: step2,
                },
            ) => {
                let (min1, max1) = match slice_min_max(axis_len, Slice::new(start1, end1, step1)) {
                    Some(m) => m,
                    None => return false,
                };
                let (min2, max2) = match slice_min_max(axis_len, Slice::new(start2, end2, step2)) {
                    Some(m) => m,
                    None => return false,
                };
                if !arith_seq_intersect((min1 as isize, max1 as isize, step1), (min2 as isize, max2 as isize, step2)) {
                    return false;
                }
            }
            (SliceInfoElem::Slice { start, end, step }, SliceInfoElem::Index(ind))
            | (SliceInfoElem::Index(ind), SliceInfoElem::Slice { start, end, step }) => {
                let ind = abs_index(axis_len, ind);
                let (min, max) = match slice_min_max(axis_len, Slice::new(start, end, step)) {
                    Some(m) => m,
                    None => return false,
                };
                if ind < min || ind > max || (ind - min) % step.unsigned_abs() != 0 {
                    return false;
                }
            }
            (SliceInfoElem::Index(ind1), SliceInfoElem::Index(ind2)) => {
                let ind1 = abs_index(axis_len, ind1);
                let ind2 = abs_index(axis_len, ind2);
                if ind1 != ind2 {
                    return false;
                }
            }
            (SliceInfoElem::NewAxis, _) | (_, SliceInfoElem::NewAxis) => unreachable!(),
        }
    }
    true
}

pub(crate) fn is_layout_c<D: Dimension>(dim: &D, strides: &D) -> bool
{
    if let Some(1) = D::NDIM {
        return strides[0] == 1 || dim[0] <= 1;
    }

    for &d in dim.slice() {
        if d == 0 {
            return true;
        }
    }

    let mut contig_stride = 1_isize;
    // check all dimensions -- a dimension of length 1 can have unequal strides
    for (&dim, &s) in izip!(dim.slice().iter().rev(), strides.slice().iter().rev()) {
        if dim != 1 {
            let s = s as isize;
            if s != contig_stride {
                return false;
            }
            contig_stride *= dim as isize;
        }
    }
    true
}

pub(crate) fn is_layout_f<D: Dimension>(dim: &D, strides: &D) -> bool
{
    if let Some(1) = D::NDIM {
        return strides[0] == 1 || dim[0] <= 1;
    }

    for &d in dim.slice() {
        if d == 0 {
            return true;
        }
    }

    let mut contig_stride = 1_isize;
    // check all dimensions -- a dimension of length 1 can have unequal strides
    for (&dim, &s) in izip!(dim.slice(), strides.slice()) {
        if dim != 1 {
            let s = s as isize;
            if s != contig_stride {
                return false;
            }
            contig_stride *= dim as isize;
        }
    }
    true
}

pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
where D: Dimension
{
    let into_len = dim.axis(into);
    let into_stride = strides.axis(into) as isize;
    let take_len = dim.axis(take);
    let take_stride = strides.axis(take) as isize;
    let merged_len = into_len * take_len;
    if take_len <= 1 {
        dim.set_axis(into, merged_len);
        dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
        true
    } else if into_len <= 1 {
        strides.set_axis(into, take_stride as usize);
        dim.set_axis(into, merged_len);
        dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
        true
    } else if take_stride == into_len as isize * into_stride {
        dim.set_axis(into, merged_len);
        dim.set_axis(take, 1);
        true
    } else {
        false
    }
}

/// Move the axis which has the smallest absolute stride and a length
/// greater than one to be the last axis.
pub fn move_min_stride_axis_to_last<D>(dim: &mut D, strides: &mut D)
where D: Dimension
{
    debug_assert_eq!(dim.ndim(), strides.ndim());
    match dim.ndim() {
        0 | 1 => {}
        2 =>
            if dim[1] <= 1 || dim[0] > 1 && (strides[0] as isize).abs() < (strides[1] as isize).abs() {
                dim.slice_mut().swap(0, 1);
                strides.slice_mut().swap(0, 1);
            },
        n => {
            if let Some(min_stride_axis) = (0..n)
                .filter(|&ax| dim[ax] > 1)
                .min_by_key(|&ax| (strides[ax] as isize).abs())
            {
                let last = n - 1;
                dim.slice_mut().swap(last, min_stride_axis);
                strides.slice_mut().swap(last, min_stride_axis);
            }
        }
    }
}

#[cfg(test)]
mod test
{
    use super::{
        arith_seq_intersect,
        can_index_slice,
        can_index_slice_not_custom,
        extended_gcd,
        max_abs_offset_check_overflow,
        slice_min_max,
        slices_intersect,
        solve_linear_diophantine_eq,
        CanIndexCheckMode,
        IntoDimension,
    };
    use crate::error::{from_kind, ErrorKind};
    use crate::slice::Slice;
    use crate::{Dim, Dimension, Ix0, Ix1, Ix2, Ix3, IxDyn, NewAxis};
    use num_integer::gcd;
    use quickcheck::{quickcheck, TestResult};

    #[test]
    fn slice_indexing_uncommon_strides()
    {
        let v: alloc::vec::Vec<_> = (0..12).collect();
        let dim = (2, 3, 2).into_dimension();
        let strides = (1, 2, 6).into_dimension();
        assert!(super::can_index_slice(&v, &dim, &strides, CanIndexCheckMode::OwnedMutable).is_ok());

        let strides = (2, 4, 12).into_dimension();
        assert_eq!(
            super::can_index_slice(&v, &dim, &strides, CanIndexCheckMode::OwnedMutable),
            Err(from_kind(ErrorKind::OutOfBounds))
        );
    }

    #[test]
    fn overlapping_strides_dim()
    {
        let dim = (2, 3, 2).into_dimension();
        let strides = (5, 2, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (-5isize as usize, 2, -1isize as usize).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 2, 1).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (6, -2isize as usize, 1).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 0, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (-6isize as usize, 0, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let dim = (2, 2).into_dimension();
        let strides = (3, 2).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (3, -2isize as usize).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
    }

    #[test]
    fn max_abs_offset_check_overflow_examples()
    {
        let dim = (1, ::std::isize::MAX as usize, 1).into_dimension();
        let strides = (1, 1, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap();
        let dim = (1, ::std::isize::MAX as usize, 2).into_dimension();
        let strides = (1, 1, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap_err();
        let dim = (0, 2, 2).into_dimension();
        let strides = (1, ::std::isize::MAX as usize, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap_err();
        let dim = (0, 2, 2).into_dimension();
        let strides = (1, ::std::isize::MAX as usize / 4, 1).into_dimension();
        max_abs_offset_check_overflow::<i32, _>(&dim, &strides).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix0()
    {
        can_index_slice::<i32, _>(&[1], &Ix0(), &Ix0(), CanIndexCheckMode::OwnedMutable).unwrap();
        can_index_slice::<i32, _>(&[], &Ix0(), &Ix0(), CanIndexCheckMode::OwnedMutable).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix1()
    {
        let mode = CanIndexCheckMode::OwnedMutable;
        can_index_slice::<i32, _>(&[], &Ix1(0), &Ix1(0), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix1(0), &Ix1(1), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix1(1), &Ix1(0), mode).unwrap_err();
        can_index_slice::<i32, _>(&[], &Ix1(1), &Ix1(1), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(0), mode).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(2), mode).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(-1isize as usize), mode).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(2), &Ix1(1), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(0), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(1), mode).unwrap();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(-1isize as usize), mode).unwrap();
    }

    #[test]
    fn can_index_slice_ix2()
    {
        let mode = CanIndexCheckMode::OwnedMutable;
        can_index_slice::<i32, _>(&[], &Ix2(0, 0), &Ix2(0, 0), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 0), &Ix2(2, 1), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 1), &Ix2(0, 0), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 1), &Ix2(2, 1), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 2), &Ix2(0, 0), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 2), &Ix2(2, 1), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix2(1, 2), &Ix2(5, 1), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix2(1, 2), &Ix2(5, 1), mode).unwrap();
        can_index_slice::<i32, _>(&[1, 2], &Ix2(1, 2), &Ix2(5, 2), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2, 3, 4, 5], &Ix2(2, 2), &Ix2(3, 1), mode).unwrap();
        can_index_slice::<i32, _>(&[1, 2, 3, 4], &Ix2(2, 2), &Ix2(3, 1), mode).unwrap_err();

        // aliasing strides: ok when readonly
        can_index_slice::<i32, _>(&[0; 4], &Ix2(2, 2), &Ix2(1, 1), CanIndexCheckMode::OwnedMutable).unwrap_err();
        can_index_slice::<i32, _>(&[0; 4], &Ix2(2, 2), &Ix2(1, 1), CanIndexCheckMode::ReadOnly).unwrap();
    }

    #[test]
    fn can_index_slice_ix3()
    {
        let mode = CanIndexCheckMode::OwnedMutable;
        can_index_slice::<i32, _>(&[], &Ix3(0, 0, 1), &Ix3(2, 1, 3), mode).unwrap();
        can_index_slice::<i32, _>(&[], &Ix3(1, 1, 1), &Ix3(2, 1, 3), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix3(1, 1, 1), &Ix3(2, 1, 3), mode).unwrap();
        can_index_slice::<i32, _>(&[1; 11], &Ix3(2, 2, 3), &Ix3(6, 3, 1), mode).unwrap_err();
        can_index_slice::<i32, _>(&[1; 12], &Ix3(2, 2, 3), &Ix3(6, 3, 1), mode).unwrap();
    }

    #[test]
    fn can_index_slice_zero_size_elem()
    {
        let mode = CanIndexCheckMode::OwnedMutable;
        can_index_slice::<(), _>(&[], &Ix1(0), &Ix1(1), mode).unwrap();
        can_index_slice::<(), _>(&[()], &Ix1(1), &Ix1(1), mode).unwrap();
        can_index_slice::<(), _>(&[(), ()], &Ix1(2), &Ix1(1), mode).unwrap();

        // These might seem okay because the element type is zero-sized, but
        // there could be a zero-sized type such that the number of instances
        // in existence are carefully controlled.
        can_index_slice::<(), _>(&[], &Ix1(1), &Ix1(1), mode).unwrap_err();
        can_index_slice::<(), _>(&[()], &Ix1(2), &Ix1(1), mode).unwrap_err();

        can_index_slice::<(), _>(&[(), ()], &Ix2(2, 1), &Ix2(1, 0), mode).unwrap();
        can_index_slice::<(), _>(&[], &Ix2(0, 2), &Ix2(0, 0), mode).unwrap();

        // This case would be probably be sound, but that's not entirely clear
        // and it's not worth the special case code.
        can_index_slice::<(), _>(&[], &Ix2(0, 2), &Ix2(2, 1), mode).unwrap_err();
    }

    quickcheck! {
        fn can_index_slice_not_custom_same_as_can_index_slice(data: alloc::vec::Vec<u8>, dim: alloc::vec::Vec<usize>) -> bool {
            let dim = IxDyn(&dim);
            let result = can_index_slice_not_custom(data.len(), &dim);
            if dim.size_checked().is_none() {
                // Avoid overflow `dim.default_strides()` or `dim.fortran_strides()`.
                result.is_err()
            } else {
                result == can_index_slice(&data, &dim, &dim.default_strides(), CanIndexCheckMode::OwnedMutable) &&
                    result == can_index_slice(&data, &dim, &dim.fortran_strides(), CanIndexCheckMode::OwnedMutable)
            }
        }
    }

    quickcheck! {
        // FIXME: This test can't handle larger values at the moment
        fn extended_gcd_solves_eq(a: i16, b: i16) -> bool {
            let (a, b) = (a as isize, b as isize);
            let (g, (x, y)) = extended_gcd(a, b);
            a * x + b * y == g
        }

        // FIXME: This test can't handle larger values at the moment
        fn extended_gcd_correct_gcd(a: i16, b: i16) -> bool {
            let (a, b) = (a as isize, b as isize);
            let (g, _) = extended_gcd(a, b);
            g == gcd(a, b)
        }
    }

    #[test]
    fn extended_gcd_zero()
    {
        assert_eq!(extended_gcd(0, 0), (0, (0, 0)));
        assert_eq!(extended_gcd(0, 5), (5, (0, 1)));
        assert_eq!(extended_gcd(5, 0), (5, (1, 0)));
        assert_eq!(extended_gcd(0, -5), (5, (0, -1)));
        assert_eq!(extended_gcd(-5, 0), (5, (-1, 0)));
    }

    quickcheck! {
        // FIXME: This test can't handle larger values at the moment
        fn solve_linear_diophantine_eq_solution_existence(
            a: i16, b: i16, c: i16
        ) -> TestResult {
            let (a, b, c) = (a as isize, b as isize, c as isize);

            if a == 0 || b == 0 {
                TestResult::discard()
            } else {
                TestResult::from_bool(
                    (c % gcd(a, b) == 0) == solve_linear_diophantine_eq(a, b, c).is_some()
                )
            }
        }

        // FIXME: This test can't handle larger values at the moment
        fn solve_linear_diophantine_eq_correct_solution(
            a: i8, b: i8, c: i8, t: i8
        ) -> TestResult {
            let (a, b, c, t) = (a as isize, b as isize, c as isize, t as isize);

            if a == 0 || b == 0 {
                TestResult::discard()
            } else {
                match solve_linear_diophantine_eq(a, b, c) {
                    Some((x0, xd)) => {
                        let x = x0 + xd * t;
                        let y = (c - a * x) / b;
                        TestResult::from_bool(a * x + b * y == c)
                    }
                    None => TestResult::discard(),
                }
            }
        }
    }

    quickcheck! {
        // FIXME: This test is extremely slow, even with i16 values, investigate
        fn arith_seq_intersect_correct(
            first1: i8, len1: i8, step1: i8,
            first2: i8, len2: i8, step2: i8
        ) -> TestResult {
            use std::cmp;

            let (len1, len2) = (len1 as isize, len2 as isize);
            let (first1, step1) = (first1 as isize, step1 as isize);
            let (first2, step2) = (first2 as isize, step2 as isize);

            if len1 == 0 || len2 == 0 {
                // This case is impossible to reach in `arith_seq_intersect()`
                // because the `min*` and `max*` arguments are inclusive.
                return TestResult::discard();
            }

            let len1 = len1.abs();
            let len2 = len2.abs();

            // Convert to `min*` and `max*` arguments for `arith_seq_intersect()`.
            let last1 = first1 + step1 * (len1 - 1);
            let (min1, max1) = (cmp::min(first1, last1), cmp::max(first1, last1));
            let last2 = first2 + step2 * (len2 - 1);
            let (min2, max2) = (cmp::min(first2, last2), cmp::max(first2, last2));

            // Naively determine if the sequences intersect.
            let seq1: alloc::vec::Vec<_> = (0..len1)
                .map(|n| first1 + step1 * n)
                .collect();
            let intersects = (0..len2)
                .map(|n| first2 + step2 * n)
                .any(|elem2| seq1.contains(&elem2));

            TestResult::from_bool(
                arith_seq_intersect(
                    (min1, max1, if step1 == 0 { 1 } else { step1 }),
                    (min2, max2, if step2 == 0 { 1 } else { step2 })
                ) == intersects
            )
        }
    }

    #[test]
    fn slice_min_max_empty()
    {
        assert_eq!(slice_min_max(0, Slice::new(0, None, 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(1, Some(1), 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(-1, Some(-1), 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(1, Some(1), -3)), None);
        assert_eq!(slice_min_max(10, Slice::new(-1, Some(-1), -3)), None);
    }

    #[test]
    fn slice_min_max_pos_step()
    {
        assert_eq!(slice_min_max(10, Slice::new(1, Some(8), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(9), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(8), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(9), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-2), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-1), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(-2), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(-1), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, None, 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, None, 3)), Some((1, 7)));
        assert_eq!(slice_min_max(11, Slice::new(1, None, 3)), Some((1, 10)));
        assert_eq!(slice_min_max(11, Slice::new(-10, None, 3)), Some((1, 10)));
    }

    #[test]
    fn slice_min_max_neg_step()
    {
        assert_eq!(slice_min_max(10, Slice::new(1, Some(8), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(2, Some(8), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(8), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-8, Some(8), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-2), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(2, Some(-2), -3)), Some((4, 7)));
        assert_eq!(
            slice_min_max(10, Slice::new(-9, Some(-2), -3)),
            Some((1, 7))
        );
        assert_eq!(
            slice_min_max(10, Slice::new(-8, Some(-2), -3)),
            Some((4, 7))
        );
        assert_eq!(slice_min_max(9, Slice::new(2, None, -3)), Some((2, 8)));
        assert_eq!(slice_min_max(9, Slice::new(-7, None, -3)), Some((2, 8)));
        assert_eq!(slice_min_max(9, Slice::new(3, None, -3)), Some((5, 8)));
        assert_eq!(slice_min_max(9, Slice::new(-6, None, -3)), Some((5, 8)));
    }

    #[test]
    fn slices_intersect_true()
    {
        assert!(slices_intersect(
            &Dim([4, 5]),
            s![NewAxis, .., NewAxis, ..],
            s![.., NewAxis, .., NewAxis]
        ));
        assert!(slices_intersect(
            &Dim([4, 5]),
            s![NewAxis, 0, ..],
            s![0, ..]
        ));
        assert!(slices_intersect(
            &Dim([4, 5]),
            s![..;2, ..],
            s![..;3, NewAxis, ..]
        ));
        assert!(slices_intersect(
            &Dim([4, 5]),
            s![.., ..;2],
            s![.., 1..;3, NewAxis]
        ));
        assert!(slices_intersect(&Dim([4, 10]), s![.., ..;9], s![.., 3..;6]));
    }

    #[test]
    fn slices_intersect_false()
    {
        assert!(!slices_intersect(
            &Dim([4, 5]),
            s![..;2, ..],
            s![NewAxis, 1..;2, ..]
        ));
        assert!(!slices_intersect(
            &Dim([4, 5]),
            s![..;2, NewAxis, ..],
            s![1..;3, ..]
        ));
        assert!(!slices_intersect(
            &Dim([4, 5]),
            s![.., ..;9],
            s![.., 3..;6, NewAxis]
        ));
    }
}
