// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Index, IndexMut};

use super::axes_of;
use super::conversion::Convert;
use super::ops::DimAdd;
use super::{stride_offset, stride_offset_checked};
use crate::itertools::{enumerate, zip};
use crate::IntoDimension;
use crate::RemoveAxis;
use crate::{ArrayView1, ArrayViewMut1};
use crate::{Axis, DimMax};
use crate::{Dim, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IxDynImpl, Ixs};

/// Array shape and index trait.
///
/// This trait defines a number of methods and operations that can be used on
/// dimensions and indices.
///
/// **Note:** *This trait can not be implemented outside the crate*
pub trait Dimension:
    Clone
    + Eq
    + Debug
    + Send
    + Sync
    + Default
    + IndexMut<usize, Output = usize>
    + Add<Self, Output = Self>
    + AddAssign
    + for<'x> AddAssign<&'x Self>
    + Sub<Self, Output = Self>
    + SubAssign
    + for<'x> SubAssign<&'x Self>
    + Mul<usize, Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign
    + for<'x> MulAssign<&'x Self>
    + MulAssign<usize>
    + DimMax<Ix0, Output = Self>
    + DimMax<Self, Output = Self>
    + DimMax<IxDyn, Output = IxDyn>
    + DimMax<<Self as Dimension>::Smaller, Output = Self>
    + DimMax<<Self as Dimension>::Larger, Output = <Self as Dimension>::Larger>
    + DimAdd<Self>
    + DimAdd<<Self as Dimension>::Smaller>
    + DimAdd<<Self as Dimension>::Larger>
    + DimAdd<Ix0, Output = Self>
    + DimAdd<Ix1, Output = <Self as Dimension>::Larger>
    + DimAdd<IxDyn, Output = IxDyn>
{
    /// For fixed-size dimension representations (e.g. `Ix2`), this should be
    /// `Some(ndim)`, and for variable-size dimension representations (e.g.
    /// `IxDyn`), this should be `None`.
    const NDIM: Option<usize>;
    /// Pattern matching friendly form of the dimension value.
    ///
    /// - For `Ix1`: `usize`,
    /// - For `Ix2`: `(usize, usize)`
    /// - and so on..
    /// - For `IxDyn`: `IxDyn`
    type Pattern: IntoDimension<Dim = Self> + Clone + Debug + PartialEq + Eq + Default;
    /// Next smaller dimension (if applicable)
    type Smaller: Dimension;
    /// Next larger dimension
    type Larger: Dimension + RemoveAxis;

    /// Returns the number of dimensions (number of axes).
    fn ndim(&self) -> usize;

    /// Convert the dimension into a pattern matching friendly value.
    fn into_pattern(self) -> Self::Pattern;

    /// Compute the size of the dimension (number of elements)
    fn size(&self) -> usize
    {
        self.slice().iter().product()
    }

    /// Compute the size while checking for overflow.
    fn size_checked(&self) -> Option<usize>
    {
        self.slice()
            .iter()
            .try_fold(1_usize, |s, &a| s.checked_mul(a))
    }

    #[doc(hidden)]
    fn slice(&self) -> &[Ix];

    #[doc(hidden)]
    fn slice_mut(&mut self) -> &mut [Ix];

    /// Borrow as a read-only array view.
    fn as_array_view(&self) -> ArrayView1<'_, Ix>
    {
        ArrayView1::from(self.slice())
    }

    /// Borrow as a read-write array view.
    fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, Ix>
    {
        ArrayViewMut1::from(self.slice_mut())
    }

    #[doc(hidden)]
    fn equal(&self, rhs: &Self) -> bool
    {
        self.slice() == rhs.slice()
    }

    /// Returns the strides for a standard layout array with the given shape.
    ///
    /// If the array is non-empty, the strides result in contiguous layout; if
    /// the array is empty, the strides are all zeros.
    #[doc(hidden)]
    fn default_strides(&self) -> Self
    {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        let mut strides = Self::zeros(self.ndim());
        // For empty arrays, use all zero strides.
        if self.slice().iter().all(|&d| d != 0) {
            let mut it = strides.slice_mut().iter_mut().rev();
            // Set first element to 1
            if let Some(rs) = it.next() {
                *rs = 1;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice().iter().rev()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    /// Returns the strides for a Fortran layout array with the given shape.
    ///
    /// If the array is non-empty, the strides result in contiguous layout; if
    /// the array is empty, the strides are all zeros.
    #[doc(hidden)]
    fn fortran_strides(&self) -> Self
    {
        // Compute fortran array strides
        // Shape (a, b, c) => Give strides (1, a, a * b)
        let mut strides = Self::zeros(self.ndim());
        // For empty arrays, use all zero strides.
        if self.slice().iter().all(|&d| d != 0) {
            let mut it = strides.slice_mut().iter_mut();
            // Set first element to 1
            if let Some(rs) = it.next() {
                *rs = 1;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    /// Creates a dimension of all zeros with the specified ndim.
    ///
    /// This method is useful for generalizing over fixed-size and
    /// variable-size dimension representations.
    ///
    /// **Panics** if `Self` has a fixed size that is not `ndim`.
    fn zeros(ndim: usize) -> Self;

    #[doc(hidden)]
    #[inline]
    fn first_index(&self) -> Option<Self>
    {
        for ax in self.slice().iter() {
            if *ax == 0 {
                return None;
            }
        }
        Some(Self::zeros(self.ndim()))
    }

    #[doc(hidden)]
    /// Iteration -- Use self as size, and return next index after `index`
    /// or None if there are no more.
    // FIXME: use &Self for index or even &mut?
    #[inline]
    fn next_for(&self, index: Self) -> Option<Self>
    {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                done = true;
                break;
            }
        }
        if done {
            Some(index)
        } else {
            None
        }
    }

    #[doc(hidden)]
    /// Iteration -- Use self as size, and create the next index after `index`
    /// Return false if iteration is done
    ///
    /// Next in f-order
    #[inline]
    fn next_for_f(&self, index: &mut Self) -> bool
    {
        let mut end_iteration = true;
        for (&dim, ix) in zip(self.slice(), index.slice_mut()) {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                end_iteration = false;
                break;
            }
        }
        !end_iteration
    }

    /// Returns `true` iff `strides1` and `strides2` are equivalent for the
    /// shape `self`.
    ///
    /// The strides are equivalent if, for each axis with length > 1, the
    /// strides are equal.
    ///
    /// Note: Returns `false` if any of the ndims don't match.
    #[doc(hidden)]
    fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
    where D: Dimension
    {
        let shape_ndim = self.ndim();
        shape_ndim == strides1.ndim()
            && shape_ndim == strides2.ndim()
            && izip!(self.slice(), strides1.slice(), strides2.slice())
                .all(|(&d, &s1, &s2)| d <= 1 || s1 as isize == s2 as isize)
    }

    #[doc(hidden)]
    /// Return stride offset for index.
    fn stride_offset(index: &Self, strides: &Self) -> isize
    {
        let mut offset = 0;
        for (&i, &s) in izip!(index.slice(), strides.slice()) {
            offset += stride_offset(i, s);
        }
        offset
    }

    #[doc(hidden)]
    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    {
        stride_offset_checked(self.slice(), strides.slice(), index.slice())
    }

    #[doc(hidden)]
    fn last_elem(&self) -> usize
    {
        if self.ndim() == 0 {
            0
        } else {
            self.slice()[self.ndim() - 1]
        }
    }

    #[doc(hidden)]
    fn set_last_elem(&mut self, i: usize)
    {
        let nd = self.ndim();
        self.slice_mut()[nd - 1] = i;
    }

    #[doc(hidden)]
    fn is_contiguous(dim: &Self, strides: &Self) -> bool
    {
        let defaults = dim.default_strides();
        if strides.equal(&defaults) {
            return true;
        }
        if dim.ndim() == 1 {
            // fast case for ndim == 1:
            // Either we have length <= 1, then stride is arbitrary,
            // or we have stride == 1 or stride == -1, but +1 case is already handled above.
            dim[0] <= 1 || strides[0] as isize == -1
        } else {
            let order = strides._fastest_varying_stride_order();
            let strides = strides.slice();

            let dim_slice = dim.slice();
            let mut cstride = 1;
            for &i in order.slice() {
                // a dimension of length 1 can have unequal strides
                if dim_slice[i] != 1 && (strides[i] as isize).unsigned_abs() != cstride {
                    return false;
                }
                cstride *= dim_slice[i];
            }
            true
        }
    }

    /// Return the axis ordering corresponding to the fastest variation
    /// (in ascending order).
    ///
    /// Assumes that no stride value appears twice.
    #[doc(hidden)]
    fn _fastest_varying_stride_order(&self) -> Self
    {
        let mut indices = self.clone();
        for (i, elt) in enumerate(indices.slice_mut()) {
            *elt = i;
        }
        let strides = self.slice();
        indices
            .slice_mut()
            .sort_by_key(|&i| (strides[i] as isize).abs());
        indices
    }

    /// Compute the minimum stride axis (absolute value), under the constraint
    /// that the length of the axis is > 1;
    #[doc(hidden)]
    fn min_stride_axis(&self, strides: &Self) -> Axis
    {
        let n = match self.ndim() {
            0 => panic!("min_stride_axis: Array must have ndim > 0"),
            1 => return Axis(0),
            n => n,
        };
        axes_of(self, strides)
            .rev()
            .min_by_key(|ax| ax.stride.abs())
            .map_or(Axis(n - 1), |ax| ax.axis)
    }

    /// Compute the maximum stride axis (absolute value), under the constraint
    /// that the length of the axis is > 1;
    #[doc(hidden)]
    fn max_stride_axis(&self, strides: &Self) -> Axis
    {
        match self.ndim() {
            0 => panic!("max_stride_axis: Array must have ndim > 0"),
            1 => return Axis(0),
            _ => {}
        }
        axes_of(self, strides)
            .filter(|ax| ax.len > 1)
            .max_by_key(|ax| ax.stride.abs())
            .map_or(Axis(0), |ax| ax.axis)
    }

    /// Convert the dimensional into a dynamic dimensional (IxDyn).
    fn into_dyn(self) -> IxDyn
    {
        IxDyn(self.slice())
    }

    #[doc(hidden)]
    fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self>
    {
        let mut s = Self::default();
        if s.ndim() == d.ndim() {
            for i in 0..d.ndim() {
                s[i] = d[i];
            }
            Some(s)
        } else {
            None
        }
    }

    #[doc(hidden)]
    fn insert_axis(&self, axis: Axis) -> Self::Larger;

    #[doc(hidden)]
    fn try_remove_axis(&self, axis: Axis) -> Self::Smaller;

    private_decl! {}
}

// Dimension impls

macro_rules! impl_insert_axis_array(
    ($n:expr) => (
        #[inline]
        fn insert_axis(&self, axis: Axis) -> Self::Larger {
            debug_assert!(axis.index() <= $n);
            let mut out = [1; $n + 1];
            out[0..axis.index()].copy_from_slice(&self.slice()[0..axis.index()]);
            out[axis.index()+1..=$n].copy_from_slice(&self.slice()[axis.index()..$n]);
            Dim(out)
        }
    );
);

impl Dimension for Dim<[Ix; 0]>
{
    const NDIM: Option<usize> = Some(0);
    type Pattern = ();
    type Smaller = Self;
    type Larger = Ix1;
    // empty product is 1 -> size is 1
    #[inline]
    fn ndim(&self) -> usize
    {
        0
    }
    #[inline]
    fn slice(&self) -> &[Ix]
    {
        &[]
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix]
    {
        &mut []
    }
    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self
    {
        Ix0()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {}
    #[inline]
    fn zeros(ndim: usize) -> Self
    {
        assert_eq!(ndim, 0);
        Self::default()
    }
    #[inline]
    fn next_for(&self, _index: Self) -> Option<Self>
    {
        None
    }
    impl_insert_axis_array!(0);
    #[inline]
    fn try_remove_axis(&self, _ignore: Axis) -> Self::Smaller
    {
        *self
    }

    private_impl! {}
}

impl Dimension for Dim<[Ix; 1]>
{
    const NDIM: Option<usize> = Some(1);
    type Pattern = Ix;
    type Smaller = Ix0;
    type Larger = Ix2;
    #[inline]
    fn ndim(&self) -> usize
    {
        1
    }
    #[inline]
    fn slice(&self) -> &[Ix]
    {
        self.ix()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix]
    {
        self.ixm()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern
    {
        get!(&self, 0)
    }
    #[inline]
    fn zeros(ndim: usize) -> Self
    {
        assert_eq!(ndim, 1);
        Self::default()
    }
    #[inline]
    fn next_for(&self, mut index: Self) -> Option<Self>
    {
        getm!(index, 0) += 1;
        if get!(&index, 0) < get!(self, 0) {
            Some(index)
        } else {
            None
        }
    }

    #[inline]
    fn equal(&self, rhs: &Self) -> bool
    {
        get!(self, 0) == get!(rhs, 0)
    }

    #[inline]
    fn size(&self) -> usize
    {
        get!(self, 0)
    }
    #[inline]
    fn size_checked(&self) -> Option<usize>
    {
        Some(get!(self, 0))
    }

    #[inline]
    fn default_strides(&self) -> Self
    {
        if get!(self, 0) == 0 {
            Ix1(0)
        } else {
            Ix1(1)
        }
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self
    {
        Ix1(0)
    }

    #[inline(always)]
    fn min_stride_axis(&self, _: &Self) -> Axis
    {
        Axis(0)
    }

    #[inline(always)]
    fn max_stride_axis(&self, _: &Self) -> Axis
    {
        Axis(0)
    }

    #[inline]
    fn first_index(&self) -> Option<Self>
    {
        if get!(self, 0) != 0 {
            Some(Ix1(0))
        } else {
            None
        }
    }

    /// Self is an index, return the stride offset
    #[inline(always)]
    fn stride_offset(index: &Self, stride: &Self) -> isize
    {
        stride_offset(get!(index, 0), get!(stride, 0))
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, stride: &Self, index: &Self) -> Option<isize>
    {
        if get!(index, 0) < get!(self, 0) {
            Some(stride_offset(get!(index, 0), get!(stride, 0)))
        } else {
            None
        }
    }
    impl_insert_axis_array!(1);
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        self.remove_axis(axis)
    }

    fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self>
    {
        if 1 == d.ndim() {
            Some(Ix1(d[0]))
        } else {
            None
        }
    }
    private_impl! {}
}

impl Dimension for Dim<[Ix; 2]>
{
    const NDIM: Option<usize> = Some(2);
    type Pattern = (Ix, Ix);
    type Smaller = Ix1;
    type Larger = Ix3;
    #[inline]
    fn ndim(&self) -> usize
    {
        2
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern
    {
        self.ix().convert()
    }
    #[inline]
    fn slice(&self) -> &[Ix]
    {
        self.ix()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix]
    {
        self.ixm()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self
    {
        assert_eq!(ndim, 2);
        Self::default()
    }
    #[inline]
    fn next_for(&self, index: Self) -> Option<Self>
    {
        let mut i = get!(&index, 0);
        let mut j = get!(&index, 1);
        let imax = get!(self, 0);
        let jmax = get!(self, 1);
        j += 1;
        if j >= jmax {
            j = 0;
            i += 1;
            if i >= imax {
                return None;
            }
        }
        Some(Ix2(i, j))
    }

    #[inline]
    fn equal(&self, rhs: &Self) -> bool
    {
        get!(self, 0) == get!(rhs, 0) && get!(self, 1) == get!(rhs, 1)
    }

    #[inline]
    fn size(&self) -> usize
    {
        get!(self, 0) * get!(self, 1)
    }

    #[inline]
    fn size_checked(&self) -> Option<usize>
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        m.checked_mul(n)
    }

    #[inline]
    fn last_elem(&self) -> usize
    {
        get!(self, 1)
    }

    #[inline]
    fn set_last_elem(&mut self, i: usize)
    {
        getm!(self, 1) = i;
    }

    #[inline]
    fn default_strides(&self) -> Self
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        if m == 0 || n == 0 {
            Ix2(0, 0)
        } else {
            Ix2(n, 1)
        }
    }
    #[inline]
    fn fortran_strides(&self) -> Self
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        if m == 0 || n == 0 {
            Ix2(0, 0)
        } else {
            Ix2(1, m)
        }
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self
    {
        if (get!(self, 0) as Ixs).abs() <= (get!(self, 1) as Ixs).abs() {
            Ix2(0, 1)
        } else {
            Ix2(1, 0)
        }
    }

    #[inline]
    fn min_stride_axis(&self, strides: &Self) -> Axis
    {
        let s = get!(strides, 0) as Ixs;
        let t = get!(strides, 1) as Ixs;
        if s.abs() < t.abs() {
            Axis(0)
        } else {
            Axis(1)
        }
    }

    #[inline]
    fn first_index(&self) -> Option<Self>
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        if m != 0 && n != 0 {
            Some(Ix2(0, 0))
        } else {
            None
        }
    }

    /// Self is an index, return the stride offset
    #[inline(always)]
    fn stride_offset(index: &Self, strides: &Self) -> isize
    {
        let i = get!(index, 0);
        let j = get!(index, 1);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        stride_offset(i, s) + stride_offset(j, t)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        let i = get!(index, 0);
        let j = get!(index, 1);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        if i < m && j < n {
            Some(stride_offset(i, s) + stride_offset(j, t))
        } else {
            None
        }
    }
    impl_insert_axis_array!(2);
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        self.remove_axis(axis)
    }
    private_impl! {}
}

impl Dimension for Dim<[Ix; 3]>
{
    const NDIM: Option<usize> = Some(3);
    type Pattern = (Ix, Ix, Ix);
    type Smaller = Ix2;
    type Larger = Ix4;
    #[inline]
    fn ndim(&self) -> usize
    {
        3
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern
    {
        self.ix().convert()
    }
    #[inline]
    fn slice(&self) -> &[Ix]
    {
        self.ix()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix]
    {
        self.ixm()
    }

    #[inline]
    fn size(&self) -> usize
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        let o = get!(self, 2);
        m * n * o
    }

    #[inline]
    fn zeros(ndim: usize) -> Self
    {
        assert_eq!(ndim, 3);
        Self::default()
    }

    #[inline]
    fn next_for(&self, index: Self) -> Option<Self>
    {
        let mut i = get!(&index, 0);
        let mut j = get!(&index, 1);
        let mut k = get!(&index, 2);
        let imax = get!(self, 0);
        let jmax = get!(self, 1);
        let kmax = get!(self, 2);
        k += 1;
        if k == kmax {
            k = 0;
            j += 1;
            if j == jmax {
                j = 0;
                i += 1;
                if i == imax {
                    return None;
                }
            }
        }
        Some(Ix3(i, j, k))
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &Self, strides: &Self) -> isize
    {
        let i = get!(index, 0);
        let j = get!(index, 1);
        let k = get!(index, 2);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        let u = get!(strides, 2);
        stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        let l = get!(self, 2);
        let i = get!(index, 0);
        let j = get!(index, 1);
        let k = get!(index, 2);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        let u = get!(strides, 2);
        if i < m && j < n && k < l {
            Some(stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u))
        } else {
            None
        }
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self
    {
        let mut stride = *self;
        let mut order = Ix3(0, 1, 2);
        macro_rules! swap {
            ($stride:expr, $order:expr, $x:expr, $y:expr) => {
                if ($stride[$x] as isize).abs() > ($stride[$y] as isize).abs() {
                    $stride.swap($x, $y);
                    $order.ixm().swap($x, $y);
                }
            };
        }
        {
            // stable sorting network for 3 elements
            let strides = stride.slice_mut();
            swap![strides, order, 1, 2];
            swap![strides, order, 0, 1];
            swap![strides, order, 1, 2];
        }
        order
    }
    impl_insert_axis_array!(3);
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        self.remove_axis(axis)
    }
    private_impl! {}
}

macro_rules! large_dim {
    ($n:expr, $name:ident, $pattern:ty, $larger:ty, { $($insert_axis:tt)* }) => (
        impl Dimension for Dim<[Ix; $n]> {
            const NDIM: Option<usize> = Some($n);
            type Pattern = $pattern;
            type Smaller = Dim<[Ix; $n - 1]>;
            type Larger = $larger;
            #[inline]
            fn ndim(&self) -> usize { $n }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                self.ix().convert()
            }
            #[inline]
            fn slice(&self) -> &[Ix] { self.ix() }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
            #[inline]
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, $n);
                Self::default()
            }
            $($insert_axis)*
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::Smaller {
                self.remove_axis(axis)
            }
            private_impl!{}
        }
    );
}

large_dim!(4, Ix4, (Ix, Ix, Ix, Ix), Ix5, {
    impl_insert_axis_array!(4);
});
large_dim!(5, Ix5, (Ix, Ix, Ix, Ix, Ix), Ix6, {
    impl_insert_axis_array!(5);
});
large_dim!(6, Ix6, (Ix, Ix, Ix, Ix, Ix, Ix), IxDyn, {
    fn insert_axis(&self, axis: Axis) -> Self::Larger {
        debug_assert!(axis.index() <= self.ndim());
        let mut out = Vec::with_capacity(self.ndim() + 1);
        out.extend_from_slice(&self.slice()[0..axis.index()]);
        out.push(1);
        out.extend_from_slice(&self.slice()[axis.index()..self.ndim()]);
        Dim(out)
    }
});

/// IxDyn is a "dynamic" index, pretty hard to use when indexing,
/// and memory wasteful, but it allows an arbitrary and dynamic number of axes.
impl Dimension for IxDyn
{
    const NDIM: Option<usize> = None;
    type Pattern = Self;
    type Smaller = Self;
    type Larger = Self;
    #[inline]
    fn ndim(&self) -> usize
    {
        self.ix().len()
    }
    #[inline]
    fn slice(&self) -> &[Ix]
    {
        self.ix()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix]
    {
        self.ixm()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern
    {
        self
    }

    #[inline]
    fn zeros(ndim: usize) -> Self
    {
        IxDyn::zeros(ndim)
    }

    #[inline]
    fn insert_axis(&self, axis: Axis) -> Self::Larger
    {
        debug_assert!(axis.index() <= self.ndim());
        Dim::new(self.ix().insert(axis.index()))
    }

    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
    {
        if self.ndim() > 0 {
            self.remove_axis(axis)
        } else {
            self.clone()
        }
    }

    fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self>
    {
        Some(IxDyn(d.slice()))
    }

    fn into_dyn(self) -> IxDyn
    {
        self
    }

    private_impl! {}
}

impl Index<usize> for Dim<IxDynImpl>
{
    type Output = <IxDynImpl as Index<usize>>::Output;
    fn index(&self, index: usize) -> &Self::Output
    {
        &self.ix()[index]
    }
}

impl IndexMut<usize> for Dim<IxDynImpl>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output
    {
        &mut self.ixm()[index]
    }
}
