// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;
use crate::slice::MultiSliceArg;
use num_complex::Complex;

/// Methods for read-only array views.
impl<'a, A, D> ArrayView<'a, A, D>
where D: Dimension
{
    /// Split the array view along `axis` and return one view strictly before the
    /// split and one view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// **Examples:**
    /// ```rust
    /// # use ndarray::prelude::*;
    /// let a = aview2(&[[0, 1, 2, 3],
    ///                  [4, 5, 6, 7],
    ///                  [8, 9, 0, 1]]);
    ///
    /// ```
    /// The array view `a` has two axes and shape 3 × 4:
    /// ```text
    ///          ──▶ Axis(1)
    ///         ┌─────┬─────┬─────┬─────┐ 0
    ///       │ │ a₀₀ │ a₀₁ │ a₀₂ │ a₀₃ │
    ///       ▼ ├─────┼─────┼─────┼─────┤ 1
    ///  Axis(0)│ a₁₀ │ a₁₁ │ a₁₂ │ a₁₃ │
    ///         ├─────┼─────┼─────┼─────┤ 2
    ///         │ a₂₀ │ a₂₁ │ a₂₂ │ a₂₃ │
    ///         └─────┴─────┴─────┴─────┘ 3 ↑
    ///         0     1     2     3     4 ← possible split_at indices.
    /// ```
    ///
    /// Row indices increase along `Axis(0)`, and column indices increase along
    /// `Axis(1)`. Note that we split “before” an element index, and that
    /// both 0 and the endpoint are valid split indices.
    ///
    /// **Example 1**: Split `a` along the first axis, in this case the rows, at
    /// index 2.<br>
    /// This produces views v1 and v2 of shapes 2 × 4 and 1 × 4:
    ///
    /// ```rust
    /// # use ndarray::prelude::*;
    /// # let a = aview2(&[[0; 4]; 3]);
    /// let (v1, v2) = a.split_at(Axis(0), 2);
    /// ```
    /// ```text
    ///         ┌─────┬─────┬─────┬─────┐       0  ↓ indices
    ///         │ a₀₀ │ a₀₁ │ a₀₂ │ a₀₃ │            along Axis(0)
    ///         ├─────┼─────┼─────┼─────┤ v1    1
    ///         │ a₁₀ │ a₁₁ │ a₁₂ │ a₁₃ │
    ///         └─────┴─────┴─────┴─────┘
    ///         ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄       2
    ///         ┌─────┬─────┬─────┬─────┐
    ///         │ a₂₀ │ a₂₁ │ a₂₂ │ a₂₃ │ v2
    ///         └─────┴─────┴─────┴─────┘       3
    /// ```
    ///
    /// **Example 2**: Split `a` along the second axis, in this case the
    /// columns, at index 2.<br>
    /// This produces views u1 and u2 of shapes 3 × 2 and 3 × 2:
    ///
    /// ```rust
    /// # use ndarray::prelude::*;
    /// # let a = aview2(&[[0; 4]; 3]);
    /// let (u1, u2) = a.split_at(Axis(1), 2);
    ///
    /// ```
    /// ```text
    ///              u1             u2
    ///         ┌─────┬─────┐┊┌─────┬─────┐
    ///         │ a₀₀ │ a₀₁ │┊│ a₀₂ │ a₀₃ │
    ///         ├─────┼─────┤┊├─────┼─────┤
    ///         │ a₁₀ │ a₁₁ │┊│ a₁₂ │ a₁₃ │
    ///         ├─────┼─────┤┊├─────┼─────┤
    ///         │ a₂₀ │ a₂₁ │┊│ a₂₂ │ a₂₃ │
    ///         └─────┴─────┘┊└─────┴─────┘
    ///         0     1      2      3     4  indices →
    ///                                      along Axis(1)
    /// ```
    #[track_caller]
    #[inline]
    pub fn split_at(self, axis: Axis, index: Ix) -> (Self, Self)
    {
        unsafe {
            let (left, right) = self.into_raw_view().split_at(axis, index);
            (left.deref_into_view(), right.deref_into_view())
        }
    }
}

impl<'a, T, D> ArrayView<'a, Complex<T>, D>
where D: Dimension
{
    /// Splits the view into views of the real and imaginary components of the
    /// elements.
    ///
    /// ```
    /// use ndarray::prelude::*;
    /// use num_complex::{Complex, Complex64};
    ///
    /// let arr = array![
    ///     [Complex64::new(1., 2.), Complex64::new(3., 4.)],
    ///     [Complex64::new(5., 6.), Complex64::new(7., 8.)],
    ///     [Complex64::new(9., 10.), Complex64::new(11., 12.)],
    /// ];
    /// let Complex { re, im } = arr.view().split_complex();
    /// assert_eq!(re, array![[1., 3.], [5., 7.], [9., 11.]]);
    /// assert_eq!(im, array![[2., 4.], [6., 8.], [10., 12.]]);
    /// ```
    pub fn split_complex(self) -> Complex<ArrayView<'a, T, D>>
    {
        unsafe {
            let Complex { re, im } = self.into_raw_view().split_complex();
            Complex {
                re: re.deref_into_view(),
                im: im.deref_into_view(),
            }
        }
    }
}

/// Methods for read-write array views.
impl<'a, A, D> ArrayViewMut<'a, A, D>
where D: Dimension
{
    /// Split the array view along `axis` and return one mutable view strictly
    /// before the split and one mutable view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    #[track_caller]
    #[inline]
    pub fn split_at(self, axis: Axis, index: Ix) -> (Self, Self)
    {
        unsafe {
            let (left, right) = self.into_raw_view_mut().split_at(axis, index);
            (left.deref_into_view_mut(), right.deref_into_view_mut())
        }
    }

    /// Split the view into multiple disjoint slices.
    ///
    /// This is similar to [`.multi_slice_mut()`], but `.multi_slice_move()`
    /// consumes `self` and produces views with lifetimes matching that of
    /// `self`.
    ///
    /// See [*Slicing*](#slicing) for full documentation. See also
    /// [`MultiSliceArg`], [`s!`], [`SliceArg`](crate::SliceArg), and
    /// [`SliceInfo`](crate::SliceInfo).
    ///
    /// [`.multi_slice_mut()`]: ArrayBase::multi_slice_mut
    ///
    /// **Panics** if any of the following occur:
    ///
    /// * if any of the views would intersect (i.e. if any element would appear in multiple slices)
    /// * if an index is out of bounds or step size is zero
    /// * if `D` is `IxDyn` and `info` does not match the number of array axes
    #[track_caller]
    pub fn multi_slice_move<M>(self, info: M) -> M::Output
    where M: MultiSliceArg<'a, A, D>
    {
        info.multi_slice_move(self)
    }
}

impl<'a, T, D> ArrayViewMut<'a, Complex<T>, D>
where D: Dimension
{
    /// Splits the view into views of the real and imaginary components of the
    /// elements.
    ///
    /// ```
    /// use ndarray::prelude::*;
    /// use num_complex::{Complex, Complex64};
    ///
    /// let mut arr = array![
    ///     [Complex64::new(1., 2.), Complex64::new(3., 4.)],
    ///     [Complex64::new(5., 6.), Complex64::new(7., 8.)],
    ///     [Complex64::new(9., 10.), Complex64::new(11., 12.)],
    /// ];
    ///
    /// let Complex { mut re, mut im } = arr.view_mut().split_complex();
    /// assert_eq!(re, array![[1., 3.], [5., 7.], [9., 11.]]);
    /// assert_eq!(im, array![[2., 4.], [6., 8.], [10., 12.]]);
    ///
    /// re[[0, 1]] = 13.;
    /// im[[2, 0]] = 14.;
    ///
    /// assert_eq!(arr[[0, 1]], Complex64::new(13., 4.));
    /// assert_eq!(arr[[2, 0]], Complex64::new(9., 14.));
    /// ```
    pub fn split_complex(self) -> Complex<ArrayViewMut<'a, T, D>>
    {
        unsafe {
            let Complex { re, im } = self.into_raw_view_mut().split_complex();
            Complex {
                re: re.deref_into_view_mut(),
                im: im.deref_into_view_mut(),
            }
        }
    }
}
