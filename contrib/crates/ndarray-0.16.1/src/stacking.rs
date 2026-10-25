// Copyright 2014-2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::dimension;
use crate::error::{from_kind, ErrorKind, ShapeError};
use crate::imp_prelude::*;

/// Concatenate arrays along the given axis.
///
/// ***Errors*** if the arrays have mismatching shapes, apart from along `axis`.
/// (may be made more flexible in the future).<br>
/// ***Errors*** if `arrays` is empty, if `axis` is out of bounds,
/// if the result is larger than is possible to represent.
///
/// ```
/// use ndarray::{arr2, Axis, concatenate};
///
/// let a = arr2(&[[2., 2.],
///                [3., 3.]]);
/// assert!(
///     concatenate(Axis(0), &[a.view(), a.view()])
///     == Ok(arr2(&[[2., 2.],
///                  [3., 3.],
///                  [2., 2.],
///                  [3., 3.]]))
/// );
/// ```
pub fn concatenate<A, D>(axis: Axis, arrays: &[ArrayView<A, D>]) -> Result<Array<A, D>, ShapeError>
where
    A: Clone,
    D: RemoveAxis,
{
    if arrays.is_empty() {
        return Err(from_kind(ErrorKind::Unsupported));
    }
    let mut res_dim = arrays[0].raw_dim();
    if axis.index() >= res_dim.ndim() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    let common_dim = res_dim.remove_axis(axis);
    if arrays
        .iter()
        .any(|a| a.raw_dim().remove_axis(axis) != common_dim)
    {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    }

    let stacked_dim = arrays.iter().fold(0, |acc, a| acc + a.len_of(axis));
    res_dim.set_axis(axis, stacked_dim);
    let new_len = dimension::size_of_shape_checked(&res_dim)?;

    // start with empty array with precomputed capacity
    // append's handling of empty arrays makes sure `axis` is ok for appending
    res_dim.set_axis(axis, 0);
    let mut res = unsafe {
        // Safety: dimension is size 0 and vec is empty
        Array::from_shape_vec_unchecked(res_dim, Vec::with_capacity(new_len))
    };

    for array in arrays {
        res.append(axis, array.clone())?;
    }
    debug_assert_eq!(res.len_of(axis), stacked_dim);
    Ok(res)
}

/// Stack arrays along the new axis.
///
/// ***Errors*** if the arrays have mismatching shapes.
/// ***Errors*** if `arrays` is empty, if `axis` is out of bounds,
/// if the result is larger than is possible to represent.
///
/// ```
/// extern crate ndarray;
///
/// use ndarray::{arr2, arr3, stack, Axis};
///
/// # fn main() {
///
/// let a = arr2(&[[2., 2.],
///                [3., 3.]]);
/// assert!(
///     stack(Axis(0), &[a.view(), a.view()])
///     == Ok(arr3(&[[[2., 2.],
///                   [3., 3.]],
///                  [[2., 2.],
///                   [3., 3.]]]))
/// );
/// # }
/// ```
pub fn stack<A, D>(axis: Axis, arrays: &[ArrayView<A, D>]) -> Result<Array<A, D::Larger>, ShapeError>
where
    A: Clone,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    if arrays.is_empty() {
        return Err(from_kind(ErrorKind::Unsupported));
    }
    let common_dim = arrays[0].raw_dim();
    // Avoid panic on `insert_axis` call, return an Err instead of it.
    if axis.index() > common_dim.ndim() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    let mut res_dim = common_dim.insert_axis(axis);

    if arrays.iter().any(|a| a.raw_dim() != common_dim) {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    }

    res_dim.set_axis(axis, arrays.len());

    let new_len = dimension::size_of_shape_checked(&res_dim)?;

    // start with empty array with precomputed capacity
    // append's handling of empty arrays makes sure `axis` is ok for appending
    res_dim.set_axis(axis, 0);
    let mut res = unsafe {
        // Safety: dimension is size 0 and vec is empty
        Array::from_shape_vec_unchecked(res_dim, Vec::with_capacity(new_len))
    };

    for array in arrays {
        res.append(axis, array.clone().insert_axis(axis))?;
    }

    debug_assert_eq!(res.len_of(axis), arrays.len());
    Ok(res)
}

/// Stack arrays along the new axis.
///
/// Uses the [`stack()`] function, calling `ArrayView::from(&a)` on each
/// argument `a`.
///
/// ***Panics*** if the `stack` function would return an error.
///
/// ```
/// extern crate ndarray;
///
/// use ndarray::{arr2, arr3, stack, Axis};
///
/// # fn main() {
///
/// let a = arr2(&[[1., 2.],
///                [3., 4.]]);
/// assert_eq!(
///     stack![Axis(0), a, a],
///     arr3(&[[[1., 2.],
///             [3., 4.]],
///            [[1., 2.],
///             [3., 4.]]]),
/// );
/// assert_eq!(
///     stack![Axis(1), a, a,],
///     arr3(&[[[1., 2.],
///             [1., 2.]],
///            [[3., 4.],
///             [3., 4.]]]),
/// );
/// assert_eq!(
///     stack![Axis(2), a, a],
///     arr3(&[[[1., 1.],
///             [2., 2.]],
///            [[3., 3.],
///             [4., 4.]]]),
/// );
/// # }
/// ```
#[macro_export]
macro_rules! stack {
    ($axis:expr, $( $array:expr ),+ ,) => {
        $crate::stack!($axis, $($array),+)
    };
    ($axis:expr, $( $array:expr ),+ ) => {
        $crate::stack($axis, &[ $($crate::ArrayView::from(&$array) ),* ]).unwrap()
    };
}

/// Concatenate arrays along the given axis.
///
/// Uses the [`concatenate()`] function, calling `ArrayView::from(&a)` on each
/// argument `a`.
///
/// ***Panics*** if the `concatenate` function would return an error.
///
/// ```
/// extern crate ndarray;
///
/// use ndarray::{arr2, concatenate, Axis};
///
/// # fn main() {
///
/// let a = arr2(&[[1., 2.],
///                [3., 4.]]);
/// assert_eq!(
///     concatenate![Axis(0), a, a],
///     arr2(&[[1., 2.],
///            [3., 4.],
///            [1., 2.],
///            [3., 4.]]),
/// );
/// assert_eq!(
///     concatenate![Axis(1), a, a,],
///     arr2(&[[1., 2., 1., 2.],
///            [3., 4., 3., 4.]]),
/// );
/// # }
/// ```
#[macro_export]
macro_rules! concatenate {
    ($axis:expr, $( $array:expr ),+ ,) => {
        $crate::concatenate!($axis, $($array),+)
    };
    ($axis:expr, $( $array:expr ),+ ) => {
        $crate::concatenate($axis, &[ $($crate::ArrayView::from(&$array) ),* ]).unwrap()
    };
}
