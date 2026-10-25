// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for two-dimensional arrays.
use crate::imp_prelude::*;

/// # Methods For 2-D Arrays
impl<A, S> ArrayBase<S, Ix2>
where S: RawData<Elem = A>
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert_eq!(array.row(0), array![1., 2.]);
    /// ```
    #[track_caller]
    pub fn row(&self, index: Ix) -> ArrayView1<'_, A>
    where S: Data
    {
        self.index_axis(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let mut array = array![[1., 2.], [3., 4.]];
    /// array.row_mut(0)[1] = 5.;
    /// assert_eq!(array, array![[1., 5.], [3., 4.]]);
    /// ```
    #[track_caller]
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A>
    where S: DataMut
    {
        self.index_axis_mut(Axis(0), index)
    }

    /// Return the number of rows (length of `Axis(0)`) in the two-dimensional array.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let array = array![[1., 2.],
    ///                    [3., 4.],
    ///                    [5., 6.]];
    /// assert_eq!(array.nrows(), 3);
    ///
    /// // equivalent ways of getting the dimensions
    /// // get nrows, ncols by using dim:
    /// let (m, n) = array.dim();
    /// assert_eq!(m, array.nrows());
    /// // get length of any particular axis with .len_of()
    /// assert_eq!(m, array.len_of(Axis(0)));
    /// ```
    pub fn nrows(&self) -> usize
    {
        self.len_of(Axis(0))
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert_eq!(array.column(0), array![1., 3.]);
    /// ```
    #[track_caller]
    pub fn column(&self, index: Ix) -> ArrayView1<'_, A>
    where S: Data
    {
        self.index_axis(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let mut array = array![[1., 2.], [3., 4.]];
    /// array.column_mut(0)[1] = 5.;
    /// assert_eq!(array, array![[1., 2.], [5., 4.]]);
    /// ```
    #[track_caller]
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A>
    where S: DataMut
    {
        self.index_axis_mut(Axis(1), index)
    }

    /// Return the number of columns (length of `Axis(1)`) in the two-dimensional array.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let array = array![[1., 2.],
    ///                    [3., 4.],
    ///                    [5., 6.]];
    /// assert_eq!(array.ncols(), 2);
    ///
    /// // equivalent ways of getting the dimensions
    /// // get nrows, ncols by using dim:
    /// let (m, n) = array.dim();
    /// assert_eq!(n, array.ncols());
    /// // get length of any particular axis with .len_of()
    /// assert_eq!(n, array.len_of(Axis(1)));
    /// ```
    pub fn ncols(&self) -> usize
    {
        self.len_of(Axis(1))
    }

    /// Return true if the array is square, false otherwise.
    ///
    /// # Examples
    /// Square:
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert!(array.is_square());
    /// ```
    /// Not square:
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2., 5.], [3., 4., 6.]];
    /// assert!(!array.is_square());
    /// ```
    pub fn is_square(&self) -> bool
    {
        let (m, n) = self.dim();
        m == n
    }
}
