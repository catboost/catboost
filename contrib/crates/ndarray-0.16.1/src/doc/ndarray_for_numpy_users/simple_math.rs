//! Example of simple math operations on 2-D arrays.
//!
//! <table>
//!
//! <tr>
//! <th>
//!
//! NumPy
//!
//! </th>
//! <th>
//!
//! `ndarray`
//!
//! </th>
//! </tr>
//! <tr>
//! <td>
//!
//! ```python
//! import numpy as np
//!
//!
//!
//!
//! a = np.full((5, 4), 3.)
//!
//!
//! a[::2, :] = 2.
//!
//!
//! a[:, 1] = np.sin(a[:, 1]) + 1.
//!
//!
//! a[a < 1.5] = 4.
//!
//!
//! odd_sum = a[:, 1::2].sum()
//!
//!
//! b = np.exp(np.arange(4))
//!
//!
//! c = a + b
//!
//!
//! d = c.T.dot(a)
//! ```
//!
//! </td>
//! <td>
//!
//! ```
//! use ndarray::prelude::*;
//!
//! # fn main() {
//! // Create a 5×4 array of threes.
//! let mut a = Array2::<f64>::from_elem((5, 4), 3.);
//!
//! // Fill the even-index rows with twos.
//! a.slice_mut(s![..;2, ..]).fill(2.);
//!
//! // Change column 1 to sin(x) + 1.
//! a.column_mut(1).mapv_inplace(|x| x.sin() + 1.);
//!
//! // Change values less than 1.5 to 4.
//! a.mapv_inplace(|x| if x < 1.5 { 4. } else { x });
//!
//! // Compute the sum of the odd-index columns.
//! let odd_sum = a.slice(s![.., 1..;2]).sum();
//!
//! // Create a 1-D array of exp(index).
//! let b = Array::from_shape_fn(4, |i| (i as f64).exp());
//!
//! // Add b to a (broadcasting to rows).
//! let c = a + &b;
//!
//! // Matrix product of c transpose with c.
//! let d = c.t().dot(&c);
//! # }
//! ```
//!
//! </td>
//! </tr>
//! </table>
