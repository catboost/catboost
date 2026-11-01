//! Example of translating `rk_step` function from SciPy.
//!
//! This snippet is a [selection of lines from
//! `rk.py`](https://github.com/scipy/scipy/blob/v1.0.0/scipy/integrate/_ivp/rk.py#L2-L78).
//! See the [license for this snippet](#scipy-license).
//!
//! ```python
//! import numpy as np
//!
//! def rk_step(fun, t, y, f, h, A, B, C, E, K):
//!     """Perform a single Runge-Kutta step.
//!     This function computes a prediction of an explicit Runge-Kutta method and
//!     also estimates the error of a less accurate method.
//!     Notation for Butcher tableau is as in [1]_.
//!     Parameters
//!     ----------
//!     fun : callable
//!         Right-hand side of the system.
//!     t : float
//!         Current time.
//!     y : ndarray, shape (n,)
//!         Current state.
//!     f : ndarray, shape (n,)
//!         Current value of the derivative, i.e. ``fun(x, y)``.
//!     h : float
//!         Step to use.
//!     A : list of ndarray, length n_stages - 1
//!         Coefficients for combining previous RK stages to compute the next
//!         stage. For explicit methods the coefficients above the main diagonal
//!         are zeros, so `A` is stored as a list of arrays of increasing lengths.
//!         The first stage is always just `f`, thus no coefficients for it
//!         are required.
//!     B : ndarray, shape (n_stages,)
//!         Coefficients for combining RK stages for computing the final
//!         prediction.
//!     C : ndarray, shape (n_stages - 1,)
//!         Coefficients for incrementing time for consecutive RK stages.
//!         The value for the first stage is always zero, thus it is not stored.
//!     E : ndarray, shape (n_stages + 1,)
//!         Coefficients for estimating the error of a less accurate method. They
//!         are computed as the difference between b's in an extended tableau.
//!     K : ndarray, shape (n_stages + 1, n)
//!         Storage array for putting RK stages here. Stages are stored in rows.
//!     Returns
//!     -------
//!     y_new : ndarray, shape (n,)
//!         Solution at t + h computed with a higher accuracy.
//!     f_new : ndarray, shape (n,)
//!         Derivative ``fun(t + h, y_new)``.
//!     error : ndarray, shape (n,)
//!         Error estimate of a less accurate method.
//!     References
//!     ----------
//!     .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
//!            Equations I: Nonstiff Problems", Sec. II.4.
//!     """
//!     K[0] = f
//!     for s, (a, c) in enumerate(zip(A, C)):
//!         dy = np.dot(K[:s + 1].T, a) * h
//!         K[s + 1] = fun(t + c * h, y + dy)
//!
//!     y_new = y + h * np.dot(K[:-1].T, B)
//!     f_new = fun(t + h, y_new)
//!
//!     K[-1] = f_new
//!     error = np.dot(K.T, E) * h
//!
//!     return y_new, f_new, error
//! ```
//!
//! A direct translation to `ndarray` looks like this:
//!
//! ```
//! use ndarray::prelude::*;
//!
//! fn rk_step<F>(
//!     mut fun: F,
//!     t: f64,
//!     y: ArrayView1<f64>,
//!     f: ArrayView1<f64>,
//!     h: f64,
//!     a: &[ArrayView1<f64>],
//!     b: ArrayView1<f64>,
//!     c: ArrayView1<f64>,
//!     e: ArrayView1<f64>,
//!     mut k: ArrayViewMut2<f64>,
//! ) -> (Array1<f64>, Array1<f64>, Array1<f64>)
//! where
//!     F: FnMut(f64, ArrayView1<f64>) -> Array1<f64>,
//! {
//!     k.slice_mut(s![0, ..]).assign(&f);
//!     for (s, (a, c)) in a.iter().zip(c).enumerate() {
//!         let dy = k.slice(s![..s + 1, ..]).t().dot(a) * h;
//!         k.slice_mut(s![s + 1, ..])
//!             .assign(&(fun(t + c * h, (&y + &dy).view())));
//!     }
//!
//!     let y_new = &y + &(h * k.slice(s![..-1, ..]).t().dot(&b));
//!     let f_new = fun(t + h, y_new.view());
//!
//!     k.slice_mut(s![-1, ..]).assign(&f_new);
//!     let error = k.t().dot(&e) * h;
//!
//!     (y_new, f_new, error)
//! }
//! #
//! # fn main() { let _ = rk_step::<fn(_, ArrayView1<'_, _>) -> _>; }
//! ```
//!
//! It's possible to improve the efficiency by doing the following:
//!
//! * Observe that `dy` is a temporary allocation. It's possible to allow the
//!   add operation to take ownership of `dy` to eliminate an extra allocation
//!   for the result of the addition. A similar situation occurs when computing
//!   `y_new`. See the comments in the example below.
//!
//! * Require the `fun` closure to mutate an existing view instead of
//!   allocating a new array for the result.
//!
//! * Don't return a newly allocated `f_new` array. If the caller wants this
//!   information, they can get it from the last row of `k`.
//!
//! * Use [`c.mul_add(h, t)`](f64::mul_add) instead of `t + c * h`. This is
//!   faster and reduces the floating-point error. It might also be beneficial
//!   to use [`.scaled_add()`] or a combination of
//!   [`azip!()`] and [`.mul_add()`](f64::mul_add) on the arrays in
//!   some places, but that's not demonstrated in the example below.
//!
//! ```
//! use ndarray::prelude::*;
//!
//! fn rk_step<F>(
//!     mut fun: F,
//!     t: f64,
//!     y: ArrayView1<f64>,
//!     f: ArrayView1<f64>,
//!     h: f64,
//!     a: &[ArrayView1<f64>],
//!     b: ArrayView1<f64>,
//!     c: ArrayView1<f64>,
//!     e: ArrayView1<f64>,
//!     mut k: ArrayViewMut2<f64>,
//! ) -> (Array1<f64>, Array1<f64>)
//! where
//!     F: FnMut(f64, ArrayView1<f64>, ArrayViewMut1<f64>),
//! {
//!     k.slice_mut(s![0, ..]).assign(&f);
//!     for (s, (a, c)) in a.iter().zip(c).enumerate() {
//!         let dy = k.slice(s![..s + 1, ..]).t().dot(a) * h;
//!         // Note that `dy` comes before `&y` in `dy + &y` in order to reuse the
//!         // `dy` allocation. (The addition operator will take ownership of `dy`
//!         // and assign the result to it instead of allocating a new array for the
//!         // result.) In contrast, you could use `&y + &dy`, but that would perform
//!         // an unnecessary memory allocation for the result, like NumPy does.
//!         fun(c.mul_add(h, t), (dy + &y).view(), k.slice_mut(s![s + 1, ..]));
//!     }
//!     // Similar case here — moving `&y` to the right hand side allows the addition
//!     // to reuse the allocated array on the left hand side.
//!     let y_new = h * k.slice(s![..-1, ..]).t().dot(&b) + &y;
//!     // Mutate the last row of `k` in-place instead of allocating a new array.
//!     fun(t + h, y_new.view(), k.slice_mut(s![-1, ..]));
//!
//!     let error = k.t().dot(&e) * h;
//!
//!     (y_new, error)
//! }
//! #
//! # fn main() { let _ = rk_step::<fn(_, ArrayView1<'_, f64>, ArrayViewMut1<'_, f64>)>; }
//! ```
//!
//! [`.scaled_add()`]: crate::ArrayBase::scaled_add
//! [`azip!()`]: crate::azip!
//!
//! ### SciPy license
//!
//! ```text
//! Copyright (c) 2001, 2002 Enthought, Inc.
//! All rights reserved.
//!
//! Copyright (c) 2003-2017 SciPy Developers.
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//!   a. Redistributions of source code must retain the above copyright notice,
//!      this list of conditions and the following disclaimer.
//!   b. Redistributions in binary form must reproduce the above copyright
//!      notice, this list of conditions and the following disclaimer in the
//!      documentation and/or other materials provided with the distribution.
//!   c. Neither the name of Enthought nor the names of the SciPy Developers
//!      may be used to endorse or promote products derived from this software
//!      without specific prior written permission.
//!
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
//! BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
//! OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//! SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//! INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//! CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
//! THE POSSIBILITY OF SUCH DAMAGE.
//! ```
