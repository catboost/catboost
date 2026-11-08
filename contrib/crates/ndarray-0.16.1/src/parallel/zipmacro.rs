// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_export]
/// Parallelized array zip macro: lock step function application across several
/// arrays and producers.
///
/// This is a version of the [`azip`] macro that requires the crate feature
/// `rayon` to be enabled.
///
/// See the [`azip`] macro for more details about the macro syntax!
///
/// This example:
///
/// ```rust,ignore
/// par_azip!((a in &mut a, &b in &b, &c in &c) { *a = b + c })
/// ```
///
/// Is equivalent to:
///
/// ```rust,ignore
/// Zip::from(&mut a).and(&b).and(&c).par_for_each(|a, &b, &c| {
///     *a = b + c;
/// });
/// ```
///
/// **Panics** if any of the arrays are not of the same shape.
///
/// ## Examples
///
/// ```rust
/// use ndarray::Array2;
/// use ndarray::parallel::par_azip;
///
/// type M = Array2<f32>;
///
/// let mut a = M::zeros((16, 16));
/// let b = M::from_elem(a.dim(), 1.);
/// let c = M::from_elem(a.dim(), 2.);
///
/// // Compute a simple ternary operation:
/// // elementwise addition of b and c, stored in a
///
/// par_azip!((a in &mut a, &b in &b, &c in &c) *a = b + c);
///
/// assert_eq!(a, &b + &c);
/// ```
macro_rules! par_azip {
    ($($t:tt)*) => {
        $crate::azip!(@build par_for_each $($t)*)
    };
}
