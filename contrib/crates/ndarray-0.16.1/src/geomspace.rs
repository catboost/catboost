// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![cfg(feature = "std")]
use num_traits::Float;

/// An iterator of a sequence of geometrically spaced floats.
///
/// Iterator element type is `F`.
pub struct Geomspace<F>
{
    sign: F,
    start: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Geomspace<F>
where F: Float
{
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F>
    {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            let exponent = self.start + self.step * F::from(i).unwrap();
            Some(self.sign * exponent.exp())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Geomspace<F>
where F: Float
{
    #[inline]
    fn next_back(&mut self) -> Option<F>
    {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            self.len -= 1;
            let i = self.len;
            let exponent = self.start + self.step * F::from(i).unwrap();
            Some(self.sign * exponent.exp())
        }
    }
}

impl<F> ExactSizeIterator for Geomspace<F> where Geomspace<F>: Iterator {}

/// An iterator of a sequence of geometrically spaced values.
///
/// The `Geomspace` has `n` geometrically spaced elements from `start` to `end`
/// (inclusive).
///
/// The iterator element type is `F`, where `F` must implement `Float`, e.g.
/// `f32` or `f64`.
///
/// Returns `None` if `start` and `end` have different signs or if either one
/// is zero. Conceptually, this means that in order to obtain a `Some` result,
/// `end / start` must be positive.
///
/// **Panics** if converting `n - 1` to type `F` fails.
#[inline]
pub fn geomspace<F>(a: F, b: F, n: usize) -> Option<Geomspace<F>>
where F: Float
{
    if a == F::zero() || b == F::zero() || a.is_sign_negative() != b.is_sign_negative() {
        return None;
    }
    let log_a = a.abs().ln();
    let log_b = b.abs().ln();
    let step = if n > 1 {
        let num_steps = F::from(n - 1).expect("Converting number of steps to `A` must not fail.");
        (log_b - log_a) / num_steps
    } else {
        F::zero()
    };
    Some(Geomspace {
        sign: a.signum(),
        start: log_a,
        step,
        index: 0,
        len: n,
    })
}

#[cfg(test)]
mod tests
{
    use super::geomspace;

    #[test]
    #[cfg(feature = "approx")]
    fn valid()
    {
        use crate::{arr1, Array1};
        use approx::assert_abs_diff_eq;

        let array: Array1<_> = geomspace(1e0, 1e3, 4).unwrap().collect();
        assert_abs_diff_eq!(array, arr1(&[1e0, 1e1, 1e2, 1e3]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(1e3, 1e0, 4).unwrap().collect();
        assert_abs_diff_eq!(array, arr1(&[1e3, 1e2, 1e1, 1e0]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(-1e3, -1e0, 4).unwrap().collect();
        assert_abs_diff_eq!(array, arr1(&[-1e3, -1e2, -1e1, -1e0]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(-1e0, -1e3, 4).unwrap().collect();
        assert_abs_diff_eq!(array, arr1(&[-1e0, -1e1, -1e2, -1e3]), epsilon = 1e-12);
    }

    #[test]
    fn iter_forward()
    {
        let mut iter = geomspace(1.0f64, 1e3, 4).unwrap();

        assert!(iter.size_hint() == (4, Some(4)));

        assert!((iter.next().unwrap() - 1e0).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e1).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e2).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e3).abs() < 1e-5);
        assert!(iter.next().is_none());

        assert!(iter.size_hint() == (0, Some(0)));
    }

    #[test]
    fn iter_backward()
    {
        let mut iter = geomspace(1.0f64, 1e3, 4).unwrap();

        assert!(iter.size_hint() == (4, Some(4)));

        assert!((iter.next_back().unwrap() - 1e3).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e2).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e1).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e0).abs() < 1e-5);
        assert!(iter.next_back().is_none());

        assert!(iter.size_hint() == (0, Some(0)));
    }

    #[test]
    fn zero_lower()
    {
        assert!(geomspace(0.0, 1.0, 4).is_none());
    }

    #[test]
    fn zero_upper()
    {
        assert!(geomspace(1.0, 0.0, 4).is_none());
    }

    #[test]
    fn zero_included()
    {
        assert!(geomspace(-1.0, 1.0, 4).is_none());
    }
}
