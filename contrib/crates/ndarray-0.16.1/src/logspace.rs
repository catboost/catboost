// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![cfg(feature = "std")]
use num_traits::Float;

/// An iterator of a sequence of logarithmically spaced number.
///
/// Iterator element type is `F`.
pub struct Logspace<F>
{
    sign: F,
    base: F,
    start: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Logspace<F>
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
            Some(self.sign * self.base.powf(exponent))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Logspace<F>
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
            Some(self.sign * self.base.powf(exponent))
        }
    }
}

impl<F> ExactSizeIterator for Logspace<F> where Logspace<F>: Iterator {}

/// An iterator of a sequence of logarithmically spaced numbers.
///
/// The [`Logspace`] has `n` elements, where the first element is `base.powf(a)`
/// and the last element is `base.powf(b)`.  If `base` is negative, this
/// iterator will return all negative values.
///
/// The iterator element type is `F`, where `F` must implement [`Float`], e.g.
/// [`f32`] or [`f64`].
///
/// **Panics** if converting `n - 1` to type `F` fails.
#[inline]
pub fn logspace<F>(base: F, a: F, b: F, n: usize) -> Logspace<F>
where F: Float
{
    let step = if n > 1 {
        let num_steps = F::from(n - 1).expect("Converting number of steps to `A` must not fail.");
        (b - a) / num_steps
    } else {
        F::zero()
    };
    Logspace {
        sign: base.signum(),
        base: base.abs(),
        start: a,
        step,
        index: 0,
        len: n,
    }
}

#[cfg(test)]
mod tests
{
    use super::logspace;

    #[test]
    #[cfg(feature = "approx")]
    fn valid()
    {
        use crate::{arr1, Array1};
        use approx::assert_abs_diff_eq;

        let array: Array1<_> = logspace(10.0, 0.0, 3.0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[1e0, 1e1, 1e2, 1e3]));

        let array: Array1<_> = logspace(10.0, 3.0, 0.0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[1e3, 1e2, 1e1, 1e0]));

        let array: Array1<_> = logspace(-10.0, 3.0, 0.0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[-1e3, -1e2, -1e1, -1e0]));

        let array: Array1<_> = logspace(-10.0, 0.0, 3.0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[-1e0, -1e1, -1e2, -1e3]));
    }

    #[test]
    fn iter_forward()
    {
        let mut iter = logspace(10.0f64, 0.0, 3.0, 4);

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
        let mut iter = logspace(10.0f64, 0.0, 3.0, 4);

        assert!(iter.size_hint() == (4, Some(4)));

        assert!((iter.next_back().unwrap() - 1e3).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e2).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e1).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e0).abs() < 1e-5);
        assert!(iter.next_back().is_none());

        assert!(iter.size_hint() == (0, Some(0)));
    }
}
