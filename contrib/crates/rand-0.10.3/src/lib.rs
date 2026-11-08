// Copyright 2018 Developers of the Rand project.
// Copyright 2013-2017 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Random-number generators and samplers
//!
//! # Quick Start
//!
//! ```
//! // rand::random() supports many common types:
//! println!("Uniform i8 sample: {}", match rand::random() {
//!     0i8 => "zero",
//!     i if i > 0 => "positive",
//!     _ => "negative",
//! });
//!
//! // Ranged sampling:
//! use std::f32::consts::PI;
//! println!("Angle: {} radians", rand::random_range(-PI..PI));
//! ```
//!
//! See also [The Book: Quick Start](https://rust-random.github.io/book/quick-start.html).

#![doc(
    html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
    html_favicon_url = "https://www.rust-lang.org/favicon.ico"
)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![doc(test(attr(allow(unused_variables), deny(warnings))))]
#![no_std]
#![cfg_attr(feature = "simd_support", feature(portable_simd))]
#![cfg_attr(
    all(feature = "simd_support", target_feature = "avx512bw"),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(
    clippy::float_cmp,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::nonminimal_bool
)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

// Re-export rand_core itself
pub use rand_core;

// Re-exports from rand_core
pub use rand_core::{CryptoRng, Rng, SeedableRng, TryCryptoRng, TryRng};

// Public modules
pub mod distr;
pub mod prelude;
mod rng;
pub mod rngs;
pub mod seq;

// Public exports
#[cfg(feature = "thread_rng")]
pub use crate::rngs::thread::rng;

pub use rng::{Fill, RngExt};

#[cfg(feature = "thread_rng")]
use crate::distr::{Distribution, StandardUniform};

/// Construct and seed an RNG
///
/// This method yields a seeded RNG, using [`rng`] ([`ThreadRng`]) if enabled or
/// [`SysRng`] otherwise.
///
/// # Examples
///
/// ```
/// let mut rng: rand::rngs::SmallRng = rand::make_rng();
/// # let _ = rand::Rng::next_u32(&mut rng);
/// ```
///
/// # Panics
///
/// If [`SysRng`] fails to obtain entropy from the OS. This is unlikely
/// outside of early boot or unusual system conditions.
///
/// # Security
///
/// Refer to [`ThreadRng#Security`].
///
/// [`SysRng`]: crate::rngs::SysRng
/// [`ThreadRng`]: crate::rngs::ThreadRng
/// [`ThreadRng#Security`]: crate::rngs::ThreadRng#security
#[cfg(feature = "sys_rng")]
#[track_caller]
pub fn make_rng<R: SeedableRng>() -> R {
    #[cfg(feature = "thread_rng")]
    {
        R::from_rng(&mut rng())
    }

    #[cfg(not(feature = "thread_rng"))]
    {
        R::try_from_rng(&mut rngs::SysRng).expect("unexpected failure from SysRng")
    }
}

/// Adapter to support [`std::io::Read`] over a [`TryRng`]
///
/// # Examples
///
/// ```no_run
/// use std::{io, io::Read};
/// use std::fs::File;
/// use rand::{rngs::SysRng, RngReader};
///
/// io::copy(
///     &mut RngReader(SysRng).take(100),
///     &mut File::create("/tmp/random.bytes").unwrap()
/// ).unwrap();
/// ```
#[cfg(feature = "std")]
pub struct RngReader<R: TryRng>(pub R);

#[cfg(feature = "std")]
impl<R: TryRng> std::io::Read for RngReader<R> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        self.0
            .try_fill_bytes(buf)
            .map_err(|err| std::io::Error::other(std::format!("RNG error: {err}")))?;
        Ok(buf.len())
    }
}

#[cfg(feature = "std")]
impl<R: TryRng> std::fmt::Debug for RngReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RngReader").finish()
    }
}

/// Generate a random value using the thread-local random number generator.
///
/// This function is shorthand for <code>[rng()].[random()](RngExt::random)</code>:
///
/// -   See [`ThreadRng`] for documentation of the generator and security
/// -   See [`StandardUniform`] for documentation of supported types and distributions
///
/// # Examples
///
/// ```
/// let x = rand::random::<u8>();
/// println!("{}", x);
///
/// let y = rand::random::<f64>();
/// println!("{}", y);
///
/// if rand::random() { // generates a boolean
///     println!("Better lucky than good!");
/// }
/// ```
///
/// If you're calling `random()` repeatedly, consider using a local `rng`
/// handle to save an initialization-check on each usage:
///
/// ```
/// use rand::RngExt; // provides the `random` method
///
/// let mut rng = rand::rng(); // a local handle to the generator
///
/// let mut v = vec![1, 2, 3];
///
/// for x in v.iter_mut() {
///     *x = rng.random();
/// }
/// ```
///
/// [`StandardUniform`]: distr::StandardUniform
/// [`ThreadRng`]: rngs::ThreadRng
#[cfg(feature = "thread_rng")]
#[inline]
pub fn random<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    rng().random()
}

/// Return an iterator over [`random()`] variates
///
/// This function is shorthand for
/// <code>[rng()].[random_iter](RngExt::random_iter)()</code>.
///
/// # Example
///
/// ```
/// let v: Vec<i32> = rand::random_iter().take(5).collect();
/// println!("{v:?}");
/// ```
#[cfg(feature = "thread_rng")]
#[inline]
pub fn random_iter<T>() -> distr::Iter<StandardUniform, rngs::ThreadRng, T>
where
    StandardUniform: Distribution<T>,
{
    rng().random_iter()
}

/// Generate a random value in the given range using the thread-local random number generator.
///
/// This function is shorthand for
/// <code>[rng()].[random_range](RngExt::random_range)(<var>range</var>)</code>.
///
/// # Example
///
/// ```
/// let y: f32 = rand::random_range(0.0..=1e9);
/// println!("{}", y);
///
/// let words: Vec<&str> = "Mary had a little lamb".split(' ').collect();
/// println!("{}", words[rand::random_range(..words.len())]);
/// ```
/// Note that the second example can also be achieved (without `collect`'ing
/// to a `Vec`) using [`seq::IteratorRandom::choose`].
#[cfg(feature = "thread_rng")]
#[inline]
pub fn random_range<T, R>(range: R) -> T
where
    T: distr::uniform::SampleUniform,
    R: distr::uniform::SampleRange<T>,
{
    rng().random_range(range)
}

/// Return a bool with a probability `p` of being true.
///
/// This function is shorthand for
/// <code>[rng()].[random_bool](RngExt::random_bool)(<var>p</var>)</code>.
///
/// # Example
///
/// ```
/// println!("{}", rand::random_bool(1.0 / 3.0));
/// ```
///
/// # Panics
///
/// If `p < 0` or `p > 1`.
#[cfg(feature = "thread_rng")]
#[inline]
#[track_caller]
pub fn random_bool(p: f64) -> bool {
    rng().random_bool(p)
}

/// Return a bool with a probability of `numerator/denominator` of being
/// true.
///
/// That is, `random_ratio(2, 3)` has chance of 2 in 3, or about 67%, of
/// returning true. If `numerator == denominator`, then the returned value
/// is guaranteed to be `true`. If `numerator == 0`, then the returned
/// value is guaranteed to be `false`.
///
/// See also the [`Bernoulli`] distribution, which may be faster if
/// sampling from the same `numerator` and `denominator` repeatedly.
///
/// This function is shorthand for
/// <code>[rng()].[random_ratio](RngExt::random_ratio)(<var>numerator</var>, <var>denominator</var>)</code>.
///
/// # Panics
///
/// If `denominator == 0` or `numerator > denominator`.
///
/// # Example
///
/// ```
/// println!("{}", rand::random_ratio(2, 3));
/// ```
///
/// [`Bernoulli`]: distr::Bernoulli
#[cfg(feature = "thread_rng")]
#[inline]
#[track_caller]
pub fn random_ratio(numerator: u32, denominator: u32) -> bool {
    rng().random_ratio(numerator, denominator)
}

/// Fill any type implementing [`Fill`] with random data
///
/// This function is shorthand for
/// <code>[rng()].[fill](RngExt::fill)(<var>dest</var>)</code>.
///
/// # Example
///
/// ```
/// let mut arr = [0i8; 20];
/// rand::fill(&mut arr[..]);
/// ```
///
/// Note that you can instead use [`random()`] to generate an array of random
/// data, though this is slower for small elements (smaller than the RNG word
/// size).
#[cfg(feature = "thread_rng")]
#[inline]
#[track_caller]
pub fn fill<T: Fill>(dest: &mut [T]) {
    Fill::fill_slice(dest, &mut rng())
}

#[cfg(test)]
mod test {
    use super::*;
    use core::convert::Infallible;

    /// Construct a deterministic RNG with the given seed
    pub fn rng(seed: u64) -> impl Rng {
        // For tests, we want a statistically good, fast, reproducible RNG.
        // PCG32 will do fine, and will be easy to embed if we ever need to.
        const INC: u64 = 11634580027462260723;
        rand_pcg::Pcg32::new(seed, INC)
    }

    /// Construct a generator yielding a constant value
    pub fn const_rng(x: u64) -> StepRng {
        StepRng(x, 0)
    }

    /// Construct a generator yielding an arithmetic sequence
    pub fn step_rng(x: u64, increment: u64) -> StepRng {
        StepRng(x, increment)
    }

    #[derive(Clone)]
    pub struct StepRng(u64, u64);
    impl TryRng for StepRng {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            self.try_next_u64().map(|x| x as u32)
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            let res = self.0;
            self.0 = self.0.wrapping_add(self.1);
            Ok(res)
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
            rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn rng_reader() {
        use std::io::Read;

        let mut rng = StepRng(255, 1);
        let mut buf = [0u8; 24];
        let expected = [
            255, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        ];

        RngReader(&mut rng).read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &expected);

        RngReader(StepRng(255, 1)).read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &expected);
    }

    #[test]
    #[cfg(feature = "thread_rng")]
    fn test_random() {
        let _n: u64 = random();
        let _f: f32 = random();
        #[allow(clippy::type_complexity)]
        let _many: (
            (),
            [(u32, bool); 3],
            (u8, i8, u16, i16, u32, i32, u64, i64),
            (f32, (f64, (f64,))),
        ) = random();
    }

    #[test]
    #[cfg(feature = "thread_rng")]
    fn test_range() {
        let _n: usize = random_range(42..=43);
        let _f: f32 = random_range(42.0..43.0);
    }
}
