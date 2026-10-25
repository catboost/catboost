// Copyright 2018 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Random number generators and adapters
//!
//! ## Generators
//!
//! This crate provides a small selection of generators.
//! See also [Types of generators] and [Our RNGs] in the book.
//!
//! ##### Non-deterministic generators
//!
//! -   [`SysRng`] is a stateless interface over the operating system's random number
//!     source. This is typically secure with some form of periodic re-seeding.
//! -   [`ThreadRng`], provided by [`crate::rng()`], is a handle to a
//!     thread-local generator with periodic seeding from [`SysRng`]. Because this
//!     is local, it is typically much faster than [`SysRng`]. It should be
//!     secure, but see documentation on [`ThreadRng`].
//!
//! ##### Standard generators
//!
//! These use selected best-in-class algorithms. They are deterministic but not
//! portable: the algorithms may be changed in any release and may be
//! platform-dependent.
//!
//! -   [`StdRng`] is a CSPRNG chosen for good performance and trust of security
//!     (based on reviews, maturity and usage). The current algorithm is
//!     [`ChaCha12Rng`], which is well established and rigorously analysed.
//!     [`StdRng`] is the deterministic generator used by [`ThreadRng`] but
//!     without the periodic reseeding or thread-local management.
//! -   [`SmallRng`] is a relatively simple, insecure generator designed to be
//!     fast, use little memory, and pass various statistical tests of
//!     randomness quality. The current algorithm is one of the Xoshiro
//!     generators below, depending on the target's pointer size.
//!
//! ##### Named portable generators
//!
//! These are similar to the [standard generators](#standard-generators), but
//! with the additional [guarantees of reproducibility]:
//!
//! -   [`Xoshiro256PlusPlus`] is a very fast 64-bit insecure generator using
//!     256 bits of state with good performance in statistical tests of quality
//! -   [`Xoshiro128PlusPlus`] is a very fast 32-bit insecure generator using
//!     128 bits of state with good performance in statistical tests of quality
//! -   [`ChaCha8Rng`], [`ChaCha12Rng`] and [`ChaCha20Rng`] are generators over
//!     the ChaCha stream cipher designed by Daniel J. Bernstein[^1].
//!
//! ### Additional generators
//!
//! -   The [`rdrand`] crate provides an interface to the RDRAND and RDSEED
//!     instructions available in modern Intel and AMD CPUs.
//! -   The [`rand_jitter`] crate provides a user-space implementation of
//!     entropy harvesting from CPU timer jitter, but is very slow and has
//!     [security issues](https://github.com/rust-random/rand/issues/699).
//! -   The [`rand_pcg`] crate provides portable implementations of a subset
//!     of the [PCG] family of small, insecure generators
//! -   The [`rand_xoshiro`] crate provides portable implementations of the
//!     [xoshiro] family of small, insecure generators
//!
//! For more, search [crates with the `rng` tag].
//!
//! ## Traits and functionality
//!
//! All generators implement [`TryRng`]. Most implement [`Rng`] (i.e.
//! `TryRng<Error = Infallible>`) and thus also implement [`RngExt`].
//! See also the [Random Values] chapter in the book.
//!
//! Secure RNGs may additionally implement the [`CryptoRng`] trait.
//!
//! Use the [`rand_core`] crate when implementing your own RNGs.
//!
//! [^1]: D. J. Bernstein, [*ChaCha, a variant of Salsa20*](https://cr.yp.to/chacha.html)
//!
//! [guarantees of reproducibility]: https://rust-random.github.io/book/crate-reprod.html
//! [Types of generators]: https://rust-random.github.io/book/guide-gen.html
//! [Our RNGs]: https://rust-random.github.io/book/guide-rngs.html
//! [Random Values]: https://rust-random.github.io/book/guide-values.html
//! [`RngExt`]: crate::RngExt
//! [`TryRng`]: crate::TryRng
//! [`Rng`]: crate::Rng
//! [`CryptoRng`]: crate::CryptoRng
//! [`SeedableRng`]: crate::SeedableRng
//! [`rdrand`]: https://crates.io/crates/rdrand
//! [`rand_jitter`]: https://crates.io/crates/rand_jitter
//! [`rand_pcg`]: https://crates.io/crates/rand_pcg
//! [`rand_xoshiro`]: https://crates.io/crates/rand_xoshiro
//! [crates with the `rng` tag]: https://crates.io/keywords/rng
//! [chacha]: https://cr.yp.to/chacha.html
//! [PCG]: https://www.pcg-random.org/
//! [xoshiro]: https://prng.di.unimi.it/

mod small;
mod xoshiro128plusplus;
mod xoshiro256plusplus;

#[cfg(feature = "std_rng")]
mod std;
#[cfg(feature = "thread_rng")]
pub(crate) mod thread;

pub use self::small::SmallRng;
pub use xoshiro128plusplus::Xoshiro128PlusPlus;
pub use xoshiro256plusplus::Xoshiro256PlusPlus;

#[cfg(feature = "std_rng")]
pub use self::std::StdRng;
#[cfg(feature = "thread_rng")]
pub use self::thread::ThreadRng;

#[cfg(feature = "chacha")]
pub use chacha20::{ChaCha8Rng, ChaCha12Rng, ChaCha20Rng};

#[cfg(feature = "sys_rng")]
pub use getrandom::{Error as SysError, SysRng};
