# `multiversion_no_op`

This crate provides a pass-through function attribute called `multiversion`.

The purpose of this crate is to optimize build times on targets that do not actually
need multiversioning. That is, targets that need multiversioning can use the real
[`multiversion`](https://crates.io/crates/multiversion) crate, and targets that do
not need multiversioning can use this crate instead.

This crate works in the `no_std` context.

## License 

Apache-2.0 OR MIT; the file named `COPYRIGHT`.

## Usage

See [`encoding_rs`](https://github.com/hsivonen/encoding_rs/) crate for a usage example.
