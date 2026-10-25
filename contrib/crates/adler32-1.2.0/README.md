[![Build Status](https://github.com/remram44/adler32-rs/workflows/Test/badge.svg)](https://github.com/remram44/adler32-rs/actions)
[![Win Build](https://ci.appveyor.com/api/projects/status/ekyg20rd6rwrus64/branch/master?svg=true)](https://ci.appveyor.com/project/remram44/adler32-rs)
[![Crates.io](https://img.shields.io/crates/v/adler32.svg)](https://crates.io/crates/adler32)
[![Documentation](https://docs.rs/adler32/badge.svg)](https://docs.rs/adler32)
[![License](https://img.shields.io/crates/l/adler32.svg)](https://github.com/remram44/adler32-rs/blob/master/LICENSE)

What is this?
=============

It is an implementation of the [Adler32 rolling hash algorithm](https://en.wikipedia.org/wiki/Adler-32) in the [Rust programming language](https://www.rust-lang.org/).

It is adapted from Jean-Loup Gailly's and Mark Adler's [original implementation in zlib](https://github.com/madler/zlib/blob/2fa463bacfff79181df1a5270fb67cc679a53e71/adler32.c).


#### Minimum Supported Version of Rust (MSRV)

`adler32-rs` can be built with Rust version 1.33 or later. This version may be raised in the future but that will be accompanied by a minor version increase.
