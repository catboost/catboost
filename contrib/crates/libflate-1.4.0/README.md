libflate
========

[![libflate](https://img.shields.io/crates/v/libflate.svg)](https://crates.io/crates/libflate)
[![Documentation](https://docs.rs/libflate/badge.svg)](https://docs.rs/libflate)
[![Actions Status](https://github.com/sile/libflate/workflows/CI/badge.svg)](https://github.com/sile/libflate/actions)
[![Coverage Status](https://coveralls.io/repos/github/sile/libflate/badge.svg?branch=master)](https://coveralls.io/github/sile/libflate?branch=master)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust implementation of DEFLATE algorithm and related formats (ZLIB, GZIP).

Documentation
-------------

See [RustDoc Documentation](https://docs.rs/libflate).

The documentation includes some examples.

Installation
------------

Add following lines to your `Cargo.toml`:

```toml
[dependencies]
libflate = "1"
```

An Example
----------

Below is a command to decode GZIP stream that is read from the standard input:

```rust
extern crate libflate;

use std::io;
use libflate::gzip::Decoder;

fn main() {
    let mut input = io::stdin();
    let mut decoder = Decoder::new(&mut input).unwrap();
    io::copy(&mut decoder, &mut io::stdout()).unwrap();
}
```

An Informal Benchmark
---------------------

A brief comparison with [flate2](https://github.com/alexcrichton/flate2-rs) and
[inflate](https://github.com/PistonDevelopers/inflate):

```bash
$ cd libflate/flate_bench/
$ curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
$ gzip -d enwiki-latest-all-titles-in-ns0.gz
> ls -lh enwiki-latest-all-titles-in-ns0
-rw-rw-r-- 1 foo foo 265M May 18 05:19 enwiki-latest-all-titles-in-ns0

$ cargo run --release -- enwiki-latest-all-titles-in-ns0
# ENCODE (input_size=277303937)
- libflate: elapsed=8.137013s, size=83259010
-   flate2: elapsed=9.814607s, size=74692153

# DECODE (input_size=74217004)
- libflate: elapsed=1.354556s, size=277303937
-   flate2: elapsed=0.960907s, size=277303937
-  inflate: elapsed=1.926142s, size=277303937
```

References
----------

- DEFLATE: [RFC-1951](https://tools.ietf.org/html/rfc1951)
- ZLIB: [RFC-1950](https://tools.ietf.org/html/rfc1950)
- GZIP: [RFC-1952](https://tools.ietf.org/html/rfc1952)
