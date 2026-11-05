/*!
This crate provides an implementation of the
[Snappy compression format](https://github.com/google/snappy/blob/master/format_description.txt),
as well as the
[framing format](https://github.com/google/snappy/blob/master/framing_format.txt).
The goal of Snappy is to provide reasonable compression at high speed. On a
modern CPU, Snappy can compress data at about 300 MB/sec or more and can
decompress data at about 800 MB/sec or more.

# Install

To use this crate with
[Cargo](https://doc.rust-lang.org/cargo/),
simply add it as a dependency to your `Cargo.toml`:

```ignore
[dependencies]
snap = "1"
```

# Overview

This crate provides two ways to use Snappy. The first way is through the
[`read::FrameDecoder`](read/struct.FrameDecoder.html)
and
[`write::FrameEncoder`](write/struct.FrameEncoder.html)
types, which implement the `std::io::Read` and `std::io::Write` traits with the
Snappy frame format. Unless you have a specific reason to the contrary, you
should only use the Snappy frame format. Specifically, the Snappy frame format
permits streaming compression or decompression.

The second way is through the
[`raw::Decoder`](raw/struct.Decoder.html)
and
[`raw::Encoder`](raw/struct.Encoder.html)
types. These types provide lower level control to the raw Snappy format, and
don't support a streaming interface directly. You should only use these types
if you know you specifically need the Snappy raw format.

Finally, the `Error` type in this crate provides an exhaustive list of error
conditions that are probably useless in most circumstances. Therefore,
`From<snap::Error> for io::Error` is implemented in this crate, which will let
you automatically convert a Snappy error to an `std::io::Error` (when using
`?`) with an appropriate error message to display to an end user.

# Example: compress data on `stdin`

This program reads data from `stdin`, compresses it and emits it to `stdout`.
This example can be found in `examples/compress.rs`:

```no_run
use std::io;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut rdr = stdin.lock();
    // Wrap the stdout writer in a Snappy writer.
    let mut wtr = snap::write::FrameEncoder::new(stdout.lock());
    io::copy(&mut rdr, &mut wtr).expect("I/O operation failed");
}
```

# Example: decompress data on `stdin`

This program reads data from `stdin`, decompresses it and emits it to `stdout`.
This example can be found in `examples/decompress.rs`:

```no_run
use std::io;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    // Wrap the stdin reader in a Snappy reader.
    let mut rdr = snap::read::FrameDecoder::new(stdin.lock());
    let mut wtr = stdout.lock();
    io::copy(&mut rdr, &mut wtr).expect("I/O operation failed");
}
```
*/

#![deny(missing_docs)]

#[cfg(test)]
doc_comment::doctest!("../README.md");

pub use crate::error::{Error, Result};

/// We don't permit compressing a block bigger than what can fit in a u32.
const MAX_INPUT_SIZE: u64 = std::u32::MAX as u64;

/// The maximum number of bytes that we process at once. A block is the unit
/// at which we scan for candidates for compression.
const MAX_BLOCK_SIZE: usize = 1 << 16;

mod bytes;
mod compress;
mod crc32;
mod crc32_table;
mod decompress;
mod error;
mod frame;
pub mod raw;
pub mod read;
mod tag;
pub mod write;
