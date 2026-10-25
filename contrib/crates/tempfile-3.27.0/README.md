# tempfile

[![Crate](https://img.shields.io/crates/v/tempfile.svg)](https://crates.io/crates/tempfile)
[![Build Status](https://github.com/Stebalien/tempfile/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Stebalien/tempfile/actions/workflows/ci.yml?query=branch%3Amaster)

A secure, cross-platform, temporary file library for Rust. In addition to creating
temporary files, this library also allows users to securely open multiple
independent references to the same temporary file (useful for consumer/producer
patterns and surprisingly difficult to implement securely).

[Documentation](https://docs.rs/tempfile/)

## Usage

Minimum required Rust version: 1.63.0

Add this to your `Cargo.toml`:

```toml
[dependencies]
tempfile = "3"
```

## Supported Platforms

This crate supports all major operating systems:

- Linux
- Android
- MacOS
- Windows
- FreeBSD (likely other BSDs but we don't have CI for them)
- RedoxOS
- Wasm (build and link only, Wasm doesn't have a filesystem)
- WASI P1 & P2.

However:

- Android, RedoxOS, Wasm, and WASI targets all require the latest stable rust compiler.
- WASI P1/P2 does not define a default temporary directory. You'll need to explicitly call `tempfile::env::override_temp_dir` with a valid directory or temporary file creation will panic on this platform.
- WASI P1/P2 does not have file permissions.
- You _may_ need to override the temporary directory in Android as well to point at your application's per-app cache directory.

## Example

```rust
use std::fs::File;
use std::io::{Write, Read, Seek, SeekFrom};

fn main() {
    // Write
    let mut tmpfile: File = tempfile::tempfile().unwrap();
    write!(tmpfile, "Hello World!").unwrap();

    // Seek to start
    tmpfile.seek(SeekFrom::Start(0)).unwrap();

    // Read
    let mut buf = String::new();
    tmpfile.read_to_string(&mut buf).unwrap();
    assert_eq!("Hello World!", buf);
}
```
