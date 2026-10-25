lz4
====

[![Build Status](https://github.com/10XGenomics/lz4-rs/workflows/Test/badge.svg?branch=master)](https://github.com/10XGenomics/lz4-rs/actions?query=branch%3Amaster)
[![Crates.io](https://img.shields.io/crates/v/lz4.svg)](https://crates.io/crates/lz4)
[![Join the chat at https://gitter.im/bozaro/lz4-rs](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/bozaro/lz4-rs?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Rustdoc](https://img.shields.io/badge/doc-rustdoc-green.svg)](https://docs.rs/lz4/)

NOTE: 10xGenomics is the new official home of lz4-rs, replacing [https://github.com/bozaro/lz4-rs](https://github.com/bozaro/lz4-rs)

This repository contains binding for lz4 compression library (https://github.com/Cyan4973/lz4).

LZ4 is a very fast lossless compression algorithm, providing compression speed at 400 MB/s per core, with near-linear scalability for multi-threaded applications. It also features an extremely fast decoder, with speed in multiple GB/s per core, typically reaching RAM speed limits on multi-core systems.

## Usage

Put this in your `Cargo.toml`:
```toml
[dependencies]
lz4 = "1.23.1"
```

Sample code for compression/decompression:
```rust
extern crate lz4;

use std::env;
use std::fs::File;
use std::io::{self, Result};
use std::path::{Path, PathBuf};

use lz4::{Decoder, EncoderBuilder};

fn main() {
    println!("LZ4 version: {}", lz4::version());

    for path in env::args().skip(1).map(PathBuf::from) {
        if let Some("lz4") = path.extension().and_then(|e| e.to_str()) {
            decompress(&path, &path.with_extension("")).unwrap();
        } else {
            compress(&path, &path.with_extension("lz4")).unwrap();
        }
    }
}

fn compress(source: &Path, destination: &Path) -> Result<()> {
    println!("Compressing: {} -> {}", source.display(), destination.display());

    let mut input_file = File::open(source)?;
    let output_file = File::create(destination)?;
    let mut encoder = EncoderBuilder::new()
        .level(4)
        .build(output_file)?;
    io::copy(&mut input_file, &mut encoder)?;
    let (_output, result) = encoder.finish();
    result
}

fn decompress(source: &Path, destination: &Path) -> Result<()> {
    println!("Decompressing: {} -> {}", source.display(), destination.display());

    let input_file = File::open(source)?;
    let mut decoder = Decoder::new(input_file)?;
    let mut output_file = File::create(destination)?;
    io::copy(&mut decoder, &mut output_file)?;

    Ok(())
}
```
