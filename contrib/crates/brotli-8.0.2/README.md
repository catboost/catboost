# rust-brotli

[![crates.io](https://img.shields.io/crates/v/brotli.svg)](https://crates.io/crates/brotli)
[![Build Status](https://travis-ci.org/dropbox/rust-brotli.svg?branch=master)](https://travis-ci.org/dropbox/rust-brotli)

## What's new in 8.0.2
Fix for memory leak in ffi API

## What's new in 8.0.1
Compatibility for ffi builds

## What's new in 8.0.0
* Fixed LZ77 to comply with the specification
  * No longer seed the context by the end of the lz77 dictionary. Instead use
    zero for the seed as would happen without a dictionary. This matches the
    behavior of brotli with a custom dictionary as specified in
  * https://datatracker.ietf.org/doc/draft-vandevenne-shared-brotli-format/

## What's new in 7.0.0
* Fix error with short writes
* allow quality=10 for certain APIs and make it default to 9.5

## What's new in 6.0.0
  * Remove unused SIMD use statements
  * hide a few warnings - these are TODOs, and should be fixed in separate PRs
  * do NOT build SIMD as part of MSRV -- doesn't make any sense to combine nightly with MSRV

## What's new in 5.0.0
* The FFI is no longer active by default to avoid ODR issues if multiple versions of brotli are included in several dependent crates.

## What's new in 4.0.0
Pinned to a rust-brotli-decompressor that can disable the ffi with the ffi-api
flag.
This can help avoid symbol conflicts with other brotli libs.

## What's new in 3.5
Updated SIMD support.
Better CI integration.
Cleaned up some of the clippy warnings.

## What's new in 3.4
Brotli decompressor's reader and writer has better behavior when operating upon brotli streams with extra bits at the end.
Optional features like stdsimd are now tested or disabled for now.

## What's new in 3.2
* into_inner conversions for both Reader and Writer classes

## What's new in 3.0
* A fully compatible FFI for drop-in compatibiltiy with the https://github.com/google/brotli binaries
  * custom allocators fully supported
* Multithreaded compression so multiple threads can operate in unison on a single file
* Concatenatability mode to add the feature requested in https://github.com/google/brotli/issues/628
  * binary tool catbrotli can accomplish this if the first file was specified with -apendable and the second with -catable
* validation mode where a file is double-checked to be able to be decompressed with the same settings; useful for benchmarking or fuzzing
* Magic Number: where the brotli file can have a useful header with a few magic bytes, concatability info and a final output size for pre-allocating memory

## What's new in 2.5
* In 2.5 The callback also passes down an allocator to make new StaticCommands and PDFs and 256 bit floating point vectors.
* In 2.4 The callback with the compression intermediate representation now passes a full metablock at a time. Also these items are mutable
in case futher optimization is desired

## What's new in 2.3

* Flush now produces output instead of calling finish on the stream. This allows you to use the writer abstraction to
get immediate output without having to resort to the CompressStream internal abstraction

## Project Requirements

Direct no-stdlib port of the C brotli compressor to Rust

no dependency on the Rust stdlib: this library would be ideal for decompressing within a rust kernel among other things.

This is useful to see how C and Rust compare in an apples-to-apples
comparison where the same algorithms and data structures and
optimizations are employed.

## Compression Usage

Rust brotli currently supports compression levels 0 - 11
They should be bitwise identical to the brotli C compression engine at compression levels 0-9
Recommended lg_window_size is between 20 and 22

### With the io::Read abstraction
```rust
let mut input = brotli::CompressorReader::new(&mut io::stdin(), 4096 /* buffer size */,
                                              quality as u32, lg_window_size as u32);
```
then you can simply read input as you would any other io::Read class

### With the io::Write abstraction

```rust
let mut writer = brotli::Compressor::new(&mut io::stdout(), 4096 /* buffer size */,
                                         quality as u32, lg_window_size as u32);
```

There are also methods to build Compressor Readers or Writers using the with_params static function

eg:
```rust
let params = BrotliEncoderParams::default();
// modify params to fit the application needs
let mut writer = brotli::Compressor::with_params(&mut io::stdout(), 4096 /* buffer size */,
                                         params);
```
or for the reader
```rust
let params = BrotliEncoderParams::default();
// modify params to fit the application needs
let mut writer = brotli::CompressorReader::with_params(&mut io::stdin(), 4096 /* buffer size */,
                                                       params);
```


### With the Stream Copy abstraction

```rust
match brotli::BrotliCompress(&mut io::stdin(), &mut io::stdout(), &brotli_encoder_params) {
    Ok(_) => {},
    Err(e) => panic!("Error {:?}", e),
}
```

## Decompression Usage

### With the io::Read abstraction

```rust
let mut input = brotli::Decompressor::new(&mut io::stdin(), 4096 /* buffer size */);
```
then you can simply read input as you would any other io::Read class

### With the io::Write abstraction

```rust
let mut writer = brotli::DecompressorWriter::new(&mut io::stdout(), 4096 /* buffer size */);
```

### With the Stream Copy abstraction

```rust
match brotli::BrotliDecompress(&mut io::stdin(), &mut io::stdout()) {
    Ok(_) => {},
    Err(e) => panic!("Error {:?}", e),
}
```

### With manual memory management

There are 3 steps to using brotli without stdlib

1. setup the memory manager
2. setup the BrotliState
3. in a loop, call BrotliDecompressStream

in Detail

```rust
// at global scope declare a MemPool type -- in this case we'll choose the heap to
// avoid unsafe code, and avoid restrictions of the stack size

declare_stack_allocator_struct!(MemPool, heap);

// at local scope, make a heap allocated buffers to hold uint8's uint32's and huffman codes
let mut u8_buffer = define_allocator_memory_pool!(4096, u8, [0; 32 * 1024 * 1024], heap);
let mut u32_buffer = define_allocator_memory_pool!(4096, u32, [0; 1024 * 1024], heap);
let mut hc_buffer = define_allocator_memory_pool!(4096, HuffmanCode, [0; 4 * 1024 * 1024], heap);
let heap_u8_allocator = HeapPrealloc::<u8>::new_allocator(4096, &mut u8_buffer, bzero);
let heap_u32_allocator = HeapPrealloc::<u32>::new_allocator(4096, &mut u32_buffer, bzero);
let heap_hc_allocator = HeapPrealloc::<HuffmanCode>::new_allocator(4096, &mut hc_buffer, bzero);

// At this point no more syscalls are going to be needed since everything can come from the allocators.

// Feel free to activate SECCOMP jailing or other mechanisms to secure your application if you wish.

// Now it's possible to setup the decompressor state
let mut brotli_state = BrotliState::new(heap_u8_allocator, heap_u32_allocator, heap_hc_allocator);

// at this point the decompressor simply needs an input and output buffer and the ability to track
// the available data left in each buffer
loop {
    result = BrotliDecompressStream(&mut available_in, &mut input_offset, &input.slice(),
                                    &mut available_out, &mut output_offset, &mut output.slice_mut(),
                                    &mut written, &mut brotli_state);

    // just end the decompression if result is BrotliResult::ResultSuccess or BrotliResult::ResultFailure
}
```

This interface is the same interface that the C brotli decompressor uses

Also feel free to use custom allocators that invoke Box directly.
This example illustrates a mechanism to avoid subsequent syscalls after the initial allocation

## Using the C interface

rust-brotli is a drop-in replacement for the official https://github.com/google/brotli C
implementation. That means you can use it from any place that supports that library.
To build rust-brotli in this manner enter the c subdirectory and run make there

cd c && make

this should build c/target/release/libbrotli.so and should build the vanilla
command line tool in C for compressing and decompressing any brotli file.

the libbrotli.so in c/target/release should be able to replace any other libbrotli.so
file, but with all the advantages of using safe rust (except in the FFI bindings)

The code also allows a wider range of options, including forcing the prediction mode
(eg UTF8 vs signed vs MSB vs LSB) and changing the weight of the literal cost from 540
 to other values.

Additionally the CATABLE and APPENDABLE options are exposed and allow concatenation of files
created in this manner.

Specifically CATABLE files can be concatenated in any order using the catbrotli tool
and APPENDABLE files can be the first file in a sequence of catable files...
eg you can combine
appendable.br catable1.br catable2.br catable3.br

or simply
catable0.br catable1.br catable2.br catable3.br

