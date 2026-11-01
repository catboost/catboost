# rust-brotli-decompressor

[![crates.io](https://img.shields.io/crates/v/brotli-decompressor.svg)](https://crates.io/crates/brotli-decompressor)
[![Build Status](https://travis-ci.org/dropbox/rust-brotli-decompressor.svg?branch=master)](https://travis-ci.org/dropbox/rust-brotli-decompressor)

## What's new in version 5.0.0
No longer seed the context by the end of the lz77 dictionary. Instead use
zero for the seed as would happen without a dictionary. This matches the
behavior of brotli with a custom dictionary as specified in
https://datatracker.ietf.org/doc/draft-vandevenne-shared-brotli-format/

## What's new in version 4.0.3
* Better handling of corrupt brotli files.

## What's new in version 4.0.2
* Better handling of corrupt brotli files.

## What's new in version 4.0.0
* The FFI is no longer active by default to avoid ODR issues if multiple versions of brotli are included in several dependent crates.

## What's new in version 3.0.0
* The FFI is feature gated behind the ffi-api feature flag (on-by-default)

## What's new in version 2.5.0
* If you call write with extra bytes it will only return the bytes consumed
* Further calls to write will return Ok(0)

## What's new in version 2.4.0
* If you call read until the file is ended, it will return 0 bytes.
* Further calls to read will error if there are extra unconsumed bytes in the file.

## What's new in version 2.3.5
* Fix bug in BrotliFillBitWindow

## What's new in version 2.3.4
* Check for fully consumed buffers in the decompressor reader/writer.

## What's new in version 2.3.x
* Error handling for Write and Read implementations.
* Fixed issue with small buffer sizes on certain files when repeatedly calling Decode stream
* Expose BrotliDecoderIsFinished

## What's new in version 2.2.x
* into_impl for reader and writer classes
* removed BrotliStateCleanup since it happens upon drop()

## What's new in version 2.1.2
* Better handling of transient errors ( fixes #4 )
* Do not panic in debug mode on unexpected bytes (handle arithmetic overflow)
* Smaller stack allocations
* Never create slice::from_raw_parts with nil
* Better panic reporting to C FFI
* Backport fixes to brotli issues 502 and 506
b
## What's new in version 2.0.0

* Legacy Custom dictionaries (mostly useful for testing multithreaded brotli encoding and experimentation)
* New mechanism to request that the library should only depend on custom allocations: the --no-default-features flag since --features=std is on by default.
* Fully compatible C FFI to match the https://github.com/google/brotli C API and become a drop-in replacement

## Project Requirements

Direct no-stdlib port of the C brotli decompressor to Rust if --no-default-features is passed into the build

no dependency on the Rust stdlib: this library would be ideal for decompressing within a rust kernel among other things.

This will be useful to see how C and Rust compare in an apples-to-apples
comparison where the same algorithms and data structures and
optimizations are employed.

The current expected performance losses come from

1. an extra indirection in the hgroups
2. array bounds checks on every access
3. no ability to load a full aligned 64 bit or 128 bit item from a [u8]

the system also enables all syscalls to be "frontloaded" in the initial generation
of a memory pool for the allocator. Afterwards, SECCOMP can be activated or
other mechanisms can be used to secure the application, if desired

## Linking rust-brotli-decompressor with C code using the zero-cost rust FFI abstraction

This library has FFI exports which comply with the original C interfaces.
To build them, enter the c directory and just type make there.
That will build a small example program and the cdylib with the appropriate ffi in place to link against

the example, called c/main.c shows how to decompress a program using the streaming interface and the nonstreaming interface.

If a nostdlib version is desired, then an unstable rust must be used (to enable the custom panic handler)
and then the BrotliDecoderDecompress function is deactivated since that has no facilities for specifying a custom malloc

a customized malloc must be used if a nostdlib build is chosen and additionally the no-stdlib-ffi-binding cargo feature must be set
eg

cargo build --features='no-stdlib no-stdlib-ffi-binding' --release


## Usage

### With the io::Read abstraction

```rust
let mut input = brotli_decompressor::Decompressor::new(&mut io::stdin(), 4096 /* buffer size */);
```
then you can simply read input as you would any other io::Read class

### With the Stream Copy abstraction

```rust
match brotli_decompressor::BrotliDecompress(&mut io::stdin(), &mut io::stdout(), 65536 /* buffer size */) {
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
