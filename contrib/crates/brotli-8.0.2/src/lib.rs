#![no_std]
#![allow(non_snake_case)]
#![allow(unused_parens)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![cfg_attr(feature = "benchmark", feature(test))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(
    feature = "no-stdlib-ffi-binding",
    cfg_attr(not(feature = "std"), feature(lang_items))
)]

#[macro_use]
// <-- for debugging, remove xprintln from bit_reader and replace with println
#[cfg(feature = "std")]
extern crate std;
extern crate alloc_no_stdlib as alloc;
#[cfg(feature = "std")]
extern crate alloc_stdlib;
extern crate brotli_decompressor;

pub mod concat;
pub mod enc;
#[cfg(feature = "ffi-api")]
pub mod ffi;

pub use alloc::{AllocatedStackMemory, Allocator, SliceWrapper, SliceWrapperMut, StackAllocator};

#[cfg(feature = "std")]
pub use alloc_stdlib::HeapAlloc;
#[cfg(feature = "std")]
pub use brotli_decompressor::copy_from_to;
pub use brotli_decompressor::io_wrappers::{CustomRead, CustomWrite};
#[cfg(feature = "std")]
pub use brotli_decompressor::io_wrappers::{IntoIoReader, IoReaderWrapper, IoWriterWrapper};
#[cfg(feature = "std")]
pub use brotli_decompressor::reader::Decompressor;
pub use brotli_decompressor::reader::DecompressorCustomIo;
pub use brotli_decompressor::transform::TransformDictionaryWord;
#[cfg(feature = "std")]
pub use brotli_decompressor::writer::DecompressorWriter;
pub use brotli_decompressor::writer::DecompressorWriterCustomIo;
#[cfg(feature = "std")]
pub use brotli_decompressor::BrotliDecompress;
#[cfg(feature = "std")]
pub use brotli_decompressor::BrotliDecompressCustomAlloc;
pub use brotli_decompressor::HuffmanCode; // so we can make custom allocator for decompression
pub use brotli_decompressor::{
    dictionary, reader, transform, writer, BrotliDecompressCustomIo,
    BrotliDecompressCustomIoCustomDict, BrotliDecompressStream, BrotliResult, BrotliState,
};

pub use self::enc::combined_alloc::CombiningAllocator;
pub use crate::enc::input_pair::{InputPair, InputReference, InputReferenceMut};
pub use crate::enc::interface::SliceOffset;
#[cfg(feature = "std")]
pub use crate::enc::reader::CompressorReader;
pub use crate::enc::reader::CompressorReaderCustomIo;
#[cfg(feature = "std")]
pub use crate::enc::writer::CompressorWriter;
pub use crate::enc::writer::CompressorWriterCustomIo;
pub use crate::enc::{interface, BrotliCompressCustomIo, BrotliCompressCustomIoCustomDict};
#[cfg(feature = "std")]
pub use crate::enc::{BrotliCompress, BrotliCompressCustomAlloc};

pub const VERSION: u8 = 1;

// interface
// pub fn BrotliDecompressStream(mut available_in: &mut usize,
//                               input_offset: &mut usize,
//                               input: &[u8],
//                               mut available_out: &mut usize,
//                               mut output_offset: &mut usize,
//                               mut output: &mut [u8],
//                               mut total_out: &mut usize,
//                               mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>);
