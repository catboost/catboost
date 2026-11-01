#![cfg(test)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

extern crate brotli_decompressor;
extern crate core;

use brotli::enc::threading::{Owned, SendAlloc};
use brotli_decompressor::{SliceWrapper, SliceWrapperMut};

use super::brotli::enc::{
    compress_multi, compress_multi_no_threadpool, BrotliEncoderMaxCompressedSizeMulti,
    BrotliEncoderParams, UnionHasher,
};
use super::integration_tests::UnlimitedBuffer;
use super::{new_brotli_heap_alloc, Rebox};

static RANDOM_THEN_UNICODE: &[u8] = include_bytes!("../../testdata/random_then_unicode");
static ALICE: &[u8] = include_bytes!("../../testdata/alice29.txt");

struct SliceRef<'a>(&'a [u8]);
impl<'a> SliceWrapper<u8> for SliceRef<'a> {
    fn slice(&self) -> &[u8] {
        self.0
    }
}

fn multi_threaded_split_compression_test(
    input_data: &'static [u8],
    num_threads: usize,
    quality: i32,
    catable: bool,
    expected_size: usize,
) {
    let mut params = BrotliEncoderParams::default();
    params.quality = quality;
    params.magic_number = true;
    if catable {
        params.catable = true;
        params.use_dictionary = false;
    }
    let mut output = Rebox::from(vec![
        0u8;
        BrotliEncoderMaxCompressedSizeMulti(
            input_data.len(),
            num_threads
        )
    ]);
    let mut alloc_per_thread = [
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
    ];
    if num_threads > alloc_per_thread.len() {
        panic!(
            "Too many threads requested {} > {}",
            num_threads,
            alloc_per_thread.len()
        );
    }
    let res = compress_multi(
        &params,
        &mut Owned::new(SliceRef(input_data)),
        output.slice_mut(),
        &mut alloc_per_thread[..num_threads],
    );
    let observed_size = res.unwrap();
    if observed_size > expected_size {
        assert_eq!(observed_size, expected_size);
    }
    assert_ne!(observed_size, 0);
    let mut compressed_version = UnlimitedBuffer::new(&output.slice()[..observed_size]);
    let mut rt = UnlimitedBuffer::new(&[]);
    match super::decompress(&mut compressed_version, &mut rt, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    assert_eq!(rt.data(), input_data);
}
#[test]
fn multi_threaded_split_compression_test_1() {
    multi_threaded_split_compression_test(RANDOM_THEN_UNICODE, 1, 3, false, 155808)
}
#[test]
fn multi_threaded_split_compression_test_2() {
    multi_threaded_split_compression_test(RANDOM_THEN_UNICODE, 2, 4, false, 151857)
}
#[test]
fn multi_threaded_split_compression_test_3() {
    multi_threaded_split_compression_test(RANDOM_THEN_UNICODE, 3, 5, false, 144325)
}
#[test]
fn multi_threaded_split_compression_test_4() {
    multi_threaded_split_compression_test(RANDOM_THEN_UNICODE, 4, 10, true, 136812)
}
#[test]
fn multi_threaded_split_compression_test_5() {
    multi_threaded_split_compression_test(RANDOM_THEN_UNICODE, 5, 9, false, 139126)
}
#[test]
fn multi_threaded_split_compression_test_1b1() {
    multi_threaded_split_compression_test(&RANDOM_THEN_UNICODE[..1], 5, 9, false, 139126)
}
#[test]
fn multi_threaded_split_compression_test_1b5() {
    multi_threaded_split_compression_test(&RANDOM_THEN_UNICODE[..1], 5, 9, false, 139125)
}

#[test]
fn multi_threaded_split_compression_test_0b1() {
    multi_threaded_split_compression_test(&[], 5, 9, false, 139125)
}
#[test]
fn multi_threaded_split_compression_test_0b5() {
    multi_threaded_split_compression_test(&[], 5, 9, false, 139125)
}

fn thread_spawn_per_job_split_compression_test(
    input_data: &'static [u8],
    num_threads: usize,
    quality: i32,
    catable: bool,
    expected_size: usize,
) {
    let mut params = BrotliEncoderParams::default();
    params.quality = quality;
    params.magic_number = true;
    if catable {
        params.catable = true;
        params.use_dictionary = false;
    }
    params.favor_cpu_efficiency = true; // this should test both paths
    let mut output = Rebox::from(vec![
        0u8;
        BrotliEncoderMaxCompressedSizeMulti(
            input_data.len(),
            num_threads
        )
    ]);
    let mut alloc_per_thread = [
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
        SendAlloc::new(new_brotli_heap_alloc(), UnionHasher::Uninit),
    ];
    if num_threads > alloc_per_thread.len() {
        panic!(
            "Too many threads requested {} > {}",
            num_threads,
            alloc_per_thread.len()
        );
    }
    let res = compress_multi_no_threadpool(
        &params,
        &mut Owned::new(SliceRef(input_data)),
        output.slice_mut(),
        &mut alloc_per_thread[..num_threads],
    );
    let observed_size = res.unwrap();
    if observed_size > expected_size {
        assert_eq!(observed_size, expected_size);
    }
    assert_ne!(observed_size, 0);
    let mut compressed_version = UnlimitedBuffer::new(&output.slice()[..observed_size]);
    let mut rt = UnlimitedBuffer::new(&[]);
    match super::decompress(&mut compressed_version, &mut rt, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    assert_eq!(rt.data(), input_data);
}
#[test]
fn thread_spawn_per_job_split_compression_test_1() {
    thread_spawn_per_job_split_compression_test(RANDOM_THEN_UNICODE, 1, 3, false, 155808)
}
#[test]
fn thread_spawn_per_job_split_compression_test_3() {
    thread_spawn_per_job_split_compression_test(RANDOM_THEN_UNICODE, 3, 5, false, 144325)
}
#[test]
fn thread_spawn_per_job_split_compression_test_1b1() {
    thread_spawn_per_job_split_compression_test(&RANDOM_THEN_UNICODE[..1], 1, 3, false, 155808)
}
#[test]
fn thread_spawn_per_job_split_compression_test_1b3() {
    thread_spawn_per_job_split_compression_test(&RANDOM_THEN_UNICODE[..1], 3, 5, false, 144325)
}
#[test]
fn thread_spawn_per_job_split_compression_test_0b1() {
    thread_spawn_per_job_split_compression_test(&[], 1, 3, false, 155808)
}
#[test]
fn thread_spawn_per_job_split_compression_test_0b3() {
    thread_spawn_per_job_split_compression_test(&[], 3, 5, false, 144325)
}
