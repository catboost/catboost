#![cfg(feature = "benchmark")]
#![cfg(feature = "std")]

extern crate test;

use alloc_stdlib::StandardAlloc;

use super::*;

static RANDOM_THEN_UNICODE: &'static [u8] = include_bytes!("../../../testdata/random_then_unicode");
static FINALIZE_DATA: &'static [u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20,
];
const TEST_LEN: usize = 256 * 1024;
const DISTANCE_CACHE: &'static [i32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // distance cache

fn make_generic_hasher() -> AdvHasher<H5Sub, StandardAlloc> {
    let params_hasher = BrotliHasherParams {
        type_: 5,
        block_bits: 6,
        bucket_bits: 15,
        num_last_distances_to_check: 10,
        hash_len: 4,
        literal_byte_score: 540,
    };
    let block_size = 1u64 << params_hasher.block_bits;
    let bucket_size = 1u64 << params_hasher.bucket_bits;
    let mut alloc = StandardAlloc::default();

    AdvHasher::<H5Sub, StandardAlloc> {
        buckets: alloc.alloc_cell((bucket_size * block_size) as usize),
        h9_opts: H9Opts::new(&params_hasher),
        num: alloc.alloc_cell(bucket_size as usize),
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H5Sub {
            hash_shift_: 32i32 - params_hasher.bucket_bits,
            bucket_size_: bucket_size as u32,
            block_bits_: params_hasher.block_bits as i32,
            block_mask_: block_size.wrapping_sub(1) as u32,
        },
    }
}
fn make_specialized_hasher() -> AdvHasher<HQ7Sub, StandardAlloc> {
    let params_hasher = BrotliHasherParams {
        type_: 5,
        block_bits: 6,
        bucket_bits: 15,
        num_last_distances_to_check: 10,
        hash_len: 4,
        literal_byte_score: 540,
    };
    let block_size = 1u64 << params_hasher.block_bits;
    let bucket_size = 1u64 << params_hasher.bucket_bits;
    let mut alloc = StandardAlloc::default();

    AdvHasher::<HQ7Sub, StandardAlloc> {
        buckets: alloc.alloc_cell((bucket_size * block_size) as usize),
        h9_opts: H9Opts::new(&params_hasher),
        num: alloc.alloc_cell(bucket_size as usize),
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: HQ7Sub {},
    }
}
#[bench]
fn bench_256k_basic_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        for i in 0..TEST_LEN {
            hasher.Store(testdata, usize::MAX, i);
        }
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_basic_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        for i in 0..TEST_LEN {
            hasher.Store(testdata, usize::MAX, i);
        }
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_opt_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptBatch(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_opt_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptBatch(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_mem_fetch_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptMemFetch(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_mem_fetch_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptMemFetch(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_mem_lazy_dupe_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptMemFetchLazyDupeUpdate(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}
#[bench]
fn bench_256k_mem_lazy_dupe_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptMemFetchLazyDupeUpdate(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_mem_random_dupe_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptRandomDupeUpdate(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_mem_random_dupe_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRangeOptRandomDupeUpdate(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_cur_generic(bench: &mut test::Bencher) {
    let mut hasher = make_generic_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRange(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}

#[bench]
fn bench_256k_cur_specialized(bench: &mut test::Bencher) {
    let mut hasher = make_specialized_hasher();
    bench.iter(|| {
        let testdata = test::black_box(RANDOM_THEN_UNICODE.split_at(TEST_LEN + 8).0);
        hasher.BulkStoreRange(testdata, usize::MAX, 0, TEST_LEN);
        let mut output = super::HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        hasher.FindLongestMatch(
            None,
            &[],
            test::black_box(FINALIZE_DATA), // data
            15,                             // ring mask
            DISTANCE_CACHE,
            8, // cur_x
            8,
            4,
            4,
            4,
            &mut output,
        );
    });
}
