#![cfg(test)]
#![cfg(feature = "std")]

use alloc_stdlib::StandardAlloc;

use super::{
    AdvHasher, AnyHasher, BrotliHasherParams, CloneWithAlloc, H5Sub, H9Opts, HQ7Sub, Struct1,
};
use crate::enc::combined_alloc::allocate;
use crate::enc::{Allocator, SliceWrapper};

static RANDOM_THEN_UNICODE: &[u8] = include_bytes!("../../../testdata/random_then_unicode"); //&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55];

#[cfg(feature = "std")]
#[test]
fn test_bulk_store_range() {
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
    let mut buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    let mut num = alloc.alloc_cell(bucket_size as usize);

    let mut hasher_a = AdvHasher::<H5Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H5Sub {
            hash_shift_: 32i32 - params_hasher.bucket_bits,
            bucket_size_: bucket_size as u32,
            block_bits_: params_hasher.block_bits,
            block_mask_: block_size.wrapping_sub(1) as u32,
        },
    };
    buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    num = alloc.alloc_cell(bucket_size as usize);
    let mut hasher_b = hasher_a.clone_with_alloc(&mut alloc);
    assert!(hasher_a == hasher_b);
    let mut hasher_e = hasher_a.clone_with_alloc(&mut alloc);
    let mut hasher_c = AdvHasher::<HQ7Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: HQ7Sub {},
    };
    let mut hasher_d = hasher_c.clone_with_alloc(&mut alloc);
    assert!(hasher_d == hasher_c);
    hasher_a.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    hasher_c.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    for i in 15..RANDOM_THEN_UNICODE.len() - 8 {
        hasher_b.Store(RANDOM_THEN_UNICODE, usize::MAX, i);
    }
    hasher_d.StoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    let ret_start = hasher_e.StoreRangeOptBatch(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    assert!(ret_start > 15);
    hasher_e.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        ret_start,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    assert_eq!(hasher_a.buckets.slice(), hasher_c.buckets.slice());
    assert_eq!(hasher_b.buckets.slice(), hasher_d.buckets.slice());
    assert_eq!(hasher_a.num.slice(), hasher_c.num.slice());
    assert_eq!(hasher_b.num.slice(), hasher_d.num.slice());
    assert_eq!(hasher_a.buckets.slice(), hasher_b.buckets.slice());
    assert_eq!(hasher_c.buckets.slice(), hasher_d.buckets.slice());
    assert_eq!(hasher_a.num.slice(), hasher_b.num.slice());
    assert_eq!(hasher_c.num.slice(), hasher_d.num.slice());
    assert!(hasher_a == hasher_b);
    assert!(hasher_d == hasher_c);
    assert!(hasher_a == hasher_e);
}
#[cfg(feature = "std")]
#[test]
// does not use the fancy optimizations for q7
fn test_bulk_store_range_off_spec() {
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
    let mut buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    let mut num = alloc.alloc_cell(bucket_size as usize);

    let mut hasher_a = AdvHasher::<H5Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H5Sub {
            hash_shift_: 32i32 - params_hasher.bucket_bits,
            bucket_size_: bucket_size as u32,
            block_bits_: params_hasher.block_bits,
            block_mask_: block_size.wrapping_sub(1) as u32,
        },
    };
    buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    num = alloc.alloc_cell(bucket_size as usize);
    let mut hasher_b = hasher_a.clone_with_alloc(&mut alloc);
    assert!(hasher_a == hasher_b);
    let mut hasher_c = AdvHasher::<HQ7Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: HQ7Sub {},
    };
    let mut hasher_d = hasher_c.clone_with_alloc(&mut alloc);
    assert!(hasher_d == hasher_c);
    hasher_a.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        0x0fff,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    hasher_c.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        0x0fff,
        15,
        RANDOM_THEN_UNICODE.len() - 8,
    );
    hasher_c.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        0x0fff,
        RANDOM_THEN_UNICODE.len(),
        RANDOM_THEN_UNICODE.len() - 8,
    ); // noop
    for i in 15..RANDOM_THEN_UNICODE.len() - 8 {
        hasher_b.Store(RANDOM_THEN_UNICODE, 0x0fff, i);
        hasher_d.Store(RANDOM_THEN_UNICODE, 0x0fff, i);
    }
    assert_eq!(hasher_a.buckets.slice(), hasher_c.buckets.slice());
    assert_eq!(hasher_b.buckets.slice(), hasher_d.buckets.slice());
    assert_eq!(hasher_a.num.slice(), hasher_c.num.slice());
    assert_eq!(hasher_b.num.slice(), hasher_d.num.slice());
    assert!(hasher_a == hasher_b);
    assert!(hasher_d == hasher_c);
}

#[cfg(feature = "std")]
#[test]
fn test_bulk_store_range_pow2() {
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
    let mut buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    let mut num = alloc.alloc_cell(bucket_size as usize);

    let mut hasher_a = AdvHasher::<H5Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H5Sub {
            hash_shift_: 32i32 - params_hasher.bucket_bits,
            bucket_size_: bucket_size as u32,
            block_bits_: params_hasher.block_bits,
            block_mask_: block_size.wrapping_sub(1) as u32,
        },
    };
    buckets = allocate::<u32, _>(&mut alloc, (bucket_size * block_size) as usize);
    num = alloc.alloc_cell(bucket_size as usize);
    let mut hasher_b = hasher_a.clone_with_alloc(&mut alloc);
    assert!(hasher_a == hasher_b);
    let mut hasher_e = hasher_a.clone_with_alloc(&mut alloc);
    let mut hasher_c = AdvHasher::<HQ7Sub, StandardAlloc> {
        buckets,
        h9_opts: H9Opts::new(&params_hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params_hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: HQ7Sub {},
    };
    let mut hasher_d = hasher_c.clone_with_alloc(&mut alloc);
    assert!(hasher_d == hasher_c);
    hasher_a.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        RANDOM_THEN_UNICODE.len() - 64 - 3,
        RANDOM_THEN_UNICODE.len() - 3,
    );
    hasher_c.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        RANDOM_THEN_UNICODE.len() - 64 - 3,
        RANDOM_THEN_UNICODE.len() - 3,
    );
    for i in RANDOM_THEN_UNICODE.len() - 64 - 3..RANDOM_THEN_UNICODE.len() - 3 {
        hasher_b.Store(RANDOM_THEN_UNICODE, usize::MAX, i);
    }
    hasher_d.StoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        RANDOM_THEN_UNICODE.len() - 64 - 3,
        RANDOM_THEN_UNICODE.len() - 3,
    );
    let ret_start = hasher_e.StoreRangeOptBatch(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        RANDOM_THEN_UNICODE.len() - 64 - 3,
        RANDOM_THEN_UNICODE.len() - 3,
    );
    assert!(ret_start > 15);
    hasher_e.BulkStoreRange(
        RANDOM_THEN_UNICODE,
        usize::MAX,
        ret_start,
        RANDOM_THEN_UNICODE.len() - 3,
    );
    assert_eq!(hasher_a.buckets.slice(), hasher_c.buckets.slice());
    assert_eq!(hasher_b.buckets.slice(), hasher_d.buckets.slice());
    assert_eq!(hasher_a.num.slice(), hasher_c.num.slice());
    assert_eq!(hasher_b.num.slice(), hasher_d.num.slice());
    assert_eq!(hasher_a.buckets.slice(), hasher_b.buckets.slice());
    assert_eq!(hasher_c.buckets.slice(), hasher_d.buckets.slice());
    assert_eq!(hasher_a.num.slice(), hasher_b.num.slice());
    assert_eq!(hasher_c.num.slice(), hasher_d.num.slice());
    assert!(hasher_a == hasher_b);
    assert!(hasher_d == hasher_c);
    assert!(hasher_a == hasher_e);
}
