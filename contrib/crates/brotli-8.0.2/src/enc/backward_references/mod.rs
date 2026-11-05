mod benchmark;
pub mod hash_to_binary_tree;
pub mod hq;
mod test;

use core::cmp::{max, min};

use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::command::{BrotliDistanceParams, Command, ComputeDistanceCode};
use super::dictionary_hash::kStaticDictionaryHash;
use super::hash_to_binary_tree::{H10Buckets, H10DefaultParams, ZopfliNode, H10};
use super::static_dict::{
    BrotliDictionary, FindMatchLengthWithLimit, FindMatchLengthWithLimitMin4,
    BROTLI_UNALIGNED_LOAD32, BROTLI_UNALIGNED_LOAD64,
};
use super::util::{floatX, Log2FloorNonZero};
use crate::enc::combined_alloc::allocate;

pub static kInvalidMatch: u32 = 0x0fff_ffff;
static kCutoffTransformsCount: u32 = 10;
static kCutoffTransforms: u64 = 0x071b_520a_da2d_3200;
pub static kHashMul32: u32 = 0x1e35_a7bd;
pub static kHashMul64: u64 = 0x1e35_a7bd_1e35_a7bd;
pub static kHashMul64Long: u64 = 0x1fe3_5a7b_d357_9bd3;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[repr(C)]
pub enum BrotliEncoderMode {
    BROTLI_MODE_GENERIC = 0,
    BROTLI_MODE_TEXT = 1,
    BROTLI_MODE_FONT = 2,
    BROTLI_FORCE_LSB_PRIOR = 3,
    BROTLI_FORCE_MSB_PRIOR = 4,
    BROTLI_FORCE_UTF8_PRIOR = 5,
    BROTLI_FORCE_SIGNED_PRIOR = 6,
}

/// This code takes a length and checks if there's an "end of dictionary" marker at the
///  "ring_buffer_break"point. This marks where a backwards reference cannot pull data through.
/// A match must stop at the end of the dictionary and cannot span the end of the dictionary
/// and beginning of the file.  ring_buffer_break is only set true for custom LZ77 dictionary.
fn fix_unbroken_len(unbroken_len: usize, prev_ix: usize, _cur_ix_masked: usize, ring_buffer_break: Option<core::num::NonZeroUsize>) -> usize
{
    if let Some(br) = ring_buffer_break {
        if prev_ix < usize::from(br) && prev_ix + unbroken_len > usize::from(br)
        {
            return usize::from(br) - prev_ix;
        }
    }
    return unbroken_len;
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrotliHasherParams {
    /// type of hasher to use (default: type 6, but others have tradeoffs of speed/memory)
    pub type_: i32,
    /// number of the number of buckets to have in the hash table (defaults to quality - 1)
    pub bucket_bits: i32,
    /// number of potential matches to hold per bucket (hash collisions)
    pub block_bits: i32,
    /// number of bytes of a potential match to hash
    pub hash_len: i32,
    /// number of previous distance matches to check for future matches (defaults to 16)
    pub num_last_distances_to_check: i32,
    /// how much to weigh distance vs an extra byte of copy match when comparing possible copy srcs
    pub literal_byte_score: i32,
}

#[derive(Clone, Debug)]
pub struct BrotliEncoderParams {
    pub dist: BrotliDistanceParams,
    /// if this brotli file is generic, font or specifically text
    pub mode: BrotliEncoderMode,
    /// quality param between 0 and 11 (11 is smallest but takes longest to encode)
    pub quality: i32,
    pub q9_5: bool,
    /// log of how big the ring buffer should be for copying prior data
    pub lgwin: i32,
    /// log of how often metablocks should be serialized
    pub lgblock: i32,
    /// how big the source file is (or 0 if no hint is provided)
    pub size_hint: usize,
    // FIXME: this should be bool
    /// avoid serializing out priors for literal sections in the favor of decode speed
    pub disable_literal_context_modeling: i32,
    pub hasher: BrotliHasherParams,
    /// produce an IR of the compression file
    pub log_meta_block: bool,
    /// attempt to detect how many bytes before the current byte generates the best prediction of it
    /// * 0 = off (stride 1 always)
    /// * 1 = on per 16th of a file
    /// * 2 = on per block type switch
    pub stride_detection_quality: u8,
    /// if nonzero, will search for high entropy strings and log them differently to the IR
    pub high_entropy_detection_quality: u8,
    /// if nonzero it will search for the temporal locality and effectiveness of the priors
    /// for literals. The best adaptation and forgetfulness will be logged per metablock to the IR
    pub cdf_adaptation_detection: u8,
    /// whether to search for whether the previous byte or the context_map are better predictors on a per-context-map basis
    pub prior_bitmask_detection: u8,
    /// for prior bitmask detection: stride_low, stride_speed, cm_low, cm_speed
    pub literal_adaptation: [(u16, u16); 4],
    pub large_window: bool,
    /// avoid search for the best ndirect vs npostfix parameters for distance
    pub avoid_distance_prefix_search: bool,
    /// construct brotli in such a way that it may be concatenated with another brotli file using appropriate bit ops
    pub catable: bool,
    /// can use the dictionary (default yes unless catable is set)
    pub use_dictionary: bool,
    /// construct brotli in such a way that another concatable brotli file may be appended
    pub appendable: bool,
    /// include a magic number and version number and size_hint at the beginning
    pub magic_number: bool,
    /// prefer to compute the map of previously seen strings
    /// just once for all the threads at the beginning, since they overlap significantly
    pub favor_cpu_efficiency: bool,
}

impl Default for BrotliEncoderParams {
    fn default() -> BrotliEncoderParams {
        super::encode::BrotliEncoderInitParams()
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct H9Opts {
    pub literal_byte_score: u32,
}
pub enum HowPrepared {
    ALREADY_PREPARED,
    NEWLY_PREPARED,
}
#[derive(Clone, PartialEq)]
pub struct Struct1 {
    pub params: BrotliHasherParams,
    /// FIXME: this should be bool
    pub is_prepared_: i32,
    pub dict_num_lookups: usize,
    pub dict_num_matches: usize,
}

fn LiteralSpreeLengthForSparseSearch(params: &BrotliEncoderParams) -> usize {
    (if params.quality < 9 { 64i32 } else { 512i32 }) as usize
}

pub struct HasherSearchResult {
    pub len: usize,
    pub len_x_code: usize,
    pub distance: usize,
    pub score: u64,
}

pub trait CloneWithAlloc<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> {
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self;
}

pub trait AnyHasher {
    fn Opts(&self) -> H9Opts;
    fn GetHasherCommon(&mut self) -> &mut Struct1;
    fn HashBytes(&self, data: &[u8]) -> usize;
    fn HashTypeLength(&self) -> usize;
    fn StoreLookahead(&self) -> usize;
    fn PrepareDistanceCache(&self, distance_cache: &mut [i32]);
    fn FindLongestMatch(
        &mut self,
        dictionary: Option<&BrotliDictionary>,
        dictionary_hash: &[u16],
        data: &[u8],
        ring_buffer_mask: usize,
        ring_buffer_break: Option<core::num::NonZeroUsize>,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        gap: usize,
        max_distance: usize,
        out: &mut HasherSearchResult,
    ) -> bool;
    fn Store(&mut self, data: &[u8], mask: usize, ix: usize);
    fn Store4Vec4(&mut self, data: &[u8], mask: usize, ix: usize) {
        for i in 0..4 {
            self.Store(data, mask, ix + i * 4);
        }
    }
    fn StoreEvenVec4(&mut self, data: &[u8], mask: usize, ix: usize) {
        for i in 0..4 {
            self.Store(data, mask, ix + i * 2);
        }
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize);
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize);
    fn Prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8]) -> HowPrepared;
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    );
}

pub fn StitchToPreviousBlockInternal<T: AnyHasher>(
    handle: &mut T,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
) {
    if num_bytes >= handle.HashTypeLength().wrapping_sub(1) && (position >= 3) {
        handle.Store(ringbuffer, ringbuffer_mask, position.wrapping_sub(3));
        handle.Store(ringbuffer, ringbuffer_mask, position.wrapping_sub(2));
        handle.Store(ringbuffer, ringbuffer_mask, position.wrapping_sub(1));
    }
}

pub fn StoreLookaheadThenStore<T: AnyHasher>(hasher: &mut T, size: usize, dict: &[u8]) {
    let overlap = hasher.StoreLookahead().wrapping_sub(1);
    if size > overlap {
        hasher.BulkStoreRange(dict, usize::MAX, 0, size - overlap);
    }
}

pub trait BasicHashComputer {
    fn HashBytes(&self, data: &[u8]) -> u32;
    fn BUCKET_BITS(&self) -> i32;
    fn USE_DICTIONARY(&self) -> i32;
    fn BUCKET_SWEEP(&self) -> i32;
}
pub struct BasicHasher<Buckets: SliceWrapperMut<u32> + SliceWrapper<u32> + BasicHashComputer> {
    pub GetHasherCommon: Struct1,
    pub buckets_: Buckets,
    pub h9_opts: H9Opts,
}

impl<A: SliceWrapperMut<u32> + SliceWrapper<u32> + BasicHashComputer> PartialEq<BasicHasher<A>>
    for BasicHasher<A>
{
    fn eq(&self, other: &BasicHasher<A>) -> bool {
        self.GetHasherCommon == other.GetHasherCommon
            && self.h9_opts == other.h9_opts
            && self.buckets_.slice() == other.buckets_.slice()
    }
}

impl<T: SliceWrapperMut<u32> + SliceWrapper<u32> + BasicHashComputer> BasicHasher<T> {
    fn StoreRangeOptBasic(
        &mut self,
        data: &[u8],
        mask: usize,
        ix_start: usize,
        ix_end: usize,
    ) -> usize {
        let lookahead = 8;
        if ix_end >= ix_start + lookahead * 2 {
            let chunk_count = (ix_end - ix_start) / 4;
            for chunk_id in 0..chunk_count {
                let i = (ix_start + chunk_id * 4) & mask;
                let word11 = data.split_at(i).1.split_at(11).0;
                let mixed0 = self.HashBytes(word11);
                let mixed1 = self.HashBytes(word11.split_at(1).1);
                let mixed2 = self.HashBytes(word11.split_at(2).1);
                let mixed3 = self.HashBytes(word11.split_at(3).1);
                let off: u32 = (i >> 3).wrapping_rem(self.buckets_.BUCKET_SWEEP() as usize) as u32;
                let offset0: usize = mixed0 + off as usize;
                let offset1: usize = mixed1 + off as usize;
                let offset2: usize = mixed2 + off as usize;
                let offset3: usize = mixed3 + off as usize;
                self.buckets_.slice_mut()[offset0] = i as u32;
                self.buckets_.slice_mut()[offset1] = i as u32 + 1;
                self.buckets_.slice_mut()[offset2] = i as u32 + 2;
                self.buckets_.slice_mut()[offset3] = i as u32 + 3;
            }
            return ix_start + chunk_count * 4;
        }
        ix_start
    }
}
pub struct H2Sub<AllocU32: alloc::Allocator<u32>> {
    pub buckets_: AllocU32::AllocatedMemory, // 65537
}
impl<T: SliceWrapperMut<u32> + SliceWrapper<u32> + BasicHashComputer> AnyHasher for BasicHasher<T> {
    #[inline(always)]
    fn Opts(&self) -> H9Opts {
        self.h9_opts
    }
    #[allow(unused_variables)]
    fn PrepareDistanceCache(&self, distance_cache: &mut [i32]) {}
    #[inline(always)]
    fn HashTypeLength(&self) -> usize {
        8
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        8
    }
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        StitchToPreviousBlockInternal(self, num_bytes, position, ringbuffer, ringbuffer_mask);
    }
    #[inline(always)]
    fn GetHasherCommon(&mut self) -> &mut Struct1 {
        &mut self.GetHasherCommon
    }
    #[inline(always)]
    fn HashBytes(&self, data: &[u8]) -> usize {
        self.buckets_.HashBytes(data) as usize
    }
    fn Store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let (_, data_window) = data.split_at((ix & mask));
        let key: u32 = self.HashBytes(data_window) as u32;
        let off: u32 = (ix >> 3).wrapping_rem(self.buckets_.BUCKET_SWEEP() as usize) as u32;
        self.buckets_.slice_mut()[key.wrapping_add(off) as usize] = ix as u32;
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        for i in self.StoreRangeOptBasic(data, mask, ix_start, ix_end)..ix_end {
            self.Store(data, mask, i);
        }
    }
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        self.StoreRange(data, mask, ix_start, ix_end);
    }
    fn Prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8]) -> HowPrepared {
        if self.GetHasherCommon.is_prepared_ != 0 {
            return HowPrepared::ALREADY_PREPARED;
        }
        let partial_prepare_threshold = (4 << self.buckets_.BUCKET_BITS()) >> 7;
        if one_shot && input_size <= partial_prepare_threshold {
            for i in 0..input_size {
                let key = self.HashBytes(&data[i..]);
                let bs = self.buckets_.BUCKET_SWEEP() as usize;
                for item in self.buckets_.slice_mut()[key..(key + bs)].iter_mut() {
                    *item = 0;
                }
            }
        } else {
            for item in self.buckets_.slice_mut().iter_mut() {
                *item = 0;
            }
        }
        self.GetHasherCommon.is_prepared_ = 1;
        HowPrepared::NEWLY_PREPARED
    }

    fn FindLongestMatch(
        &mut self,
        dictionary: Option<&BrotliDictionary>,
        dictionary_hash: &[u16],
        data: &[u8],
        ring_buffer_mask: usize,
        ring_buffer_break: Option<core::num::NonZeroUsize>,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        gap: usize,
        max_distance: usize,
        out: &mut HasherSearchResult,
    ) -> bool {
        let opts = self.Opts();
        let best_len_in: usize = out.len;
        let cur_ix_masked: usize = cur_ix & ring_buffer_mask;
        let key: u32 = self.HashBytes(&data[cur_ix_masked..]) as u32;
        let mut compare_char: i32 = data[cur_ix_masked.wrapping_add(best_len_in)] as i32;
        let mut best_score: u64 = out.score;
        let mut best_len: usize = best_len_in;
        let cached_backward: usize = distance_cache[0] as usize;
        let mut prev_ix: usize = cur_ix.wrapping_sub(cached_backward);
        let mut is_match_found = false;
        out.len_x_code = 0usize;
        if prev_ix < cur_ix {
            prev_ix &= ring_buffer_mask as u32 as usize;
            if compare_char == data[prev_ix.wrapping_add(best_len)] as i32 {
                let unbroken_len: usize = FindMatchLengthWithLimitMin4(
                    &data[prev_ix..],
                    &data[cur_ix_masked..],
                    max_length,
                );
                if unbroken_len != 0 {
                    let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                    best_score = BackwardReferenceScoreUsingLastDistance(len, opts);
                    best_len = len;
                    out.len = len;
                    out.distance = cached_backward;
                    out.score = best_score;
                    compare_char = data[cur_ix_masked.wrapping_add(best_len)] as i32;
                    if self.buckets_.BUCKET_SWEEP() == 1i32 {
                        self.buckets_.slice_mut()[key as usize] = cur_ix as u32;
                        return true;
                    } else {
                        is_match_found = true;
                    }
                }
            }
        }
        let bucket_sweep = self.buckets_.BUCKET_SWEEP();
        if bucket_sweep == 1i32 {
            prev_ix = self.buckets_.slice()[key as usize] as usize;
            self.buckets_.slice_mut()[key as usize] = cur_ix as u32;
            let backward: usize = cur_ix.wrapping_sub(prev_ix);
            prev_ix &= ring_buffer_mask as u32 as usize;
            if compare_char != data[prev_ix.wrapping_add(best_len_in)] as i32 {
                return false;
            }
            if backward == 0usize || backward > max_backward {
                return false;
            }
            let unbroken_len: usize =
                FindMatchLengthWithLimitMin4(&data[prev_ix..], &data[cur_ix_masked..], max_length);
            if unbroken_len != 0 {
                let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                out.len = len;
                out.distance = backward;
                out.score = BackwardReferenceScore(len, backward, opts);
                return true;
            }
        } else {
            for prev_ix_ref in
                self.buckets_.slice().split_at(key as usize).1[..bucket_sweep as usize].iter()
            {
                let mut prev_ix = *prev_ix_ref as usize;
                let backward: usize = cur_ix.wrapping_sub(prev_ix);
                prev_ix &= ring_buffer_mask as u32 as usize;
                if compare_char != data[prev_ix.wrapping_add(best_len)] as i32 {
                    continue;
                }
                if backward == 0usize || backward > max_backward {
                    continue;
                }
                let unbroken_len = FindMatchLengthWithLimitMin4(
                    &data[prev_ix..],
                    &data[cur_ix_masked..],
                    max_length,
                );
                
                if unbroken_len != 0 {
                    let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                    let score: u64 = BackwardReferenceScore(len, backward, opts);
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = score;
                        compare_char = data[cur_ix_masked.wrapping_add(best_len)] as i32;
                        is_match_found = true;
                    }
                }
            }
        }
        if dictionary.is_some() && self.buckets_.USE_DICTIONARY() != 0 && !is_match_found {
            is_match_found = SearchInStaticDictionary(
                dictionary.unwrap(),
                dictionary_hash,
                self,
                &data[cur_ix_masked..],
                max_length,
                max_backward.wrapping_add(gap),
                max_distance,
                out,
                true,
            );
        }
        self.buckets_.slice_mut()
            [(key as usize).wrapping_add((cur_ix >> 3).wrapping_rem(bucket_sweep as usize))] =
            cur_ix as u32;
        is_match_found
    }
}
impl<AllocU32: alloc::Allocator<u32>> BasicHashComputer for H2Sub<AllocU32> {
    fn HashBytes(&self, data: &[u8]) -> u32 {
        let h: u64 =
            (BROTLI_UNALIGNED_LOAD64(data) << (64i32 - 8i32 * 5i32)).wrapping_mul(kHashMul64);
        (h >> (64i32 - 16i32)) as u32
    }
    fn BUCKET_BITS(&self) -> i32 {
        16
    }
    fn BUCKET_SWEEP(&self) -> i32 {
        1
    }
    fn USE_DICTIONARY(&self) -> i32 {
        1
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapperMut<u32> for H2Sub<AllocU32> {
    fn slice_mut(&mut self) -> &mut [u32] {
        return self.buckets_.slice_mut();
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapper<u32> for H2Sub<AllocU32> {
    fn slice(&self) -> &[u32] {
        return self.buckets_.slice();
    }
}
pub struct H3Sub<AllocU32: alloc::Allocator<u32>> {
    pub buckets_: AllocU32::AllocatedMemory, // 65538
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapperMut<u32> for H3Sub<AllocU32> {
    fn slice_mut(&mut self) -> &mut [u32] {
        return self.buckets_.slice_mut();
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapper<u32> for H3Sub<AllocU32> {
    fn slice(&self) -> &[u32] {
        return self.buckets_.slice();
    }
}
impl<AllocU32: alloc::Allocator<u32>> BasicHashComputer for H3Sub<AllocU32> {
    fn BUCKET_BITS(&self) -> i32 {
        16
    }
    fn BUCKET_SWEEP(&self) -> i32 {
        2
    }
    fn USE_DICTIONARY(&self) -> i32 {
        0
    }
    fn HashBytes(&self, data: &[u8]) -> u32 {
        let h: u64 =
            (BROTLI_UNALIGNED_LOAD64(data) << (64i32 - 8i32 * 5i32)).wrapping_mul(kHashMul64);
        (h >> (64i32 - 16i32)) as u32
    }
}
pub struct H4Sub<AllocU32: alloc::Allocator<u32>> {
    pub buckets_: AllocU32::AllocatedMemory, // 131076
}
impl<AllocU32: alloc::Allocator<u32>> BasicHashComputer for H4Sub<AllocU32> {
    fn BUCKET_BITS(&self) -> i32 {
        17
    }
    fn BUCKET_SWEEP(&self) -> i32 {
        4
    }
    fn USE_DICTIONARY(&self) -> i32 {
        1
    }
    fn HashBytes(&self, data: &[u8]) -> u32 {
        let h: u64 =
            (BROTLI_UNALIGNED_LOAD64(data) << (64i32 - 8i32 * 5i32)).wrapping_mul(kHashMul64);
        (h >> (64i32 - 17i32)) as u32
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapperMut<u32> for H4Sub<AllocU32> {
    fn slice_mut(&mut self) -> &mut [u32] {
        return self.buckets_.slice_mut();
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapper<u32> for H4Sub<AllocU32> {
    fn slice(&self) -> &[u32] {
        return self.buckets_.slice();
    }
}
pub struct H54Sub<AllocU32: alloc::Allocator<u32>> {
    pub buckets_: AllocU32::AllocatedMemory,
}
impl<AllocU32: alloc::Allocator<u32>> BasicHashComputer for H54Sub<AllocU32> {
    fn BUCKET_BITS(&self) -> i32 {
        20
    }
    fn BUCKET_SWEEP(&self) -> i32 {
        4
    }
    fn USE_DICTIONARY(&self) -> i32 {
        0
    }
    fn HashBytes(&self, data: &[u8]) -> u32 {
        let h: u64 =
            (BROTLI_UNALIGNED_LOAD64(data) << (64i32 - 8i32 * 7i32)).wrapping_mul(kHashMul64);
        (h >> (64i32 - 20i32)) as u32
    }
}

impl<AllocU32: alloc::Allocator<u32>> SliceWrapperMut<u32> for H54Sub<AllocU32> {
    fn slice_mut(&mut self) -> &mut [u32] {
        return self.buckets_.slice_mut();
    }
}
impl<AllocU32: alloc::Allocator<u32>> SliceWrapper<u32> for H54Sub<AllocU32> {
    fn slice(&self) -> &[u32] {
        return self.buckets_.slice();
    }
}
pub const H9_BUCKET_BITS: usize = 15;
pub const H9_BLOCK_BITS: usize = 8;
pub const H9_NUM_LAST_DISTANCES_TO_CHECK: usize = 16;
pub const H9_BLOCK_SIZE: usize = 1 << H9_BLOCK_BITS;
const H9_BLOCK_MASK: usize = (1 << H9_BLOCK_BITS) - 1;

impl H9Opts {
    pub fn new(params: &BrotliHasherParams) -> H9Opts {
        H9Opts {
            literal_byte_score: if params.literal_byte_score != 0 {
                params.literal_byte_score as u32
            } else {
                540
            },
        }
    }
}

pub struct H9<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> {
    pub num_: <Alloc as Allocator<u16>>::AllocatedMemory, //[u16;1 << H9_BUCKET_BITS],
    pub buckets_: <Alloc as Allocator<u32>>::AllocatedMemory, //[u32; H9_BLOCK_SIZE << H9_BUCKET_BITS],
    pub dict_search_stats_: Struct1,
    pub h9_opts: H9Opts,
}

impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> PartialEq<H9<Alloc>> for H9<Alloc> {
    fn eq(&self, other: &H9<Alloc>) -> bool {
        self.dict_search_stats_ == other.dict_search_stats_
            && self.num_.slice() == other.num_.slice()
            && self.buckets_.slice() == other.buckets_.slice()
            && self.h9_opts == other.h9_opts
    }
}

fn adv_prepare_distance_cache(distance_cache: &mut [i32], num_distances: i32) {
    if num_distances > 4i32 {
        let last_distance: i32 = distance_cache[0];
        distance_cache[4] = last_distance - 1i32;
        distance_cache[5] = last_distance + 1i32;
        distance_cache[6] = last_distance - 2i32;
        distance_cache[7] = last_distance + 2i32;
        distance_cache[8] = last_distance - 3i32;
        distance_cache[9] = last_distance + 3i32;
        if num_distances > 10i32 {
            let next_last_distance: i32 = distance_cache[1];
            distance_cache[10] = next_last_distance - 1i32;
            distance_cache[11] = next_last_distance + 1i32;
            distance_cache[12] = next_last_distance - 2i32;
            distance_cache[13] = next_last_distance + 2i32;
            distance_cache[14] = next_last_distance - 3i32;
            distance_cache[15] = next_last_distance + 3i32;
        }
    }
}

pub const kDistanceCacheIndex: [u8; 16] = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

pub const kDistanceCacheOffset: [i8; 16] = [0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3];

//const BROTLI_LITERAL_BYTE_SCORE: u64 = 540;
const BROTLI_DISTANCE_BIT_PENALTY: u32 = 120;

// Score must be positive after applying maximal penalty.
const BROTLI_SCORE_BASE: u32 = (BROTLI_DISTANCE_BIT_PENALTY * 8 * 8/* sizeof usize*/);
const kDistanceShortCodeCost: [u32; 16] = [
    /* Repeat last */
    BROTLI_SCORE_BASE + 60,
    /* 2nd, 3rd, 4th last */
    BROTLI_SCORE_BASE - 95,
    BROTLI_SCORE_BASE - 117,
    BROTLI_SCORE_BASE - 127,
    /* Last with offset */
    BROTLI_SCORE_BASE - 93,
    BROTLI_SCORE_BASE - 93,
    BROTLI_SCORE_BASE - 96,
    BROTLI_SCORE_BASE - 96,
    BROTLI_SCORE_BASE - 99,
    BROTLI_SCORE_BASE - 99,
    /* 2nd last with offset */
    BROTLI_SCORE_BASE - 105,
    BROTLI_SCORE_BASE - 105,
    BROTLI_SCORE_BASE - 115,
    BROTLI_SCORE_BASE - 115,
    BROTLI_SCORE_BASE - 125,
    BROTLI_SCORE_BASE - 125,
];

fn BackwardReferenceScoreH9(
    copy_length: usize,
    backward_reference_offset: usize,
    h9_opts: H9Opts,
) -> u64 {
    (u64::from(BROTLI_SCORE_BASE)
        .wrapping_add((h9_opts.literal_byte_score as u64).wrapping_mul(copy_length as u64))
        .wrapping_sub(
            (BROTLI_DISTANCE_BIT_PENALTY as u64)
                .wrapping_mul(Log2FloorNonZero(backward_reference_offset as u64) as u64),
        ))
        >> 2
}

fn BackwardReferenceScoreUsingLastDistanceH9(
    copy_length: usize,
    distance_short_code: usize,
    h9_opts: H9Opts,
) -> u64 {
    ((h9_opts.literal_byte_score as u64)
        .wrapping_mul(copy_length as u64)
        .wrapping_add(u64::from(kDistanceShortCodeCost[distance_short_code])))
        >> 2
}

impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> AnyHasher for H9<Alloc> {
    #[inline(always)]
    fn Opts(&self) -> H9Opts {
        self.h9_opts
    }
    #[inline(always)]
    fn GetHasherCommon(&mut self) -> &mut Struct1 {
        &mut self.dict_search_stats_
    }
    #[inline(always)]
    fn HashBytes(&self, data: &[u8]) -> usize {
        let h: u32 = BROTLI_UNALIGNED_LOAD32(data).wrapping_mul(kHashMul32);
        let thirty_two: usize = 32;
        (h >> (thirty_two.wrapping_sub(H9_BUCKET_BITS))) as usize
    }
    #[inline(always)]
    fn HashTypeLength(&self) -> usize {
        4
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        4
    }
    fn PrepareDistanceCache(&self, distance_cache: &mut [i32]) {
        let num_distances = H9_NUM_LAST_DISTANCES_TO_CHECK as i32;
        adv_prepare_distance_cache(distance_cache, num_distances);
    }
    fn FindLongestMatch(
        &mut self,
        dictionary: Option<&BrotliDictionary>,
        dictionary_hash: &[u16],
        data: &[u8],
        ring_buffer_mask: usize,
        ring_buffer_break: Option<core::num::NonZeroUsize>,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        gap: usize,
        max_distance: usize,
        out: &mut HasherSearchResult,
    ) -> bool {
        let best_len_in: usize = out.len;
        let cur_ix_masked: usize = cur_ix & ring_buffer_mask;
        let mut best_score: u64 = out.score;
        let mut best_len: usize = best_len_in;
        let mut is_match_found = false;
        out.len_x_code = 0usize;
        for i in 0..H9_NUM_LAST_DISTANCES_TO_CHECK {
            let idx = kDistanceCacheIndex[i] as usize;
            let backward =
                (distance_cache[idx] as usize).wrapping_add(kDistanceCacheOffset[i] as usize);
            let mut prev_ix = cur_ix.wrapping_sub(backward);
            if prev_ix >= cur_ix {
                continue;
            }
            if backward > max_backward {
                continue;
            }
            prev_ix &= ring_buffer_mask;
            if cur_ix_masked.wrapping_add(best_len) > ring_buffer_mask
                || prev_ix.wrapping_add(best_len) > ring_buffer_mask
                || data[cur_ix_masked.wrapping_add(best_len)]
                    != data[prev_ix.wrapping_add(best_len)]
            {
                continue;
            }
            {
                let unbroken_len: usize =
                    FindMatchLengthWithLimit(&data[prev_ix..], &data[cur_ix_masked..], max_length);
                if unbroken_len >= 3 || (unbroken_len == 2 && i < 2) {
                    let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                    let score = BackwardReferenceScoreUsingLastDistanceH9(len, i, self.h9_opts);
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = best_score;
                        is_match_found = true;
                    }
                }
            }
        }
        if max_length >= 4 && cur_ix_masked.wrapping_add(best_len) <= ring_buffer_mask {
            let key = self.HashBytes(data.split_at(cur_ix_masked).1);
            let bucket = &mut self
                .buckets_
                .slice_mut()
                .split_at_mut(key << H9_BLOCK_BITS)
                .1
                .split_at_mut(H9_BLOCK_SIZE)
                .0;
            assert!(bucket.len() > H9_BLOCK_MASK);
            assert_eq!(bucket.len(), H9_BLOCK_MASK + 1);
            let self_num_key = &mut self.num_.slice_mut()[key];
            let down = if *self_num_key > H9_BLOCK_SIZE as u16 {
                (*self_num_key as usize) - H9_BLOCK_SIZE
            } else {
                0usize
            };
            let mut i: usize = *self_num_key as usize;
            let mut prev_best_val = data[cur_ix_masked.wrapping_add(best_len)];
            while i > down {
                i -= 1;
                let mut prev_ix = bucket[i & H9_BLOCK_MASK] as usize;
                let backward = cur_ix.wrapping_sub(prev_ix);
                if (backward > max_backward) {
                    break;
                }
                prev_ix &= ring_buffer_mask;
                if (prev_ix.wrapping_add(best_len) > ring_buffer_mask
                    || prev_best_val != data[prev_ix.wrapping_add(best_len)])
                {
                    continue;
                }
                {
                    let unbroken_len = FindMatchLengthWithLimit(
                        data.split_at(prev_ix).1,
                        data.split_at(cur_ix_masked).1,
                        max_length,
                    );
                    if (unbroken_len >= 4) {
                        let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                        /* Comparing for >= 3 does not change the semantics, but just saves
                        for a few unnecessary binary logarithms in backward reference
                        score, since we are not interested in such short matches. */
                        let score = BackwardReferenceScoreH9(len, backward, self.h9_opts);
                        if (best_score < score) {
                            best_score = score;
                            best_len = len;
                            out.len = best_len;
                            out.distance = backward;
                            out.score = best_score;
                            is_match_found = true;
                            if cur_ix_masked.wrapping_add(best_len) > ring_buffer_mask {
                                break;
                            }
                            prev_best_val = data[cur_ix_masked.wrapping_add(best_len)];
                        }
                    }
                }
            }
            bucket[*self_num_key as usize & H9_BLOCK_MASK] = cur_ix as u32;
            *self_num_key = self_num_key.wrapping_add(1);
        }
        if !is_match_found && dictionary.is_some() {
            let (_, cur_data) = data.split_at(cur_ix_masked);
            is_match_found = SearchInStaticDictionary(
                dictionary.unwrap(),
                dictionary_hash,
                self,
                cur_data,
                max_length,
                max_backward.wrapping_add(gap),
                max_distance,
                out,
                false,
            );
        }
        is_match_found
    }

    fn Store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let (_, data_window) = data.split_at((ix & mask));
        let key: u32 = self.HashBytes(data_window) as u32;
        let self_num_key = &mut self.num_.slice_mut()[key as usize];
        let minor_ix: usize = (*self_num_key as usize & H9_BLOCK_MASK);
        self.buckets_.slice_mut()[minor_ix.wrapping_add((key as usize) << H9_BLOCK_BITS)] =
            ix as u32;
        *self_num_key = self_num_key.wrapping_add(1);
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        for i in ix_start..ix_end {
            self.Store(data, mask, i);
        }
    }
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        for i in ix_start..ix_end {
            self.Store(data, mask, i);
        }
    }
    fn Prepare(&mut self, _one_shot: bool, _input_size: usize, _data: &[u8]) -> HowPrepared {
        if self.GetHasherCommon().is_prepared_ != 0 {
            return HowPrepared::ALREADY_PREPARED;
        }
        for item in self.num_.slice_mut().iter_mut() {
            *item = 0;
        }
        self.GetHasherCommon().is_prepared_ = 1;
        HowPrepared::NEWLY_PREPARED
    }
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        StitchToPreviousBlockInternal(self, num_bytes, position, ringbuffer, ringbuffer_mask)
    }
}

pub trait AdvHashSpecialization: PartialEq<Self> {
    fn get_hash_mask(&self) -> u64;
    fn set_hash_mask(&mut self, params_hash_len: i32);
    fn get_k_hash_mul(&self) -> u64;
    fn HashTypeLength(&self) -> usize;
    fn StoreLookahead(&self) -> usize;
    fn load_and_mix_word(&self, data: &[u8]) -> u64;
    fn hash_shift(&self) -> i32;
    fn bucket_size(&self) -> u32;
    fn block_mask(&self) -> u32;
    fn block_size(&self) -> u32;
    fn block_bits(&self) -> i32;
}
pub struct AdvHasher<
    Specialization: AdvHashSpecialization + Sized + Clone,
    Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>,
> {
    pub GetHasherCommon: Struct1,
    pub specialization: Specialization, // contains hash_mask_
    pub num: <Alloc as Allocator<u16>>::AllocatedMemory,
    pub buckets: <Alloc as Allocator<u32>>::AllocatedMemory,
    pub h9_opts: H9Opts,
}

impl<
        Specialization: AdvHashSpecialization + Sized + Clone,
        Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>,
    > PartialEq<AdvHasher<Specialization, Alloc>> for AdvHasher<Specialization, Alloc>
{
    fn eq(&self, other: &Self) -> bool {
        self.GetHasherCommon == other.GetHasherCommon
            && self.specialization == other.specialization
            && self.num.slice() == other.num.slice()
            && self.buckets.slice() == other.buckets.slice()
            && self.h9_opts == other.h9_opts
    }
}

#[derive(Clone, PartialEq)]
pub struct HQ5Sub {}
impl AdvHashSpecialization for HQ5Sub {
    #[inline(always)]
    fn hash_shift(&self) -> i32 {
        32i32 - 14 // 32 - bucket_bits
    }
    #[inline(always)]
    fn bucket_size(&self) -> u32 {
        1 << 14
    }
    #[inline(always)]
    fn block_bits(&self) -> i32 {
        4
    }
    #[inline(always)]
    fn block_size(&self) -> u32 {
        1 << 4
    }
    #[inline(always)]
    fn block_mask(&self) -> u32 {
        (1 << 4) - 1
    }
    #[inline(always)]
    fn get_hash_mask(&self) -> u64 {
        //return 0xffff_ffff_ffff_ffff;
        0xffff_ffff // make it 32 bit
    }
    #[inline(always)]
    fn get_k_hash_mul(&self) -> u64 {
        kHashMul32 as u64
    }
    #[inline(always)]
    fn load_and_mix_word(&self, data: &[u8]) -> u64 {
        (BROTLI_UNALIGNED_LOAD32(data) as u64 * self.get_k_hash_mul()) & self.get_hash_mask()
    }
    #[inline(always)]
    fn set_hash_mask(&mut self, _params_hash_len: i32) {}
    fn HashTypeLength(&self) -> usize {
        4
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        4
    }
}

#[derive(Clone, PartialEq)]
pub struct HQ7Sub {}
impl AdvHashSpecialization for HQ7Sub {
    #[inline(always)]
    fn hash_shift(&self) -> i32 {
        32i32 - 15 // 32 - bucket_bits
    }
    #[inline(always)]
    fn bucket_size(&self) -> u32 {
        1 << 15
    }
    #[inline(always)]
    fn block_bits(&self) -> i32 {
        6
    }
    #[inline(always)]
    fn block_size(&self) -> u32 {
        1 << 6
    }
    #[inline(always)]
    fn block_mask(&self) -> u32 {
        (1 << 6) - 1
    }
    #[inline(always)]
    fn get_hash_mask(&self) -> u64 {
        //return 0xffff_ffff_ffff_ffff;
        0xffff_ffff // make it 32 bit
    }
    #[inline(always)]
    fn get_k_hash_mul(&self) -> u64 {
        kHashMul32 as u64
    }
    #[inline(always)]
    fn load_and_mix_word(&self, data: &[u8]) -> u64 {
        (BROTLI_UNALIGNED_LOAD32(data) as u64 * self.get_k_hash_mul()) & self.get_hash_mask()
    }
    #[inline(always)]
    fn set_hash_mask(&mut self, _params_hash_len: i32) {}
    fn HashTypeLength(&self) -> usize {
        4
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        4
    }
}

#[derive(Clone, PartialEq)]
pub struct H5Sub {
    pub hash_shift_: i32,
    pub bucket_size_: u32,
    pub block_mask_: u32,
    pub block_bits_: i32,
}

impl AdvHashSpecialization for H5Sub {
    #[inline(always)]
    fn hash_shift(&self) -> i32 {
        self.hash_shift_
    }
    fn bucket_size(&self) -> u32 {
        self.bucket_size_
    }
    fn block_bits(&self) -> i32 {
        self.block_bits_
    }
    fn block_size(&self) -> u32 {
        1 << self.block_bits_
    }
    fn block_mask(&self) -> u32 {
        self.block_mask_
    }
    fn get_hash_mask(&self) -> u64 {
        //return 0xffff_ffff_ffff_ffff;
        0xffff_ffff // make it 32 bit
    }
    fn get_k_hash_mul(&self) -> u64 {
        kHashMul32 as u64
    }
    fn load_and_mix_word(&self, data: &[u8]) -> u64 {
        (BROTLI_UNALIGNED_LOAD32(data) as u64 * self.get_k_hash_mul()) & self.get_hash_mask()
    }
    #[allow(unused_variables)]
    fn set_hash_mask(&mut self, params_hash_len: i32) {}
    fn HashTypeLength(&self) -> usize {
        4
    }
    fn StoreLookahead(&self) -> usize {
        4
    }
}

#[derive(Clone, PartialEq)]
pub struct H6Sub {
    pub hash_mask: u64,
    pub hash_shift_: i32,
    pub bucket_size_: u32,
    pub block_mask_: u32,
    pub block_bits_: i32,
}

impl AdvHashSpecialization for H6Sub {
    #[inline(always)]
    fn hash_shift(&self) -> i32 {
        self.hash_shift_
    }
    #[inline(always)]
    fn bucket_size(&self) -> u32 {
        self.bucket_size_
    }
    fn block_bits(&self) -> i32 {
        self.block_bits_
    }
    fn block_size(&self) -> u32 {
        1 << self.block_bits_
    }
    #[inline(always)]
    fn block_mask(&self) -> u32 {
        self.block_mask_
    }
    #[inline(always)]
    fn get_hash_mask(&self) -> u64 {
        self.hash_mask
    }
    #[inline(always)]
    fn set_hash_mask(&mut self, params_hash_len: i32) {
        // FIXME: this assumes params_hash_len is fairly small, or else it may result in a negative shift value
        self.hash_mask = u64::MAX >> (64i32 - 8i32 * params_hash_len);
    }
    #[inline(always)]
    fn get_k_hash_mul(&self) -> u64 {
        kHashMul64Long
    }
    #[inline(always)]
    fn load_and_mix_word(&self, data: &[u8]) -> u64 {
        (BROTLI_UNALIGNED_LOAD64(data) & self.get_hash_mask()).wrapping_mul(self.get_k_hash_mul())
    }
    #[inline(always)]
    fn HashTypeLength(&self) -> usize {
        8
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        8
    }
}

fn BackwardReferencePenaltyUsingLastDistance(distance_short_code: usize) -> u64 {
    // FIXME?: double bitwise AND with the same value?
    (39u64).wrapping_add((0x0001_ca10_u64 >> (distance_short_code & 0x0e) & 0x0e))
}

impl<
        Specialization: AdvHashSpecialization + Clone,
        Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>,
    > AdvHasher<Specialization, Alloc>
{
    // 7 opt
    // returns a new ix_start
    fn StoreRangeOptBatch(
        &mut self,
        data: &[u8],
        mask: usize,
        ix_start: usize,
        ix_end: usize,
    ) -> usize {
        let lookahead = self.specialization.StoreLookahead();
        if ix_end >= ix_start + lookahead * 2 && lookahead == 4 {
            let num = self.num.slice_mut();
            let buckets = self.buckets.slice_mut();
            assert_eq!(num.len(), self.specialization.bucket_size() as usize);
            assert_eq!(
                buckets.len(),
                self.specialization.bucket_size() as usize
                    * self.specialization.block_size() as usize
            );
            let shift = self.specialization.hash_shift();
            let chunk_count = (ix_end - ix_start) / 4;
            for chunk_id in 0..chunk_count {
                let i = (ix_start + chunk_id * 4) & mask;
                let ffffffff = 0xffff_ffff;
                let word = u64::from(data[i])
                    | (u64::from(data[i + 1]) << 8)
                    | (u64::from(data[i + 2]) << 16)
                    | (u64::from(data[i + 3]) << 24)
                    | (u64::from(data[i + 4]) << 32)
                    | (u64::from(data[i + 5]) << 40)
                    | (u64::from(data[i + 6]) << 48);
                let mixed0 = ((((word & ffffffff) * self.specialization.get_k_hash_mul())
                    & self.specialization.get_hash_mask())
                    >> shift) as usize;
                let mixed1 = (((((word >> 8) & ffffffff) * self.specialization.get_k_hash_mul())
                    & self.specialization.get_hash_mask())
                    >> shift) as usize;
                let mixed2 = (((((word >> 16) & ffffffff) * self.specialization.get_k_hash_mul())
                    & self.specialization.get_hash_mask())
                    >> shift) as usize;
                let mixed3 = (((((word >> 24) & ffffffff) * self.specialization.get_k_hash_mul())
                    & self.specialization.get_hash_mask())
                    >> shift) as usize;
                let mut num_ref0 = u32::from(num[mixed0]);
                num[mixed0] = num_ref0.wrapping_add(1) as u16;
                num_ref0 &= self.specialization.block_mask();
                let mut num_ref1 = u32::from(num[mixed1]);
                num[mixed1] = num_ref1.wrapping_add(1) as u16;
                num_ref1 &= self.specialization.block_mask();
                let mut num_ref2 = u32::from(num[mixed2]);
                num[mixed2] = num_ref2.wrapping_add(1) as u16;
                num_ref2 &= self.specialization.block_mask();
                let mut num_ref3 = u32::from(num[mixed3]);
                num[mixed3] = num_ref3.wrapping_add(1) as u16;
                num_ref3 &= self.specialization.block_mask();
                let offset0: usize =
                    (mixed0 << self.specialization.block_bits()) + num_ref0 as usize;
                let offset1: usize =
                    (mixed1 << self.specialization.block_bits()) + num_ref1 as usize;
                let offset2: usize =
                    (mixed2 << self.specialization.block_bits()) + num_ref2 as usize;
                let offset3: usize =
                    (mixed3 << self.specialization.block_bits()) + num_ref3 as usize;
                buckets[offset0] = (i) as u32;
                buckets[offset1] = (i + 1) as u32;
                buckets[offset2] = (i + 2) as u32;
                buckets[offset3] = (i + 3) as u32;
            }
            return ix_start + chunk_count * 4;
        }
        ix_start
    }

    fn BulkStoreRangeOptMemFetch(
        &mut self,
        data: &[u8],
        mask: usize,
        ix_start: usize,
        ix_end: usize,
    ) -> usize {
        const REG_SIZE: usize = 32usize;
        let lookahead = self.specialization.StoreLookahead();
        if mask == usize::MAX && ix_end > ix_start + REG_SIZE && lookahead == 4 {
            const lookahead4: usize = 4;
            assert_eq!(lookahead4, lookahead);
            let mut data64 = [0u8; REG_SIZE + lookahead4 - 1];
            let del = (ix_end - ix_start) / REG_SIZE;
            let num = self.num.slice_mut();
            let buckets = self.buckets.slice_mut();
            assert_eq!(num.len(), self.specialization.bucket_size() as usize);
            assert_eq!(
                buckets.len(),
                self.specialization.bucket_size() as usize
                    * self.specialization.block_size() as usize
            );
            let shift = self.specialization.hash_shift();
            for chunk_id in 0..del {
                let ix_offset = ix_start + chunk_id * REG_SIZE;
                data64[..REG_SIZE + lookahead4 - 1].clone_from_slice(
                    data.split_at(ix_offset)
                        .1
                        .split_at(REG_SIZE + lookahead4 - 1)
                        .0,
                );
                for quad_index in 0..(REG_SIZE >> 2) {
                    let i = quad_index << 2;
                    let ffffffff = 0xffff_ffff;
                    let word = u64::from(data64[i])
                        | (u64::from(data64[i + 1]) << 8)
                        | (u64::from(data64[i + 2]) << 16)
                        | (u64::from(data64[i + 3]) << 24)
                        | (u64::from(data64[i + 4]) << 32)
                        | (u64::from(data64[i + 5]) << 40)
                        | (u64::from(data64[i + 6]) << 48);
                    let mixed0 = ((((word & ffffffff) * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed1 = (((((word >> 8) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed2 = (((((word >> 16) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed3 = (((((word >> 24) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mut num_ref0 = u32::from(num[mixed0]);
                    num[mixed0] = num_ref0.wrapping_add(1) as u16;
                    num_ref0 &= self.specialization.block_mask();
                    let mut num_ref1 = u32::from(num[mixed1]);
                    num[mixed1] = num_ref1.wrapping_add(1) as u16;
                    num_ref1 &= self.specialization.block_mask();
                    let mut num_ref2 = u32::from(num[mixed2]);
                    num[mixed2] = num_ref2.wrapping_add(1) as u16;
                    num_ref2 &= self.specialization.block_mask();
                    let mut num_ref3 = u32::from(num[mixed3]);
                    num[mixed3] = num_ref3.wrapping_add(1) as u16;
                    num_ref3 &= self.specialization.block_mask();
                    let offset0: usize =
                        (mixed0 << self.specialization.block_bits()) + num_ref0 as usize;
                    let offset1: usize =
                        (mixed1 << self.specialization.block_bits()) + num_ref1 as usize;
                    let offset2: usize =
                        (mixed2 << self.specialization.block_bits()) + num_ref2 as usize;
                    let offset3: usize =
                        (mixed3 << self.specialization.block_bits()) + num_ref3 as usize;
                    buckets[offset0] = (ix_offset + i) as u32;
                    buckets[offset1] = (ix_offset + i + 1) as u32;
                    buckets[offset2] = (ix_offset + i + 2) as u32;
                    buckets[offset3] = (ix_offset + i + 3) as u32;
                }
            }
            return ix_start + del * REG_SIZE;
        }
        ix_start
    }

    #[cfg(feature = "benchmark")]
    fn BulkStoreRangeOptMemFetchLazyDupeUpdate(
        &mut self,
        data: &[u8],
        mask: usize,
        ix_start: usize,
        ix_end: usize,
    ) -> usize {
        const REG_SIZE: usize = 32usize;
        let lookahead = self.specialization.StoreLookahead();
        if mask == usize::MAX && ix_end > ix_start + REG_SIZE && lookahead == 4 {
            const lookahead4: usize = 4;
            assert_eq!(lookahead4, lookahead);
            let mut data64 = [0u8; REG_SIZE + lookahead4];
            let del = (ix_end - ix_start) / REG_SIZE;
            let num = self.num.slice_mut();
            let buckets = self.buckets.slice_mut();
            assert_eq!(num.len(), self.specialization.bucket_size() as usize);
            assert_eq!(
                buckets.len(),
                self.specialization.bucket_size() as usize
                    * self.specialization.block_size() as usize
            );
            let shift = self.specialization.hash_shift();
            for chunk_id in 0..del {
                let ix_offset = ix_start + chunk_id * REG_SIZE;
                data64[..REG_SIZE + lookahead4]
                    .clone_from_slice(data.split_at(ix_offset).1.split_at(REG_SIZE + lookahead4).0);
                for quad_index in 0..(REG_SIZE >> 2) {
                    let i = quad_index << 2;
                    let ffffffff = 0xffff_ffff;
                    let word = u64::from(data64[i])
                        | (u64::from(data64[i + 1]) << 8)
                        | (u64::from(data64[i + 2]) << 16)
                        | (u64::from(data64[i + 3]) << 24)
                        | (u64::from(data64[i + 4]) << 32)
                        | (u64::from(data64[i + 5]) << 40)
                        | (u64::from(data64[i + 6]) << 48);
                    let mixed0 = ((((word & ffffffff) * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed1 = (((((word >> 8) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed2 = (((((word >> 16) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mixed3 = (((((word >> 24) & ffffffff)
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask())
                        >> shift) as usize;
                    let mut num_ref0 = u32::from(num[mixed0]);
                    let mut num_ref1 = u32::from(num[mixed1]);
                    let mut num_ref2 = u32::from(num[mixed2]);
                    let mut num_ref3 = u32::from(num[mixed3]);
                    num[mixed0] = num_ref0.wrapping_add(1) as u16;
                    num[mixed1] = num_ref1.wrapping_add(1) as u16;
                    num[mixed2] = num_ref2.wrapping_add(1) as u16;
                    num[mixed3] = num_ref3.wrapping_add(1) as u16;
                    num_ref0 &= self.specialization.block_mask();
                    num_ref1 &= self.specialization.block_mask();
                    num_ref2 &= self.specialization.block_mask();
                    num_ref3 &= self.specialization.block_mask();
                    let offset0: usize =
                        (mixed0 << self.specialization.block_bits()) + num_ref0 as usize;
                    let offset1: usize =
                        (mixed1 << self.specialization.block_bits()) + num_ref1 as usize;
                    let offset2: usize =
                        (mixed2 << self.specialization.block_bits()) + num_ref2 as usize;
                    let offset3: usize =
                        (mixed3 << self.specialization.block_bits()) + num_ref3 as usize;
                    buckets[offset0] = (ix_offset + i) as u32;
                    buckets[offset1] = (ix_offset + i + 1) as u32;
                    buckets[offset2] = (ix_offset + i + 2) as u32;
                    buckets[offset3] = (ix_offset + i + 3) as u32;
                }
            }
            return ix_start + del * REG_SIZE;
        }
        ix_start
    }

    #[cfg(feature = "benchmark")]
    fn BulkStoreRangeOptRandomDupeUpdate(
        &mut self,
        data: &[u8],
        mask: usize,
        ix_start: usize,
        ix_end: usize,
    ) -> usize {
        const REG_SIZE: usize = 32usize;
        let lookahead = self.specialization.StoreLookahead();
        if mask == usize::MAX && ix_end > ix_start + REG_SIZE && lookahead == 4 {
            const lookahead4: usize = 4;
            assert_eq!(lookahead4, lookahead);
            let mut data64 = [0u8; REG_SIZE + lookahead4];
            let del = (ix_end - ix_start) / REG_SIZE;
            let num = self.num.slice_mut();
            let buckets = self.buckets.slice_mut();
            assert_eq!(num.len(), self.specialization.bucket_size() as usize);
            assert_eq!(
                buckets.len(),
                self.specialization.bucket_size() as usize
                    * self.specialization.block_size() as usize
            );
            let shift = self.specialization.hash_shift();
            for chunk_id in 0..del {
                let ix_offset = ix_start + chunk_id * REG_SIZE;
                data64[..REG_SIZE + lookahead4]
                    .clone_from_slice(data.split_at(ix_offset).1.split_at(REG_SIZE + lookahead4).0);
                for i in 0..REG_SIZE {
                    let mixed_word = ((u32::from(data64[i])
                        | (u32::from(data64[i + 1]) << 8)
                        | (u32::from(data64[i + 2]) << 16)
                        | (u32::from(data64[i + 3]) << 24))
                        as u64
                        * self.specialization.get_k_hash_mul())
                        & self.specialization.get_hash_mask();
                    let key = mixed_word >> shift;
                    let minor_ix: usize = chunk_id & self.specialization.block_mask() as usize; //   *num_ref as usize & self.specialization.block_mask() as usize; //GIGANTIC HAX: overwrite firsst option
                    let offset: usize =
                        minor_ix + (key << self.specialization.block_bits()) as usize;
                    buckets[offset] = (ix_offset + i) as u32;
                }
            }
            for (bucket_index, num_ref) in num.iter_mut().enumerate() {
                let region = buckets
                    .split_at_mut(bucket_index << self.specialization.block_bits())
                    .1
                    .split_at_mut(self.specialization.block_size() as usize)
                    .0;
                let mut lnum = 0usize;
                for block_index in 0..self.specialization.block_size() as usize {
                    if region[block_index] != 0 {
                        let byte_addr = region[block_index];
                        region[lnum] = byte_addr;
                        lnum += 1;
                    }
                }
                *num_ref = lnum as u16;
            }
            return ix_start + del * REG_SIZE;
        }
        ix_start
    }
}

impl<
        Specialization: AdvHashSpecialization + Clone,
        Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>,
    > AnyHasher for AdvHasher<Specialization, Alloc>
{
    fn Opts(&self) -> H9Opts {
        self.h9_opts
    }
    fn PrepareDistanceCache(&self, distance_cache: &mut [i32]) {
        let num_distances = self.GetHasherCommon.params.num_last_distances_to_check;
        adv_prepare_distance_cache(distance_cache, num_distances);
    }
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        StitchToPreviousBlockInternal(self, num_bytes, position, ringbuffer, ringbuffer_mask);
    }
    fn Prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8]) -> HowPrepared {
        if self.GetHasherCommon.is_prepared_ != 0 {
            return HowPrepared::ALREADY_PREPARED;
        }
        let partial_prepare_threshold = self.specialization.bucket_size() as usize >> 6;
        if one_shot && input_size <= partial_prepare_threshold {
            for i in 0..input_size {
                let key = self.HashBytes(&data[i..]);
                self.num.slice_mut()[key] = 0;
            }
        } else {
            for item in
                self.num.slice_mut()[..(self.specialization.bucket_size() as usize)].iter_mut()
            {
                *item = 0;
            }
        }
        self.GetHasherCommon.is_prepared_ = 1;
        HowPrepared::NEWLY_PREPARED
    }

    fn GetHasherCommon(&mut self) -> &mut Struct1 {
        &mut self.GetHasherCommon
    }
    fn HashTypeLength(&self) -> usize {
        self.specialization.HashTypeLength()
    }
    fn StoreLookahead(&self) -> usize {
        self.specialization.StoreLookahead()
    }
    fn HashBytes(&self, data: &[u8]) -> usize {
        let shift = self.specialization.hash_shift();
        let h: u64 = self.specialization.load_and_mix_word(data);
        (h >> shift) as u32 as usize
    }
    fn StoreEvenVec4(&mut self, data: &[u8], mask: usize, ix: usize) {
        if self.specialization.StoreLookahead() != 4 {
            for i in 0..4 {
                self.Store(data, mask, ix + i * 2);
            }
            return;
        }
        let shift = self.specialization.hash_shift();
        let num = self.num.slice_mut();
        let buckets = self.buckets.slice_mut();
        let li = ix & mask;
        let lword = u64::from(data[li])
            | (u64::from(data[li + 1]) << 8)
            | (u64::from(data[li + 2]) << 16)
            | (u64::from(data[li + 3]) << 24)
            | (u64::from(data[li + 4]) << 32)
            | (u64::from(data[li + 5]) << 40)
            | (u64::from(data[li + 6]) << 48)
            | (u64::from(data[li + 7]) << 56);
        let hi = (ix + 8) & mask;
        let hword = u64::from(data[hi]) | (u64::from(data[hi + 1]) << 8);
        let mixed0 = ((((lword & 0xffff_ffff) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed1 = (((((lword >> 16) & 0xffff_ffff) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed2 = (((((lword >> 32) & 0xffff_ffff) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed3 = ((((((hword & 0xffff) << 16) | ((lword >> 48) & 0xffff))
            * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mut num_ref0 = u32::from(num[mixed0]);
        num[mixed0] = num_ref0.wrapping_add(1) as u16;
        num_ref0 &= self.specialization.block_mask();
        let mut num_ref1 = u32::from(num[mixed1]);
        num[mixed1] = num_ref1.wrapping_add(1) as u16;
        num_ref1 &= self.specialization.block_mask();
        let mut num_ref2 = u32::from(num[mixed2]);
        num[mixed2] = num_ref2.wrapping_add(1) as u16;
        num_ref2 &= self.specialization.block_mask();
        let mut num_ref3 = u32::from(num[mixed3]);
        num[mixed3] = num_ref3.wrapping_add(1) as u16;
        num_ref3 &= self.specialization.block_mask();
        let offset0: usize = (mixed0 << self.specialization.block_bits()) + num_ref0 as usize;
        let offset1: usize = (mixed1 << self.specialization.block_bits()) + num_ref1 as usize;
        let offset2: usize = (mixed2 << self.specialization.block_bits()) + num_ref2 as usize;
        let offset3: usize = (mixed3 << self.specialization.block_bits()) + num_ref3 as usize;
        buckets[offset0] = ix as u32;
        buckets[offset1] = (ix + 2) as u32;
        buckets[offset2] = (ix + 4) as u32;
        buckets[offset3] = (ix + 6) as u32;
    }
    fn Store4Vec4(&mut self, data: &[u8], mask: usize, ix: usize) {
        if self.specialization.StoreLookahead() != 4 {
            for i in 0..4 {
                self.Store(data, mask, ix + i * 4);
            }
            return;
        }
        let shift = self.specialization.hash_shift();
        let num = self.num.slice_mut();
        let buckets = self.buckets.slice_mut();
        let li = ix & mask;
        let llword = u32::from(data[li])
            | (u32::from(data[li + 1]) << 8)
            | (u32::from(data[li + 2]) << 16)
            | (u32::from(data[li + 3]) << 24);
        let luword = u32::from(data[li + 4])
            | (u32::from(data[li + 5]) << 8)
            | (u32::from(data[li + 6]) << 16)
            | (u32::from(data[li + 7]) << 24);
        let ui = (ix + 8) & mask;
        let ulword = u32::from(data[ui])
            | (u32::from(data[ui + 1]) << 8)
            | (u32::from(data[ui + 2]) << 16)
            | (u32::from(data[ui + 3]) << 24);

        let uuword = u32::from(data[ui + 4])
            | (u32::from(data[ui + 5]) << 8)
            | (u32::from(data[ui + 6]) << 16)
            | (u32::from(data[ui + 7]) << 24);

        let mixed0 = (((u64::from(llword) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed1 = (((u64::from(luword) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed2 = (((u64::from(ulword) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mixed3 = (((u64::from(uuword) * self.specialization.get_k_hash_mul())
            & self.specialization.get_hash_mask())
            >> shift) as usize;
        let mut num_ref0 = u32::from(num[mixed0]);
        num[mixed0] = num_ref0.wrapping_add(1) as u16;
        num_ref0 &= self.specialization.block_mask();
        let mut num_ref1 = u32::from(num[mixed1]);
        num[mixed1] = num_ref1.wrapping_add(1) as u16;
        num_ref1 &= self.specialization.block_mask();
        let mut num_ref2 = u32::from(num[mixed2]);
        num[mixed2] = num_ref2.wrapping_add(1) as u16;
        num_ref2 &= self.specialization.block_mask();
        let mut num_ref3 = u32::from(num[mixed3]);
        num[mixed3] = num_ref3.wrapping_add(1) as u16;
        num_ref3 &= self.specialization.block_mask();
        let offset0: usize = (mixed0 << self.specialization.block_bits()) + num_ref0 as usize;
        let offset1: usize = (mixed1 << self.specialization.block_bits()) + num_ref1 as usize;
        let offset2: usize = (mixed2 << self.specialization.block_bits()) + num_ref2 as usize;
        let offset3: usize = (mixed3 << self.specialization.block_bits()) + num_ref3 as usize;
        buckets[offset0] = ix as u32;
        buckets[offset1] = (ix + 4) as u32;
        buckets[offset2] = (ix + 8) as u32;
        buckets[offset3] = (ix + 12) as u32;
    }
    fn Store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let (_, data_window) = data.split_at((ix & mask));
        let key: u32 = self.HashBytes(data_window) as u32;
        let minor_ix: usize =
            (self.num.slice()[(key as usize)] as u32 & self.specialization.block_mask()) as usize;
        let offset: usize =
            minor_ix.wrapping_add((key << self.specialization.block_bits()) as usize);
        self.buckets.slice_mut()[offset] = ix as u32;
        {
            let _lhs = &mut self.num.slice_mut()[(key as usize)];
            *_lhs = (*_lhs as i32 + 1) as u16;
        }
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        for i in self.StoreRangeOptBatch(data, mask, ix_start, ix_end)..ix_end {
            self.Store(data, mask, i);
        }
    }
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, mut ix_start: usize, ix_end: usize) {
        /*
        if ix_start + 4096 < ix_end {
          for vec_offset in 0..(ix_end - ix_start - 4096) / 16 {
            self.Store4Vec4(data, mask, ix_start + vec_offset * 16);
          }
          ix_start += 16 * ((ix_end - ix_start - 4096) / 16);
        }
        if ix_start + 512 < ix_end {
          for vec_offset in 0..(ix_end - ix_start - 512) / 8 {
            self.StoreEvenVec4(data, mask, ix_start + vec_offset * 8);
            //self.StoreRange(data, mask, ix_start + vec_offset * 8, ix_start + (1+ vec_offset) * 8);
          }
          ix_start += 8 * ((ix_end - ix_start - 512) / 8);
        }
         */
        ix_start = self.BulkStoreRangeOptMemFetch(data, mask, ix_start, ix_end);
        for i in ix_start..ix_end {
            self.Store(data, mask, i);
        }
    }

    fn FindLongestMatch(
        &mut self,
        dictionary: Option<&BrotliDictionary>,
        dictionary_hash: &[u16],
        data: &[u8],
        ring_buffer_mask: usize,
        ring_buffer_break: Option<core::num::NonZeroUsize>,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        gap: usize,
        max_distance: usize,
        out: &mut HasherSearchResult,
    ) -> bool {
        let opts = self.Opts();
        let cur_ix_masked: usize = cur_ix & ring_buffer_mask;
        let mut is_match_found = false;
        let mut best_score: u64 = out.score;
        let mut best_len: usize = out.len;
        out.len = 0usize;
        out.len_x_code = 0usize;
        let cur_data = data.split_at(cur_ix_masked).1;
        for i in 0..self.GetHasherCommon.params.num_last_distances_to_check as usize {
            let backward: usize = distance_cache[i] as usize;
            let mut prev_ix: usize = cur_ix.wrapping_sub(backward);
            if prev_ix >= cur_ix || backward > max_backward {
                continue;
            }
            prev_ix &= ring_buffer_mask;
            if (cur_ix_masked.wrapping_add(best_len) > ring_buffer_mask
                || prev_ix.wrapping_add(best_len) > ring_buffer_mask
                || cur_data[best_len] != data[prev_ix.wrapping_add(best_len)])
            {
                continue;
            }
            let prev_data = data.split_at(prev_ix).1;

            let unbroken_len = FindMatchLengthWithLimit(prev_data, cur_data, max_length);
            if unbroken_len >= 3 || (unbroken_len == 2 && i < 2) {
                let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                let mut score: u64 = BackwardReferenceScoreUsingLastDistance(len, opts);
                if best_score < score {
                    if i != 0 {
                        score = score.wrapping_sub(BackwardReferencePenaltyUsingLastDistance(i));
                    }
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = best_score;
                        is_match_found = true;
                    }
                }
            }
        }

        let key: u32 = self.HashBytes(cur_data) as u32;
        let common_block_bits = self.specialization.block_bits();
        let num_ref_mut = &mut self.num.slice_mut()[key as usize];
        let num_copy = *num_ref_mut;
        let bucket: &mut [u32] = self
            .buckets
            .slice_mut()
            .split_at_mut((key << common_block_bits) as usize)
            .1
            .split_at_mut(self.specialization.block_size() as usize)
            .0;
        assert!(bucket.len() > self.specialization.block_mask() as usize);
        if num_copy != 0 {
            let down: usize = max(
                i32::from(num_copy) - self.specialization.block_size() as i32,
                0,
            ) as usize;
            let mut i = num_copy as usize;
            while i > down {
                i -= 1;
                let mut prev_ix = bucket[i & self.specialization.block_mask() as usize] as usize;
                let backward = cur_ix.wrapping_sub(prev_ix);
                prev_ix &= ring_buffer_mask;
                if (cur_ix_masked.wrapping_add(best_len) > ring_buffer_mask
                    || prev_ix.wrapping_add(best_len) > ring_buffer_mask
                    || cur_data[best_len] != data[prev_ix.wrapping_add(best_len)])
                {
                    if backward > max_backward {
                        break;
                    }
                    continue;
                }
                if backward > max_backward {
                    break;
                }
                let prev_data = data.split_at(prev_ix).1;
                let unbroken_len = FindMatchLengthWithLimitMin4(prev_data, cur_data, max_length);
                if unbroken_len != 0 {
                    let len = fix_unbroken_len(unbroken_len, prev_ix, cur_ix_masked, ring_buffer_break);
                    let score: u64 = BackwardReferenceScore(len, backward, opts);
                    if best_score < score {
                        best_score = score;
                        best_len = len;
                        out.len = best_len;
                        out.distance = backward;
                        out.score = best_score;
                        is_match_found = true;
                    }
                }
            }
        }
        bucket[(num_copy as u32 & self.specialization.block_mask()) as usize] = cur_ix as u32;
        *num_ref_mut = num_ref_mut.wrapping_add(1);

        if !is_match_found && dictionary.is_some() {
            let (_, cur_data) = data.split_at(cur_ix_masked);
            is_match_found = SearchInStaticDictionary(
                dictionary.unwrap(),
                dictionary_hash,
                self,
                cur_data,
                max_length,
                max_backward.wrapping_add(gap),
                max_distance,
                out,
                false,
            );
        }
        is_match_found
    }
}

pub struct BankH40 {
    pub slots: [SlotH40; 65536],
}

pub struct BankH41 {
    pub slots: [SlotH41; 65536],
}

pub struct BankH42 {
    pub slots: [SlotH42; 512],
}

pub struct SlotH40 {
    pub delta: u16,
    pub next: u16,
}
pub struct SlotH41 {
    pub delta: u16,
    pub next: u16,
}

pub struct SlotH42 {
    pub delta: u16,
    pub next: u16,
}

// UNSUPPORTED, for now.
pub struct H40 {
    pub common: Struct1,
    pub addr: [u32; 32768],
    pub head: [u16; 32768],
    pub tiny_hash: [u8; 65536],
    pub banks: [BankH40; 1],
    pub free_slot_idx: [u16; 1],
    pub max_hops: usize,
}

pub struct H41 {
    pub common: Struct1,
    pub addr: [u32; 32768],
    pub head: [u16; 32768],
    pub tiny_hash: [u8; 65536],
    pub banks: [BankH41; 1],
    pub free_slot_idx: [u16; 1],
    pub max_hops: usize,
}

pub struct H42 {
    pub common: Struct1,
    pub addr: [u32; 32768],
    pub head: [u16; 32768],
    pub tiny_hash: [u8; 65536],
    pub banks: [BankH42; 512],
    pub max_hops: usize,
}

fn BackwardReferenceScoreUsingLastDistance(copy_length: usize, h9_opts: H9Opts) -> u64 {
    ((h9_opts.literal_byte_score as u64) >> 2)
        .wrapping_mul(copy_length as u64)
        .wrapping_add((30u64 * 8u64).wrapping_mul(::core::mem::size_of::<u64>() as u64))
        .wrapping_add(15)
}

fn BackwardReferenceScore(
    copy_length: usize,
    backward_reference_offset: usize,
    h9_opts: H9Opts,
) -> u64 {
    (30u64 * 8u64)
        .wrapping_mul(::core::mem::size_of::<u64>() as u64)
        .wrapping_add(((h9_opts.literal_byte_score as usize) >> 2).wrapping_mul(copy_length) as u64)
        .wrapping_sub(
            (30u64).wrapping_mul(Log2FloorNonZero(backward_reference_offset as u64) as u64),
        )
}

fn Hash14(data: &[u8]) -> u32 {
    let h: u32 = BROTLI_UNALIGNED_LOAD32(data).wrapping_mul(kHashMul32);
    h >> (32i32 - 14i32)
}

fn TestStaticDictionaryItem(
    dictionary: &BrotliDictionary,
    item: usize,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    h9_opts: H9Opts,
    out: &mut HasherSearchResult,
) -> i32 {
    let backward: usize;

    let len: usize = item & 0x1fusize;
    let dist: usize = item >> 5;
    let offset: usize =
        (dictionary.offsets_by_length[len] as usize).wrapping_add(len.wrapping_mul(dist));
    if len > max_length {
        return 0i32;
    }
    let matchlen: usize = FindMatchLengthWithLimit(data, &dictionary.data[offset..], len);
    if matchlen.wrapping_add(kCutoffTransformsCount as usize) <= len || matchlen == 0usize {
        return 0i32;
    }
    {
        let cut: u64 = len.wrapping_sub(matchlen) as u64;
        let transform_id: usize =
            (cut << 2).wrapping_add(kCutoffTransforms >> cut.wrapping_mul(6) & 0x3f) as usize;
        backward = max_backward
            .wrapping_add(dist)
            .wrapping_add(1)
            .wrapping_add(transform_id << dictionary.size_bits_by_length[len] as i32);
    }
    if backward > max_distance {
        return 0i32;
    }
    let score: u64 = BackwardReferenceScore(matchlen, backward, h9_opts);
    if score < out.score {
        return 0i32;
    }
    out.len = matchlen;
    out.len_x_code = len ^ matchlen;
    out.distance = backward;
    out.score = score;
    1i32
}

fn SearchInStaticDictionary<HasherType: AnyHasher>(
    dictionary: &BrotliDictionary,
    dictionary_hash: &[u16],
    handle: &mut HasherType,
    data: &[u8],
    max_length: usize,
    max_backward: usize,
    max_distance: usize,
    out: &mut HasherSearchResult,
    shallow: bool,
) -> bool {
    let mut key: usize;
    let mut i: usize;
    let mut is_match_found = false;
    let opts = handle.Opts();
    let xself: &mut Struct1 = handle.GetHasherCommon();
    if xself.dict_num_matches < xself.dict_num_lookups >> 7 {
        return false;
    }
    key = (Hash14(data) << 1) as usize; //FIXME: works for any kind of hasher??
    i = 0usize;
    while i < if shallow { 1 } else { 2 } {
        {
            let item: usize = dictionary_hash[key] as usize;
            xself.dict_num_lookups = xself.dict_num_lookups.wrapping_add(1);
            if item != 0usize {
                let item_matches: i32 = TestStaticDictionaryItem(
                    dictionary,
                    item,
                    data,
                    max_length,
                    max_backward,
                    max_distance,
                    opts,
                    out,
                );
                if item_matches != 0 {
                    xself.dict_num_matches = xself.dict_num_matches.wrapping_add(1);
                    is_match_found = true;
                }
            }
        }
        i = i.wrapping_add(1);
        key = key.wrapping_add(1);
    }
    is_match_found
}

impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc>
    for BasicHasher<H2Sub<Alloc>>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut ret = BasicHasher::<H2Sub<Alloc>> {
            GetHasherCommon: self.GetHasherCommon.clone(),
            buckets_: H2Sub {
                buckets_: allocate::<u32, _>(m, self.buckets_.buckets_.len()),
            },
            h9_opts: self.h9_opts,
        };
        ret.buckets_
            .buckets_
            .slice_mut()
            .clone_from_slice(self.buckets_.buckets_.slice());
        ret
    }
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc>
    for BasicHasher<H3Sub<Alloc>>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut ret = BasicHasher::<H3Sub<Alloc>> {
            GetHasherCommon: self.GetHasherCommon.clone(),
            buckets_: H3Sub::<Alloc> {
                buckets_: allocate::<u32, _>(m, self.buckets_.buckets_.len()),
            },
            h9_opts: self.h9_opts,
        };
        ret.buckets_
            .buckets_
            .slice_mut()
            .clone_from_slice(self.buckets_.buckets_.slice());
        ret
    }
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc>
    for BasicHasher<H4Sub<Alloc>>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut ret = BasicHasher::<H4Sub<Alloc>> {
            GetHasherCommon: self.GetHasherCommon.clone(),
            buckets_: H4Sub::<Alloc> {
                buckets_: allocate::<u32, _>(m, self.buckets_.buckets_.len()),
            },
            h9_opts: self.h9_opts,
        };
        ret.buckets_
            .buckets_
            .slice_mut()
            .clone_from_slice(self.buckets_.buckets_.slice());
        ret
    }
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc>
    for BasicHasher<H54Sub<Alloc>>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut ret = BasicHasher::<H54Sub<Alloc>> {
            GetHasherCommon: self.GetHasherCommon.clone(),
            buckets_: H54Sub::<Alloc> {
                buckets_: allocate::<u32, _>(m, self.buckets_.len()),
            },
            h9_opts: self.h9_opts,
        };
        ret.buckets_
            .buckets_
            .slice_mut()
            .clone_from_slice(self.buckets_.buckets_.slice());
        ret
    }
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc> for H9<Alloc> {
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut num = allocate::<u16, _>(m, self.num_.len());
        num.slice_mut().clone_from_slice(self.num_.slice());
        let mut buckets = allocate::<u32, _>(m, self.buckets_.len());
        buckets.slice_mut().clone_from_slice(self.buckets_.slice());
        H9::<Alloc> {
            num_: num,
            buckets_: buckets,
            dict_search_stats_: self.dict_search_stats_.clone(),
            h9_opts: self.h9_opts,
        }
    }
}
impl<
        Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>,
        Special: AdvHashSpecialization + Sized + Clone,
    > CloneWithAlloc<Alloc> for AdvHasher<Special, Alloc>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut num = allocate::<u16, _>(m, self.num.len());
        num.slice_mut().clone_from_slice(self.num.slice());
        let mut buckets = allocate::<u32, _>(m, self.buckets.len());
        buckets.slice_mut().clone_from_slice(self.buckets.slice());
        AdvHasher::<Special, Alloc> {
            GetHasherCommon: self.GetHasherCommon.clone(),
            specialization: self.specialization.clone(),
            num,
            buckets,
            h9_opts: self.h9_opts,
        }
    }
}

pub enum UnionHasher<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> {
    Uninit,
    H2(BasicHasher<H2Sub<Alloc>>),
    H3(BasicHasher<H3Sub<Alloc>>),
    H4(BasicHasher<H4Sub<Alloc>>),
    H54(BasicHasher<H54Sub<Alloc>>),
    H5(AdvHasher<H5Sub, Alloc>),
    H5q7(AdvHasher<HQ7Sub, Alloc>),
    H5q5(AdvHasher<HQ5Sub, Alloc>),
    H6(AdvHasher<H6Sub, Alloc>),
    H9(H9<Alloc>),
    H10(H10<Alloc, H10Buckets<Alloc>, H10DefaultParams>),
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> PartialEq<UnionHasher<Alloc>>
    for UnionHasher<Alloc>
{
    fn eq(&self, other: &UnionHasher<Alloc>) -> bool {
        match *self {
            UnionHasher::H2(ref hasher) => match *other {
                UnionHasher::H2(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H3(ref hasher) => match *other {
                UnionHasher::H3(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H4(ref hasher) => match *other {
                UnionHasher::H4(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H54(ref hasher) => match *other {
                UnionHasher::H54(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H5(ref hasher) => match *other {
                UnionHasher::H5(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H5q7(ref hasher) => match *other {
                UnionHasher::H5q7(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H5q5(ref hasher) => match *other {
                UnionHasher::H5q5(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H6(ref hasher) => match *other {
                UnionHasher::H6(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H9(ref hasher) => match *other {
                UnionHasher::H9(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::H10(ref hasher) => match *other {
                UnionHasher::H10(ref otherh) => *hasher == *otherh,
                _ => false,
            },
            UnionHasher::Uninit => match *other {
                UnionHasher::Uninit => true,
                _ => false,
            },
        }
    }
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> CloneWithAlloc<Alloc>
    for UnionHasher<Alloc>
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        match *self {
            UnionHasher::H2(ref hasher) => UnionHasher::H2(hasher.clone_with_alloc(m)),
            UnionHasher::H3(ref hasher) => UnionHasher::H3(hasher.clone_with_alloc(m)),
            UnionHasher::H4(ref hasher) => UnionHasher::H4(hasher.clone_with_alloc(m)),
            UnionHasher::H5(ref hasher) => UnionHasher::H5(hasher.clone_with_alloc(m)),
            UnionHasher::H5q7(ref hasher) => UnionHasher::H5q7(hasher.clone_with_alloc(m)),
            UnionHasher::H5q5(ref hasher) => UnionHasher::H5q5(hasher.clone_with_alloc(m)),
            UnionHasher::H6(ref hasher) => UnionHasher::H6(hasher.clone_with_alloc(m)),
            UnionHasher::H54(ref hasher) => UnionHasher::H54(hasher.clone_with_alloc(m)),
            UnionHasher::H9(ref hasher) => UnionHasher::H9(hasher.clone_with_alloc(m)),
            UnionHasher::H10(ref hasher) => UnionHasher::H10(hasher.clone_with_alloc(m)),
            UnionHasher::Uninit => UnionHasher::Uninit,
        }
    }
}
macro_rules! match_all_hashers_mut {
    ($xself : expr, $func_call : ident, $( $args:expr),*) => {
        match $xself {
     &mut UnionHasher::H2(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H3(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H4(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H5(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H5q7(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H5q5(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H6(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H54(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H9(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::H10(ref mut hasher) => hasher.$func_call($($args),*),
     &mut UnionHasher::Uninit => panic!("UNINTIALIZED"),
        }
    };
}
macro_rules! match_all_hashers {
    ($xself : expr, $func_call : ident, $( $args:expr),*) => {
        match $xself {
     &UnionHasher::H2(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H3(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H4(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H5(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H5q7(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H5q5(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H6(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H54(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H9(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::H10(ref hasher) => hasher.$func_call($($args),*),
     &UnionHasher::Uninit => panic!("UNINTIALIZED"),
        }
    };
}
impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> AnyHasher for UnionHasher<Alloc> {
    fn Opts(&self) -> H9Opts {
        match_all_hashers!(self, Opts,)
    }
    fn GetHasherCommon(&mut self) -> &mut Struct1 {
        match_all_hashers_mut!(self, GetHasherCommon,)
    } /*
      fn GetH10Tree(&mut self) -> Option<&mut H10<AllocU32, H10Buckets, H10DefaultParams>> {
        return match_all_hashers_mut!(self, GetH10Tree,);
      }*/
    fn Prepare(&mut self, one_shot: bool, input_size: usize, data: &[u8]) -> HowPrepared {
        match_all_hashers_mut!(self, Prepare, one_shot, input_size, data)
    }
    fn HashBytes(&self, data: &[u8]) -> usize {
        match_all_hashers!(self, HashBytes, data)
    }
    fn HashTypeLength(&self) -> usize {
        match_all_hashers!(self, HashTypeLength,)
    }
    fn StoreLookahead(&self) -> usize {
        match_all_hashers!(self, StoreLookahead,)
    }
    fn PrepareDistanceCache(&self, distance_cache: &mut [i32]) {
        match_all_hashers!(self, PrepareDistanceCache, distance_cache)
    }
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        match_all_hashers_mut!(
            self,
            StitchToPreviousBlock,
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask
        )
    }
    fn FindLongestMatch(
        &mut self,
        dictionary: Option<&BrotliDictionary>,
        dictionary_hash: &[u16],
        data: &[u8],
        ring_buffer_mask: usize,
        ring_buffer_break: Option<core::num::NonZeroUsize>,
        distance_cache: &[i32],
        cur_ix: usize,
        max_length: usize,
        max_backward: usize,
        gap: usize,
        max_distance: usize,
        out: &mut HasherSearchResult,
    ) -> bool {
        match_all_hashers_mut!(
            self,
            FindLongestMatch,
            dictionary,
            dictionary_hash,
            data,
            ring_buffer_mask,
            ring_buffer_break,
            distance_cache,
            cur_ix,
            max_length,
            max_backward,
            gap,
            max_distance,
            out
        )
    }
    fn Store(&mut self, data: &[u8], mask: usize, ix: usize) {
        match_all_hashers_mut!(self, Store, data, mask, ix)
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        match_all_hashers_mut!(self, StoreRange, data, mask, ix_start, ix_end)
    }
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        match_all_hashers_mut!(self, BulkStoreRange, data, mask, ix_start, ix_end)
    }
}

impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> UnionHasher<Alloc> {
    pub fn free(&mut self, alloc: &mut Alloc) {
        match self {
            &mut UnionHasher::H2(ref mut hasher) => {
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::take(&mut hasher.buckets_.buckets_),
                );
            }
            &mut UnionHasher::H3(ref mut hasher) => {
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::take(&mut hasher.buckets_.buckets_),
                );
            }
            &mut UnionHasher::H4(ref mut hasher) => {
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::take(&mut hasher.buckets_.buckets_),
                );
            }
            &mut UnionHasher::H54(ref mut hasher) => {
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::take(&mut hasher.buckets_.buckets_),
                );
            }
            &mut UnionHasher::H5q7(ref mut hasher) => {
                <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut hasher.num));
                <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut hasher.buckets));
            }
            &mut UnionHasher::H5q5(ref mut hasher) => {
                <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut hasher.num));
                <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut hasher.buckets));
            }
            &mut UnionHasher::H5(ref mut hasher) => {
                <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut hasher.num));
                <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut hasher.buckets));
            }
            &mut UnionHasher::H6(ref mut hasher) => {
                <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut hasher.num));
                <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut hasher.buckets));
            }
            &mut UnionHasher::H9(ref mut hasher) => {
                <Alloc as Allocator<u16>>::free_cell(alloc, core::mem::take(&mut hasher.num_));
                <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut hasher.buckets_));
            }
            &mut UnionHasher::H10(ref mut hasher) => {
                hasher.free(alloc);
            }
            &mut UnionHasher::Uninit => {}
        }
        *self = UnionHasher::<Alloc>::default();
    }
}

impl<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>> Default for UnionHasher<Alloc> {
    fn default() -> Self {
        UnionHasher::Uninit
    }
}

/*UnionHasher::H2(BasicHasher {
GetHasherCommon:Struct1{params:BrotliHasherParams{
 type_:2,
 block_bits: 8,
 bucket_bits:16,
 hash_len: 4,
 num_last_distances_to_check:0},
is_prepared_:0,
dict_num_lookups:0,
dict_num_matches:0,
},
buckets_:H2Sub{
buckets_:[0;65537],
},
})
*/
fn CreateBackwardReferences<AH: AnyHasher>(
    dictionary: Option<&BrotliDictionary>,
    dictionary_hash: &[u16],
    num_bytes: usize,
    mut position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    hasher: &mut AH,
    dist_cache: &mut [i32],
    last_insert_len: &mut usize,
    mut commands: &mut [Command],
    num_commands: &mut usize,
    num_literals: &mut usize,
) {
    let gap = 0usize;
    let max_backward_limit: usize = (1usize << params.lgwin).wrapping_sub(16);
    let mut new_commands_count: usize = 0;
    let mut insert_length: usize = *last_insert_len;
    let pos_end: usize = position.wrapping_add(num_bytes);
    let store_end: usize = if num_bytes >= hasher.StoreLookahead() {
        position
            .wrapping_add(num_bytes)
            .wrapping_sub(hasher.StoreLookahead())
            .wrapping_add(1)
    } else {
        position
    };
    let random_heuristics_window_size: usize = LiteralSpreeLengthForSparseSearch(params);
    let mut apply_random_heuristics: usize = position.wrapping_add(random_heuristics_window_size);
    let kMinScore: u64 = (30u64 * 8)
        .wrapping_mul(::core::mem::size_of::<u64>() as u64)
        .wrapping_add(100);
    hasher.PrepareDistanceCache(dist_cache);
    while position.wrapping_add(hasher.HashTypeLength()) < pos_end {
        let mut max_length: usize = pos_end.wrapping_sub(position);
        let mut max_distance: usize = min(position, max_backward_limit);
        let mut sr = HasherSearchResult {
            len: 0,
            len_x_code: 0,
            distance: 0,
            score: 0,
        };
        sr.len = 0usize;
        sr.len_x_code = 0usize;
        sr.distance = 0usize;
        sr.score = kMinScore;
        if hasher.FindLongestMatch(
            dictionary,
            dictionary_hash,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            dist_cache,
            position,
            max_length,
            max_distance,
            gap,
            params.dist.max_distance,
            &mut sr,
        ) {
            let mut delayed_backward_references_in_row: i32 = 0i32;
            max_length = max_length.wrapping_sub(1);
            'break6: loop {
                'continue7: loop {
                    let cost_diff_lazy: u64 = 175;

                    let mut sr2 = HasherSearchResult {
                        len: 0,
                        len_x_code: 0,
                        distance: 0,
                        score: 0,
                    };
                    sr2.len = if params.quality < 5 {
                        min(sr.len.wrapping_sub(1), max_length)
                    } else {
                        0usize
                    };
                    sr2.len_x_code = 0usize;
                    sr2.distance = 0usize;
                    sr2.score = kMinScore;
                    max_distance = min(position.wrapping_add(1), max_backward_limit);
                    let is_match_found: bool = hasher.FindLongestMatch(
                        dictionary,
                        dictionary_hash,
                        ringbuffer,
                        ringbuffer_mask,
                        ringbuffer_break,
                        dist_cache,
                        position.wrapping_add(1),
                        max_length,
                        max_distance,
                        gap,
                        params.dist.max_distance,
                        &mut sr2,
                    );
                    if is_match_found && (sr2.score >= sr.score.wrapping_add(cost_diff_lazy)) {
                        position = position.wrapping_add(1);
                        insert_length = insert_length.wrapping_add(1);
                        sr = sr2;
                        if {
                            delayed_backward_references_in_row += 1;
                            delayed_backward_references_in_row
                        } < 4i32
                            && (position.wrapping_add(hasher.HashTypeLength()) < pos_end)
                        {
                            break 'continue7;
                        }
                    }
                    break 'break6;
                }
                max_length = max_length.wrapping_sub(1);
            }
            apply_random_heuristics = position
                .wrapping_add((2usize).wrapping_mul(sr.len))
                .wrapping_add(random_heuristics_window_size);
            max_distance = min(position, max_backward_limit);
            {
                let distance_code: usize =
                    ComputeDistanceCode(sr.distance, max_distance, dist_cache);
                if sr.distance <= max_distance && (distance_code > 0usize) {
                    dist_cache[3] = dist_cache[2];
                    dist_cache[2] = dist_cache[1];
                    dist_cache[1] = dist_cache[0];
                    dist_cache[0] = sr.distance as i32;
                    hasher.PrepareDistanceCache(dist_cache);
                }
                new_commands_count += 1;

                let (old, new_commands) = core::mem::take(&mut commands).split_at_mut(1);
                commands = new_commands;
                old[0].init(
                    &params.dist,
                    insert_length,
                    sr.len,
                    sr.len ^ sr.len_x_code,
                    distance_code,
                );
            }
            *num_literals = num_literals.wrapping_add(insert_length);
            insert_length = 0usize;
            hasher.StoreRange(
                ringbuffer,
                ringbuffer_mask,
                position.wrapping_add(2),
                min(position.wrapping_add(sr.len), store_end),
            );
            position = position.wrapping_add(sr.len);
        } else {
            insert_length = insert_length.wrapping_add(1);
            position = position.wrapping_add(1);

            if position > apply_random_heuristics {
                let kMargin: usize = max(hasher.StoreLookahead().wrapping_sub(1), 4);
                if position.wrapping_add(16) >= pos_end.wrapping_sub(kMargin) {
                    insert_length = insert_length.wrapping_add(pos_end - position);
                    position = pos_end;
                } else if position
                    > apply_random_heuristics
                        .wrapping_add((4usize).wrapping_mul(random_heuristics_window_size))
                {
                    hasher.Store4Vec4(ringbuffer, ringbuffer_mask, position);
                    insert_length = insert_length.wrapping_add(16);
                    position = position.wrapping_add(16);
                } else {
                    hasher.StoreEvenVec4(ringbuffer, ringbuffer_mask, position);
                    insert_length = insert_length.wrapping_add(8);
                    position = position.wrapping_add(8);
                }
            }
        }
    }
    insert_length = insert_length.wrapping_add(pos_end.wrapping_sub(position));
    *last_insert_len = insert_length;
    *num_commands = num_commands.wrapping_add(new_commands_count);
}
pub fn BrotliCreateBackwardReferences<
    Alloc: alloc::Allocator<u16>
        + alloc::Allocator<u32>
        + alloc::Allocator<u64>
        + alloc::Allocator<floatX>
        + alloc::Allocator<ZopfliNode>,
>(
    alloc: &mut Alloc,
    dictionary: &BrotliDictionary,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>
,
    params: &BrotliEncoderParams,
    hasher_union: &mut UnionHasher<Alloc>,
    dist_cache: &mut [i32],
    last_insert_len: &mut usize,
    commands: &mut [Command],
    num_commands: &mut usize,
    num_literals: &mut usize,
) {
    match (hasher_union) {
        &mut UnionHasher::Uninit => panic!("working with uninitialized hash map"),
        &mut UnionHasher::H10(ref mut hasher) => {
            if params.quality >= 11 {
                super::backward_references_hq::BrotliCreateHqZopfliBackwardReferences(
                    alloc,
                    if params.use_dictionary {
                        Some(dictionary)
                    } else {
                        None
                    },
                    num_bytes,
                    position,
                    ringbuffer,
                    ringbuffer_mask,
                    ringbuffer_break,
                    params,
                    hasher,
                    dist_cache,
                    last_insert_len,
                    commands,
                    num_commands,
                    num_literals,
                )
            } else {
                super::backward_references_hq::BrotliCreateZopfliBackwardReferences(
                    alloc,
                    if params.use_dictionary {
                        Some(dictionary)
                    } else {
                        None
                    },
                    num_bytes,
                    position,
                    ringbuffer,
                    ringbuffer_mask,
                    ringbuffer_break,
                    params,
                    hasher,
                    dist_cache,
                    last_insert_len,
                    commands,
                    num_commands,
                    num_literals,
                )
            }
        }
        &mut UnionHasher::H2(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H3(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H4(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H5(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H5q7(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H5q5(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H6(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H9(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
        &mut UnionHasher::H54(ref mut hasher) => CreateBackwardReferences(
            if params.use_dictionary {
                Some(dictionary)
            } else {
                None
            },
            &kStaticDictionaryHash[..],
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            hasher,
            dist_cache,
            last_insert_len,
            commands,
            num_commands,
            num_literals,
        ),
    }
}
