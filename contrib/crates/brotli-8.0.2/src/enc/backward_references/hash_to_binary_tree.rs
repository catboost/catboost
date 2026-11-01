use alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use core;
use core::cmp::min;

use super::{
    kHashMul32, AnyHasher, BrotliEncoderParams, CloneWithAlloc, H9Opts, HasherSearchResult,
    HowPrepared, Struct1, fix_unbroken_len,
};
use crate::enc::combined_alloc::allocate;
use crate::enc::static_dict::{
    BrotliDictionary, FindMatchLengthWithLimit, BROTLI_UNALIGNED_LOAD32,
};
use crate::enc::util::floatX;

pub const kInfinity: floatX = 1.7e38;

#[derive(Clone, Copy, Debug)]
pub enum Union1 {
    cost(floatX),
    next(u32),
    shortcut(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct ZopfliNode {
    //highest 7 bit is used to reconstruct the length code
    pub length: u32,
    // distance associated with the length
    pub distance: u32,
    // number of literal inserts before the copy; highest 5 bits contain distance short code + 1 (or zero if no short code)
    pub dcode_insert_length: u32,
    pub u: Union1,
}
impl Default for ZopfliNode {
    fn default() -> Self {
        ZopfliNode {
            length: 1,
            distance: 0,
            dcode_insert_length: 0,
            u: Union1::cost(kInfinity),
        }
    }
}

pub trait Allocable<T: Copy, AllocT: Allocator<T>> {
    fn new(m: &mut AllocT, init: T) -> Self;
    fn new_uninit(m: &mut AllocT) -> Self;
    fn free(&mut self, m: &mut AllocT);
}
pub trait H10Params {
    fn max_tree_search_depth() -> u32;
    fn max_tree_comp_length() -> u32;
}

pub struct H10DefaultParams {}
impl H10Params for H10DefaultParams {
    #[inline(always)]
    fn max_tree_search_depth() -> u32 {
        64
    }
    #[inline(always)]
    fn max_tree_comp_length() -> u32 {
        128
    }
}

const BUCKET_BITS: usize = 17;

pub struct H10Buckets<AllocU32: Allocator<u32>>(AllocU32::AllocatedMemory);

impl<AllocU32: Allocator<u32>> Allocable<u32, AllocU32> for H10Buckets<AllocU32> {
    fn new(m: &mut AllocU32, initializer: u32) -> H10Buckets<AllocU32> {
        let mut ret = m.alloc_cell(1 << BUCKET_BITS);
        for item in ret.slice_mut().iter_mut() {
            *item = initializer;
        }
        H10Buckets::<AllocU32>(ret)
    }
    fn new_uninit(m: &mut AllocU32) -> H10Buckets<AllocU32> {
        H10Buckets::<AllocU32>(m.alloc_cell(1 << BUCKET_BITS))
    }
    fn free(&mut self, m: &mut AllocU32) {
        m.free_cell(core::mem::take(&mut self.0));
    }
}

impl<AllocU32: Allocator<u32>> PartialEq<H10Buckets<AllocU32>> for H10Buckets<AllocU32> {
    fn eq(&self, other: &H10Buckets<AllocU32>) -> bool {
        return self.0.slice() == other.0.slice();
    }
}

impl<AllocU32: Allocator<u32>> SliceWrapper<u32> for H10Buckets<AllocU32> {
    #[inline(always)]
    fn slice(&self) -> &[u32] {
        self.0.slice()
    }
}
impl<AllocU32: Allocator<u32>> SliceWrapperMut<u32> for H10Buckets<AllocU32> {
    #[inline(always)]
    fn slice_mut(&mut self) -> &mut [u32] {
        self.0.slice_mut()
    }
}

pub struct H10<
    AllocU32: Allocator<u32>,
    Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
> where
    Buckets: PartialEq<Buckets>,
{
    pub window_mask_: usize,
    pub ringbuffer_break: Option<core::num::NonZeroUsize>,
    pub common: Struct1,
    pub buckets_: Buckets,
    pub invalid_pos_: u32,
    pub forest: AllocU32::AllocatedMemory,
    pub _params: core::marker::PhantomData<Params>,
}

impl<
        AllocU32: Allocator<u32>,
        Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
        Params: H10Params,
    > PartialEq<H10<AllocU32, Buckets, Params>> for H10<AllocU32, Buckets, Params>
where
    Buckets: PartialEq<Buckets>,
{
    fn eq(&self, other: &H10<AllocU32, Buckets, Params>) -> bool {
        self.window_mask_ == other.window_mask_
            && self.common == other.common
            && self.buckets_ == other.buckets_
            && self.invalid_pos_ == other.invalid_pos_
            && self.forest.slice() == other.forest.slice()
            && self._params == other._params
            && self.ringbuffer_break == other.ringbuffer_break
    }
}

pub fn InitializeH10<AllocU32: Allocator<u32>>(
    m32: &mut AllocU32,
    one_shot: bool,
    params: &BrotliEncoderParams,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    input_size: usize,
) -> H10<AllocU32, H10Buckets<AllocU32>, H10DefaultParams> {
    initialize_h10::<AllocU32, H10Buckets<AllocU32>>(m32, one_shot, params, input_size, ringbuffer_break)
}
fn initialize_h10<
    AllocU32: Allocator<u32>,
    Buckets: SliceWrapperMut<u32> + SliceWrapper<u32> + Allocable<u32, AllocU32>,
>(
    m32: &mut AllocU32,
    one_shot: bool,
    params: &BrotliEncoderParams,
    input_size: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
) -> H10<AllocU32, Buckets, H10DefaultParams>
where
    Buckets: PartialEq<Buckets>,
{
    let mut num_nodes = 1 << params.lgwin;
    if one_shot && input_size < num_nodes {
        num_nodes = input_size;
    }
    let window_mask = (1 << params.lgwin) - 1;
    let invalid_pos = 0u32.wrapping_sub(window_mask);
    let buckets = <Buckets as Allocable<u32, AllocU32>>::new(m32, invalid_pos);
    H10::<AllocU32, Buckets, H10DefaultParams> {
        common: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        _params: core::marker::PhantomData::<H10DefaultParams>,
        window_mask_: window_mask as usize,
        invalid_pos_: invalid_pos,
        buckets_: buckets,
        forest: m32.alloc_cell(num_nodes * 2),
        ringbuffer_break,
    }
}

impl<
        AllocU32: Allocator<u32>,
        Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
        Params: H10Params,
    > H10<AllocU32, Buckets, Params>
where
    Buckets: PartialEq<Buckets>,
{
    pub fn free(&mut self, m32: &mut AllocU32) {
        m32.free_cell(core::mem::take(&mut self.forest));
        self.buckets_.free(m32);
    }
}
impl<
        Alloc: Allocator<u16> + Allocator<u32>,
        Buckets: Allocable<u32, Alloc> + SliceWrapperMut<u32> + SliceWrapper<u32>,
        Params: H10Params,
    > CloneWithAlloc<Alloc> for H10<Alloc, Buckets, Params>
where
    Buckets: PartialEq<Buckets>,
{
    fn clone_with_alloc(&self, m: &mut Alloc) -> Self {
        let mut ret = H10::<Alloc, Buckets, Params> {
            window_mask_: self.window_mask_,
            common: self.common.clone(),
            buckets_: Buckets::new_uninit(m),
            invalid_pos_: self.invalid_pos_,
            forest: allocate::<u32, _>(m, self.forest.len()),
            _params: core::marker::PhantomData::<Params>,
            ringbuffer_break: self.ringbuffer_break,
        };
        ret.buckets_
            .slice_mut()
            .clone_from_slice(self.buckets_.slice());
        ret.forest.slice_mut().clone_from_slice(self.forest.slice());
        ret
    }
}

impl<
        AllocU32: Allocator<u32>,
        Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
        Params: H10Params,
    > AnyHasher for H10<AllocU32, Buckets, Params>
where
    Buckets: PartialEq<Buckets>,
{
    /*  fn GetH10Tree(&mut self) -> Option<&mut H10<AllocU32, Buckets, H10Params>> {
      Some(self)
    }*/
    #[inline(always)]
    fn Opts(&self) -> H9Opts {
        H9Opts {
            literal_byte_score: 340,
        }
    }
    #[inline(always)]
    fn PrepareDistanceCache(&self, _distance_cache: &mut [i32]) {}
    #[inline(always)]
    fn HashTypeLength(&self) -> usize {
        4
    }
    #[inline(always)]
    fn StoreLookahead(&self) -> usize {
        Params::max_tree_comp_length() as usize
    }
    fn StitchToPreviousBlock(
        &mut self,
        num_bytes: usize,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        super::hq::StitchToPreviousBlockH10(self, num_bytes, position, ringbuffer, ringbuffer_mask, self.ringbuffer_break)
    }
    #[inline(always)]
    fn GetHasherCommon(&mut self) -> &mut Struct1 {
        &mut self.common
    }
    #[inline(always)]
    fn HashBytes(&self, data: &[u8]) -> usize {
        let h = BROTLI_UNALIGNED_LOAD32(data).wrapping_mul(kHashMul32);
        (h >> (32i32 - BUCKET_BITS as i32)) as usize
    }
    #[inline(always)]
    fn Store(&mut self, data: &[u8], mask: usize, ix: usize) {
        let max_backward: usize = self.window_mask_.wrapping_sub(16).wrapping_add(1);
        StoreAndFindMatchesH10(
            self,
            data,
            ix,
            mask,
            self.ringbuffer_break,
            Params::max_tree_comp_length() as usize,
            max_backward,
            &mut 0,
            &mut [],
        );
    }
    fn StoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        let mut i: usize = ix_start;
        let mut j: usize = ix_start;
        if ix_start.wrapping_add(63) <= ix_end {
            i = ix_end.wrapping_sub(63);
        }
        if ix_start.wrapping_add(512) <= i {
            while j < i {
                {
                    self.Store(data, mask, j);
                }
                j = j.wrapping_add(8);
            }
        }
        while i < ix_end {
            {
                self.Store(data, mask, i);
            }
            i = i.wrapping_add(1);
        }
    }
    fn BulkStoreRange(&mut self, data: &[u8], mask: usize, ix_start: usize, ix_end: usize) {
        for i in ix_start..ix_end {
            self.Store(data, mask, i);
        }
    }
    fn Prepare(&mut self, _one_shot: bool, _input_size: usize, _data: &[u8]) -> HowPrepared {
        if self.common.is_prepared_ != 0 {
            return HowPrepared::ALREADY_PREPARED;
        }
        let invalid_pos = self.invalid_pos_;
        for bucket in self.buckets_.slice_mut().iter_mut() {
            *bucket = invalid_pos;
        }
        self.common.is_prepared_ = 1;
        HowPrepared::NEWLY_PREPARED
    }

    fn FindLongestMatch(
        &mut self,
        _dictionary: Option<&BrotliDictionary>,
        _dictionary_hash: &[u16],
        _data: &[u8],
        _ring_buffer_mask: usize,
        _ring_buffer_break: Option<core::num::NonZeroUsize>,
        _distance_cache: &[i32],
        _cur_ix: usize,
        _max_length: usize,
        _max_backward: usize,
        _gap: usize,
        _max_distance: usize,
        _out: &mut HasherSearchResult,
    ) -> bool {
        unimplemented!();
    }
}

pub struct BackwardMatch(pub u64);

//    pub distance : u32,
//    pub length_and_code : u32,
impl BackwardMatch {
    #[inline(always)]
    pub fn distance(&self) -> u32 {
        self.0 as u32
    }
    #[inline(always)]
    pub fn length_and_code(&self) -> u32 {
        (self.0 >> 32) as u32
    }
}
pub struct BackwardMatchMut<'a>(pub &'a mut u64);

//    pub distance : u32,
//    pub length_and_code : u32,
impl<'a> BackwardMatchMut<'a> {
    #[inline(always)]
    pub fn distance(&self) -> u32 {
        *self.0 as u32
    }
    #[inline(always)]
    pub fn length_and_code(&self) -> u32 {
        (*self.0 >> 32) as u32
    }
    #[inline(always)]
    pub fn set_distance(&mut self, data: u32) {
        *self.0 &= 0xffffffff00000000;
        *self.0 |= u64::from(data)
    }
    #[inline(always)]
    pub fn set_length_and_code(&mut self, data: u32) {
        *self.0 = u64::from((*self.0) as u32) | (u64::from(data) << 32);
    }
    #[inline(always)]
    pub fn init(&mut self, dist: usize, len: usize) {
        self.set_distance(dist as u32);
        self.set_length_and_code((len << 5) as u32);
    }
    #[inline(always)]
    pub(crate) fn init_dictionary(&mut self, dist: usize, len: usize, len_code: usize) {
        self.set_distance(dist as u32);
        self.set_length_and_code((len << 5 | if len == len_code { 0 } else { len_code }) as u32);
    }
}

macro_rules! LeftChildIndexH10 {
    ($xself: expr, $pos: expr) => {
        (2usize).wrapping_mul($pos & (*$xself).window_mask_)
    };
}
macro_rules! RightChildIndexH10 {
    ($xself: expr, $pos: expr) => {
        (2usize)
            .wrapping_mul($pos & (*$xself).window_mask_)
            .wrapping_add(1)
    };
}
/*
fn LeftChildIndexH10<AllocU32: Allocator<u32>,
     Buckets: Allocable<u32, AllocU32>+SliceWrapperMut<u32>+SliceWrapper<u32>,
     Params:H10Params>(
    mut xself : &mut H10<AllocU32, Buckets, Params>, pos : usize
) -> usize {
    (2usize).wrapping_mul(pos & xself.window_mask_)
}

fn RightChildIndexH10<AllocU32: Allocator<u32>,
     Buckets: Allocable<u32, AllocU32>+SliceWrapperMut<u32>+SliceWrapper<u32>,
     Params:H10Params>(
    mut xself : &mut H10<AllocU32, Buckets, Params>, pos : usize
) -> usize {
    (2usize).wrapping_mul(
        pos & xself.window_mask_
    ).wrapping_add(
        1
    )
}
*/

pub fn StoreAndFindMatchesH10<
    AllocU32: Allocator<u32>,
    Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
>(
    xself: &mut H10<AllocU32, Buckets, Params>,
    data: &[u8],
    cur_ix: usize,
    ring_buffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    max_length: usize,
    max_backward: usize,
    best_len: &mut usize,
    matches: &mut [u64],
) -> usize
where
    Buckets: PartialEq<Buckets>,
{
    let mut matches_offset = 0_usize;
    let cur_ix_masked = cur_ix & ring_buffer_mask;
    let max_comp_len = min(max_length, 128);
    let should_reroot_tree = max_length >= 128;
    let key = xself.HashBytes(&data[cur_ix_masked..]);
    let forest = xself.forest.slice_mut();
    let mut prev_ix = xself.buckets_.slice()[key] as usize;
    let mut node_left = LeftChildIndexH10!(xself, cur_ix);
    let mut node_right = RightChildIndexH10!(xself, cur_ix);
    let mut best_len_left = 0_usize;
    let mut best_len_right = 0_usize;
    let mut depth_remaining = 64_usize;

    if should_reroot_tree {
        xself.buckets_.slice_mut()[key] = cur_ix as u32;
    }

    loop {
        let backward = cur_ix.wrapping_sub(prev_ix);
        let prev_ix_masked = prev_ix & ring_buffer_mask;
        if backward == 0 || backward > max_backward || depth_remaining == 0 {
            if should_reroot_tree {
                forest[node_left] = xself.invalid_pos_;
                forest[node_right] = xself.invalid_pos_;
            }
            break;
        }

        let cur_len = min(best_len_left, best_len_right);

        let len = fix_unbroken_len(
            cur_len.wrapping_add(FindMatchLengthWithLimit(
                &data[cur_ix_masked.wrapping_add(cur_len)..],
                &data[prev_ix_masked.wrapping_add(cur_len)..],
                max_length.wrapping_sub(cur_len),
            )), prev_ix_masked, cur_ix_masked, ringbuffer_break);

        if matches_offset != matches.len() && len > *best_len {
            *best_len = len;
            BackwardMatchMut(&mut matches[matches_offset]).init(backward, len);
            matches_offset += 1;
        }

        if len >= max_comp_len {
            if should_reroot_tree {
                forest[node_left] = forest[LeftChildIndexH10!(xself, prev_ix)];
                forest[node_right] = forest[RightChildIndexH10!(xself, prev_ix)];
            }
            break;
        }

        if data[cur_ix_masked.wrapping_add(len)] > data[prev_ix_masked.wrapping_add(len)] {
            best_len_left = len;
            if should_reroot_tree {
                forest[node_left] = prev_ix as u32;
            }
            node_left = RightChildIndexH10!(xself, prev_ix);
            prev_ix = forest[node_left] as usize;
        } else {
            best_len_right = len;
            if should_reroot_tree {
                forest[node_right] = prev_ix as u32;
            }
            node_right = LeftChildIndexH10!(xself, prev_ix);
            prev_ix = forest[node_right] as usize;
        }

        depth_remaining = depth_remaining.wrapping_sub(1);
    }

    matches_offset
}
