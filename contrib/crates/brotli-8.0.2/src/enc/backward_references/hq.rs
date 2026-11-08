use alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use core;
use core::cmp::{max, min};

use super::hash_to_binary_tree::{
    kInfinity, Allocable, BackwardMatch, BackwardMatchMut, H10Params, StoreAndFindMatchesH10,
    Union1, ZopfliNode, H10,
};
use super::{
    kDistanceCacheIndex, kDistanceCacheOffset, kInvalidMatch, AnyHasher, BrotliEncoderParams, fix_unbroken_len,
};
use crate::enc::combined_alloc::{alloc_if, alloc_or_default};
use crate::enc::command::{
    combine_length_codes, BrotliDistanceParams, Command, GetCopyLengthCode, GetInsertLengthCode,
    PrefixEncodeCopyDistance,
};
use crate::enc::constants::{kCopyExtra, kInsExtra};
use crate::enc::encode;
use crate::enc::literal_cost::BrotliEstimateBitCostsForLiterals;
use crate::enc::static_dict::{
    BrotliDictionary, BrotliFindAllStaticDictionaryMatches, FindMatchLengthWithLimit,
};
use crate::enc::util::{floatX, FastLog2, FastLog2f64};

const BROTLI_WINDOW_GAP: usize = 16;
const BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN: usize = 37;

/*
static kBrotliMinWindowBits: i32 = 10i32;

static kBrotliMaxWindowBits: i32 = 24i32;

static kInvalidMatch: u32 = 0xfffffffu32;

static kCutoffTransformsCount: u32 = 10u32;

static kCutoffTransforms: u64 = 0x71b520au64 << 32 | 0xda2d3200u32 as (u64);

pub static kHashMul32: u32 = 0x1e35a7bdu32;

pub static kHashMul64: u64 = 0x1e35a7bdu64 << 32 | 0x1e35a7bdu64;

pub static kHashMul64Long: u64 = 0x1fe35a7bu32 as (u64) << 32 | 0xd3579bd3u32 as (u64);

*/
pub const BROTLI_MAX_EFFECTIVE_DISTANCE_ALPHABET_SIZE: usize = 544;
pub const BROTLI_NUM_LITERAL_SYMBOLS: usize = 256;
pub const BROTLI_NUM_COMMAND_SYMBOLS: usize = 704;

pub const BROTLI_SIMPLE_DISTANCE_ALPHABET_SIZE: usize = encode::BROTLI_NUM_DISTANCE_SHORT_CODES
    as usize
    + (2 * encode::BROTLI_LARGE_MAX_DISTANCE_BITS as usize);

const STORE_LOOKAHEAD_H_10: usize = 128;

#[inline(always)]
pub fn BrotliInitZopfliNodes(array: &mut [ZopfliNode], length: usize) {
    let stub = ZopfliNode::default();
    let mut i: usize;
    i = 0usize;
    while i < length {
        array[i] = stub;
        i = i.wrapping_add(1);
    }
}

impl ZopfliNode {
    #[inline(always)]
    fn copy_length(&self) -> u32 {
        self.length & 0x01ff_ffff
    }

    #[inline(always)]
    fn copy_distance(&self) -> u32 {
        self.distance
    }

    #[inline(always)]
    fn length_code(&self) -> u32 {
        self.copy_length()
            .wrapping_add(9)
            .wrapping_sub(self.length >> 25)
    }

    #[inline(always)]
    fn distance_code(&self) -> u32 {
        let short_code: u32 = self.dcode_insert_length >> 27;
        if short_code == 0u32 {
            self.copy_distance().wrapping_add(16).wrapping_sub(1)
        } else {
            short_code.wrapping_sub(1)
        }
    }
}

pub fn BrotliZopfliCreateCommands(
    num_bytes: usize,
    block_start: usize,
    max_backward_limit: usize,
    nodes: &[ZopfliNode],
    dist_cache: &mut [i32],
    last_insert_len: &mut usize,
    params: &BrotliEncoderParams,
    commands: &mut [Command],
    num_literals: &mut usize,
) {
    let mut pos: usize = 0usize;
    let mut offset: u32 = match (nodes[0]).u {
        Union1::next(off) => off,
        _ => 0,
    };
    let mut i: usize;
    let gap: usize = 0usize;
    i = 0usize;
    while offset != !(0u32) {
        {
            let next: &ZopfliNode = &nodes[pos.wrapping_add(offset as usize)];
            let copy_length = next.copy_length() as usize;
            let mut insert_length: usize = (next.dcode_insert_length & 0x07ff_ffff) as usize;
            pos = pos.wrapping_add(insert_length);
            offset = match next.u {
                Union1::next(off) => off,
                _ => 0,
            };
            if i == 0usize {
                insert_length = insert_length.wrapping_add(*last_insert_len);
                *last_insert_len = 0usize;
            }
            {
                let distance: usize = next.copy_distance() as usize;
                let len_code: usize = next.length_code() as usize;
                let max_distance: usize = min(block_start.wrapping_add(pos), max_backward_limit);
                let is_dictionary = distance > max_distance.wrapping_add(gap);
                let dist_code: usize = next.distance_code() as usize;
                commands[i].init(
                    &params.dist,
                    insert_length,
                    copy_length,
                    len_code,
                    dist_code,
                );
                if !is_dictionary && dist_code > 0 {
                    dist_cache[3] = dist_cache[2];
                    dist_cache[2] = dist_cache[1];
                    dist_cache[1] = dist_cache[0];
                    dist_cache[0] = distance as i32;
                }
            }
            *num_literals = num_literals.wrapping_add(insert_length);
            pos = pos.wrapping_add(copy_length);
        }
        i = i.wrapping_add(1);
    }
    *last_insert_len = last_insert_len.wrapping_add(num_bytes.wrapping_sub(pos));
}

#[inline(always)]
fn MaxZopfliLen(params: &BrotliEncoderParams) -> usize {
    (if params.quality <= 10i32 {
        150i32
    } else {
        325i32
    }) as usize
}

pub struct ZopfliCostModel<AllocF: Allocator<floatX>> {
    pub cost_cmd_: [floatX; BROTLI_NUM_COMMAND_SYMBOLS],
    pub cost_dist_: AllocF::AllocatedMemory,
    pub distance_histogram_size: u32,
    pub literal_costs_: AllocF::AllocatedMemory,
    pub min_cost_cmd_: floatX,
    pub num_bytes_: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct PosData {
    pub pos: usize,
    pub distance_cache: [i32; 4],
    pub costdiff: floatX,
    pub cost: floatX,
}

#[derive(Copy, Clone, Debug)]
pub struct StartPosQueue {
    pub q_: [PosData; 8],
    pub idx_: usize,
}
impl Default for StartPosQueue {
    #[inline(always)]
    fn default() -> Self {
        StartPosQueue {
            q_: [PosData {
                pos: 0,
                distance_cache: [0; 4],
                costdiff: 0.0,
                cost: 0.0,
            }; 8],
            idx_: 0,
        }
    }
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    fn init(m: &mut AllocF, dist: &BrotliDistanceParams, num_bytes: usize) -> Self {
        Self {
            num_bytes_: num_bytes,
            cost_cmd_: [0.0; 704],
            min_cost_cmd_: 0.0,
            // FIXME: makes little sense to test if N+2 > 0 -- always true unless wrapping. Perhaps use allocate() instead?
            literal_costs_: alloc_or_default::<floatX, _>(m, num_bytes + 2),
            // FIXME: possible bug because allocation size is different from the condition
            cost_dist_: alloc_if::<floatX, _>(
                dist.alphabet_size > 0,
                m,
                num_bytes + dist.alphabet_size as usize,
            ),
            distance_histogram_size: min(dist.alphabet_size, 544),
        }
    }

    fn set_from_literal_costs(
        &mut self,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
    ) {
        let literal_costs = self.literal_costs_.slice_mut();
        let mut literal_carry: floatX = 0.0;
        let cost_dist = self.cost_dist_.slice_mut();
        let cost_cmd = &mut self.cost_cmd_[..];
        let num_bytes: usize = self.num_bytes_;
        BrotliEstimateBitCostsForLiterals(
            position,
            num_bytes,
            ringbuffer_mask,
            ringbuffer,
            &mut literal_costs[1..],
        );
        literal_costs[0] = 0.0 as (floatX);
        for i in 0usize..num_bytes {
            literal_carry = literal_carry + literal_costs[i.wrapping_add(1)];
            literal_costs[i.wrapping_add(1)] = literal_costs[i] + literal_carry;
            literal_carry -= literal_costs[i.wrapping_add(1)] - literal_costs[i];
        }
        for i in 0..BROTLI_NUM_COMMAND_SYMBOLS {
            cost_cmd[i] = FastLog2(11 + i as u64);
        }
        for i in 0usize..self.distance_histogram_size as usize {
            cost_dist[i] = FastLog2((20u64).wrapping_add(i as (u64))) as (floatX);
        }
        self.min_cost_cmd_ = FastLog2(11) as (floatX);
    }
}

pub fn StitchToPreviousBlockH10<
    AllocU32: Allocator<u32>,
    Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
>(
    handle: &mut H10<AllocU32, Buckets, Params>,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
) where
    Buckets: PartialEq<Buckets>,
{
    if (num_bytes >= handle.HashTypeLength() - 1
        && position >= Params::max_tree_comp_length() as usize)
    {
        /* Store the last `MAX_TREE_COMP_LENGTH - 1` positions in the hasher.
        These could not be calculated before, since they require knowledge
        of both the previous and the current block. */
        let i_start = position - Params::max_tree_comp_length() as usize;
        let i_end = min(position, i_start.wrapping_add(num_bytes));
        for i in i_start..i_end {
            /* Maximum distance is window size - 16, see section 9.1. of the spec.
            Furthermore, we have to make sure that we don't look further back
            from the start of the next block than the window size, otherwise we
            could access already overwritten areas of the ring-buffer. */
            let max_backward = handle.window_mask_ - max(BROTLI_WINDOW_GAP - 1, position - i);
            let mut _best_len = 0;
            /* We know that i + MAX_TREE_COMP_LENGTH <= position + num_bytes, i.e. the
            end of the current block and that we have at least
            MAX_TREE_COMP_LENGTH tail in the ring-buffer. */
            StoreAndFindMatchesH10(
                handle,
                ringbuffer,
                i,
                ringbuffer_mask,
                ringbuffer_break,
                <Params as H10Params>::max_tree_comp_length() as usize,
                max_backward,
                &mut _best_len,
                &mut [],
            );
        }
    }
}
fn FindAllMatchesH10<
    AllocU32: Allocator<u32>,
    Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
>(
    handle: &mut H10<AllocU32, Buckets, Params>,
    dictionary: Option<&BrotliDictionary>,
    data: &[u8],
    ring_buffer_mask: usize,
    ring_buffer_break: Option<core::num::NonZeroUsize>,
    cur_ix: usize,
    max_length: usize,
    max_backward: usize,
    gap: usize,
    params: &BrotliEncoderParams,
    matches: &mut [u64],
) -> usize
where
    Buckets: PartialEq<Buckets>,
{
    let mut matches_offset = 0usize;
    let cur_ix_masked: usize = cur_ix & ring_buffer_mask;
    let mut best_len: usize = 1usize;
    let short_match_max_backward: usize = (if params.quality != 11i32 {
        16i32
    } else {
        64i32
    }) as usize;
    let mut stop: usize = cur_ix.wrapping_sub(short_match_max_backward);
    let mut dict_matches = [kInvalidMatch; BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN + 1];
    let mut i: usize;
    if cur_ix < short_match_max_backward {
        stop = 0usize;
    }
    i = cur_ix.wrapping_sub(1);
    while i > stop && best_len <= 2 {
        let mut prev_ix: usize = i;
        let backward: usize = cur_ix.wrapping_sub(prev_ix);
        if backward > max_backward {
            break;
        }
        prev_ix &= ring_buffer_mask;
        if data[cur_ix_masked] == data[prev_ix]
            && data[cur_ix_masked.wrapping_add(1)] == data[prev_ix.wrapping_add(1)]
        {
            let len =
                FindMatchLengthWithLimit(&data[prev_ix..], &data[cur_ix_masked..], max_length);
            if len > best_len {
                best_len = len;
                BackwardMatchMut(&mut matches[matches_offset]).init(backward, len);
                matches_offset += 1;
            }
        }
        i = i.wrapping_sub(1);
    }
    if best_len < max_length {
        let loc_offset = StoreAndFindMatchesH10(
            handle,
            data,
            cur_ix,
            ring_buffer_mask,
            ring_buffer_break,
            max_length,
            max_backward,
            &mut best_len,
            matches.split_at_mut(matches_offset).1,
        );
        matches_offset += loc_offset;
    }
    for i in 0..=37 {
        dict_matches[i] = kInvalidMatch
    }
    {
        let minlen = max(4, best_len.wrapping_add(1));
        if dictionary.is_some()
            && BrotliFindAllStaticDictionaryMatches(
                dictionary.unwrap(),
                &data[cur_ix_masked..],
                minlen,
                max_length,
                &mut dict_matches[..],
            ) != 0
        {
            assert!(params.use_dictionary);
            let maxlen = min(37, max_length);
            for l in minlen..=maxlen {
                let dict_id: u32 = dict_matches[l];
                if dict_id < kInvalidMatch {
                    let distance: usize = max_backward
                        .wrapping_add(gap)
                        .wrapping_add((dict_id >> 5) as usize)
                        .wrapping_add(1);
                    if distance <= params.dist.max_distance {
                        BackwardMatchMut(&mut matches[matches_offset]).init_dictionary(
                            distance,
                            l,
                            (dict_id & 31u32) as usize,
                        );
                        matches_offset += 1;
                    }
                }
            }
        }
    }
    matches_offset
}

impl BackwardMatch {
    #[inline(always)]
    fn length(&self) -> usize {
        (self.length_and_code() >> 5) as usize
    }
}

#[inline(always)]
fn MaxZopfliCandidates(params: &BrotliEncoderParams) -> usize {
    (if params.quality <= 10i32 { 1i32 } else { 5i32 }) as usize
}

#[inline(always)]
fn ComputeDistanceShortcut(
    block_start: usize,
    pos: usize,
    max_backward: usize,
    gap: usize,
    nodes: &[ZopfliNode],
) -> u32 {
    let clen: usize = nodes[pos].copy_length() as usize;
    let ilen: usize = ((nodes[pos]).dcode_insert_length) as usize & 0x07ff_ffff;
    let dist: usize = nodes[pos].copy_distance() as usize;
    if pos == 0usize {
        0u32
    } else if dist.wrapping_add(clen) <= block_start.wrapping_add(pos).wrapping_add(gap)
        && dist <= max_backward.wrapping_add(gap)
        && nodes[pos].distance_code() > 0
    {
        pos as u32
    } else {
        match (nodes[(pos.wrapping_sub(clen).wrapping_sub(ilen) as usize)]).u {
            Union1::shortcut(shrt) => shrt,
            _ => 0,
        }
    }
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    #[inline(always)]
    fn get_literal_costs(&self, from: usize, to: usize) -> floatX {
        self.literal_costs_.slice()[to] - self.literal_costs_.slice()[from]
    }
}

fn ComputeDistanceCache(
    pos: usize,
    mut starting_dist_cache: &[i32],
    nodes: &[ZopfliNode],
    dist_cache: &mut [i32],
) {
    let mut idx: i32 = 0i32;
    let mut p: usize = match (nodes[pos]).u {
        Union1::shortcut(shrt) => shrt,
        _ => 0,
    } as usize;
    while idx < 4i32 && (p > 0usize) {
        let ilen: usize = ((nodes[p]).dcode_insert_length) as usize & 0x07ff_ffff;
        let clen = nodes[p].copy_length() as usize;
        let dist = nodes[p].copy_distance() as usize;
        dist_cache[({
            let _old = idx;
            idx += 1;
            _old
        } as usize)] = dist as i32;
        p = match (nodes[(p.wrapping_sub(clen).wrapping_sub(ilen) as usize)]).u {
            Union1::shortcut(shrt) => shrt,
            _ => 0,
        } as usize;
    }
    while idx < 4i32 {
        {
            dist_cache[(idx as usize)] = {
                let (_old, _upper) = starting_dist_cache.split_at(1);
                starting_dist_cache = _upper;
                _old[0]
            };
        }
        idx += 1;
    }
}

impl StartPosQueue {
    #[inline(always)]
    fn size(&self) -> usize {
        min(self.idx_, 8)
    }

    fn push(&mut self, posdata: &PosData) {
        let mut offset: usize = !self.idx_ & 7usize;
        self.idx_ = self.idx_.wrapping_add(1);
        let len: usize = self.size();
        let q: &mut [PosData; 8] = &mut self.q_;
        q[offset] = *posdata;
        for _i in 1..len {
            if q[offset & 7].costdiff > q[(offset + 1) & 7].costdiff {
                q.swap(offset & 7, (offset + 1) & 7);
            }
            offset = offset.wrapping_add(1);
        }
    }
}

fn EvaluateNode<AllocF: Allocator<floatX>>(
    block_start: usize,
    pos: usize,
    max_backward_limit: usize,
    gap: usize,
    starting_dist_cache: &[i32],
    model: &ZopfliCostModel<AllocF>,
    queue: &mut StartPosQueue,
    nodes: &mut [ZopfliNode],
) {
    let node_cost: floatX = match (nodes[pos]).u {
        Union1::cost(cst) => cst,
        _ => 0.0,
    };
    (nodes[pos]).u = Union1::shortcut(ComputeDistanceShortcut(
        block_start,
        pos,
        max_backward_limit,
        gap,
        nodes,
    ));
    if node_cost <= model.get_literal_costs(0, pos) {
        let mut posdata = PosData {
            pos,
            cost: node_cost,
            costdiff: node_cost - model.get_literal_costs(0, pos),
            distance_cache: [0; 4],
        };
        ComputeDistanceCache(
            pos,
            starting_dist_cache,
            nodes,
            &mut posdata.distance_cache[..],
        );
        queue.push(&mut posdata);
    }
}

impl StartPosQueue {
    #[inline(always)]
    fn at(&self, k: usize) -> &PosData {
        &self.q_[k.wrapping_sub(self.idx_) & 7usize]
    }
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    #[inline(always)]
    fn get_min_cost_cmd(&self) -> floatX {
        self.min_cost_cmd_
    }
}

#[inline(always)]
fn ComputeMinimumCopyLength(
    start_cost: floatX,
    nodes: &[ZopfliNode],
    num_bytes: usize,
    pos: usize,
) -> usize {
    let mut min_cost: floatX = start_cost;
    let mut len: usize = 2usize;
    let mut next_len_bucket: usize = 4usize;
    let mut next_len_offset: usize = 10usize;
    while pos.wrapping_add(len) <= num_bytes
        && (match (nodes[pos.wrapping_add(len)]).u {
            Union1::cost(cst) => cst,
            _ => 0.0,
        } <= min_cost)
    {
        len = len.wrapping_add(1);
        if len == next_len_offset {
            min_cost += 1.0;
            next_len_offset = next_len_offset.wrapping_add(next_len_bucket);
            next_len_bucket = next_len_bucket.wrapping_mul(2);
        }
    }
    len
}
#[inline(always)]
fn GetInsertExtra(inscode: u16) -> u32 {
    kInsExtra[(inscode as usize)]
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    #[inline(always)]
    fn get_distance_cost(&self, distcode: usize) -> floatX {
        self.cost_dist_.slice()[distcode]
    }
}

#[inline(always)]
fn GetCopyExtra(copycode: u16) -> u32 {
    kCopyExtra[(copycode as usize)]
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    #[inline(always)]
    fn get_command_cost(&self, cmdcode: u16) -> floatX {
        self.cost_cmd_[cmdcode as usize]
    }
}

#[inline(always)]
fn UpdateZopfliNode(
    nodes: &mut [ZopfliNode],
    pos: usize,
    start_pos: usize,
    len: usize,
    len_code: usize,
    dist: usize,
    short_code: usize,
    cost: floatX,
) {
    let next = &mut nodes[pos.wrapping_add(len)];
    next.length = (len | len.wrapping_add(9u32 as usize).wrapping_sub(len_code) << 25) as u32;
    next.distance = dist as u32;
    next.dcode_insert_length = pos.wrapping_sub(start_pos) as u32 | (short_code << 27) as u32;
    next.u = Union1::cost(cost);
}

impl BackwardMatch {
    #[inline(always)]
    fn length_code(&self) -> usize {
        let code = (self.length_and_code() & 31u32) as usize;
        if code != 0 {
            code
        } else {
            self.length()
        }
    }
}

fn UpdateNodes<AllocF: Allocator<floatX>>(
    num_bytes: usize,
    block_start: usize,
    pos: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    max_backward_limit: usize,
    starting_dist_cache: &[i32],
    num_matches: usize,
    matches: &[u64],
    model: &ZopfliCostModel<AllocF>,
    queue: &mut StartPosQueue,
    nodes: &mut [ZopfliNode],
) -> usize {
    let cur_ix: usize = block_start.wrapping_add(pos);
    let cur_ix_masked: usize = cur_ix & ringbuffer_mask;
    let max_distance: usize = min(cur_ix, max_backward_limit);
    let max_len: usize = num_bytes.wrapping_sub(pos);
    let max_zopfli_len: usize = MaxZopfliLen(params);
    let min_len: usize;
    let mut result: usize = 0usize;
    let gap: usize = 0usize;
    EvaluateNode(
        block_start,
        pos,
        max_backward_limit,
        gap,
        starting_dist_cache,
        model,
        queue,
        nodes,
    );
    {
        let posdata = queue.at(0);
        let min_cost =
            posdata.cost + model.get_min_cost_cmd() + model.get_literal_costs(posdata.pos, pos);
        min_len = ComputeMinimumCopyLength(min_cost, nodes, num_bytes, pos);
    }

    for k in 0..min(MaxZopfliCandidates(params), queue.size()) {
        let posdata = queue.at(k);
        let start: usize = posdata.pos;
        let inscode: u16 = GetInsertLengthCode(pos.wrapping_sub(start));
        let start_costdiff: floatX = posdata.costdiff;
        let base_cost: floatX =
            start_costdiff + GetInsertExtra(inscode) as (floatX) + model.get_literal_costs(0, pos);
        let mut best_len: usize = min_len.wrapping_sub(1);
        for j in 0..16 {
            if best_len >= max_len {
                break;
            }

            let idx: usize = kDistanceCacheIndex[j] as usize;
            let distance_cache_len_minus_1 = 3;
            debug_assert_eq!(distance_cache_len_minus_1 + 1, posdata.distance_cache.len());
            let backward: usize = (posdata.distance_cache[(idx & distance_cache_len_minus_1)]
                + i32::from(kDistanceCacheOffset[j])) as usize;
            let mut prev_ix: usize = cur_ix.wrapping_sub(backward);
            let len: usize;
            let continuation: u8 = ringbuffer[cur_ix_masked.wrapping_add(best_len)];
            /*
            if let Some(breakpoint) = ringbuffer_break {
                if prev_ix < usize::from(breakpoint) && 
                    cur_ix_masked.wrapping_add(best_len) > usize::from(breakpoint)
                {
                    break;
                }    
                let tmp_ix = prev_ix & ringbuffer_mask;
                if tmp_ix < usize::from(breakpoint) &&
                    tmp_ix.wrapping_add(best_len) > breakpoint
                {
                    continue;
                }
            }*/
            if cur_ix_masked.wrapping_add(best_len) > ringbuffer_mask {
                break;
            }
            if backward > max_distance.wrapping_add(gap) {
                continue;
            }
            if backward > max_distance {
                continue;
            }
            if prev_ix >= cur_ix {
                continue;
            }
            prev_ix &= ringbuffer_mask;
            if prev_ix.wrapping_add(best_len) > ringbuffer_mask
                || continuation != ringbuffer[prev_ix.wrapping_add(best_len)]
            {
                continue;
            }
            len = fix_unbroken_len(
                FindMatchLengthWithLimit(
                    &ringbuffer[prev_ix..],
                    &ringbuffer[cur_ix_masked..],
                    max_len,
                ), prev_ix, cur_ix_masked, ringbuffer_break);

            let dist_cost = base_cost + model.get_distance_cost(j);
            for l in best_len.wrapping_add(1)..=len {
                let copycode: u16 = GetCopyLengthCode(l);
                let cmdcode = combine_length_codes(inscode, copycode, j == 0);
                let cost: floatX = (if cmdcode < 128 { base_cost } else { dist_cost })
                    + (GetCopyExtra(copycode) as floatX)
                    + model.get_command_cost(cmdcode);
                if cost
                    < match nodes[pos.wrapping_add(l)].u {
                        Union1::cost(cost) => cost,
                        _ => 0.0,
                    }
                {
                    UpdateZopfliNode(nodes, pos, start, l, l, backward, j.wrapping_add(1), cost);
                    result = max(result, l);
                }
                best_len = l;
            }
        }

        if k >= 2 {
            continue;
        }

        let mut len: usize = min_len;
        for j in 0usize..num_matches {
            let match_ = BackwardMatch(matches[j]);
            let dist: usize = match_.distance() as usize;
            let is_dictionary_match = dist > max_distance.wrapping_add(gap);
            let dist_code: usize = dist.wrapping_add(16).wrapping_sub(1);
            let mut dist_symbol: u16 = 0;
            let mut distextra: u32 = 0;

            PrefixEncodeCopyDistance(
                dist_code,
                params.dist.num_direct_distance_codes as usize,
                u64::from(params.dist.distance_postfix_bits),
                &mut dist_symbol,
                &mut distextra,
            );
            let distnumextra: u32 = u32::from(dist_symbol) >> 10;
            let dist_cost = base_cost
                + (distnumextra as floatX)
                + model.get_distance_cost((dist_symbol as i32 & 0x03ff) as usize);
            let max_match_len = match_.length();
            if len < max_match_len && (is_dictionary_match || max_match_len > max_zopfli_len) {
                len = max_match_len;
            }
            while len <= max_match_len {
                {
                    let len_code: usize = if is_dictionary_match {
                        match_.length_code()
                    } else {
                        len
                    };
                    let copycode: u16 = GetCopyLengthCode(len_code);
                    let cmdcode = combine_length_codes(inscode, copycode, false);
                    let cost: floatX = dist_cost
                        + GetCopyExtra(copycode) as (floatX)
                        + model.get_command_cost(cmdcode);
                    if let Union1::cost(nodeCost) = (nodes[pos.wrapping_add(len)]).u {
                        if cost < nodeCost {
                            UpdateZopfliNode(nodes, pos, start, len, len_code, dist, 0usize, cost);
                            result = max(result, len);
                        }
                    }
                }
                len = len.wrapping_add(1);
            }
        }
    }
    result
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    #[inline(always)]
    fn cleanup(&mut self, m: &mut AllocF) {
        m.free_cell(core::mem::take(&mut self.literal_costs_));
        m.free_cell(core::mem::take(&mut self.cost_dist_));
    }
}

impl ZopfliNode {
    #[inline(always)]
    fn command_length(&self) -> u32 {
        self.copy_length()
            .wrapping_add(self.dcode_insert_length & 0x07ff_ffff)
    }
}

#[inline(always)]
fn ComputeShortestPathFromNodes(num_bytes: usize, nodes: &mut [ZopfliNode]) -> usize {
    let mut index: usize = num_bytes;
    let mut num_commands: usize = 0usize;
    while (nodes[index].dcode_insert_length & 0x07ff_ffff) == 0 && nodes[index].length == 1 {
        index = index.wrapping_sub(1);
    }
    nodes[index].u = Union1::next(!(0u32));
    while index != 0 {
        let len = nodes[index].command_length() as usize;
        index = index.wrapping_sub(len);
        (nodes[index]).u = Union1::next(len as u32);
        num_commands = num_commands.wrapping_add(1);
    }
    num_commands
}

const MAX_NUM_MATCHES_H10: usize = 128;
pub fn BrotliZopfliComputeShortestPath<
    AllocU32: Allocator<u32>,
    Buckets: Allocable<u32, AllocU32> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
    AllocF: Allocator<floatX>,
>(
    m: &mut AllocF,
    dictionary: Option<&BrotliDictionary>,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    max_backward_limit: usize,
    dist_cache: &[i32],
    handle: &mut H10<AllocU32, Buckets, Params>,
    nodes: &mut [ZopfliNode],
) -> usize
where
    Buckets: PartialEq<Buckets>,
{
    let max_zopfli_len: usize = MaxZopfliLen(params);
    let mut model: ZopfliCostModel<AllocF>;
    let mut queue: StartPosQueue;
    let mut matches = [0; MAX_NUM_MATCHES_H10];
    let store_end: usize = if num_bytes >= STORE_LOOKAHEAD_H_10 {
        position
            .wrapping_add(num_bytes)
            .wrapping_sub(STORE_LOOKAHEAD_H_10)
            .wrapping_add(1)
    } else {
        position
    };
    let mut i: usize;
    let gap: usize = 0usize;
    let lz_matches_offset: usize = 0usize;
    (nodes[0]).length = 0u32;
    (nodes[0]).u = Union1::cost(0.0);
    model = ZopfliCostModel::init(m, &params.dist, num_bytes);
    if !(0i32 == 0) {
        return 0usize;
    }
    model.set_from_literal_costs(position, ringbuffer, ringbuffer_mask);
    queue = StartPosQueue::default();
    i = 0usize;
    while i.wrapping_add(handle.HashTypeLength()).wrapping_sub(1) < num_bytes {
        {
            let pos: usize = position.wrapping_add(i);
            let max_distance: usize = min(pos, max_backward_limit);
            let mut skip: usize;
            let mut num_matches: usize = FindAllMatchesH10(
                handle,
                dictionary,
                ringbuffer,
                ringbuffer_mask,
                ringbuffer_break,
                pos,
                num_bytes.wrapping_sub(i),
                max_distance,
                gap,
                params,
                &mut matches[lz_matches_offset..],
            );
            if num_matches > 0
                && BackwardMatch(matches[num_matches.wrapping_sub(1)]).length() > max_zopfli_len
            {
                matches[0] = matches[num_matches.wrapping_sub(1)];
                num_matches = 1usize;
            }
            skip = UpdateNodes(
                num_bytes,
                position,
                i,
                ringbuffer,
                ringbuffer_mask,
                ringbuffer_break,
                params,
                max_backward_limit,
                dist_cache,
                num_matches,
                &matches[..],
                &mut model,
                &mut queue,
                nodes,
            );
            if skip < 16384usize {
                skip = 0usize;
            }
            if num_matches == 1 && BackwardMatch(matches[0]).length() > max_zopfli_len {
                skip = max(BackwardMatch(matches[0]).length(), skip);
            }
            if skip > 1usize {
                handle.StoreRange(
                    ringbuffer,
                    ringbuffer_mask,
                    pos.wrapping_add(1),
                    min(pos.wrapping_add(skip), store_end),
                );
                skip = skip.wrapping_sub(1);
                while skip != 0 {
                    i = i.wrapping_add(1);
                    if i.wrapping_add(handle.HashTypeLength()).wrapping_sub(1) >= num_bytes {
                        break;
                    }
                    EvaluateNode(
                        position,
                        i,
                        max_backward_limit,
                        gap,
                        dist_cache,
                        &mut model,
                        &mut queue,
                        nodes,
                    );
                    skip = skip.wrapping_sub(1);
                }
            }
        }
        i = i.wrapping_add(1);
    }

    model.cleanup(m);

    ComputeShortestPathFromNodes(num_bytes, nodes)
}

pub fn BrotliCreateZopfliBackwardReferences<
    Alloc: Allocator<u32> + Allocator<floatX> + Allocator<ZopfliNode>,
    Buckets: Allocable<u32, Alloc> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
>(
    alloc: &mut Alloc,
    dictionary: Option<&BrotliDictionary>,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    hasher: &mut H10<Alloc, Buckets, Params>,
    dist_cache: &mut [i32],
    last_insert_len: &mut usize,
    commands: &mut [Command],
    num_commands: &mut usize,
    num_literals: &mut usize,
) where
    Buckets: PartialEq<Buckets>,
{
    let max_backward_limit: usize = (1usize << params.lgwin).wrapping_sub(16);
    // FIXME: makes little sense to test if N+1 > 0 -- always true unless wrapping. Perhaps use allocate() instead?
    let mut nodes = alloc_or_default::<ZopfliNode, _>(alloc, num_bytes + 1);
    if !(0i32 == 0) {
        return;
    }
    BrotliInitZopfliNodes(nodes.slice_mut(), num_bytes.wrapping_add(1));
    *num_commands = num_commands.wrapping_add(BrotliZopfliComputeShortestPath(
        alloc,
        dictionary,
        num_bytes,
        position,
        ringbuffer,
        ringbuffer_mask,
        ringbuffer_break,
        params,
        max_backward_limit,
        dist_cache,
        hasher,
        nodes.slice_mut(),
    ));
    if !(0i32 == 0) {
        return;
    }
    BrotliZopfliCreateCommands(
        num_bytes,
        position,
        max_backward_limit,
        nodes.slice(),
        dist_cache,
        last_insert_len,
        params,
        commands,
        num_literals,
    );
    {
        <Alloc as Allocator<ZopfliNode>>::free_cell(alloc, core::mem::take(&mut nodes));
    }
}

fn SetCost(histogram: &[u32], histogram_size: usize, literal_histogram: bool, cost: &mut [floatX]) {
    let mut sum: u64 = 0;
    for i in 0..histogram_size {
        sum = sum.wrapping_add(u64::from(histogram[i]));
    }
    let log2sum = FastLog2(sum);

    let mut missing_symbol_sum = sum;
    if !literal_histogram {
        for i in 0..histogram_size {
            if histogram[i] == 0 {
                missing_symbol_sum = missing_symbol_sum.wrapping_add(1);
            }
        }
    }

    let missing_symbol_cost = FastLog2f64(missing_symbol_sum) + 2.0;
    for i in 0..histogram_size {
        if histogram[i] == 0 {
            cost[i] = missing_symbol_cost;
        } else {
            cost[i] = log2sum - FastLog2(u64::from(histogram[i]));
            if cost[i] < 1.0 {
                cost[i] = 1.0;
            }
        }
    }
}

impl<AllocF: Allocator<floatX>> ZopfliCostModel<AllocF> {
    fn set_from_commands(
        &mut self,
        position: usize,
        ringbuffer: &[u8],
        ringbuffer_mask: usize,
        commands: &[Command],
        num_commands: usize,
        last_insert_len: usize,
    ) {
        let mut histogram_literal = [0u32; BROTLI_NUM_LITERAL_SYMBOLS];
        let mut histogram_cmd = [0u32; BROTLI_NUM_COMMAND_SYMBOLS];
        let mut histogram_dist = [0u32; BROTLI_SIMPLE_DISTANCE_ALPHABET_SIZE];
        let mut cost_literal = [0.0; BROTLI_NUM_LITERAL_SYMBOLS];
        let mut pos: usize = position.wrapping_sub(last_insert_len);
        let mut min_cost_cmd: floatX = kInfinity;
        let mut i: usize;
        let cost_cmd: &mut [floatX] = &mut self.cost_cmd_[..];
        i = 0usize;
        while i < num_commands {
            {
                let inslength: usize = (commands[i]).insert_len_ as usize;
                let copylength: usize = commands[i].copy_len() as usize;
                let distcode: usize = (commands[i].dist_prefix_ as i32 & 0x03ff) as usize;
                let cmdcode: usize = (commands[i]).cmd_prefix_ as usize;
                {
                    let _rhs = 1;
                    let _lhs = &mut histogram_cmd[cmdcode];
                    *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                }
                if cmdcode >= 128usize {
                    let _rhs = 1;
                    let _lhs = &mut histogram_dist[distcode];
                    *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                }
                for j in 0usize..inslength {
                    let _rhs = 1;
                    let _lhs = &mut histogram_literal
                        [(ringbuffer[(pos.wrapping_add(j) & ringbuffer_mask)] as usize)];
                    *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                }
                pos = pos.wrapping_add(inslength.wrapping_add(copylength));
            }
            i = i.wrapping_add(1);
        }
        SetCost(
            &histogram_literal[..],
            BROTLI_NUM_LITERAL_SYMBOLS,
            true,
            &mut cost_literal,
        );
        SetCost(
            &histogram_cmd[..],
            BROTLI_NUM_COMMAND_SYMBOLS,
            false,
            &mut cost_cmd[..],
        );
        SetCost(
            &histogram_dist[..],
            self.distance_histogram_size as usize,
            false,
            self.cost_dist_.slice_mut(),
        );
        for i in 0usize..704usize {
            min_cost_cmd = min_cost_cmd.min(cost_cmd[i]);
        }
        self.min_cost_cmd_ = min_cost_cmd;
        {
            let literal_costs: &mut [floatX] = self.literal_costs_.slice_mut();
            let mut literal_carry: floatX = 0.0;
            let num_bytes: usize = self.num_bytes_;
            literal_costs[0] = 0.0 as (floatX);
            for i in 0usize..num_bytes {
                literal_carry +=
                    cost_literal[ringbuffer[position.wrapping_add(i) & ringbuffer_mask] as usize];
                literal_costs[i.wrapping_add(1)] = literal_costs[i] + literal_carry;
                literal_carry -= literal_costs[i.wrapping_add(1)] - literal_costs[i];
            }
        }
    }
}

fn ZopfliIterate<AllocF: Allocator<floatX>>(
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    max_backward_limit: usize,
    gap: usize,
    dist_cache: &[i32],
    model: &ZopfliCostModel<AllocF>,
    num_matches: &[u32],
    matches: &[u64],
    nodes: &mut [ZopfliNode],
) -> usize {
    let max_zopfli_len: usize = MaxZopfliLen(params);
    let mut queue: StartPosQueue;
    let mut cur_match_pos: usize = 0usize;
    let mut i: usize;
    (nodes[0]).length = 0u32;
    (nodes[0]).u = Union1::cost(0.0);
    queue = StartPosQueue::default();
    i = 0usize;
    while i.wrapping_add(3) < num_bytes {
        {
            let mut skip: usize = UpdateNodes(
                num_bytes,
                position,
                i,
                ringbuffer,
                ringbuffer_mask,
                ringbuffer_break,
                params,
                max_backward_limit,
                dist_cache,
                num_matches[i] as usize,
                &matches[cur_match_pos..],
                model,
                &mut queue,
                nodes,
            );
            if skip < 16384usize {
                skip = 0usize;
            }
            cur_match_pos = cur_match_pos.wrapping_add(num_matches[i] as usize);
            if num_matches[i] == 1
                && BackwardMatch(matches[cur_match_pos.wrapping_sub(1)]).length() > max_zopfli_len
            {
                skip = max(
                    BackwardMatch(matches[cur_match_pos.wrapping_sub(1)]).length(),
                    skip,
                );
            }
            if skip > 1usize {
                skip = skip.wrapping_sub(1);
                while skip != 0 {
                    i = i.wrapping_add(1);
                    if i.wrapping_add(3) >= num_bytes {
                        break;
                    }
                    EvaluateNode(
                        position,
                        i,
                        max_backward_limit,
                        gap,
                        dist_cache,
                        model,
                        &mut queue,
                        nodes,
                    );
                    cur_match_pos = cur_match_pos.wrapping_add(num_matches[i] as usize);
                    skip = skip.wrapping_sub(1);
                }
            }
        }
        i = i.wrapping_add(1);
    }
    ComputeShortestPathFromNodes(num_bytes, nodes)
}

pub fn BrotliCreateHqZopfliBackwardReferences<
    Alloc: Allocator<u32> + Allocator<u64> + Allocator<floatX> + Allocator<ZopfliNode>,
    Buckets: Allocable<u32, Alloc> + SliceWrapperMut<u32> + SliceWrapper<u32>,
    Params: H10Params,
>(
    alloc: &mut Alloc,
    dictionary: Option<&BrotliDictionary>,
    num_bytes: usize,
    position: usize,
    ringbuffer: &[u8],
    ringbuffer_mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &BrotliEncoderParams,
    hasher: &mut H10<Alloc, Buckets, Params>,
    dist_cache: &mut [i32],
    last_insert_len: &mut usize,
    commands: &mut [Command],
    num_commands: &mut usize,
    num_literals: &mut usize,
) where
    Buckets: PartialEq<Buckets>,
{
    let max_backward_limit: usize = (1usize << params.lgwin).wrapping_sub(16);
    let mut num_matches = alloc_or_default::<u32, _>(alloc, num_bytes);
    let mut matches_size: usize = (4usize).wrapping_mul(num_bytes);
    let store_end: usize = if num_bytes >= STORE_LOOKAHEAD_H_10 {
        position
            .wrapping_add(num_bytes)
            .wrapping_sub(STORE_LOOKAHEAD_H_10)
            .wrapping_add(1)
    } else {
        position
    };
    let mut cur_match_pos: usize = 0usize;
    let mut i: usize;

    let mut orig_dist_cache = [0i32; 4];

    let mut model: ZopfliCostModel<Alloc>;
    let mut matches = alloc_or_default::<u64, _>(alloc, matches_size);
    let gap: usize = 0usize;
    let shadow_matches: usize = 0usize;
    i = 0usize;
    while i.wrapping_add(hasher.HashTypeLength()).wrapping_sub(1) < num_bytes {
        {
            let pos: usize = position.wrapping_add(i);
            let max_distance: usize = min(pos, max_backward_limit);
            let max_length: usize = num_bytes.wrapping_sub(i);

            let mut j: usize;
            {
                if matches_size < cur_match_pos.wrapping_add(128).wrapping_add(shadow_matches) {
                    let mut new_size: usize = if matches_size == 0usize {
                        cur_match_pos.wrapping_add(128).wrapping_add(shadow_matches)
                    } else {
                        matches_size
                    };
                    while new_size < cur_match_pos.wrapping_add(128).wrapping_add(shadow_matches) {
                        new_size = new_size.wrapping_mul(2);
                    }
                    let mut new_array = alloc_or_default::<u64, _>(alloc, new_size);
                    if matches_size != 0 {
                        for (dst, src) in new_array
                            .slice_mut()
                            .split_at_mut(matches_size)
                            .0
                            .iter_mut()
                            .zip(matches.slice().split_at(matches_size).0.iter())
                        {
                            *dst = *src;
                        }
                    }
                    {
                        <Alloc as Allocator<u64>>::free_cell(alloc, core::mem::take(&mut matches));
                    }
                    matches = new_array;
                    matches_size = new_size;
                }
            }
            if !(0i32 == 0) {
                return;
            }
            let num_found_matches: usize = FindAllMatchesH10(
                hasher,
                dictionary, //&params.dictionary ,
                ringbuffer,
                ringbuffer_mask,
                ringbuffer_break,
                pos,
                max_length,
                max_distance,
                gap,
                params,
                &mut matches.slice_mut()[cur_match_pos.wrapping_add(shadow_matches)..],
            );
            let cur_match_end: usize = cur_match_pos.wrapping_add(num_found_matches);
            j = cur_match_pos;
            while j.wrapping_add(1) < cur_match_end {
                {}
                j = j.wrapping_add(1);
            }
            num_matches.slice_mut()[i] = num_found_matches as u32;
            if num_found_matches > 0usize {
                let match_len =
                    BackwardMatch(matches.slice()[cur_match_end.wrapping_sub(1)]).length();
                if match_len > 325usize {
                    let skip: usize = match_len.wrapping_sub(1);
                    let tmp = matches.slice()[(cur_match_end.wrapping_sub(1) as usize)];
                    matches.slice_mut()[cur_match_pos] = tmp;
                    cur_match_pos = cur_match_pos.wrapping_add(1);
                    num_matches.slice_mut()[i] = 1u32;
                    hasher.StoreRange(
                        ringbuffer,
                        ringbuffer_mask,
                        pos.wrapping_add(1),
                        min(pos.wrapping_add(match_len), store_end),
                    );
                    for item in num_matches
                        .slice_mut()
                        .split_at_mut(i.wrapping_add(1))
                        .1
                        .split_at_mut(skip)
                        .0
                        .iter_mut()
                    {
                        *item = 0;
                    }
                    i = i.wrapping_add(skip);
                } else {
                    cur_match_pos = cur_match_end;
                }
            }
        }
        i = i.wrapping_add(1);
    }
    let orig_num_literals: usize = *num_literals;
    let orig_last_insert_len: usize = *last_insert_len;
    for (i, j) in orig_dist_cache
        .split_at_mut(4)
        .0
        .iter_mut()
        .zip(dist_cache.split_at(4).0)
    {
        *i = *j;
    }
    let orig_num_commands: usize = *num_commands;
    // FIXME: makes little sense to test if N+1 > 0 -- always true unless wrapping. Perhaps use allocate() instead?
    let mut nodes = alloc_or_default::<ZopfliNode, _>(alloc, num_bytes + 1);
    if !(0i32 == 0) {
        return;
    }
    model = ZopfliCostModel::init(alloc, &params.dist, num_bytes);
    if !(0i32 == 0) {
        return;
    }
    for i in 0usize..2usize {
        BrotliInitZopfliNodes(nodes.slice_mut(), num_bytes.wrapping_add(1));
        if i == 0usize {
            model.set_from_literal_costs(position, ringbuffer, ringbuffer_mask);
        } else {
            model.set_from_commands(
                position,
                ringbuffer,
                ringbuffer_mask,
                commands,
                num_commands.wrapping_sub(orig_num_commands),
                orig_last_insert_len,
            );
        }
        *num_commands = orig_num_commands;
        *num_literals = orig_num_literals;
        *last_insert_len = orig_last_insert_len;
        for (i, j) in dist_cache
            .split_at_mut(4)
            .0
            .iter_mut()
            .zip(orig_dist_cache.split_at(4).0)
        {
            *i = *j;
        }
        *num_commands = num_commands.wrapping_add(ZopfliIterate(
            num_bytes,
            position,
            ringbuffer,
            ringbuffer_mask,
            ringbuffer_break,
            params,
            max_backward_limit,
            gap,
            dist_cache,
            &mut model,
            num_matches.slice(),
            matches.slice(),
            nodes.slice_mut(),
        ));
        BrotliZopfliCreateCommands(
            num_bytes,
            position,
            max_backward_limit,
            nodes.slice(),
            dist_cache,
            last_insert_len,
            params,
            commands,
            num_literals,
        );
    }
    model.cleanup(alloc);
    <Alloc as Allocator<ZopfliNode>>::free_cell(alloc, nodes);
    <Alloc as Allocator<u64>>::free_cell(alloc, matches);
    <Alloc as Allocator<u32>>::free_cell(alloc, num_matches);
}
