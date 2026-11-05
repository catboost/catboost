use core;
use core::cmp::{max, min};

use super::super::alloc;
use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::backward_references::BrotliEncoderParams;
use super::bit_cost::{BitsEntropy, BrotliPopulationCost};
use super::block_split::BlockSplit;
use super::block_splitter::BrotliSplitBlock;
use super::brotli_bit_stream::MetaBlockSplit;
use super::cluster::BrotliClusterHistograms;
use super::combined_alloc::BrotliAlloc;
use super::command::{BrotliDistanceParams, Command, PrefixEncodeCopyDistance};
use super::constants::BROTLI_MAX_NPOSTFIX;
use super::encode::{
    BROTLI_DISTANCE_ALPHABET_SIZE, BROTLI_LARGE_MAX_DISTANCE_BITS, BROTLI_MAX_ALLOWED_DISTANCE,
    BROTLI_MAX_DISTANCE_BITS,
};
use super::entropy_encode::BrotliOptimizeHuffmanCountsForRle;
use super::histogram::{
    BrotliBuildHistogramsWithContext, ClearHistograms, Context, ContextType, CostAccessors,
    HistogramAddHistogram, HistogramAddItem, HistogramClear, HistogramCommand, HistogramDistance,
    HistogramLiteral,
};
use crate::enc::combined_alloc::{alloc_default, allocate};
use crate::enc::floatX;

pub fn BrotliInitDistanceParams(params: &mut BrotliEncoderParams, npostfix: u32, ndirect: u32) {
    let dist_params = &mut params.dist;
    let mut alphabet_size;
    let mut max_distance;

    dist_params.distance_postfix_bits = npostfix;
    dist_params.num_direct_distance_codes = ndirect;

    alphabet_size = BROTLI_DISTANCE_ALPHABET_SIZE(npostfix, ndirect, BROTLI_MAX_DISTANCE_BITS);
    max_distance =
        ndirect + (1u32 << (BROTLI_MAX_DISTANCE_BITS + npostfix + 2)) - (1u32 << (npostfix + 2));

    if (params.large_window) {
        let bound: [u32; BROTLI_MAX_NPOSTFIX + 1] = [0, 4, 12, 28];
        let postfix = 1u32 << npostfix;
        alphabet_size =
            BROTLI_DISTANCE_ALPHABET_SIZE(npostfix, ndirect, BROTLI_LARGE_MAX_DISTANCE_BITS);
        /* The maximum distance is set so that no distance symbol used can encode
        a distance larger than BROTLI_MAX_ALLOWED_DISTANCE with all
        its extra bits set. */
        if (ndirect < bound[npostfix as usize]) {
            max_distance =
                BROTLI_MAX_ALLOWED_DISTANCE as u32 - (bound[npostfix as usize] - ndirect);
        } else if (ndirect >= bound[npostfix as usize] + postfix) {
            max_distance = (3u32 << 29) - 4 + (ndirect - bound[npostfix as usize]);
        } else {
            max_distance = BROTLI_MAX_ALLOWED_DISTANCE as u32;
        }
    }

    dist_params.alphabet_size = alphabet_size;
    dist_params.max_distance = max_distance as usize;
}

fn RecomputeDistancePrefixes(
    cmds: &mut [Command],
    num_commands: usize,
    orig_params: &BrotliDistanceParams,
    new_params: &BrotliDistanceParams,
) {
    if orig_params.distance_postfix_bits == new_params.distance_postfix_bits
        && orig_params.num_direct_distance_codes == new_params.num_direct_distance_codes
    {
        return;
    }

    for cmd in cmds.split_at_mut(num_commands).0.iter_mut() {
        if (cmd.copy_len() != 0 && cmd.cmd_prefix_ >= 128) {
            let ret = cmd.restore_distance_code(orig_params);
            PrefixEncodeCopyDistance(
                ret as usize,
                new_params.num_direct_distance_codes as usize,
                new_params.distance_postfix_bits as u64,
                &mut cmd.dist_prefix_,
                &mut cmd.dist_extra_,
            );
        }
    }
}

fn ComputeDistanceCost(
    cmds: &[Command],
    num_commands: usize,
    orig_params: &BrotliDistanceParams,
    new_params: &BrotliDistanceParams,
    scratch: &mut <HistogramDistance as CostAccessors>::i32vec,
    cost: &mut f64,
) -> bool {
    let mut equal_params = false;
    let mut dist_prefix: u16 = 0;
    let mut dist_extra: u32 = 0;
    let mut extra_bits: f64 = 0.0;
    let mut histo = HistogramDistance::default();

    if (orig_params.distance_postfix_bits == new_params.distance_postfix_bits
        && orig_params.num_direct_distance_codes == new_params.num_direct_distance_codes)
    {
        equal_params = true;
    }
    for cmd in cmds.split_at(num_commands).0 {
        if cmd.copy_len() != 0 && cmd.cmd_prefix_ >= 128 {
            if equal_params {
                dist_prefix = cmd.dist_prefix_;
            } else {
                let distance = cmd.restore_distance_code(orig_params);
                if distance > new_params.max_distance as u32 {
                    return false;
                }
                PrefixEncodeCopyDistance(
                    distance as usize,
                    new_params.num_direct_distance_codes as usize,
                    new_params.distance_postfix_bits as u64,
                    &mut dist_prefix,
                    &mut dist_extra,
                );
            }
            HistogramAddItem(&mut histo, (dist_prefix & 0x03ff) as usize);
            extra_bits += (dist_prefix >> 10) as f64;
        }
    }

    *cost = BrotliPopulationCost(&histo, scratch) as f64 + extra_bits;
    true
}

pub fn BrotliBuildMetaBlock<Alloc: BrotliAlloc>(
    alloc: &mut Alloc,
    ringbuffer: &[u8],
    pos: usize,
    mask: usize,
    params: &mut BrotliEncoderParams,
    prev_byte: u8,
    prev_byte2: u8,
    cmds: &mut [Command],
    num_commands: usize,
    literal_context_mode: ContextType,
    lit_scratch_space: &mut <HistogramLiteral as CostAccessors>::i32vec,
    cmd_scratch_space: &mut <HistogramCommand as CostAccessors>::i32vec,
    dst_scratch_space: &mut <HistogramDistance as CostAccessors>::i32vec,
    mb: &mut MetaBlockSplit<Alloc>,
) {
    static kMaxNumberOfHistograms: usize = 256usize;
    let mut distance_histograms: <Alloc as Allocator<HistogramDistance>>::AllocatedMemory;
    let mut literal_histograms: <Alloc as Allocator<HistogramLiteral>>::AllocatedMemory;
    let mut literal_context_modes = alloc_default::<ContextType, Alloc>();

    let mut i: usize;
    let mut literal_context_multiplier: usize = 1;
    let mut ndirect_msb: u32 = 0;
    let mut check_orig = true;
    if !params.avoid_distance_prefix_search {
        let mut best_dist_cost: f64 = 1e99;
        let orig_params = params.clone();
        let mut new_params = params.clone();

        for npostfix in 0..(BROTLI_MAX_NPOSTFIX + 1) {
            while ndirect_msb < 16 {
                let ndirect = ndirect_msb << npostfix;

                let mut dist_cost: f64 = 0.0;
                BrotliInitDistanceParams(&mut new_params, npostfix as u32, ndirect);
                if npostfix as u32 == orig_params.dist.distance_postfix_bits
                    && ndirect == orig_params.dist.num_direct_distance_codes
                {
                    check_orig = false;
                }
                let skip: bool = !ComputeDistanceCost(
                    cmds,
                    num_commands,
                    &orig_params.dist,
                    &new_params.dist,
                    dst_scratch_space,
                    &mut dist_cost,
                );
                if skip || (dist_cost > best_dist_cost) {
                    break;
                }
                best_dist_cost = dist_cost;
                params.dist = new_params.dist;
                ndirect_msb += 1;
            }
            ndirect_msb = ndirect_msb.saturating_sub(1);
            ndirect_msb /= 2;
        }
        if check_orig {
            let mut dist_cost: f64 = 0.0;
            ComputeDistanceCost(
                cmds,
                num_commands,
                &orig_params.dist,
                &orig_params.dist,
                dst_scratch_space,
                &mut dist_cost,
            );
            if dist_cost < best_dist_cost {
                // best_dist_cost = dist_cost; unused
                params.dist = orig_params.dist;
            }
        }
        RecomputeDistancePrefixes(cmds, num_commands, &orig_params.dist, &params.dist);
    }
    BrotliSplitBlock(
        alloc,
        cmds,
        num_commands,
        ringbuffer,
        pos,
        mask,
        params,
        lit_scratch_space,
        cmd_scratch_space,
        dst_scratch_space,
        &mut mb.literal_split,
        &mut mb.command_split,
        &mut mb.distance_split,
    );
    if params.disable_literal_context_modeling == 0 {
        literal_context_multiplier = (1i32 << 6) as usize;
        literal_context_modes = allocate::<ContextType, _>(alloc, mb.literal_split.num_types);
        for item in literal_context_modes.slice_mut().iter_mut() {
            *item = literal_context_mode;
        }
    }
    let literal_histograms_size: usize = mb
        .literal_split
        .num_types
        .wrapping_mul(literal_context_multiplier);
    literal_histograms = allocate::<HistogramLiteral, _>(alloc, literal_histograms_size);
    let distance_histograms_size: usize = mb.distance_split.num_types << 2;
    distance_histograms = allocate::<HistogramDistance, _>(alloc, distance_histograms_size);
    mb.command_histograms_size = mb.command_split.num_types;
    mb.command_histograms = allocate::<HistogramCommand, _>(alloc, mb.command_histograms_size);
    BrotliBuildHistogramsWithContext(
        cmds,
        num_commands,
        &mut mb.literal_split,
        &mut mb.command_split,
        &mut mb.distance_split,
        ringbuffer,
        pos,
        mask,
        prev_byte,
        prev_byte2,
        literal_context_modes.slice(),
        literal_histograms.slice_mut(),
        mb.command_histograms.slice_mut(),
        distance_histograms.slice_mut(),
    );
    <Alloc as Allocator<ContextType>>::free_cell(alloc, literal_context_modes);
    mb.literal_context_map_size = mb.literal_split.num_types << 6;
    mb.literal_context_map = allocate::<u32, _>(alloc, mb.literal_context_map_size);
    mb.literal_histograms_size = mb.literal_context_map_size;
    mb.literal_histograms = allocate::<HistogramLiteral, _>(alloc, mb.literal_histograms_size);
    BrotliClusterHistograms(
        alloc,
        literal_histograms.slice(),
        literal_histograms_size,
        kMaxNumberOfHistograms,
        lit_scratch_space,
        mb.literal_histograms.slice_mut(),
        &mut mb.literal_histograms_size,
        mb.literal_context_map.slice_mut(),
    );
    <Alloc as Allocator<HistogramLiteral>>::free_cell(alloc, literal_histograms);
    if params.disable_literal_context_modeling != 0 {
        i = mb.literal_split.num_types;
        while i != 0usize {
            let mut j: usize = 0usize;
            i = i.wrapping_sub(1);
            while j < (1i32 << 6) as usize {
                {
                    let val = mb.literal_context_map.slice()[i];
                    mb.literal_context_map.slice_mut()[(i << 6).wrapping_add(j)] = val;
                }
                j = j.wrapping_add(1);
            }
        }
    }
    mb.distance_context_map_size = mb.distance_split.num_types << 2;
    mb.distance_context_map = allocate::<u32, _>(alloc, mb.distance_context_map_size);
    mb.distance_histograms_size = mb.distance_context_map_size;
    mb.distance_histograms = allocate::<HistogramDistance, _>(alloc, mb.distance_histograms_size);
    BrotliClusterHistograms(
        alloc,
        distance_histograms.slice(),
        mb.distance_context_map_size,
        kMaxNumberOfHistograms,
        dst_scratch_space,
        mb.distance_histograms.slice_mut(),
        &mut mb.distance_histograms_size,
        mb.distance_context_map.slice_mut(),
    );
    <Alloc as Allocator<HistogramDistance>>::free_cell(alloc, distance_histograms);
}

/*
pub struct BlockSplitter<'a, HistogramType:SliceWrapper<u32>+SliceWrapperMut<u32> +CostAccessors,
                         AllocU8:alloc::Allocator<u8>+'a,
                         AllocU32:alloc::Allocator<u32>+'a,
                         AllocHT:alloc::Allocator<HistogramType>+'a > {
                         */
pub struct BlockSplitter {
    pub alphabet_size_: usize,
    pub min_block_size_: usize,
    pub split_threshold_: floatX,
    pub num_blocks_: usize,
    //  pub split_: &'a mut BlockSplit<AllocU8, AllocU32>,
    //  pub histograms_: AllocHT::AllocatedMemory, // FIXME: pull this one out at the end
    //  pub histograms_size_: &'a mut usize, // FIXME: pull this one out at the end
    pub target_block_size_: usize,
    pub block_size_: usize,
    pub curr_histogram_ix_: usize,
    pub last_histogram_ix_: [usize; 2],
    pub last_entropy_: [floatX; 2],
    pub merge_last_count_: usize,
}

pub struct ContextBlockSplitter {
    pub alphabet_size_: usize,
    pub num_contexts_: usize,
    pub max_block_types_: usize,
    pub min_block_size_: usize,
    pub split_threshold_: floatX,
    pub num_blocks_: usize,
    //  pub split_: &'a mut BlockSplit<AllocU8, AllocU32>,
    //  pub histograms_: AllocHL::AllocatedMemory,
    //  pub histograms_size_: &'a mut usize, // FIXME: pull this one out at the end
    pub target_block_size_: usize,
    pub block_size_: usize,
    pub curr_histogram_ix_: usize,
    pub last_histogram_ix_: [usize; 2],
    pub last_entropy_: [floatX; 2 * BROTLI_MAX_STATIC_CONTEXTS],
    pub merge_last_count_: usize,
}

enum LitBlocks {
    plain(BlockSplitter),      //<'a, HistogramLiteral, AllocU8, AllocU32, AllocHL>,
    ctx(ContextBlockSplitter), //<'a, AllocU8, AllocU32, AllocHL>,
}

/*

pub struct BlockSplitterCommand {
  pub alphabet_size_: usize,
  pub min_block_size_: usize,
  pub split_threshold_: floatX,
  pub num_blocks_: usize,
  pub split_: *mut BlockSplit,
  pub histograms_: *mut HistogramCommand,
  pub histograms_size_: *mut usize,
  pub target_block_size_: usize,
  pub block_size_: usize,
  pub curr_histogram_ix_: usize,
  pub last_histogram_ix_: [usize; 2],
  pub last_entropy_: [floatX; 2],
  pub merge_last_count_: usize,
}



pub struct BlockSplitterDistance {
  pub alphabet_size_: usize,
  pub min_block_size_: usize,
  pub split_threshold_: floatX,
  pub num_blocks_: usize,
  pub split_: *mut BlockSplit,
  pub histograms_: *mut HistogramDistance,
  pub histograms_size_: *mut usize,
  pub target_block_size_: usize,
  pub block_size_: usize,
  pub curr_histogram_ix_: usize,
  pub last_histogram_ix_: [usize; 2],
  pub last_entropy_: [floatX; 2],
  pub merge_last_count_: usize,
}
*/

fn InitBlockSplitter<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32> + alloc::Allocator<HistogramType>,
>(
    alloc: &mut Alloc,
    alphabet_size: usize,
    min_block_size: usize,
    split_threshold: floatX,
    num_symbols: usize,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut <Alloc as Allocator<HistogramType>>::AllocatedMemory,
    histograms_size: &mut usize,
) -> BlockSplitter {
    let max_num_blocks: usize = num_symbols.wrapping_div(min_block_size).wrapping_add(1);
    let max_num_types: usize = min(max_num_blocks, (256i32 + 1i32) as usize);
    let mut xself = BlockSplitter {
        last_entropy_: [0.0; 2],
        alphabet_size_: alphabet_size,
        min_block_size_: min_block_size,
        split_threshold_: split_threshold,
        num_blocks_: 0usize,
        //xself.split_ : split,
        //xself.histograms_size_ : histograms_size,
        target_block_size_: min_block_size,
        block_size_: 0usize,
        curr_histogram_ix_: 0usize,
        merge_last_count_: 0usize,
        last_histogram_ix_: [0; 2],
    };
    {
        if split.types.slice().len() < max_num_blocks {
            let mut _new_size: usize = if split.types.slice().is_empty() {
                max_num_blocks
            } else {
                split.types.slice().len()
            };
            let mut new_array: <Alloc as Allocator<u8>>::AllocatedMemory;
            while _new_size < max_num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            new_array = allocate::<u8, _>(alloc, _new_size);
            if (!split.types.slice().is_empty()) {
                new_array.slice_mut()[..split.types.slice().len()]
                    .clone_from_slice(split.types.slice());
            }
            <Alloc as Allocator<u8>>::free_cell(
                alloc,
                core::mem::replace(&mut split.types, new_array),
            );
        }
    }
    {
        if split.lengths.slice().len() < max_num_blocks {
            let mut _new_size: usize = if split.lengths.slice().is_empty() {
                max_num_blocks
            } else {
                split.lengths.slice().len()
            };
            while _new_size < max_num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u32, _>(alloc, _new_size);
            new_array.slice_mut()[..split.lengths.slice().len()]
                .clone_from_slice(split.lengths.slice());
            <Alloc as Allocator<u32>>::free_cell(
                alloc,
                core::mem::replace(&mut split.lengths, new_array),
            );
        }
    }
    split.num_blocks = max_num_blocks;
    *histograms_size = max_num_types;
    let hlocal = allocate::<HistogramType, _>(alloc, *histograms_size);
    <Alloc as Allocator<HistogramType>>::free_cell(
        alloc,
        core::mem::replace(&mut *histograms, hlocal),
    );
    HistogramClear(&mut histograms.slice_mut()[0]);
    xself.last_histogram_ix_[0] = 0;
    xself.last_histogram_ix_[1] = 0;
    xself
}
fn InitContextBlockSplitter<
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32> + alloc::Allocator<HistogramLiteral>,
>(
    alloc: &mut Alloc,
    alphabet_size: usize,
    num_contexts: usize,
    min_block_size: usize,
    split_threshold: floatX,
    num_symbols: usize,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut <Alloc as Allocator<HistogramLiteral>>::AllocatedMemory,
    histograms_size: &mut usize,
) -> ContextBlockSplitter {
    let max_num_blocks: usize = num_symbols.wrapping_div(min_block_size).wrapping_add(1);

    assert!(num_contexts <= BROTLI_MAX_STATIC_CONTEXTS);
    let mut xself = ContextBlockSplitter {
        alphabet_size_: alphabet_size,
        num_contexts_: num_contexts,
        max_block_types_: (256usize).wrapping_div(num_contexts),
        min_block_size_: min_block_size,
        split_threshold_: split_threshold,
        num_blocks_: 0usize,
        //        histograms_size_: histograms_size,
        target_block_size_: min_block_size,
        block_size_: 0usize,
        curr_histogram_ix_: 0usize,
        merge_last_count_: 0usize,
        last_histogram_ix_: [0; 2],
        last_entropy_: [0.0; 2 * BROTLI_MAX_STATIC_CONTEXTS],
    };
    let max_num_types: usize = min(max_num_blocks, xself.max_block_types_.wrapping_add(1));
    {
        if split.types.slice().len() < max_num_blocks {
            let mut _new_size: usize = if split.types.slice().is_empty() {
                max_num_blocks
            } else {
                split.types.slice().len()
            };
            while _new_size < max_num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u8, _>(alloc, _new_size);
            if (!split.types.slice().is_empty()) {
                new_array.slice_mut()[..split.types.slice().len()]
                    .clone_from_slice(split.types.slice());
            }
            <Alloc as Allocator<u8>>::free_cell(
                alloc,
                core::mem::replace(&mut split.types, new_array),
            );
        }
    }
    {
        if split.lengths.slice().len() < max_num_blocks {
            let mut _new_size: usize = if split.lengths.slice().is_empty() {
                max_num_blocks
            } else {
                split.lengths.slice().len()
            };
            while _new_size < max_num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u32, _>(alloc, _new_size);
            if (!split.lengths.slice().is_empty()) {
                new_array.slice_mut()[..split.lengths.slice().len()]
                    .clone_from_slice(split.lengths.slice());
            }
            <Alloc as Allocator<u32>>::free_cell(
                alloc,
                core::mem::replace(&mut split.lengths, new_array),
            );
        }
    }
    split.num_blocks = max_num_blocks;
    *histograms_size = max_num_types.wrapping_mul(num_contexts);
    *histograms = allocate::<HistogramLiteral, _>(alloc, *histograms_size);
    //xself.histograms_ = *histograms;
    ClearHistograms(&mut histograms.slice_mut()[0..], num_contexts);
    xself.last_histogram_ix_[0] = 0;
    xself.last_histogram_ix_[1] = 0;
    xself
}

fn BlockSplitterFinishBlock<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + Clone,
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32>,
>(
    xself: &mut BlockSplitter,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut [HistogramType],
    histograms_size: &mut usize,
    is_final: bool,
) {
    xself.block_size_ = max(xself.block_size_, xself.min_block_size_);
    if xself.num_blocks_ == 0usize {
        split.lengths.slice_mut()[0] = xself.block_size_ as u32;
        split.types.slice_mut()[0] = 0u8;
        xself.last_entropy_[0] = BitsEntropy((histograms[0]).slice(), xself.alphabet_size_);
        xself.last_entropy_[1] = xself.last_entropy_[0];
        xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
        split.num_types = split.num_types.wrapping_add(1);
        xself.curr_histogram_ix_ = xself.curr_histogram_ix_.wrapping_add(1);
        if xself.curr_histogram_ix_ < *histograms_size {
            HistogramClear(&mut histograms[xself.curr_histogram_ix_]);
        }
        xself.block_size_ = 0usize;
    } else if xself.block_size_ > 0usize {
        let entropy = BitsEntropy(
            (histograms[xself.curr_histogram_ix_]).slice(),
            xself.alphabet_size_,
        );
        let mut combined_histo: [HistogramType; 2] = [
            histograms[xself.curr_histogram_ix_].clone(),
            histograms[xself.curr_histogram_ix_].clone(),
        ];

        let mut combined_entropy: [floatX; 2] = [0.0, 0.0];
        let mut diff: [floatX; 2] = [0.0, 0.0];
        for j in 0..2 {
            let last_histogram_ix: usize = xself.last_histogram_ix_[j];
            HistogramAddHistogram(&mut combined_histo[j], &histograms[last_histogram_ix]);
            combined_entropy[j] = BitsEntropy(
                &mut combined_histo[j].slice_mut()[0..],
                xself.alphabet_size_,
            );
            diff[j] = combined_entropy[j] - entropy - xself.last_entropy_[j];
        }
        if split.num_types < 256usize
            && (diff[0] > xself.split_threshold_)
            && (diff[1] > xself.split_threshold_)
        {
            split.lengths.slice_mut()[xself.num_blocks_] = xself.block_size_ as u32;
            split.types.slice_mut()[xself.num_blocks_] = split.num_types as u8;
            xself.last_histogram_ix_[1] = xself.last_histogram_ix_[0];
            xself.last_histogram_ix_[0] = split.num_types as u8 as usize;
            xself.last_entropy_[1] = xself.last_entropy_[0];
            xself.last_entropy_[0] = entropy;
            xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
            split.num_types = split.num_types.wrapping_add(1);
            xself.curr_histogram_ix_ = xself.curr_histogram_ix_.wrapping_add(1);
            if xself.curr_histogram_ix_ < *histograms_size {
                HistogramClear(&mut histograms[xself.curr_histogram_ix_]);
            }
            xself.block_size_ = 0usize;
            xself.merge_last_count_ = 0usize;
            xself.target_block_size_ = xself.min_block_size_;
        } else if diff[1] < diff[0] - 20.0 {
            split.lengths.slice_mut()[xself.num_blocks_] = xself.block_size_ as u32;
            split.types.slice_mut()[xself.num_blocks_] =
                split.types.slice()[xself.num_blocks_.wrapping_sub(2)]; //FIXME: investigate copy?
            {
                xself.last_histogram_ix_.swap(0, 1);
            }
            histograms[xself.last_histogram_ix_[0]] = combined_histo[1].clone();
            xself.last_entropy_[1] = xself.last_entropy_[0];
            xself.last_entropy_[0] = combined_entropy[1];
            xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
            xself.block_size_ = 0usize;
            HistogramClear(&mut histograms[xself.curr_histogram_ix_]);
            xself.merge_last_count_ = 0usize;
            xself.target_block_size_ = xself.min_block_size_;
        } else {
            {
                let _rhs = xself.block_size_ as u32;
                let _lhs = &mut split.lengths.slice_mut()[xself.num_blocks_.wrapping_sub(1)];
                *_lhs = (*_lhs).wrapping_add(_rhs);
            }
            histograms[xself.last_histogram_ix_[0]] = combined_histo[0].clone();
            xself.last_entropy_[0] = combined_entropy[0];
            if split.num_types == 1 {
                xself.last_entropy_[1] = xself.last_entropy_[0];
            }
            xself.block_size_ = 0usize;
            HistogramClear(&mut histograms[xself.curr_histogram_ix_]);
            if {
                xself.merge_last_count_ = xself.merge_last_count_.wrapping_add(1);
                xself.merge_last_count_
            } > 1
            {
                xself.target_block_size_ =
                    xself.target_block_size_.wrapping_add(xself.min_block_size_);
            }
        }
    }
    if is_final {
        *histograms_size = split.num_types;
        split.num_blocks = xself.num_blocks_;
    }
}
const BROTLI_MAX_STATIC_CONTEXTS: usize = 13;

fn ContextBlockSplitterFinishBlock<
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32> + alloc::Allocator<HistogramLiteral>,
    AllocHL: alloc::Allocator<HistogramLiteral>,
>(
    xself: &mut ContextBlockSplitter,
    m: &mut AllocHL,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut [HistogramLiteral],
    histograms_size: &mut usize,
    is_final: bool,
) {
    let num_contexts: usize = xself.num_contexts_;
    if xself.block_size_ < xself.min_block_size_ {
        xself.block_size_ = xself.min_block_size_;
    }
    if xself.num_blocks_ == 0usize {
        split.lengths.slice_mut()[0] = xself.block_size_ as u32;
        split.types.slice_mut()[0] = 0u8;
        for i in 0usize..num_contexts {
            xself.last_entropy_[i] = BitsEntropy((histograms[i]).slice(), xself.alphabet_size_);
            xself.last_entropy_[num_contexts.wrapping_add(i)] = xself.last_entropy_[i];
        }
        xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
        split.num_types = split.num_types.wrapping_add(1);
        xself.curr_histogram_ix_ = xself.curr_histogram_ix_.wrapping_add(num_contexts);
        if xself.curr_histogram_ix_ < *histograms_size {
            ClearHistograms(
                &mut histograms[xself.curr_histogram_ix_..],
                xself.num_contexts_,
            );
        }
        xself.block_size_ = 0usize;
    } else if xself.block_size_ > 0usize {
        let mut entropy = [0.0; BROTLI_MAX_STATIC_CONTEXTS];
        let mut combined_histo = m.alloc_cell(2 * num_contexts);
        let mut combined_entropy = [0.0; 2 * BROTLI_MAX_STATIC_CONTEXTS];
        let mut diff = [0.0; 2];
        for i in 0usize..num_contexts {
            let curr_histo_ix: usize = xself.curr_histogram_ix_.wrapping_add(i);
            let mut j: usize;
            entropy[i] = BitsEntropy((histograms[curr_histo_ix]).slice(), xself.alphabet_size_);
            j = 0usize;
            while j < 2usize {
                {
                    let jx: usize = j.wrapping_mul(num_contexts).wrapping_add(i);
                    let last_histogram_ix: usize = xself.last_histogram_ix_[j].wrapping_add(i);
                    combined_histo.slice_mut()[jx] = histograms[curr_histo_ix].clone();
                    HistogramAddHistogram(
                        &mut combined_histo.slice_mut()[jx],
                        &mut histograms[last_histogram_ix],
                    );
                    combined_entropy[jx] =
                        BitsEntropy(combined_histo.slice()[jx].slice(), xself.alphabet_size_);
                    diff[j] += combined_entropy[jx] - entropy[i] - xself.last_entropy_[jx];
                }
                j = j.wrapping_add(1);
            }
        }
        if split.num_types < xself.max_block_types_
            && (diff[0] > xself.split_threshold_)
            && (diff[1] > xself.split_threshold_)
        {
            split.lengths.slice_mut()[xself.num_blocks_] = xself.block_size_ as u32;
            split.types.slice_mut()[xself.num_blocks_] = split.num_types as u8;
            xself.last_histogram_ix_[1] = xself.last_histogram_ix_[0];
            xself.last_histogram_ix_[0] = split.num_types.wrapping_mul(num_contexts);
            for i in 0usize..num_contexts {
                xself.last_entropy_[num_contexts.wrapping_add(i)] = xself.last_entropy_[i];
                xself.last_entropy_[i] = entropy[i];
            }
            xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
            split.num_types = split.num_types.wrapping_add(1);
            xself.curr_histogram_ix_ = xself.curr_histogram_ix_.wrapping_add(num_contexts);
            if xself.curr_histogram_ix_ < *histograms_size {
                ClearHistograms(
                    &mut histograms[xself.curr_histogram_ix_..],
                    xself.num_contexts_,
                );
            }
            xself.block_size_ = 0usize;
            xself.merge_last_count_ = 0usize;
            xself.target_block_size_ = xself.min_block_size_;
        } else if diff[1] < diff[0] - 20.0 {
            split.lengths.slice_mut()[xself.num_blocks_] = xself.block_size_ as u32;
            let nbm2 = split.types.slice()[xself.num_blocks_.wrapping_sub(2)];
            split.types.slice_mut()[xself.num_blocks_] = nbm2;

            {
                xself.last_histogram_ix_.swap(0, 1);
            }
            for i in 0usize..num_contexts {
                histograms[xself.last_histogram_ix_[0].wrapping_add(i)] =
                    combined_histo.slice()[num_contexts.wrapping_add(i)].clone();
                xself.last_entropy_[num_contexts.wrapping_add(i)] = xself.last_entropy_[i];
                xself.last_entropy_[i] = combined_entropy[num_contexts.wrapping_add(i)];
                HistogramClear(&mut histograms[xself.curr_histogram_ix_.wrapping_add(i)]);
            }
            xself.num_blocks_ = xself.num_blocks_.wrapping_add(1);
            xself.block_size_ = 0usize;
            xself.merge_last_count_ = 0usize;
            xself.target_block_size_ = xself.min_block_size_;
        } else {
            {
                let _rhs = xself.block_size_ as u32;
                let _lhs = &mut split.lengths.slice_mut()[xself.num_blocks_.wrapping_sub(1)];
                let old_split_length = *_lhs;
                *_lhs = old_split_length.wrapping_add(_rhs);
            }
            for i in 0usize..num_contexts {
                histograms[xself.last_histogram_ix_[0].wrapping_add(i)] =
                    combined_histo.slice()[i].clone();
                xself.last_entropy_[i] = combined_entropy[i];
                if split.num_types == 1 {
                    xself.last_entropy_[num_contexts.wrapping_add(i)] = xself.last_entropy_[i];
                }
                HistogramClear(&mut histograms[xself.curr_histogram_ix_.wrapping_add(i)]);
            }
            xself.block_size_ = 0usize;
            if {
                xself.merge_last_count_ = xself.merge_last_count_.wrapping_add(1);
                xself.merge_last_count_
            } > 1
            {
                xself.target_block_size_ =
                    xself.target_block_size_.wrapping_add(xself.min_block_size_);
            }
        }
        m.free_cell(combined_histo);
    }
    if is_final {
        *histograms_size = split.num_types.wrapping_mul(num_contexts);
        split.num_blocks = xself.num_blocks_;
    }
}

fn BlockSplitterAddSymbol<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + Clone,
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32>,
>(
    xself: &mut BlockSplitter,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut [HistogramType],
    histograms_size: &mut usize,
    symbol: usize,
) {
    HistogramAddItem(&mut histograms[xself.curr_histogram_ix_], symbol);
    xself.block_size_ = xself.block_size_.wrapping_add(1);
    if xself.block_size_ == xself.target_block_size_ {
        BlockSplitterFinishBlock(xself, split, histograms, histograms_size, false);
    }
}

fn ContextBlockSplitterAddSymbol<
    Alloc: alloc::Allocator<u8> + alloc::Allocator<u32> + alloc::Allocator<HistogramLiteral>,
>(
    xself: &mut ContextBlockSplitter,
    m: &mut Alloc,
    split: &mut BlockSplit<Alloc>,
    histograms: &mut [HistogramLiteral],
    histograms_size: &mut usize,
    symbol: usize,
    context: usize,
) {
    HistogramAddItem(
        &mut histograms[xself.curr_histogram_ix_.wrapping_add(context)],
        symbol,
    );
    xself.block_size_ = xself.block_size_.wrapping_add(1);
    if xself.block_size_ == xself.target_block_size_ {
        ContextBlockSplitterFinishBlock(xself, m, split, histograms, histograms_size, false);
    }
}

fn MapStaticContexts<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>,
>(
    m32: &mut Alloc,
    num_contexts: usize,
    static_context_map: &[u32],
    mb: &mut MetaBlockSplit<Alloc>,
) {
    mb.literal_context_map_size = mb.literal_split.num_types << 6;
    let new_literal_context_map = allocate::<u32, _>(m32, mb.literal_context_map_size);
    <Alloc as Allocator<u32>>::free_cell(
        m32,
        core::mem::replace(&mut mb.literal_context_map, new_literal_context_map),
    );
    for i in 0usize..mb.literal_split.num_types {
        let offset: u32 = i.wrapping_mul(num_contexts) as u32;
        for j in 0usize..(1u32 << 6) as usize {
            mb.literal_context_map.slice_mut()[(i << 6).wrapping_add(j)] =
                offset.wrapping_add(static_context_map[j]);
        }
    }
}
pub fn BrotliBuildMetaBlockGreedyInternal<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>,
>(
    alloc: &mut Alloc,
    ringbuffer: &[u8],
    mut pos: usize,
    mask: usize,
    mut prev_byte: u8,
    mut prev_byte2: u8,
    literal_context_mode: ContextType,
    num_contexts: usize,
    static_context_map: &[u32],
    commands: &[Command],
    n_commands: usize,
    mb: &mut MetaBlockSplit<Alloc>,
) {
    let mut lit_blocks: LitBlocks;
    let mut cmd_blocks: BlockSplitter;
    let mut dist_blocks: BlockSplitter;
    let mut num_literals: usize = 0usize;
    for i in 0usize..n_commands {
        num_literals = num_literals.wrapping_add((commands[i]).insert_len_ as usize);
    }
    lit_blocks = if num_contexts == 1 {
        LitBlocks::plain(InitBlockSplitter::<HistogramLiteral, Alloc>(
            alloc,
            256usize,
            512usize,
            400.0,
            num_literals,
            &mut mb.literal_split,
            &mut mb.literal_histograms,
            &mut mb.literal_histograms_size,
        ))
    } else {
        LitBlocks::ctx(InitContextBlockSplitter::<Alloc>(
            alloc,
            256usize,
            num_contexts,
            512usize,
            400.0,
            num_literals,
            &mut mb.literal_split,
            &mut mb.literal_histograms,
            &mut mb.literal_histograms_size,
        ))
    };
    cmd_blocks = InitBlockSplitter::<HistogramCommand, Alloc>(
        alloc,
        704usize,
        1024usize,
        500.0,
        n_commands,
        &mut mb.command_split,
        &mut mb.command_histograms,
        &mut mb.command_histograms_size,
    );
    dist_blocks = InitBlockSplitter::<HistogramDistance, Alloc>(
        alloc,
        64usize,
        512usize,
        100.0,
        n_commands,
        &mut mb.distance_split,
        &mut mb.distance_histograms,
        &mut mb.distance_histograms_size,
    );

    for i in 0usize..n_commands {
        let cmd: Command = commands[i];
        let mut j: usize;
        BlockSplitterAddSymbol(
            &mut cmd_blocks,
            &mut mb.command_split,
            mb.command_histograms.slice_mut(),
            &mut mb.command_histograms_size,
            cmd.cmd_prefix_ as usize,
        );
        j = cmd.insert_len_ as usize;
        while j != 0usize {
            {
                let literal: u8 = ringbuffer[(pos & mask)];
                match (&mut lit_blocks) {
                    &mut LitBlocks::plain(ref mut lit_blocks_plain) => BlockSplitterAddSymbol(
                        lit_blocks_plain,
                        &mut mb.literal_split,
                        mb.literal_histograms.slice_mut(),
                        &mut mb.literal_histograms_size,
                        literal as usize,
                    ),
                    &mut LitBlocks::ctx(ref mut lit_blocks_ctx) => {
                        let context: usize =
                            Context(prev_byte, prev_byte2, literal_context_mode) as usize;
                        ContextBlockSplitterAddSymbol(
                            lit_blocks_ctx,
                            alloc,
                            &mut mb.literal_split,
                            mb.literal_histograms.slice_mut(),
                            &mut mb.literal_histograms_size,
                            literal as usize,
                            static_context_map[(context as usize)] as usize,
                        );
                    }
                }
                prev_byte2 = prev_byte;
                prev_byte = literal;
                pos = pos.wrapping_add(1);
            }
            j = j.wrapping_sub(1);
        }
        pos = pos.wrapping_add(cmd.copy_len() as usize);
        if cmd.copy_len() != 0 {
            prev_byte2 = ringbuffer[(pos.wrapping_sub(2) & mask)];
            prev_byte = ringbuffer[(pos.wrapping_sub(1) & mask)];
            if cmd.cmd_prefix_ as i32 >= 128i32 {
                BlockSplitterAddSymbol(
                    &mut dist_blocks,
                    &mut mb.distance_split,
                    mb.distance_histograms.slice_mut(),
                    &mut mb.distance_histograms_size,
                    cmd.dist_prefix_ as usize & 0x3ff,
                );
            }
        }
    }
    match (&mut lit_blocks) {
        &mut LitBlocks::plain(ref mut lit_blocks_plain) => BlockSplitterFinishBlock(
            lit_blocks_plain,
            &mut mb.literal_split,
            mb.literal_histograms.slice_mut(),
            &mut mb.literal_histograms_size,
            true,
        ),
        &mut LitBlocks::ctx(ref mut lit_blocks_ctx) => ContextBlockSplitterFinishBlock(
            lit_blocks_ctx,
            alloc,
            &mut mb.literal_split,
            mb.literal_histograms.slice_mut(),
            &mut mb.literal_histograms_size,
            true,
        ),
    }
    BlockSplitterFinishBlock(
        &mut cmd_blocks,
        &mut mb.command_split,
        mb.command_histograms.slice_mut(),
        &mut mb.command_histograms_size,
        true,
    );
    BlockSplitterFinishBlock(
        &mut dist_blocks,
        &mut mb.distance_split,
        mb.distance_histograms.slice_mut(),
        &mut mb.distance_histograms_size,
        true,
    );
    if num_contexts > 1 {
        MapStaticContexts(alloc, num_contexts, static_context_map, mb);
    }
}
pub fn BrotliBuildMetaBlockGreedy<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>,
>(
    alloc: &mut Alloc,
    ringbuffer: &[u8],
    pos: usize,
    mask: usize,
    prev_byte: u8,
    prev_byte2: u8,
    literal_context_mode: ContextType,
    _literal_context_lut: &[u8],
    num_contexts: usize,
    static_context_map: &[u32],
    commands: &[Command],
    n_commands: usize,
    mb: &mut MetaBlockSplit<Alloc>,
) {
    if num_contexts == 1 {
        BrotliBuildMetaBlockGreedyInternal(
            alloc,
            ringbuffer,
            pos,
            mask,
            prev_byte,
            prev_byte2,
            literal_context_mode,
            1,
            &[],
            commands,
            n_commands,
            mb,
        );
    } else {
        BrotliBuildMetaBlockGreedyInternal(
            alloc,
            ringbuffer,
            pos,
            mask,
            prev_byte,
            prev_byte2,
            literal_context_mode,
            num_contexts,
            static_context_map,
            commands,
            n_commands,
            mb,
        );
    }
}

pub fn BrotliOptimizeHistograms<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>,
>(
    num_distance_codes: usize,
    mb: &mut MetaBlockSplit<Alloc>,
) {
    let mut good_for_rle: [u8; 704] = [0; 704];
    for i in 0usize..mb.literal_histograms_size {
        BrotliOptimizeHuffmanCountsForRle(
            256usize,
            mb.literal_histograms.slice_mut()[i].slice_mut(),
            &mut good_for_rle[..],
        );
    }
    for i in 0usize..mb.command_histograms_size {
        BrotliOptimizeHuffmanCountsForRle(
            704usize,
            mb.command_histograms.slice_mut()[i].slice_mut(),
            &mut good_for_rle[..],
        );
    }
    for i in 0usize..mb.distance_histograms_size {
        BrotliOptimizeHuffmanCountsForRle(
            num_distance_codes,
            mb.distance_histograms.slice_mut()[i].slice_mut(),
            &mut good_for_rle[..],
        );
    }
}
