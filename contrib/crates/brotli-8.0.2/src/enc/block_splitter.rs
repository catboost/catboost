use core;
use core::cmp::{max, min};
#[cfg(feature = "simd")]
use core::simd::prelude::{SimdFloat, SimdPartialOrd};

use super::super::alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use super::backward_references::BrotliEncoderParams;
use super::bit_cost::BrotliPopulationCost;
use super::block_split::BlockSplit;
use super::cluster::{BrotliHistogramBitCostDistance, BrotliHistogramCombine, HistogramPair};
use super::command::Command;
use super::histogram::{
    ClearHistograms, CostAccessors, HistogramAddHistogram, HistogramAddItem, HistogramAddVector,
    HistogramClear, HistogramCommand, HistogramDistance, HistogramLiteral,
};
use super::util::FastLog2;
use super::vectorization::{sum8i, v256, v256i, Mem256f};
use crate::enc::combined_alloc::allocate;
use crate::enc::floatX;

static kMaxLiteralHistograms: usize = 100usize;

static kMaxCommandHistograms: usize = 50usize;

static kLiteralBlockSwitchCost: floatX = 28.1;

static kCommandBlockSwitchCost: floatX = 13.5;

static kDistanceBlockSwitchCost: floatX = 14.6;

static kLiteralStrideLength: usize = 70usize;

static kCommandStrideLength: usize = 40usize;

static kSymbolsPerLiteralHistogram: usize = 544usize;

static kSymbolsPerCommandHistogram: usize = 530usize;

static kSymbolsPerDistanceHistogram: usize = 544usize;

static kMinLengthForBlockSplitting: usize = 128usize;

static kIterMulForRefining: usize = 2usize;

static kMinItersForRefining: usize = 100usize;

#[inline(always)]
fn update_cost_and_signal(
    num_histograms32: u32,
    ix: usize,
    min_cost: floatX,
    block_switch_cost: floatX,
    cost: &mut [Mem256f],
    switch_signal: &mut [u8],
) {
    let ymm_min_cost = v256::splat(min_cost);
    let ymm_block_switch_cost = v256::splat(block_switch_cost);
    let ymm_and_mask = v256i::from([
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
    ]);

    for (index, cost_it) in cost[..((num_histograms32 as usize + 7) >> 3)]
        .iter_mut()
        .enumerate()
    {
        let mut ymm_cost = *cost_it;
        let costk_minus_min_cost = ymm_cost - ymm_min_cost;
        let ymm_cmpge: v256i = costk_minus_min_cost.simd_ge(ymm_block_switch_cost).to_int();
        let ymm_bits = ymm_cmpge & ymm_and_mask;
        let result = sum8i(ymm_bits);
        //super::vectorization::sum8(ymm_bits) as u8;
        switch_signal[ix + index] |= result as u8;
        ymm_cost = costk_minus_min_cost.simd_min(ymm_block_switch_cost);
        *cost_it = Mem256f::from(ymm_cost);
        //println_stderr!("{:} ss {:} c {:?}", (index << 3) + 7, switch_signal[ix + index],*cost_it);
    }
}
fn CountLiterals(cmds: &[Command], num_commands: usize) -> usize {
    let mut total_length: usize = 0usize;
    for i in 0usize..num_commands {
        total_length = total_length.wrapping_add((cmds[i]).insert_len_ as usize);
    }
    total_length
}

fn CopyLiteralsToByteArray(
    cmds: &[Command],
    num_commands: usize,
    data: &[u8],
    offset: usize,
    mask: usize,
    literals: &mut [u8],
) {
    let mut pos: usize = 0usize;
    let mut from_pos: usize = offset & mask;
    for i in 0usize..num_commands {
        let mut insert_len: usize = (cmds[i]).insert_len_ as usize;
        if from_pos.wrapping_add(insert_len) > mask {
            let head_size: usize = mask.wrapping_add(1).wrapping_sub(from_pos);
            literals[pos..(pos + head_size)]
                .clone_from_slice(&data[from_pos..(from_pos + head_size)]);
            from_pos = 0usize;
            pos = pos.wrapping_add(head_size);
            insert_len = insert_len.wrapping_sub(head_size);
        }
        if insert_len > 0usize {
            literals[pos..(pos + insert_len)]
                .clone_from_slice(&data[from_pos..(from_pos + insert_len)]);
            pos = pos.wrapping_add(insert_len);
        }
        from_pos = from_pos
            .wrapping_add(insert_len)
            .wrapping_add(cmds[i].copy_len() as usize)
            & mask;
    }
}

fn MyRand(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(16807);
    if *seed == 0u32 {
        *seed = 1u32;
    }
    *seed
}

fn InitialEntropyCodes<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    stride: usize,
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    let mut seed: u32 = 7u32;
    let block_length: usize = length.wrapping_div(num_histograms);
    ClearHistograms(histograms, num_histograms);
    for i in 0usize..num_histograms {
        let mut pos: usize = length.wrapping_mul(i).wrapping_div(num_histograms);
        if i != 0usize {
            pos = pos.wrapping_add((MyRand(&mut seed) as usize).wrapping_rem(block_length));
        }
        if pos.wrapping_add(stride) >= length {
            pos = length.wrapping_sub(stride).wrapping_sub(1);
        }
        HistogramAddVector(&mut histograms[i], &data[pos..], stride);
    }
}

fn RandomSample<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    seed: &mut u32,
    data: &[IntegerType],
    length: usize,
    mut stride: usize,
    sample: &mut HistogramType,
) where
    u64: core::convert::From<IntegerType>,
{
    let pos: usize;
    if stride >= length {
        pos = 0usize;
        stride = length;
    } else {
        pos = (MyRand(seed) as usize).wrapping_rem(length.wrapping_sub(stride).wrapping_add(1));
    }
    HistogramAddVector(sample, &data[pos..], stride);
}

fn RefineEntropyCodes<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    stride: usize,
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    let mut iters: usize = kIterMulForRefining
        .wrapping_mul(length)
        .wrapping_div(stride)
        .wrapping_add(kMinItersForRefining);
    let mut seed: u32 = 7u32;
    iters = iters
        .wrapping_add(num_histograms)
        .wrapping_sub(1)
        .wrapping_div(num_histograms)
        .wrapping_mul(num_histograms);
    for iter in 0usize..iters {
        let mut sample = HistogramType::default();
        HistogramClear(&mut sample);
        RandomSample(&mut seed, data, length, stride, &mut sample);
        HistogramAddHistogram(
            &mut histograms[iter.wrapping_rem(num_histograms)],
            &mut sample,
        );
    }
}

fn BitCost(count: usize) -> floatX {
    if count == 0usize {
        -2.0
    } else {
        FastLog2(count as u64)
    }
}

fn FindBlocks<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    block_switch_bitcost: floatX,
    num_histograms: usize,
    histograms: &[HistogramType],
    insert_cost: &mut [floatX],
    cost: &mut [Mem256f],
    switch_signal: &mut [u8],
    block_id: &mut [u8],
) -> usize
where
    u64: core::convert::From<IntegerType>,
{
    if num_histograms == 0 {
        return 0;
    }
    let data_size: usize = histograms[0].slice().len();
    let bitmaplen: usize = num_histograms.wrapping_add(7) >> 3;
    let mut num_blocks: usize = 1;
    let mut i: usize;
    if num_histograms <= 1 {
        for i in 0usize..length {
            block_id[i] = 0u8;
        }
        return 1;
    }
    for item in insert_cost[..(data_size * num_histograms)].iter_mut() {
        *item = 0.0;
    }
    for i in 0usize..num_histograms {
        insert_cost[i] = FastLog2((histograms[i]).total_count() as u32 as (u64));
    }
    i = data_size;
    while i != 0usize {
        i = i.wrapping_sub(1);
        for j in 0usize..num_histograms {
            insert_cost[i.wrapping_mul(num_histograms).wrapping_add(j)] =
                insert_cost[j] - BitCost((histograms[j]).slice()[i] as usize);
        }
    }
    for item in cost.iter_mut() {
        *item = Mem256f::default();
    }
    for item in switch_signal[..(length * bitmaplen)].iter_mut() {
        *item = 0;
    }
    for (byte_ix, data_byte_ix) in data[..length].iter().enumerate() {
        let block_id_ptr = &mut block_id[byte_ix];
        let ix: usize = byte_ix.wrapping_mul(bitmaplen);
        let insert_cost_ix: usize =
            u64::from(data_byte_ix.clone()).wrapping_mul(num_histograms as u64) as usize;
        let mut min_cost: floatX = 1e38;
        let mut block_switch_cost: floatX = block_switch_bitcost;
        // main (vectorized) loop
        let insert_cost_slice = insert_cost.split_at(insert_cost_ix).1;
        for (v_index, cost_iter) in cost
            .split_at_mut(num_histograms >> 3)
            .0
            .iter_mut()
            .enumerate()
        {
            let base_index = v_index << 3;
            let mut local_insert_cost = [0.0; 8];
            local_insert_cost
                .clone_from_slice(insert_cost_slice.split_at(base_index).1.split_at(8).0);
            for sub_index in 0usize..8usize {
                cost_iter[sub_index] += local_insert_cost[sub_index];
                let final_cost = cost_iter[sub_index];
                if final_cost < min_cost {
                    min_cost = final_cost;
                    *block_id_ptr = (base_index + sub_index) as u8;
                }
            }
        }
        let vectorized_offset = ((num_histograms >> 3) << 3);
        let mut k = vectorized_offset;
        //remainder loop for
        for insert_cost_iter in insert_cost
            .split_at(insert_cost_ix + vectorized_offset)
            .1
            .split_at(num_histograms & 7)
            .0
            .iter()
        {
            let cost_iter = &mut cost[(k >> 3)];
            cost_iter[k & 7] += *insert_cost_iter;
            if cost_iter[k & 7] < min_cost {
                min_cost = cost_iter[k & 7];
                *block_id_ptr = k as u8;
            }
            k += 1;
        }
        if byte_ix < 2000usize {
            block_switch_cost *= (0.77 + 0.07 * (byte_ix as floatX) / 2000.0);
        }
        update_cost_and_signal(
            num_histograms as u32,
            ix,
            min_cost,
            block_switch_cost,
            cost,
            switch_signal,
        );
    }
    {
        let mut byte_ix: usize = length.wrapping_sub(1);
        let mut ix: usize = byte_ix.wrapping_mul(bitmaplen);
        let mut cur_id: u8 = block_id[byte_ix];
        while byte_ix > 0usize {
            let mask: u8 = (1u32 << (cur_id as i32 & 7i32)) as u8;
            byte_ix -= 1;
            ix = ix.wrapping_sub(bitmaplen);
            if switch_signal[ix.wrapping_add((cur_id as i32 >> 3) as usize)] as i32 & mask as i32
                != 0
                && cur_id as i32 != block_id[byte_ix] as i32
            {
                cur_id = block_id[byte_ix];
                num_blocks = num_blocks.wrapping_add(1);
            }
            block_id[byte_ix] = cur_id;
        }
    }
    num_blocks
}

fn RemapBlockIds(
    block_ids: &mut [u8],
    length: usize,
    new_id: &mut [u16],
    num_histograms: usize,
) -> usize {
    static kInvalidId: u16 = 256u16;
    let mut next_id: u16 = 0u16;
    for i in 0usize..num_histograms {
        new_id[i] = kInvalidId;
    }
    for i in 0usize..length {
        if new_id[(block_ids[i] as usize)] as i32 == kInvalidId as i32 {
            new_id[(block_ids[i] as usize)] = {
                let _old = next_id;
                next_id = (next_id as i32 + 1) as u16;
                _old
            };
        }
    }
    for i in 0usize..length {
        block_ids[i] = new_id[(block_ids[i] as usize)] as u8;
    }
    next_id as usize
}

fn BuildBlockHistograms<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors,
    IntegerType: Sized + Clone,
>(
    data: &[IntegerType],
    length: usize,
    block_ids: &[u8],
    num_histograms: usize,
    histograms: &mut [HistogramType],
) where
    u64: core::convert::From<IntegerType>,
{
    ClearHistograms(histograms, num_histograms);
    for i in 0usize..length {
        HistogramAddItem(
            &mut histograms[(block_ids[i] as usize)],
            u64::from(data[i].clone()) as usize,
        );
    }
}

fn ClusterBlocks<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default + Clone,
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u32>
        + alloc::Allocator<HistogramType>
        + alloc::Allocator<HistogramPair>,
    IntegerType: Sized + Clone,
>(
    alloc: &mut Alloc,
    data: &[IntegerType],
    length: usize,
    num_blocks: usize,
    scratch_space: &mut HistogramType::i32vec,
    block_ids: &mut [u8],
    split: &mut BlockSplit<Alloc>,
) where
    u64: core::convert::From<IntegerType>,
{
    let mut histogram_symbols = allocate::<u32, _>(alloc, num_blocks);
    let mut block_lengths = allocate::<u32, _>(alloc, num_blocks);
    let expected_num_clusters: usize = (16usize)
        .wrapping_mul(num_blocks.wrapping_add(64).wrapping_sub(1))
        .wrapping_div(64);
    let mut all_histograms_size: usize = 0usize;
    let mut all_histograms_capacity: usize = expected_num_clusters;
    let mut all_histograms = allocate::<HistogramType, _>(alloc, all_histograms_capacity);
    let mut cluster_size_size: usize = 0usize;
    let mut cluster_size_capacity: usize = expected_num_clusters;
    let mut cluster_size = allocate::<u32, _>(alloc, cluster_size_capacity);
    let mut num_clusters: usize = 0usize;
    let mut histograms = allocate::<HistogramType, _>(alloc, min(num_blocks, 64));
    let mut max_num_pairs: usize = (64i32 * 64i32 / 2i32) as usize;
    let pairs_capacity: usize = max_num_pairs.wrapping_add(1);
    let mut pairs = allocate::<HistogramPair, _>(alloc, pairs_capacity);
    let mut pos: usize = 0usize;
    let mut clusters: <Alloc as Allocator<u32>>::AllocatedMemory;

    static kInvalidIndex: u32 = u32::MAX;
    let mut i: usize;
    let mut sizes: [u32; 64] = [0; 64];
    let mut new_clusters: [u32; 64] = [0; 64];
    let mut symbols: [u32; 64] = [0; 64];
    let mut remap: [u32; 64] = [0; 64];
    {
        let mut block_idx: usize = 0usize;
        i = 0usize;
        while i < length {
            {
                {
                    let _rhs = 1;
                    let _lhs = &mut block_lengths.slice_mut()[block_idx];
                    *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                }
                if i.wrapping_add(1) == length
                    || block_ids[i] as i32 != block_ids[i.wrapping_add(1)] as i32
                {
                    block_idx = block_idx.wrapping_add(1);
                }
            }
            i = i.wrapping_add(1);
        }
    }
    i = 0usize;
    while i < num_blocks {
        {
            let num_to_combine: usize = min(num_blocks.wrapping_sub(i), 64);

            for j in 0usize..num_to_combine {
                HistogramClear(&mut histograms.slice_mut()[j]);
                for _k in 0usize..block_lengths.slice()[i.wrapping_add(j)] as usize {
                    HistogramAddItem(
                        &mut histograms.slice_mut()[j],
                        u64::from(data[pos].clone()) as usize,
                    );
                    pos = pos.wrapping_add(1);
                }
                let new_cost = BrotliPopulationCost(&histograms.slice()[j], scratch_space);
                (histograms.slice_mut()[j]).set_bit_cost(new_cost);

                new_clusters[j] = j as u32;
                symbols[j] = j as u32;
                sizes[j] = 1u32;
            }
            let num_new_clusters: usize = BrotliHistogramCombine(
                histograms.slice_mut(),
                &mut sizes[..],
                &mut symbols[..],
                &mut new_clusters[..],
                pairs.slice_mut(),
                num_to_combine,
                num_to_combine,
                64usize,
                max_num_pairs,
                scratch_space,
            );
            {
                if all_histograms_capacity < all_histograms_size.wrapping_add(num_new_clusters) {
                    let mut _new_size: usize = if all_histograms_capacity == 0usize {
                        all_histograms_size.wrapping_add(num_new_clusters)
                    } else {
                        all_histograms_capacity
                    };
                    while _new_size < all_histograms_size.wrapping_add(num_new_clusters) {
                        _new_size = _new_size.wrapping_mul(2);
                    }
                    let mut new_array = allocate::<HistogramType, _>(alloc, _new_size);
                    new_array.slice_mut()[..all_histograms_capacity]
                        .clone_from_slice(&all_histograms.slice()[..all_histograms_capacity]);
                    <Alloc as Allocator<HistogramType>>::free_cell(
                        alloc,
                        core::mem::replace(&mut all_histograms, new_array),
                    );
                    all_histograms_capacity = _new_size;
                }
            }
            {
                if cluster_size_capacity < cluster_size_size.wrapping_add(num_new_clusters) {
                    let mut _new_size: usize = if cluster_size_capacity == 0usize {
                        cluster_size_size.wrapping_add(num_new_clusters)
                    } else {
                        cluster_size_capacity
                    };
                    while _new_size < cluster_size_size.wrapping_add(num_new_clusters) {
                        _new_size = _new_size.wrapping_mul(2);
                    }
                    let mut new_array = allocate::<u32, _>(alloc, _new_size);
                    new_array.slice_mut()[..cluster_size_capacity]
                        .clone_from_slice(&cluster_size.slice()[..cluster_size_capacity]);
                    <Alloc as Allocator<u32>>::free_cell(
                        alloc,
                        core::mem::replace(&mut cluster_size, new_array),
                    );
                    cluster_size_capacity = _new_size;
                }
            }
            for j in 0usize..num_new_clusters {
                all_histograms.slice_mut()[all_histograms_size] =
                    histograms.slice()[new_clusters[j] as usize].clone();
                all_histograms_size = all_histograms_size.wrapping_add(1);
                cluster_size.slice_mut()[cluster_size_size] = sizes[new_clusters[j] as usize];
                cluster_size_size = cluster_size_size.wrapping_add(1);
                remap[new_clusters[j] as usize] = j as u32;
            }
            for j in 0usize..num_to_combine {
                histogram_symbols.slice_mut()[i.wrapping_add(j)] =
                    (num_clusters as u32).wrapping_add(remap[symbols[j] as usize]);
            }
            num_clusters = num_clusters.wrapping_add(num_new_clusters);
        }
        i = i.wrapping_add(64);
    }
    <Alloc as Allocator<HistogramType>>::free_cell(alloc, core::mem::take(&mut histograms));
    max_num_pairs = min(
        (64usize).wrapping_mul(num_clusters),
        num_clusters.wrapping_div(2).wrapping_mul(num_clusters),
    );
    if pairs_capacity < max_num_pairs.wrapping_add(1) {
        let new_cell = allocate::<HistogramPair, _>(alloc, max_num_pairs.wrapping_add(1));
        <Alloc as Allocator<HistogramPair>>::free_cell(
            alloc,
            core::mem::replace(&mut pairs, new_cell),
        );
    }
    clusters = allocate::<u32, _>(alloc, num_clusters);
    i = 0usize;
    for item in clusters.slice_mut()[..num_clusters].iter_mut() {
        *item = i as u32;
        i = i.wrapping_add(1);
    }
    let num_final_clusters: usize = BrotliHistogramCombine(
        all_histograms.slice_mut(),
        cluster_size.slice_mut(),
        histogram_symbols.slice_mut(),
        clusters.slice_mut(),
        pairs.slice_mut(),
        num_clusters,
        num_blocks,
        256usize,
        max_num_pairs,
        scratch_space,
    );
    <Alloc as Allocator<HistogramPair>>::free_cell(alloc, core::mem::take(&mut pairs));
    <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut cluster_size));

    let mut new_index = allocate::<u32, _>(alloc, num_clusters);
    for item in new_index.slice_mut().iter_mut() {
        *item = kInvalidIndex;
    }
    pos = 0usize;
    {
        let mut next_index: u32 = 0u32;
        for i in 0usize..num_blocks {
            let mut histo: HistogramType = HistogramType::default();
            let mut best_out: u32;
            let mut best_bits: floatX;
            HistogramClear(&mut histo);
            for _j in 0usize..block_lengths.slice()[i] as usize {
                HistogramAddItem(&mut histo, u64::from(data[pos].clone()) as usize);
                pos = pos.wrapping_add(1);
            }
            best_out = if i == 0usize {
                histogram_symbols.slice()[0]
            } else {
                histogram_symbols.slice()[i.wrapping_sub(1)]
            };
            best_bits = BrotliHistogramBitCostDistance(
                &mut histo,
                &mut all_histograms.slice_mut()[(best_out as usize)],
                scratch_space,
            );
            for j in 0usize..num_final_clusters {
                let cur_bits: floatX = BrotliHistogramBitCostDistance(
                    &mut histo,
                    &mut all_histograms.slice_mut()[(clusters.slice()[j] as usize)],
                    scratch_space,
                );
                if cur_bits < best_bits {
                    best_bits = cur_bits;
                    best_out = clusters.slice()[j];
                }
            }
            histogram_symbols.slice_mut()[i] = best_out;
            if new_index.slice()[best_out as usize] == kInvalidIndex {
                new_index.slice_mut()[best_out as usize] = next_index;
                next_index = next_index.wrapping_add(1);
            }
        }
    }
    <Alloc as Allocator<u32>>::free_cell(alloc, core::mem::take(&mut clusters));
    <Alloc as Allocator<HistogramType>>::free_cell(alloc, core::mem::take(&mut all_histograms));
    {
        if split.types_alloc_size() < num_blocks {
            let mut _new_size: usize = if split.types_alloc_size() == 0usize {
                num_blocks
            } else {
                split.types_alloc_size()
            };
            while _new_size < num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u8, _>(alloc, _new_size);
            new_array.slice_mut()[..split.types_alloc_size()]
                .clone_from_slice(&split.types.slice()[..split.types_alloc_size()]);
            <Alloc as Allocator<u8>>::free_cell(
                alloc,
                core::mem::replace(&mut split.types, new_array),
            );
        }
    }
    {
        if split.lengths_alloc_size() < num_blocks {
            let mut _new_size: usize = if split.lengths_alloc_size() == 0usize {
                num_blocks
            } else {
                split.lengths_alloc_size()
            };
            while _new_size < num_blocks {
                _new_size = _new_size.wrapping_mul(2);
            }
            let mut new_array = allocate::<u32, _>(alloc, _new_size);
            new_array.slice_mut()[..split.lengths_alloc_size()]
                .clone_from_slice(split.lengths.slice());
            <Alloc as Allocator<u32>>::free_cell(
                alloc,
                core::mem::replace(&mut split.lengths, new_array),
            );
        }
    }
    {
        let mut cur_length: u32 = 0u32;
        let mut block_idx: usize = 0usize;
        let mut max_type: u8 = 0u8;
        for i in 0usize..num_blocks {
            cur_length = cur_length.wrapping_add(block_lengths.slice()[i]);
            if i.wrapping_add(1) == num_blocks
                || histogram_symbols.slice()[i] != histogram_symbols.slice()[i.wrapping_add(1)]
            {
                let id: u8 = new_index.slice()[(histogram_symbols.slice()[i] as usize)] as u8;
                split.types.slice_mut()[block_idx] = id;
                split.lengths.slice_mut()[block_idx] = cur_length;
                max_type = max(max_type, id);
                cur_length = 0u32;
                block_idx = block_idx.wrapping_add(1);
            }
        }
        split.num_blocks = block_idx;
        split.num_types = (max_type as usize).wrapping_add(1);
    }
    <Alloc as Allocator<u32>>::free_cell(alloc, new_index);
    <Alloc as Allocator<u32>>::free_cell(alloc, block_lengths);
    <Alloc as Allocator<u32>>::free_cell(alloc, histogram_symbols);
}

fn SplitByteVector<
    HistogramType: SliceWrapper<u32> + SliceWrapperMut<u32> + CostAccessors + core::default::Default + Clone,
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u16>
        + alloc::Allocator<u32>
        + alloc::Allocator<floatX>
        + alloc::Allocator<Mem256f>
        + alloc::Allocator<HistogramType>
        + alloc::Allocator<HistogramPair>,
    IntegerType: Sized + Clone,
>(
    alloc: &mut Alloc,
    data: &[IntegerType],
    length: usize,
    literals_per_histogram: usize,
    max_histograms: usize,
    sampling_stride_length: usize,
    block_switch_cost: floatX,
    params: &BrotliEncoderParams,
    scratch_space: &mut HistogramType::i32vec,
    split: &mut BlockSplit<Alloc>,
) where
    u64: core::convert::From<IntegerType>,
{
    let data_size: usize = HistogramType::default().slice().len();
    let mut num_histograms: usize = length.wrapping_div(literals_per_histogram).wrapping_add(1);
    if num_histograms > max_histograms {
        num_histograms = max_histograms;
    }
    if length == 0usize {
        split.num_types = 1;
        return;
    } else if length < kMinLengthForBlockSplitting {
        {
            if split.types_alloc_size() < split.num_blocks.wrapping_add(1) {
                let mut _new_size: usize = if split.types_alloc_size() == 0usize {
                    split.num_blocks.wrapping_add(1)
                } else {
                    split.types_alloc_size()
                };

                while _new_size < split.num_blocks.wrapping_add(1) {
                    _new_size = _new_size.wrapping_mul(2);
                }
                let mut new_array = allocate::<u8, _>(alloc, _new_size);
                new_array.slice_mut()[..split.types_alloc_size()]
                    .clone_from_slice(&split.types.slice()[..split.types_alloc_size()]);
                <Alloc as Allocator<u8>>::free_cell(
                    alloc,
                    core::mem::replace(&mut split.types, new_array),
                );
            }
        }
        {
            if split.lengths_alloc_size() < split.num_blocks.wrapping_add(1) {
                let mut _new_size: usize = if split.lengths_alloc_size() == 0usize {
                    split.num_blocks.wrapping_add(1)
                } else {
                    split.lengths_alloc_size()
                };
                while _new_size < split.num_blocks.wrapping_add(1) {
                    _new_size = _new_size.wrapping_mul(2);
                }
                let mut new_array = allocate::<u32, _>(alloc, _new_size);
                new_array.slice_mut()[..split.lengths_alloc_size()]
                    .clone_from_slice(&split.lengths.slice()[..split.lengths_alloc_size()]);
                <Alloc as Allocator<u32>>::free_cell(
                    alloc,
                    core::mem::replace(&mut split.lengths, new_array),
                );
            }
        }
        split.num_types = 1;
        split.types.slice_mut()[split.num_blocks] = 0u8;
        split.lengths.slice_mut()[split.num_blocks] = length as u32;
        split.num_blocks = split.num_blocks.wrapping_add(1);
        return;
    }
    let mut histograms = allocate::<HistogramType, _>(alloc, num_histograms);

    InitialEntropyCodes(
        data,
        length,
        sampling_stride_length,
        num_histograms,
        histograms.slice_mut(),
    );
    RefineEntropyCodes(
        data,
        length,
        sampling_stride_length,
        num_histograms,
        histograms.slice_mut(),
    );
    {
        let mut block_ids = allocate::<u8, _>(alloc, length);
        let mut num_blocks: usize = 0usize;
        let bitmaplen: usize = num_histograms.wrapping_add(7) >> 3;
        let mut insert_cost = allocate::<floatX, _>(alloc, data_size.wrapping_mul(num_histograms));
        let mut cost = allocate::<Mem256f, _>(alloc, ((num_histograms + 7) >> 3));
        let mut switch_signal = allocate::<u8, _>(alloc, length.wrapping_mul(bitmaplen));
        let mut new_id = allocate::<u16, _>(alloc, num_histograms);
        let iters: usize = (if params.quality <= 11 { 3i32 } else { 10i32 }) as usize;
        for _i in 0usize..iters {
            num_blocks = FindBlocks(
                data,
                length,
                block_switch_cost,
                num_histograms,
                histograms.slice_mut(),
                insert_cost.slice_mut(),
                cost.slice_mut(),
                switch_signal.slice_mut(),
                block_ids.slice_mut(),
            );
            num_histograms = RemapBlockIds(
                block_ids.slice_mut(),
                length,
                new_id.slice_mut(),
                num_histograms,
            );
            BuildBlockHistograms(
                data,
                length,
                block_ids.slice(),
                num_histograms,
                histograms.slice_mut(),
            );
        }
        <Alloc as Allocator<floatX>>::free_cell(alloc, insert_cost);
        <Alloc as Allocator<Mem256f>>::free_cell(alloc, cost);
        <Alloc as Allocator<u8>>::free_cell(alloc, switch_signal);
        <Alloc as Allocator<u16>>::free_cell(alloc, new_id);
        <Alloc as Allocator<HistogramType>>::free_cell(alloc, histograms);
        ClusterBlocks::<HistogramType, Alloc, IntegerType>(
            alloc,
            data,
            length,
            num_blocks,
            scratch_space,
            block_ids.slice_mut(),
            split,
        );
        <Alloc as Allocator<u8>>::free_cell(alloc, block_ids);
    }
}

pub fn BrotliSplitBlock<
    Alloc: alloc::Allocator<u8>
        + alloc::Allocator<u16>
        + alloc::Allocator<u32>
        + alloc::Allocator<floatX>
        + alloc::Allocator<Mem256f>
        + alloc::Allocator<HistogramLiteral>
        + alloc::Allocator<HistogramCommand>
        + alloc::Allocator<HistogramDistance>
        + alloc::Allocator<HistogramPair>,
>(
    alloc: &mut Alloc,
    cmds: &[Command],
    num_commands: usize,
    data: &[u8],
    pos: usize,
    mask: usize,
    params: &BrotliEncoderParams,
    lit_scratch_space: &mut <HistogramLiteral as CostAccessors>::i32vec,
    cmd_scratch_space: &mut <HistogramCommand as CostAccessors>::i32vec,
    dst_scratch_space: &mut <HistogramDistance as CostAccessors>::i32vec,
    literal_split: &mut BlockSplit<Alloc>,
    insert_and_copy_split: &mut BlockSplit<Alloc>,
    dist_split: &mut BlockSplit<Alloc>,
) {
    {
        /*for (i, cmd) in cmds[..num_commands].iter().enumerate() {
            println_stderr!("C {:} {:} {:} {:} {:} {:}",
                            i, cmd.insert_len_, cmd.copy_len_, cmd.dist_extra_, cmd.cmd_prefix_, cmd.dist_prefix_);
        }*/
        let literals_count: usize = CountLiterals(cmds, num_commands);
        let mut literals = allocate::<u8, _>(alloc, literals_count);
        CopyLiteralsToByteArray(cmds, num_commands, data, pos, mask, literals.slice_mut());
        SplitByteVector::<HistogramLiteral, Alloc, u8>(
            alloc,
            literals.slice(),
            literals_count,
            kSymbolsPerLiteralHistogram,
            kMaxLiteralHistograms,
            kLiteralStrideLength,
            kLiteralBlockSwitchCost,
            params,
            lit_scratch_space,
            literal_split,
        );
        <Alloc as Allocator<u8>>::free_cell(alloc, literals);
    }
    {
        let mut insert_and_copy_codes = allocate::<u16, _>(alloc, num_commands);
        for i in 0..min(num_commands, cmds.len()) {
            insert_and_copy_codes.slice_mut()[i] = (cmds[i]).cmd_prefix_;
        }
        SplitByteVector::<HistogramCommand, Alloc, u16>(
            alloc,
            insert_and_copy_codes.slice(),
            num_commands,
            kSymbolsPerCommandHistogram,
            kMaxCommandHistograms,
            kCommandStrideLength,
            kCommandBlockSwitchCost,
            params,
            cmd_scratch_space,
            insert_and_copy_split,
        );
        <Alloc as Allocator<u16>>::free_cell(alloc, insert_and_copy_codes);
    }
    {
        let mut distance_prefixes = allocate::<u16, _>(alloc, num_commands);
        let mut j: usize = 0usize;
        for i in 0usize..num_commands {
            let cmd = &cmds[i];
            if cmd.copy_len() != 0 && cmd.cmd_prefix_ >= 128 {
                distance_prefixes.slice_mut()[j] = cmd.dist_prefix_ & 0x03ff;
                j = j.wrapping_add(1);
            }
        }
        SplitByteVector::<HistogramDistance, Alloc, u16>(
            alloc,
            distance_prefixes.slice(),
            j,
            kSymbolsPerDistanceHistogram,
            kMaxCommandHistograms,
            kCommandStrideLength,
            kDistanceBlockSwitchCost,
            params,
            dst_scratch_space,
            dist_split,
        );
        <Alloc as Allocator<u16>>::free_cell(alloc, distance_prefixes);
    }
}
