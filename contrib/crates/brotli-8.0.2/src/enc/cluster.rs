use alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use core::cmp::min;

use {alloc, core};

use super::bit_cost::BrotliPopulationCost;
use super::histogram::{
    CostAccessors, HistogramAddHistogram, HistogramClear, HistogramSelfAddHistogram,
};
use super::util::FastLog2;
use crate::enc::combined_alloc::{alloc_or_default, allocate};

#[derive(Clone, Copy)]
pub struct HistogramPair {
    pub idx1: u32,
    pub idx2: u32,
    pub cost_combo: super::util::floatX,
    pub cost_diff: super::util::floatX,
}

impl Default for HistogramPair {
    #[inline(always)]
    fn default() -> HistogramPair {
        HistogramPair {
            idx1: 0,
            idx2: 0,
            cost_combo: 0.0,
            cost_diff: 0.0,
        }
    }
}
/* Returns entropy reduction of the context map when we combine two clusters. */
#[inline(always)]
fn ClusterCostDiff(size_a: usize, size_b: usize) -> super::util::floatX {
    let size_c: usize = size_a.wrapping_add(size_b);
    size_a as (super::util::floatX) * FastLog2(size_a as u64)
        + size_b as (super::util::floatX) * FastLog2(size_b as u64)
        - size_c as (super::util::floatX) * FastLog2(size_c as u64)
}

#[inline(always)]
fn HistogramPairIsLess(p1: &HistogramPair, p2: &HistogramPair) -> bool {
    if p1.cost_diff != p2.cost_diff {
        p1.cost_diff > p2.cost_diff
    } else {
        p1.idx2.wrapping_sub(p1.idx1) > p2.idx2.wrapping_sub(p2.idx1)
    }
}

/* Computes the bit cost reduction by combining out[idx1] and out[idx2] and if
it is below a threshold, stores the pair (idx1, idx2) in the *pairs queue. */
fn BrotliCompareAndPushToQueue<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
>(
    out: &[HistogramType],
    cluster_size: &[u32],
    mut idx1: u32,
    mut idx2: u32,
    max_num_pairs: usize,
    scratch_space: &mut HistogramType::i32vec,
    pairs: &mut [HistogramPair],
    num_pairs: &mut usize,
) {
    let mut is_good_pair = false;
    let mut p: HistogramPair = HistogramPair {
        idx1: 0,
        idx2: 0,
        cost_combo: 0.0,
        cost_diff: 0.0,
    };
    if idx1 == idx2 {
    } else {
        if idx2 < idx1 {
            core::mem::swap(&mut idx2, &mut idx1);
        }
        p.idx1 = idx1;
        p.idx2 = idx2;
        p.cost_diff = 0.5
            * ClusterCostDiff(
                cluster_size[idx1 as usize] as usize,
                cluster_size[idx2 as usize] as usize,
            );
        p.cost_diff -= (out[idx1 as usize]).bit_cost();
        p.cost_diff -= (out[idx2 as usize]).bit_cost();
        if (out[idx1 as usize]).total_count() == 0usize {
            p.cost_combo = (out[idx2 as usize]).bit_cost();
            is_good_pair = true;
        } else if (out[idx2 as usize]).total_count() == 0usize {
            p.cost_combo = (out[idx1 as usize]).bit_cost();
            is_good_pair = true;
        } else {
            let threshold = if *num_pairs == 0 {
                1e38
            } else {
                pairs[0].cost_diff.max(0.0)
            };

            let mut combo: HistogramType = out[idx1 as usize].clone();
            HistogramAddHistogram(&mut combo, &out[idx2 as usize]);
            let cost_combo: super::util::floatX = BrotliPopulationCost(&combo, scratch_space);
            if cost_combo < threshold - p.cost_diff {
                p.cost_combo = cost_combo;
                is_good_pair = true;
            }
        }
        if is_good_pair {
            p.cost_diff += p.cost_combo;
            if *num_pairs > 0usize && HistogramPairIsLess(&pairs[0], &p) {
                /* Replace the top of the queue if needed. */
                if *num_pairs < max_num_pairs {
                    pairs[*num_pairs] = pairs[0];
                    *num_pairs = num_pairs.wrapping_add(1);
                }
                pairs[0] = p;
            } else if *num_pairs < max_num_pairs {
                pairs[*num_pairs] = p;
                *num_pairs = num_pairs.wrapping_add(1);
            }
        }
    }
}

pub fn BrotliHistogramCombine<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
>(
    out: &mut [HistogramType],
    cluster_size: &mut [u32],
    symbols: &mut [u32],
    clusters: &mut [u32],
    pairs: &mut [HistogramPair],
    mut num_clusters: usize,
    symbols_size: usize,
    max_clusters: usize,
    max_num_pairs: usize,
    scratch_space: &mut HistogramType::i32vec,
) -> usize {
    let mut cost_diff_threshold: super::util::floatX = 0.0;
    let mut min_cluster_size: usize = 1;
    let mut num_pairs: usize = 0usize;
    {
        /* We maintain a vector of histogram pairs, with the property that the pair
        with the maximum bit cost reduction is the first. */
        for idx1 in 0..num_clusters {
            for idx2 in idx1 + 1..num_clusters {
                BrotliCompareAndPushToQueue(
                    out,
                    cluster_size,
                    clusters[idx1],
                    clusters[idx2],
                    max_num_pairs,
                    scratch_space,
                    pairs,
                    &mut num_pairs,
                );
            }
        }
    }
    while num_clusters > min_cluster_size {
        let mut i: usize;
        if (pairs[0]).cost_diff >= cost_diff_threshold {
            cost_diff_threshold = 1e38;
            min_cluster_size = max_clusters;
            {
                continue;
            }
        }
        /* Take the best pair from the top of heap. */
        let best_idx1: u32 = (pairs[0]).idx1;
        let best_idx2: u32 = (pairs[0]).idx2;
        HistogramSelfAddHistogram(out, (best_idx1 as usize), (best_idx2 as usize));
        (out[(best_idx1 as usize)]).set_bit_cost((pairs[0]).cost_combo);
        {
            let _rhs = cluster_size[(best_idx2 as usize)];
            let _lhs = &mut cluster_size[(best_idx1 as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs);
        }
        for i in 0usize..symbols_size {
            if symbols[i] == best_idx2 {
                symbols[i] = best_idx1;
            }
        }
        i = 0usize;
        'break9: while i < num_clusters {
            {
                if clusters[i] == best_idx2 {
                    for offset in 0..(num_clusters - i - 1) {
                        clusters[i + offset] = clusters[i + 1 + offset];
                    }
                    break 'break9;
                }
            }
            i = i.wrapping_add(1);
        }
        num_clusters = num_clusters.wrapping_sub(1);
        {
            /* Remove pairs intersecting the just combined best pair. */
            let mut copy_to_idx: usize = 0usize;
            i = 0usize;
            while i < num_pairs {
                'continue12: loop {
                    {
                        let p: HistogramPair = pairs[i];
                        if (p).idx1 == best_idx1
                            || (p).idx2 == best_idx1
                            || (p).idx1 == best_idx2
                            || (p).idx2 == best_idx2
                        {
                            /* Remove invalid pair from the queue. */
                            break 'continue12;
                        }
                        if HistogramPairIsLess(&pairs[0], &p) {
                            /* Replace the top of the queue if needed. */
                            let front: HistogramPair = pairs[0];
                            pairs[0] = p;
                            pairs[copy_to_idx] = front;
                        } else {
                            pairs[copy_to_idx] = p;
                        }
                        copy_to_idx = copy_to_idx.wrapping_add(1);
                    }
                    break;
                }
                i = i.wrapping_add(1);
            }
            num_pairs = copy_to_idx;
        }
        for i in 0usize..num_clusters {
            BrotliCompareAndPushToQueue(
                out,
                cluster_size,
                best_idx1,
                clusters[i],
                max_num_pairs,
                scratch_space,
                pairs,
                &mut num_pairs,
            );
        }
    }
    num_clusters
}

/* What is the bit cost of moving histogram from cur_symbol to candidate. */
#[inline(always)]
pub fn BrotliHistogramBitCostDistance<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
>(
    histogram: &HistogramType,
    candidate: &HistogramType,
    scratch_space: &mut HistogramType::i32vec,
) -> super::util::floatX {
    if histogram.total_count() == 0usize {
        0.0
    } else {
        let mut tmp: HistogramType = histogram.clone();
        HistogramAddHistogram(&mut tmp, candidate);
        BrotliPopulationCost(&tmp, scratch_space) - candidate.bit_cost()
    }
}

/* Find the best 'out' histogram for each of the 'in' histograms.
When called, clusters[0..num_clusters) contains the unique values from
symbols[0..in_size), but this property is not preserved in this function.
Note: we assume that out[]->bit_cost_ is already up-to-date. */

pub fn BrotliHistogramRemap<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
>(
    inp: &[HistogramType],
    in_size: usize,
    clusters: &[u32],
    num_clusters: usize,
    scratch_space: &mut HistogramType::i32vec,
    out: &mut [HistogramType],
    symbols: &mut [u32],
) {
    for i in 0usize..in_size {
        let mut best_out: u32 = if i == 0usize {
            symbols[0]
        } else {
            symbols[i.wrapping_sub(1)]
        };
        let mut best_bits: super::util::floatX =
            BrotliHistogramBitCostDistance(&inp[i], &mut out[(best_out as usize)], scratch_space);
        for j in 0usize..num_clusters {
            let cur_bits: super::util::floatX = BrotliHistogramBitCostDistance(
                &inp[i],
                &mut out[(clusters[j] as usize)],
                scratch_space,
            );
            if cur_bits < best_bits {
                best_bits = cur_bits;
                best_out = clusters[j];
            }
        }
        symbols[i] = best_out;
    }
    for i in 0usize..num_clusters {
        HistogramClear(&mut out[(clusters[i] as usize)]);
    }
    for i in 0usize..in_size {
        HistogramAddHistogram(&mut out[(symbols[i] as usize)], &inp[i]);
    }
}

/* Reorders elements of the out[0..length) array and changes values in
symbols[0..length) array in the following way:
  * when called, symbols[] contains indexes into out[], and has N unique
    values (possibly N < length)
  * on return, symbols'[i] = f(symbols[i]) and
               out'[symbols'[i]] = out[symbols[i]], for each 0 <= i < length,
    where f is a bijection between the range of symbols[] and [0..N), and
    the first occurrences of values in symbols'[i] come in consecutive
    increasing order.
Returns N, the number of unique values in symbols[]. */
pub fn BrotliHistogramReindex<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
    Alloc: alloc::Allocator<u32> + alloc::Allocator<HistogramType>,
>(
    alloc: &mut Alloc,
    out: &mut [HistogramType],
    symbols: &mut [u32],
    length: usize,
) -> usize {
    static kInvalidIndex: u32 = u32::MAX;
    let mut new_index = alloc_or_default::<u32, _>(alloc, length);
    let mut next_index: u32;
    let mut tmp: <Alloc as Allocator<HistogramType>>::AllocatedMemory;
    for i in 0usize..length {
        new_index.slice_mut()[i] = kInvalidIndex;
    }
    next_index = 0u32;
    for i in 0usize..length {
        if new_index.slice()[(symbols[i] as usize)] == kInvalidIndex {
            new_index.slice_mut()[(symbols[i] as usize)] = next_index;
            next_index = next_index.wrapping_add(1);
        }
    }
    tmp = alloc_or_default::<HistogramType, _>(alloc, next_index as usize);
    next_index = 0u32;
    for i in 0usize..length {
        if new_index.slice()[(symbols[i] as usize)] == next_index {
            tmp.slice_mut()[(next_index as usize)] = out[(symbols[i] as usize)].clone();
            next_index = next_index.wrapping_add(1);
        }
        symbols[i] = new_index.slice()[(symbols[i] as usize)];
    }
    {
        <Alloc as Allocator<u32>>::free_cell(alloc, new_index);
    }
    for i in 0usize..next_index as usize {
        out[i] = tmp.slice()[i].clone();
    }
    {
        <Alloc as Allocator<HistogramType>>::free_cell(alloc, tmp)
    }
    next_index as usize
}

pub fn BrotliClusterHistograms<
    HistogramType: SliceWrapperMut<u32> + SliceWrapper<u32> + CostAccessors + Clone,
    Alloc: alloc::Allocator<u32> + alloc::Allocator<HistogramPair> + alloc::Allocator<HistogramType>,
>(
    alloc: &mut Alloc,
    inp: &[HistogramType],
    in_size: usize,
    max_histograms: usize,
    scratch_space: &mut HistogramType::i32vec,
    out: &mut [HistogramType],
    out_size: &mut usize,
    histogram_symbols: &mut [u32],
) {
    let mut cluster_size = alloc_or_default::<u32, Alloc>(alloc, in_size);
    let mut clusters = alloc_or_default::<u32, Alloc>(alloc, in_size);
    let mut num_clusters: usize = 0usize;
    let max_input_histograms: usize = 64usize;
    let pairs_capacity: usize = max_input_histograms
        .wrapping_mul(max_input_histograms)
        .wrapping_div(2);
    let mut pairs = allocate::<HistogramPair, _>(alloc, pairs_capacity.wrapping_add(1));
    let mut i: usize;
    for i in 0usize..in_size {
        cluster_size.slice_mut()[i] = 1u32;
    }
    for i in 0usize..in_size {
        out[i] = inp[i].clone();
        (out[i]).set_bit_cost(BrotliPopulationCost(&inp[i], scratch_space));
        histogram_symbols[i] = i as u32;
    }
    i = 0usize;
    while i < in_size {
        {
            let num_to_combine: usize = min(in_size.wrapping_sub(i), max_input_histograms);

            for j in 0usize..num_to_combine {
                clusters.slice_mut()[num_clusters.wrapping_add(j)] = i.wrapping_add(j) as u32;
            }
            let num_new_clusters: usize = BrotliHistogramCombine(
                out,
                cluster_size.slice_mut(),
                &mut histogram_symbols[i..],
                &mut clusters.slice_mut()[num_clusters..],
                pairs.slice_mut(),
                num_to_combine,
                num_to_combine,
                max_histograms,
                pairs_capacity,
                scratch_space,
            );
            num_clusters = num_clusters.wrapping_add(num_new_clusters);
        }
        i = i.wrapping_add(max_input_histograms);
    }
    {
        let max_num_pairs: usize = min(
            (64usize).wrapping_mul(num_clusters),
            num_clusters.wrapping_div(2).wrapping_mul(num_clusters),
        );
        {
            if pairs_capacity < max_num_pairs.wrapping_add(1) {
                let mut _new_size: usize = if pairs_capacity == 0usize {
                    max_num_pairs.wrapping_add(1)
                } else {
                    pairs_capacity
                };
                let mut new_array: <Alloc as Allocator<HistogramPair>>::AllocatedMemory;
                while _new_size < max_num_pairs.wrapping_add(1) {
                    _new_size = _new_size.wrapping_mul(2);
                }
                new_array = alloc_or_default::<HistogramPair, _>(alloc, _new_size);
                new_array.slice_mut()[..pairs_capacity]
                    .clone_from_slice(&pairs.slice()[..pairs_capacity]);
                <Alloc as Allocator<HistogramPair>>::free_cell(
                    alloc,
                    core::mem::replace(&mut pairs, new_array),
                );
            }
        }
        num_clusters = BrotliHistogramCombine(
            out,
            cluster_size.slice_mut(),
            histogram_symbols,
            clusters.slice_mut(),
            pairs.slice_mut(),
            num_clusters,
            in_size,
            max_histograms,
            max_num_pairs,
            scratch_space,
        );
    }
    <Alloc as Allocator<HistogramPair>>::free_cell(alloc, pairs);
    <Alloc as Allocator<u32>>::free_cell(alloc, cluster_size);
    BrotliHistogramRemap(
        inp,
        in_size,
        clusters.slice(),
        num_clusters,
        scratch_space,
        out,
        histogram_symbols,
    );
    <Alloc as Allocator<u32>>::free_cell(alloc, clusters);
    *out_size = BrotliHistogramReindex(alloc, out, histogram_symbols, in_size);
}

/////////// DONE //////////////////////////
