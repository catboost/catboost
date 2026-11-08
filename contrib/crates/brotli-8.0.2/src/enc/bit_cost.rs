use alloc::SliceWrapperMut;
use core::cmp::{max, min};

use super::super::alloc::SliceWrapper;
use super::histogram::CostAccessors;
use super::util::{FastLog2, FastLog2u16};
use super::vectorization::Mem256i;
use crate::enc::floatX;

const BROTLI_REPEAT_ZERO_CODE_LENGTH: usize = 17;
const BROTLI_CODE_LENGTH_CODES: usize = BROTLI_REPEAT_ZERO_CODE_LENGTH + 1;

pub(crate) fn shannon_entropy(mut population: &[u32], size: usize) -> (floatX, usize) {
    let mut sum: usize = 0;
    let mut retval: floatX = 0.0;

    if (size & 1) != 0 && !population.is_empty() {
        let p = population[0] as usize;
        population = population.split_at(1).1;
        sum = sum.wrapping_add(p);
        retval -= p as floatX * FastLog2u16(p as u16);
    }
    for pop_iter in population.split_at((size >> 1) << 1).0 {
        let p = *pop_iter as usize;
        sum = sum.wrapping_add(p);
        retval -= p as floatX * FastLog2u16(p as u16);
    }
    if sum != 0 {
        retval += sum as floatX * FastLog2(sum as u64); // not sure it's 16 bit
    }

    (retval, sum)
}

#[inline(always)]
pub fn BitsEntropy(population: &[u32], size: usize) -> floatX {
    let (mut retval, sum) = shannon_entropy(population, size);
    if retval < sum as floatX {
        retval = sum as floatX;
    }
    retval
}

#[allow(clippy::excessive_precision)]
fn CostComputation<T: SliceWrapper<Mem256i>>(
    depth_histo: &mut [u32; BROTLI_CODE_LENGTH_CODES],
    nnz_data: &T,
    nnz: usize,
    _total_count: floatX,
    log2total: floatX,
) -> floatX {
    let mut bits: floatX = 0.0;
    let mut max_depth: usize = 1;
    for i in 0..nnz {
        // Compute -log2(P(symbol)) = -log2(count(symbol)/total_count) =
        //                            = log2(total_count) - log2(count(symbol))
        let element = nnz_data.slice()[i >> 3][i & 7];
        let log2p = log2total - FastLog2u16(element as u16);
        // Approximate the bit depth by round(-log2(P(symbol)))
        let depth = min((log2p + 0.5) as u8, 15u8);
        bits += (element as floatX) * log2p;
        if (depth as usize) > max_depth {
            max_depth = depth as usize;
        }
        depth_histo[depth as usize] += 1;
    }

    // Add the estimated encoding cost of the code length code histogram.
    bits += (18 + 2 * max_depth) as floatX;
    // Add the entropy of the code length code histogram.
    bits += BitsEntropy(depth_histo, BROTLI_CODE_LENGTH_CODES);
    //println_stderr!("{:?} {:?}", &depth_histo[..], bits);
    bits
}

pub fn BrotliPopulationCost<HistogramType: SliceWrapper<u32> + CostAccessors>(
    histogram: &HistogramType,
    nnz_data: &mut HistogramType::i32vec,
) -> floatX {
    static kOneSymbolHistogramCost: floatX = 12.0;
    static kTwoSymbolHistogramCost: floatX = 20.0;
    static kThreeSymbolHistogramCost: floatX = 28.0;
    static kFourSymbolHistogramCost: floatX = 37.0;

    let data_size: usize = histogram.slice().len();
    let mut count = 0;
    let mut s: [usize; 5] = [0; 5];
    let mut bits: floatX = 0.0;

    if histogram.total_count() == 0 {
        return kOneSymbolHistogramCost;
    }
    for i in 0..data_size {
        if histogram.slice()[i] > 0 {
            s[count] = i;
            count += 1;
            if count > 4 {
                break;
            }
        }
    }
    match count {
        1 => return kOneSymbolHistogramCost,
        2 => return kTwoSymbolHistogramCost + histogram.total_count() as floatX,
        3 => {
            let histo0: u32 = histogram.slice()[s[0]];
            let histo1: u32 = histogram.slice()[s[1]];
            let histo2: u32 = histogram.slice()[s[2]];
            let histomax: u32 = max(histo0, max(histo1, histo2));
            return kThreeSymbolHistogramCost
                + (2u32).wrapping_mul(histo0.wrapping_add(histo1).wrapping_add(histo2)) as floatX
                - histomax as floatX;
        }
        4 => {
            let mut histo: [u32; 4] = [0; 4];

            for i in 0..4 {
                histo[i] = histogram.slice()[s[i]];
            }
            for i in 0..4 {
                for j in i + 1..4 {
                    if histo[j] > histo[i] {
                        histo.swap(j, i);
                    }
                }
            }
            let h23: u32 = histo[2].wrapping_add(histo[3]);
            let histomax: u32 = max(h23, histo[0]);
            return kFourSymbolHistogramCost
                + (3u32).wrapping_mul(h23) as floatX
                + (2u32).wrapping_mul(histo[0].wrapping_add(histo[1])) as floatX
                - histomax as floatX;
        }
        _ => {}
    }

    if cfg!(feature = "vector_scratch_space") {
        // vectorization failed: it's faster to do things inline than split into two loops
        let mut nnz: usize = 0;
        let mut depth_histo = [0u32; 18];
        let total_count = histogram.total_count() as floatX;
        let log2total = FastLog2(histogram.total_count() as u64);
        let mut i: usize = 0;
        while i < data_size {
            if histogram.slice()[i] > 0 {
                let nnz_val = &mut nnz_data.slice_mut()[nnz >> 3];
                nnz_val[nnz & 7] = histogram.slice()[i] as i32;
                i += 1;
                nnz += 1;
            } else {
                let mut reps: u32 = 1;
                for hd in histogram.slice()[i + 1..data_size].iter() {
                    if *hd != 0 {
                        break;
                    }
                    reps += 1
                }
                i += reps as usize;
                if i == data_size {
                    break;
                }
                if reps < 3 {
                    depth_histo[0] += reps;
                } else {
                    reps -= 2;
                    let mut depth_histo_adds: u32 = 0;
                    while reps > 0 {
                        depth_histo_adds += 1;
                        bits += 3.0;
                        reps >>= 3;
                    }
                    depth_histo[BROTLI_REPEAT_ZERO_CODE_LENGTH] += depth_histo_adds;
                }
            }
        }
        bits += CostComputation(&mut depth_histo, nnz_data, nnz, total_count, log2total);
    } else {
        let mut max_depth: usize = 1;
        let mut depth_histo = [0u32; 18];
        let log2total: floatX = FastLog2(histogram.total_count() as u64); // 64 bit here
        let mut reps: u32 = 0;
        for histo in histogram.slice()[..data_size].iter() {
            if *histo != 0 {
                if reps != 0 {
                    if reps < 3 {
                        depth_histo[0] += reps;
                    } else {
                        reps -= 2;
                        while reps > 0 {
                            depth_histo[17] += 1;
                            bits += 3.0;
                            reps >>= 3;
                        }
                    }
                    reps = 0;
                }
                let log2p = log2total - FastLog2u16(*histo as u16);
                let mut depth = (log2p + 0.5) as usize;
                bits += *histo as floatX * log2p;
                depth = min(depth, 15);
                max_depth = max(depth, max_depth);
                depth_histo[depth] += 1;
            } else {
                reps += 1;
            }
        }
        bits += (18usize).wrapping_add((2usize).wrapping_mul(max_depth)) as floatX;
        bits += BitsEntropy(&depth_histo[..], 18);
    }
    bits
}
