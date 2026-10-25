use core::cmp::{max, min};
use core::ops::{Index, IndexMut, Range};

use super::super::alloc;
use super::super::alloc::{SliceWrapper, SliceWrapperMut};
use super::input_pair::{InputPair, InputReference};
use super::interface;
use super::util::FastLog2;
use crate::enc::combined_alloc::alloc_if;
// float32 doesn't have enough resolution for blocks of data more than 3.5 megs
pub type floatY = f64;
// the cost of storing a particular population of data including the approx
// cost of a huffman table to describe the frequencies of each symbol
pub fn HuffmanCost(population: &[u32]) -> floatY {
    assert_eq!(population.len(), 256 * 256);
    let mut cost: floatY = 0.0 as floatY;
    let mut sum: floatY = 0.0 as floatY;
    let mut buckets: floatY = 0.0 as floatY;
    for pop in population.iter() {
        if *pop == 0 {
            continue;
        }
        cost -= *pop as floatY * FastLog2(*pop as u64) as floatY;
        sum += *pop as floatY;
        buckets += 1.0 as floatY;
    }

    //println!("Observed {} nonzero buckets with a sum of {}, hc={}", buckets, sum, cost);

    16.0 as floatY * buckets + cost + sum * FastLog2(sum as u64) as floatY
}

// this holds a population of data assuming 1 byte of prior for that data
// bucket_populations is therefore a 65536-long dynamically allocated buffer
pub struct EntropyBucketPopulation<AllocU32: alloc::Allocator<u32>> {
    pub bucket_populations: AllocU32::AllocatedMemory,
    pub cached_bit_entropy: floatY,
}

impl<AllocU32: alloc::Allocator<u32>> EntropyBucketPopulation<AllocU32> {
    pub fn new(m32: &mut AllocU32) -> Self {
        let size = 256 * 256;
        EntropyBucketPopulation::<AllocU32> {
            cached_bit_entropy: 0.0,
            bucket_populations: m32.alloc_cell(size),
        }
    }
    pub fn free(&mut self, m32: &mut AllocU32) {
        m32.free_cell(core::mem::take(&mut self.bucket_populations));
    }
    fn clone_from(&mut self, other: &EntropyBucketPopulation<AllocU32>) {
        self.bucket_populations
            .slice_mut()
            .clone_from_slice(other.bucket_populations.slice());
    }
    fn add_assign(&mut self, other: &EntropyBucketPopulation<AllocU32>) {
        assert_eq!(
            self.bucket_populations.slice().len(),
            other.bucket_populations.slice().len()
        );
        for (item, other_item) in self
            .bucket_populations
            .slice_mut()
            .iter_mut()
            .zip(other.bucket_populations.slice().iter())
        {
            *item += *other_item;
        }
        self.cached_bit_entropy = HuffmanCost(self.bucket_populations.slice());
    }
    // clear the allocated memory and reset literal population to zero
    fn bzero(&mut self) {
        self.cached_bit_entropy = 0.0;
        for bp in self.bucket_populations.slice_mut().iter_mut() {
            *bp = 0;
        }
    }
    // setup population to the sum of an array of populations where the stride of that row matches. Additionally allow another optional
    fn initiate_from(
        &mut self,
        rows: [&[Self]; 2],
        rows_stride: [&[u8]; 2],
        stride: u8,
        do_clear: bool,
    ) {
        self.cached_bit_entropy = 0.0;
        let mut found_any = false;
        for (sub_row, sub_stride) in rows.iter().zip(rows_stride.iter()) {
            for (item, istride) in sub_row.iter().zip(sub_stride.iter()) {
                if *istride != stride {
                    continue; // if we chain, then optional was already filtered by stride
                }
                if do_clear && !found_any {
                    self.bucket_populations
                        .slice_mut()
                        .clone_from_slice(item.bucket_populations.slice());
                    found_any = true;
                } else {
                    for (dst, src) in self
                        .bucket_populations
                        .slice_mut()
                        .iter_mut()
                        .zip(item.bucket_populations.slice().iter())
                    {
                        *dst += *src;
                    }
                }
            }
        }
        if do_clear && !found_any {
            self.bzero();
        } else {
            self.cached_bit_entropy = HuffmanCost(self.bucket_populations.slice());
        }
    }
    fn bit_cost_of_data_subset(
        &mut self,
        data0: &[u8],
        mut stride: u8,
        mut prev_bytes: [u8; NUM_STRIDES],
        scratch: &mut EntropyBucketPopulation<AllocU32>,
    ) -> floatY {
        prev_bytes.reverse();
        stride = max(1, stride); // we return stride=1 to mean 1 away
        scratch
            .bucket_populations
            .slice_mut()
            .clone_from_slice(self.bucket_populations.slice());
        scratch.bucket_populations.slice_mut()[65535] += 1; // to demonstrate that we have
        scratch.bucket_populations.slice_mut()[65535] -= 1; // to demonstrate that we have write capability
        let mut stray_count = 0.0 as floatY;
        assert_eq!((NUM_STRIDES - 1) & NUM_STRIDES, 0); // must be power of two
        for (index, val) in data0.iter().enumerate() {
            let prior_byte =
                prev_bytes[(index + (NUM_STRIDES - stride as usize)) & (NUM_STRIDES - 1)];
            let loc = &mut scratch.bucket_populations.slice_mut()
                [prior_byte as usize * 256 + *val as usize];
            if *loc == 0 {
                stray_count += 1.0;
            } else {
                *loc -= 1;
            }
            prev_bytes[index & (NUM_STRIDES - 1)] = *val;
        }
        if self.cached_bit_entropy == 0.0 as floatY {
            self.cached_bit_entropy = HuffmanCost(self.bucket_populations.slice());
        }
        debug_assert_eq!(
            HuffmanCost(self.bucket_populations.slice()),
            self.cached_bit_entropy
        );

        scratch.cached_bit_entropy = HuffmanCost(scratch.bucket_populations.slice());
        self.cached_bit_entropy - scratch.cached_bit_entropy + stray_count * 8.0
    }
}

const NUM_STRIDES: usize = 8;
#[derive(Copy, Clone)]
pub struct BucketPopIndex {
    pub val: u8,
    pub six_bits: u8,
    pub stride: u8,
}

impl<AllocU32: alloc::Allocator<u32>> Index<BucketPopIndex> for EntropyBucketPopulation<AllocU32> {
    type Output = u32;
    fn index(&self, index: BucketPopIndex) -> &u32 {
        &self.bucket_populations.slice()
            [index.val as usize + index.six_bits as usize * 256 + index.stride as usize * 256 * 64]
    }
}
impl<AllocU32: alloc::Allocator<u32>> IndexMut<BucketPopIndex>
    for EntropyBucketPopulation<AllocU32>
{
    fn index_mut(&mut self, index: BucketPopIndex) -> &mut u32 {
        &mut self.bucket_populations.slice_mut()
            [index.val as usize + index.six_bits as usize * 256 + index.stride as usize * 256 * 64]
    }
}

pub struct EntropyTally<AllocU32: alloc::Allocator<u32>> {
    pop: [EntropyBucketPopulation<AllocU32>; NUM_STRIDES],
}

const NUM_LEVELS: usize = 4;
const NUM_NODES: usize = (1 << (NUM_LEVELS)) - 1;
pub const NUM_LEAF_NODES: usize = (NUM_NODES + 1) >> 1;

pub struct EntropyPyramid<AllocU32: alloc::Allocator<u32>> {
    pop: [EntropyBucketPopulation<AllocU32>; NUM_NODES],
    stride: [u8; NUM_NODES],
}

impl<AllocU32: alloc::Allocator<u32>> EntropyPyramid<AllocU32> {
    pub fn last_level_range(&self) -> Range<usize> {
        (NUM_NODES - (1 << (NUM_LEVELS - 1)))..NUM_NODES
    }
    pub fn byte_index_to_pyramid_index(&self, byte_index: usize, metablock_size: usize) -> usize {
        let range = self.last_level_range();
        min(
            range.start + (range.end - range.start) * byte_index / metablock_size,
            range.end - 1,
        ) // since we tally after the end of the literal block, it could be after the pyramid
    }
    pub fn reset_scratch_to_deepest_level(&self, output: &mut EntropyTally<AllocU32>) {
        let mut has_modified = [false; NUM_STRIDES];
        //println!("Last level range {:?}", self.last_level_range());
        for index in self.last_level_range() {
            if has_modified[self.stride[index] as usize] {
                output.pop[self.stride[index] as usize].add_assign(&self.pop[index]);
            } else {
                output.pop[self.stride[index] as usize].clone_from(&self.pop[index]);
                has_modified[self.stride[index] as usize] = true;
            }
        }
        for stride in 0..NUM_STRIDES {
            if !has_modified[stride] {
                output.pop[stride].bzero();
                output.pop[stride].cached_bit_entropy = 0.0;
            } else {
                output.pop[stride].cached_bit_entropy =
                    HuffmanCost(output.pop[stride].bucket_populations.slice());
            }
            //println!("BASE PYRAMID {} = {}", stride,output.pop[stride].cached_bit_entropy);
        }
    }
    pub fn stride_last_level_range(&self) -> [u8; NUM_LEAF_NODES] {
        let mut ret = [0u8; NUM_LEAF_NODES];
        ret.clone_from_slice(self.stride.split_at(self.stride.len() - NUM_LEAF_NODES).1);
        ret
    }
    pub fn free(&mut self, m32: &mut AllocU32) {
        for item in self.pop.iter_mut() {
            item.free(m32);
        }
    }
    pub fn disabled_placeholder(_m32: &mut AllocU32) -> Self {
        EntropyPyramid::<AllocU32> {
            pop: [
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: AllocU32::AllocatedMemory::default(),
                },
            ],
            stride: [0; NUM_NODES],
        }
    }
    pub fn new(m32: &mut AllocU32) -> Self {
        let size = 256 * 256;
        EntropyPyramid::<AllocU32> {
            pop: [
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
                EntropyBucketPopulation::<AllocU32> {
                    cached_bit_entropy: 0.0,
                    bucket_populations: m32.alloc_cell(size),
                },
            ],
            stride: [0; NUM_NODES],
        }
    }
    pub fn bit_cost_of_literals(
        &mut self,
        data0: &[u8],
        start_index: u32,
        metablock_len: usize,
        stride: u8,
        previous_bytes: [u8; NUM_STRIDES],
        scratch: &mut EntropyTally<AllocU32>,
    ) -> floatY {
        assert!(stride as usize <= NUM_STRIDES);

        self.pop[self.byte_index_to_pyramid_index(start_index as usize, metablock_len)]
            .bit_cost_of_data_subset(data0, stride, previous_bytes, &mut scratch.pop[0])
    }
    fn populate_entry_stride1(&mut self, input: InputPair, index: u32) {
        let mut prev_val = 0;
        let pyr_item = &mut self.pop[index as usize];
        pyr_item.bzero();
        assert_eq!(pyr_item.bucket_populations.slice()[65535], 0);
        for val in input.0.slice().iter().chain(input.1.slice().iter()) {
            pyr_item.bucket_populations.slice_mut()[prev_val as usize * 256 + *val as usize] += 1;
            prev_val = *val;
        }
        pyr_item.cached_bit_entropy = HuffmanCost(pyr_item.bucket_populations.slice());
        self.stride[index as usize] = 0;
    }
    fn populate_entry(
        &mut self,
        input: InputPair,
        scratch: &mut EntropyTally<AllocU32>,
        index: u32,
        mirror_range: Option<Range<usize>>,
        prev_range: Option<Range<usize>>,
    ) {
        let mut initial_entropies = [0.0 as floatY; NUM_STRIDES];
        let nothing: &[EntropyBucketPopulation<AllocU32>] = &[];
        let nothing_u8: &[u8] = &[];
        {
            let pop_ranges = [
                match mirror_range {
                    None => nothing,
                    Some(ref ir) => &self.pop[ir.clone()],
                },
                match prev_range {
                    None => nothing,
                    Some(ref pr) => &self.pop[pr.clone()],
                },
            ];
            let stride_ranges = [
                match mirror_range {
                    None => nothing_u8,
                    Some(ref ir) => &self.stride[ir.clone()],
                },
                match prev_range {
                    None => nothing_u8,
                    Some(ref pr) => &self.stride[pr.clone()],
                },
            ];
            for stride in 0..NUM_STRIDES {
                scratch.pop[stride].initiate_from(pop_ranges, stride_ranges, stride as u8, true);
                initial_entropies[stride] = scratch.pop[stride].cached_bit_entropy;
            }
        }
        scratch.observe_input_stream(input.0.slice(), input.1.slice());
        let mut best_entropy_index = 0;
        let mut min_entropy_value = (scratch.pop[0].cached_bit_entropy - initial_entropies[0]);
        //println!("{} OLD ENTROPY {:} NEW_ENTROPY {:}", best_entropy_index, scratch.pop[0].cached_bit_entropy, initial_entropies[0]);
        for stride in 1..NUM_STRIDES {
            let entropy_value = scratch.pop[stride].cached_bit_entropy - initial_entropies[stride];
            //println!("{} OLD ENTROPY {:} NEW_ENTROPY {:}", stride, scratch.pop[stride].cached_bit_entropy, initial_entropies[stride]);
            if entropy_value < min_entropy_value {
                best_entropy_index = stride;
                min_entropy_value = entropy_value;
            }
        }
        self.pop[index as usize].clone_from(&scratch.pop[best_entropy_index]);
        self.stride[index as usize] = best_entropy_index as u8;
    }
    pub fn populate_stride1(&mut self, input0: &[u8], input1: &[u8]) {
        let input = InputPair(
            InputReference {
                data: input0,
                orig_offset: 0,
            },
            InputReference {
                data: input1,
                orig_offset: input0.len(),
            },
        );
        for i in 0..2 {
            let first_range = if i == 0 {
                input.split_at(input.len() >> 1).0
            } else {
                input.split_at(input.len() >> 1).1
            };
            for j in 0..2 {
                let second_range = if j == 0 {
                    first_range.split_at(input.len() >> 2).0
                } else {
                    first_range.split_at(input.len() >> 2).1
                };
                if NUM_LEVELS == 4 {
                    for k in 0..2 {
                        let third_range = if j == 0 {
                            second_range.split_at(input.len() >> 3).0
                        } else {
                            second_range.split_at(input.len() >> 3).1
                        };
                        self.populate_entry_stride1(third_range, 7 + ((i << 2) + (j << 1) + k));
                    }
                } else {
                    assert_eq!(NUM_LEVELS, 3); // we hard coded the 3 levels for now... we can add more later or make this into some kind of recursion
                    self.populate_entry_stride1(second_range, 3 + ((i << 1) + j));
                }
            }
        }
    }
    pub fn populate(&mut self, input0: &[u8], input1: &[u8], scratch: &mut EntropyTally<AllocU32>) {
        let input = InputPair(
            InputReference {
                data: input0,
                orig_offset: 0,
            },
            InputReference {
                data: input1,
                orig_offset: input0.len(),
            },
        );
        self.populate_entry(input, scratch, 0, None, None); // BASE

        // LEVEL 1
        self.populate_entry(
            input.split_at(input.len() >> 1).0,
            scratch,
            1,
            Some(0..1),
            None,
        );
        self.populate_entry(
            input.split_at(input.len() >> 1).1,
            scratch,
            2,
            None,
            Some(1..2),
        ); // should we use the range from 0..1??

        // LEVEL 2
        self.populate_entry(
            input.split_at(input.len() >> 2).0,
            scratch,
            3,
            Some(1..3),
            None,
        );
        self.populate_entry(
            input
                .split_at(input.len() >> 1)
                .0
                .split_at(input.len() >> 2)
                .1,
            scratch,
            4,
            Some(2..3),
            Some(3..4),
        );
        self.populate_entry(
            input
                .split_at(input.len() >> 1)
                .1
                .split_at(input.len() >> 2)
                .0,
            scratch,
            5,
            Some(3..5),
            None,
        );
        self.populate_entry(
            input
                .split_at(input.len() >> 1)
                .1
                .split_at(input.len() >> 2)
                .1,
            scratch,
            6,
            Some(3..6),
            None,
        );
        if NUM_LEVELS == 4 {
            // level 4
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .0
                    .split_at(input.len() >> 2)
                    .0
                    .split_at(input.len() >> 3)
                    .0,
                scratch,
                7,
                Some(4..7),
                None,
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .0
                    .split_at(input.len() >> 2)
                    .0
                    .split_at(input.len() >> 3)
                    .1,
                scratch,
                8,
                Some(4..7),
                Some(7..8),
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .0
                    .split_at(input.len() >> 2)
                    .1
                    .split_at(input.len() >> 3)
                    .0,
                scratch,
                9,
                Some(5..7),
                Some(7..9),
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .0
                    .split_at(input.len() >> 2)
                    .1
                    .split_at(input.len() >> 3)
                    .1,
                scratch,
                0x0a,
                Some(5..7),
                Some(7..0xa),
            );

            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .1
                    .split_at(input.len() >> 2)
                    .0
                    .split_at(input.len() >> 3)
                    .0,
                scratch,
                0xb,
                Some(6..7),
                Some(7..0xb),
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .1
                    .split_at(input.len() >> 2)
                    .0
                    .split_at(input.len() >> 3)
                    .1,
                scratch,
                0xc,
                Some(6..7),
                Some(7..0xc),
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .1
                    .split_at(input.len() >> 2)
                    .1
                    .split_at(input.len() >> 3)
                    .0,
                scratch,
                0xd,
                None,
                Some(7..0xd),
            );
            self.populate_entry(
                input
                    .split_at(input.len() >> 1)
                    .1
                    .split_at(input.len() >> 2)
                    .1
                    .split_at(input.len() >> 3)
                    .1,
                scratch,
                0xe,
                None,
                Some(7..0xe),
            );
        } else {
            assert_eq!(NUM_LEVELS, 3); // we hard coded the 3 levels for now... we can add more later or make this into some kind of recursion
        }
    }
}

impl<AllocU32: alloc::Allocator<u32>> EntropyTally<AllocU32> {
    pub fn new(m32: &mut AllocU32, max_stride_arg: Option<u8>) -> EntropyTally<AllocU32> {
        let size = 256 * 256;
        let max_stride = max_stride_arg.unwrap_or(NUM_STRIDES as u8);
        EntropyTally::<AllocU32> {
            pop: [
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 0, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 1, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 2, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 3, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 4, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 5, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 6, m32, size),
                },
                EntropyBucketPopulation {
                    cached_bit_entropy: 0.0,
                    bucket_populations: alloc_if(max_stride > 7, m32, size),
                },
            ],
        }
    }
    pub fn disabled_placeholder(m32: &mut AllocU32) -> EntropyTally<AllocU32> {
        Self::new(m32, Some(0))
    }
    fn observe_input_stream(&mut self, input0: &[u8], input1: &[u8]) {
        let mut priors = [0u8; NUM_STRIDES];
        for val in input0.iter().chain(input1.iter()) {
            for stride in 0..NUM_STRIDES {
                self.pop[stride].bucket_populations.slice_mut()
                    [priors[stride] as usize * 256 + (*val as usize)] += 1;
            }
            {
                let mut tmp = [0u8; NUM_STRIDES - 1];
                tmp.clone_from_slice(&priors[..(NUM_STRIDES - 1)]);
                priors[1..].clone_from_slice(&tmp[..]);
                priors[0] = *val;
            }
        }
        for stride in 0..NUM_STRIDES {
            self.pop[stride].cached_bit_entropy =
                HuffmanCost(self.pop[stride].bucket_populations.slice());
        }
    }
    fn identify_best_population_and_update_cache(&mut self) -> u8 {
        let mut old_bit_entropy: [floatY; NUM_STRIDES] = [0.0; NUM_STRIDES];
        for (obe, be) in old_bit_entropy.iter_mut().zip(self.pop.iter_mut()) {
            *obe = be.cached_bit_entropy;
            if *obe != 0.0 {
                be.cached_bit_entropy = HuffmanCost(be.bucket_populations.slice());
            }
        }
        let mut best_stride = 0u8;
        let mut best_entropy = self.pop[0].cached_bit_entropy - old_bit_entropy[0];
        //println!("Weighing {} as {}", best_stride, best_entropy);
        for index in 1..NUM_STRIDES {
            let cur = self.pop[index].cached_bit_entropy - old_bit_entropy[index];
            //println!("Weighing {} as {} = [{} - {}]", index, cur, self.pop[index].cached_bit_entropy, old_bit_entropy[index]);
            if (best_entropy == 0.0 || cur < best_entropy) && old_bit_entropy[index] > 0.0 {
                best_stride = index as u8;
                best_entropy = cur;
            }
        }
        best_stride
    }
    pub fn peek(&mut self) -> &mut EntropyBucketPopulation<AllocU32> {
        &mut self.pop[0]
    }
    pub fn get_previous_bytes(
        &self,
        input0: &[u8],
        input1: &[u8],
        bytes_processed: usize,
    ) -> [u8; NUM_STRIDES] {
        let mut retval = [0u8; NUM_STRIDES];
        for index in 0..NUM_STRIDES {
            let bp_offset = index + 1;
            if bp_offset <= bytes_processed {
                let offset = bytes_processed - bp_offset;
                if offset >= input0.len() {
                    retval[index] = input1[offset - input0.len()];
                } else {
                    retval[index] = input0[offset];
                }
            }
        }
        retval
    }
    pub fn pick_best_stride<InputReference: SliceWrapper<u8>>(
        &mut self,
        commands: &[interface::Command<InputReference>],
        input0: &[u8],
        input1: &[u8],
        bytes_processed: &mut usize,
        entropy_pyramid: &EntropyPyramid<AllocU32>,
        stride_detection_quality: u8,
    ) -> u8 {
        if stride_detection_quality == 0 {
            return 0;
        }
        //println!("ENTROPY PYRAMID {:?}", entropy_pyramid.stride);
        if stride_detection_quality > 1 {
            entropy_pyramid.reset_scratch_to_deepest_level(self);
        }
        let mut pyramid_byte_index: usize = 0;
        for cmd in commands.iter() {
            match *cmd {
                interface::Command::Copy(ref copy) => {
                    *bytes_processed += copy.num_bytes as usize;
                }
                interface::Command::Dict(ref dict) => {
                    *bytes_processed += dict.final_size as usize;
                }
                interface::Command::Literal(ref lit) => {
                    if stride_detection_quality > 1 {
                        let mut priors = self.get_previous_bytes(input0, input1, *bytes_processed);
                        for (lindex, val) in lit.data.slice().iter().enumerate() {
                            if lindex == NUM_STRIDES {
                                let vpriors = self.get_previous_bytes(
                                    input0,
                                    input1,
                                    NUM_STRIDES + *bytes_processed,
                                );
                                assert_eq!(vpriors, priors);
                            }
                            for (index, prior) in priors.iter().enumerate() {
                                self.pop[index].bucket_populations.slice_mut()
                                    [256 * (*prior as usize) + *val as usize] += 1;
                                // increment the population value of this literal
                                // for the respective prior for the stride index
                            }
                            {
                                //reset prior values for the next item
                                let mut tmp = [0u8; 7];
                                tmp.clone_from_slice(&priors[..7]);
                                priors[1..].clone_from_slice(&tmp[..]);
                                priors[0] = *val;
                            }
                        }
                    }
                    *bytes_processed += lit.data.slice().len();
                    pyramid_byte_index = *bytes_processed;
                }
                interface::Command::BlockSwitchCommand(_)
                | interface::Command::BlockSwitchLiteral(_)
                | interface::Command::BlockSwitchDistance(_)
                | interface::Command::PredictionMode(_) => {}
            }
        }

        //println!("ENTROPY PYRAMID {:?} selected {}", entropy_pyramid.stride, best_stride);

        if stride_detection_quality > 1 {
            self.identify_best_population_and_update_cache() + 1
        } else {
            entropy_pyramid.stride[entropy_pyramid
                .byte_index_to_pyramid_index(pyramid_byte_index, input0.len() + input1.len())]
                + 1
        }
    }
    pub fn free(&mut self, m32: &mut AllocU32) {
        for item in self.pop.iter_mut() {
            m32.free_cell(core::mem::take(&mut item.bucket_populations))
        }
    }
    pub fn is_free(&mut self) -> bool {
        self.pop[0].bucket_populations.slice().is_empty()
    }
}
