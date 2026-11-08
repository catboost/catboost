/* Copyright 2010 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Entropy encoding (Huffman) utilities. */
use core::cmp::max;

#[derive(Clone, Copy, Default)]
pub struct HuffmanTree {
    pub total_count_: u32,
    pub index_left_: i16,
    pub index_right_or_value_: i16,
}

impl HuffmanTree {
    pub fn new(count: u32, left: i16, right: i16) -> Self {
        Self {
            total_count_: count,
            index_left_: left,
            index_right_or_value_: right,
        }
    }
}

pub fn BrotliSetDepth(p0: i32, pool: &mut [HuffmanTree], depth: &mut [u8], max_depth: i32) -> bool {
    let mut stack: [i32; 16] = [0; 16];
    let mut level: i32 = 0i32;
    let mut p: i32 = p0;
    stack[0] = -1i32;
    loop {
        if (pool[(p as usize)]).index_left_ as i32 >= 0i32 {
            level += 1;
            if level > max_depth {
                return false;
            }
            stack[level as usize] = (pool[(p as usize)]).index_right_or_value_ as i32;
            p = (pool[(p as usize)]).index_left_ as i32;
            {
                continue;
            }
        } else {
            let pp = pool[(p as usize)];
            depth[((pp).index_right_or_value_ as usize)] = level as u8;
        }
        while level >= 0i32 && (stack[level as usize] == -1i32) {
            level -= 1;
        }
        if level < 0i32 {
            return true;
        }
        p = stack[level as usize];
        stack[level as usize] = -1i32;
    }
}

pub trait HuffmanComparator {
    fn Cmp(&self, a: &HuffmanTree, b: &HuffmanTree) -> bool;
}
pub struct SortHuffmanTree {}
impl HuffmanComparator for SortHuffmanTree {
    fn Cmp(&self, v0: &HuffmanTree, v1: &HuffmanTree) -> bool {
        if v0.total_count_ != v1.total_count_ {
            v0.total_count_ < v1.total_count_
        } else {
            v0.index_right_or_value_ > v1.index_right_or_value_
        }
    }
}
pub fn SortHuffmanTreeItems<Comparator: HuffmanComparator>(
    items: &mut [HuffmanTree],
    n: usize,
    comparator: Comparator,
) {
    static gaps: [usize; 6] = [132, 57, 23, 10, 4, 1];
    if n < 13 {
        for i in 1..n {
            let mut tmp: HuffmanTree = items[i];
            let mut k: usize = i;
            let mut j: usize = i.wrapping_sub(1);
            while comparator.Cmp(&mut tmp, &mut items[j]) {
                items[k] = items[j];
                k = j;
                if {
                    let _old = j;
                    j = j.wrapping_sub(1);
                    _old
                } == 0
                {
                    break;
                }
            }
            items[k] = tmp;
        }
    } else {
        let mut g: i32 = if n < 57usize { 2i32 } else { 0i32 };
        while g < 6i32 {
            {
                let gap: usize = gaps[g as usize];
                for i in gap..n {
                    let mut j: usize = i;
                    let mut tmp: HuffmanTree = items[i];
                    while j >= gap && (comparator.Cmp(&mut tmp, &mut items[j.wrapping_sub(gap)])) {
                        {
                            items[j] = items[j.wrapping_sub(gap)];
                        }
                        j = j.wrapping_sub(gap);
                    }
                    items[j] = tmp;
                }
            }
            g += 1;
        }
    }
}

/* This function will create a Huffman tree.

The catch here is that the tree cannot be arbitrarily deep.
Brotli specifies a maximum depth of 15 bits for "code trees"
and 7 bits for "code length code trees."

count_limit is the value that is to be faked as the minimum value
and this minimum value is raised until the tree matches the
maximum length requirement.

This algorithm is not of excellent performance for very long data blocks,
especially when population counts are longer than 2**tree_limit, but
we are not planning to use this with extremely long blocks.

See https://en.wikipedia.org/wiki/Huffman_coding */
pub fn BrotliCreateHuffmanTree(
    data: &[u32],
    length: usize,
    tree_limit: i32,
    tree: &mut [HuffmanTree],
    depth: &mut [u8],
) {
    let sentinel = HuffmanTree::new(u32::MAX, -1, -1);
    let mut count_limit = 1u32;
    'break1: loop {
        {
            let mut n: usize = 0usize;
            let mut i: usize;
            let mut j: usize;
            let mut k: usize;
            i = length;
            while i != 0usize {
                i = i.wrapping_sub(1);
                if data[i] != 0 {
                    let count: u32 = max(data[i], count_limit);
                    tree[n] = HuffmanTree::new(count, -1, i as i16);
                    n = n.wrapping_add(1);
                }
            }
            if n == 1 {
                depth[((tree[0]).index_right_or_value_ as usize)] = 1u8;
                {
                    break 'break1;
                }
            }
            SortHuffmanTreeItems(tree, n, SortHuffmanTree {});
            tree[n] = sentinel;
            tree[n.wrapping_add(1)] = sentinel;
            i = 0usize;
            j = n.wrapping_add(1);
            k = n.wrapping_sub(1);
            while k != 0usize {
                {
                    let left: usize;
                    let right: usize;
                    if (tree[i]).total_count_ <= (tree[j]).total_count_ {
                        left = i;
                        i = i.wrapping_add(1);
                    } else {
                        left = j;
                        j = j.wrapping_add(1);
                    }
                    if (tree[i]).total_count_ <= (tree[j]).total_count_ {
                        right = i;
                        i = i.wrapping_add(1);
                    } else {
                        right = j;
                        j = j.wrapping_add(1);
                    }
                    {
                        let j_end: usize = (2usize).wrapping_mul(n).wrapping_sub(k);
                        (tree[j_end]).total_count_ = (tree[left])
                            .total_count_
                            .wrapping_add((tree[right]).total_count_);
                        (tree[j_end]).index_left_ = left as i16;
                        (tree[j_end]).index_right_or_value_ = right as i16;
                        tree[j_end.wrapping_add(1)] = sentinel;
                    }
                }
                k = k.wrapping_sub(1);
            }
            if BrotliSetDepth(
                (2usize).wrapping_mul(n).wrapping_sub(1) as i32,
                tree,
                depth,
                tree_limit,
            ) {
                break 'break1;
            }
        }
        count_limit = count_limit.wrapping_mul(2);
    }
}
pub fn BrotliOptimizeHuffmanCountsForRle(
    mut length: usize,
    counts: &mut [u32],
    good_for_rle: &mut [u8],
) {
    let mut nonzero_count: usize = 0usize;
    let mut stride: usize;
    let mut limit: usize;
    let mut sum: usize;
    let streak_limit: usize = 1240usize;
    for i in 0usize..length {
        if counts[i] != 0 {
            nonzero_count = nonzero_count.wrapping_add(1);
        }
    }
    if nonzero_count < 16usize {
        return;
    }
    while length != 0usize && (counts[length.wrapping_sub(1)] == 0u32) {
        length = length.wrapping_sub(1);
    }
    if length == 0usize {
        return;
    }
    {
        let mut nonzeros: usize = 0usize;
        let mut smallest_nonzero: u32 = (1i32 << 30) as u32;
        for i in 0usize..length {
            if counts[i] != 0u32 {
                nonzeros = nonzeros.wrapping_add(1);
                if smallest_nonzero > counts[i] {
                    smallest_nonzero = counts[i];
                }
            }
        }
        if nonzeros < 5usize {
            return;
        }
        if smallest_nonzero < 4u32 {
            let zeros: usize = length.wrapping_sub(nonzeros);
            if zeros < 6 {
                for i in 1..length.wrapping_sub(1) {
                    if counts[i - 1] != 0 && counts[i] == 0 && counts[i + 1] != 0 {
                        counts[i] = 1;
                    }
                }
            }
        }
        if nonzeros < 28usize {
            return;
        }
    }
    for rle_item in good_for_rle.iter_mut() {
        *rle_item = 0;
    }
    {
        let mut symbol: u32 = counts[0];
        let mut step: usize = 0usize;
        for i in 0..=length {
            if i == length || counts[i] != symbol {
                if symbol == 0u32 && (step >= 5usize) || symbol != 0u32 && (step >= 7usize) {
                    for k in 0usize..step {
                        good_for_rle[i.wrapping_sub(k).wrapping_sub(1)] = 1u8;
                    }
                }
                step = 1;
                if i != length {
                    symbol = counts[i];
                }
            } else {
                step = step.wrapping_add(1);
            }
        }
    }
    stride = 0usize;
    limit = (256u32)
        .wrapping_mul((counts[0]).wrapping_add(counts[1]).wrapping_add(counts[2]))
        .wrapping_div(3)
        .wrapping_add(420) as usize;
    sum = 0usize;
    for i in 0..=length {
        if i == length
            || good_for_rle[i] != 0
            || i != 0usize && (good_for_rle[i.wrapping_sub(1)] != 0)
            || ((256u32).wrapping_mul(counts[i]) as usize)
                .wrapping_sub(limit)
                .wrapping_add(streak_limit)
                >= (2usize).wrapping_mul(streak_limit)
        {
            if stride >= 4usize || stride >= 3usize && (sum == 0usize) {
                let mut count: usize = sum
                    .wrapping_add(stride.wrapping_div(2))
                    .wrapping_div(stride);
                if count == 0usize {
                    count = 1;
                }
                if sum == 0usize {
                    count = 0usize;
                }
                for k in 0usize..stride {
                    counts[i.wrapping_sub(k).wrapping_sub(1)] = count as u32;
                }
            }
            stride = 0usize;
            sum = 0usize;
            if i < length.wrapping_sub(2) {
                limit = (256u32)
                    .wrapping_mul(
                        (counts[i])
                            .wrapping_add(counts[i.wrapping_add(1)])
                            .wrapping_add(counts[i.wrapping_add(2)]),
                    )
                    .wrapping_div(3)
                    .wrapping_add(420) as usize;
            } else if i < length {
                limit = (256u32).wrapping_mul(counts[i]) as usize;
            } else {
                limit = 0usize;
            }
        }
        stride = stride.wrapping_add(1);
        if i != length {
            sum = sum.wrapping_add(counts[i] as usize);
            if stride >= 4usize {
                limit = (256usize)
                    .wrapping_mul(sum)
                    .wrapping_add(stride.wrapping_div(2))
                    .wrapping_div(stride);
            }
            if stride == 4usize {
                limit = limit.wrapping_add(120);
            }
        }
    }
}


pub(crate) fn decide_over_rle_use(depth: &[u8], length: usize) -> (bool, bool) {
    let mut total_reps_zero: usize = 0usize;
    let mut total_reps_non_zero: usize = 0usize;
    let mut count_reps_zero: usize = 1;
    let mut count_reps_non_zero: usize = 1;
    let mut i: usize;
    i = 0usize;
    while i < length {
        let value: u8 = depth[i];
        let mut reps: usize = 1;
        let mut k: usize;
        k = i.wrapping_add(1);
        while k < length && (depth[k] as i32 == value as i32) {
            {
                reps = reps.wrapping_add(1);
            }
            k = k.wrapping_add(1);
        }
        if reps >= 3usize && (value as i32 == 0i32) {
            total_reps_zero = total_reps_zero.wrapping_add(reps);
            count_reps_zero = count_reps_zero.wrapping_add(1);
        }
        if reps >= 4usize && (value as i32 != 0i32) {
            total_reps_non_zero = total_reps_non_zero.wrapping_add(reps);
            count_reps_non_zero = count_reps_non_zero.wrapping_add(1);
        }
        i = i.wrapping_add(reps);
    }
    let use_rle_for_non_zero = total_reps_non_zero > count_reps_non_zero.wrapping_mul(2);
    let use_rle_for_zero = total_reps_zero > count_reps_zero.wrapping_mul(2);

    (use_rle_for_non_zero, use_rle_for_zero)
}

fn Reverse(v: &mut [u8], mut start: usize, mut end: usize) {
    end = end.wrapping_sub(1);
    while start < end {
        v.swap(start, end);
        start = start.wrapping_add(1);
        end = end.wrapping_sub(1);
    }
}

fn BrotliWriteHuffmanTreeRepetitions(
    previous_value: u8,
    value: u8,
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    if previous_value as i32 != value as i32 {
        tree[*tree_size] = value;
        extra_bits_data[*tree_size] = 0u8;
        *tree_size = tree_size.wrapping_add(1);
        repetitions = repetitions.wrapping_sub(1);
    }
    if repetitions == 7usize {
        tree[*tree_size] = value;
        extra_bits_data[*tree_size] = 0u8;
        *tree_size = tree_size.wrapping_add(1);
        repetitions = repetitions.wrapping_sub(1);
    }
    if repetitions < 3usize {
        for _i in 0usize..repetitions {
            tree[*tree_size] = value;
            extra_bits_data[*tree_size] = 0u8;
            *tree_size = tree_size.wrapping_add(1);
        }
    } else {
        let start: usize = *tree_size;
        repetitions = repetitions.wrapping_sub(3);
        loop {
            tree[*tree_size] = 16u8;
            extra_bits_data[*tree_size] = (repetitions & 0x03) as u8;
            *tree_size = tree_size.wrapping_add(1);
            repetitions >>= 2i32;
            if repetitions == 0usize {
                break;
            }
            repetitions = repetitions.wrapping_sub(1);
        }
        Reverse(tree, start, *tree_size);
        Reverse(extra_bits_data, start, *tree_size);
    }
}

fn BrotliWriteHuffmanTreeRepetitionsZeros(
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    if repetitions == 11 {
        tree[*tree_size] = 0u8;
        extra_bits_data[*tree_size] = 0u8;
        *tree_size = tree_size.wrapping_add(1);
        repetitions = repetitions.wrapping_sub(1);
    }
    if repetitions < 3usize {
        for _i in 0usize..repetitions {
            tree[*tree_size] = 0u8;
            extra_bits_data[*tree_size] = 0u8;
            *tree_size = tree_size.wrapping_add(1);
        }
    } else {
        let start: usize = *tree_size;
        repetitions = repetitions.wrapping_sub(3);
        loop {
            tree[*tree_size] = 17u8;
            extra_bits_data[*tree_size] = (repetitions & 0x7usize) as u8;
            *tree_size = tree_size.wrapping_add(1);
            repetitions >>= 3i32;
            if repetitions == 0usize {
                break;
            }
            repetitions = repetitions.wrapping_sub(1);
        }
        Reverse(tree, start, *tree_size);
        Reverse(extra_bits_data, start, *tree_size);
    }
}

pub fn BrotliWriteHuffmanTree(
    depth: &[u8],
    length: usize,
    tree_size: &mut usize,
    tree: &mut [u8],
    extra_bits_data: &mut [u8],
) {
    let mut previous_value: u8 = 8u8;
    let mut i: usize;
    let mut use_rle_for_non_zero = false;
    let mut use_rle_for_zero = false;
    let mut new_length: usize = length;
    i = 0usize;
    'break27: while i < length {
        {
            if depth[length.wrapping_sub(i).wrapping_sub(1)] as i32 == 0i32 {
                new_length = new_length.wrapping_sub(1);
            } else {
                break 'break27;
            }
        }
        i = i.wrapping_add(1);
    }
    if length > 50 {
        (use_rle_for_non_zero, use_rle_for_zero) = decide_over_rle_use(depth, new_length);
    }
    i = 0usize;
    while i < new_length {
        let value: u8 = depth[i];
        let mut reps: usize = 1;
        if value != 0 && use_rle_for_non_zero || value == 0 && use_rle_for_zero {
            let mut k: usize;
            k = i.wrapping_add(1);
            while k < new_length && (depth[k] as i32 == value as i32) {
                {
                    reps = reps.wrapping_add(1);
                }
                k = k.wrapping_add(1);
            }
        }
        if value as i32 == 0i32 {
            BrotliWriteHuffmanTreeRepetitionsZeros(reps, tree_size, tree, extra_bits_data);
        } else {
            BrotliWriteHuffmanTreeRepetitions(
                previous_value,
                value,
                reps,
                tree_size,
                tree,
                extra_bits_data,
            );
            previous_value = value;
        }
        i = i.wrapping_add(reps);
    }
}

fn BrotliReverseBits(num_bits: usize, mut bits: u16) -> u16 {
    static kLut: [usize; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];
    let mut retval: usize = kLut[(bits as i32 & 0xfi32) as usize];
    let mut i: usize;
    i = 4usize;
    while i < num_bits {
        {
            retval <<= 4i32;
            bits = (bits as i32 >> 4) as u16;
            retval |= kLut[(bits as i32 & 0xfi32) as usize];
        }
        i = i.wrapping_add(4);
    }
    retval >>= (0usize.wrapping_sub(num_bits) & 0x3usize);
    retval as u16
}
const MAX_HUFFMAN_BITS: usize = 16;
pub fn BrotliConvertBitDepthsToSymbols(depth: &[u8], len: usize, bits: &mut [u16]) {
    /* In Brotli, all bit depths are [1..15]
    0 bit depth means that the symbol does not exist. */

    let mut bl_count: [u16; MAX_HUFFMAN_BITS] = [0; MAX_HUFFMAN_BITS];
    let mut next_code: [u16; MAX_HUFFMAN_BITS] = [0; MAX_HUFFMAN_BITS];
    let mut code: i32 = 0i32;
    for i in 0usize..len {
        let _rhs = 1;
        let _lhs = &mut bl_count[depth[i] as usize];
        *_lhs = (*_lhs as i32 + _rhs) as u16;
    }
    bl_count[0] = 0u16;
    next_code[0] = 0u16;
    for i in 1..MAX_HUFFMAN_BITS {
        code = (code + bl_count[i - 1] as i32) << 1;
        next_code[i] = code as u16;
    }
    for i in 0usize..len {
        if depth[i] != 0 {
            bits[i] = BrotliReverseBits(depth[i] as usize, {
                let _rhs = 1;
                let _lhs = &mut next_code[depth[i] as usize];
                let _old = *_lhs;
                *_lhs = (*_lhs as i32 + _rhs) as u16;
                _old
            });
        }
    }
}
