use core::cmp::min;

//caution: lots of the functions look structurally the same as two_pass,
// but have subtle index differences
// examples: IsMatch checks p1[4] and p1[5]
// the hoops that BuildAndStoreCommandPrefixCode goes through are subtly different in order
// (eg memcpy x+24, y instead of +24, y+40
// pretty much assume compress_fragment_two_pass is a trap! except for store_meta_block_header
use super::super::alloc;
use super::backward_references::kHashMul32;
use super::brotli_bit_stream::{BrotliBuildAndStoreHuffmanTreeFast, BrotliStoreHuffmanTree};
use super::compress_fragment_two_pass::{memcpy, BrotliWriteBits};
use super::entropy_encode::{
    BrotliConvertBitDepthsToSymbols, BrotliCreateHuffmanTree, HuffmanTree,
};
use super::static_dict::{
    FindMatchLengthWithLimit, BROTLI_UNALIGNED_LOAD32, BROTLI_UNALIGNED_LOAD64,
};
use super::util::{FastLog2, Log2FloorNonZero};
use crate::enc::compress_fragment_two_pass::store_meta_block_header;
use crate::enc::floatX;

//static kHashMul32: u32 = 0x1e35a7bdu32;

static kCmdHistoSeed: [u32; 128] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
];

fn Hash(p: &[u8], shift: usize) -> u32 {
    let h: u64 = (BROTLI_UNALIGNED_LOAD64(p) << 24).wrapping_mul(kHashMul32 as (u64));
    (h >> shift) as u32
}

fn IsMatch(p1: &[u8], p2: &[u8]) -> bool {
    BROTLI_UNALIGNED_LOAD32(p1) == BROTLI_UNALIGNED_LOAD32(p2) && (p1[4] as i32 == p2[4] as i32)
}

fn BuildAndStoreLiteralPrefixCode<AllocHT: alloc::Allocator<HuffmanTree>>(
    mht: &mut AllocHT,
    input: &[u8],
    input_size: usize,
    depths: &mut [u8],
    bits: &mut [u16],
    storage_ix: &mut usize,
    storage: &mut [u8],
) -> usize {
    let mut histogram: [u32; 256] = [0; 256];
    let mut histogram_total: usize;
    let mut i: usize;
    if input_size < (1i32 << 15) as usize {
        for i in 0usize..input_size {
            let _rhs = 1;
            let _lhs = &mut histogram[input[i] as usize];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
        histogram_total = input_size;
        i = 0usize;
        while i < 256usize {
            {
                let adjust: u32 = (2u32).wrapping_mul(min(histogram[i], 11u32));
                {
                    let _rhs = adjust;
                    let _lhs = &mut histogram[i];
                    *_lhs = (*_lhs).wrapping_add(_rhs);
                }
                histogram_total = histogram_total.wrapping_add(adjust as usize);
            }
            i = i.wrapping_add(1);
        }
    } else {
        static kSampleRate: usize = 29usize;
        i = 0usize;
        while i < input_size {
            {
                let _rhs = 1;
                let _lhs = &mut histogram[input[i] as usize];
                *_lhs = (*_lhs).wrapping_add(_rhs as u32);
            }
            i = i.wrapping_add(kSampleRate);
        }
        histogram_total = input_size
            .wrapping_add(kSampleRate)
            .wrapping_sub(1)
            .wrapping_div(kSampleRate);
        i = 0usize;
        while i < 256usize {
            {
                let adjust: u32 =
                    (1u32).wrapping_add((2u32).wrapping_mul(min(histogram[i], 11u32)));
                {
                    let _rhs = adjust;
                    let _lhs = &mut histogram[i];
                    *_lhs = (*_lhs).wrapping_add(_rhs);
                }
                histogram_total = histogram_total.wrapping_add(adjust as usize);
            }
            i = i.wrapping_add(1);
        }
    }
    BrotliBuildAndStoreHuffmanTreeFast(
        mht,
        &mut histogram[..],
        histogram_total,
        8usize,
        depths,
        bits,
        storage_ix,
        storage,
    );
    {
        let mut literal_ratio: usize = 0usize;
        for i in 0usize..256usize {
            if histogram[i] != 0 {
                literal_ratio = literal_ratio
                    .wrapping_add(histogram[i].wrapping_mul(depths[i] as u32) as usize);
            }
        }
        literal_ratio
            .wrapping_mul(125)
            .wrapping_div(histogram_total)
    }
}
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum CodeBlockState {
    EMIT_REMAINDER,
    EMIT_COMMANDS,
    NEXT_BLOCK,
}

fn EmitInsertLen(
    insertlen: usize,
    depth: &[u8],
    bits: &[u16],
    histo: &mut [u32],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    if insertlen < 6usize {
        let code: usize = insertlen.wrapping_add(40);
        BrotliWriteBits(
            depth[code] as usize,
            bits[code] as (u64),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[code];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if insertlen < 130usize {
        let tail: usize = insertlen.wrapping_sub(2);
        let nbits: u32 = Log2FloorNonZero(tail as u64).wrapping_sub(1);
        let prefix: usize = tail >> nbits;
        let inscode: usize = ((nbits << 1) as usize)
            .wrapping_add(prefix)
            .wrapping_add(42);
        BrotliWriteBits(
            depth[(inscode as usize)] as usize,
            bits[(inscode as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            (tail as u64).wrapping_sub((prefix as u64) << nbits),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[(inscode as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if insertlen < 2114usize {
        let tail: usize = insertlen.wrapping_sub(66);
        let nbits: u32 = Log2FloorNonZero(tail as u64);
        let code: usize = nbits.wrapping_add(50) as usize;
        BrotliWriteBits(
            depth[(code as usize)] as usize,
            bits[(code as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            (tail as u64).wrapping_sub(1 << nbits),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[(code as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else {
        BrotliWriteBits(depth[61] as usize, bits[61] as (u64), storage_ix, storage);
        BrotliWriteBits(
            12usize,
            (insertlen as u64).wrapping_sub(2114),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[61];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    }
}

fn ShouldUseUncompressedMode(delta: isize, insertlen: usize, literal_ratio: usize) -> bool {
    let compressed = delta as usize;
    if compressed.wrapping_mul(50) > insertlen {
        false
    } else if literal_ratio > 980 {
        true
    } else {
        false
    }
}
fn RewindBitPosition(new_storage_ix: usize, storage_ix: &mut usize, storage: &mut [u8]) {
    let bitpos: usize = new_storage_ix & 7usize;
    let mask: usize = (1u32 << bitpos).wrapping_sub(1) as usize;
    {
        let _rhs = mask as u8;
        let _lhs = &mut storage[(new_storage_ix >> 3)];
        *_lhs = (*_lhs as i32 & _rhs as i32) as u8;
    }
    *storage_ix = new_storage_ix;
}

fn EmitUncompressedMetaBlock(
    begin: &[u8],
    len: usize,
    storage_ix_start: usize,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    RewindBitPosition(storage_ix_start, storage_ix, storage);
    store_meta_block_header(len, true, storage_ix, storage);
    *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
    memcpy(storage, (*storage_ix >> 3), begin, 0, len);
    *storage_ix = storage_ix.wrapping_add(len << 3);
    storage[(*storage_ix >> 3)] = 0u8;
}

fn EmitLongInsertLen(
    insertlen: usize,
    depth: &[u8],
    bits: &[u16],
    histo: &mut [u32],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    if insertlen < 22594usize {
        BrotliWriteBits(depth[62] as usize, bits[62] as (u64), storage_ix, storage);
        BrotliWriteBits(
            14usize,
            (insertlen as u64).wrapping_sub(6210),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[62];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else {
        BrotliWriteBits(depth[63] as usize, bits[63] as (u64), storage_ix, storage);
        BrotliWriteBits(
            24usize,
            (insertlen as u64).wrapping_sub(22594),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[63];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    }
}

fn EmitLiterals(
    input: &[u8],
    len: usize,
    depth: &[u8],
    bits: &[u16],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    for j in 0usize..len {
        let lit: u8 = input[j];
        BrotliWriteBits(
            depth[(lit as usize)] as usize,
            bits[(lit as usize)] as (u64),
            storage_ix,
            storage,
        );
    }
}

fn EmitDistance(
    distance: usize,
    depth: &[u8],
    bits: &[u16],
    histo: &mut [u32],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let d: u64 = distance.wrapping_add(3) as u64;
    let nbits: u32 = Log2FloorNonZero(d).wrapping_sub(1);
    let prefix: u64 = d >> nbits & 1;
    let offset: u64 = (2u64).wrapping_add(prefix) << nbits;
    let distcode: u64 = ((2u32).wrapping_mul(nbits.wrapping_sub(1)) as (u64))
        .wrapping_add(prefix)
        .wrapping_add(80);
    BrotliWriteBits(
        depth[(distcode as usize)] as usize,
        bits[(distcode as usize)] as (u64),
        storage_ix,
        storage,
    );
    BrotliWriteBits(nbits as usize, d.wrapping_sub(offset), storage_ix, storage);
    {
        let _rhs = 1;
        let _lhs = &mut histo[(distcode as usize)];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
}

fn EmitCopyLenLastDistance(
    copylen: usize,
    depth: &[u8],
    bits: &[u16],
    histo: &mut [u32],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    if copylen < 12usize {
        BrotliWriteBits(
            depth[copylen.wrapping_sub(4)] as usize,
            bits[copylen.wrapping_sub(4)] as (u64),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[copylen.wrapping_sub(4)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if copylen < 72usize {
        let tail: usize = copylen.wrapping_sub(8);
        let nbits: u32 = Log2FloorNonZero(tail as u64).wrapping_sub(1);
        let prefix: usize = tail >> nbits;
        let code: usize = ((nbits << 1) as usize).wrapping_add(prefix).wrapping_add(4);
        BrotliWriteBits(
            depth[(code as usize)] as usize,
            bits[(code as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            tail.wrapping_sub(prefix << nbits) as u64,
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[(code as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if copylen < 136usize {
        let tail: usize = copylen.wrapping_sub(8);
        let code: usize = (tail >> 5).wrapping_add(30);
        BrotliWriteBits(
            depth[code] as usize,
            bits[code] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(5usize, tail as u64 & 31, storage_ix, storage);
        BrotliWriteBits(depth[64] as usize, bits[64] as (u64), storage_ix, storage);
        {
            let _rhs = 1;
            let _lhs = &mut histo[code];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
        {
            let _rhs = 1;
            let _lhs = &mut histo[64];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if copylen < 2120usize {
        let tail: usize = copylen.wrapping_sub(72);
        let nbits: u32 = Log2FloorNonZero(tail as u64);
        let code: usize = nbits.wrapping_add(28) as usize;
        BrotliWriteBits(
            depth[(code as usize)] as usize,
            bits[(code as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            (tail as u64).wrapping_sub(1u64 << nbits),
            storage_ix,
            storage,
        );
        BrotliWriteBits(depth[64] as usize, bits[64] as (u64), storage_ix, storage);
        {
            let _rhs = 1;
            let _lhs = &mut histo[(code as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
        {
            let _rhs = 1;
            let _lhs = &mut histo[64];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else {
        BrotliWriteBits(depth[39] as usize, bits[39] as (u64), storage_ix, storage);
        BrotliWriteBits(
            24usize,
            copylen.wrapping_sub(2120) as u64,
            storage_ix,
            storage,
        );
        BrotliWriteBits(depth[64] as usize, bits[64] as (u64), storage_ix, storage);
        {
            let _rhs = 1;
            let _lhs = &mut histo[39];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
        {
            let _rhs = 1;
            let _lhs = &mut histo[64];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    }
}

fn HashBytesAtOffset(v: u64, offset: i32, shift: usize) -> u32 {
    let h: u64 = (v >> (8i32 * offset) << 24).wrapping_mul(kHashMul32 as (u64));
    (h >> shift) as u32
}

fn EmitCopyLen(
    copylen: usize,
    depth: &[u8],
    bits: &[u16],
    histo: &mut [u32],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    if copylen < 10usize {
        BrotliWriteBits(
            depth[copylen.wrapping_add(14)] as usize,
            bits[copylen.wrapping_add(14)] as (u64),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[copylen.wrapping_add(14)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if copylen < 134usize {
        let tail: usize = copylen.wrapping_sub(6);
        let nbits: u32 = Log2FloorNonZero(tail as u64).wrapping_sub(1);
        let prefix: usize = tail >> nbits;
        let code: usize = ((nbits << 1) as usize)
            .wrapping_add(prefix)
            .wrapping_add(20);
        BrotliWriteBits(
            depth[(code as usize)] as usize,
            bits[(code as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            (tail as u64).wrapping_sub((prefix as u64) << nbits),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[(code as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else if copylen < 2118usize {
        let tail: usize = copylen.wrapping_sub(70);
        let nbits: u32 = Log2FloorNonZero(tail as u64);
        let code: usize = nbits.wrapping_add(28) as usize;
        BrotliWriteBits(
            depth[(code as usize)] as usize,
            bits[(code as usize)] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            nbits as usize,
            (tail as u64).wrapping_sub(1 << nbits),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[(code as usize)];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    } else {
        BrotliWriteBits(depth[39] as usize, bits[39] as (u64), storage_ix, storage);
        BrotliWriteBits(
            24usize,
            (copylen as u64).wrapping_sub(2118),
            storage_ix,
            storage,
        );
        {
            let _rhs = 1;
            let _lhs = &mut histo[39];
            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
        }
    }
}

fn ShouldMergeBlock(data: &[u8], len: usize, depths: &[u8]) -> bool {
    let mut histo: [usize; 256] = [0; 256];
    static kSampleRate: usize = 43usize;
    let mut i: usize;
    i = 0usize;
    while i < len {
        {
            let _rhs = 1;
            let _lhs = &mut histo[data[i] as usize];
            *_lhs = (*_lhs).wrapping_add(_rhs as usize);
        }
        i = i.wrapping_add(kSampleRate);
    }
    {
        let total: usize = len
            .wrapping_add(kSampleRate)
            .wrapping_sub(1)
            .wrapping_div(kSampleRate);
        let mut r: floatX = (FastLog2(total as u64) + 0.5) * (total as floatX) + 200.0;
        for i in 0usize..256usize {
            r -= (histo[i] as floatX) * ((depths[i] as floatX) + FastLog2(histo[i] as u64));
        }
        r >= 0.0
    }
}

fn UpdateBits(mut n_bits: usize, mut bits: u32, mut pos: usize, array: &mut [u8]) {
    while n_bits > 0usize {
        let byte_pos: usize = pos >> 3;
        let n_unchanged_bits: usize = pos & 7usize;
        let n_changed_bits: usize = min(n_bits, (8usize).wrapping_sub(n_unchanged_bits));
        let total_bits: usize = n_unchanged_bits.wrapping_add(n_changed_bits);
        let mask: u32 =
            !(1u32 << total_bits).wrapping_sub(1) | (1u32 << n_unchanged_bits).wrapping_sub(1);
        let unchanged_bits: u32 = array[byte_pos] as u32 & mask;
        let changed_bits: u32 = bits & (1u32 << n_changed_bits).wrapping_sub(1);
        array[byte_pos] = (changed_bits << n_unchanged_bits | unchanged_bits) as u8;
        n_bits = n_bits.wrapping_sub(n_changed_bits);
        bits >>= n_changed_bits;
        pos = pos.wrapping_add(n_changed_bits);
    }
}

fn BuildAndStoreCommandPrefixCode(
    histogram: &[u32],
    depth: &mut [u8],
    bits: &mut [u16],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let mut tree = [HuffmanTree::new(0, 0, 0); 129];
    let mut cmd_depth: [u8; 704] = [0u8; 704];

    let mut cmd_bits: [u16; 64] = [0; 64];
    BrotliCreateHuffmanTree(histogram, 64usize, 15i32, &mut tree[..], depth);
    BrotliCreateHuffmanTree(
        &histogram[64..],
        64usize,
        14i32,
        &mut tree[..],
        &mut depth[64..],
    );
    /* We have to jump through a few hoops here in order to compute
    the command bits because the symbols are in a different order than in
    the full alphabet. This looks complicated, but having the symbols
    in this order in the command bits saves a few branches in the Emit*
    functions. */
    memcpy(&mut cmd_depth[..], 0, depth, 0, 24usize);
    memcpy(&mut cmd_depth[..], 24usize, depth, (40usize), 8usize);
    memcpy(&mut cmd_depth[..], 32usize, depth, (24usize), 8usize);
    memcpy(&mut cmd_depth[..], 40usize, depth, (48usize), 8usize);
    memcpy(&mut cmd_depth[..], 48usize, depth, (32usize), 8usize);
    memcpy(&mut cmd_depth[..], 56usize, depth, (56usize), 8usize);
    BrotliConvertBitDepthsToSymbols(&mut cmd_depth[..], 64usize, &mut cmd_bits[..]);
    memcpy(bits, 0, &cmd_bits[..], 0, 24usize);
    memcpy(bits, (24usize), &cmd_bits[..], 32usize, 8usize);
    memcpy(bits, (32usize), &cmd_bits[..], 48usize, 8usize);
    memcpy(bits, (40usize), &cmd_bits[..], 24usize, 8usize);
    memcpy(bits, (48usize), &cmd_bits[..], 40usize, 8usize);
    memcpy(bits, (56usize), &cmd_bits[..], 56usize, 8usize);
    BrotliConvertBitDepthsToSymbols(&mut depth[64..], 64usize, &mut bits[64..]);
    {
        for item in cmd_depth[..64].iter_mut() {
            *item = 0;
        }
        memcpy(&mut cmd_depth[..], 0, depth, 0, 8usize);
        memcpy(&mut cmd_depth[..], 64usize, depth, (8usize), 8usize);
        memcpy(&mut cmd_depth[..], 128usize, depth, (16usize), 8usize);
        memcpy(&mut cmd_depth[..], 192usize, depth, (24usize), 8usize);
        memcpy(&mut cmd_depth[..], 384usize, depth, (32usize), 8usize);
        for i in 0usize..8usize {
            cmd_depth[(128usize).wrapping_add((8usize).wrapping_mul(i))] =
                depth[i.wrapping_add(40)];
            cmd_depth[(256usize).wrapping_add((8usize).wrapping_mul(i))] =
                depth[i.wrapping_add(48)];
            cmd_depth[(448usize).wrapping_add((8usize).wrapping_mul(i))] =
                depth[i.wrapping_add(56)];
        }
        BrotliStoreHuffmanTree(
            &mut cmd_depth[..],
            704usize,
            &mut tree[..],
            storage_ix,
            storage,
        );
    }
    BrotliStoreHuffmanTree(
        &mut depth[64..],
        64usize,
        &mut tree[..],
        storage_ix,
        storage,
    );
}

#[allow(unused_assignments)]
fn compress_fragment_fast_impl<AllocHT: alloc::Allocator<HuffmanTree>>(
    m: &mut AllocHT,
    input_ptr: &[u8],
    mut input_size: usize,
    is_last: bool,
    table: &mut [i32],
    table_bits: usize,
    cmd_depth: &mut [u8],
    cmd_bits: &mut [u16],
    cmd_code_numbits: &mut usize,
    cmd_code: &mut [u8],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let mut cmd_histo = [0u32; 128];
    let mut ip_end = 0usize;
    let mut next_emit = 0usize;
    let base_ip = 0usize;
    static kFirstBlockSize: usize = (3i32 << 15) as usize;
    static kMergeBlockSize: usize = (1i32 << 16) as usize;
    let kInputMarginBytes = 16usize;
    let kMinMatchLen = 5usize;
    let mut metablock_start = 0usize;
    let mut block_size = min(input_size, kFirstBlockSize);
    let mut total_block_size = block_size;
    let mut mlen_storage_ix = storage_ix.wrapping_add(3);
    let mut lit_depth = [0u8; 256];
    let mut lit_bits = [0u16; 256];
    let mut literal_ratio: usize;
    let mut input_index = 0usize;
    let mut last_distance: i32;
    let shift: usize = (64u32 as usize).wrapping_sub(table_bits);
    store_meta_block_header(block_size, false, storage_ix, storage);
    BrotliWriteBits(13usize, 0, storage_ix, storage);
    literal_ratio = BuildAndStoreLiteralPrefixCode(
        m,
        &input_ptr[input_index..],
        block_size,
        &mut lit_depth[..],
        &mut lit_bits[..],
        storage_ix,
        storage,
    );
    {
        let mut i = 0usize;
        while i.wrapping_add(7) < *cmd_code_numbits {
            BrotliWriteBits(8usize, cmd_code[i >> 3] as u64, storage_ix, storage);
            i = i.wrapping_add(8);
        }
    }
    BrotliWriteBits(
        *cmd_code_numbits & 7usize,
        cmd_code[*cmd_code_numbits >> 3] as u64,
        storage_ix,
        storage,
    );
    let mut code_block_selection = CodeBlockState::EMIT_COMMANDS;
    'continue_to_next_block: loop {
        let mut ip_index: usize;
        if code_block_selection == CodeBlockState::EMIT_COMMANDS {
            cmd_histo[..128].clone_from_slice(&kCmdHistoSeed[..]);
            ip_index = input_index;
            last_distance = -1i32;
            ip_end = input_index.wrapping_add(block_size);
            if block_size >= kInputMarginBytes {
                let len_limit: usize = min(
                    block_size.wrapping_sub(kMinMatchLen),
                    input_size.wrapping_sub(kInputMarginBytes),
                );
                let ip_limit: usize = input_index.wrapping_add(len_limit);
                let mut next_hash = Hash(
                    &input_ptr[{
                        ip_index = ip_index.wrapping_add(1);
                        ip_index
                    }..],
                    shift,
                );
                loop {
                    let mut skip = 32u32;
                    let mut next_ip = ip_index;
                    let mut candidate = 0usize;
                    loop {
                        loop {
                            let hash = next_hash;
                            let bytes_between_hash_lookups: u32 = skip >> 5;
                            skip = skip.wrapping_add(1);
                            ip_index = next_ip;
                            next_ip = ip_index.wrapping_add(bytes_between_hash_lookups as usize);
                            if next_ip > ip_limit {
                                code_block_selection = CodeBlockState::EMIT_REMAINDER;
                                break;
                            }
                            next_hash = Hash(&input_ptr[next_ip..], shift);
                            candidate = ip_index.wrapping_sub(last_distance as usize);
                            if IsMatch(&input_ptr[ip_index..], &input_ptr[candidate..])
                                && candidate < ip_index
                            {
                                table[hash as usize] = ip_index.wrapping_sub(base_ip) as i32;
                                break;
                            }
                            candidate = base_ip.wrapping_add(table[hash as usize] as usize);
                            table[hash as usize] = ip_index.wrapping_sub(base_ip) as i32;
                            if IsMatch(&input_ptr[ip_index..], &input_ptr[candidate..]) {
                                break;
                            }
                        }
                        if !(ip_index.wrapping_sub(candidate)
                            > (1usize << 18).wrapping_sub(16) as isize as usize
                            && code_block_selection == CodeBlockState::EMIT_COMMANDS)
                        {
                            break;
                        }
                    }
                    if code_block_selection != CodeBlockState::EMIT_COMMANDS {
                        break;
                    }

                    let base: usize = ip_index;
                    let matched = (5usize).wrapping_add(FindMatchLengthWithLimit(
                        &input_ptr[candidate + 5..],
                        &input_ptr[ip_index + 5..],
                        ip_end.wrapping_sub(ip_index).wrapping_sub(5),
                    ));
                    let distance = base.wrapping_sub(candidate) as i32;
                    let insert = base.wrapping_sub(next_emit);
                    ip_index = ip_index.wrapping_add(matched);
                    if insert < 6210 {
                        EmitInsertLen(
                            insert,
                            cmd_depth,
                            cmd_bits,
                            &mut cmd_histo[..],
                            storage_ix,
                            storage,
                        );
                    } else if ShouldUseUncompressedMode(
                        (next_emit as isize) - (metablock_start as isize),
                        insert,
                        literal_ratio,
                    ) {
                        EmitUncompressedMetaBlock(
                            &input_ptr[metablock_start..],
                            base.wrapping_sub(metablock_start),
                            mlen_storage_ix.wrapping_sub(3),
                            storage_ix,
                            storage,
                        );
                        input_size = input_size.wrapping_sub(base.wrapping_sub(input_index));
                        input_index = base;
                        next_emit = input_index;
                        code_block_selection = CodeBlockState::NEXT_BLOCK;
                        continue 'continue_to_next_block;
                    } else {
                        EmitLongInsertLen(
                            insert,
                            cmd_depth,
                            cmd_bits,
                            &mut cmd_histo[..],
                            storage_ix,
                            storage,
                        );
                    }
                    EmitLiterals(
                        &input_ptr[next_emit..],
                        insert,
                        &mut lit_depth[..],
                        &mut lit_bits[..],
                        storage_ix,
                        storage,
                    );
                    if distance == last_distance {
                        BrotliWriteBits(
                            cmd_depth[64] as usize,
                            cmd_bits[64] as u64,
                            storage_ix,
                            storage,
                        );
                        {
                            let _rhs = 1u32;
                            let _lhs = &mut cmd_histo[64];
                            *_lhs = (*_lhs).wrapping_add(_rhs);
                        }
                    } else {
                        EmitDistance(
                            distance as usize,
                            cmd_depth,
                            cmd_bits,
                            &mut cmd_histo[..],
                            storage_ix,
                            storage,
                        );
                        last_distance = distance;
                    }
                    EmitCopyLenLastDistance(
                        matched,
                        cmd_depth,
                        cmd_bits,
                        &mut cmd_histo[..],
                        storage_ix,
                        storage,
                    );
                    next_emit = ip_index;
                    if ip_index >= ip_limit {
                        code_block_selection = CodeBlockState::EMIT_REMAINDER;
                        continue 'continue_to_next_block;
                    }

                    assert!(ip_index >= 3);
                    let input_bytes: u64 = BROTLI_UNALIGNED_LOAD64(&input_ptr[ip_index - 3..]);
                    let mut prev_hash: u32 = HashBytesAtOffset(input_bytes, 0, shift);
                    let cur_hash: u32 = HashBytesAtOffset(input_bytes, 3, shift);
                    table[prev_hash as usize] =
                        ip_index.wrapping_sub(base_ip).wrapping_sub(3) as i32;
                    prev_hash = HashBytesAtOffset(input_bytes, 1, shift);
                    table[prev_hash as usize] =
                        ip_index.wrapping_sub(base_ip).wrapping_sub(2) as i32;
                    prev_hash = HashBytesAtOffset(input_bytes, 2, shift);
                    table[prev_hash as usize] =
                        ip_index.wrapping_sub(base_ip).wrapping_sub(1) as i32;
                    candidate = base_ip.wrapping_add(table[cur_hash as usize] as usize);
                    table[cur_hash as usize] = ip_index.wrapping_sub(base_ip) as i32;

                    while IsMatch(&input_ptr[ip_index..], &input_ptr[candidate..]) {
                        let base: usize = ip_index;
                        let matched: usize = (5usize).wrapping_add(FindMatchLengthWithLimit(
                            &input_ptr[candidate + 5..],
                            &input_ptr[ip_index + 5..],
                            ip_end.wrapping_sub(ip_index).wrapping_sub(5),
                        ));
                        if ip_index.wrapping_sub(candidate) > (1usize << 18).wrapping_sub(16) {
                            break;
                        }
                        ip_index = ip_index.wrapping_add(matched);
                        last_distance = base.wrapping_sub(candidate) as i32;
                        EmitCopyLen(
                            matched,
                            cmd_depth,
                            cmd_bits,
                            &mut cmd_histo[..],
                            storage_ix,
                            storage,
                        );
                        EmitDistance(
                            last_distance as usize,
                            cmd_depth,
                            cmd_bits,
                            &mut cmd_histo[..],
                            storage_ix,
                            storage,
                        );
                        next_emit = ip_index;
                        if ip_index >= ip_limit {
                            code_block_selection = CodeBlockState::EMIT_REMAINDER;
                            continue 'continue_to_next_block;
                        }

                        assert!(ip_index >= 3);
                        let input_bytes: u64 = BROTLI_UNALIGNED_LOAD64(&input_ptr[ip_index - 3..]);
                        let mut prev_hash: u32 = HashBytesAtOffset(input_bytes, 0, shift);
                        let cur_hash: u32 = HashBytesAtOffset(input_bytes, 3, shift);
                        table[prev_hash as usize] =
                            ip_index.wrapping_sub(base_ip).wrapping_sub(3) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1, shift);
                        table[prev_hash as usize] =
                            ip_index.wrapping_sub(base_ip).wrapping_sub(2) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 2, shift);
                        table[prev_hash as usize] =
                            ip_index.wrapping_sub(base_ip).wrapping_sub(1) as i32;
                        candidate = base_ip.wrapping_add(table[cur_hash as usize] as usize);
                        table[cur_hash as usize] = ip_index.wrapping_sub(base_ip) as i32;
                    }
                    if code_block_selection == CodeBlockState::EMIT_REMAINDER {
                        break;
                    }
                    if code_block_selection == CodeBlockState::EMIT_COMMANDS {
                        next_hash = Hash(
                            &input_ptr[{
                                ip_index = ip_index.wrapping_add(1);
                                ip_index
                            }..],
                            shift,
                        );
                    }
                }
            }
            code_block_selection = CodeBlockState::EMIT_REMAINDER;
            continue 'continue_to_next_block;
        } else if code_block_selection == CodeBlockState::EMIT_REMAINDER {
            input_index = input_index.wrapping_add(block_size);
            input_size = input_size.wrapping_sub(block_size);
            block_size = min(input_size, kMergeBlockSize);
            if input_size > 0
                && (total_block_size.wrapping_add(block_size) <= (1i32 << 20) as usize)
                && ShouldMergeBlock(&input_ptr[input_index..], block_size, &mut lit_depth[..])
            {
                total_block_size = total_block_size.wrapping_add(block_size);
                UpdateBits(
                    20usize,
                    total_block_size.wrapping_sub(1) as u32,
                    mlen_storage_ix,
                    storage,
                );
                code_block_selection = CodeBlockState::EMIT_COMMANDS;
                continue 'continue_to_next_block;
            }
            if next_emit < ip_end {
                let insert: usize = ip_end.wrapping_sub(next_emit);
                if insert < 6210 {
                    EmitInsertLen(
                        insert,
                        cmd_depth,
                        cmd_bits,
                        &mut cmd_histo[..],
                        storage_ix,
                        storage,
                    );
                    EmitLiterals(
                        &input_ptr[next_emit..],
                        insert,
                        &mut lit_depth[..],
                        &mut lit_bits[..],
                        storage_ix,
                        storage,
                    );
                } else if ShouldUseUncompressedMode(
                    next_emit as isize - metablock_start as isize,
                    insert,
                    literal_ratio,
                ) {
                    EmitUncompressedMetaBlock(
                        &input_ptr[metablock_start..],
                        ip_end.wrapping_sub(metablock_start),
                        mlen_storage_ix.wrapping_sub(3),
                        storage_ix,
                        storage,
                    );
                } else {
                    EmitLongInsertLen(
                        insert,
                        cmd_depth,
                        cmd_bits,
                        &mut cmd_histo[..],
                        storage_ix,
                        storage,
                    );
                    EmitLiterals(
                        &input_ptr[next_emit..],
                        insert,
                        &mut lit_depth[..],
                        &mut lit_bits[..],
                        storage_ix,
                        storage,
                    );
                }
            }
            next_emit = ip_end;
            code_block_selection = CodeBlockState::NEXT_BLOCK;
            continue 'continue_to_next_block;
        } else if code_block_selection == CodeBlockState::NEXT_BLOCK {
            if input_size > 0 {
                metablock_start = input_index;
                block_size = min(input_size, kFirstBlockSize);
                total_block_size = block_size;
                mlen_storage_ix = storage_ix.wrapping_add(3);
                store_meta_block_header(block_size, false, storage_ix, storage);
                BrotliWriteBits(13usize, 0, storage_ix, storage);
                literal_ratio = BuildAndStoreLiteralPrefixCode(
                    m,
                    &input_ptr[input_index..],
                    block_size,
                    &mut lit_depth[..],
                    &mut lit_bits[..],
                    storage_ix,
                    storage,
                );
                BuildAndStoreCommandPrefixCode(
                    &mut cmd_histo[..],
                    cmd_depth,
                    cmd_bits,
                    storage_ix,
                    storage,
                );
                code_block_selection = CodeBlockState::EMIT_COMMANDS;
                continue 'continue_to_next_block;
            }
            break;
        }
    }
    if !is_last {
        cmd_code[0] = 0;
        *cmd_code_numbits = 0;
        BuildAndStoreCommandPrefixCode(
            &mut cmd_histo[..],
            cmd_depth,
            cmd_bits,
            cmd_code_numbits,
            cmd_code,
        );
    }
}

macro_rules! compress_specialization {
    ($table_bits : expr, $fname: ident) => {
        fn $fname<AllocHT: alloc::Allocator<HuffmanTree>>(
            mht: &mut AllocHT,
            input: &[u8],
            input_size: usize,
            is_last: bool,
            table: &mut [i32],
            cmd_depth: &mut [u8],
            cmd_bits: &mut [u16],
            cmd_code_numbits: &mut usize,
            cmd_code: &mut [u8],
            storage_ix: &mut usize,
            storage: &mut [u8],
        ) {
            compress_fragment_fast_impl(
                mht,
                input,
                input_size,
                is_last,
                table,
                $table_bits,
                cmd_depth,
                cmd_bits,
                cmd_code_numbits,
                cmd_code,
                storage_ix,
                storage,
            );
        }
    };
}

compress_specialization!(9, BrotliCompressFragmentFastImpl9);
compress_specialization!(11, BrotliCompressFragmentFastImpl11);
compress_specialization!(13, BrotliCompressFragmentFastImpl13);
compress_specialization!(15, BrotliCompressFragmentFastImpl15);


pub(crate) fn compress_fragment_fast<AllocHT: alloc::Allocator<HuffmanTree>>(
    m: &mut AllocHT,
    input: &[u8],
    input_size: usize,
    is_last: bool,
    table: &mut [i32],
    table_size: usize,
    cmd_depth: &mut [u8],
    cmd_bits: &mut [u16],
    cmd_code_numbits: &mut usize,
    cmd_code: &mut [u8],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let initial_storage_ix: usize = *storage_ix;
    let table_bits: usize = Log2FloorNonZero(table_size as u64) as usize;
    if input_size == 0usize {
        BrotliWriteBits(1usize, 1, storage_ix, storage);
        BrotliWriteBits(1usize, 1, storage_ix, storage);
        *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
        return;
    }
    if table_bits == 9usize {
        BrotliCompressFragmentFastImpl9(
            m,
            input,
            input_size,
            is_last,
            table,
            cmd_depth,
            cmd_bits,
            cmd_code_numbits,
            cmd_code,
            storage_ix,
            storage,
        );
    }
    if table_bits == 11usize {
        BrotliCompressFragmentFastImpl11(
            m,
            input,
            input_size,
            is_last,
            table,
            cmd_depth,
            cmd_bits,
            cmd_code_numbits,
            cmd_code,
            storage_ix,
            storage,
        );
    }
    if table_bits == 13usize {
        BrotliCompressFragmentFastImpl13(
            m,
            input,
            input_size,
            is_last,
            table,
            cmd_depth,
            cmd_bits,
            cmd_code_numbits,
            cmd_code,
            storage_ix,
            storage,
        );
    }
    if table_bits == 15usize {
        BrotliCompressFragmentFastImpl15(
            m,
            input,
            input_size,
            is_last,
            table,
            cmd_depth,
            cmd_bits,
            cmd_code_numbits,
            cmd_code,
            storage_ix,
            storage,
        );
    }
    if storage_ix.wrapping_sub(initial_storage_ix) > (31usize).wrapping_add(input_size << 3) {
        EmitUncompressedMetaBlock(input, input_size, initial_storage_ix, storage_ix, storage);
    }
    if is_last {
        BrotliWriteBits(1usize, 1, storage_ix, storage);
        BrotliWriteBits(1usize, 1, storage_ix, storage);
        *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
    }
}
