use core;
use core::cmp::min;

use super::super::alloc;
use super::backward_references::kHashMul32;
use super::bit_cost::BitsEntropy;
use super::brotli_bit_stream::{BrotliBuildAndStoreHuffmanTreeFast, BrotliStoreHuffmanTree};
use super::entropy_encode::{
    BrotliConvertBitDepthsToSymbols, BrotliCreateHuffmanTree, HuffmanTree,
};
use super::static_dict::{
    FindMatchLengthWithLimit, BROTLI_UNALIGNED_LOAD32, BROTLI_UNALIGNED_LOAD64,
    BROTLI_UNALIGNED_STORE64,
};
use super::util::{floatX, Log2FloorNonZero};
static kCompressFragmentTwoPassBlockSize: usize = (1i32 << 17) as usize;

// returns number of commands inserted
fn EmitInsertLen(insertlen: u32, commands: &mut &mut [u32]) -> usize {
    if insertlen < 6u32 {
        (*commands)[0] = insertlen;
    } else if insertlen < 130u32 {
        let tail: u32 = insertlen.wrapping_sub(2);
        let nbits: u32 = Log2FloorNonZero(tail as (u64)).wrapping_sub(1);
        let prefix: u32 = tail >> nbits;
        let inscode: u32 = (nbits << 1).wrapping_add(prefix).wrapping_add(2);
        let extra: u32 = tail.wrapping_sub(prefix << nbits);
        (*commands)[0] = inscode | extra << 8;
    } else if insertlen < 2114u32 {
        let tail: u32 = insertlen.wrapping_sub(66);
        let nbits: u32 = Log2FloorNonZero(tail as (u64));
        let code: u32 = nbits.wrapping_add(10);
        let extra: u32 = tail.wrapping_sub(1u32 << nbits);
        (*commands)[0] = code | extra << 8;
    } else if insertlen < 6210u32 {
        let extra: u32 = insertlen.wrapping_sub(2114);
        (*commands)[0] = 21u32 | extra << 8;
    } else if insertlen < 22594u32 {
        let extra: u32 = insertlen.wrapping_sub(6210);
        (*commands)[0] = 22u32 | extra << 8;
    } else {
        let extra: u32 = insertlen.wrapping_sub(22594);
        (*commands)[0] = 23u32 | extra << 8;
    }
    let remainder = core::mem::take(commands);
    let _ = core::mem::replace(commands, &mut remainder[1..]);
    1
}

fn EmitDistance(distance: u32, commands: &mut &mut [u32]) -> usize {
    let d: u32 = distance.wrapping_add(3);
    let nbits: u32 = Log2FloorNonZero(d as (u64)).wrapping_sub(1);
    let prefix: u32 = d >> nbits & 1u32;
    let offset: u32 = (2u32).wrapping_add(prefix) << nbits;
    let distcode: u32 = (2u32)
        .wrapping_mul(nbits.wrapping_sub(1))
        .wrapping_add(prefix)
        .wrapping_add(80);
    let extra: u32 = d.wrapping_sub(offset);
    (*commands)[0] = distcode | extra << 8;
    let remainder = core::mem::take(commands);
    let _ = core::mem::replace(commands, &mut remainder[1..]);
    1
}

fn EmitCopyLenLastDistance(copylen: usize, commands: &mut &mut [u32]) -> usize {
    if copylen < 12usize {
        (*commands)[0] = copylen.wrapping_add(20) as u32;
        let remainder = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder[1..]);
        1
    } else if copylen < 72usize {
        let tail: usize = copylen.wrapping_sub(8);
        let nbits: usize = Log2FloorNonZero(tail as u64).wrapping_sub(1) as usize;
        let prefix: usize = tail >> nbits;
        let code: usize = (nbits << 1).wrapping_add(prefix).wrapping_add(28);
        let extra: usize = tail.wrapping_sub(prefix << nbits);
        (*commands)[0] = (code | extra << 8) as u32;
        let remainder = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder[1..]);
        1
    } else if copylen < 136usize {
        let tail: usize = copylen.wrapping_sub(8);
        let code: usize = (tail >> 5).wrapping_add(54);
        let extra: usize = tail & 31usize;
        (*commands)[0] = (code | extra << 8) as u32;
        let remainder = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder[1..]);
        (*commands)[0] = 64u32;
        let remainder2 = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder2[1..]);
        2
    } else if copylen < 2120usize {
        let tail: usize = copylen.wrapping_sub(72);
        let nbits: usize = Log2FloorNonZero(tail as u64) as usize;
        let code: usize = nbits.wrapping_add(52);
        let extra: usize = tail.wrapping_sub(1usize << nbits);
        (*commands)[0] = (code | extra << 8) as u32;
        let remainder = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder[1..]);
        (*commands)[0] = 64u32;
        let remainder2 = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder2[1..]);
        2
    } else {
        let extra: usize = copylen.wrapping_sub(2120);
        (*commands)[0] = (63usize | extra << 8) as u32;
        let remainder = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder[1..]);
        (*commands)[0] = 64u32;
        let remainder2 = core::mem::take(commands);
        let _ = core::mem::replace(commands, &mut remainder2[1..]);
        2
    }
}
fn HashBytesAtOffset(v: u64, offset: i32, shift: usize, length: usize) -> u32 {
    let h: u64 = (v >> (8i32 * offset) << ((8 - length) * 8)).wrapping_mul(kHashMul32 as (u64));
    (h >> shift) as u32
}

fn EmitCopyLen(copylen: usize, commands: &mut &mut [u32]) -> usize {
    if copylen < 10usize {
        (*commands)[0] = copylen.wrapping_add(38) as u32;
    } else if copylen < 134usize {
        let tail: usize = copylen.wrapping_sub(6);
        let nbits: usize = Log2FloorNonZero(tail as u64).wrapping_sub(1) as usize;
        let prefix: usize = tail >> nbits;
        let code: usize = (nbits << 1).wrapping_add(prefix).wrapping_add(44);
        let extra: usize = tail.wrapping_sub(prefix << nbits);
        (*commands)[0] = (code | extra << 8) as u32;
    } else if copylen < 2118usize {
        let tail: usize = copylen.wrapping_sub(70);
        let nbits: usize = Log2FloorNonZero(tail as u64) as usize;
        let code: usize = nbits.wrapping_add(52);
        let extra: usize = tail.wrapping_sub(1usize << nbits);
        (*commands)[0] = (code | extra << 8) as u32;
    } else {
        let extra: usize = copylen.wrapping_sub(2118);
        (*commands)[0] = (63usize | extra << 8) as u32;
    }
    let remainder = core::mem::take(commands);
    let _ = core::mem::replace(commands, &mut remainder[1..]);
    1
}
fn Hash(p: &[u8], shift: usize, length: usize) -> u32 {
    let h: u64 =
        (BROTLI_UNALIGNED_LOAD64(p) << ((8 - length) * 8)).wrapping_mul(kHashMul32 as (u64));
    (h >> shift) as u32
}

fn IsMatch(p1: &[u8], p2: &[u8], length: usize) -> bool {
    BROTLI_UNALIGNED_LOAD32(p1) == BROTLI_UNALIGNED_LOAD32(p2)
        && (length == 4 || (p1[4] == p2[4] && p1[5] == p2[5]))
}

#[allow(unused_assignments)]
fn CreateCommands(
    input_index: usize,
    block_size: usize,
    input_size: usize,
    base_ip: &[u8],
    table: &mut [i32],
    table_bits: usize,
    min_match: usize,
    literals: &mut &mut [u8],
    num_literals: &mut usize,
    commands: &mut &mut [u32],
    num_commands: &mut usize,
) {
    let mut ip_index: usize = input_index;
    let shift: usize = (64u32 as usize).wrapping_sub(table_bits);
    let ip_end: usize = input_index.wrapping_add(block_size);
    let mut next_emit: usize = input_index;
    let mut last_distance: i32 = -1i32;
    let kInputMarginBytes: usize = 16usize;

    if block_size >= kInputMarginBytes {
        let len_limit: usize = min(
            block_size.wrapping_sub(min_match),
            input_size.wrapping_sub(kInputMarginBytes),
        );
        let ip_limit: usize = input_index.wrapping_add(len_limit);
        let mut next_hash: u32;
        let mut goto_emit_remainder = false;
        next_hash = Hash(
            &base_ip[{
                ip_index = ip_index.wrapping_add(1);
                ip_index
            }..],
            shift,
            min_match,
        );
        while !goto_emit_remainder {
            let mut skip: u32 = 32u32;
            let mut next_ip: usize = ip_index;
            let mut candidate: usize = 0;
            loop {
                {
                    'break3: loop {
                        {
                            let hash: u32 = next_hash;
                            let bytes_between_hash_lookups: u32 = skip >> 5;
                            skip = skip.wrapping_add(1);
                            ip_index = next_ip;
                            next_ip = ip_index.wrapping_add(bytes_between_hash_lookups as usize);
                            if next_ip > ip_limit {
                                goto_emit_remainder = true;
                                {
                                    break 'break3;
                                }
                            }
                            next_hash = Hash(&base_ip[next_ip..], shift, min_match);
                            candidate = ip_index.wrapping_sub(last_distance as usize);
                            if IsMatch(&base_ip[ip_index..], &base_ip[candidate..], min_match)
                                && candidate < ip_index
                            {
                                table[(hash as usize)] = ip_index.wrapping_sub(0) as i32;
                                {
                                    break 'break3;
                                }
                            }
                            candidate = table[(hash as usize)] as usize;
                            table[(hash as usize)] = ip_index.wrapping_sub(0) as i32;
                        }
                        if IsMatch(&base_ip[ip_index..], &base_ip[candidate..], min_match) {
                            break;
                        }
                    }
                }
                if !(ip_index.wrapping_sub(candidate)
                    > (1usize << 18).wrapping_sub(16) as isize as usize
                    && !goto_emit_remainder)
                {
                    break;
                }
            }
            if goto_emit_remainder {
                break;
            }
            {
                let base: usize = ip_index;
                let matched: usize = min_match.wrapping_add(FindMatchLengthWithLimit(
                    &base_ip[(candidate + min_match)..],
                    &base_ip[(ip_index + min_match)..],
                    ip_end.wrapping_sub(ip_index).wrapping_sub(min_match),
                ));
                let distance: i32 = base.wrapping_sub(candidate) as i32;
                let insert: i32 = base.wrapping_sub(next_emit) as i32;
                ip_index = ip_index.wrapping_add(matched);
                *num_commands += EmitInsertLen(insert as u32, commands);
                (*literals)[..(insert as usize)]
                    .clone_from_slice(&base_ip[next_emit..(next_emit + insert as usize)]);
                *num_literals += insert as usize;
                let new_literals = core::mem::take(literals);
                let _ = core::mem::replace(literals, &mut new_literals[(insert as usize)..]);
                if distance == last_distance {
                    (*commands)[0] = 64u32;
                    let remainder = core::mem::take(commands);
                    let _ = core::mem::replace(commands, &mut remainder[1..]);
                    *num_commands += 1;
                } else {
                    *num_commands += EmitDistance(distance as u32, commands);
                    last_distance = distance;
                }
                *num_commands += EmitCopyLenLastDistance(matched, commands);
                next_emit = ip_index;
                if ip_index >= ip_limit {
                    goto_emit_remainder = true;
                    {
                        break;
                    }
                }
                {
                    let mut input_bytes: u64;
                    let mut prev_hash: u32;
                    let cur_hash: u32;
                    if min_match == 4 {
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 3)..]);
                        cur_hash = HashBytesAtOffset(input_bytes, 3i32, shift, min_match);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(3) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(2) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(1) as i32;
                    } else {
                        assert!(ip_index >= 5);
                        // could this be off the end FIXME
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 5)..]);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(5) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(4) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 2i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(3) as i32;
                        assert!(ip_index >= 2);
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 2)..]);
                        cur_hash = HashBytesAtOffset(input_bytes, 2i32, shift, min_match);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(2) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(1) as i32;
                    }
                    candidate = table[(cur_hash as usize)] as usize;
                    table[(cur_hash as usize)] = ip_index as i32;
                }
            }
            while ip_index.wrapping_sub(candidate)
                <= (1usize << 18).wrapping_sub(16) as isize as usize
                && IsMatch(&base_ip[ip_index..], &base_ip[candidate..], min_match)
            {
                let base_index: usize = ip_index;
                let matched: usize = min_match.wrapping_add(FindMatchLengthWithLimit(
                    &base_ip[(candidate + min_match)..],
                    &base_ip[(ip_index + min_match)..],
                    ip_end.wrapping_sub(ip_index).wrapping_sub(min_match),
                ));
                ip_index = ip_index.wrapping_add(matched);
                last_distance = base_index.wrapping_sub(candidate) as i32;
                *num_commands += EmitCopyLen(matched, commands);
                *num_commands += EmitDistance(last_distance as u32, commands);
                next_emit = ip_index;
                if ip_index >= ip_limit {
                    goto_emit_remainder = true;
                    {
                        break;
                    }
                }
                {
                    assert!(ip_index >= 5);
                    let mut input_bytes: u64;

                    let cur_hash: u32;
                    let mut prev_hash: u32;
                    if min_match == 4 {
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 3)..]);
                        cur_hash = HashBytesAtOffset(input_bytes, 3i32, shift, min_match);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(3) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(2) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 2i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(1) as i32;
                    } else {
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 5)..]);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(5) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(4) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 2i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(3) as i32;
                        assert!(ip_index >= 2);
                        input_bytes = BROTLI_UNALIGNED_LOAD64(&base_ip[(ip_index - 2)..]);
                        cur_hash = HashBytesAtOffset(input_bytes, 2i32, shift, min_match);
                        prev_hash = HashBytesAtOffset(input_bytes, 0i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(2) as i32;
                        prev_hash = HashBytesAtOffset(input_bytes, 1i32, shift, min_match);
                        table[(prev_hash as usize)] = ip_index.wrapping_sub(1) as i32;
                    }
                    candidate = table[(cur_hash as usize)] as usize;
                    table[(cur_hash as usize)] = ip_index as i32;
                }
            }
            if !goto_emit_remainder {
                next_hash = Hash(
                    &base_ip[{
                        ip_index = ip_index.wrapping_add(1);
                        ip_index
                    }..],
                    shift,
                    min_match,
                );
            }
        }
    }
    if next_emit < ip_end {
        let insert: u32 = ip_end.wrapping_sub(next_emit) as u32;
        *num_commands += EmitInsertLen(insert, commands);
        literals[..insert as usize]
            .clone_from_slice(&base_ip[next_emit..(next_emit + insert as usize)]);
        let mut xliterals = core::mem::take(literals);
        *literals = &mut core::mem::take(&mut xliterals)[(insert as usize)..];
        *num_literals += insert as usize;
    }
}

fn ShouldCompress(input: &[u8], input_size: usize, num_literals: usize) -> bool {
    let corpus_size = input_size as floatX;
    if (num_literals as floatX) < 0.98 * corpus_size {
        true
    } else {
        let mut literal_histo: [u32; 256] = [0; 256];
        let max_total_bit_cost: floatX = corpus_size * 8.0 * 0.98 / 43.0;
        let mut i: usize;
        i = 0usize;
        while i < input_size {
            {
                let _rhs = 1;
                let _lhs = &mut literal_histo[input[i] as usize];
                *_lhs = (*_lhs).wrapping_add(_rhs as u32);
            }
            i = i.wrapping_add(43);
        }
        BitsEntropy(&mut literal_histo[..], 256) < max_total_bit_cost
    }
}

pub fn BrotliWriteBits(n_bits: usize, bits: u64, pos: &mut usize, array: &mut [u8]) {
    let p = &mut array[(*pos >> 3)..];
    let mut v: u64 = p[0] as (u64);
    v |= bits << (*pos & 7);
    BROTLI_UNALIGNED_STORE64(p, v);
    *pos = pos.wrapping_add(n_bits);
}

pub(crate) fn store_meta_block_header(
    len: usize,
    is_uncompressed: bool,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let mut nibbles: u64 = 6;
    BrotliWriteBits(1, 0, storage_ix, storage);
    if len <= (1u32 << 16) as usize {
        nibbles = 4;
    } else if len <= (1u32 << 20) as usize {
        nibbles = 5;
    }
    BrotliWriteBits(2, nibbles.wrapping_sub(4), storage_ix, storage);
    BrotliWriteBits(
        nibbles.wrapping_mul(4) as usize,
        len.wrapping_sub(1) as u64,
        storage_ix,
        storage,
    );
    BrotliWriteBits(1, u64::from(is_uncompressed), storage_ix, storage);
}

pub fn memcpy<T: Sized + Clone>(
    dst: &mut [T],
    dst_offset: usize,
    src: &[T],
    src_offset: usize,
    size_to_copy: usize,
) {
    dst[dst_offset..(dst_offset + size_to_copy)]
        .clone_from_slice(&src[src_offset..(src_offset + size_to_copy)]);
}
fn BuildAndStoreCommandPrefixCode(
    histogram: &[u32],
    depth: &mut [u8],
    bits: &mut [u16],
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let mut tree = [HuffmanTree::new(0, 0, 0); 129];
    let mut cmd_depth: [u8; 704] = [0; 704];
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
    memcpy(&mut cmd_depth[..], 0, depth, 24, 24);
    memcpy(&mut cmd_depth[..], 24, depth, 0, 8);
    memcpy(&mut cmd_depth[..], 32usize, depth, (48usize), 8usize);
    memcpy(&mut cmd_depth[..], 40usize, depth, (8usize), 8usize);
    memcpy(&mut cmd_depth[..], 48usize, depth, (56usize), 8usize);
    memcpy(&mut cmd_depth[..], 56usize, depth, (16usize), 8usize);
    BrotliConvertBitDepthsToSymbols(&mut cmd_depth[..], 64usize, &mut cmd_bits[..]);
    memcpy(bits, 0, &cmd_bits[..], 24usize, 16usize);
    memcpy(bits, (8usize), &cmd_bits[..], 40usize, 8usize);
    memcpy(bits, (16usize), &cmd_bits[..], 56usize, 8usize);
    memcpy(bits, (24usize), &cmd_bits[..], 0, 48usize);
    memcpy(bits, (48usize), &cmd_bits[..], 32usize, 8usize);
    memcpy(bits, (56usize), &cmd_bits[..], 48usize, 8usize);
    BrotliConvertBitDepthsToSymbols(&mut depth[64..], 64usize, &mut bits[64..]);
    {
        for item in cmd_depth[..64].iter_mut() {
            *item = 0;
        }
        //memset(&mut cmd_depth[..], 0i32, 64usize);
        memcpy(&mut cmd_depth[..], 0, depth, (24usize), 8usize);
        memcpy(&mut cmd_depth[..], 64usize, depth, (32usize), 8usize);
        memcpy(&mut cmd_depth[..], 128usize, depth, (40usize), 8usize);
        memcpy(&mut cmd_depth[..], 192usize, depth, (48usize), 8usize);
        memcpy(&mut cmd_depth[..], 384usize, depth, (56usize), 8usize);
        for i in 0usize..8usize {
            cmd_depth[(128usize).wrapping_add((8usize).wrapping_mul(i))] = depth[i];
            cmd_depth[(256usize).wrapping_add((8usize).wrapping_mul(i))] = depth[i.wrapping_add(8)];
            cmd_depth[(448usize).wrapping_add((8usize).wrapping_mul(i))] =
                depth[i.wrapping_add(16)];
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

fn StoreCommands<AllocHT: alloc::Allocator<HuffmanTree>>(
    mht: &mut AllocHT,
    mut literals: &[u8],
    num_literals: usize,
    commands: &[u32],
    num_commands: usize,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    static kNumExtraBits: [u32; 128] = [
        0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        7, 8, 9, 10, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
        5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
        18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
    ];
    static kInsertOffset: [u32; 24] = [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114,
        6210, 22594,
    ];
    let mut lit_depths: [u8; 256] = [0; 256];
    let mut lit_bits: [u16; 256] = [0; 256]; // maybe return this instead
    let mut lit_histo: [u32; 256] = [0; 256]; // maybe return this instead of init
    let mut cmd_depths: [u8; 128] = [0; 128];
    let mut cmd_bits: [u16; 128] = [0; 128];
    let mut cmd_histo: [u32; 128] = [0; 128];
    let mut i: usize;
    for i in 0usize..num_literals {
        let _rhs = 1;
        let _lhs = &mut lit_histo[literals[i] as usize];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
    BrotliBuildAndStoreHuffmanTreeFast(
        mht,
        &lit_histo[..],
        num_literals,
        8usize,
        &mut lit_depths[..],
        &mut lit_bits[..],
        storage_ix,
        storage,
    );
    i = 0usize;
    while i < num_commands {
        {
            let code: u32 = commands[i] & 0xffu32;
            {
                let _rhs = 1;
                let _lhs = &mut cmd_histo[code as usize];
                *_lhs = (*_lhs).wrapping_add(_rhs as u32);
            }
        }
        i = i.wrapping_add(1);
    }
    {
        let _rhs = 1i32;
        let _lhs = &mut cmd_histo[1];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
    {
        let _rhs = 1i32;
        let _lhs = &mut cmd_histo[2];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
    {
        let _rhs = 1i32;
        let _lhs = &mut cmd_histo[64];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
    {
        let _rhs = 1i32;
        let _lhs = &mut cmd_histo[84];
        *_lhs = (*_lhs).wrapping_add(_rhs as u32);
    }
    BuildAndStoreCommandPrefixCode(
        &mut cmd_histo[..],
        &mut cmd_depths[..],
        &mut cmd_bits[..],
        storage_ix,
        storage,
    );
    for i in 0usize..num_commands {
        let cmd: u32 = commands[i];
        let code: u32 = cmd & 0xffu32;
        let extra: u32 = cmd >> 8;
        BrotliWriteBits(
            cmd_depths[code as usize] as usize,
            cmd_bits[code as usize] as (u64),
            storage_ix,
            storage,
        );
        BrotliWriteBits(
            kNumExtraBits[code as usize] as usize,
            extra as (u64),
            storage_ix,
            storage,
        );
        if code < 24u32 {
            let insert: u32 = kInsertOffset[code as usize].wrapping_add(extra);
            for literal in literals[..(insert as usize)].iter() {
                let lit: u8 = *literal;
                BrotliWriteBits(
                    lit_depths[lit as usize] as usize,
                    lit_bits[lit as usize] as (u64),
                    storage_ix,
                    storage,
                );
            }
            literals = &literals[insert as usize..];
        }
    }
}
fn EmitUncompressedMetaBlock(
    input: &[u8],
    input_size: usize,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    store_meta_block_header(input_size, true, storage_ix, storage);
    *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
    memcpy(storage, (*storage_ix >> 3), input, 0, input_size);
    *storage_ix = storage_ix.wrapping_add(input_size << 3);
    storage[(*storage_ix >> 3)] = 0u8;
}

#[allow(unused_variables)]
#[inline(always)]
fn compress_fragment_two_pass_impl<AllocHT: alloc::Allocator<HuffmanTree>>(
    m: &mut AllocHT,
    base_ip: &[u8],
    mut input_size: usize,
    is_last: bool,
    command_buf: &mut [u32],
    literal_buf: &mut [u8],
    table: &mut [i32],
    table_bits: usize,
    min_match: usize,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let mut input_index: usize = 0usize;
    while input_size > 0usize {
        let block_size: usize = min(input_size, kCompressFragmentTwoPassBlockSize);
        let mut num_literals: usize = 0;
        let mut num_commands: usize = 0;
        {
            let mut literals = &mut literal_buf[..];
            let mut commands = &mut command_buf[..];
            CreateCommands(
                input_index,
                block_size,
                input_size,
                base_ip,
                table,
                table_bits,
                min_match,
                &mut literals,
                &mut num_literals,
                &mut commands,
                &mut num_commands,
            );
        }
        if ShouldCompress(&base_ip[input_index..], block_size, num_literals) {
            store_meta_block_header(block_size, false, storage_ix, storage);
            BrotliWriteBits(13usize, 0, storage_ix, storage);
            StoreCommands(
                m,
                literal_buf,
                num_literals,
                command_buf,
                num_commands,
                storage_ix,
                storage,
            );
        } else {
            EmitUncompressedMetaBlock(&base_ip[input_index..], block_size, storage_ix, storage);
        }
        input_index = input_index.wrapping_add(block_size);
        input_size = input_size.wrapping_sub(block_size);
    }
}
macro_rules! compress_specialization {
    ($table_bits : expr, $fname: ident) => {
        fn $fname<AllocHT: alloc::Allocator<HuffmanTree>>(
            mht: &mut AllocHT,
            input: &[u8],
            input_size: usize,
            is_last: bool,
            command_buf: &mut [u32],
            literal_buf: &mut [u8],
            table: &mut [i32],
            storage_ix: &mut usize,
            storage: &mut [u8],
        ) {
            let min_match = if $table_bits < 15 { 4 } else { 6 };
            compress_fragment_two_pass_impl(
                mht,
                input,
                input_size,
                is_last,
                command_buf,
                literal_buf,
                table,
                $table_bits,
                min_match,
                storage_ix,
                storage,
            );
        }
    };
}
compress_specialization!(8, BrotliCompressFragmentTwoPassImpl8);
compress_specialization!(9, BrotliCompressFragmentTwoPassImpl9);
compress_specialization!(10, BrotliCompressFragmentTwoPassImpl10);
compress_specialization!(11, BrotliCompressFragmentTwoPassImpl11);
compress_specialization!(12, BrotliCompressFragmentTwoPassImpl12);
compress_specialization!(13, BrotliCompressFragmentTwoPassImpl13);
compress_specialization!(14, BrotliCompressFragmentTwoPassImpl14);
compress_specialization!(15, BrotliCompressFragmentTwoPassImpl15);
compress_specialization!(16, BrotliCompressFragmentTwoPassImpl16);
compress_specialization!(17, BrotliCompressFragmentTwoPassImpl17);

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


pub(crate) fn compress_fragment_two_pass<AllocHT: alloc::Allocator<HuffmanTree>>(
    m: &mut AllocHT,
    input: &[u8],
    input_size: usize,
    is_last: bool,
    command_buf: &mut [u32],
    literal_buf: &mut [u8],
    table: &mut [i32],
    table_size: usize,
    storage_ix: &mut usize,
    storage: &mut [u8],
) {
    let initial_storage_ix: usize = *storage_ix;
    let table_bits: usize = Log2FloorNonZero(table_size as u64) as usize;
    if table_bits == 8usize {
        BrotliCompressFragmentTwoPassImpl8(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 9usize {
        BrotliCompressFragmentTwoPassImpl9(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 10usize {
        BrotliCompressFragmentTwoPassImpl10(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 11usize {
        BrotliCompressFragmentTwoPassImpl11(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 12usize {
        BrotliCompressFragmentTwoPassImpl12(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 13usize {
        BrotliCompressFragmentTwoPassImpl13(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 14usize {
        BrotliCompressFragmentTwoPassImpl14(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 15usize {
        BrotliCompressFragmentTwoPassImpl15(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 16usize {
        BrotliCompressFragmentTwoPassImpl16(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if table_bits == 17usize {
        BrotliCompressFragmentTwoPassImpl17(
            m,
            input,
            input_size,
            is_last,
            command_buf,
            literal_buf,
            table,
            storage_ix,
            storage,
        );
    }
    if storage_ix.wrapping_sub(initial_storage_ix) > (31usize).wrapping_add(input_size << 3) {
        RewindBitPosition(initial_storage_ix, storage_ix, storage);
        EmitUncompressedMetaBlock(input, input_size, storage_ix, storage);
    }
    if is_last {
        BrotliWriteBits(1, 1, storage_ix, storage);
        BrotliWriteBits(1, 1, storage_ix, storage);
        *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
    }
}
