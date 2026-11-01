use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr;

use crate::bytes;
use crate::error::{Error, Result};
use crate::{MAX_BLOCK_SIZE, MAX_INPUT_SIZE};

/// The total number of slots we permit for our hash table of 4 byte repeat
/// sequences.
const MAX_TABLE_SIZE: usize = 1 << 14;

/// The size of a small hash table. This is useful for reducing overhead when
/// compressing very small blocks of bytes.
const SMALL_TABLE_SIZE: usize = 1 << 10;

/// The total number of bytes that we always leave uncompressed at the end
/// of the buffer. This in particular affords us some wiggle room during
/// compression such that faster copy operations can be used.
const INPUT_MARGIN: usize = 16 - 1;

/// The minimum block size that we're willing to consider for compression.
/// Anything smaller than this gets emitted as a literal.
const MIN_NON_LITERAL_BLOCK_SIZE: usize = 1 + 1 + INPUT_MARGIN;

/// Nice names for the various Snappy tags.
enum Tag {
    Literal = 0b00,
    Copy1 = 0b01,
    Copy2 = 0b10,
    // Compression never actually emits a Copy4 operation and decompression
    // uses tricks so that we never explicitly do case analysis on the copy
    // operation type, therefore leading to the fact that we never use Copy4.
    #[allow(dead_code)]
    Copy4 = 0b11,
}

/// Returns the maximum compressed size given the uncompressed size.
///
/// If the uncompressed size exceeds the maximum allowable size then this
/// returns 0.
pub fn max_compress_len(input_len: usize) -> usize {
    let input_len = input_len as u64;
    if input_len > MAX_INPUT_SIZE {
        return 0;
    }
    let max = 32 + input_len + (input_len / 6);
    if max > MAX_INPUT_SIZE {
        0
    } else {
        max as usize
    }
}

/// Encoder is a raw encoder for compressing bytes in the Snappy format.
///
/// Thie encoder does not use the Snappy frame format and simply compresses the
/// given bytes in one big Snappy block (that is, it has a single header).
///
/// Unless you explicitly need the low-level control, you should use
/// [`read::FrameEncoder`](../read/struct.FrameEncoder.html)
/// or
/// [`write::FrameEncoder`](../write/struct.FrameEncoder.html)
/// instead, which compresses to the Snappy frame format.
///
/// It is beneficial to reuse an Encoder when possible.
pub struct Encoder {
    small: [u16; SMALL_TABLE_SIZE],
    big: Vec<u16>,
}

impl fmt::Debug for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Encoder(...)")
    }
}

impl Encoder {
    /// Return a new encoder that can be used for compressing bytes.
    pub fn new() -> Encoder {
        Encoder { small: [0; SMALL_TABLE_SIZE], big: vec![] }
    }

    /// Compresses all bytes in `input` into `output`.
    ///
    /// `input` can be any arbitrary sequence of bytes.
    ///
    /// `output` must be large enough to hold the maximum possible compressed
    /// size of `input`, which can be computed using `max_compress_len`.
    ///
    /// On success, this returns the number of bytes written to `output`.
    ///
    /// # Errors
    ///
    /// This method returns an error in the following circumstances:
    ///
    /// * The total number of bytes to compress exceeds `2^32 - 1`.
    /// * `output` has length less than `max_compress_len(input.len())`.
    pub fn compress(
        &mut self,
        mut input: &[u8],
        output: &mut [u8],
    ) -> Result<usize> {
        match max_compress_len(input.len()) {
            0 => {
                return Err(Error::TooBig {
                    given: input.len() as u64,
                    max: MAX_INPUT_SIZE,
                });
            }
            min if output.len() < min => {
                return Err(Error::BufferTooSmall {
                    given: output.len() as u64,
                    min: min as u64,
                });
            }
            _ => {}
        }
        // Handle an edge case specially.
        if input.is_empty() {
            // Encodes a varint of 0, denoting the total size of uncompressed
            // bytes.
            output[0] = 0;
            return Ok(1);
        }
        // Write the Snappy header, which is just the total number of
        // uncompressed bytes.
        let mut d = bytes::write_varu64(output, input.len() as u64);
        while !input.is_empty() {
            // Find the next block.
            let mut src = input;
            if src.len() > MAX_BLOCK_SIZE {
                src = &src[..MAX_BLOCK_SIZE as usize];
            }
            input = &input[src.len()..];

            // If the block is smallish, then don't waste time on it and just
            // emit a literal.
            let mut block = Block::new(src, output, d);
            if block.src.len() < MIN_NON_LITERAL_BLOCK_SIZE {
                let lit_end = block.src.len();
                unsafe {
                    // SAFETY: next_emit is zero (in bounds) and the end is
                    // the length of the block (in bounds).
                    block.emit_literal(lit_end);
                }
            } else {
                let table = self.block_table(block.src.len());
                block.compress(table);
            }
            d = block.d;
        }
        Ok(d)
    }

    /// Compresses all bytes in `input` into a freshly allocated `Vec`.
    ///
    /// This is just like the `compress` method, except it allocates a `Vec`
    /// with the right size for you. (This is intended to be a convenience
    /// method.)
    ///
    /// This method returns an error under the same circumstances that
    /// `compress` does.
    pub fn compress_vec(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let mut buf = vec![0; max_compress_len(input.len())];
        let n = self.compress(input, &mut buf)?;
        buf.truncate(n);
        Ok(buf)
    }
}

struct Block<'s, 'd> {
    src: &'s [u8],
    s: usize,
    s_limit: usize,
    dst: &'d mut [u8],
    d: usize,
    next_emit: usize,
}

impl<'s, 'd> Block<'s, 'd> {
    #[inline(always)]
    fn new(src: &'s [u8], dst: &'d mut [u8], d: usize) -> Block<'s, 'd> {
        Block {
            src: src,
            s: 0,
            s_limit: src.len(),
            dst: dst,
            d: d,
            next_emit: 0,
        }
    }

    #[inline(always)]
    fn compress(&mut self, mut table: BlockTable<'_>) {
        debug_assert!(!table.is_empty());
        debug_assert!(self.src.len() >= MIN_NON_LITERAL_BLOCK_SIZE);

        self.s += 1;
        self.s_limit -= INPUT_MARGIN;
        let mut next_hash =
            table.hash(bytes::read_u32_le(&self.src[self.s..]));
        loop {
            let mut skip = 32;
            let mut candidate;
            let mut s_next = self.s;
            loop {
                self.s = s_next;
                let bytes_between_hash_lookups = skip >> 5;
                s_next = self.s + bytes_between_hash_lookups;
                skip += bytes_between_hash_lookups;
                if s_next > self.s_limit {
                    return self.done();
                }
                unsafe {
                    // SAFETY: next_hash is always computed by table.hash
                    // which is guaranteed to be in bounds.
                    candidate = *table.get_unchecked(next_hash) as usize;
                    *table.get_unchecked_mut(next_hash) = self.s as u16;

                    let srcp = self.src.as_ptr();
                    // SAFETY: s_next is guaranteed to be less than s_limit by
                    // the conditional above, which implies s_next is in
                    // bounds.
                    let x = bytes::loadu_u32_le(srcp.add(s_next));
                    next_hash = table.hash(x);
                    // SAFETY: self.s is always less than s_next, so it is also
                    // in bounds by the argument above.
                    //
                    // candidate is extracted from table, which is only ever
                    // set to valid positions in the block and is therefore
                    // also in bounds.
                    //
                    // We only need to compare y/z for equality, so we don't
                    // need to both with endianness. cur corresponds to the
                    // bytes at the current position and cand corresponds to
                    // a potential match. If they're equal, we declare victory
                    // and move below to try and extend the match.
                    let cur = bytes::loadu_u32_ne(srcp.add(self.s));
                    let cand = bytes::loadu_u32_ne(srcp.add(candidate));
                    if cur == cand {
                        break;
                    }
                }
            }
            // While the above found a candidate for compression, before we
            // emit a copy operation for it, we need to make sure that we emit
            // any bytes between the last copy operation and this one as a
            // literal.
            let lit_end = self.s;
            unsafe {
                // SAFETY: next_emit is set to a previous value of self.s,
                // which is guaranteed to be less than s_limit (in bounds).
                // lit_end is set to the current value of self.s, also
                // guaranteed to be less than s_limit (in bounds).
                self.emit_literal(lit_end);
            }
            loop {
                // Look for more matching bytes starting at the position of
                // the candidate and the current src position. We increment
                // self.s and candidate by 4 since we already know the first 4
                // bytes match.
                let base = self.s;
                self.s += 4;
                unsafe {
                    // SAFETY: candidate is always set to a value from our
                    // hash table, which only contains positions in self.src
                    // that have been seen for this block that occurred before
                    // self.s.
                    self.extend_match(candidate + 4);
                }
                let (offset, len) = (base - candidate, self.s - base);
                self.emit_copy(offset, len);
                self.next_emit = self.s;
                if self.s >= self.s_limit {
                    return self.done();
                }
                // Update the hash table with the byte sequences
                // self.src[self.s - 1..self.s + 3] and
                // self.src[self.s..self.s + 4]. Instead of reading 4 bytes
                // twice, we read 8 bytes once.
                //
                // If we happen to get a hit on self.src[self.s..self.s + 4],
                // then continue this loop and extend the match.
                unsafe {
                    let srcp = self.src.as_ptr();
                    // SAFETY: self.s can never exceed s_limit given by the
                    // conditional above and self.s is guaranteed to be
                    // non-zero and is therefore in bounds.
                    let x = bytes::loadu_u64_le(srcp.add(self.s - 1));
                    // The lower 4 bytes of x correspond to
                    // self.src[self.s - 1..self.s + 3].
                    let prev_hash = table.hash(x as u32);
                    // SAFETY: Hash values are guaranteed to be in bounds.
                    *table.get_unchecked_mut(prev_hash) = (self.s - 1) as u16;
                    // The lower 4 bytes of x>>8 correspond to
                    // self.src[self.s..self.s + 4].
                    let cur_hash = table.hash((x >> 8) as u32);
                    // SAFETY: Hash values are guaranteed to be in bounds.
                    candidate = *table.get_unchecked(cur_hash) as usize;
                    *table.get_unchecked_mut(cur_hash) = self.s as u16;

                    // SAFETY: candidate is set from table, which always
                    // contains valid positions in the current block.
                    let y = bytes::loadu_u32_le(srcp.add(candidate));
                    if (x >> 8) as u32 != y {
                        // If we didn't get a hit, update the next hash
                        // and move on. Our initial 8 byte read continues to
                        // pay off.
                        next_hash = table.hash((x >> 16) as u32);
                        self.s += 1;
                        break;
                    }
                }
            }
        }
    }

    /// Emits one or more copy operations with the given offset and length.
    /// offset must be in the range [1, 65535] and len must be in the range
    /// [4, 65535].
    #[inline(always)]
    fn emit_copy(&mut self, offset: usize, mut len: usize) {
        debug_assert!(1 <= offset && offset <= 65535);
        // Copy operations only allow lengths up to 64, but we'll allow bigger
        // lengths and emit as many operations as we need.
        //
        // N.B. Since our block size is 64KB, we never actually emit a copy 4
        // operation.
        debug_assert!(4 <= len && len <= 65535);

        // Emit copy 2 operations until we don't have to.
        // We check on 68 here and emit a shorter copy than 64 below because
        // it is cheaper to, e.g., encode a length 67 copy as a length 60
        // copy 2 followed by a length 7 copy 1 than to encode it as a length
        // 64 copy 2 followed by a length 3 copy 2. They key here is that a
        // copy 1 operation requires at least length 4 which forces a length 3
        // copy to use a copy 2 operation.
        while len >= 68 {
            self.emit_copy2(offset, 64);
            len -= 64;
        }
        if len > 64 {
            self.emit_copy2(offset, 60);
            len -= 60;
        }
        // If we can squeeze the last copy into a copy 1 operation, do it.
        if len <= 11 && offset <= 2047 {
            self.dst[self.d] = (((offset >> 8) as u8) << 5)
                | (((len - 4) as u8) << 2)
                | (Tag::Copy1 as u8);
            self.dst[self.d + 1] = offset as u8;
            self.d += 2;
        } else {
            self.emit_copy2(offset, len);
        }
    }

    /// Emits a "copy 2" operation with the given offset and length. The
    /// offset and length must be valid for a copy 2 operation. i.e., offset
    /// must be in the range [1, 65535] and len must be in the range [1, 64].
    #[inline(always)]
    fn emit_copy2(&mut self, offset: usize, len: usize) {
        debug_assert!(1 <= offset && offset <= 65535);
        debug_assert!(1 <= len && len <= 64);
        self.dst[self.d] = (((len - 1) as u8) << 2) | (Tag::Copy2 as u8);
        bytes::write_u16_le(offset as u16, &mut self.dst[self.d + 1..]);
        self.d += 3;
    }

    /// Attempts to extend a match from the current position in self.src with
    /// the candidate position given.
    ///
    /// This method uses unaligned loads and elides bounds checks, so the
    /// caller must guarantee that cand points to a valid location in self.src
    /// and is less than the current position in src.
    #[inline(always)]
    unsafe fn extend_match(&mut self, mut cand: usize) {
        debug_assert!(cand < self.s);
        while self.s + 8 <= self.src.len() {
            let srcp = self.src.as_ptr();
            // SAFETY: The loop invariant guarantees that there is at least
            // 8 bytes to read at self.src + self.s. Since cand must be
            // guaranteed by the caller to be valid and less than self.s, it
            // also has enough room to read 8 bytes.
            //
            // TODO(ag): Despite my best efforts, I couldn't get this to
            // autovectorize with 128-bit loads. The logic after the loads
            // appears to be a little too clever...
            let x = bytes::loadu_u64_ne(srcp.add(self.s));
            let y = bytes::loadu_u64_ne(srcp.add(cand));
            if x == y {
                // If all 8 bytes are equal, move on...
                self.s += 8;
                cand += 8;
            } else {
                // Otherwise, find the last byte that was equal. We can do
                // this efficiently by interpreted x/y as little endian
                // numbers, which lets us use the number of trailing zeroes
                // as a proxy for the number of equivalent bits (after an XOR).
                let z = x.to_le() ^ y.to_le();
                self.s += z.trailing_zeros() as usize / 8;
                return;
            }
        }
        // When we have fewer than 8 bytes left in the block, fall back to the
        // slow loop.
        while self.s < self.src.len() && self.src[self.s] == self.src[cand] {
            self.s += 1;
            cand += 1;
        }
    }

    /// Executes any cleanup when the current block has finished compressing.
    /// In particular, it emits any leftover bytes as a literal.
    #[inline(always)]
    fn done(&mut self) {
        if self.next_emit < self.src.len() {
            let lit_end = self.src.len();
            unsafe {
                // SAFETY: Both next_emit and lit_end are trivially in bounds
                // given the conditional and definition above.
                self.emit_literal(lit_end);
            }
        }
    }

    /// Emits a literal from self.src[self.next_emit..lit_end].
    ///
    /// This uses unaligned loads and elides bounds checks, so the caller must
    /// guarantee that self.src[self.next_emit..lit_end] is valid.
    #[inline(always)]
    unsafe fn emit_literal(&mut self, lit_end: usize) {
        let lit_start = self.next_emit;
        let len = lit_end - lit_start;
        let n = len.checked_sub(1).unwrap();
        if n <= 59 {
            self.dst[self.d] = ((n as u8) << 2) | (Tag::Literal as u8);
            self.d += 1;
            if len <= 16 && lit_start + 16 <= self.src.len() {
                // SAFETY: lit_start is equivalent to self.next_emit, which is
                // only set to self.s immediately following a copy emit. The
                // conditional above also ensures that there is at least 16
                // bytes of room in both src and dst.
                //
                // dst is big enough because the buffer is guaranteed to
                // be big enough to hold biggest possible compressed size plus
                // an extra 32 bytes, which exceeds the 16 byte copy here.
                let srcp = self.src.as_ptr().add(lit_start);
                let dstp = self.dst.as_mut_ptr().add(self.d);
                ptr::copy_nonoverlapping(srcp, dstp, 16);
                self.d += len;
                return;
            }
        } else if n < 256 {
            self.dst[self.d] = (60 << 2) | (Tag::Literal as u8);
            self.dst[self.d + 1] = n as u8;
            self.d += 2;
        } else {
            self.dst[self.d] = (61 << 2) | (Tag::Literal as u8);
            bytes::write_u16_le(n as u16, &mut self.dst[self.d + 1..]);
            self.d += 3;
        }
        // SAFETY: lit_start is equivalent to self.next_emit, which is only set
        // to self.s immediately following a copy, which implies that it always
        // points to valid bytes in self.src.
        //
        // We can't guarantee that there are at least len bytes though, which
        // must be guaranteed by the caller and is why this method is unsafe.
        let srcp = self.src.as_ptr().add(lit_start);
        let dstp = self.dst.as_mut_ptr().add(self.d);
        ptr::copy_nonoverlapping(srcp, dstp, len);
        self.d += len;
    }
}

/// `BlockTable` is a map from 4 byte sequences to positions of their most
/// recent occurrence in a block. In particular, this table lets us quickly
/// find candidates for compression.
///
/// We expose the `hash` method so that callers can be fastidious about the
/// number of times a hash is computed.
struct BlockTable<'a> {
    table: &'a mut [u16],
    /// The number of bits required to shift the hash such that the result
    /// is less than table.len().
    shift: u32,
}

impl Encoder {
    fn block_table(&mut self, block_size: usize) -> BlockTable<'_> {
        let mut shift: u32 = 32 - 8;
        let mut table_size = 256;
        while table_size < MAX_TABLE_SIZE && table_size < block_size {
            shift -= 1;
            table_size *= 2;
        }
        // If our block size is small, then use a small stack allocated table
        // instead of putting a bigger one on the heap. This particular
        // optimization is important if the caller is using Snappy to compress
        // many small blocks. (The memset savings alone is considerable.)
        let table: &mut [u16] = if table_size <= SMALL_TABLE_SIZE {
            &mut self.small[0..table_size]
        } else {
            if self.big.is_empty() {
                // Interestingly, using `self.big.resize` here led to some
                // very weird code getting generated that led to a large
                // slow down. Forcing the issue with a new vec seems to
                // fix it. ---AG
                self.big = vec![0; MAX_TABLE_SIZE];
            }
            &mut self.big[0..table_size]
        };
        for x in &mut *table {
            *x = 0;
        }
        BlockTable { table: table, shift: shift }
    }
}

impl<'a> BlockTable<'a> {
    #[inline(always)]
    fn hash(&self, x: u32) -> usize {
        (x.wrapping_mul(0x1E35A7BD) >> self.shift) as usize
    }
}

impl<'a> Deref for BlockTable<'a> {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        self.table
    }
}

impl<'a> DerefMut for BlockTable<'a> {
    fn deref_mut(&mut self) -> &mut [u16] {
        self.table
    }
}
