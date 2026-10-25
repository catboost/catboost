#![allow(unsafe_op_in_unsafe_fn)]
use crate::{Rounds, Variant};
use core::marker::PhantomData;

#[cfg(feature = "cipher")]
use crate::{STATE_WORDS, chacha::Block};

#[cfg(feature = "cipher")]
use cipher::{
    BlockSizeUser, ParBlocks, ParBlocksSizeUser, StreamCipherBackend, StreamCipherClosure,
    consts::{U16, U64},
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Maximum number of blocks processed in parallel.
/// We also support 8 and 4 in gen_tail_blocks.
const MAX_PAR_BLOCKS: usize = 16;

/// Divisor to compute `N`, the number of __m512i needed
/// to represent a number of parallel blocks.
const BLOCKS_PER_VECTOR: usize = 4;

const MAX_N: usize = MAX_PAR_BLOCKS / BLOCKS_PER_VECTOR;

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg(feature = "cipher")]
pub(crate) unsafe fn inner<R, F, V>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Rounds,
    F: StreamCipherClosure<BlockSize = U64>,
    V: Variant,
{
    let simd_state = state.as_mut_ptr().cast::<i32>();

    let mut backend = Backend::<R, V> {
        state: [
            _mm_loadu_epi32(simd_state),
            _mm_loadu_epi32(simd_state.add(4)),
            _mm_loadu_epi32(simd_state.add(8)),
        ],
        ctr: _mm_loadu_epi32(simd_state.add(12)),
        _pd: PhantomData,
    };

    f.call(&mut backend);

    // Update counter in the persistent state
    state[12] = _mm_extract_epi32::<0>(backend.ctr) as u32;
    if size_of::<V::Counter>() == 8 {
        state[13] = _mm_extract_epi32::<1>(backend.ctr) as u32;
    }
}

struct Backend<R: Rounds, V: Variant> {
    state: [__m128i; 3],
    ctr: __m128i,
    _pd: PhantomData<(R, V)>,
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> Backend<R, V> {
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vl")]
    unsafe fn increment_ctr(&mut self, amount: usize) {
        match size_of::<V::Counter>() {
            4 => {
                self.ctr = _mm_add_epi32(self.ctr, _mm_set_epi32(0, 0, 0, amount as i32));
            }
            8 => {
                self.ctr = _mm_add_epi64(self.ctr, _mm_set_epi64x(0, amount as i64));
            }
            _ => unreachable!(),
        }
    }

    /// Generates blocks using the 512-bit-wide dispatch
    /// with up to `N` vectors processed in parallel, producing
    /// `N * BLOCKS_PER_VECTOR` blocks.
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vl")]
    unsafe fn gen_blocks_fullwidth<const N: usize>(&mut self, blocks: &mut [Block]) {
        let par_blocks = N * BLOCKS_PER_VECTOR;
        assert!(blocks.len() <= par_blocks);

        let mut ctrs = [_mm512_broadcast_i32x4(self.ctr); N];
        for i in 0..ctrs.len() {
            match size_of::<V::Counter>() {
                4 => {
                    ctrs[i] = _mm512_add_epi32(
                        ctrs[i],
                        _mm512_set_epi32(
                            0,
                            0,
                            0,
                            (i * BLOCKS_PER_VECTOR + 3) as i32,
                            0,
                            0,
                            0,
                            (i * BLOCKS_PER_VECTOR + 2) as i32,
                            0,
                            0,
                            0,
                            (i * BLOCKS_PER_VECTOR + 1) as i32,
                            0,
                            0,
                            0,
                            (i * BLOCKS_PER_VECTOR) as i32,
                        ),
                    );
                }
                8 => {
                    ctrs[i] = _mm512_add_epi64(
                        ctrs[i],
                        _mm512_set_epi64(
                            0,
                            (i * BLOCKS_PER_VECTOR + 3) as i64,
                            0,
                            (i * BLOCKS_PER_VECTOR + 2) as i64,
                            0,
                            (i * BLOCKS_PER_VECTOR + 1) as i64,
                            0,
                            (i * BLOCKS_PER_VECTOR) as i64,
                        ),
                    );
                }
                _ => unreachable!(),
            }
        }

        self.increment_ctr(blocks.len());

        let result = rounds::<N, R>(&self.state.map(|v| _mm512_broadcast_i32x4(v)), &ctrs);

        for i in 0..N {
            let result_vectors = result[i];

            // We have our data in SIMD vectors in the following layout
            // (using a, b, c, and d to indicate the resp. 4 rows of each block,
            // and Bn to denote the nth block):
            // result_vectors[0]:
            // B0a0 B0a1 B0a2 B0a3
            // B1a0 B1a1 B1a2 B1a3
            // ...
            // B3a0 B3a1 B3a2 B3a2
            //
            // result_vectors[1]:
            // B0b0 B0b1 B0b2 B0b3
            // B1b0 B1b1 B1b2 B1b3
            // ...
            // B3b0 B3b1 B3b2 B3b2
            //
            // and so on for result_vectors[2] (storing c values) and result_vectors[3] (storing d values).
            //
            // To store to memory, we need to transpose to the following format:
            // transposed[0]:
            // B0a0 B0a1 B0a2 B0a3
            // B0b0 B0b1 B0b2 B0b3
            // B0c0 B0c1 B0c2 B0c3
            // B0d0 B0d1 B0d2 B0d3
            //
            // and so on, such that each 512-bit SIMD vector
            // contains a single contiguous block.
            //
            // We achieve this transposition using the following
            // sequence of shuffles.

            let temp_abab_block01 = _mm512_permutex2var_epi64(
                result_vectors[0],
                _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11),
                result_vectors[1],
            );
            let temp_abab_block23 = _mm512_permutex2var_epi64(
                result_vectors[0],
                _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15),
                result_vectors[1],
            );

            let temp_cdcd_block01 = _mm512_permutex2var_epi64(
                result_vectors[2],
                _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11),
                result_vectors[3],
            );
            let temp_cdcd_block23 = _mm512_permutex2var_epi64(
                result_vectors[2],
                _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15),
                result_vectors[3],
            );

            let block0 =
                _mm512_shuffle_i32x4::<0b01_00_01_00>(temp_abab_block01, temp_cdcd_block01);
            let block1 =
                _mm512_shuffle_i32x4::<0b11_10_11_10>(temp_abab_block01, temp_cdcd_block01);
            let block2 =
                _mm512_shuffle_i32x4::<0b01_00_01_00>(temp_abab_block23, temp_cdcd_block23);
            let block3 =
                _mm512_shuffle_i32x4::<0b11_10_11_10>(temp_abab_block23, temp_cdcd_block23);

            for (j, src_block) in [block0, block1, block2, block3].into_iter().enumerate() {
                let dst_index = i * BLOCKS_PER_VECTOR + j;
                if dst_index < blocks.len() {
                    _mm512_storeu_si512((&raw mut blocks[dst_index]).cast(), src_block);
                }
            }
        }
    }

    /// Generates up to 2 blocks using 256-bit vectors.
    #[inline]
    #[target_feature(enable = "avx512f", enable = "avx512vl")]
    unsafe fn gen_blocks_halfwidth(&mut self, blocks: &mut [Block]) {
        assert!(blocks.len() <= 2);

        let mut ctr = _mm256_broadcast_i32x4(self.ctr);

        match size_of::<V::Counter>() {
            4 => {
                ctr = _mm256_add_epi32(ctr, _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 0));
            }
            8 => {
                ctr = _mm256_add_epi64(ctr, _mm256_set_epi64x(0, 1, 0, 0));
            }
            _ => unreachable!(),
        }

        self.increment_ctr(blocks.len());

        let block_vectors = rounds_halfwide::<R>([
            _mm256_broadcast_i32x4(self.state[0]),
            _mm256_broadcast_i32x4(self.state[1]),
            _mm256_broadcast_i32x4(self.state[2]),
            ctr,
        ]);

        // Similar transpose operation as
        // in gen_blocks_fullwidth.

        let block0_ab = _mm256_permutex2var_epi64(
            block_vectors[0],
            _mm256_setr_epi64x(0, 1, 4, 5),
            block_vectors[1],
        );
        let block0_cd = _mm256_permutex2var_epi64(
            block_vectors[2],
            _mm256_setr_epi64x(0, 1, 4, 5),
            block_vectors[3],
        );
        let block1_ab = _mm256_permutex2var_epi64(
            block_vectors[0],
            _mm256_setr_epi64x(2, 3, 6, 7),
            block_vectors[1],
        );
        let block1_cd = _mm256_permutex2var_epi64(
            block_vectors[2],
            _mm256_setr_epi64x(2, 3, 6, 7),
            block_vectors[3],
        );

        for (i, (block_part_ab, block_part_cd)) in [(block0_ab, block0_cd), (block1_ab, block1_cd)]
            .into_iter()
            .enumerate()
        {
            if i < blocks.len() {
                let dst = (&raw mut blocks[i]).cast::<i32>();
                _mm256_storeu_epi32(dst, block_part_ab);
                _mm256_storeu_epi32(
                    dst.add(size_of::<Block>() / 2 / size_of::<i32>()),
                    block_part_cd,
                );
            }
        }
    }
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> BlockSizeUser for Backend<R, V> {
    type BlockSize = U64;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> ParBlocksSizeUser for Backend<R, V> {
    type ParBlocksSize = U16;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> StreamCipherBackend for Backend<R, V> {
    #[inline]
    fn gen_par_ks_blocks(&mut self, blocks: &mut ParBlocks<Self>) {
        unsafe { self.gen_blocks_fullwidth::<MAX_N>(blocks) }
    }

    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        // Fallback for generating a single block using quarter-width vectors
        // (128).

        unsafe {
            let state = [self.state[0], self.state[1], self.state[2], self.ctr];

            self.increment_ctr(1);

            let result = rounds_quarterwide::<R>(state);

            for row in 0..4 {
                let dst = block.as_mut_ptr().cast::<i32>().add(row * 4);
                _mm_storeu_epi32(dst, result[row]);
            }
        }
    }

    #[inline]
    fn gen_tail_blocks(&mut self, blocks: &mut [cipher::Block<Self>]) {
        assert!(blocks.len() < MAX_PAR_BLOCKS);

        if blocks.is_empty() {
            return;
        }

        // Fallback for generating a number of blocks less than
        // MAX_PAR_BLOCKS.
        unsafe {
            if blocks.len() == 1 {
                self.gen_ks_block(&mut blocks[0]);
            } else if blocks.len() == 2 {
                self.gen_blocks_halfwidth(blocks);
            } else if blocks.len() <= 4 {
                self.gen_blocks_fullwidth::<1>(blocks);
            } else if blocks.len() <= 8 {
                self.gen_blocks_fullwidth::<2>(blocks);
            } else {
                self.gen_blocks_fullwidth::<MAX_N>(blocks);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn rounds<const N: usize, R: Rounds>(
    v: &[__m512i; 3],
    c: &[__m512i; N],
) -> [[__m512i; 4]; N] {
    let mut vs: [[__m512i; 4]; N] = [[_mm512_setzero_si512(); 4]; N];
    for i in 0..N {
        vs[i] = [v[0], v[1], v[2], c[i]];
    }
    for _ in 0..R::COUNT {
        double_quarter_round(&mut vs);
    }

    for i in 0..N {
        for j in 0..3 {
            vs[i][j] = _mm512_add_epi32(vs[i][j], v[j]);
        }
        vs[i][3] = _mm512_add_epi32(vs[i][3], c[i]);
    }

    vs
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn double_quarter_round<const N: usize>(v: &mut [[__m512i; 4]; N]) {
    add_xor_rot(v);
    rows_to_cols(v);
    add_xor_rot(v);
    cols_to_rows(v);
}

/// The goal of this function is to transform the state words from:
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c0, c1, c2, c3]    [ 8,  9, 10, 11]
/// [d0, d1, d2, d3]    [12, 13, 14, 15]
/// ```
///
/// to:
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b1, b2, b3, b0] == [ 5,  6,  7,  4]
/// [c2, c3, c0, c1]    [10, 11,  8,  9]
/// [d3, d0, d1, d2]    [15, 12, 13, 14]
/// ```
///
/// so that we can apply [`add_xor_rot`] to the resulting columns, and have it compute the
/// "diagonal rounds" (as defined in RFC 7539) in parallel. In practice, this shuffle is
/// non-optimal: the last state word to be altered in `add_xor_rot` is `b`, so the shuffle
/// blocks on the result of `b` being calculated.
///
/// We can optimize this by observing that the four quarter rounds in `add_xor_rot` are
/// data-independent: they only access a single column of the state, and thus the order of
/// the columns does not matter. We therefore instead shuffle the other three state words,
/// to obtain the following equivalent layout:
/// ```text
/// [a3, a0, a1, a2]    [ 3,  0,  1,  2]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c1, c2, c3, c0]    [ 9, 10, 11,  8]
/// [d2, d3, d0, d1]    [14, 15, 12, 13]
/// ```
///
/// See https://github.com/sneves/blake2-avx2/pull/4 for additional details. The earliest
/// known occurrence of this optimization is in floodyberry's SSE4 ChaCha code from 2014:
/// - https://github.com/floodyberry/chacha-opt/blob/0ab65cb99f5016633b652edebaf3691ceb4ff753/chacha_blocks_ssse3-64.S#L639-L643
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn rows_to_cols<const N: usize>(vs: &mut [[__m512i; 4]; N]) {
    // c >>>= 32; d >>>= 64; a >>>= 96;
    for [a, _, c, d] in vs {
        *c = _mm512_shuffle_epi32::<0b_00_11_10_01>(*c); // _MM_SHUFFLE(0, 3, 2, 1)
        *d = _mm512_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
        *a = _mm512_shuffle_epi32::<0b_10_01_00_11>(*a); // _MM_SHUFFLE(2, 1, 0, 3)
    }
}

/// The goal of this function is to transform the state words from:
/// ```text
/// [a3, a0, a1, a2]    [ 3,  0,  1,  2]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c1, c2, c3, c0]    [ 9, 10, 11,  8]
/// [d2, d3, d0, d1]    [14, 15, 12, 13]
/// ```
///
/// to:
/// ```text
/// [a0, a1, a2, a3]    [ 0,  1,  2,  3]
/// [b0, b1, b2, b3] == [ 4,  5,  6,  7]
/// [c0, c1, c2, c3]    [ 8,  9, 10, 11]
/// [d0, d1, d2, d3]    [12, 13, 14, 15]
/// ```
///
/// reversing the transformation of [`rows_to_cols`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cols_to_rows<const N: usize>(vs: &mut [[__m512i; 4]; N]) {
    // c <<<= 32; d <<<= 64; a <<<= 96;
    for [a, _, c, d] in vs {
        *c = _mm512_shuffle_epi32::<0b_10_01_00_11>(*c); // _MM_SHUFFLE(2, 1, 0, 3)
        *d = _mm512_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
        *a = _mm512_shuffle_epi32::<0b_00_11_10_01>(*a); // _MM_SHUFFLE(0, 3, 2, 1)
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_xor_rot<const N: usize>(vs: &mut [[__m512i; 4]; N]) {
    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    for [a, b, _, d] in vs.iter_mut() {
        *a = _mm512_add_epi32(*a, *b);
        *d = _mm512_xor_si512(*d, *a);
        *d = _mm512_rol_epi32::<16>(*d);
    }

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    for [_, b, c, d] in vs.iter_mut() {
        *c = _mm512_add_epi32(*c, *d);
        *b = _mm512_xor_si512(*b, *c);
        *b = _mm512_rol_epi32::<12>(*b);
    }

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    for [a, b, _, d] in vs.iter_mut() {
        *a = _mm512_add_epi32(*a, *b);
        *d = _mm512_xor_si512(*d, *a);
        *d = _mm512_rol_epi32::<8>(*d);
    }

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    for [_, b, c, d] in vs.iter_mut() {
        *c = _mm512_add_epi32(*c, *d);
        *b = _mm512_xor_si512(*b, *c);
        *b = _mm512_rol_epi32::<7>(*b);
    }
}

// Below is another implementation of the round application
// that uses 256-bit vectors instead of 512-bit (but, unlike
// the avx2 module, can use new AVX-512 instructions like rotates).
// It is used for tail processing of shorter outputs,
// since 256-bit instructions can be faster and lower latency
// than 512-bit instructions on certain microarchitectures (e.g. Zen 4).

#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn rounds_halfwide<R: Rounds>(v_in: [__m256i; 4]) -> [__m256i; 4] {
    let mut v = v_in;

    for _ in 0..R::COUNT {
        double_quarter_round_halfwide(&mut v);
    }

    for (a, b) in v.iter_mut().zip(v_in) {
        *a = _mm256_add_epi32(*a, b);
    }

    v
}

#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn double_quarter_round_halfwide(v: &mut [__m256i; 4]) {
    add_xor_rot_halfwide(v);
    rows_to_cols_halfwide(v);
    add_xor_rot_halfwide(v);
    cols_to_rows_halfwide(v);
}

#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn rows_to_cols_halfwide(v: &mut [__m256i; 4]) {
    // c >>>= 32; d >>>= 64; a >>>= 96;
    let [a, _, c, d] = v;
    *c = _mm256_shuffle_epi32::<0b_00_11_10_01>(*c); // _MM_SHUFFLE(0, 3, 2, 1)
    *d = _mm256_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
    *a = _mm256_shuffle_epi32::<0b_10_01_00_11>(*a); // _MM_SHUFFLE(2, 1, 0, 3)
}

#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn cols_to_rows_halfwide(v: &mut [__m256i; 4]) {
    // c <<<= 32; d <<<= 64; a <<<= 96;
    let [a, _, c, d] = v;
    *c = _mm256_shuffle_epi32::<0b_10_01_00_11>(*c); // _MM_SHUFFLE(2, 1, 0, 3)
    *d = _mm256_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
    *a = _mm256_shuffle_epi32::<0b_00_11_10_01>(*a); // _MM_SHUFFLE(0, 3, 2, 1)
}

#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn add_xor_rot_halfwide(v: &mut [__m256i; 4]) {
    let [a, b, c, d] = v;

    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    *a = _mm256_add_epi32(*a, *b);
    *d = _mm256_xor_si256(*d, *a);
    *d = _mm256_rol_epi32::<16>(*d);

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    *c = _mm256_add_epi32(*c, *d);
    *b = _mm256_xor_si256(*b, *c);
    *b = _mm256_rol_epi32::<12>(*b);

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    *a = _mm256_add_epi32(*a, *b);
    *d = _mm256_xor_si256(*d, *a);
    *d = _mm256_rol_epi32::<8>(*d);

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    *c = _mm256_add_epi32(*c, *d);
    *b = _mm256_xor_si256(*b, *c);
    *b = _mm256_rol_epi32::<7>(*b);
}

// Finally, below is an implementation using 128-bit vectors
// for the case of generating a single block.

#[inline(always)]
unsafe fn rounds_quarterwide<R: Rounds>(v_in: [__m128i; 4]) -> [__m128i; 4] {
    let mut v = v_in;

    for _ in 0..R::COUNT {
        double_quarter_round_quarterwide(&mut v);
    }

    for (a, b) in v.iter_mut().zip(v_in) {
        *a = _mm_add_epi32(*a, b);
    }

    v
}

#[inline(always)]
unsafe fn double_quarter_round_quarterwide(v: &mut [__m128i; 4]) {
    add_xor_rot_quarterwide(v);
    rows_to_cols_quarterwide(v);
    add_xor_rot_quarterwide(v);
    cols_to_rows_quarterwide(v);
}

#[inline(always)]
unsafe fn rows_to_cols_quarterwide(v: &mut [__m128i; 4]) {
    // c >>>= 32; d >>>= 64; a >>>= 96;
    let [a, _, c, d] = v;
    *c = _mm_shuffle_epi32::<0b_00_11_10_01>(*c); // _MM_SHUFFLE(0, 3, 2, 1)
    *d = _mm_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
    *a = _mm_shuffle_epi32::<0b_10_01_00_11>(*a); // _MM_SHUFFLE(2, 1, 0, 3)
}

#[inline(always)]
unsafe fn cols_to_rows_quarterwide(v: &mut [__m128i; 4]) {
    // c <<<= 32; d <<<= 64; a <<<= 96;
    let [a, _, c, d] = v;
    *c = _mm_shuffle_epi32::<0b_10_01_00_11>(*c); // _MM_SHUFFLE(2, 1, 0, 3)
    *d = _mm_shuffle_epi32::<0b_01_00_11_10>(*d); // _MM_SHUFFLE(1, 0, 3, 2)
    *a = _mm_shuffle_epi32::<0b_00_11_10_01>(*a); // _MM_SHUFFLE(0, 3, 2, 1)
}

#[inline(always)]
unsafe fn add_xor_rot_quarterwide(v: &mut [__m128i; 4]) {
    let [a, b, c, d] = v;

    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_xor_si128(*d, *a);
    *d = _mm_rol_epi32::<16>(*d);

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_xor_si128(*b, *c);
    *b = _mm_rol_epi32::<12>(*b);

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_xor_si128(*d, *a);
    *d = _mm_rol_epi32::<8>(*d);

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_xor_si128(*b, *c);
    *b = _mm_rol_epi32::<7>(*b);
}
