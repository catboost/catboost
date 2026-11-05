//! AVX2 backend.

#![allow(unsafe_op_in_unsafe_fn, reason = "needs triage")]
#![allow(clippy::cast_possible_truncation, reason = "needs triage")]
#![allow(clippy::cast_possible_wrap, reason = "needs triage")]
#![allow(clippy::cast_sign_loss, reason = "needs triage")]
#![allow(clippy::undocumented_unsafe_blocks, reason = "TODO")]

use crate::{Rounds, Variant};
use core::marker::PhantomData;

#[cfg(feature = "rng")]
use crate::ChaChaCore;

#[cfg(feature = "cipher")]
use crate::{STATE_WORDS, chacha::Block};

#[cfg(feature = "cipher")]
use cipher::{
    BlockSizeUser, ParBlocks, ParBlocksSizeUser, StreamCipherBackend, StreamCipherClosure,
    consts::{U4, U64},
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Number of blocks processed in parallel.
const PAR_BLOCKS: usize = 4;
/// Number of `__m256i` to store parallel blocks.
const N: usize = PAR_BLOCKS / 2;

#[inline]
#[target_feature(enable = "avx2")]
#[cfg(feature = "cipher")]
#[cfg_attr(chacha20_backend = "avx512", expect(unused))]
pub(crate) unsafe fn inner<R, F, V>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Rounds,
    F: StreamCipherClosure<BlockSize = U64>,
    V: Variant,
{
    let state_ptr = state.as_ptr().cast::<__m128i>();
    let v = [
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(0))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(1))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(2))),
    ];
    let mut c = _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(3)));
    c = match size_of::<V::Counter>() {
        4 => _mm256_add_epi32(c, _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 0)),
        8 => _mm256_add_epi64(c, _mm256_set_epi64x(0, 1, 0, 0)),
        _ => unreachable!(),
    };
    let mut ctr = [c; N];
    for i in 0..N {
        ctr[i] = c;
        c = match size_of::<V::Counter>() {
            4 => _mm256_add_epi32(c, _mm256_set_epi32(0, 0, 0, 2, 0, 0, 0, 2)),
            8 => _mm256_add_epi64(c, _mm256_set_epi64x(0, 2, 0, 2)),
            _ => unreachable!(),
        };
    }
    let mut backend = Backend::<R, V> {
        v,
        ctr,
        _pd: PhantomData,
    };

    f.call(&mut backend);

    state[12] = _mm256_extract_epi32(backend.ctr[0], 0) as u32;
    match size_of::<V::Counter>() {
        4 => {}
        8 => state[13] = _mm256_extract_epi32(backend.ctr[0], 1) as u32,
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[cfg(feature = "rng")]
pub(crate) unsafe fn rng_inner<R, V>(core: &mut ChaChaCore<R, V>, buffer: &mut [u32; 64])
where
    R: Rounds,
    V: Variant,
{
    let state_ptr = core.state.as_ptr().cast::<__m128i>();
    let v = [
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(0))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(1))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(2))),
    ];
    let mut c = _mm256_broadcastsi128_si256(_mm_loadu_si128(state_ptr.add(3)));
    c = _mm256_add_epi64(c, _mm256_set_epi64x(0, 1, 0, 0));
    let mut ctr = [c; N];
    for i in 0..N {
        ctr[i] = c;
        c = _mm256_add_epi64(c, _mm256_set_epi64x(0, 2, 0, 2));
    }
    let mut backend = Backend::<R, V> {
        v,
        ctr,
        _pd: PhantomData,
    };

    backend.rng_gen_par_ks_blocks(buffer);

    core.state[12] = _mm256_extract_epi32(backend.ctr[0], 0) as u32;
    core.state[13] = _mm256_extract_epi32(backend.ctr[0], 1) as u32;
}

struct Backend<R: Rounds, V: Variant> {
    v: [__m256i; 3],
    ctr: [__m256i; N],
    _pd: PhantomData<(R, V)>,
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> BlockSizeUser for Backend<R, V> {
    type BlockSize = U64;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> ParBlocksSizeUser for Backend<R, V> {
    type ParBlocksSize = U4;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> StreamCipherBackend for Backend<R, V> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        unsafe {
            let res = rounds::<R>(&self.v, &self.ctr);
            for c in self.ctr.iter_mut() {
                *c = match size_of::<V::Counter>() {
                    4 => _mm256_add_epi32(*c, _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 1)),
                    8 => _mm256_add_epi64(*c, _mm256_set_epi64x(0, 1, 0, 1)),
                    _ => unreachable!(),
                };
            }

            let res0: [__m128i; 8] = core::mem::transmute(res[0]);

            let block_ptr = block.as_mut_ptr().cast::<__m128i>();
            for i in 0..4 {
                _mm_storeu_si128(block_ptr.add(i), res0[2 * i]);
            }
        }
    }

    #[inline(always)]
    fn gen_par_ks_blocks(&mut self, blocks: &mut ParBlocks<Self>) {
        unsafe {
            let vs = rounds::<R>(&self.v, &self.ctr);

            let pb = PAR_BLOCKS as i32;
            for c in self.ctr.iter_mut() {
                *c = match size_of::<V::Counter>() {
                    4 => _mm256_add_epi32(*c, _mm256_set_epi32(0, 0, 0, pb, 0, 0, 0, pb)),
                    8 => {
                        _mm256_add_epi64(*c, _mm256_set_epi64x(0, i64::from(pb), 0, i64::from(pb)))
                    }
                    _ => unreachable!(),
                }
            }

            let mut block_ptr = blocks.as_mut_ptr().cast::<__m128i>();
            for v in vs {
                let t: [__m128i; 8] = core::mem::transmute(v);
                for i in 0..4 {
                    _mm_storeu_si128(block_ptr.add(i), t[2 * i]);
                    _mm_storeu_si128(block_ptr.add(4 + i), t[2 * i + 1]);
                }
                block_ptr = block_ptr.add(8);
            }
        }
    }
}

#[cfg(feature = "rng")]
impl<R: Rounds, V: Variant> Backend<R, V> {
    #[inline(always)]
    fn rng_gen_par_ks_blocks(&mut self, blocks: &mut [u32; 64]) {
        unsafe {
            let vs = rounds::<R>(&self.v, &self.ctr);

            let pb = PAR_BLOCKS as i32;
            for c in self.ctr.iter_mut() {
                *c = _mm256_add_epi64(*c, _mm256_set_epi64x(0, i64::from(pb), 0, i64::from(pb)));
            }

            let mut block_ptr = blocks.as_mut_ptr().cast::<__m128i>();
            for v in vs {
                let t: [__m128i; 8] = core::mem::transmute(v);
                for i in 0..4 {
                    _mm_storeu_si128(block_ptr.add(i), t[2 * i]);
                    _mm_storeu_si128(block_ptr.add(4 + i), t[2 * i + 1]);
                }
                block_ptr = block_ptr.add(8);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rounds<R: Rounds>(v: &[__m256i; 3], c: &[__m256i; N]) -> [[__m256i; 4]; N] {
    let mut vs: [[__m256i; 4]; N] = [[_mm256_setzero_si256(); 4]; N];
    for i in 0..N {
        vs[i] = [v[0], v[1], v[2], c[i]];
    }
    for _ in 0..R::COUNT {
        double_quarter_round(&mut vs);
    }

    for i in 0..N {
        for j in 0..3 {
            vs[i][j] = _mm256_add_epi32(vs[i][j], v[j]);
        }
        vs[i][3] = _mm256_add_epi32(vs[i][3], c[i]);
    }

    vs
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn double_quarter_round(v: &mut [[__m256i; 4]; N]) {
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
#[target_feature(enable = "avx2")]
unsafe fn rows_to_cols(vs: &mut [[__m256i; 4]; N]) {
    // c >>>= 32; d >>>= 64; a >>>= 96;
    for [a, _, c, d] in vs {
        *c = _mm256_shuffle_epi32(*c, 0b_00_11_10_01); // _MM_SHUFFLE(0, 3, 2, 1)
        *d = _mm256_shuffle_epi32(*d, 0b_01_00_11_10); // _MM_SHUFFLE(1, 0, 3, 2)
        *a = _mm256_shuffle_epi32(*a, 0b_10_01_00_11); // _MM_SHUFFLE(2, 1, 0, 3)
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
#[target_feature(enable = "avx2")]
unsafe fn cols_to_rows(vs: &mut [[__m256i; 4]; N]) {
    // c <<<= 32; d <<<= 64; a <<<= 96;
    for [a, _, c, d] in vs {
        *c = _mm256_shuffle_epi32(*c, 0b_10_01_00_11); // _MM_SHUFFLE(2, 1, 0, 3)
        *d = _mm256_shuffle_epi32(*d, 0b_01_00_11_10); // _MM_SHUFFLE(1, 0, 3, 2)
        *a = _mm256_shuffle_epi32(*a, 0b_00_11_10_01); // _MM_SHUFFLE(0, 3, 2, 1)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_xor_rot(vs: &mut [[__m256i; 4]; N]) {
    let rol16_mask = _mm256_set_epi64x(
        0x0d0c_0f0e_0908_0b0a,
        0x0504_0706_0100_0302,
        0x0d0c_0f0e_0908_0b0a,
        0x0504_0706_0100_0302,
    );
    let rol8_mask = _mm256_set_epi64x(
        0x0e0d_0c0f_0a09_080b,
        0x0605_0407_0201_0003,
        0x0e0d_0c0f_0a09_080b,
        0x0605_0407_0201_0003,
    );

    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    for [a, b, _, d] in vs.iter_mut() {
        *a = _mm256_add_epi32(*a, *b);
        *d = _mm256_xor_si256(*d, *a);
        *d = _mm256_shuffle_epi8(*d, rol16_mask);
    }

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    for [_, b, c, d] in vs.iter_mut() {
        *c = _mm256_add_epi32(*c, *d);
        *b = _mm256_xor_si256(*b, *c);
        *b = _mm256_xor_si256(_mm256_slli_epi32(*b, 12), _mm256_srli_epi32(*b, 20));
    }

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    for [a, b, _, d] in vs.iter_mut() {
        *a = _mm256_add_epi32(*a, *b);
        *d = _mm256_xor_si256(*d, *a);
        *d = _mm256_shuffle_epi8(*d, rol8_mask);
    }

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    for [_, b, c, d] in vs.iter_mut() {
        *c = _mm256_add_epi32(*c, *d);
        *b = _mm256_xor_si256(*b, *c);
        *b = _mm256_xor_si256(_mm256_slli_epi32(*b, 7), _mm256_srli_epi32(*b, 25));
    }
}
