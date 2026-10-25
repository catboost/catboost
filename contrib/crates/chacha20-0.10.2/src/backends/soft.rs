//! Portable implementation which does not rely on architecture-specific intrinsics.

#![allow(clippy::cast_possible_truncation)]

use crate::{ChaChaCore, Rounds, STATE_WORDS, Variant, quarter_round};

#[cfg(feature = "cipher")]
use crate::chacha::Block;
#[cfg(feature = "cipher")]
use cipher::{
    BlockSizeUser, ParBlocksSizeUser, StreamCipherBackend,
    consts::{U1, U64},
};

#[cfg(feature = "rng")]
use crate::rng::BLOCK_WORDS;

pub(crate) struct Backend<'a, R: Rounds, V: Variant>(pub(crate) &'a mut ChaChaCore<R, V>);

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> BlockSizeUser for Backend<'_, R, V> {
    type BlockSize = U64;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> ParBlocksSizeUser for Backend<'_, R, V> {
    type ParBlocksSize = U1;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> StreamCipherBackend for Backend<'_, R, V> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        let res = run_rounds::<R>(&self.0.state);
        let mut ctr = (u64::from(self.0.state[13]) << 32) | u64::from(self.0.state[12]);
        ctr = ctr.wrapping_add(1);
        self.0.state[12] = ctr as u32;
        if size_of::<V::Counter>() == 8 {
            self.0.state[13] = (ctr >> 32) as u32;
        }

        for (chunk, val) in block.chunks_exact_mut(4).zip(res.iter()) {
            chunk.copy_from_slice(&val.to_le_bytes());
        }
    }
}

#[cfg(feature = "rng")]
impl<R: Rounds, V: Variant> Backend<'_, R, V> {
    #[inline(always)]
    pub(crate) fn gen_ks_blocks(&mut self, buffer: &mut [u32; 64]) {
        for block in 0..4 {
            let res = run_rounds::<R>(&self.0.state);
            let mut ctr = (u64::from(self.0.state[13]) << 32) | u64::from(self.0.state[12]);
            ctr = ctr.wrapping_add(1);
            self.0.state[12] = ctr as u32;
            self.0.state[13] = (ctr >> 32) as u32;

            buffer[block * BLOCK_WORDS as usize..(block + 1) * BLOCK_WORDS as usize]
                .copy_from_slice(&res);
        }
    }
}

#[inline(always)]
fn run_rounds<R: Rounds>(state: &[u32; STATE_WORDS]) -> [u32; STATE_WORDS] {
    let mut res = *state;

    for _ in 0..R::COUNT {
        // column rounds
        quarter_round(0, 4, 8, 12, &mut res);
        quarter_round(1, 5, 9, 13, &mut res);
        quarter_round(2, 6, 10, 14, &mut res);
        quarter_round(3, 7, 11, 15, &mut res);

        // diagonal rounds
        quarter_round(0, 5, 10, 15, &mut res);
        quarter_round(1, 6, 11, 12, &mut res);
        quarter_round(2, 7, 8, 13, &mut res);
        quarter_round(3, 4, 9, 14, &mut res);
    }

    for (s1, s0) in res.iter_mut().zip(state.iter()) {
        *s1 = s1.wrapping_add(*s0);
    }
    res
}
