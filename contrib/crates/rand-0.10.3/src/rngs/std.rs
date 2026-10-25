// Copyright 2018 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The standard RNG

use core::convert::Infallible;
use rand_core::{SeedableRng, TryCryptoRng, TryRng};

use chacha20::ChaCha12Rng as Rng;

/// A strong, fast (amortized), non-portable RNG
///
/// This is the "standard" RNG, a generator with the following properties:
///
/// - Non-[portable]: any future library version may replace the algorithm
///   and results may be platform-dependent.
///   (For a portable version, use the [chacha20] crate directly.)
/// - [CSPRNG]: statistically good quality of randomness and [unpredictable]
/// - Fast ([amortized](https://en.wikipedia.org/wiki/Amortized_analysis)):
///   the RNG is fast for bulk generation, but the cost of method calls is not
///   consistent due to usage of an output buffer.
///
/// The current algorithm used is the ChaCha block cipher with 12 rounds. Please
/// see this relevant [rand issue] for the discussion. This may change as new
/// evidence of cipher security and performance becomes available.
///
/// ## Seeding (construction)
///
/// This generator implements the [`SeedableRng`] trait. Any method may be used,
/// but note that `seed_from_u64` is not suitable for usage where security is
/// important. Also note that, even with a fixed seed, output is not [portable].
///
/// Using a fresh seed **direct from the OS** is the most secure option:
/// ```
/// # use rand::{SeedableRng, rngs::{StdRng, SysRng}};
/// let rng = StdRng::try_from_rng(&mut SysRng).unwrap();
/// # let _: StdRng = rng;
/// ```
///
/// Seeding via [`rand::make_rng()`] or [`rand::rng()`] may be
/// faster:
/// ```
/// # use rand::rngs::StdRng;
/// let mut rng: StdRng = rand::make_rng();
/// # let _ = rand::Rng::next_u32(&mut rng);
/// ```
///
/// Any [`SeedableRng`] method may be used, but note that `seed_from_u64` is not
/// suitable where security is required. See also [Seeding RNGs] in the book.
///
/// ## Generation
///
/// The generator implements [`Rng`] and thus also [`RngExt`].
/// See also the [Random Values] chapter in the book.
///
/// [portable]: https://rust-random.github.io/book/crate-reprod.html
/// [Seeding RNGs]: https://rust-random.github.io/book/guide-seeding.html
/// [unpredictable]: https://rust-random.github.io/book/guide-rngs.html#security
/// [Random Values]: https://rust-random.github.io/book/guide-values.html
/// [CSPRNG]: https://rust-random.github.io/book/guide-gen.html#cryptographically-secure-pseudo-random-number-generator
/// [chacha20]: https://crates.io/crates/chacha20
/// [rand issue]: https://github.com/rust-random/rand/issues/932
/// [`Rng`]: rand_core::Rng
/// [`RngExt`]: crate::RngExt
/// [`rand::make_rng()`]: crate::make_rng
/// [`rand::rng()`]: crate::rng
#[derive(Debug, PartialEq, Eq)]
pub struct StdRng(Rng);

impl TryRng for StdRng {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        self.0.try_next_u32()
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        self.0.try_next_u64()
    }

    #[inline(always)]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.0.try_fill_bytes(dst)
    }
}

impl SeedableRng for StdRng {
    // Fix to 256 bits. Changing this is a breaking change!
    type Seed = [u8; 32];

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        StdRng(Rng::from_seed(seed))
    }
}

impl TryCryptoRng for StdRng {}

#[cfg(test)]
mod test {
    use crate::rngs::StdRng;
    use crate::{Rng, RngExt, SeedableRng};

    #[test]
    fn test_stdrng_construction() {
        // Test value-stability of StdRng. This is expected to break any time
        // the algorithm is changed.
        #[rustfmt::skip]
        let seed = [1,0,0,0, 23,0,0,0, 200,1,0,0, 210,30,0,0,
                    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0];

        let target = [10719222850664546238, 14064965282130556830];

        let mut rng0 = StdRng::from_seed(seed);

        let x0 = rng0.next_u64();

        let mut rng1 = StdRng::from_rng(&mut rng0);
        let x1 = rng1.next_u64();

        assert_eq!([x0, x1], target);
    }

    #[test]
    fn test_chacha_true_values_1() {
        // Source: Strombergson 2013, Test Vectors for the Stream Cipher ChaCha
        // draft-strombergson-chacha-test-vectors-01
        // https://datatracker.ietf.org/doc/html/draft-strombergson-chacha-test-vectors-01
        // Converted to LE u128 form (four u128 to one block).
        // TC: all zero key and IV, rounds 12, 256-bit key

        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);

        let mut results = [0u128; 8];
        rng.fill(&mut results);
        let expected = [
            0xd583265f12ce1f8153f955076a9af49b,
            0x5f15ae2ea589007e1474e049bbc32904,
            0x798cfaac3428e82cc0e37ad279f86405,
            0xbe2613412fe80b611969dea02c9f623a,
            0x3d17e08c3371fc86fe743e204188d50b,
            0xb489c04c21851515cccbbd19b7eb28c6,
            0x43c88c1b97b802c611f14ca1cd8d2542,
            0x1693e617b0a64427c0515190ca461ee9,
        ];
        assert_eq!(results, expected);

        assert_eq!(rng.0.get_word_pos(), 32);
    }

    #[test]
    fn test_chacha_true_values_2() {
        // Source: Strombergson 2013, Test Vectors for the Stream Cipher ChaCha
        // TC2: single bit set in key, all zero IV, rounds 12, 256-bit key

        let mut seed = [0u8; 32];
        seed[0] = 1;
        let mut rng = StdRng::from_seed(seed);

        let mut results = [0u128; 8];
        rng.fill(&mut results);
        let expected = [
            0x9a225cdf090f0eef6b0565d596e0512,
            0x10dd4d0bff1802930f5d5290278c2449,
            0xfefdfe067d7a109ee254a4d9392200a6,
            0xc029dc60c972179bf2f944a0eb0f21f0,
            0x2a37692ab05e660e2404c6cbc566730c,
            0xc8a72980b8c4c72a0978bb6fb279f97a,
            0xaf15ba8e302e43907dfcbb17c23b5154,
            0xa9177125baafe601560d10ef48eb5ac6,
        ];
        assert_eq!(results, expected);

        assert_eq!(rng.0.get_word_pos(), 32);
    }

    #[test]
    fn test_chacha_true_values_3() {
        // Source: Strombergson 2013, Test Vectors for the Stream Cipher ChaCha
        // TC3: all zero key, single bit set in IV, rounds 12, 256-bit key

        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        rng.0.set_stream(1);

        let mut results = [0u128; 8];
        rng.fill(&mut results);
        let expected = [
            0x3de08d69eff7ba6d4b8c827bf8bdb864,
            0x6929e19be5ad36988f411457633fb3f8,
            0xa5995d1de898cb9efccf8ef3a053c946,
            0xf1d8f021fb3f31ee4b9450a9a8ffced,
            0x28886a59a2b923fe42c422f2a7b49d55,
            0x23c72a9150a17ca76e8963134fee2251,
            0x67b7d07029cb2037e802f6a024bf0bf,
            0x6fa2523bbd836d3a01c8137c82b91afc,
        ];
        assert_eq!(results, expected);

        assert_eq!(rng.0.get_word_pos(), 32);
    }

    #[test]
    fn test_chacha_true_values_8() {
        // Source: Strombergson 2013, Test Vectors for the Stream Cipher ChaCha
        // TC8: key: 'All your base are belong to us!', IV: IETF2013, rounds 12, 256-bit key

        #[rustfmt::skip]
        let seed = [
            0xc4, 0x6e, 0xc1, 0xb1, 0x8c, 0xe8, 0xa8, 0x78,
            0x72, 0x5a, 0x37, 0xe7, 0x80, 0xdf, 0xb7, 0x35,
            0x1f, 0x68, 0xed, 0x2e, 0x19, 0x4c, 0x79, 0xfb,
            0xc6, 0xae, 0xbe, 0xe1, 0xa6, 0x67, 0x97, 0x5d,
        ];
        let iv = [0x1a, 0xda, 0x31, 0xd5, 0xcf, 0x68, 0x82, 0x21];
        let mut rng = StdRng::from_seed(seed);
        rng.0.set_stream(u64::from_le_bytes(iv));

        let mut results = [0u128; 8];
        rng.fill(&mut results);
        let expected = [
            0x10c08b11dc3be7b4066dbc8427078214,
            0xc19c7e1f25aa8669e018a96c7876793c,
            0x207c8db0992e2d24b483ee160a9a74b2,
            0xabfb0f9db3b1613b28876c46bc802b09,
            0x5495b60d624f9e9b32dbebc16b114bd9,
            0x31d66e96ad465a970c3d47689b3d8e4a,
            0x3c11e5a1df7a04d8c7ead50a53ff2ae4,
            0x2ba4a57be08f1cac89d1f183b8e3f391,
        ];
        assert_eq!(results, expected);

        assert_eq!(rng.0.get_word_pos(), 32);
    }

    #[test]
    fn test_chacha_counter() {
        // Source: rand_chacha implementation
        // We test six blocks: counter=u32::MAX, four blocks from 2^32 (backends
        // which yield four blocks at a time may need to handle this specially)
        // and the first block after this wrap-logic completes.
        // Test: all zero key and IV, block set to u32::MAX, rounds 12, 256-bit key

        let seed = [0u8; 32];
        let mut rng = StdRng::from_seed(seed);
        let block = u32::MAX;
        let words_per_block = 16;
        rng.0.set_word_pos((block as u128) * words_per_block);

        let mut results = [0u128; 4 * 6];
        rng.fill(&mut results);
        let expected = [
            0xf106e2fcbb524248292ac9f150afa6d7,
            0x12032ef6c183b50a83a3309513dd017d,
            0x2c93ff300438eaed6c958a9aa1619382,
            0x74fc0624270ab858508377945edb52d0,
            0xe5f4f4a8b8810524264d8911dc537bcc,
            0x18a6a6cbdc1f823fb1231280056740af,
            0xabdae0a44b1f45edbccc83dcd3f8638a,
            0xad6b649f12f70de567cc39740dbb8a22,
            0x37512785327825dc30ecfaf37a38f5a0,
            0x5af852d2df0dc286c2dd19af39b54e39,
            0xb04dc185c27497ac9f4a4f6769d1b5d,
            0x816492be66439cecd2498c9865284377,
            0x724fe95e0b6cbb8a55b707c06166147f,
            0xe3e7cda19d92b5318024abb34aa31329,
            0x1a3594d7283c077017cd511144bf3db3,
            0x99ab26cf14f38b11d78e413bdce6424c,
            0x553deaed89d3bf630de05408c0f655e8,
            0x86c46a5676fef18f0dc0dff3ee16507c,
            0xd33d6cf5ade97b000b29e3ce614faf51,
            0x5b62dcc48c0fc60326afc5783c40d40c,
            0x44eedc777ed030f43d382d4921eba244,
            0xa2d66a5893ade34a0d17c706e8d89dba,
            0xd229d1f3a07526e47cabd035135012fd,
            0xefae0722059b654dea6945547e535052,
        ];
        assert_eq!(results, expected);

        assert_eq!(rng.0.get_word_pos(), (block as u128) * words_per_block + 96);
    }
}
