//! The adler32 checksum algorithm.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
mod avx512_vnni;
mod generic;
#[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
mod lsx;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
mod wasm;

pub fn adler32(start_checksum: u32, data: &[u8]) -> u32 {
    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    if cfg!(all(target_feature = "avx512f", target_feature = "avx512bw")) {
        return avx512::adler32_avx512(start_checksum, data);
    }

    #[cfg(target_arch = "x86_64")]
    if crate::cpu_features::is_enabled_avx2_and_bmi2() {
        return avx2::adler32_avx2(start_checksum, data);
    }

    #[cfg(target_arch = "aarch64")]
    if crate::cpu_features::is_enabled_neon() {
        return self::neon::adler32_neon(start_checksum, data);
    }

    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    if crate::cpu_features::is_enabled_simd128() {
        return self::wasm::adler32_wasm(start_checksum, data);
    }

    #[cfg(all(target_arch = "loongarch64", feature = "lsx"))]
    if crate::cpu_features::is_enabled_lsx() {
        return self::lsx::adler32_lsx(start_checksum, data);
    }

    generic::adler32_rust(start_checksum, data)
}

pub(crate) fn adler32_fold_copy(start_checksum: u32, dst: &mut [u8], src: &[u8]) -> u32 {
    debug_assert!(dst.len() >= src.len(), "{} < {}", dst.len(), src.len());

    // integrating the memcpy into the adler32 function did not have any benefits, and in fact was
    // a bit slower for very small chunk sizes.
    dst[..src.len()].copy_from_slice(src);
    adler32(start_checksum, src)
}

pub fn adler32_combine(adler1: u32, adler2: u32, len2: u64) -> u32 {
    const BASE: u64 = self::BASE as u64;

    let rem = len2 % BASE;

    let adler1 = adler1 as u64;
    let adler2 = adler2 as u64;

    /* the derivation of this formula is left as an exercise for the reader */
    let mut sum1 = adler1 & 0xffff;
    let mut sum2 = rem * sum1;
    sum2 %= BASE;
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;

    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum2 >= (BASE << 1) {
        sum2 -= BASE << 1;
    }
    if sum2 >= BASE {
        sum2 -= BASE;
    }

    (sum1 | (sum2 << 16)) as u32
}

// inefficient but correct, useful for testing
#[cfg(test)]
fn naive_adler32(start_checksum: u32, data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521; // Largest prime smaller than 2^16

    let mut a = start_checksum & 0xFFFF;
    let mut b = (start_checksum >> 16) & 0xFFFF;

    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    (b << 16) | a
}

const BASE: u32 = 65521; /* largest prime smaller than 65536 */
const NMAX: u32 = 5552;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn naive_is_fancy_small_inputs() {
        for i in 0..128 {
            let v = (0u8..i).collect::<Vec<_>>();
            assert_eq!(naive_adler32(1, &v), generic::adler32_rust(1, &v));
        }
    }

    #[test]
    fn test_adler32_combine() {
        ::quickcheck::quickcheck(test as fn(_) -> _);

        fn test(data: Vec<u8>) -> bool {
            let Some(buf_len) = data.first().copied() else {
                return true;
            };

            let buf_size = Ord::max(buf_len, 1) as usize;

            let mut adler1 = 1;
            let mut adler2 = 1;

            for chunk in data.chunks(buf_size) {
                adler1 = adler32(adler1, chunk);
            }

            adler2 = adler32(adler2, &data);

            assert_eq!(adler1, adler2);

            let combine1 = adler32_combine(adler1, adler2, data.len() as _);
            let combine2 = adler32_combine(adler1, adler1, data.len() as _);
            assert_eq!(combine1, combine2);

            true
        }
    }
}
