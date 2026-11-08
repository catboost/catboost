use super::braid::CRC32_LSB_POLY;

pub const fn crc32_combine(crc1: u32, crc2: u32, len2: u64) -> u32 {
    crc32_combine_op(crc1, crc2, crc32_combine_gen(len2))
}

#[inline(always)]
pub const fn crc32_combine_gen(len2: u64) -> u32 {
    x2nmodp(len2, 3)
}

#[inline(always)]
pub const fn crc32_combine_op(crc1: u32, crc2: u32, op: u32) -> u32 {
    multmodp(op, crc1) ^ crc2
}

const X2N_TABLE: [u32; 32] = [
    0x40000000, 0x20000000, 0x08000000, 0x00800000, 0x00008000, 0xedb88320, 0xb1e6b092, 0xa06a2517,
    0xed627dae, 0x88d14467, 0xd7bbfe6a, 0xec447f11, 0x8e7ea170, 0x6427800e, 0x4d47bae0, 0x09fe548f,
    0x83852d0f, 0x30362f1a, 0x7b5a9cc3, 0x31fec169, 0x9fec022a, 0x6c8dedc4, 0x15d6874d, 0x5fde7a4e,
    0xbad90e37, 0x2e4e5eef, 0x4eaba214, 0xa8a472c0, 0x429a969e, 0x148d302a, 0xc40ba6d0, 0xc4e22c3c,
];

// Return a(x) multiplied by b(x) modulo p(x), where p(x) is the CRC polynomial,
// reflected. For speed, this requires that a not be zero.
pub(crate) const fn multmodp(a: u32, mut b: u32) -> u32 {
    let mut m = 1 << 31;
    let mut p = 0;

    loop {
        if (a & m) != 0 {
            p ^= b;
            if (a & (m - 1)) == 0 {
                break;
            }
        }
        m >>= 1;
        b = if (b & 1) != 0 {
            (b >> 1) ^ CRC32_LSB_POLY as u32
        } else {
            b >> 1
        };
    }

    p
}

// Return x^(n * 2^k) modulo p(x).
pub(crate) const fn x2nmodp(mut n: u64, mut k: u32) -> u32 {
    let mut p: u32 = 1 << 31; /* x^0 == 1 */

    while n > 0 {
        if (n & 1) != 0 {
            p = multmodp(X2N_TABLE[k as usize & 31], p);
        }
        n >>= 1;
        k += 1;
    }

    p
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::crc32::crc32;

    #[test]
    fn test_crc32_combine() {
        ::quickcheck::quickcheck(test as fn(_) -> _);

        fn test(data: Vec<u8>) -> bool {
            let Some(buf_len) = data.first().copied() else {
                return true;
            };

            let buf_size = Ord::max(buf_len, 1) as usize;

            let crc0 = 0;
            let mut crc1 = crc0;
            let mut crc2 = crc0;

            /* CRC32 */
            for chunk in data.chunks(buf_size) {
                let crc3 = crc32(crc0, chunk);
                let op = crc32_combine_gen(chunk.len() as _);
                let crc4 = crc32_combine_op(crc1, crc3, op);
                crc1 = crc32(crc1, chunk);

                assert_eq!(crc1, crc4);
            }

            crc2 = crc32(crc2, &data);

            assert_eq!(crc1, crc2);

            let combine1 = crc32_combine(crc1, crc2, data.len() as _);
            let combine2 = crc32_combine(crc1, crc1, data.len() as _);
            assert_eq!(combine1, combine2);

            // Fast CRC32 combine.
            let op = crc32_combine_gen(data.len() as _);
            let combine1 = crc32_combine_op(crc1, crc2, op);
            let combine2 = crc32_combine_op(crc2, crc1, op);
            assert_eq!(combine1, combine2);

            let combine1 = crc32_combine(crc1, crc2, data.len() as _);
            let combine2 = crc32_combine_op(crc2, crc1, op);
            assert_eq!(combine1, combine2);

            true
        }
    }
}
