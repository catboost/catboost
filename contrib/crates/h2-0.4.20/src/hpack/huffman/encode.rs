use crate::hpack::huffman::table::{ENCODE_CODES, ENCODE_CODE_LENGTHS};

use bytes::{BufMut, BytesMut};

pub fn encode(src: &[u8], dst: &mut BytesMut) {
    let mut bits = 0u64;
    let mut bits_len = 0;

    for &b in src {
        let nbits = ENCODE_CODE_LENGTHS[b as usize] as usize;
        let code = ENCODE_CODES[b as usize] as u64;

        bits = (bits << nbits) | code;
        bits_len += nbits;

        if bits_len >= 32 {
            let remaining = bits_len - 32;
            dst.put_u32((bits >> remaining) as u32);

            bits_len = remaining;
            if remaining == 0 {
                bits = 0;
            } else {
                bits &= (1 << remaining) - 1;
            }
        }
    }

    if bits_len != 0 {
        // Pad the final byte with the EOS prefix (all ones).
        let padding = 8 - (bits_len % 8);
        let padding = if padding == 8 { 0 } else { padding };
        bits = (bits << padding) | ((1 << padding) - 1);
        bits_len += padding;

        while bits_len != 0 {
            bits_len -= 8;
            dst.put_u8((bits >> bits_len) as u8);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn encode_single_byte() {
        let mut dst = BytesMut::with_capacity(1);

        encode(b"o", &mut dst);
        assert_eq!(&dst[..], &[0b00111111]);

        dst.clear();
        encode(b"0", &mut dst);
        assert_eq!(&dst[..], &[7]);

        dst.clear();
        encode(b"A", &mut dst);
        assert_eq!(&dst[..], &[(0x21 << 2) + 3]);
    }

    #[test]
    fn encode_rfc_example() {
        let mut dst = BytesMut::new();

        encode(b"www.example.com", &mut dst);

        assert_eq!(
            &dst[..],
            &[0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff]
        );
    }
}

/*
// uncomment to run benchmarks
#[cfg(test)]
mod bench {
    extern crate test;

    use self::test::{black_box, Bencher};
    use super::*;

    fn encode_input(b: &mut Bencher, input: &[u8]) {
        let mut encoded = BytesMut::new();
        encode(input, &mut encoded);

        b.bytes = input.len() as u64;
        b.iter(|| {
            encoded.clear();
            encode(black_box(input), &mut encoded);
            black_box(encoded.as_ref());
        });
    }

    #[bench]
    fn encode_short_ascii(b: &mut Bencher) {
        encode_input(b, b"www.example.com");
    }

    #[bench]
    fn encode_header_value(b: &mut Bencher) {
        encode_input(
            b,
            b"text/html,application/xhtml+xml,application/xml;q=0.9;q=0.8",
        );
    }

    #[bench]
    fn encode_long_ascii(b: &mut Bencher) {
        encode_input(
            b,
            b"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:16.0) Gecko/20100101 Firefox/16.0",
        );
    }

    #[bench]
    fn encode_all_octets(b: &mut Bencher) {
        let input: Vec<_> = (0..=u8::MAX).collect();
        encode_input(b, &input);
    }
}
*/
