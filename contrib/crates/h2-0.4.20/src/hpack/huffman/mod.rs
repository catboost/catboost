mod decode;
mod encode;
mod table;

pub use self::decode::decode;
pub use self::encode::encode;

#[cfg(test)]
mod test {
    use super::*;

    use bytes::BytesMut;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn decode(src: &[u8]) -> BytesMut {
        let mut buf = BytesMut::new();
        super::decode(src, &mut buf).unwrap()
    }

    #[test]
    fn encode_decode_str() {
        const DATA: &[&str] = &[
            "hello world",
            ":method",
            ":scheme",
            ":authority",
            "yahoo.co.jp",
            "GET",
            "http",
            ":path",
            "/images/top/sp2/cmn/logo-ns-130528.png",
            "example.com",
            "hpack-test",
            "xxxxxxx1",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:16.0) Gecko/20100101 Firefox/16.0",
            "accept",
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "cookie",
            "B=76j09a189a6h4&b=3&s=0b",
            "TE",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi non bibendum libero. \
             Etiam ultrices lorem ut.",
        ];

        for s in DATA {
            let mut dst = BytesMut::with_capacity(s.len());
            encode(s.as_bytes(), &mut dst);
            let decoded = decode(&dst);
            assert_eq!(&decoded[..], s.as_bytes());
        }
    }

    #[test]
    fn encode_decode_u8() {
        const DATA: &[&[u8]] = &[b"\0", b"\0\0\0", b"\0\x01\x02\x03\x04\x05", b"\xFF\xF8"];

        for s in DATA {
            let mut dst = BytesMut::with_capacity(s.len());
            encode(s, &mut dst);
            let decoded = decode(&dst);
            assert_eq!(&decoded[..], &s[..]);
        }
    }

    #[test]
    fn encode_decode_all_octets() {
        let src: Vec<_> = (0..=u8::MAX).collect();
        let mut encoded = BytesMut::new();
        encode(&src, &mut encoded);
        assert_eq!(decode(&encoded), src);
    }

    // '0' has a shortest possible code (5 bits), so all-'0' strings expand
    // the most when decoding, hitting the decoder's output reservation
    // exactly.
    #[test]
    fn encode_decode_max_expansion() {
        for len in 0..100 {
            let src = vec![b'0'; len];
            let mut encoded = BytesMut::new();
            encode(&src, &mut encoded);
            assert_eq!(&decode(&encoded)[..], &src[..], "len={}", len);
        }
    }

    // Round-trip strings of random bytes of every length
    #[test]
    fn encode_decode_random() {
        let mut rng = StdRng::seed_from_u64(0x123456789abcdef);

        for len in 0..600 {
            let src: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let mut encoded = BytesMut::new();
            encode(&src, &mut encoded);
            assert_eq!(&decode(&encoded)[..], &src[..], "len={}", len);
        }
    }
}
