//! The encoder and decoder of the DEFLATE format and algorithm.
//!
//! The DEFLATE is defined in [RFC-1951](https://tools.ietf.org/html/rfc1951).
//!
//! # Examples
//! ```
//! #[cfg(feature = "no_std")]
//! use core2::io::{Read, Write};
//! #[cfg(not(feature = "no_std"))]
//! use std::io::{Read, Write};
//! use libflate::deflate::{Encoder, Decoder};
//!
//! // Encoding
//! let mut encoder = Encoder::new(Vec::new());
//! encoder.write_all(b"Hello World!".as_ref()).unwrap();
//! let encoded_data = encoder.finish().into_result().unwrap();
//!
//! // Decoding
//! let mut decoder = Decoder::new(&encoded_data[..]);
//! let mut decoded_data = Vec::new();
//! decoder.read_to_end(&mut decoded_data).unwrap();
//!
//! assert_eq!(decoded_data, b"Hello World!");
//! ```
pub use self::decode::Decoder;
pub use self::encode::EncodeOptions;
pub use self::encode::Encoder;
pub use self::encode::DEFAULT_BLOCK_SIZE;

mod decode;
mod encode;
pub(crate) mod symbol;

#[cfg(test)]
pub(crate) mod test_data;

#[derive(Debug, Clone, Copy)]
enum BlockType {
    Raw = 0b00,
    Fixed = 0b01,
    Dynamic = 0b10,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz77;
    #[cfg(feature = "no_std")]
    use core2::io::{Read, Write};
    #[cfg(not(feature = "no_std"))]
    use std::io::{Read, Write};

    #[test]
    fn encode_and_decode_works() {
        let plain = (0..lz77::MAX_DISTANCE as u32 * 32)
            .map(|i| i as u8)
            .collect::<Vec<_>>();

        let buffer = Vec::new();
        let mut encoder = Encoder::new(buffer);
        encoder.write_all(&plain[..]).expect("encode");
        let encoded = encoder.finish().into_result().unwrap();

        let mut buffer = Vec::new();
        let mut decoder = Decoder::new(&encoded[..]);
        decoder.read_to_end(&mut buffer).expect("decode");

        assert_eq!(buffer, plain);
    }
}
