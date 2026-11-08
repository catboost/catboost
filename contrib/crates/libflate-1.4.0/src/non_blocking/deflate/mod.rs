//! The decoder of the DEFLATE format and algorithm.
//!
//! The DEFLATE is defined in [RFC-1951](https://tools.ietf.org/html/rfc1951).
//!
//! # Examples
//! ```
//! #[cfg(feature = "no_std")]
//! use core2::io::{Read, Write};
//! #[cfg(not(feature = "no_std"))]
//! use std::io::{Read, Write};
//! use libflate::deflate::Encoder;
//! use libflate::non_blocking::deflate::Decoder;
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

mod decode;
