//! The encoder and decoder of the ZLIB format.
//!
//! The ZLIB format is defined in [RFC-1950](https://tools.ietf.org/html/rfc1950).
//!
//! # Examples
//! ```
//! #[cfg(feature = "no_std")]
//! use core2::io::{Read, Write};
//! #[cfg(not(feature = "no_std"))]
//! use std::io::{Read, Write};
//! use libflate::zlib::Encoder;
//! use libflate::non_blocking::zlib::Decoder;
//!
//! // Encoding
//! let mut encoder = Encoder::new(Vec::new()).unwrap();
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
use crate::checksum;
use crate::non_blocking::deflate;
use crate::zlib::Header;
#[cfg(feature = "no_std")]
use core2::io::{self, Read};
#[cfg(not(feature = "no_std"))]
use std::io::{self, Read};

/// ZLIB decoder which supports non-blocking I/O.
#[derive(Debug)]
pub struct Decoder<R> {
    header: Option<Header>,
    reader: deflate::Decoder<R>,
    adler32: checksum::Adler32,
    eos: bool,
}
impl<R: Read> Decoder<R> {
    /// Makes a new decoder instance.
    ///
    /// `inner` is to be decoded ZLIB stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Read;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Read;
    /// use libflate::non_blocking::zlib::Decoder;
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let mut decoder = Decoder::new(&encoded_data[..]);
    /// let mut buf = Vec::new();
    /// decoder.read_to_end(&mut buf).unwrap();
    ///
    /// assert_eq!(buf, b"Hello World!");
    /// ```
    pub fn new(inner: R) -> Self {
        Decoder {
            header: None,
            reader: deflate::Decoder::new(inner),
            adler32: checksum::Adler32::new(),
            eos: false,
        }
    }

    /// Returns the header of the ZLIB stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::CompressionLevel;
    /// use libflate::non_blocking::zlib::Decoder;
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let mut decoder = Decoder::new(&encoded_data[..]);
    /// assert_eq!(decoder.header().unwrap().compression_level(),
    ///            CompressionLevel::Default);
    /// ```
    pub fn header(&mut self) -> io::Result<&Header> {
        if let Some(ref header) = self.header {
            Ok(header)
        } else {
            let header = self
                .reader
                .bit_reader_mut()
                .transaction(|r| Header::read_from(r.as_inner_mut()))?;
            self.header = Some(header);
            self.header()
        }
    }

    /// Returns the immutable reference to the inner stream.
    pub fn as_inner_ref(&self) -> &R {
        self.reader.as_inner_ref()
    }

    /// Returns the mutable reference to the inner stream.
    pub fn as_inner_mut(&mut self) -> &mut R {
        self.reader.as_inner_mut()
    }

    /// Unwraps this `Decoder`, returning the underlying reader.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Cursor;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Cursor;
    /// use libflate::non_blocking::zlib::Decoder;
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let decoder = Decoder::new(Cursor::new(&encoded_data));
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data);
    /// ```
    pub fn into_inner(self) -> R {
        self.reader.into_inner()
    }
}
impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.header.is_none() {
            self.header()?;
        }
        if self.eos {
            Ok(0)
        } else {
            let read_size = self.reader.read(buf)?;
            if read_size == 0 {
                let adler32 = self.reader.bit_reader_mut().transaction(|r| {
                    let mut buf = [0; 4];
                    r.as_inner_mut()
                        .read_exact(&mut buf)
                        .and(Ok(u32::from_be_bytes(buf)))
                })?;
                self.eos = true;
                // checksum verification is skipped during fuzzing
                // so that random data from fuzzer can reach actually interesting code
                // Compilation flag 'fuzzing' is automatically set by all 3 Rust fuzzers.
                if cfg!(not(fuzzing)) && adler32 != self.adler32.value() {
                    Err(invalid_data_error!(
                        "Adler32 checksum mismatched: value={}, expected={}",
                        self.adler32.value(),
                        adler32
                    ))
                } else {
                    Ok(0)
                }
            } else {
                self.adler32.update(&buf[..read_size]);
                Ok(read_size)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{nb_read_to_end, WouldBlockReader};
    use crate::zlib::{EncodeOptions, Encoder};
    #[cfg(feature = "no_std")]
    use core2::io::Write;
    #[cfg(not(feature = "no_std"))]
    use std::io::Write;

    fn decode_all(buf: &[u8]) -> io::Result<Vec<u8>> {
        let decoder = Decoder::new(WouldBlockReader::new(buf));
        nb_read_to_end(decoder)
    }
    #[cfg(not(feature = "no_std"))]
    fn default_encode(buf: &[u8]) -> io::Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(buf.as_ref()).unwrap();
        encoder.finish().into_result()
    }
    #[cfg(not(feature = "no_std"))]
    macro_rules! assert_encode_decode {
        ($input:expr) => {{
            let encoded = default_encode(&$input[..]).unwrap();
            assert_eq!(decode_all(&encoded).unwrap(), &$input[..]);
        }};
    }

    const DECODE_WORKS_TESTDATA: [u8; 20] = [
        120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73, 81, 4, 0, 28, 73, 4, 62,
    ];
    #[test]
    fn decode_works() {
        let encoded = DECODE_WORKS_TESTDATA;
        let buf = decode_all(&encoded[..]).unwrap();
        let expected = b"Hello World!";
        assert_eq!(buf, expected);
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn default_encode_works() {
        let plain = b"Hello World! Hello ZLIB!!";
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        io::copy(&mut &plain[..], &mut encoder).unwrap();
        let encoded = encoder.finish().into_result().unwrap();
        assert_eq!(decode_all(&encoded).unwrap(), plain);
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn best_speed_encode_works() {
        let plain = b"Hello World! Hello ZLIB!!";
        let mut encoder =
            Encoder::with_options(Vec::new(), EncodeOptions::default().fixed_huffman_codes())
                .unwrap();
        io::copy(&mut &plain[..], &mut encoder).unwrap();
        let encoded = encoder.finish().into_result().unwrap();
        assert_eq!(decode_all(&encoded).unwrap(), plain);
    }

    const RAW_ENCODE_WORKS_EXPECTED: [u8; 23] = [
        120, 1, 1, 12, 0, 243, 255, 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 28, 73,
        4, 62,
    ];
    #[test]
    fn raw_encode_works() {
        let plain = b"Hello World!";
        let mut encoder =
            Encoder::with_options(Vec::new(), EncodeOptions::new().no_compression()).unwrap();
        encoder.write_all(plain.as_ref()).unwrap();
        let encoded = encoder.finish().into_result().unwrap();
        let expected = RAW_ENCODE_WORKS_EXPECTED;
        assert_eq!(encoded, expected);
        assert_eq!(decode_all(&encoded).unwrap(), plain);
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn test_issue_2() {
        // See: https://github.com/sile/libflate/issues/2
        assert_encode_decode!([
            163, 181, 167, 40, 62, 239, 41, 125, 189, 217, 61, 122, 20, 136, 160, 178, 119, 217,
            217, 41, 125, 189, 97, 195, 101, 47, 170,
        ]);
        assert_encode_decode!([
            162, 58, 99, 211, 7, 64, 96, 36, 57, 155, 53, 166, 76, 14, 238, 66, 66, 148, 154, 124,
            162, 58, 99, 188, 138, 131, 171, 189, 54, 229, 192, 38, 29, 240, 122, 28,
        ]);
        assert_encode_decode!([
            239, 238, 212, 42, 5, 46, 186, 67, 122, 247, 30, 61, 219, 62, 228, 202, 164, 205, 139,
            109, 99, 181, 99, 181, 99, 122, 30, 12, 62, 46, 27, 145, 241, 183, 137,
        ]);
        assert_encode_decode!([
            88, 202, 64, 12, 125, 108, 153, 49, 164, 250, 71, 19, 4, 108, 111, 108, 237, 205, 208,
            77, 217, 100, 118, 49, 10, 64, 12, 125, 51, 202, 69, 67, 181, 146, 86,
        ]);
    }
}
