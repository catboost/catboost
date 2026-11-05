use super::symbol;
use crate::bit;
use crate::lz77;
#[cfg(feature = "no_std")]
use core2::io::{self, Read};
#[cfg(not(feature = "no_std"))]
use std::io::{self, Read};

/// DEFLATE decoder.
#[derive(Debug)]
pub struct Decoder<R> {
    bit_reader: bit::BitReader<R>,
    lz77_decoder: lz77::Lz77Decoder,
    eos: bool,
}
impl<R> Decoder<R>
where
    R: Read,
{
    /// Makes a new decoder instance.
    ///
    /// `inner` is to be decoded DEFLATE stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::{Cursor, Read};
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::{Cursor, Read};
    /// use libflate::deflate::Decoder;
    ///
    /// let encoded_data = [243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73, 81, 4, 0];
    /// let mut decoder = Decoder::new(&encoded_data[..]);
    /// let mut buf = Vec::new();
    /// decoder.read_to_end(&mut buf).unwrap();
    ///
    /// assert_eq!(buf, b"Hello World!");
    /// ```
    pub fn new(inner: R) -> Self {
        Decoder {
            bit_reader: bit::BitReader::new(inner),
            lz77_decoder: lz77::Lz77Decoder::new(),
            eos: false,
        }
    }

    /// Returns the immutable reference to the inner stream.
    pub fn as_inner_ref(&self) -> &R {
        self.bit_reader.as_inner_ref()
    }

    /// Returns the mutable reference to the inner stream.
    pub fn as_inner_mut(&mut self) -> &mut R {
        self.bit_reader.as_inner_mut()
    }

    /// Unwraps this `Decoder`, returning the underlying reader.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Cursor;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Cursor;
    /// use libflate::deflate::Decoder;
    ///
    /// let encoded_data = [243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73, 81, 4, 0];
    /// let decoder = Decoder::new(Cursor::new(&encoded_data));
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data);
    /// ```
    pub fn into_inner(self) -> R {
        self.bit_reader.into_inner()
    }

    /// Returns the data that has been decoded but has not yet been read.
    ///
    /// This method is useful to retrieve partial decoded data when the decoding process is failed.
    pub fn unread_decoded_data(&self) -> &[u8] {
        self.lz77_decoder.buffer()
    }

    pub(crate) fn reset(&mut self) {
        self.bit_reader.reset();
        self.lz77_decoder.clear();
        self.eos = false
    }

    fn read_non_compressed_block(&mut self) -> io::Result<()> {
        self.bit_reader.reset();
        let mut buf = [0; 2];
        self.bit_reader.as_inner_mut().read_exact(&mut buf)?;
        let len = u16::from_le_bytes(buf);
        self.bit_reader.as_inner_mut().read_exact(&mut buf)?;
        let nlen = u16::from_le_bytes(buf);
        if !len != nlen {
            Err(invalid_data_error!(
                "LEN={} is not the one's complement of NLEN={}",
                len,
                nlen
            ))
        } else {
            self.lz77_decoder
                .extend_from_reader(self.bit_reader.as_inner_mut().take(len.into()))
                .and_then(|used| {
                    if used != len.into() {
                        Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            #[cfg(not(feature = "no_std"))]
                            format!("The reader has incorrect length: expected {len}, read {used}"),
                            #[cfg(feature = "no_std")]
                            "The reader has incorrect length",
                        ))
                    } else {
                        Ok(())
                    }
                })
        }
    }
    fn read_compressed_block<H>(&mut self, huffman: &H) -> io::Result<()>
    where
        H: symbol::HuffmanCodec,
    {
        let symbol_decoder = huffman.load(&mut self.bit_reader)?;
        loop {
            let s = symbol_decoder.decode_unchecked(&mut self.bit_reader);
            self.bit_reader.check_last_error()?;
            match s {
                symbol::Symbol::Code(code) => {
                    self.lz77_decoder.decode(code)?;
                }
                symbol::Symbol::EndOfBlock => {
                    break;
                }
            }
        }
        Ok(())
    }
}
impl<R> Read for Decoder<R>
where
    R: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.lz77_decoder.buffer().is_empty() {
            self.lz77_decoder.read(buf)
        } else if self.eos {
            Ok(0)
        } else {
            let bfinal = self.bit_reader.read_bit()?;
            let btype = self.bit_reader.read_bits(2)?;
            self.eos = bfinal;
            match btype {
                0b00 => {
                    self.read_non_compressed_block()?;
                    self.read(buf)
                }
                0b01 => {
                    self.read_compressed_block(&symbol::FixedHuffmanCodec)?;
                    self.read(buf)
                }
                0b10 => {
                    self.read_compressed_block(&symbol::DynamicHuffmanCodec)?;
                    self.read(buf)
                }
                0b11 => Err(invalid_data_error!(
                    "btype 0x11 of DEFLATE is reserved(error) value"
                )),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "no_std"))]
    use super::*;
    use crate::deflate::symbol::{DynamicHuffmanCodec, HuffmanCodec};
    #[cfg(not(feature = "no_std"))]
    use std::io;

    #[test]
    fn test_issues_3() {
        // see: https://github.com/sile/libflate/issues/3
        let input = [
            180, 253, 73, 143, 28, 201, 150, 46, 8, 254, 150, 184, 139, 75, 18, 69, 247, 32, 157,
            51, 27, 141, 132, 207, 78, 210, 167, 116, 243, 160, 223, 136, 141, 66, 205, 76, 221,
            76, 195, 213, 84, 236, 234, 224, 78, 227, 34, 145, 221, 139, 126, 232, 69, 173, 170,
            208, 192, 219, 245, 67, 3, 15, 149, 120, 171, 70, 53, 106, 213, 175, 23, 21, 153, 139,
            254, 27, 249, 75, 234, 124, 71, 116, 56, 71, 68, 212, 204, 121, 115, 64, 222, 160, 203,
            119, 142, 170, 169, 138, 202, 112, 228, 140, 38,
        ];
        let mut bit_reader = crate::bit::BitReader::new(&input[..]);
        assert_eq!(bit_reader.read_bit().unwrap(), false); // not final block
        assert_eq!(bit_reader.read_bits(2).unwrap(), 0b10); // DynamicHuffmanCodec
        DynamicHuffmanCodec.load(&mut bit_reader).unwrap();
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn it_works() {
        let input = [
            180, 253, 73, 143, 28, 201, 150, 46, 8, 254, 150, 184, 139, 75, 18, 69, 247, 32, 157,
            51, 27, 141, 132, 207, 78, 210, 167, 116, 243, 160, 223, 136, 141, 66, 205, 76, 221,
            76, 195, 213, 84, 236, 234, 224, 78, 227, 34, 145, 221, 139, 126, 232, 69, 173, 170,
            208, 192, 219, 245, 67, 3, 15, 149, 120, 171, 70, 53, 106, 213, 175, 23, 21, 153, 139,
            254, 27, 249, 75, 234, 124, 71, 116, 56, 71, 68, 212, 204, 121, 115, 64, 222, 160, 203,
            119, 142, 170, 169, 138, 202, 112, 228, 140, 38, 171, 162, 88, 212, 235, 56, 136, 231,
            233, 239, 113, 249, 163, 252, 16, 42, 138, 49, 226, 108, 73, 28, 153,
        ];
        let mut decoder = Decoder::new(&input[..]);

        let result = io::copy(&mut decoder, &mut io::sink());
        assert!(result.is_err());

        let error = result.err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().starts_with("Too long backword reference"));
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn test_issue_64() {
        let input = b"\x04\x04\x04\x05:\x1az*\xfc\x06\x01\x90\x01\x06\x01";
        let mut decoder = Decoder::new(&input[..]);
        assert!(io::copy(&mut decoder, &mut io::sink()).is_err());
    }
}
