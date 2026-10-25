use crate::deflate::symbol::{self, HuffmanCodec};
use crate::lz77;
use crate::non_blocking::transaction::TransactionalBitReader;
#[cfg(feature = "no_std")]
use core::cmp;
#[cfg(feature = "no_std")]
use core2::io::{self, Read};
#[cfg(not(feature = "no_std"))]
use std::{
    cmp,
    io::{self, Read},
};

/// DEFLATE decoder which supports non-blocking I/O.
#[derive(Debug)]
pub struct Decoder<R> {
    state: DecoderState,
    eos: bool,
    bit_reader: TransactionalBitReader<R>,
    block_decoder: BlockDecoder,
}
impl<R: Read> Decoder<R> {
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
    /// use libflate::non_blocking::deflate::Decoder;
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
            state: DecoderState::ReadBlockHeader,
            eos: false,
            bit_reader: TransactionalBitReader::new(inner),
            block_decoder: BlockDecoder::new(),
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
    /// use libflate::non_blocking::deflate::Decoder;
    ///
    /// let encoded_data = [243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73, 81, 4, 0];
    /// let decoder = Decoder::new(Cursor::new(&encoded_data));
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data);
    /// ```
    pub fn into_inner(self) -> R {
        self.bit_reader.into_inner()
    }

    pub(crate) fn bit_reader_mut(&mut self) -> &mut TransactionalBitReader<R> {
        &mut self.bit_reader
    }
}
impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut read_size;
        loop {
            let next = match self.state {
                DecoderState::ReadBlockHeader => {
                    let (bfinal, btype) = self.bit_reader.transaction(|r| {
                        let bfinal = r.read_bit()?;
                        let btype = r.read_bits(2)?;
                        Ok((bfinal, btype))
                    })?;
                    self.eos = bfinal;
                    self.block_decoder.enter_new_block();
                    match btype {
                        0b00 => DecoderState::ReadNonCompressedBlockLen,
                        0b01 => DecoderState::LoadFixedHuffmanCode,
                        0b10 => DecoderState::LoadDynamicHuffmanCode,
                        0b11 => {
                            return Err(invalid_data_error!(
                                "btype 0x11 of DEFLATE is reserved(error) value"
                            ));
                        }
                        _ => unreachable!(),
                    }
                }
                DecoderState::ReadNonCompressedBlockLen => {
                    let len = self.bit_reader.transaction(|r| {
                        r.reset();
                        let mut buf = [0; 2];
                        r.as_inner_mut().read_exact(&mut buf)?;
                        let len = u16::from_le_bytes(buf);
                        r.as_inner_mut().read_exact(&mut buf)?;
                        let nlen = u16::from_le_bytes(buf);
                        if !len != nlen {
                            Err(invalid_data_error!(
                                "LEN={} is not the one's complement of NLEN={}",
                                len,
                                nlen
                            ))
                        } else {
                            Ok(len)
                        }
                    })?;
                    DecoderState::ReadNonCompressedBlock { len }
                }
                DecoderState::ReadNonCompressedBlock { len: 0 } => {
                    if self.eos {
                        read_size = 0;
                        break;
                    } else {
                        DecoderState::ReadBlockHeader
                    }
                }
                DecoderState::ReadNonCompressedBlock { ref mut len } => {
                    let buf_len = buf.len();
                    let buf = &mut buf[..cmp::min(buf_len, *len as usize)];
                    read_size = self.bit_reader.as_inner_mut().read(buf)?;

                    self.block_decoder.extend(&buf[..read_size]);
                    *len -= read_size as u16;
                    break;
                }
                DecoderState::LoadFixedHuffmanCode => {
                    let symbol_decoder = self
                        .bit_reader
                        .transaction(|r| symbol::FixedHuffmanCodec.load(r))?;
                    DecoderState::DecodeBlock(symbol_decoder)
                }
                DecoderState::LoadDynamicHuffmanCode => {
                    let symbol_decoder = self
                        .bit_reader
                        .transaction(|r| symbol::DynamicHuffmanCodec.load(r))?;
                    DecoderState::DecodeBlock(symbol_decoder)
                }
                DecoderState::DecodeBlock(ref mut symbol_decoder) => {
                    self.block_decoder
                        .decode(&mut self.bit_reader, symbol_decoder)?;
                    read_size = self.block_decoder.read(buf)?;
                    if read_size == 0 && !buf.is_empty() && !self.eos {
                        DecoderState::ReadBlockHeader
                    } else {
                        break;
                    }
                }
            };
            self.state = next;
        }
        Ok(read_size)
    }
}

#[derive(Debug)]
enum DecoderState {
    ReadBlockHeader,
    ReadNonCompressedBlockLen,
    ReadNonCompressedBlock { len: u16 },
    LoadFixedHuffmanCode,
    LoadDynamicHuffmanCode,
    DecodeBlock(symbol::Decoder),
}

#[derive(Debug)]
struct BlockDecoder {
    lz77_decoder: lz77::Lz77Decoder,
    eob: bool,
}
impl BlockDecoder {
    pub fn new() -> Self {
        BlockDecoder {
            lz77_decoder: lz77::Lz77Decoder::new(),
            eob: false,
        }
    }
    pub fn enter_new_block(&mut self) {
        self.eob = false;
    }
    pub fn decode<R: Read>(
        &mut self,
        bit_reader: &mut TransactionalBitReader<R>,
        symbol_decoder: &mut symbol::Decoder,
    ) -> io::Result<()> {
        if self.eob {
            return Ok(());
        }
        while let Some(s) = self.decode_symbol(bit_reader, symbol_decoder)? {
            match s {
                symbol::Symbol::Code(code) => {
                    self.lz77_decoder.decode(code)?;
                }
                symbol::Symbol::EndOfBlock => {
                    self.eob = true;
                    break;
                }
            }
        }
        Ok(())
    }

    fn extend(&mut self, buf: &[u8]) {
        self.lz77_decoder.extend_from_slice(buf);
    }

    fn decode_symbol<R: Read>(
        &mut self,
        bit_reader: &mut TransactionalBitReader<R>,
        symbol_decoder: &mut symbol::Decoder,
    ) -> io::Result<Option<symbol::Symbol>> {
        let result = bit_reader.transaction(|bit_reader| {
            let s = symbol_decoder.decode_unchecked(bit_reader);
            bit_reader.check_last_error().map(|()| s)
        });
        match result {
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
            Ok(s) => Ok(Some(s)),
        }
    }
}
impl Read for BlockDecoder {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.lz77_decoder.buffer().is_empty() {
            self.lz77_decoder.read(buf)
        } else if self.eob {
            Ok(0)
        } else {
            Err(io::Error::new(io::ErrorKind::WouldBlock, "Would block"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::{EncodeOptions, Encoder};
    use crate::util::{nb_read_to_end, WouldBlockReader};
    #[cfg(feature = "no_std")]
    use core2::io::{Read, Write};
    #[cfg(not(feature = "no_std"))]
    use std::io::{Read, Write};

    #[test]
    fn it_works() {
        let mut encoder = Encoder::new(Vec::new());
        encoder.write_all(b"Hello World!".as_ref()).unwrap();
        let encoded_data = encoder.finish().into_result().unwrap();

        let mut decoder = Decoder::new(&encoded_data[..]);
        let mut decoded_data = Vec::new();
        decoder.read_to_end(&mut decoded_data).unwrap();

        assert_eq!(decoded_data, b"Hello World!");
    }

    #[test]
    fn non_blocking_io_works() {
        let mut encoder = Encoder::new(Vec::new());
        encoder.write_all(b"Hello World!".as_ref()).unwrap();
        let encoded_data = encoder.finish().into_result().unwrap();

        let decoder = Decoder::new(WouldBlockReader::new(&encoded_data[..]));
        let decoded_data = nb_read_to_end(decoder).unwrap();

        assert_eq!(decoded_data, b"Hello World!");
    }

    #[test]
    fn non_blocking_io_for_large_text_works() {
        let text: String = (0..10000)
            .into_iter()
            .map(|i| format!("test {}", i))
            .collect();

        let mut encoder = crate::deflate::Encoder::new(Vec::new());
        encoder.write_all(text.as_bytes()).unwrap();
        let encoded_data = encoder.finish().into_result().unwrap();

        let decoder = Decoder::new(WouldBlockReader::new(&encoded_data[..]));
        let decoded_data = nb_read_to_end(decoder).unwrap();
        assert_eq!(decoded_data, text.as_bytes());
    }

    #[test]
    fn non_compressed_non_blocking_io_works() {
        let mut encoder = Encoder::with_options(Vec::new(), EncodeOptions::new().no_compression());
        encoder.write_all(b"Hello World!".as_ref()).unwrap();
        let encoded_data = encoder.finish().into_result().unwrap();

        let decoder = Decoder::new(WouldBlockReader::new(&encoded_data[..]));
        let decoded_data = nb_read_to_end(decoder).unwrap();

        assert_eq!(decoded_data, b"Hello World!");
    }
}
