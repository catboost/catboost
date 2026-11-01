//! The encoder and decoder of the ZLIB format.
//!
//! The ZLIB format is defined in [RFC-1950](https://tools.ietf.org/html/rfc1950).
//!
//! # Examples
//! ```
//! #[cfg(not(feature = "no_std"))]
//! use std::io::{Read, Write};
//! #[cfg(feature = "no_std")]
//! use core2::io::{Read, Write};
//! use libflate::zlib::{Encoder, Decoder};
//!
//! // Encoding
//! let mut encoder = Encoder::new(Vec::new()).unwrap();
//! encoder.write_all(&b"Hello World!"[..]).unwrap();
//! let encoded_data = encoder.finish().into_result().unwrap();
//!
//! // Decoding
//! let mut decoder = Decoder::new(&encoded_data[..]).unwrap();
//! let mut decoded_data = Vec::new();
//! decoder.read_to_end(&mut decoded_data).unwrap();
//!
//! assert_eq!(decoded_data, b"Hello World!");
//! ```
use crate::checksum;
use crate::deflate;
use crate::finish::{Complete, Finish};
use crate::lz77;
#[cfg(feature = "no_std")]
use core2::io;
#[cfg(not(feature = "no_std"))]
use std::io;

const COMPRESSION_METHOD_DEFLATE: u8 = 8;

/// Compression levels defined by the ZLIB format.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// Compressor used fastest algorithm.
    Fastest = 0,

    /// Compressor used fast algorithm.
    Fast = 1,

    /// Compressor used default algorithm.
    Default = 2,

    /// Compressor used maximum compression, slowest algorithm.
    Slowest = 3,
}
impl CompressionLevel {
    fn from_u2(level: u8) -> Self {
        match level {
            0 => CompressionLevel::Fastest,
            1 => CompressionLevel::Fast,
            2 => CompressionLevel::Default,
            3 => CompressionLevel::Slowest,
            _ => unreachable!(),
        }
    }
    fn as_u2(&self) -> u8 {
        self.clone() as u8
    }
}
impl From<lz77::CompressionLevel> for CompressionLevel {
    fn from(f: lz77::CompressionLevel) -> Self {
        match f {
            lz77::CompressionLevel::None => CompressionLevel::Fastest,
            lz77::CompressionLevel::Fast => CompressionLevel::Fast,
            lz77::CompressionLevel::Balance => CompressionLevel::Default,
            lz77::CompressionLevel::Best => CompressionLevel::Slowest,
        }
    }
}

/// LZ77 Window sizes defined by the ZLIB format.
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
// TODO: Use `#[allow(clippy::upper_case_acronyms)]` instead once it gets available on the stable branch
#[allow(clippy::all)]
pub enum Lz77WindowSize {
    /// 256 bytes
    B256 = 0,

    /// 512 btyes
    B512 = 1,

    /// 1 kilobyte
    KB1 = 2,

    /// 2 kilobytes
    KB2 = 3,

    /// 4 kitobytes
    KB4 = 4,

    /// 8 kitobytes
    KB8 = 5,

    /// 16 kitobytes
    KB16 = 6,

    /// 32 kitobytes
    KB32 = 7,
}
impl Lz77WindowSize {
    fn from_u4(compression_info: u8) -> Option<Self> {
        match compression_info {
            0 => Some(Lz77WindowSize::B256),
            1 => Some(Lz77WindowSize::B512),
            2 => Some(Lz77WindowSize::KB1),
            3 => Some(Lz77WindowSize::KB2),
            4 => Some(Lz77WindowSize::KB4),
            5 => Some(Lz77WindowSize::KB8),
            6 => Some(Lz77WindowSize::KB16),
            7 => Some(Lz77WindowSize::KB32),
            _ => None,
        }
    }
    fn as_u4(&self) -> u8 {
        self.clone() as u8
    }

    /// Converts from `u16` to Lz77WindowSize`.
    ///
    /// Fractions are rounded to next upper window size.
    /// If `size` exceeds maximum window size,
    /// `lz77::MAX_WINDOW_SIZE` will be used instead.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::Lz77WindowSize;
    ///
    /// assert_eq!(Lz77WindowSize::from_u16(15000), Lz77WindowSize::KB16);
    /// assert_eq!(Lz77WindowSize::from_u16(16384), Lz77WindowSize::KB16);
    /// assert_eq!(Lz77WindowSize::from_u16(16385), Lz77WindowSize::KB32);
    /// assert_eq!(Lz77WindowSize::from_u16(40000), Lz77WindowSize::KB32);
    /// ```
    pub fn from_u16(size: u16) -> Self {
        use self::Lz77WindowSize::*;
        if 16_384 < size {
            KB32
        } else if 8192 < size {
            KB16
        } else if 4096 < size {
            KB8
        } else if 2048 < size {
            KB4
        } else if 1024 < size {
            KB2
        } else if 512 < size {
            KB1
        } else if 256 < size {
            B512
        } else {
            B256
        }
    }

    /// Converts from `Lz77WindowSize` to `u16`.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::Lz77WindowSize;
    ///
    /// assert_eq!(Lz77WindowSize::KB16.to_u16(), 16384u16);
    /// ```
    pub fn to_u16(&self) -> u16 {
        use self::Lz77WindowSize::*;
        match *self {
            B256 => 256,
            B512 => 512,
            KB1 => 1024,
            KB2 => 2048,
            KB4 => 4096,
            KB8 => 8192,
            KB16 => 16_384,
            KB32 => 32_768,
        }
    }
}

/// [zlib] library specific parameter for defining behavior when `Write::flush` method is called.
///
/// # References
///
/// - [Zlib Manual](https://www.zlib.net/manual.html)
/// - [Zlib Flush Modes](https://www.bolet.org/~pornin/deflate-flush.html)
///
/// [zlib]: https://www.zlib.net/
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FlushMode {
    /// `Z_NO_FLUSH` (default).
    ///
    /// Note that when this parameter is specified,
    /// no `zlib` specific processing will not be executed but ordinal DEFLATE layer flushing will be performed.
    #[default]
    None = 0,

    /// `Z_SYNC_FLUSH`.
    Sync = 2,
}

/// ZLIB header.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Header {
    window_size: Lz77WindowSize,
    compression_level: CompressionLevel,
}
impl Header {
    /// Returns the LZ77 window size stored in the header.
    pub fn window_size(&self) -> Lz77WindowSize {
        self.window_size.clone()
    }
    /// Returns the compression level stored in the header.
    pub fn compression_level(&self) -> CompressionLevel {
        self.compression_level.clone()
    }
    fn from_lz77<E>(lz77: &E) -> Self
    where
        E: lz77::Lz77Encode,
    {
        Header {
            compression_level: From::from(lz77.compression_level()),
            window_size: Lz77WindowSize::from_u16(lz77.window_size()),
        }
    }
    pub(crate) fn read_from<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut buf = [0; 2];
        reader.read_exact(&mut buf)?;
        let (cmf, flg) = (buf[0], buf[1]);
        let check = (u16::from(cmf) << 8) + u16::from(flg);
        if check % 31 != 0 {
            return Err(invalid_data_error!(
                "Inconsistent ZLIB check bits: `CMF({}) * 256 + \
                 FLG({})` must be a multiple of 31",
                cmf,
                flg
            ));
        }

        let compression_method = cmf & 0b1111;
        let compression_info = cmf >> 4;
        if compression_method != COMPRESSION_METHOD_DEFLATE {
            return Err(invalid_data_error!(
                "Compression methods other than DEFLATE(8) are \
                 unsupported: method={}",
                compression_method
            ));
        }
        let window_size = Lz77WindowSize::from_u4(compression_info).ok_or_else(|| {
            invalid_data_error!("CINFO above 7 are not allowed: value={}", compression_info)
        })?;

        let dict_flag = (flg & 0b10_0000) != 0;
        if dict_flag {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            return Err(invalid_data_error!(
                "Preset dictionaries are not supported: \
                 dictionary_id=0x{:X}",
                u32::from_be_bytes(buf)
            ));
        }
        let compression_level = CompressionLevel::from_u2(flg >> 6);
        Ok(Header {
            window_size,
            compression_level,
        })
    }
    fn write_to<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        let cmf = (self.window_size.as_u4() << 4) | COMPRESSION_METHOD_DEFLATE;
        let mut flg = self.compression_level.as_u2() << 6;
        let check = (u16::from(cmf) << 8) + u16::from(flg);
        if check % 31 != 0 {
            flg += (31 - check % 31) as u8;
        }
        writer.write_all(&[cmf, flg])?;
        Ok(())
    }
}

/// ZLIB decoder.
#[derive(Debug)]
pub struct Decoder<R> {
    header: Header,
    reader: deflate::Decoder<R>,
    adler32: checksum::Adler32,
    eos: bool,
}
impl<R> Decoder<R>
where
    R: io::Read,
{
    /// Makes a new decoder instance.
    ///
    /// `inner` is to be decoded ZLIB stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Read;
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Read;
    /// use libflate::zlib::Decoder;
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let mut decoder = Decoder::new(&encoded_data[..]).unwrap();
    /// let mut buf = Vec::new();
    /// decoder.read_to_end(&mut buf).unwrap();
    ///
    /// assert_eq!(buf, b"Hello World!");
    /// ```
    pub fn new(mut inner: R) -> io::Result<Self> {
        let header = Header::read_from(&mut inner)?;
        Ok(Decoder {
            header,
            reader: deflate::Decoder::new(inner),
            adler32: checksum::Adler32::new(),
            eos: false,
        })
    }

    /// Returns the header of the ZLIB stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::{Decoder, CompressionLevel};
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let decoder = Decoder::new(&encoded_data[..]).unwrap();
    /// assert_eq!(decoder.header().compression_level(),
    ///            CompressionLevel::Default);
    /// ```
    pub fn header(&self) -> &Header {
        &self.header
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
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Cursor;
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Cursor;
    /// use libflate::zlib::Decoder;
    ///
    /// let encoded_data = [120, 156, 243, 72, 205, 201, 201, 87, 8, 207, 47,
    ///                     202, 73, 81, 4, 0, 28, 73, 4, 62];
    ///
    /// let decoder = Decoder::new(Cursor::new(&encoded_data)).unwrap();
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data);
    /// ```
    pub fn into_inner(self) -> R {
        self.reader.into_inner()
    }

    /// Returns the data that has been decoded but has not yet been read.
    ///
    /// This method is useful to retrieve partial decoded data when the decoding process is failed.
    pub fn unread_decoded_data(&self) -> &[u8] {
        self.reader.unread_decoded_data()
    }
}
impl<R> io::Read for Decoder<R>
where
    R: io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.eos {
            Ok(0)
        } else {
            let read_size = self.reader.read(buf)?;
            if read_size == 0 {
                if buf.is_empty() {
                    return Ok(0);
                }

                self.eos = true;
                let mut buf = [0; 4];
                self.reader.as_inner_mut().read_exact(&mut buf)?;
                let adler32 = u32::from_be_bytes(buf);

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

/// Options for a ZLIB encoder.
#[derive(Debug)]
pub struct EncodeOptions<E>
where
    E: lz77::Lz77Encode,
{
    header: Header,
    options: deflate::EncodeOptions<E>,
    flush_mode: FlushMode,
}
impl Default for EncodeOptions<lz77::DefaultLz77Encoder> {
    fn default() -> Self {
        EncodeOptions {
            header: Header::from_lz77(&lz77::DefaultLz77Encoder::new()),
            options: Default::default(),
            flush_mode: FlushMode::None,
        }
    }
}
impl EncodeOptions<lz77::DefaultLz77Encoder> {
    /// Makes a default instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new();
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
}
impl<E> EncodeOptions<E>
where
    E: lz77::Lz77Encode,
{
    /// Specifies the LZ77 encoder used to compress input data.
    ///
    /// # Example
    /// ```
    /// use libflate::lz77::DefaultLz77Encoder;
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::with_lz77(DefaultLz77Encoder::new());
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn with_lz77(lz77: E) -> Self {
        EncodeOptions {
            header: Header::from_lz77(&lz77),
            options: deflate::EncodeOptions::with_lz77(lz77),
            flush_mode: FlushMode::None,
        }
    }

    /// Disables LZ77 compression.
    ///
    /// # Example
    /// ```
    /// use libflate::lz77::DefaultLz77Encoder;
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().no_compression();
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn no_compression(mut self) -> Self {
        self.options = self.options.no_compression();
        self.header.compression_level = CompressionLevel::Fastest;
        self
    }

    /// Specifies the hint of the size of a DEFLATE block.
    ///
    /// The default value is `deflate::DEFAULT_BLOCK_SIZE`.
    ///
    /// # Example
    /// ```
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().block_size(512 * 1024);
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn block_size(mut self, size: usize) -> Self {
        self.options = self.options.block_size(size);
        self
    }

    /// Specifies to compress with fixed huffman codes.
    ///
    /// # Example
    /// ```
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().fixed_huffman_codes();
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn fixed_huffman_codes(mut self) -> Self {
        self.options = self.options.fixed_huffman_codes();
        self
    }

    /// Specifies flush mode.
    pub fn flush_mode(mut self, mode: FlushMode) -> Self {
        self.flush_mode = mode;
        self
    }
}

/// ZLIB encoder.
#[derive(Debug)]
pub struct Encoder<W, E = lz77::DefaultLz77Encoder> {
    header: Header,
    flush_mode: FlushMode,
    writer: deflate::Encoder<W, E>,
    adler32: checksum::Adler32,
}
impl<W> Encoder<W, lz77::DefaultLz77Encoder>
where
    W: io::Write,
{
    /// Makes a new encoder instance.
    ///
    /// Encoded ZLIB stream is written to `inner`.
    ///
    /// # Examples
    /// ```
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// use libflate::zlib::Encoder;
    ///
    /// let mut encoder = Encoder::new(Vec::new()).unwrap();
    /// encoder.write_all(b"Hello World!").unwrap();
    ///
    /// assert_eq!(encoder.finish().into_result().unwrap(),
    ///            vec![120, 156, 5, 192, 49, 13, 0, 0, 8, 3, 65, 43, 224, 6, 7, 24, 128,
    ///                 237, 147, 38, 245, 63, 244, 230, 65, 181, 50, 215, 1, 28, 73, 4, 62]);
    /// ```
    pub fn new(inner: W) -> io::Result<Self> {
        Self::with_options(inner, EncodeOptions::default())
    }
}
impl<W, E> Encoder<W, E>
where
    W: io::Write,
    E: lz77::Lz77Encode,
{
    /// Makes a new encoder instance with specified options.
    ///
    /// Encoded ZLIB stream is written to `inner`.
    ///
    /// # Examples
    /// ```
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// use libflate::zlib::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().no_compression();
    /// let mut encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// encoder.write_all(b"Hello World!").unwrap();
    ///
    /// assert_eq!(encoder.finish().into_result().unwrap(),
    ///            [120, 1, 1, 12, 0, 243, 255, 72, 101, 108, 108, 111, 32, 87, 111,
    ///             114, 108, 100, 33, 28, 73, 4, 62]);
    /// ```
    pub fn with_options(mut inner: W, options: EncodeOptions<E>) -> io::Result<Self> {
        options.header.write_to(&mut inner)?;
        Ok(Encoder {
            header: options.header,
            flush_mode: options.flush_mode,
            writer: deflate::Encoder::with_options(inner, options.options),
            adler32: checksum::Adler32::new(),
        })
    }

    /// Returns the header of the ZLIB stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::zlib::{Encoder, Lz77WindowSize};
    ///
    /// let encoder = Encoder::new(Vec::new()).unwrap();
    /// assert_eq!(encoder.header().window_size(), Lz77WindowSize::KB32);
    /// ```
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Writes the ZLIB trailer and returns the inner stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// use libflate::zlib::Encoder;
    ///
    /// let mut encoder = Encoder::new(Vec::new()).unwrap();
    /// encoder.write_all(b"Hello World!").unwrap();
    ///
    /// assert_eq!(encoder.finish().into_result().unwrap(),
    ///            vec![120, 156, 5, 192, 49, 13, 0, 0, 8, 3, 65, 43, 224, 6, 7, 24, 128,
    ///                 237, 147, 38, 245, 63, 244, 230, 65, 181, 50, 215, 1, 28, 73, 4, 62]);
    /// ```
    ///
    /// # Note
    ///
    /// If you are not concerned the result of this encoding,
    /// it may be convenient to use `AutoFinishUnchecked` instead of the explicit invocation of this method.
    ///
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::finish::AutoFinishUnchecked;
    /// use libflate::zlib::Encoder;
    ///
    /// let plain = b"Hello World!";
    /// let mut buf = Vec::new();
    /// let mut encoder = AutoFinishUnchecked::new(Encoder::new(&mut buf).unwrap());
    /// encoder.write_all(plain.as_ref()).unwrap();
    /// ```
    pub fn finish(self) -> Finish<W, io::Error> {
        let mut inner = finish_try!(self.writer.finish());
        match inner
            .write_all(&self.adler32.value().to_be_bytes())
            .and_then(|_| inner.flush())
        {
            Ok(_) => Finish::new(inner, None),
            Err(e) => Finish::new(inner, Some(e)),
        }
    }

    /// Returns the immutable reference to the inner stream.
    pub fn as_inner_ref(&self) -> &W {
        self.writer.as_inner_ref()
    }

    /// Returns the mutable reference to the inner stream.
    pub fn as_inner_mut(&mut self) -> &mut W {
        self.writer.as_inner_mut()
    }

    /// Unwraps the `Encoder`, returning the inner stream.
    pub fn into_inner(self) -> W {
        self.writer.into_inner()
    }
}
impl<W, E> io::Write for Encoder<W, E>
where
    W: io::Write,
    E: lz77::Lz77Encode,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written_size = self.writer.write(buf)?;
        self.adler32.update(&buf[..written_size]);
        Ok(written_size)
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.flush_mode {
            FlushMode::None => self.writer.flush(),
            FlushMode::Sync => self.writer.zlib_sync_flush(),
        }
    }
}
impl<W, E> Complete for Encoder<W, E>
where
    W: io::Write,
    E: lz77::Lz77Encode,
{
    fn complete(self) -> io::Result<()> {
        self.finish().into_result().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finish::AutoFinish;
    #[cfg(feature = "no_std")]
    use core2::io::{Read as _, Write as _};
    #[cfg(not(feature = "no_std"))]
    use std::io::{Read as _, Write as _};

    fn decode_all(buf: &[u8]) -> io::Result<Vec<u8>> {
        let mut decoder = Decoder::new(buf).unwrap();
        let mut buf = Vec::with_capacity(buf.len());
        decoder.read_to_end(&mut buf)?;
        Ok(buf)
    }
    fn default_encode(buf: &[u8]) -> io::Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(buf).unwrap();
        encoder.finish().into_result()
    }
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
        let mut decoder = Decoder::new(&encoded[..]).unwrap();
        assert_eq!(
            *decoder.header(),
            Header {
                window_size: Lz77WindowSize::KB32,
                compression_level: CompressionLevel::Default,
            }
        );

        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf).unwrap();

        let expected = b"Hello World!";
        assert_eq!(buf, expected);
    }

    #[test]
    fn default_encode_works() {
        let plain = b"Hello World! Hello ZLIB!!";
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(plain.as_ref()).unwrap();
        let encoded = encoder.finish().into_result().unwrap();
        assert_eq!(decode_all(&encoded).unwrap(), plain);
    }

    #[test]
    fn best_speed_encode_works() {
        let plain = b"Hello World! Hello ZLIB!!";
        let mut encoder =
            Encoder::with_options(Vec::new(), EncodeOptions::default().fixed_huffman_codes())
                .unwrap();
        encoder.write_all(plain.as_ref()).unwrap();
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
    fn encoder_auto_finish_works() {
        let plain = b"Hello World! Hello ZLIB!!";
        let mut buf = Vec::new();
        {
            let mut encoder = AutoFinish::new(Encoder::new(&mut buf).unwrap());
            encoder.write_all(plain.as_ref()).unwrap();
        }
        assert_eq!(decode_all(&buf).unwrap(), plain);
    }

    #[test]
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

    #[test]
    fn test_issues_16() {
        // See: https://github.com/sile/libflate/issues/16

        let encoded =
            include_bytes!("../data/issues_16/crash-1bb6d408475a5bd57247ee40f290830adfe2086e");
        assert_eq!(
            &decode_all(&encoded[..])
                .err()
                .map(|e| e.to_string())
                .unwrap()[..31],
            "The value of HDIST is too big: max=30, actual=32"[..31]
                .to_owned()
                .as_str()
        );

        let encoded =
            include_bytes!("../data/issues_16/crash-369e8509a0e76356f4549c292ceedee429cfe125");
        assert_eq!(
            &decode_all(&encoded[..])
                .err()
                .map(|e| e.to_string())
                .unwrap()[..31],
            "The value of HDIST is too big: max=30, actual=32"[..31]
                .to_owned()
                .as_str()
        );

        let encoded =
            include_bytes!("../data/issues_16/crash-e75959d935650306881140df7f6d1d73e33425cb");
        assert_eq!(
            &decode_all(&encoded[..])
                .err()
                .map(|e| e.to_string())
                .unwrap()[..31],
            "The value of HDIST is too big: max=30, actual=32"[..31]
                .to_owned()
                .as_str()
        );
    }

    #[test]
    fn test_issues_27() {
        // See: https://github.com/sile/libflate/issues/27

        let writes = ["fooooooooooooooooo", "bar", "baz"];

        // FlushMode::None
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        for _ in 0..2 {
            for string in &writes {
                encoder.write(string.as_bytes()).expect("Write failed");
            }
            encoder.flush().expect("Flush failed");
        }
        let finished = encoder.finish().unwrap();
        let expected = vec![
            120, 156, // header
            92, 192, 161, 17, 0, 0, 0, 1, 192, 89, 9, 170, 59, 209, 244, 186, 151, 31, 17, 162,
            227, 2, 14, 141, 0, 0, 0, 8, 0, 206, 74, 80, 221, 137, 166, 215, 189, 252, 136, 16, 93,
            1, 112, 32, 0, 0, 0, 0, 0, 228, 255, 26, 246, 95, 20, 111,
        ];
        assert_eq!(finished.0, expected);

        let mut output = Vec::new();
        Decoder::new(&finished.0[..])
            .unwrap()
            .read_to_end(&mut output)
            .unwrap();
        assert_eq!(
            output,
            "fooooooooooooooooobarbazfooooooooooooooooobarbaz".as_bytes()
        );

        // FlushMode::Sync
        let mut encoder =
            Encoder::with_options(Vec::new(), EncodeOptions::new().flush_mode(FlushMode::Sync))
                .unwrap();
        for _ in 0..2 {
            for string in &writes {
                encoder.write(string.as_bytes()).expect("Write failed");
            }
            encoder.flush().expect("Flush failed");
        }
        let finished = encoder.finish().unwrap();
        let expected = vec![
            120, 156, // header
            92, 192, 161, 17, 0, 0, 0, 1, 192, 89, 9, 170, 59, 209, 244, 186, 151, 31, 17, 162, 3,
            0, 0, 255, 255, // sync bytes
            92, 192, 161, 17, 0, 0, 0, 1, 192, 89, 9, 170, 59, 209, 244, 186, 151, 31, 17, 162, 3,
            0, 0, 255, 255, // sync bytes
            5, 192, 129, 0, 0, 0, 0, 0, 144, 255, 107, 0, 246, 95, 20, 111,
        ];
        assert_eq!(finished.0, expected);

        let mut output = Vec::new();
        Decoder::new(&finished.0[..])
            .unwrap()
            .read_to_end(&mut output)
            .unwrap();
        assert_eq!(
            output,
            "fooooooooooooooooobarbazfooooooooooooooooobarbaz".as_bytes()
        );
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    /// See: https://github.com/sile/libflate/issues/61
    fn issue_61() {
        let data = default_encode(b"Hello World").unwrap();
        let mut decoder = Decoder::new(&data[..]).unwrap();
        let mut buf = Vec::new();
        decoder.read(&mut buf).unwrap();
        decoder.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"Hello World");
    }

    #[test]
    fn issue71() {
        let encoded_data = [
            120, 218, 251, 255, 207, 144, 193, 138, 193, 151, 161, 146, 33, 143, 33, 149, 161, 156,
            161, 24, 72, 38, 51, 148, 48, 100, 50, 228, 3, 69, 120, 25, 184, 24,
        ];

        let mut decoder = Decoder::new(&encoded_data[..]).unwrap();
        let mut buf = Vec::new();
        let result = decoder.read_to_end(&mut buf);
        assert!(result.is_err());

        let decoded_data = [
            255, 254, 49, 0, 58, 0, 77, 0, 121, 0, 110, 0, 101, 0, 119, 0, 115, 0, 101, 0, 99, 0,
            116, 0, 105, 0, 111, 0, 110, 0, 13, 0, 10,
        ];
        assert_eq!(decoder.unread_decoded_data(), decoded_data);
    }
}
