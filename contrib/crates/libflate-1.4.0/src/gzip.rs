//! The encoder and decoder of the GZIP format.
//!
//! The GZIP format is defined in [RFC-1952](https://tools.ietf.org/html/rfc1952).
//!
//! # Examples
//! ```
//! #[cfg(feature = "no_std")]
//! use core2::io::{Read, Write};
//! #[cfg(not(feature = "no_std"))]
//! use std::io::{Read, Write};
//! use libflate::gzip::{Encoder, Decoder};
//!
//! // Encoding
//! let mut encoder = Encoder::new(Vec::new()).unwrap();
//! encoder.write_all(b"Hello World!".as_ref()).unwrap();
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
use alloc::{ffi::CString, vec::Vec};
#[cfg(feature = "no_std")]
use core2::io;
#[cfg(not(feature = "no_std"))]
use std::{ffi::CString, io, time};

const GZIP_ID: [u8; 2] = [31, 139];
const COMPRESSION_METHOD_DEFLATE: u8 = 8;

const OS_FAT: u8 = 0;
const OS_AMIGA: u8 = 1;
const OS_VMS: u8 = 2;
const OS_UNIX: u8 = 3;
const OS_VM_CMS: u8 = 4;
const OS_ATARI_TOS: u8 = 5;
const OS_HPFS: u8 = 6;
const OS_MACINTOSH: u8 = 7;
const OS_Z_SYSTEM: u8 = 8;
const OS_CPM: u8 = 9;
const OS_TOPS20: u8 = 10;
const OS_NTFS: u8 = 11;
const OS_QDOS: u8 = 12;
const OS_ACORN_RISCOS: u8 = 13;
const OS_UNKNOWN: u8 = 255;

const F_TEXT: u8 = 0b00_0001;
const F_HCRC: u8 = 0b00_0010;
const F_EXTRA: u8 = 0b00_0100;
const F_NAME: u8 = 0b00_1000;
const F_COMMENT: u8 = 0b01_0000;

/// Compression levels defined by the GZIP format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionLevel {
    /// Compressor used fastest algorithm.
    Fastest,

    /// Compressor used maximum compression, slowest algorithm.
    Slowest,

    /// No information about compression method.
    Unknown,
}
impl CompressionLevel {
    fn to_u8(&self) -> u8 {
        match *self {
            CompressionLevel::Fastest => 4,
            CompressionLevel::Slowest => 2,
            CompressionLevel::Unknown => 0,
        }
    }
    fn from_u8(x: u8) -> Self {
        match x {
            4 => CompressionLevel::Fastest,
            2 => CompressionLevel::Slowest,
            _ => CompressionLevel::Unknown,
        }
    }
}
impl From<lz77::CompressionLevel> for CompressionLevel {
    fn from(f: lz77::CompressionLevel) -> Self {
        match f {
            lz77::CompressionLevel::Fast => CompressionLevel::Fastest,
            lz77::CompressionLevel::Best => CompressionLevel::Slowest,
            _ => CompressionLevel::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Trailer {
    crc32: u32,
    input_size: u32,
}
impl Trailer {
    pub fn crc32(&self) -> u32 {
        self.crc32
    }
    pub fn read_from<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        let crc32 = u32::from_le_bytes(buf);
        reader.read_exact(&mut buf)?;
        let input_size = u32::from_le_bytes(buf);
        Ok(Trailer { crc32, input_size })
    }
    fn write_to<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        writer.write_all(&self.crc32.to_le_bytes())?;
        writer.write_all(&self.input_size.to_le_bytes())?;
        Ok(())
    }
}

/// GZIP header builder.
#[derive(Debug, Clone)]
pub struct HeaderBuilder {
    header: Header,
}
impl HeaderBuilder {
    /// Makes a new builder instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{HeaderBuilder, CompressionLevel, Os};
    ///
    /// let header = HeaderBuilder::new().finish();
    /// assert_eq!(header.compression_level(), CompressionLevel::Unknown);
    /// assert_eq!(header.os(), Os::Unix);
    /// assert_eq!(header.is_text(), false);
    /// assert_eq!(header.is_verified(), false);
    /// assert_eq!(header.extra_field(), None);
    /// assert_eq!(header.filename(), None);
    /// assert_eq!(header.comment(), None);
    /// ```
    pub fn new() -> Self {
        // wasm-unknown-unknown does not implement the time module
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "no_std")))]
        let modification_time = time::UNIX_EPOCH
            .elapsed()
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        #[cfg(any(target_arch = "wasm32", feature = "no_std"))]
        let modification_time = 0;

        let header = Header {
            modification_time,
            compression_level: CompressionLevel::Unknown,
            os: Os::Unix,
            is_text: false,
            is_verified: false,
            extra_field: None,
            filename: None,
            comment: None,
        };
        HeaderBuilder { header }
    }

    /// Sets the modification time (UNIX timestamp).
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::HeaderBuilder;
    ///
    /// let header = HeaderBuilder::new().modification_time(10).finish();
    /// assert_eq!(header.modification_time(), 10);
    /// ```
    pub fn modification_time(&mut self, modification_time: u32) -> &mut Self {
        self.header.modification_time = modification_time;
        self
    }

    /// Sets the OS type.
    ///
    /// ```
    /// use libflate::gzip::{HeaderBuilder, Os};
    ///
    /// let header = HeaderBuilder::new().os(Os::Ntfs).finish();
    /// assert_eq!(header.os(), Os::Ntfs);
    /// ```
    pub fn os(&mut self, os: Os) -> &mut Self {
        self.header.os = os;
        self
    }

    /// Indicates the encoding data is a ASCII text.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::HeaderBuilder;
    ///
    /// let header = HeaderBuilder::new().text().finish();
    /// assert_eq!(header.is_text(), true);
    /// ```
    pub fn text(&mut self) -> &mut Self {
        self.header.is_text = true;
        self
    }

    /// Specifies toe verify header bytes using CRC-16.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::HeaderBuilder;
    ///
    /// let header = HeaderBuilder::new().verify().finish();
    /// assert_eq!(header.is_verified(), true);
    /// ```
    pub fn verify(&mut self) -> &mut Self {
        self.header.is_verified = true;
        self
    }

    /// Sets the extra field.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{HeaderBuilder, ExtraField, ExtraSubField};
    ///
    /// let subfield = ExtraSubField{id: [0, 1], data: vec![2, 3, 4]};
    /// let extra = ExtraField{subfields: vec![subfield]};
    /// let header = HeaderBuilder::new().extra_field(extra.clone()).finish();
    /// assert_eq!(header.extra_field(), Some(&extra));
    /// ```
    pub fn extra_field(&mut self, extra: ExtraField) -> &mut Self {
        self.header.extra_field = Some(extra);
        self
    }

    /// Sets the file name.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// extern crate alloc;
    /// #[cfg(feature = "no_std")]
    /// use alloc::ffi::CString;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::ffi::CString;
    /// use libflate::gzip::HeaderBuilder;
    ///
    /// let header = HeaderBuilder::new().filename(CString::new("foo").unwrap()).finish();
    /// assert_eq!(header.filename(), Some(&CString::new("foo").unwrap()));
    /// ```
    pub fn filename(&mut self, filename: CString) -> &mut Self {
        self.header.filename = Some(filename);
        self
    }

    /// Sets the comment.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// extern crate alloc;
    /// #[cfg(feature = "no_std")]
    /// use alloc::ffi::CString;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::ffi::CString;
    /// use libflate::gzip::HeaderBuilder;
    ///
    /// let header = HeaderBuilder::new().comment(CString::new("foo").unwrap()).finish();
    /// assert_eq!(header.comment(), Some(&CString::new("foo").unwrap()));
    /// ```
    pub fn comment(&mut self, comment: CString) -> &mut Self {
        self.header.comment = Some(comment);
        self
    }

    /// Returns the result header.
    pub fn finish(&self) -> Header {
        self.header.clone()
    }
}
impl Default for HeaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// GZIP Header.
#[derive(Debug, Clone)]
pub struct Header {
    modification_time: u32,
    compression_level: CompressionLevel,
    os: Os,
    is_text: bool,
    is_verified: bool,
    extra_field: Option<ExtraField>,
    filename: Option<CString>,
    comment: Option<CString>,
}
impl Header {
    /// Returns the modification time (UNIX timestamp).
    pub fn modification_time(&self) -> u32 {
        self.modification_time
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> CompressionLevel {
        self.compression_level.clone()
    }

    /// Returns the OS type.
    pub fn os(&self) -> Os {
        self.os.clone()
    }

    /// Returns `true` if the stream is probably ASCII text, `false` otherwise.
    pub fn is_text(&self) -> bool {
        self.is_text
    }

    /// Returns `true` if the header bytes is verified by CRC-16, `false` otherwise.
    pub fn is_verified(&self) -> bool {
        self.is_verified
    }

    /// Returns the extra field.
    pub fn extra_field(&self) -> Option<&ExtraField> {
        self.extra_field.as_ref()
    }

    /// Returns the file name.
    pub fn filename(&self) -> Option<&CString> {
        self.filename.as_ref()
    }

    /// Returns the comment.
    pub fn comment(&self) -> Option<&CString> {
        self.comment.as_ref()
    }

    fn flags(&self) -> u8 {
        [
            (F_TEXT, self.is_text),
            (F_HCRC, self.is_verified),
            (F_EXTRA, self.extra_field.is_some()),
            (F_NAME, self.filename.is_some()),
            (F_COMMENT, self.comment.is_some()),
        ]
        .iter()
        .filter(|e| e.1)
        .map(|e| e.0)
        .sum()
    }
    fn crc16(&self) -> u16 {
        let mut crc = checksum::Crc32::new();
        let mut buf = Vec::new();
        Header {
            is_verified: false,
            ..self.clone()
        }
        .write_to(&mut buf)
        .unwrap();
        crc.update(&buf);
        crc.value() as u16
    }
    fn write_to<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        writer.write_all(&GZIP_ID)?;
        writer.write_all(&[COMPRESSION_METHOD_DEFLATE, self.flags()])?;
        writer.write_all(&self.modification_time.to_le_bytes())?;
        writer.write_all(&[self.compression_level.to_u8(), self.os.to_u8()])?;
        if let Some(ref x) = self.extra_field {
            x.write_to(&mut writer)?;
        }
        if let Some(ref x) = self.filename {
            writer.write_all(x.as_bytes_with_nul())?;
        }
        if let Some(ref x) = self.comment {
            writer.write_all(x.as_bytes_with_nul())?;
        }
        if self.is_verified {
            writer.write_all(&self.crc16().to_le_bytes())?;
        }
        Ok(())
    }
    pub(crate) fn read_from<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut this = HeaderBuilder::new().finish();
        let mut buf = [0; 2 + 1 + 1 + 4 + 1 + 1];
        reader.read_exact(&mut buf)?;
        let id = &buf[0..2];
        if id != GZIP_ID {
            return Err(invalid_data_error!(
                "Unexpected GZIP ID: value={:?}, \
                 expected={:?}",
                id,
                GZIP_ID
            ));
        }
        let compression_method = buf[2];
        if compression_method != COMPRESSION_METHOD_DEFLATE {
            return Err(invalid_data_error!(
                "Compression methods other than DEFLATE(8) are \
                 unsupported: method={}",
                compression_method
            ));
        }
        let flags = buf[3];
        this.modification_time = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        this.compression_level = CompressionLevel::from_u8(buf[8]);
        this.os = Os::from_u8(buf[9]);
        if flags & F_EXTRA != 0 {
            this.extra_field = Some(ExtraField::read_from(&mut reader)?);
        }
        if flags & F_NAME != 0 {
            this.filename = Some(read_cstring(&mut reader)?);
        }
        if flags & F_COMMENT != 0 {
            this.comment = Some(read_cstring(&mut reader)?);
        }
        // Checksum verification is skipped during fuzzing
        // so that random data from fuzzer can reach actually interesting code.
        // Compilation flag 'fuzzing' is automatically set by all 3 Rust fuzzers.
        if flags & F_HCRC != 0 && cfg!(not(fuzzing)) {
            let mut buf = [0; 2];
            reader.read_exact(&mut buf)?;
            let crc = u16::from_le_bytes(buf);
            let expected = this.crc16();
            if crc != expected {
                return Err(invalid_data_error!(
                    "CRC16 of GZIP header mismatched: value={}, \
                     expected={}",
                    crc,
                    expected
                ));
            }
            this.is_verified = true;
        }
        Ok(this)
    }
}

fn read_cstring<R>(mut reader: R) -> io::Result<CString>
where
    R: io::Read,
{
    let mut buf = Vec::new();
    loop {
        let mut cbuf = [0; 1];
        reader.read_exact(&mut cbuf)?;
        if cbuf[0] == 0 {
            return Ok(CString::new(buf).unwrap());
        }
        buf.push(cbuf[0]);
    }
}

/// Extra field of a GZIP header.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraField {
    /// Data of the extra field.
    pub subfields: Vec<ExtraSubField>,
}
impl ExtraField {
    fn read_from<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut subfields = Vec::new();
        let mut buf = [0; 2];
        reader.read_exact(&mut buf)?;
        let data_size = u16::from_le_bytes(buf) as usize;
        let mut reader = reader.take(data_size as u64);
        while reader.limit() > 0 {
            subfields.push(ExtraSubField::read_from(&mut reader)?);
        }
        Ok(ExtraField { subfields })
    }
    fn write_to<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        let len = self.subfields.iter().map(|f| f.write_len()).sum::<usize>();
        if len > 0xFFFF {
            return Err(invalid_data_error!("extra field too long: {}", len));
        }
        writer.write_all(&(len as u16).to_le_bytes())?;
        for f in &self.subfields {
            f.write_to(&mut writer)?;
        }
        Ok(())
    }
}

/// A sub field in the extra field of a GZIP header.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtraSubField {
    /// ID of the field.
    pub id: [u8; 2],

    /// Data of the field.
    pub data: Vec<u8>,
}
impl ExtraSubField {
    fn read_from<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut field = ExtraSubField {
            id: [0; 2],
            data: Vec::new(),
        };

        reader.read_exact(&mut field.id)?;
        let mut buf = [0; 2];
        reader.read_exact(&mut buf)?;
        let data_size = u16::from_le_bytes(buf) as usize;
        field.data.resize(data_size, 0);
        reader.read_exact(&mut field.data)?;

        Ok(field)
    }
    fn write_to<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        writer.write_all(&self.id)?;
        writer.write_all(&(self.data.len() as u16).to_le_bytes())?;
        writer.write_all(&self.data)?;
        Ok(())
    }
    fn write_len(&self) -> usize {
        4 + self.data.len()
    }
}

/// OS type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Os {
    /// FAT filesystem (MS-DOS, OS/2, NT/Win32)
    Fat,

    /// Amiga
    Amiga,

    /// VMS (or OpenVMS)
    Vms,

    /// Unix
    Unix,

    /// VM/CMS
    VmCms,

    /// Atari TOS
    AtariTos,

    /// HPFS filesystem (OS/2, NT)
    Hpfs,

    /// Macintosh
    Macintosh,

    /// Z-System
    ZSystem,

    /// CP/M
    CpM,

    /// TOPS-20
    Tops20,

    /// NTFS filesystem (NT)
    Ntfs,

    /// QDOS
    Qdos,

    /// Acorn RISCOS
    AcornRiscos,

    /// Unknown
    Unknown,

    /// Undefined value in RFC-1952
    Undefined(u8),
}
impl Os {
    fn to_u8(&self) -> u8 {
        match *self {
            Os::Fat => OS_FAT,
            Os::Amiga => OS_AMIGA,
            Os::Vms => OS_VMS,
            Os::Unix => OS_UNIX,
            Os::VmCms => OS_VM_CMS,
            Os::AtariTos => OS_ATARI_TOS,
            Os::Hpfs => OS_HPFS,
            Os::Macintosh => OS_MACINTOSH,
            Os::ZSystem => OS_Z_SYSTEM,
            Os::CpM => OS_CPM,
            Os::Tops20 => OS_TOPS20,
            Os::Ntfs => OS_NTFS,
            Os::Qdos => OS_QDOS,
            Os::AcornRiscos => OS_ACORN_RISCOS,
            Os::Unknown => OS_UNKNOWN,
            Os::Undefined(os) => os,
        }
    }
    fn from_u8(x: u8) -> Self {
        match x {
            OS_FAT => Os::Fat,
            OS_AMIGA => Os::Amiga,
            OS_VMS => Os::Vms,
            OS_UNIX => Os::Unix,
            OS_VM_CMS => Os::VmCms,
            OS_ATARI_TOS => Os::AtariTos,
            OS_HPFS => Os::Hpfs,
            OS_MACINTOSH => Os::Macintosh,
            OS_Z_SYSTEM => Os::ZSystem,
            OS_CPM => Os::CpM,
            OS_TOPS20 => Os::Tops20,
            OS_NTFS => Os::Ntfs,
            OS_QDOS => Os::Qdos,
            OS_ACORN_RISCOS => Os::AcornRiscos,
            OS_UNKNOWN => Os::Unknown,
            os => Os::Undefined(os),
        }
    }
}

/// Options for a GZIP encoder.
#[derive(Debug)]
pub struct EncodeOptions<E>
where
    E: lz77::Lz77Encode,
{
    header: Header,
    options: deflate::EncodeOptions<E>,
}
impl Default for EncodeOptions<lz77::DefaultLz77Encoder> {
    fn default() -> Self {
        EncodeOptions {
            header: HeaderBuilder::new().finish(),
            options: Default::default(),
        }
    }
}
impl EncodeOptions<lz77::DefaultLz77Encoder> {
    /// Makes a default instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{Encoder, EncodeOptions};
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
    /// use libflate::gzip::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::with_lz77(DefaultLz77Encoder::new());
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn with_lz77(lz77: E) -> Self {
        let mut header = HeaderBuilder::new().finish();
        header.compression_level = From::from(lz77.compression_level());
        EncodeOptions {
            header,
            options: deflate::EncodeOptions::with_lz77(lz77),
        }
    }

    /// Disables LZ77 compression.
    ///
    /// # Example
    /// ```
    /// use libflate::lz77::DefaultLz77Encoder;
    /// use libflate::gzip::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().no_compression();
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn no_compression(mut self) -> Self {
        self.options = self.options.no_compression();
        self.header.compression_level = CompressionLevel::Unknown;
        self
    }

    /// Sets the GZIP header which will be written to the output stream.
    ///
    /// # Example
    /// ```
    /// use libflate::gzip::{Encoder, EncodeOptions, HeaderBuilder};
    ///
    /// let header = HeaderBuilder::new().text().modification_time(100).finish();
    /// let options = EncodeOptions::new().header(header);
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn header(mut self, header: Header) -> Self {
        self.header = header;
        self
    }

    /// Specifies the hint of the size of a DEFLATE block.
    ///
    /// The default value is `deflate::DEFAULT_BLOCK_SIZE`.
    ///
    /// # Example
    /// ```
    /// use libflate::gzip::{Encoder, EncodeOptions};
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
    /// use libflate::gzip::{Encoder, EncodeOptions};
    ///
    /// let options = EncodeOptions::new().fixed_huffman_codes();
    /// let encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// ```
    pub fn fixed_huffman_codes(mut self) -> Self {
        self.options = self.options.fixed_huffman_codes();
        self
    }
}

/// GZIP encoder.
pub struct Encoder<W, E = lz77::DefaultLz77Encoder> {
    header: Header,
    crc32: checksum::Crc32,
    input_size: u32,
    writer: deflate::Encoder<W, E>,
}
impl<W> Encoder<W, lz77::DefaultLz77Encoder>
where
    W: io::Write,
{
    /// Makes a new encoder instance.
    ///
    /// Encoded GZIP stream is written to `inner`.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::gzip::Encoder;
    ///
    /// let mut encoder = Encoder::new(Vec::new()).unwrap();
    /// encoder.write_all(&b"Hello World!"[..]).unwrap();
    /// encoder.finish().into_result().unwrap();
    /// ```
    pub fn new(inner: W) -> io::Result<Self> {
        Self::with_options(inner, EncodeOptions::new())
    }
}
impl<W, E> Encoder<W, E>
where
    W: io::Write,
    E: lz77::Lz77Encode,
{
    /// Makes a new encoder instance with specified options.
    ///
    /// Encoded GZIP stream is written to `inner`.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::gzip::{Encoder, EncodeOptions, HeaderBuilder};
    ///
    /// let header = HeaderBuilder::new().modification_time(123).finish();
    /// let options = EncodeOptions::new().no_compression().header(header);
    /// let mut encoder = Encoder::with_options(Vec::new(), options).unwrap();
    /// encoder.write_all(&b"Hello World!"[..]).unwrap();
    ///
    /// assert_eq!(encoder.finish().into_result().unwrap(),
    ///            &[31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255, 72, 101, 108, 108,
    ///              111, 32, 87, 111, 114, 108, 100, 33, 163, 28, 41, 28, 12, 0, 0, 0][..]);
    /// ```
    pub fn with_options(mut inner: W, options: EncodeOptions<E>) -> io::Result<Self> {
        options.header.write_to(&mut inner)?;
        Ok(Encoder {
            header: options.header.clone(),
            crc32: checksum::Crc32::new(),
            input_size: 0,
            writer: deflate::Encoder::with_options(inner, options.options),
        })
    }

    /// Returns the header of the GZIP stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{Encoder, Os};
    ///
    /// let encoder = Encoder::new(Vec::new()).unwrap();
    /// assert_eq!(encoder.header().os(), Os::Unix);
    /// ```
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Writes the GZIP trailer and returns the inner stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::gzip::Encoder;
    ///
    /// let mut encoder = Encoder::new(Vec::new()).unwrap();
    /// encoder.write_all(&b"Hello World!"[..]).unwrap();
    ///
    /// assert!(encoder.finish().as_result().is_ok())
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
    /// use std::io;
    /// use libflate::finish::AutoFinishUnchecked;
    /// use libflate::gzip::Encoder;
    ///
    /// let plain = b"Hello World!";
    /// let mut buf = Vec::new();
    /// let mut encoder = AutoFinishUnchecked::new(Encoder::new(&mut buf).unwrap());
    /// #[cfg(feature = "no_std")]
    /// encoder.write_all(plain.as_ref()).unwrap();
    /// #[cfg(not(feature = "no_std"))]
    /// io::copy(&mut &plain[..], &mut encoder).unwrap();
    /// ```
    pub fn finish(self) -> Finish<W, io::Error> {
        let trailer = Trailer {
            crc32: self.crc32.value(),
            input_size: self.input_size,
        };
        let mut inner = finish_try!(self.writer.finish());
        match trailer.write_to(&mut inner).and_then(|_| inner.flush()) {
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
        self.crc32.update(&buf[..written_size]);
        self.input_size = self.input_size.wrapping_add(written_size as u32);
        Ok(written_size)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
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

/// GZIP decoder.
#[derive(Debug)]
pub struct Decoder<R> {
    header: Header,
    reader: deflate::Decoder<R>,
    crc32: checksum::Crc32,
    eos: bool,
}
impl<R> Decoder<R>
where
    R: io::Read,
{
    /// Makes a new decoder instance.
    ///
    /// `inner` is to be decoded GZIP stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Read;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Read;
    /// use libflate::gzip::Decoder;
    ///
    /// let encoded_data = [31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255,
    ///                     72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ///                     163, 28, 41, 28, 12, 0, 0, 0];
    ///
    /// let mut decoder = Decoder::new(&encoded_data[..]).unwrap();
    /// let mut buf = Vec::new();
    /// decoder.read_to_end(&mut buf).unwrap();
    ///
    /// assert_eq!(buf, b"Hello World!");
    /// ```
    pub fn new(mut inner: R) -> io::Result<Self> {
        let header = Header::read_from(&mut inner)?;
        Ok(Self::with_header(inner, header))
    }

    /// Returns the header of the GZIP stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{Decoder, Os};
    ///
    /// let encoded_data = [31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255,
    ///                     72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ///                     163, 28, 41, 28, 12, 0, 0, 0];
    ///
    /// let decoder = Decoder::new(&encoded_data[..]).unwrap();
    /// assert_eq!(decoder.header().os(), Os::Unix);
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
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Cursor;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Cursor;
    /// use libflate::gzip::Decoder;
    ///
    /// let encoded_data = [31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255,
    ///                     72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ///                     163, 28, 41, 28, 12, 0, 0, 0];
    ///
    /// let decoder = Decoder::new(Cursor::new(&encoded_data[..])).unwrap();
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data[..]);
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

    fn with_header(inner: R, header: Header) -> Self {
        Decoder {
            header,
            reader: deflate::Decoder::new(inner),
            crc32: checksum::Crc32::new(),
            eos: false,
        }
    }

    fn reset(&mut self, header: Header) {
        self.header = header;
        self.reader.reset();
        self.crc32 = checksum::Crc32::new();
        self.eos = false;
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
            self.crc32.update(&buf[..read_size]);
            if read_size == 0 {
                if buf.is_empty() {
                    return Ok(0);
                }

                self.eos = true;
                let trailer = Trailer::read_from(self.reader.as_inner_mut())?;
                // checksum verification is skipped during fuzzing
                // so that random data from fuzzer can reach actually interesting code
                // Compilation flag 'fuzzing' is automatically set by all 3 Rust fuzzers.
                if cfg!(not(fuzzing)) && trailer.crc32 != self.crc32.value() {
                    Err(invalid_data_error!(
                        "CRC32 mismatched: value={}, expected={}",
                        self.crc32.value(),
                        trailer.crc32
                    ))
                } else {
                    Ok(0)
                }
            } else {
                Ok(read_size)
            }
        }
    }
}

/// A decoder that decodes all members in a GZIP stream.
#[derive(Debug)]
pub struct MultiDecoder<R> {
    decoder: Decoder<R>,
    eos: bool,
}
impl<R> MultiDecoder<R>
where
    R: io::Read,
{
    /// Makes a new decoder instance.
    ///
    /// `inner` is to be decoded GZIP stream.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Read;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Read;
    /// use libflate::gzip::MultiDecoder;
    ///
    /// let mut encoded_data = Vec::new();
    ///
    /// // Add a member (a GZIP binary that represents "Hello ")
    /// encoded_data.extend(&[31, 139, 8, 0, 51, 206, 75, 90, 0, 3, 5, 128, 49, 9, 0, 0, 0, 194, 170, 24,
    ///                       199, 34, 126, 3, 251, 127, 163, 131, 71, 192, 252, 45, 234, 6, 0, 0, 0][..]);
    ///
    /// // Add another member (a GZIP binary that represents "World!")
    /// encoded_data.extend(&[31, 139, 8, 0, 227, 207, 75, 90, 0, 3, 5, 128, 49, 9, 0, 0, 0, 194, 178, 152,
    ///                       202, 2, 158, 130, 96, 255, 99, 120, 111, 4, 222, 157, 40, 118, 6, 0, 0, 0][..]);
    ///
    /// let mut decoder = MultiDecoder::new(&encoded_data[..]).unwrap();
    /// let mut buf = Vec::new();
    /// decoder.read_to_end(&mut buf).unwrap();
    ///
    /// assert_eq!(buf, b"Hello World!");
    /// ```
    pub fn new(inner: R) -> io::Result<Self> {
        let decoder = Decoder::new(inner)?;
        Ok(MultiDecoder {
            decoder,
            eos: false,
        })
    }

    /// Returns the header of the current member in the GZIP stream.
    ///
    /// # Examples
    /// ```
    /// use libflate::gzip::{MultiDecoder, Os};
    ///
    /// let encoded_data = [31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255,
    ///                     72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ///                     163, 28, 41, 28, 12, 0, 0, 0];
    ///
    /// let decoder = MultiDecoder::new(&encoded_data[..]).unwrap();
    /// assert_eq!(decoder.header().os(), Os::Unix);
    /// ```
    pub fn header(&self) -> &Header {
        self.decoder.header()
    }

    /// Returns the immutable reference to the inner stream.
    pub fn as_inner_ref(&self) -> &R {
        self.decoder.as_inner_ref()
    }

    /// Returns the mutable reference to the inner stream.
    pub fn as_inner_mut(&mut self) -> &mut R {
        self.decoder.as_inner_mut()
    }

    /// Unwraps this `MultiDecoder`, returning the underlying reader.
    ///
    /// # Examples
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Cursor;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Cursor;
    /// use libflate::gzip::MultiDecoder;
    ///
    /// let encoded_data = [31, 139, 8, 0, 123, 0, 0, 0, 0, 3, 1, 12, 0, 243, 255,
    ///                     72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33,
    ///                     163, 28, 41, 28, 12, 0, 0, 0];
    ///
    /// let decoder = MultiDecoder::new(Cursor::new(&encoded_data[..])).unwrap();
    /// assert_eq!(decoder.into_inner().into_inner(), &encoded_data[..]);
    /// ```
    pub fn into_inner(self) -> R {
        self.decoder.into_inner()
    }
}
impl<R> io::Read for MultiDecoder<R>
where
    R: io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.eos {
            return Ok(0);
        }

        let read_size = self.decoder.read(buf)?;
        if read_size == 0 {
            match Header::read_from(self.as_inner_mut()) {
                Err(e) => {
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        self.eos = true;
                        Ok(0)
                    } else {
                        Err(e)
                    }
                }
                Ok(header) => {
                    self.decoder.reset(header);
                    self.read(buf)
                }
            }
        } else {
            Ok(read_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finish::AutoFinish;
    #[cfg(feature = "no_std")]
    use core2::io::{Read, Write};
    #[cfg(not(feature = "no_std"))]
    use std::io::{self, Read, Write};

    fn decode(buf: &[u8]) -> io::Result<Vec<u8>> {
        let mut decoder = Decoder::new(buf).unwrap();
        let mut buf = Vec::with_capacity(buf.len());
        decoder.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn decode_multi(buf: &[u8]) -> io::Result<Vec<u8>> {
        let mut decoder = MultiDecoder::new(buf).unwrap();
        let mut buf = Vec::with_capacity(buf.len());
        decoder.read_to_end(&mut buf).unwrap();
        Ok(buf)
    }

    fn encode(text: &[u8]) -> io::Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(text).unwrap();
        encoder.finish().into_result()
    }

    #[test]
    fn encode_works() {
        let plain = b"Hello World! Hello GZIP!!";
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(plain.as_ref()).unwrap();
        let encoded = encoder.finish().into_result().unwrap();
        assert_eq!(decode(&encoded).unwrap(), plain);
    }

    #[test]
    fn encoder_auto_finish_works() {
        let plain = b"Hello World! Hello GZIP!!";
        let mut buf = Vec::new();
        {
            let mut encoder = AutoFinish::new(Encoder::new(&mut buf).unwrap());
            encoder.write_all(plain.as_ref()).unwrap();
        }
        assert_eq!(decode(&buf).unwrap(), plain);
    }

    #[test]
    fn multi_decode_works() {
        #[cfg(feature = "no_std")]
        use core::iter;
        #[cfg(not(feature = "no_std"))]
        use std::iter;
        let text = b"Hello World!";
        let encoded: Vec<u8> = iter::repeat(encode(text).unwrap())
            .take(2)
            .flat_map(|b| b)
            .collect();
        assert_eq!(decode(&encoded).unwrap(), b"Hello World!");
        assert_eq!(decode_multi(&encoded).unwrap(), b"Hello World!Hello World!");
    }

    #[test]
    /// See: https://github.com/sile/libflate/issues/15 and https://github.com/RazrFalcon/usvg/issues/20
    fn issue_15_1() {
        let data = b"\x1F\x8B\x08\xC1\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x7B\x80\x80\x80\x80\x7B\x7B\x7B\x7B\x7B\x7B\x97\x7B\x7B\x7B\x86\x27\xEB\x60\xA7\xA8\x46\x6E\x1F\x33\x51\x5C\x34\xE0\xD2\x2E\xE8\x0C\x19\x1D\x3D\x3C\xFD\x3B\x6A\xFA\x63\xDF\x28\x87\x86\xF2\xA6\xAC\x87\x86\xF2\xA6\xAC\xD5";
        assert!(decode(&data[..]).is_err());
    }

    #[test]
    /// See: https://github.com/sile/libflate/issues/15 and https://github.com/RazrFalcon/usvg/issues/21
    fn issue_15_2() {
        let data = b"\x1F\x8B\x08\xC1\x7B\x7B\x7B\x7B\x7B\xFC\x5D\x2D\xDC\x08\xC1\x7B\x7B\x7B\x7B\x7B\xFC\x5D\x2D\xDC\x08\xC1\x7B\x7F\x7B\x7B\x7B\xFC\x5D\x2D\xDC\x69\x32\x48\x22\x5A\x81\x81\x42\x42\x81\x7E\x81\x81\x81\x81\xF2\x17";
        assert!(decode(&data[..]).is_err());
    }

    #[test]
    /// See: https://github.com/sile/libflate/issues/15 and https://github.com/RazrFalcon/usvg/issues/22
    fn issue_15_3() {
        let data = b"\x1F\x8B\x08\xC1\x91\x28\x71\xDC\xF2\x2D\x34\x35\x31\x35\x34\x30\x70\x6E\x60\x35\x31\x32\x32\x33\x32\x33\x37\x32\x36\x38\xDD\x1C\xE5\x2A\xDD\xDD\xDD\x22\xDD\xDD\xDD\xDC\x88\x13\xC9\x40\x60\xA7";
        assert!(decode(&data[..]).is_err());
    }

    #[test]
    /// See: https://github.com/sile/libflate/issues/61
    fn issue_61() {
        let data = encode(b"Hello World").unwrap();
        let mut decoder = Decoder::new(&data[..]).unwrap();
        let mut buf = Vec::new();
        decoder.read(&mut buf).unwrap();
        decoder.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"Hello World");
    }

    #[test]
    fn extra_field() {
        let f = ExtraField {
            subfields: vec![ExtraSubField {
                id: [0, 0x42],
                data: "abc".into(),
            }],
        };

        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();

        assert_eq!(ExtraField::read_from(&buf[..]).unwrap(), f);
    }

    #[test]
    #[cfg(not(feature = "no_std"))]
    fn encode_with_extra_field() {
        let mut buf = Vec::new();
        let extra_field = ExtraField {
            subfields: vec![ExtraSubField {
                id: [0, 0x42],
                data: "abc".into(),
            }],
        };
        {
            // encode
            let header = HeaderBuilder::new()
                .extra_field(extra_field.clone())
                .finish();

            let ops = EncodeOptions::new().header(header);
            let mut encoder = Encoder::with_options(&mut buf, ops).unwrap();
            write!(encoder, "hello world").unwrap();
            encoder.finish().as_result().unwrap();
        }
        {
            // decode
            let mut decoder = Decoder::new(&buf[..]).unwrap();
            io::copy(&mut decoder, &mut io::sink()).unwrap();
            assert_eq!(decoder.header().extra_field(), Some(&extra_field));
        }
    }
}
