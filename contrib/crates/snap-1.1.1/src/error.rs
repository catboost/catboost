use std::fmt;
use std::io;
use std::result;

/// A convenient type alias for `Result<T, snap::Error>`.
pub type Result<T> = result::Result<T, Error>;

/// `IntoInnerError` occurs when consuming an encoder fails.
///
/// Consuming the encoder causes a flush to happen. If the flush fails, then
/// this error is returned, which contains both the original encoder and the
/// error that occurred.
///
/// The type parameter `W` is the unconsumed writer.
pub struct IntoInnerError<W> {
    wtr: W,
    err: io::Error,
}

impl<W> IntoInnerError<W> {
    pub(crate) fn new(wtr: W, err: io::Error) -> IntoInnerError<W> {
        IntoInnerError { wtr, err }
    }

    /// Returns the error which caused the call to `into_inner` to fail.
    ///
    /// This error was returned when attempting to flush the internal buffer.
    pub fn error(&self) -> &io::Error {
        &self.err
    }

    /// Returns the error which caused the call to `into_inner` to fail.
    ///
    /// This error was returned when attempting to flush the internal buffer.
    pub fn into_error(self) -> io::Error {
        self.err
    }

    /// Returns the underlying writer which generated the error.
    ///
    /// The returned value can be used for error recovery, such as
    /// re-inspecting the buffer.
    pub fn into_inner(self) -> W {
        self.wtr
    }
}

impl<W: std::any::Any> std::error::Error for IntoInnerError<W> {}

impl<W> fmt::Display for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.err.fmt(f)
    }
}

impl<W> fmt::Debug for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.err.fmt(f)
    }
}

/// Error describes all the possible errors that may occur during Snappy
/// compression or decompression.
///
/// Note that it's unlikely that you'll need to care about the specific error
/// reported since all of them indicate a corrupt Snappy data or a limitation
/// that cannot be worked around. Therefore,
/// `From<snap::Error> for std::io::Error` is provided so that any Snappy
/// errors will be converted to a `std::io::Error` automatically when using
/// `try!`.
#[derive(Clone, Debug)]
pub enum Error {
    /// This error occurs when the given input is too big. This can happen
    /// during compression or decompression.
    TooBig {
        /// The size of the given input.
        given: u64,
        /// The maximum allowed size of an input buffer.
        max: u64,
    },
    /// This error occurs when the given buffer is too small to contain the
    /// maximum possible compressed bytes or the total number of decompressed
    /// bytes.
    BufferTooSmall {
        /// The size of the given output buffer.
        given: u64,
        /// The minimum size of the output buffer.
        min: u64,
    },
    /// This error occurs when trying to decompress a zero length buffer.
    Empty,
    /// This error occurs when an invalid header is found during decompression.
    Header,
    /// This error occurs when there is a mismatch between the number of
    /// decompressed bytes reported in the header and the number of
    /// actual decompressed bytes. In this error case, the number of actual
    /// decompressed bytes is always less than the number reported in the
    /// header.
    HeaderMismatch {
        /// The total number of decompressed bytes expected (i.e., the header
        /// value).
        expected_len: u64,
        /// The total number of actual decompressed bytes.
        got_len: u64,
    },
    /// This error occurs during decompression when there was a problem
    /// reading a literal.
    Literal {
        /// The expected length of the literal.
        len: u64,
        /// The number of remaining bytes in the compressed bytes.
        src_len: u64,
        /// The number of remaining slots in the decompression buffer.
        dst_len: u64,
    },
    /// This error occurs during decompression when there was a problem
    /// reading a copy.
    CopyRead {
        /// The expected length of the copy (as encoded in the compressed
        /// bytes).
        len: u64,
        /// The number of remaining bytes in the compressed bytes.
        src_len: u64,
    },
    /// This error occurs during decompression when there was a problem
    /// writing a copy to the decompression buffer.
    CopyWrite {
        /// The length of the copy (i.e., the total number of bytes to be
        /// produced by this copy in the decompression buffer).
        len: u64,
        /// The number of remaining bytes in the decompression buffer.
        dst_len: u64,
    },
    /// This error occurs during decompression when an invalid copy offset
    /// is found. An offset is invalid if it is zero or if it is out of bounds.
    Offset {
        /// The offset that was read.
        offset: u64,
        /// The current position in the decompression buffer. If the offset is
        /// non-zero, then the offset must be greater than this position.
        dst_pos: u64,
    },
    /// This error occurs when a stream header chunk type was expected but got
    /// a different chunk type.
    /// This error only occurs when reading a Snappy frame formatted stream.
    StreamHeader {
        /// The chunk type byte that was read.
        byte: u8,
    },
    /// This error occurs when the magic stream headers bytes do not match
    /// what is expected.
    /// This error only occurs when reading a Snappy frame formatted stream.
    StreamHeaderMismatch {
        /// The bytes that were read.
        bytes: Vec<u8>,
    },
    /// This error occurs when an unsupported chunk type is seen.
    /// This error only occurs when reading a Snappy frame formatted stream.
    UnsupportedChunkType {
        /// The chunk type byte that was read.
        byte: u8,
    },
    /// This error occurs when trying to read a chunk with an unexpected or
    /// incorrect length when reading a Snappy frame formatted stream.
    /// This error only occurs when reading a Snappy frame formatted stream.
    UnsupportedChunkLength {
        /// The length of the chunk encountered.
        len: u64,
        /// True when this error occured while reading the stream header.
        header: bool,
    },
    /// This error occurs when a checksum validity check fails.
    /// This error only occurs when reading a Snappy frame formatted stream.
    Checksum {
        /// The expected checksum read from the stream.
        expected: u32,
        /// The computed checksum.
        got: u32,
    },
}

impl From<Error> for io::Error {
    fn from(err: Error) -> io::Error {
        io::Error::new(io::ErrorKind::Other, err)
    }
}

impl Eq for Error {}

impl PartialEq for Error {
    fn eq(&self, other: &Error) -> bool {
        use self::Error::*;
        match (self, other) {
            (
                &TooBig { given: given1, max: max1 },
                &TooBig { given: given2, max: max2 },
            ) => (given1, max1) == (given2, max2),
            (
                &BufferTooSmall { given: given1, min: min1 },
                &BufferTooSmall { given: given2, min: min2 },
            ) => (given1, min1) == (given2, min2),
            (&Empty, &Empty) | (&Header, &Header) => true,
            (
                &HeaderMismatch { expected_len: elen1, got_len: glen1 },
                &HeaderMismatch { expected_len: elen2, got_len: glen2 },
            ) => (elen1, glen1) == (elen2, glen2),
            (
                &Literal { len: len1, src_len: src_len1, dst_len: dst_len1 },
                &Literal { len: len2, src_len: src_len2, dst_len: dst_len2 },
            ) => (len1, src_len1, dst_len1) == (len2, src_len2, dst_len2),
            (
                &CopyRead { len: len1, src_len: src_len1 },
                &CopyRead { len: len2, src_len: src_len2 },
            ) => (len1, src_len1) == (len2, src_len2),
            (
                &CopyWrite { len: len1, dst_len: dst_len1 },
                &CopyWrite { len: len2, dst_len: dst_len2 },
            ) => (len1, dst_len1) == (len2, dst_len2),
            (
                &Offset { offset: offset1, dst_pos: dst_pos1 },
                &Offset { offset: offset2, dst_pos: dst_pos2 },
            ) => (offset1, dst_pos1) == (offset2, dst_pos2),
            (&StreamHeader { byte: byte1 }, &StreamHeader { byte: byte2 }) => {
                byte1 == byte2
            }
            (
                &StreamHeaderMismatch { bytes: ref bytes1 },
                &StreamHeaderMismatch { bytes: ref bytes2 },
            ) => bytes1 == bytes2,
            (
                &UnsupportedChunkType { byte: byte1 },
                &UnsupportedChunkType { byte: byte2 },
            ) => byte1 == byte2,
            (
                &UnsupportedChunkLength { len: len1, header: header1 },
                &UnsupportedChunkLength { len: len2, header: header2 },
            ) => (len1, header1) == (len2, header2),
            (
                &Checksum { expected: e1, got: g1 },
                &Checksum { expected: e2, got: g2 },
            ) => (e1, g1) == (e2, g2),
            _ => false,
        }
    }
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::TooBig { given, max } => write!(
                f,
                "snappy: input buffer (size = {}) is larger than \
                         allowed (size = {})",
                given, max
            ),
            Error::BufferTooSmall { given, min } => write!(
                f,
                "snappy: output buffer (size = {}) is smaller than \
                         required (size = {})",
                given, min
            ),
            Error::Empty => write!(f, "snappy: corrupt input (empty)"),
            Error::Header => {
                write!(f, "snappy: corrupt input (invalid header)")
            }
            Error::HeaderMismatch { expected_len, got_len } => write!(
                f,
                "snappy: corrupt input (header mismatch; expected \
                         {} decompressed bytes but got {})",
                expected_len, got_len
            ),
            Error::Literal { len, src_len, dst_len } => write!(
                f,
                "snappy: corrupt input (expected literal read of \
                         length {}; remaining src: {}; remaining dst: {})",
                len, src_len, dst_len
            ),
            Error::CopyRead { len, src_len } => write!(
                f,
                "snappy: corrupt input (expected copy read of \
                         length {}; remaining src: {})",
                len, src_len
            ),
            Error::CopyWrite { len, dst_len } => write!(
                f,
                "snappy: corrupt input (expected copy write of \
                         length {}; remaining dst: {})",
                len, dst_len
            ),
            Error::Offset { offset, dst_pos } => write!(
                f,
                "snappy: corrupt input (expected valid offset but \
                         got offset {}; dst position: {})",
                offset, dst_pos
            ),
            Error::StreamHeader { byte } => write!(
                f,
                "snappy: corrupt input (expected stream header but \
                         got unexpected chunk type byte {})",
                byte
            ),
            Error::StreamHeaderMismatch { ref bytes } => write!(
                f,
                "snappy: corrupt input (expected sNaPpY stream \
                         header but got {})",
                escape(&**bytes)
            ),
            Error::UnsupportedChunkType { byte } => write!(
                f,
                "snappy: corrupt input (unsupported chunk type: {})",
                byte
            ),
            Error::UnsupportedChunkLength { len, header: false } => write!(
                f,
                "snappy: corrupt input \
                         (unsupported chunk length: {})",
                len
            ),
            Error::UnsupportedChunkLength { len, header: true } => write!(
                f,
                "snappy: corrupt input \
                         (invalid stream header length: {})",
                len
            ),
            Error::Checksum { expected, got } => write!(
                f,
                "snappy: corrupt input (bad checksum; \
                         expected: {}, got: {})",
                expected, got
            ),
        }
    }
}

fn escape(bytes: &[u8]) -> String {
    use std::ascii::escape_default;
    bytes.iter().flat_map(|&b| escape_default(b)).map(|b| b as char).collect()
}
