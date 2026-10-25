#![doc = core::include_str!("../README.md")]
#![cfg_attr(not(any(test, feature = "rust-allocator")), no_std)]
#![cfg_attr(
    all(any(miri, feature = "lsx"), target_arch = "loongarch64"),
    feature(stdarch_loongarch)
)]

#[cfg(any(feature = "rust-allocator", feature = "c-allocator"))]
extern crate alloc;

pub mod adler32;
pub mod crc32;

cfg_select! {
    feature = "__internal-api" => {
        pub mod allocate;
        pub mod c_api;
        pub mod deflate;
        pub mod inflate;

        pub const MIN_WBITS: i32 = 8; // 256b LZ77 window
        pub const MAX_WBITS: i32 = 15; // 32kb LZ77 window
    }
    _ => {
        pub(crate) mod allocate;
        pub(crate) mod c_api;
        pub(crate) mod deflate;
        pub(crate) mod inflate;

        pub(crate) const MIN_WBITS: i32 = 8; // 256b LZ77 window
        pub(crate) const MAX_WBITS: i32 = 15; // 32kb LZ77 window
    }
}

mod cpu_features;
mod stable;
mod weak_slice;

pub use stable::{Deflate, DeflateError, Inflate, InflateError, Status};

pub use deflate::{DeflateConfig, Method, Strategy};
pub use inflate::InflateConfig;

pub use deflate::{compress_bound, compress_slice};
pub use inflate::decompress_slice;

macro_rules! traceln {
    ($($arg:tt)*) => {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            eprintln!($($arg)*)
        }
    };
}
pub(crate) use traceln;

macro_rules! trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            eprint!($($arg)*)
        }
    };
}
pub(crate) use trace;

macro_rules! cfg_select {
    ({ $($tt:tt)* }) => {{
        $crate::cfg_select! { $($tt)* }
    }};
    (_ => { $($output:tt)* }) => {
        $($output)*
    };
    (
        $cfg:meta => $output:tt
        $($( $rest:tt )+)?
    ) => {
        #[cfg($cfg)]
        $crate::cfg_select! { _ => $output }
        $(
            #[cfg(not($cfg))]
            $crate::cfg_select! { $($rest)+ }
        )?
    }
}
use cfg_select;

/// Maximum size of the dynamic table.  The maximum number of code structures is
/// 1924, which is the sum of 1332 for literal/length codes and 592 for distance
/// codes.  These values were found by exhaustive searches using the program
/// examples/enough.c found in the zlib distributions.  The arguments to that
/// program are the number of symbols, the initial root table size, and the
/// maximum bit length of a code.  "enough 286 10 15" for literal/length codes
/// returns 1332, and "enough 30 9 15" for distance codes returns 592.
/// The initial root table size (10 or 9) is found in the fifth argument of the
/// inflate_table() calls in inflate.c and infback.c.  If the root table size is
/// changed, then these maximum sizes would be need to be recalculated and
/// updated.
#[allow(unused)]
pub(crate) const ENOUGH: usize = ENOUGH_LENS + ENOUGH_DISTS;
pub(crate) const ENOUGH_LENS: usize = 1332;
pub(crate) const ENOUGH_DISTS: usize = 592;

/// initial adler-32 hash value
pub(crate) const ADLER32_INITIAL_VALUE: usize = 1;
/// initial crc-32 hash value
pub(crate) const CRC32_INITIAL_VALUE: u32 = 0;

pub(crate) const DEF_WBITS: i32 = MAX_WBITS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub enum DeflateFlush {
    #[default]
    /// if flush is set to `NoFlush`, that allows deflate to decide how much data
    /// to accumulate before producing output, in order to maximize compression.
    NoFlush = 0,

    /// If flush is set to `PartialFlush`, all pending output is flushed to the
    /// output buffer, but the output is not aligned to a byte boundary.  All of the
    /// input data so far will be available to the decompressor, as for Z_SYNC_FLUSH.
    /// This completes the current deflate block and follows it with an empty fixed
    /// codes block that is 10 bits long.  This assures that enough bytes are output
    /// in order for the decompressor to finish the block before the empty fixed
    /// codes block.
    PartialFlush = 1,

    /// If the parameter flush is set to `SyncFlush`, all pending output is
    /// flushed to the output buffer and the output is aligned on a byte boundary, so
    /// that the decompressor can get all input data available so far.  (In
    /// particular avail_in is zero after the call if enough output space has been
    /// provided before the call.) Flushing may degrade compression for some
    /// compression algorithms and so it should be used only when necessary.  This
    /// completes the current deflate block and follows it with an empty stored block
    /// that is three bits plus filler bits to the next byte, followed by four bytes
    /// (00 00 ff ff).
    SyncFlush = 2,

    /// If flush is set to `FullFlush`, all output is flushed as with
    /// Z_SYNC_FLUSH, and the compression state is reset so that decompression can
    /// restart from this point if previous compressed data has been damaged or if
    /// random access is desired.  Using `FullFlush` too often can seriously degrade
    /// compression.
    FullFlush = 3,

    /// If the parameter flush is set to `Finish`, pending input is processed,
    /// pending output is flushed and deflate returns with `StreamEnd` if there was
    /// enough output space.  If deflate returns with `Ok` or `BufError`, this
    /// function must be called again with `Finish` and more output space (updated
    /// avail_out) but no more input data, until it returns with `StreamEnd` or an
    /// error.  After deflate has returned `StreamEnd`, the only possible operations
    /// on the stream are deflateReset or deflateEnd.
    ///
    /// `Finish` can be used in the first deflate call after deflateInit if all the
    /// compression is to be done in a single step.  In order to complete in one
    /// call, avail_out must be at least the value returned by deflateBound (see
    /// below).  Then deflate is guaranteed to return `StreamEnd`.  If not enough
    /// output space is provided, deflate will not return `StreamEnd`, and it must
    /// be called again as described above.
    Finish = 4,

    /// If flush is set to `Block`, a deflate block is completed and emitted, as
    /// for `SyncFlush`, but the output is not aligned on a byte boundary, and up to
    /// seven bits of the current block are held to be written as the next byte after
    /// the next deflate block is completed.  In this case, the decompressor may not
    /// be provided enough bits at this point in order to complete decompression of
    /// the data provided so far to the compressor.  It may need to wait for the next
    /// block to be emitted.  This is for advanced applications that need to control
    /// the emission of deflate blocks.
    Block = 5,
}

impl TryFrom<i32> for DeflateFlush {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::NoFlush),
            1 => Ok(Self::PartialFlush),
            2 => Ok(Self::SyncFlush),
            3 => Ok(Self::FullFlush),
            4 => Ok(Self::Finish),
            5 => Ok(Self::Block),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InflateFlush {
    #[default]
    NoFlush = 0,
    SyncFlush = 2,
    Finish = 4,
    Block = 5,
    Trees = 6,
}

impl TryFrom<i32> for InflateFlush {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::NoFlush),
            2 => Ok(Self::SyncFlush),
            4 => Ok(Self::Finish),
            5 => Ok(Self::Block),
            6 => Ok(Self::Trees),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Code {
    /// operation, extra bits, table bits
    pub op: u8,
    /// bits in this part of the code
    pub bits: u8,
    /// offset in table or code value
    pub val: u16,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum ReturnCode {
    Ok = 0,
    StreamEnd = 1,
    NeedDict = 2,
    ErrNo = -1,
    StreamError = -2,
    DataError = -3,
    MemError = -4,
    BufError = -5,
    VersionError = -6,
}

impl From<i32> for ReturnCode {
    fn from(value: i32) -> Self {
        match Self::try_from_c_int(value) {
            Some(value) => value,
            None => panic!("invalid return code {value}"),
        }
    }
}

impl ReturnCode {
    fn error_message_str(self) -> &'static str {
        self.error_message_str_with_null().trim_end_matches('\0')
    }

    const fn error_message_str_with_null(self) -> &'static str {
        match self {
            ReturnCode::Ok => "\0",
            ReturnCode::StreamEnd => "stream end\0",
            ReturnCode::NeedDict => "need dictionary\0",
            ReturnCode::ErrNo => "file error\0",
            ReturnCode::StreamError => "stream error\0",
            ReturnCode::DataError => "data error\0",
            ReturnCode::MemError => "insufficient memory\0",
            ReturnCode::BufError => "buffer error\0",
            ReturnCode::VersionError => "incompatible version\0",
        }
    }

    pub const fn error_message(self) -> *const core::ffi::c_char {
        let msg = self.error_message_str_with_null();
        msg.as_ptr().cast::<core::ffi::c_char>()
    }

    pub const fn try_from_c_int(err: core::ffi::c_int) -> Option<Self> {
        match err {
            0 => Some(Self::Ok),
            1 => Some(Self::StreamEnd),
            2 => Some(Self::NeedDict),
            -1 => Some(Self::ErrNo),
            -2 => Some(Self::StreamError),
            -3 => Some(Self::DataError),
            -4 => Some(Self::MemError),
            -5 => Some(Self::BufError),
            -6 => Some(Self::VersionError),
            _ => None,
        }
    }
}
