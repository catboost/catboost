#![allow(non_snake_case)] // TODO ultimately remove this
#![allow(clippy::missing_safety_doc)] // obviously needs to be fixed long-term

use core::ffi::{c_char, c_int, c_long, c_ulong};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::ControlFlow;

mod bitreader;
mod infback;
mod inffixed_tbl;
mod inftrees;
mod window;
mod writer;

use crate::allocate::Allocator;
use crate::c_api::internal_state;
use crate::cpu_features::CpuFeatures;
use crate::{
    adler32::adler32,
    c_api::{gz_header, z_checksum, z_size, z_stream, Z_DEFLATED},
    inflate::writer::Writer,
    traceln, Code, InflateFlush, ReturnCode, DEF_WBITS, MAX_WBITS, MIN_WBITS,
};

use crate::crc32::{crc32, Crc32Fold};

pub use self::infback::{back, back_end, back_init};
pub use self::window::Window;
use self::{
    bitreader::BitReader,
    inftrees::{inflate_table, CodeType, InflateTable},
};

const INFLATE_STRICT: bool = false;

// SAFETY: This struct must have the same layout as [`z_stream`], so that casts and transmutations
// between the two can work without UB.
#[repr(C)]
pub struct InflateStream<'a> {
    pub(crate) next_in: *mut crate::c_api::Bytef,
    pub(crate) avail_in: crate::c_api::uInt,
    pub(crate) total_in: crate::c_api::z_size,
    pub(crate) next_out: *mut crate::c_api::Bytef,
    pub(crate) avail_out: crate::c_api::uInt,
    pub(crate) total_out: crate::c_api::z_size,
    pub(crate) msg: *mut c_char,
    pub(crate) state: &'a mut State<'a>,
    pub(crate) alloc: Allocator<'a>,
    pub(crate) data_type: c_int,
    pub(crate) adler: crate::c_api::z_checksum,
    pub(crate) reserved: crate::c_api::uLong,
}

unsafe impl Sync for InflateStream<'_> {}
unsafe impl Send for InflateStream<'_> {}

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub const INFLATE_STATE_SIZE: usize = core::mem::size_of::<crate::inflate::State>();

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub unsafe fn set_mode_dict(strm: &mut z_stream) {
    unsafe {
        (*(strm.state as *mut State)).mode = Mode::Dict;
    }
}

#[cfg(feature = "__internal-test")]
#[doc(hidden)]
pub unsafe fn set_mode_sync(strm: *mut z_stream) {
    unsafe {
        (*((*strm).state as *mut State)).mode = Mode::Sync;
    }
}

impl<'a> InflateStream<'a> {
    // z_stream and DeflateStream must have the same layout. Do our best to check if this is true.
    // (imperfect check, but should catch most mistakes.)
    const _S: () = assert!(core::mem::size_of::<z_stream>() == core::mem::size_of::<Self>());
    const _A: () = assert!(core::mem::align_of::<z_stream>() == core::mem::align_of::<Self>());

    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `strm` satisfies the conditions of [`pointer::as_ref`]
    /// - if not `NULL`, `strm` as initialized using [`init`] or similar
    ///
    /// [`pointer::as_ref`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.as_ref
    #[inline(always)]
    pub unsafe fn from_stream_ref(strm: *const z_stream) -> Option<&'a Self> {
        {
            // Safety: ptr points to a valid value of type z_stream (if non-null)
            let stream = unsafe { strm.as_ref() }?;

            if stream.zalloc.is_none() || stream.zfree.is_none() {
                return None;
            }

            if stream.state.is_null() {
                return None;
            }
        }

        // Safety: InflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<InflateStream>().as_ref() }
    }

    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `strm` satisfies the conditions of [`pointer::as_mut`]
    /// - if not `NULL`, `strm` as initialized using [`init`] or similar
    ///
    /// [`pointer::as_mut`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.as_mut
    #[inline(always)]
    pub unsafe fn from_stream_mut(strm: *mut z_stream) -> Option<&'a mut Self> {
        {
            // Safety: ptr points to a valid value of type z_stream (if non-null)
            let stream = unsafe { strm.as_ref() }?;

            if stream.zalloc.is_none() || stream.zfree.is_none() {
                return None;
            }

            if stream.state.is_null() {
                return None;
            }
        }

        // Safety: InflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<InflateStream>().as_mut() }
    }

    fn as_z_stream_mut(&mut self) -> &mut z_stream {
        // safety: a valid &mut InflateStream is also a valid &mut z_stream
        unsafe { &mut *(self as *mut _ as *mut z_stream) }
    }

    pub fn new(config: InflateConfig) -> Self {
        let mut inner = crate::c_api::z_stream::default();

        let ret = crate::inflate::init(&mut inner, config);
        assert_eq!(ret, ReturnCode::Ok);

        unsafe { core::mem::transmute(inner) }
    }
}

const MAX_BITS: u8 = 15; // maximum number of bits in a code
const MAX_DIST_EXTRA_BITS: u8 = 13; // maximum number of extra distance bits

/// Decompresses `input` into the provided `output` buffer.
///
/// Returns a subslice of `output` containing the decompressed bytes and a
/// [`ReturnCode`] indicating the result of the operation. Returns [`ReturnCode::BufError`] if
/// there is insufficient output space.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
/// # fn foo(compressed: &[u8]) {
/// let mut buffer = [0u8; 1024];
/// let (decompressed, rc) = decompress_slice(&mut buffer, compressed, InflateConfig::default());
/// assert_eq!(rc, ReturnCode::Ok);
/// # }
/// ```
pub fn decompress_slice<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    // SAFETY: [u8] is also a valid [MaybeUninit<u8>]
    let output_uninit = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut MaybeUninit<u8>, output.len())
    };

    uncompress(output_uninit, input, config)
}

/// Inflates `source` into `dest`, and writes the final inflated size into `dest_len`.
pub fn uncompress<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: InflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    let (_consumed, output, ret) = uncompress2(output, input, config);
    (output, ret)
}

pub fn uncompress2<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: InflateConfig,
) -> (u64, &'a mut [u8], ReturnCode) {
    let mut dest_len_ptr = output.len() as z_checksum;

    // for detection of incomplete stream when *destLen == 0
    let mut buf = [0u8];

    let mut left;
    let mut len = input.len() as u64;

    let dest = if output.is_empty() {
        left = 1;

        buf.as_mut_ptr()
    } else {
        left = output.len() as u64;
        dest_len_ptr = 0;

        output.as_mut_ptr() as *mut u8
    };

    let mut stream = z_stream {
        next_in: input.as_ptr() as *mut u8,
        avail_in: 0,

        zalloc: None,
        zfree: None,
        opaque: core::ptr::null_mut(),

        ..z_stream::default()
    };

    let err = init(&mut stream, config);
    if err != ReturnCode::Ok {
        return (0, &mut [], err);
    }

    stream.next_out = dest;
    stream.avail_out = 0;

    let Some(stream) = (unsafe { InflateStream::from_stream_mut(&mut stream) }) else {
        return (0, &mut [], ReturnCode::StreamError);
    };

    let err = loop {
        if stream.avail_out == 0 {
            stream.avail_out = Ord::min(left, u32::MAX as u64) as u32;
            left -= stream.avail_out as u64;
        }

        if stream.avail_in == 0 {
            stream.avail_in = Ord::min(len, u32::MAX as u64) as u32;
            len -= stream.avail_in as u64;
        }

        let err = unsafe { inflate(stream, InflateFlush::NoFlush) };

        if err != ReturnCode::Ok {
            break err;
        }
    };

    let consumed = len + u64::from(stream.avail_in);
    if !output.is_empty() {
        dest_len_ptr = stream.total_out;
    } else if stream.total_out != 0 && err == ReturnCode::BufError {
        left = 1;
    }

    let avail_out = stream.avail_out;

    end(stream);

    let ret = match err {
        ReturnCode::StreamEnd => ReturnCode::Ok,
        ReturnCode::NeedDict => ReturnCode::DataError,
        ReturnCode::BufError if (left + avail_out as u64) != 0 => ReturnCode::DataError,
        _ => err,
    };

    // SAFETY: we have now initialized these bytes
    let output_slice = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, dest_len_ptr as usize)
    };

    (consumed, output_slice, ret)
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Mode {
    Head,
    Flags,
    Time,
    Os,
    ExLen,
    Extra,
    Name,
    Comment,
    HCrc,
    Sync,
    Mem,
    Length,
    Type,
    TypeDo,
    Stored,
    CopyBlock,
    Check,
    Len_,
    Len,
    Lit,
    LenExt,
    Dist,
    DistExt,
    Match,
    Table,
    LenLens,
    CodeLens,
    DictId,
    Dict,
    Done,
    Bad,
}

#[derive(Default, Clone, Copy)]
#[allow(clippy::enum_variant_names)]
enum Codes {
    #[default]
    Fixed,
    Codes,
    Len,
    Dist,
}

#[derive(Default, Clone, Copy)]
struct Table {
    codes: Codes,
    bits: usize,
}

#[derive(Clone, Copy)]
struct Flags(u8);

impl Default for Flags {
    fn default() -> Self {
        Self::SANE
    }
}

impl Flags {
    /// set if currently processing the last block
    const IS_LAST_BLOCK: Self = Self(0b0000_0001);

    /// set if a custom dictionary was provided
    const HAVE_DICT: Self = Self(0b0000_0010);

    /// if false, allow invalid distance too far
    const SANE: Self = Self(0b0000_0100);

    pub(crate) const fn contains(self, other: Self) -> bool {
        debug_assert!(other.0.count_ones() == 1);

        self.0 & other.0 != 0
    }

    #[inline(always)]
    pub(crate) fn update(&mut self, other: Self, value: bool) {
        if value {
            *self = Self(self.0 | other.0);
        } else {
            *self = Self(self.0 & !other.0);
        }
    }
}

#[repr(C, align(64))]
pub(crate) struct State<'a> {
    /// Current inflate mode
    mode: Mode,

    flags: Flags,

    /// log base 2 of requested window size
    wbits: u8,

    /// bitflag
    ///
    /// - bit 0 true if zlib
    /// - bit 1 true if gzip
    /// - bit 2 true to validate check value
    wrap: u8,

    flush: InflateFlush,

    // allocated window if needed (capacity == 0 if unused)
    window: Window<'a>,

    //
    /// number of code length code lengths
    ncode: usize,
    /// number of length code lengths
    nlen: usize,
    /// number of distance code lengths
    ndist: usize,
    /// number of code lengths in lens[]
    have: usize,
    /// next available space in codes[]
    next: usize, // represented as an index, don't want a self-referential structure here

    // IO
    bit_reader: BitReader<'a>,

    writer: Writer<'a>,
    total: usize,

    /// length of a block to copy
    length: usize,
    /// distance back to copy the string from
    offset: usize,

    /// extra bits needed
    extra: usize,

    /// bits back of last unprocessed length/lit
    back: usize,

    /// initial length of match
    was: usize,

    /// size of memory copying chunk
    chunksize: usize,

    in_available: usize,
    out_available: usize,

    gzip_flags: i32,

    checksum: u32,
    crc_fold: Crc32Fold,

    error_message: Option<&'static str>,

    /// place to store gzip header if needed
    head: Option<&'a mut gz_header>,
    dmax: usize,

    /// table for length/literal codes
    len_table: Table,

    /// table for dist codes
    dist_table: Table,

    codes_codes: [Code; crate::ENOUGH_LENS],
    len_codes: [Code; crate::ENOUGH_LENS],
    dist_codes: [Code; crate::ENOUGH_DISTS],

    /// temporary storage space for code lengths
    lens: [u16; 320],
    /// work area for code table building
    work: [u16; 288],

    allocation_start: *mut u8,
    total_allocation_size: usize,
}

impl<'a> State<'a> {
    fn new(reader: &'a [u8], writer: Writer<'a>) -> Self {
        let in_available = reader.len();
        let out_available = writer.capacity();

        Self {
            flush: InflateFlush::NoFlush,

            flags: Flags::default(),
            wrap: 0,
            mode: Mode::Head,
            length: 0,

            len_table: Table::default(),
            dist_table: Table::default(),

            wbits: 0,
            offset: 0,
            extra: 0,
            back: 0,
            was: 0,
            chunksize: 0,
            in_available,
            out_available,

            bit_reader: BitReader::new(reader),

            writer,
            total: 0,

            window: Window::empty(),
            head: None,

            lens: [0u16; 320],
            work: [0u16; 288],

            ncode: 0,
            nlen: 0,
            ndist: 0,
            have: 0,
            next: 0,

            error_message: None,

            checksum: 0,
            crc_fold: Crc32Fold::new(),

            dmax: 0,
            gzip_flags: 0,

            codes_codes: [Code::default(); crate::ENOUGH_LENS],
            len_codes: [Code::default(); crate::ENOUGH_LENS],
            dist_codes: [Code::default(); crate::ENOUGH_DISTS],

            allocation_start: core::ptr::null_mut(),
            total_allocation_size: 0,
        }
    }

    fn len_table_ref(&self) -> &[Code] {
        match self.len_table.codes {
            Codes::Fixed => &self::inffixed_tbl::LENFIX,
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    fn dist_table_ref(&self) -> &[Code] {
        match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX,
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        }
    }

    fn len_table_get(&self, index: usize) -> Code {
        self.len_table_ref()[index]
    }

    fn dist_table_get(&self, index: usize) -> Code {
        self.dist_table_ref()[index]
    }
}

// swaps endianness
const fn zswap32(q: u32) -> u32 {
    u32::from_be(q.to_le())
}

const INFLATE_FAST_MIN_HAVE: usize = 15;
const INFLATE_FAST_MIN_LEFT: usize = 260;

impl State<'_> {
    // This logic is split into its own function for two reasons
    //
    // - We get to load state to the stack; doing this in all cases is expensive, but doing it just
    //      for Len and related states is very helpful.
    // - The `-Cllvm-args=-enable-dfa-jump-thread` llvm arg is able to optimize this function, but
    //      not the entirity of `dispatch`. We get a massive boost from that pass.
    //
    // It unfortunately does duplicate the code for some of the states; deduplicating it by having
    // more of the states call this function is slower.
    fn len_and_friends(&mut self) -> ControlFlow<ReturnCode, ()> {
        let avail_in = self.bit_reader.bytes_remaining();
        let avail_out = self.writer.remaining();

        if avail_in >= INFLATE_FAST_MIN_HAVE && avail_out >= INFLATE_FAST_MIN_LEFT {
            // SAFETY: INFLATE_FAST_MIN_HAVE is enough bytes remaining to satisfy the precondition.
            unsafe { inflate_fast_help(self, 0) };
            match self.mode {
                Mode::Len => {}
                _ => return ControlFlow::Continue(()),
            }
        }

        let mut mode;
        let mut writer;
        let mut bit_reader;

        macro_rules! load {
            () => {
                mode = self.mode;
                writer = core::mem::replace(&mut self.writer, Writer::new(&mut []));
                bit_reader = self.bit_reader;
            };
        }

        macro_rules! restore {
            () => {
                self.mode = mode;
                self.writer = writer;
                self.bit_reader = bit_reader;
            };
        }

        load!();

        let len_table = match self.len_table.codes {
            Codes::Fixed => &self::inffixed_tbl::LENFIX[..],
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        };

        let dist_table = match self.dist_table.codes {
            Codes::Fixed => &self::inffixed_tbl::DISTFIX[..],
            Codes::Codes => &self.codes_codes,
            Codes::Len => &self.len_codes,
            Codes::Dist => &self.dist_codes,
        };

        loop {
            mode = 'top: {
                match mode {
                    Mode::Len => {
                        let avail_in = bit_reader.bytes_remaining();
                        let avail_out = writer.remaining();

                        // INFLATE_FAST_MIN_LEFT is important. It makes sure there is at least 32 bytes of free
                        // space available. This means for many SIMD operations we don't need to process a
                        // remainder; we just copy blindly, and a later operation will overwrite the extra copied
                        // bytes
                        if avail_in >= INFLATE_FAST_MIN_HAVE && avail_out >= INFLATE_FAST_MIN_LEFT {
                            restore!();
                            // SAFETY: INFLATE_FAST_MIN_HAVE >= 15.
                            // Note that the restore macro does not do anything that would
                            // reduce the number of bytes available.
                            unsafe { inflate_fast_help(self, 0) };
                            return ControlFlow::Continue(());
                        }

                        self.back = 0;

                        // get a literal, length, or end-of-block code
                        let mut here;
                        loop {
                            let bits = bit_reader.bits(self.len_table.bits);
                            here = len_table[bits as usize];

                            if here.bits <= bit_reader.bits_in_buffer() {
                                break;
                            }

                            if let Err(return_code) = bit_reader.pull_byte() {
                                restore!();
                                return ControlFlow::Break(return_code);
                            };
                        }

                        if here.op != 0 && here.op & 0xf0 == 0 {
                            let last = here;
                            loop {
                                let bits = bit_reader.bits((last.bits + last.op) as usize) as u16;
                                here = len_table[(last.val + (bits >> last.bits)) as usize];
                                if last.bits + here.bits <= bit_reader.bits_in_buffer() {
                                    break;
                                }

                                if let Err(return_code) = bit_reader.pull_byte() {
                                    restore!();
                                    return ControlFlow::Break(return_code);
                                };
                            }

                            bit_reader.drop_bits(last.bits);
                            self.back += last.bits as usize;
                        }

                        bit_reader.drop_bits(here.bits);
                        self.back += here.bits as usize;
                        self.length = here.val as usize;

                        if here.op == 0 {
                            break 'top Mode::Lit;
                        } else if here.op & 32 != 0 {
                            // end of block

                            traceln!("inflate:         end of block");

                            self.back = usize::MAX;
                            mode = Mode::Type;

                            restore!();
                            return ControlFlow::Continue(());
                        } else if here.op & 64 != 0 {
                            mode = Mode::Bad;
                            {
                                restore!();
                                let this = &mut *self;
                                let msg: &'static str = "invalid literal/length code\0";
                                #[cfg(all(feature = "std", test))]
                                dbg!(msg);
                                this.error_message = Some(msg);
                                return ControlFlow::Break(ReturnCode::DataError);
                            }
                        } else {
                            // length code
                            self.extra = (here.op & MAX_BITS) as usize;
                            break 'top Mode::LenExt;
                        }
                    }
                    Mode::Lit => {
                        // NOTE: this branch must be kept in sync with its counterpart in `dispatch`
                        if writer.is_full() {
                            restore!();
                            traceln!("Ok: writer is full ({} bytes)", self.writer.capacity());
                            return ControlFlow::Break(ReturnCode::Ok);
                        }

                        writer.push(self.length as u8);

                        break 'top Mode::Len;
                    }
                    Mode::LenExt => {
                        // NOTE: this branch must be kept in sync with its counterpart in `dispatch`
                        let extra = self.extra;

                        // get extra bits, if any
                        if extra != 0 {
                            match bit_reader.need_bits(extra) {
                                Err(return_code) => {
                                    restore!();
                                    return ControlFlow::Break(return_code);
                                }
                                Ok(v) => v,
                            };
                            self.length += bit_reader.bits(extra) as usize;
                            bit_reader.drop_bits(extra as u8);
                            self.back += extra;
                        }

                        traceln!("inflate: length {}", self.length);

                        self.was = self.length;

                        break 'top Mode::Dist;
                    }
                    Mode::Dist => {
                        // NOTE: this branch must be kept in sync with its counterpart in `dispatch`

                        // get distance code
                        let mut here;
                        loop {
                            let bits = bit_reader.bits(self.dist_table.bits) as usize;
                            here = dist_table[bits];
                            if here.bits <= bit_reader.bits_in_buffer() {
                                break;
                            }

                            if let Err(return_code) = bit_reader.pull_byte() {
                                restore!();
                                return ControlFlow::Break(return_code);
                            };
                        }

                        if here.op & 0xf0 == 0 {
                            let last = here;

                            loop {
                                let bits = bit_reader.bits((last.bits + last.op) as usize);
                                here =
                                    dist_table[last.val as usize + ((bits as usize) >> last.bits)];

                                if last.bits + here.bits <= bit_reader.bits_in_buffer() {
                                    break;
                                }

                                if let Err(return_code) = bit_reader.pull_byte() {
                                    restore!();
                                    return ControlFlow::Break(return_code);
                                };
                            }

                            bit_reader.drop_bits(last.bits);
                            self.back += last.bits as usize;
                        }

                        bit_reader.drop_bits(here.bits);

                        if here.op & 64 != 0 {
                            restore!();
                            self.mode = Mode::Bad;
                            return ControlFlow::Break(self.bad("invalid distance code\0"));
                        }

                        self.offset = here.val as usize;

                        self.extra = (here.op & MAX_BITS) as usize;

                        break 'top Mode::DistExt;
                    }
                    Mode::DistExt => {
                        // NOTE: this branch must be kept in sync with its counterpart in `dispatch`
                        let extra = self.extra;

                        if extra > 0 {
                            match bit_reader.need_bits(extra) {
                                Err(return_code) => {
                                    restore!();
                                    return ControlFlow::Break(return_code);
                                }
                                Ok(v) => v,
                            };
                            self.offset += bit_reader.bits(extra) as usize;
                            bit_reader.drop_bits(extra as u8);
                            self.back += extra;
                        }

                        if INFLATE_STRICT && self.offset > self.dmax {
                            restore!();
                            self.mode = Mode::Bad;
                            return ControlFlow::Break(
                                self.bad("invalid distance code too far back\0"),
                            );
                        }

                        traceln!("inflate: distance {}", self.offset);

                        break 'top Mode::Match;
                    }
                    Mode::Match => {
                        // NOTE: this branch must be kept in sync with its counterpart in `dispatch`
                        if writer.is_full() {
                            restore!();
                            traceln!(
                                "BufError: writer is full ({} bytes)",
                                self.writer.capacity()
                            );
                            return ControlFlow::Break(ReturnCode::Ok);
                        }

                        let left = writer.remaining();
                        let copy = writer.len();

                        let copy = if self.offset > copy {
                            // copy from window to output

                            let mut copy = self.offset - copy;

                            if copy > self.window.have() {
                                if self.flags.contains(Flags::SANE) {
                                    restore!();
                                    self.mode = Mode::Bad;
                                    return ControlFlow::Break(
                                        self.bad("invalid distance too far back\0"),
                                    );
                                }

                                // TODO INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
                                panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
                            }

                            let wnext = self.window.next();
                            let wsize = self.window.size();

                            let from = if copy > wnext {
                                copy -= wnext;
                                wsize - copy
                            } else {
                                wnext - copy
                            };

                            copy = Ord::min(copy, self.length);
                            copy = Ord::min(copy, left);

                            writer.extend_from_window(&self.window, from..from + copy);

                            copy
                        } else {
                            let copy = Ord::min(self.length, left);
                            writer.copy_match(self.offset, copy);

                            copy
                        };

                        self.length -= copy;

                        if self.length == 0 {
                            break 'top Mode::Len;
                        } else {
                            // otherwise it seems to recurse?
                            // self.match_()
                            break 'top Mode::Match;
                        }
                    }
                    _ => unsafe { core::hint::unreachable_unchecked() },
                }
            }
        }
    }

    fn dispatch(&mut self) -> ReturnCode {
        // Note: All early returns must save mode into self.mode again.
        let mut mode = self.mode;

        macro_rules! pull_byte {
            ($self:expr) => {
                match $self.bit_reader.pull_byte() {
                    Err(return_code) => {
                        self.mode = mode;
                        return $self.inflate_leave(return_code);
                    }
                    Ok(_) => (),
                }
            };
        }

        macro_rules! need_bits {
            ($self:expr, $n:expr) => {
                match $self.bit_reader.need_bits($n) {
                    Err(return_code) => {
                        self.mode = mode;
                        return $self.inflate_leave(return_code);
                    }
                    Ok(v) => v,
                }
            };
        }

        let ret = 'label: loop {
            mode = 'blk: {
                match mode {
                    Mode::Head => {
                        if self.wrap == 0 {
                            break 'blk Mode::TypeDo;
                        }

                        need_bits!(self, 16);

                        // Gzip
                        if (self.wrap & 2) != 0 && self.bit_reader.hold() == 0x8b1f {
                            if self.wbits == 0 {
                                self.wbits = 15;
                            }

                            let b0 = self.bit_reader.bits(8) as u8;
                            let b1 = (self.bit_reader.hold() >> 8) as u8;
                            self.checksum = crc32(crate::CRC32_INITIAL_VALUE, &[b0, b1]);
                            self.bit_reader.init_bits();

                            break 'blk Mode::Flags;
                        }

                        if let Some(header) = &mut self.head {
                            header.done = -1;
                        }

                        // check if zlib header is allowed
                        if (self.wrap & 1) == 0
                            || ((self.bit_reader.bits(8) << 8) + (self.bit_reader.hold() >> 8)) % 31
                                != 0
                        {
                            mode = Mode::Bad;
                            break 'label self.bad("incorrect header check\0");
                        }

                        if self.bit_reader.bits(4) != Z_DEFLATED as u64 {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown compression method\0");
                        }

                        self.bit_reader.drop_bits(4);
                        let len = self.bit_reader.bits(4) as u8 + 8;

                        if self.wbits == 0 {
                            self.wbits = len;
                        }

                        if len as i32 > MAX_WBITS || len > self.wbits {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid window size\0");
                        }

                        self.dmax = 1 << len;
                        self.gzip_flags = 0; // indicate zlib header
                        self.checksum = crate::ADLER32_INITIAL_VALUE as _;

                        if self.bit_reader.hold() & 0x200 != 0 {
                            self.bit_reader.init_bits();

                            break 'blk Mode::DictId;
                        } else {
                            self.bit_reader.init_bits();

                            break 'blk Mode::Type;
                        }
                    }
                    Mode::Flags => {
                        need_bits!(self, 16);
                        self.gzip_flags = self.bit_reader.hold() as i32;

                        // Z_DEFLATED = 8 is the only supported method
                        if self.gzip_flags & 0xff != Z_DEFLATED {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown compression method\0");
                        }

                        if self.gzip_flags & 0xe000 != 0 {
                            mode = Mode::Bad;
                            break 'label self.bad("unknown header flags set\0");
                        }

                        if let Some(head) = self.head.as_mut() {
                            head.text = ((self.bit_reader.hold() >> 8) & 1) as i32;
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let b0 = self.bit_reader.bits(8) as u8;
                            let b1 = (self.bit_reader.hold() >> 8) as u8;
                            self.checksum = crc32(self.checksum, &[b0, b1]);
                        }

                        self.bit_reader.init_bits();

                        break 'blk Mode::Time;
                    }
                    Mode::Time => {
                        need_bits!(self, 32);
                        if let Some(head) = self.head.as_mut() {
                            head.time = self.bit_reader.hold() as z_size;
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let bytes = (self.bit_reader.hold() as u32).to_le_bytes();
                            self.checksum = crc32(self.checksum, &bytes);
                        }

                        self.bit_reader.init_bits();

                        break 'blk Mode::Os;
                    }
                    Mode::Os => {
                        need_bits!(self, 16);
                        if let Some(head) = self.head.as_mut() {
                            head.xflags = (self.bit_reader.hold() & 0xff) as i32;
                            head.os = (self.bit_reader.hold() >> 8) as i32;
                        }

                        if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                            let bytes = (self.bit_reader.hold() as u16).to_le_bytes();
                            self.checksum = crc32(self.checksum, &bytes);
                        }

                        self.bit_reader.init_bits();

                        break 'blk Mode::ExLen;
                    }
                    Mode::ExLen => {
                        if (self.gzip_flags & 0x0400) != 0 {
                            need_bits!(self, 16);

                            // self.length (and head.extra_len) represent the length of the extra field
                            self.length = self.bit_reader.hold() as usize;
                            if let Some(head) = self.head.as_mut() {
                                head.extra_len = self.length as u32;
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                let bytes = (self.bit_reader.hold() as u16).to_le_bytes();
                                self.checksum = crc32(self.checksum, &bytes);
                            }
                            self.bit_reader.init_bits();
                        } else if let Some(head) = self.head.as_mut() {
                            head.extra = core::ptr::null_mut();
                        }

                        break 'blk Mode::Extra;
                    }
                    Mode::Extra => {
                        if (self.gzip_flags & 0x0400) != 0 {
                            // self.length is the number of remaining `extra` bytes. But they may not all be available
                            let extra_available =
                                Ord::min(self.length, self.bit_reader.bytes_remaining());

                            if extra_available > 0 {
                                if let Some(head) = self.head.as_mut() {
                                    if !head.extra.is_null() {
                                        // at `head.extra`, the caller has reserved `head.extra_max` bytes.
                                        // in the deflated byte stream, we've found a gzip header with
                                        // `head.extra_len` bytes of data. We must be careful because
                                        // `head.extra_len` may be larger than `head.extra_max`.

                                        // how many bytes we've already written into `head.extra`
                                        let written_so_far = head.extra_len as usize - self.length;

                                        // min of number of bytes available at dst and at src
                                        let count = Ord::min(
                                            (head.extra_max as usize)
                                                .saturating_sub(written_so_far),
                                            extra_available,
                                        );

                                        // SAFETY: location where we'll write: this saturates at the
                                        // `head.extra.add(head.extra.max)` to prevent UB
                                        let next_write_offset =
                                            Ord::min(written_so_far, head.extra_max as usize);

                                        unsafe {
                                            // SAFETY: count is effectively bounded by head.extra_max
                                            // and bit_reader.bytes_remaining(), so the count won't
                                            // go out of bounds.
                                            core::ptr::copy_nonoverlapping(
                                                self.bit_reader.as_mut_ptr(),
                                                head.extra.add(next_write_offset),
                                                count,
                                            );
                                        }
                                    }
                                }

                                // Checksum
                                if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                    let extra_slice =
                                        &self.bit_reader.as_slice()[..extra_available];
                                    self.checksum = crc32(self.checksum, extra_slice)
                                }

                                self.in_available -= extra_available;
                                self.bit_reader.advance(extra_available);
                                self.length -= extra_available;
                            }

                            // Checks for errors occur after returning
                            if self.length != 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }
                        }

                        self.length = 0;

                        break 'blk Mode::Name;
                    }
                    Mode::Name => {
                        if (self.gzip_flags & 0x0800) != 0 {
                            if self.in_available == 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }

                            // the name string will always be null-terminated, but might be longer than we have
                            // space for in the header struct. Nonetheless, we read the whole thing.
                            let slice = self.bit_reader.as_slice();
                            let null_terminator_index = slice.iter().position(|c| *c == 0);

                            // we include the null terminator if it exists
                            let name_slice = match null_terminator_index {
                                Some(i) => &slice[..=i],
                                None => slice,
                            };

                            // if the header has space, store as much as possible in there
                            if let Some(head) = self.head.as_mut() {
                                if !head.name.is_null() {
                                    let remaining_name_bytes = (head.name_max as usize)
                                        .checked_sub(self.length)
                                        .expect("name out of bounds");
                                    let copy = Ord::min(name_slice.len(), remaining_name_bytes);

                                    unsafe {
                                        // SAFETY: copy is effectively bound by the name length and
                                        // head.name_max, so this won't go out of bounds.
                                        core::ptr::copy_nonoverlapping(
                                            name_slice.as_ptr(),
                                            head.name.add(self.length),
                                            copy,
                                        )
                                    };

                                    self.length += copy;
                                }
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                self.checksum = crc32(self.checksum, name_slice);
                            }

                            let reached_end = name_slice.last() == Some(&0);
                            self.bit_reader.advance(name_slice.len());

                            if !reached_end && self.bit_reader.bytes_remaining() == 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }
                        } else if let Some(head) = self.head.as_mut() {
                            head.name = core::ptr::null_mut();
                        }

                        self.length = 0;

                        break 'blk Mode::Comment;
                    }
                    Mode::Comment => {
                        if (self.gzip_flags & 0x01000) != 0 {
                            if self.in_available == 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }

                            // the comment string will always be null-terminated, but might be longer than we have
                            // space for in the header struct. Nonetheless, we read the whole thing.
                            let slice = self.bit_reader.as_slice();
                            let null_terminator_index = slice.iter().position(|c| *c == 0);

                            // we include the null terminator if it exists
                            let comment_slice = match null_terminator_index {
                                Some(i) => &slice[..=i],
                                None => slice,
                            };

                            // if the header has space, store as much as possible in there
                            if let Some(head) = self.head.as_mut() {
                                if !head.comment.is_null() {
                                    let remaining_comm_bytes = (head.comm_max as usize)
                                        .checked_sub(self.length)
                                        .expect("comm out of bounds");
                                    let copy = Ord::min(comment_slice.len(), remaining_comm_bytes);

                                    unsafe {
                                        // SAFETY: copy is effectively bound by the comment length and
                                        // head.comm_max, so this won't go out of bounds.
                                        core::ptr::copy_nonoverlapping(
                                            comment_slice.as_ptr(),
                                            head.comment.add(self.length),
                                            copy,
                                        )
                                    };

                                    self.length += copy;
                                }
                            }

                            if (self.gzip_flags & 0x0200) != 0 && (self.wrap & 4) != 0 {
                                self.checksum = crc32(self.checksum, comment_slice);
                            }

                            let reached_end = comment_slice.last() == Some(&0);
                            self.bit_reader.advance(comment_slice.len());

                            if !reached_end && self.bit_reader.bytes_remaining() == 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }
                        } else if let Some(head) = self.head.as_mut() {
                            head.comment = core::ptr::null_mut();
                        }

                        break 'blk Mode::HCrc;
                    }
                    Mode::HCrc => {
                        if (self.gzip_flags & 0x0200) != 0 {
                            need_bits!(self, 16);

                            if (self.wrap & 4) != 0
                                && self.bit_reader.hold() as u32 != (self.checksum & 0xffff)
                            {
                                mode = Mode::Bad;
                                break 'label self.bad("header crc mismatch\0");
                            }

                            self.bit_reader.init_bits();
                        }

                        if let Some(head) = self.head.as_mut() {
                            head.hcrc = (self.gzip_flags >> 9) & 1;
                            head.done = 1;
                        }

                        // compute crc32 checksum if not in raw mode
                        if (self.wrap & 4 != 0) && self.gzip_flags != 0 {
                            self.crc_fold = Crc32Fold::new();
                            self.checksum = crate::CRC32_INITIAL_VALUE;
                        }

                        break 'blk Mode::Type;
                    }
                    Mode::Type => {
                        use InflateFlush::*;

                        match self.flush {
                            Block | Trees => break 'label ReturnCode::Ok,
                            NoFlush | SyncFlush | Finish => {
                                // NOTE: this is slightly different to what zlib-rs does!
                                break 'blk Mode::TypeDo;
                            }
                        }
                    }
                    Mode::TypeDo => {
                        if self.flags.contains(Flags::IS_LAST_BLOCK) {
                            self.bit_reader.next_byte_boundary();
                            break 'blk Mode::Check;
                        }

                        need_bits!(self, 3);
                        let last = self.bit_reader.bits(1) != 0;
                        self.flags.update(Flags::IS_LAST_BLOCK, last);
                        self.bit_reader.drop_bits(1);

                        match self.bit_reader.bits(2) {
                            0b00 => {
                                traceln!("inflate:     stored block (last = {last})");

                                self.bit_reader.drop_bits(2);

                                break 'blk Mode::Stored;
                            }
                            0b01 => {
                                traceln!("inflate:     fixed codes block (last = {last})");

                                self.len_table = Table {
                                    codes: Codes::Fixed,
                                    bits: 9,
                                };

                                self.dist_table = Table {
                                    codes: Codes::Fixed,
                                    bits: 5,
                                };

                                mode = Mode::Len_;

                                self.bit_reader.drop_bits(2);

                                if let InflateFlush::Trees = self.flush {
                                    break 'label self.inflate_leave(ReturnCode::Ok);
                                } else {
                                    break 'blk Mode::Len_;
                                }
                            }
                            0b10 => {
                                traceln!("inflate:     dynamic codes block (last = {last})");

                                self.bit_reader.drop_bits(2);

                                break 'blk Mode::Table;
                            }
                            0b11 => {
                                traceln!("inflate:     invalid block type");

                                self.bit_reader.drop_bits(2);

                                mode = Mode::Bad;
                                break 'label self.bad("invalid block type\0");
                            }
                            _ => {
                                // LLVM will optimize this branch away
                                unreachable!("BitReader::bits(2) only yields a value of two bits, so this match is already exhaustive")
                            }
                        }
                    }
                    Mode::Stored => {
                        self.bit_reader.next_byte_boundary();

                        need_bits!(self, 32);

                        let hold = self.bit_reader.bits(32) as u32;

                        traceln!("hold {hold:#x}");

                        if hold as u16 != !((hold >> 16) as u16) {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid stored block lengths\0");
                        }

                        self.length = hold as usize & 0xFFFF;
                        traceln!("inflate:     stored length {}", self.length);

                        self.bit_reader.init_bits();

                        if let InflateFlush::Trees = self.flush {
                            break 'label self.inflate_leave(ReturnCode::Ok);
                        } else {
                            break 'blk Mode::CopyBlock;
                        }
                    }
                    Mode::CopyBlock => {
                        loop {
                            let mut copy = self.length;

                            if copy == 0 {
                                break;
                            }

                            copy = Ord::min(copy, self.writer.remaining());
                            copy = Ord::min(copy, self.bit_reader.bytes_remaining());

                            if copy == 0 {
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }

                            self.writer.extend(&self.bit_reader.as_slice()[..copy]);
                            self.bit_reader.advance(copy);

                            self.length -= copy;
                        }

                        break 'blk Mode::Type;
                    }
                    Mode::Check => {
                        if !cfg!(feature = "__internal-fuzz-disable-checksum") && self.wrap != 0 {
                            need_bits!(self, 32);

                            self.total += self.writer.len();

                            if self.wrap & 4 != 0 {
                                if self.gzip_flags != 0 {
                                    self.crc_fold.fold(self.writer.filled(), self.checksum);
                                    self.checksum = self.crc_fold.finish();
                                } else {
                                    self.checksum = adler32(self.checksum, self.writer.filled());
                                }
                            }

                            let given_checksum = if self.gzip_flags != 0 {
                                self.bit_reader.hold() as u32
                            } else {
                                zswap32(self.bit_reader.hold() as u32)
                            };

                            self.out_available = self.writer.capacity() - self.writer.len();

                            if self.wrap & 4 != 0 && given_checksum != self.checksum {
                                mode = Mode::Bad;
                                break 'label self.bad("incorrect data check\0");
                            }

                            self.bit_reader.init_bits();
                        }

                        break 'blk Mode::Length;
                    }
                    Mode::Len_ => {
                        break 'blk Mode::Len;
                    }
                    Mode::Len => {
                        self.mode = mode;
                        let val = self.len_and_friends();
                        mode = self.mode;
                        match val {
                            ControlFlow::Break(return_code) => break 'label return_code,
                            ControlFlow::Continue(()) => continue 'label,
                        }
                    }
                    Mode::LenExt => {
                        // NOTE: this branch must be kept in sync with its counterpart in `len_and_friends`
                        let extra = self.extra;

                        // get extra bits, if any
                        if extra != 0 {
                            need_bits!(self, extra);
                            self.length += self.bit_reader.bits(extra) as usize;
                            self.bit_reader.drop_bits(extra as u8);
                            self.back += extra;
                        }

                        traceln!("inflate: length {}", self.length);

                        self.was = self.length;

                        break 'blk Mode::Dist;
                    }
                    Mode::Lit => {
                        // NOTE: this branch must be kept in sync with its counterpart in `len_and_friends`
                        if self.writer.is_full() {
                            traceln!("Ok: writer is full ({} bytes)", self.writer.capacity());
                            break 'label self.inflate_leave(ReturnCode::Ok);
                        }

                        self.writer.push(self.length as u8);

                        break 'blk Mode::Len;
                    }
                    Mode::Dist => {
                        // NOTE: this branch must be kept in sync with its counterpart in `len_and_friends`

                        // get distance code
                        let mut here;
                        loop {
                            let bits = self.bit_reader.bits(self.dist_table.bits) as usize;
                            here = self.dist_table_get(bits);
                            if here.bits <= self.bit_reader.bits_in_buffer() {
                                break;
                            }

                            pull_byte!(self);
                        }

                        if here.op & 0xf0 == 0 {
                            let last = here;

                            loop {
                                let bits = self.bit_reader.bits((last.bits + last.op) as usize);
                                here = self.dist_table_get(
                                    last.val as usize + ((bits as usize) >> last.bits),
                                );

                                if last.bits + here.bits <= self.bit_reader.bits_in_buffer() {
                                    break;
                                }

                                pull_byte!(self);
                            }

                            self.bit_reader.drop_bits(last.bits);
                            self.back += last.bits as usize;
                        }

                        self.bit_reader.drop_bits(here.bits);

                        if here.op & 64 != 0 {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid distance code\0");
                        }

                        self.offset = here.val as usize;

                        self.extra = (here.op & MAX_BITS) as usize;

                        break 'blk Mode::DistExt;
                    }
                    Mode::DistExt => {
                        // NOTE: this branch must be kept in sync with its counterpart in `len_and_friends`
                        let extra = self.extra;

                        if extra > 0 {
                            need_bits!(self, extra);
                            self.offset += self.bit_reader.bits(extra) as usize;
                            self.bit_reader.drop_bits(extra as u8);
                            self.back += extra;
                        }

                        if INFLATE_STRICT && self.offset > self.dmax {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid distance code too far back\0");
                        }

                        traceln!("inflate: distance {}", self.offset);

                        break 'blk Mode::Match;
                    }
                    Mode::Match => {
                        // NOTE: this branch must be kept in sync with its counterpart in `len_and_friends`

                        'match_: loop {
                            if self.writer.is_full() {
                                traceln!(
                                    "BufError: writer is full ({} bytes)",
                                    self.writer.capacity()
                                );
                                break 'label self.inflate_leave(ReturnCode::Ok);
                            }

                            let left = self.writer.remaining();
                            let copy = self.writer.len();

                            let copy = if self.offset > copy {
                                // copy from window to output

                                let mut copy = self.offset - copy;

                                if copy > self.window.have() {
                                    if self.flags.contains(Flags::SANE) {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid distance too far back\0");
                                    }

                                    // TODO INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
                                    panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
                                }

                                let wnext = self.window.next();
                                let wsize = self.window.size();

                                let from = if copy > wnext {
                                    copy -= wnext;
                                    wsize - copy
                                } else {
                                    wnext - copy
                                };

                                copy = Ord::min(copy, self.length);
                                copy = Ord::min(copy, left);

                                self.writer
                                    .extend_from_window(&self.window, from..from + copy);

                                copy
                            } else {
                                let copy = Ord::min(self.length, left);
                                self.writer.copy_match(self.offset, copy);

                                copy
                            };

                            self.length -= copy;

                            if self.length == 0 {
                                break 'blk Mode::Len;
                            } else {
                                // otherwise it seems to recurse?
                                continue 'match_;
                            }
                        }
                    }
                    Mode::Table => {
                        need_bits!(self, 14);
                        self.nlen = self.bit_reader.bits(5) as usize + 257;
                        self.bit_reader.drop_bits(5);
                        self.ndist = self.bit_reader.bits(5) as usize + 1;
                        self.bit_reader.drop_bits(5);
                        self.ncode = self.bit_reader.bits(4) as usize + 4;
                        self.bit_reader.drop_bits(4);

                        // TODO pkzit_bug_workaround
                        if self.nlen > 286 || self.ndist > 30 {
                            mode = Mode::Bad;
                            break 'label self.bad("too many length or distance symbols\0");
                        }

                        self.have = 0;

                        break 'blk Mode::LenLens;
                    }
                    Mode::LenLens => {
                        // permutation of code lengths ;
                        const ORDER: [u8; 19] = [
                            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                        ];

                        while self.have < self.ncode {
                            need_bits!(self, 3);
                            self.lens[usize::from(ORDER[self.have])] =
                                self.bit_reader.bits(3) as u16;
                            self.have += 1;
                            self.bit_reader.drop_bits(3);
                        }

                        while self.have < 19 {
                            self.lens[usize::from(ORDER[self.have])] = 0;
                            self.have += 1;
                        }

                        let InflateTable::Success { root, used } = inflate_table(
                            CodeType::Codes,
                            &self.lens[..19],
                            &mut self.codes_codes,
                            7,
                            &mut self.work,
                        ) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid code lengths set\0");
                        };

                        self.next = used;
                        self.len_table.codes = Codes::Codes;
                        self.len_table.bits = root;

                        self.have = 0;

                        break 'blk Mode::CodeLens;
                    }
                    Mode::CodeLens => {
                        while self.have < self.nlen + self.ndist {
                            let here = loop {
                                let bits = self.bit_reader.bits(self.len_table.bits);
                                let here = self.len_table_get(bits as usize);
                                if here.bits <= self.bit_reader.bits_in_buffer() {
                                    break here;
                                }

                                pull_byte!(self);
                            };

                            let here_bits = here.bits;

                            match here.val {
                                0..=15 => {
                                    self.bit_reader.drop_bits(here_bits);
                                    self.lens[self.have] = here.val;
                                    self.have += 1;
                                }
                                16 => {
                                    need_bits!(self, usize::from(here_bits) + 2);
                                    self.bit_reader.drop_bits(here_bits);
                                    if self.have == 0 {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    let len = self.lens[self.have - 1];
                                    let copy = 3 + self.bit_reader.bits(2) as usize;
                                    self.bit_reader.drop_bits(2);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(len);
                                    self.have += copy;
                                }
                                17 => {
                                    need_bits!(self, usize::from(here_bits) + 3);
                                    self.bit_reader.drop_bits(here_bits);
                                    let copy = 3 + self.bit_reader.bits(3) as usize;
                                    self.bit_reader.drop_bits(3);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(0);
                                    self.have += copy;
                                }
                                18.. => {
                                    need_bits!(self, usize::from(here_bits) + 7);
                                    self.bit_reader.drop_bits(here_bits);
                                    let copy = 11 + self.bit_reader.bits(7) as usize;
                                    self.bit_reader.drop_bits(7);

                                    if self.have + copy > self.nlen + self.ndist {
                                        mode = Mode::Bad;
                                        break 'label self.bad("invalid bit length repeat\0");
                                    }

                                    self.lens[self.have..][..copy].fill(0);
                                    self.have += copy;
                                }
                            }
                        }

                        // check for end-of-block code (better have one)
                        if self.lens[256] == 0 {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid code -- missing end-of-block\0");
                        }

                        // build code tables

                        let InflateTable::Success { root, used } = inflate_table(
                            CodeType::Lens,
                            &self.lens[..self.nlen],
                            &mut self.len_codes,
                            10,
                            &mut self.work,
                        ) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid literal/lengths set\0");
                        };

                        self.len_table.codes = Codes::Len;
                        self.len_table.bits = root;
                        self.next = used;

                        let InflateTable::Success { root, used } = inflate_table(
                            CodeType::Dists,
                            &self.lens[self.nlen..][..self.ndist],
                            &mut self.dist_codes,
                            9,
                            &mut self.work,
                        ) else {
                            mode = Mode::Bad;
                            break 'label self.bad("invalid distances set\0");
                        };

                        self.dist_table.bits = root;
                        self.dist_table.codes = Codes::Dist;
                        self.next += used;

                        mode = Mode::Len_;

                        if matches!(self.flush, InflateFlush::Trees) {
                            break 'label self.inflate_leave(ReturnCode::Ok);
                        }

                        break 'blk Mode::Len_;
                    }
                    Mode::Dict => {
                        if !self.flags.contains(Flags::HAVE_DICT) {
                            break 'label self.inflate_leave(ReturnCode::NeedDict);
                        }

                        self.checksum = crate::ADLER32_INITIAL_VALUE as _;

                        break 'blk Mode::Type;
                    }
                    Mode::DictId => {
                        need_bits!(self, 32);

                        self.checksum = zswap32(self.bit_reader.hold() as u32);

                        self.bit_reader.init_bits();

                        break 'blk Mode::Dict;
                    }
                    Mode::Done => {
                        // Inflate stream terminated properly.
                        break 'label ReturnCode::StreamEnd;
                    }
                    Mode::Bad => {
                        let msg = "repeated call with bad state\0";
                        #[cfg(all(feature = "std", test))]
                        dbg!(msg);
                        self.error_message = Some(msg);

                        break 'label ReturnCode::DataError;
                    }
                    Mode::Mem => {
                        break 'label ReturnCode::MemError;
                    }
                    Mode::Sync => {
                        break 'label ReturnCode::StreamError;
                    }
                    Mode::Length => {
                        // for gzip, last bytes contain LENGTH
                        if self.wrap != 0 && self.gzip_flags != 0 {
                            need_bits!(self, 32);
                            if (self.wrap & 0b100) != 0
                                && self.bit_reader.hold() as u32 != self.total as u32
                            {
                                mode = Mode::Bad;
                                break 'label self.bad("incorrect length check\0");
                            }

                            self.bit_reader.init_bits();
                        }

                        mode = Mode::Done;
                        // Inflate stream terminated properly.
                        break 'label ReturnCode::StreamEnd;
                    }
                };
            }
        };

        self.mode = mode;

        ret
    }

    fn bad(&mut self, msg: &'static str) -> ReturnCode {
        #[cfg(all(feature = "std", test))]
        dbg!(msg);
        self.error_message = Some(msg);
        self.inflate_leave(ReturnCode::DataError)
    }

    // NOTE: it is crucial for the internal bookkeeping that this is the only route for actually
    // leaving the inflate function call chain
    fn inflate_leave(&mut self, return_code: ReturnCode) -> ReturnCode {
        // actual logic is in `inflate` itself
        return_code
    }

    /// Stored in the `z_stream.data_type` field
    fn decoding_state(&self) -> i32 {
        let bit_reader_bits = self.bit_reader.bits_in_buffer() as i32;
        debug_assert!(bit_reader_bits < 64);

        let last = if self.flags.contains(Flags::IS_LAST_BLOCK) {
            64
        } else {
            0
        };

        let mode = match self.mode {
            Mode::Type => 128,
            Mode::Len_ | Mode::CopyBlock => 256,
            _ => 0,
        };

        bit_reader_bits | last | mode
    }
}

/// # Safety
///
/// `state.bit_reader` must have at least 15 bytes available to read, as
/// indicated by `state.bit_reader.bytes_remaining() >= 15`
unsafe fn inflate_fast_help(state: &mut State, start: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if crate::cpu_features::is_enabled_avx2_and_bmi2() {
        // SAFETY: we've verified the target features and the caller ensured enough bytes_remaining
        return unsafe { inflate_fast_help_avx2(state, start) };
    }

    // SAFETY: The caller ensured enough bytes_remaining
    unsafe { inflate_fast_help_vanilla(state, start) };
}

/// # Safety
///
/// `state.bit_reader` must have at least 15 bytes available to read, as
/// indicated by `state.bit_reader.bytes_remaining() >= 15`
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
#[target_feature(enable = "bmi1")]
unsafe fn inflate_fast_help_avx2(state: &mut State, start: usize) {
    // SAFETY: `bytes_remaining` checked by our caller
    unsafe { inflate_fast_help_impl::<{ CpuFeatures::AVX2 }>(state, start) };
}

/// # Safety
///
/// `state.bit_reader` must have at least 15 bytes available to read, as
/// indicated by `state.bit_reader.bytes_remaining() >= 15`
unsafe fn inflate_fast_help_vanilla(state: &mut State, start: usize) {
    // SAFETY: `bytes_remaining` checked by our caller
    unsafe { inflate_fast_help_impl::<{ CpuFeatures::NONE }>(state, start) };
}

/// # Safety
///
/// `state.bit_reader` must have at least 15 bytes available to read, as
/// indicated by `state.bit_reader.bytes_remaining() >= 15`
#[inline(always)]
unsafe fn inflate_fast_help_impl<const FEATURES: usize>(state: &mut State, _start: usize) {
    let mut bit_reader = BitReader::new(&[]);
    core::mem::swap(&mut bit_reader, &mut state.bit_reader);
    debug_assert!(bit_reader.bytes_remaining() >= 15);

    let mut writer = Writer::new(&mut []);
    core::mem::swap(&mut writer, &mut state.writer);

    let lcode = state.len_table_ref();
    let dcode = state.dist_table_ref();

    // IDEA: use const generics for the bits here?
    let lmask = (1u64 << state.len_table.bits) - 1;
    let dmask = (1u64 << state.dist_table.bits) - 1;

    // TODO verify if this is relevant for us
    let extra_safe = false;

    let window_size = state.window.size();

    let mut bad = None;

    if bit_reader.bits_in_buffer() < 10 {
        debug_assert!(bit_reader.bytes_remaining() >= 15);
        // Safety: Caller ensured that bit_reader has >= 15 bytes available; refill only needs 8.
        unsafe { bit_reader.refill() };
    }
    // We had at least 15 bytes in the slice, plus whatever was in the buffer. After filling the
    // buffer from the slice, we now have at least 8 bytes remaining in the slice, plus a full buffer.
    debug_assert!(
        bit_reader.bytes_remaining() >= 8 && bit_reader.bytes_remaining_including_buffer() >= 15
    );

    'outer: loop {
        // This condition is ensured above for the first iteration of the `outer` loop. For
        // subsequent iterations, the loop continuation condition is
        // `bit_reader.bytes_remaining_including_buffer() > 15`. And because the buffer
        // contributes at most 7 bytes to the result of bit_reader.bytes_remaining_including_buffer(),
        // that means that the slice contains at least 8 bytes.
        debug_assert!(
            bit_reader.bytes_remaining() >= 8
                && bit_reader.bytes_remaining_including_buffer() >= 15
        );

        let mut here = {
            let bits = bit_reader.bits_in_buffer();
            let hold = bit_reader.hold();

            // Safety: As described in the comments for the debug_assert at the start of
            // the `outer` loop, it is guaranteed that `bit_reader.bytes_remaining() >= 8` here,
            // which satisfies the safety precondition for `refill`. And, because the total
            // number of bytes in `bit_reader`'s buffer plus its slice is at least 15, and
            // `refill` moves at most 7 bytes from the slice to the buffer, the slice will still
            // contain at least 8 bytes after this `refill` call.
            unsafe { bit_reader.refill() };
            // After the refill, there will be at least 8 bytes left in the bit_reader's slice.
            debug_assert!(bit_reader.bytes_remaining() >= 8);

            // in most cases, the read can be interleaved with the logic
            // based on benchmarks this matters in practice. wild.
            if bits as usize >= state.len_table.bits {
                lcode[(hold & lmask) as usize]
            } else {
                lcode[(bit_reader.hold() & lmask) as usize]
            }
        };

        if here.op == 0 {
            writer.push(here.val as u8);
            bit_reader.drop_bits(here.bits);
            here = lcode[(bit_reader.hold() & lmask) as usize];

            if here.op == 0 {
                writer.push(here.val as u8);
                bit_reader.drop_bits(here.bits);
                here = lcode[(bit_reader.hold() & lmask) as usize];
            }
        }

        'dolen: loop {
            bit_reader.drop_bits(here.bits);
            let op = here.op;

            if op == 0 {
                writer.push(here.val as u8);
            } else if op & 16 != 0 {
                let op = op & MAX_BITS;
                let mut len = here.val + bit_reader.bits(op as usize) as u16;
                bit_reader.drop_bits(op);

                here = dcode[(bit_reader.hold() & dmask) as usize];

                // we have two fast-path loads: 10+10 + 15+5 = 40,
                // but we may need to refill here in the worst case
                if bit_reader.bits_in_buffer() < MAX_BITS + MAX_DIST_EXTRA_BITS {
                    debug_assert!(bit_reader.bytes_remaining() >= 8);
                    // Safety: On the first iteration of the `dolen` loop, we can rely on the
                    // invariant documented for the previous `refill` call above: after that
                    // operation, `bit_reader.bytes_remining >= 8`, which satisfies the safety
                    // precondition for this call. For subsequent iterations, this invariant
                    // remains true because nothing else within the `dolen` loop consumes data
                    // from the slice.
                    unsafe { bit_reader.refill() };
                }

                'dodist: loop {
                    bit_reader.drop_bits(here.bits);
                    let op = here.op;

                    if op & 16 != 0 {
                        let op = op & MAX_BITS;
                        let dist = here.val + bit_reader.bits(op as usize) as u16;

                        if INFLATE_STRICT && dist as usize > state.dmax {
                            bad = Some("invalid distance too far back\0");
                            state.mode = Mode::Bad;
                            break 'outer;
                        }

                        bit_reader.drop_bits(op);

                        // max distance in output
                        let written = writer.len();

                        if dist as usize > written {
                            // copy fropm the window
                            if (dist as usize - written) > state.window.have() {
                                if state.flags.contains(Flags::SANE) {
                                    bad = Some("invalid distance too far back\0");
                                    state.mode = Mode::Bad;
                                    break 'outer;
                                }

                                panic!("INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR")
                            }

                            let mut op = dist as usize - written;
                            let mut from;

                            let window_next = state.window.next();

                            if window_next == 0 {
                                // This case is hit when the window has just wrapped around
                                // by logic in `Window::extend`. It is special-cased because
                                // apparently this is quite common.
                                //
                                // the match is at the end of the window, even though the next
                                // position has now wrapped around.
                                from = window_size - op;
                            } else if window_next >= op {
                                // the standard case: a contiguous copy from the window, no wrapping
                                from = window_next - op;
                            } else {
                                // This case is hit when the window has recently wrapped around
                                // by logic in `Window::extend`.
                                //
                                // The match is (partially) at the end of the window
                                op -= window_next;
                                from = window_size - op;

                                if op < len as usize {
                                    // This case is hit when part of the match is at the end of the
                                    // window, and part of it has wrapped around to the start. Copy
                                    // the end section here, the start section will be copied below.
                                    len -= op as u16;
                                    writer.extend_from_window_with_features::<FEATURES>(
                                        &state.window,
                                        from..from + op,
                                    );
                                    from = 0;
                                    op = window_next;
                                }
                            }

                            let copy = Ord::min(op, len as usize);
                            writer.extend_from_window_with_features::<FEATURES>(
                                &state.window,
                                from..from + copy,
                            );

                            if op < len as usize {
                                // here we need some bytes from the output itself
                                writer.copy_match_with_features::<FEATURES>(
                                    dist as usize,
                                    len as usize - op,
                                );
                            }
                        } else if extra_safe {
                            todo!()
                        } else {
                            writer.copy_match_with_features::<FEATURES>(dist as usize, len as usize)
                        }
                    } else if (op & 64) == 0 {
                        // 2nd level distance code
                        here = dcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                        continue 'dodist;
                    } else {
                        bad = Some("invalid distance code\0");
                        state.mode = Mode::Bad;
                        break 'outer;
                    }

                    break 'dodist;
                }
            } else if (op & 64) == 0 {
                // 2nd level length code
                here = lcode[(here.val + bit_reader.bits(op as usize) as u16) as usize];
                continue 'dolen;
            } else if op & 32 != 0 {
                // end of block
                state.mode = Mode::Type;
                break 'outer;
            } else {
                bad = Some("invalid literal/length code\0");
                state.mode = Mode::Bad;
                break 'outer;
            }

            break 'dolen;
        }

        // For normal `inflate`, include the bits in the bit_reader buffer in the count of available bytes.
        let remaining = bit_reader.bytes_remaining_including_buffer();
        if remaining >= INFLATE_FAST_MIN_HAVE && writer.remaining() >= INFLATE_FAST_MIN_LEFT {
            continue;
        }

        break 'outer;
    }

    // return unused bytes (on entry, bits < 8, so in won't go too far back)
    bit_reader.return_unused_bytes();

    state.bit_reader = bit_reader;
    state.writer = writer;

    if let Some(error_message) = bad {
        debug_assert!(matches!(state.mode, Mode::Bad));
        state.bad(error_message);
    }
}

pub fn prime(stream: &mut InflateStream, bits: i32, value: i32) -> ReturnCode {
    if bits == 0 {
        /* fall through */
    } else if bits < 0 {
        stream.state.bit_reader.init_bits();
    } else if bits > 16 || stream.state.bit_reader.bits_in_buffer() + bits as u8 > 32 {
        return ReturnCode::StreamError;
    } else {
        stream.state.bit_reader.prime(bits as u8, value as u64);
    }

    ReturnCode::Ok
}

struct InflateAllocOffsets {
    total_size: usize,
    state_pos: usize,
    window_pos: usize,
}

impl InflateAllocOffsets {
    fn new() -> Self {
        use core::mem::size_of;

        // 64B padding for SIMD operations. This allows unaligned operations (up to 512-bit) to run
        // off the end of the object without issue.
        const WINDOW_PAD_SIZE: usize = 64;

        // 64B alignment of individual items in the alloc.
        // Note that changing this also requires changes in 'init' and 'copy'.
        const ALIGN_SIZE: usize = 64;
        let mut curr_size = 0usize;

        /* Define sizes */
        let state_size = size_of::<State>();
        let window_size = (1 << MAX_WBITS) + WINDOW_PAD_SIZE;

        /* Calculate relative buffer positions and paddings */
        let state_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = state_pos + state_size;

        let window_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = window_pos + window_size;

        /* Add ALIGN_SIZE-1 to allow alignment (done in the 'init' and 'copy' functions), and round
         * size of buffer up to next multiple of ALIGN_SIZE */
        let total_size = (curr_size + (ALIGN_SIZE - 1)).next_multiple_of(ALIGN_SIZE);

        Self {
            total_size,
            state_pos,
            window_pos,
        }
    }
}

/// Configuration for decompresssion.
///
/// Used with [`decompress_slice`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InflateConfig {
    pub window_bits: i32,
}

impl Default for InflateConfig {
    fn default() -> Self {
        Self {
            window_bits: DEF_WBITS,
        }
    }
}

/// Initialize the stream in an inflate state
pub fn init(stream: &mut z_stream, config: InflateConfig) -> ReturnCode {
    stream.msg = core::ptr::null_mut();

    // for safety we must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` configures the C allocator
    #[cfg(feature = "rust-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_rust_allocator()
    }

    #[cfg(feature = "c-allocator")]
    if stream.zalloc.is_none() || stream.zfree.is_none() {
        stream.configure_default_c_allocator()
    }

    if stream.zalloc.is_none() || stream.zfree.is_none() {
        return ReturnCode::StreamError;
    }

    let mut state = State::new(&[], Writer::new(&mut []));

    // TODO this can change depending on the used/supported SIMD instructions
    state.chunksize = 32;

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
        opaque: stream.opaque,
        _marker: PhantomData,
    };
    let allocs = InflateAllocOffsets::new();

    let Some(allocation_start) = alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let address = allocation_start.as_ptr() as usize;
    let align_offset = address.next_multiple_of(64) - address;
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    let window_allocation = unsafe { buf.add(allocs.window_pos) };
    let window = unsafe { Window::from_raw_parts(window_allocation, (1 << MAX_WBITS) + 64) };
    state.window = window;

    let state_allocation = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { state_allocation.write(state) };
    stream.state = state_allocation.cast::<internal_state>();

    // SAFETY: we've correctly initialized the stream to be an InflateStream
    if let Some(stream) = unsafe { InflateStream::from_stream_mut(stream) } {
        stream.state.allocation_start = allocation_start.as_ptr();
        stream.state.total_allocation_size = allocs.total_size;
        let ret = reset_with_config(stream, config);

        if ret != ReturnCode::Ok {
            end(stream);
        }

        ret
    } else {
        ReturnCode::StreamError
    }
}

pub fn reset_with_config(stream: &mut InflateStream, config: InflateConfig) -> ReturnCode {
    let mut window_bits = config.window_bits;
    let wrap;

    if window_bits < 0 {
        wrap = 0;

        if window_bits < -MAX_WBITS {
            return ReturnCode::StreamError;
        }

        window_bits = -window_bits;
    } else {
        wrap = (window_bits >> 4) + 5; // TODO wth?

        if window_bits < 48 {
            window_bits &= MAX_WBITS;
        }
    }

    if window_bits != 0 && !(MIN_WBITS..=MAX_WBITS).contains(&window_bits) {
        traceln!("invalid windowBits");
        return ReturnCode::StreamError;
    }

    stream.state.wrap = wrap as u8;
    stream.state.wbits = window_bits as _;

    reset(stream)
}

pub fn reset(stream: &mut InflateStream) -> ReturnCode {
    // reset the state of the window
    stream.state.window.clear();

    stream.state.error_message = None;

    reset_keep(stream)
}

pub fn reset_keep(stream: &mut InflateStream) -> ReturnCode {
    stream.total_in = 0;
    stream.total_out = 0;
    stream.state.total = 0;

    stream.msg = core::ptr::null_mut();

    let state = &mut stream.state;

    if state.wrap != 0 {
        // to support ill-conceived Java test suite
        stream.adler = (state.wrap & 1) as _;
    }

    state.mode = Mode::Head;
    state.checksum = crate::ADLER32_INITIAL_VALUE as u32;

    state.flags.update(Flags::IS_LAST_BLOCK, false);
    state.flags.update(Flags::HAVE_DICT, false);
    state.flags.update(Flags::SANE, true);
    state.gzip_flags = -1;
    state.dmax = 32768;
    state.head = None;
    state.bit_reader = BitReader::new(&[]);

    state.next = 0;
    state.len_table = Table::default();
    state.dist_table = Table::default();

    state.back = usize::MAX;

    ReturnCode::Ok
}

pub fn codes_used(stream: &InflateStream) -> usize {
    stream.state.next
}

pub unsafe fn inflate(stream: &mut InflateStream, flush: InflateFlush) -> ReturnCode {
    if stream.next_out.is_null() || (stream.next_in.is_null() && stream.avail_in != 0) {
        return ReturnCode::StreamError;
    }

    let state = &mut stream.state;

    // skip check
    if let Mode::Type = state.mode {
        state.mode = Mode::TypeDo;
    }

    state.flush = flush;

    unsafe {
        state
            .bit_reader
            .update_slice(stream.next_in, stream.avail_in as usize)
    };
    // Safety: `stream.next_out` is non-null and points to at least `stream.avail_out` bytes.
    state.writer = unsafe { Writer::new_uninit(stream.next_out.cast(), stream.avail_out as usize) };

    state.in_available = stream.avail_in as _;
    state.out_available = stream.avail_out as _;

    let err = state.dispatch();

    let in_read = state.bit_reader.as_ptr() as usize - stream.next_in as usize;
    let out_written = state.out_available - (state.writer.capacity() - state.writer.len());

    stream.total_in += in_read as z_size;
    state.total = state.total.wrapping_add(out_written);
    stream.total_out = state.total as _;

    stream.avail_in = state.bit_reader.bytes_remaining() as u32;
    stream.next_in = state.bit_reader.as_ptr() as *mut u8;

    stream.avail_out = (state.writer.capacity() - state.writer.len()) as u32;
    stream.next_out = state.writer.next_out() as *mut u8;

    stream.adler = state.checksum as z_checksum;

    let valid_mode = |mode| !matches!(mode, Mode::Bad | Mode::Mem | Mode::Sync);
    let not_done = |mode| {
        !matches!(
            mode,
            Mode::Check | Mode::Length | Mode::Bad | Mode::Mem | Mode::Sync
        )
    };

    let must_update_window = state.window.size() != 0
        || (out_written != 0
            && valid_mode(state.mode)
            && (not_done(state.mode) || !matches!(state.flush, InflateFlush::Finish)));

    let update_checksum = state.wrap & 4 != 0;

    if must_update_window {
        state.window.extend(
            &state.writer.filled()[..out_written],
            state.gzip_flags,
            update_checksum,
            &mut state.checksum,
            &mut state.crc_fold,
        );
    }

    if let Some(msg) = state.error_message {
        assert!(msg.ends_with('\0'));
        stream.msg = msg.as_ptr() as *mut u8 as *mut core::ffi::c_char;
    }

    stream.data_type = state.decoding_state();

    if ((in_read == 0 && out_written == 0) || flush == InflateFlush::Finish)
        && err == ReturnCode::Ok
    {
        ReturnCode::BufError
    } else {
        err
    }
}

fn syncsearch(mut got: usize, buf: &[u8]) -> (usize, usize) {
    let len = buf.len();
    let mut next = 0;

    while next < len && got < 4 {
        if buf[next] == if got < 2 { 0 } else { 0xff } {
            got += 1;
        } else if buf[next] != 0 {
            got = 0;
        } else {
            got = 4 - got;
        }
        next += 1;
    }

    (got, next)
}

pub fn sync(stream: &mut InflateStream) -> ReturnCode {
    let state = &mut stream.state;

    if stream.avail_in == 0 && state.bit_reader.bits_in_buffer() < 8 {
        return ReturnCode::BufError;
    }
    /* if first time, start search in bit buffer */
    if !matches!(state.mode, Mode::Sync) {
        state.mode = Mode::Sync;

        let (buf, len) = state.bit_reader.start_sync_search();

        (state.have, _) = syncsearch(0, &buf[..len]);
    }

    // search available input
    // SAFETY: user guarantees that pointer and length are valid.
    let slice = unsafe { core::slice::from_raw_parts(stream.next_in, stream.avail_in as usize) };

    let len;
    (state.have, len) = syncsearch(state.have, slice);
    // SAFETY: syncsearch() returns an index that is in-bounds of the slice.
    stream.next_in = unsafe { stream.next_in.add(len) };
    stream.avail_in -= len as u32;
    stream.total_in += len as z_size;

    /* return no joy or set up to restart inflate() on a new block */
    if state.have != 4 {
        return ReturnCode::DataError;
    }

    if state.gzip_flags == -1 {
        state.wrap = 0; /* if no header yet, treat as raw */
    } else {
        state.wrap &= !4; /* no point in computing a check value now */
    }

    let flags = state.gzip_flags;
    let total_in = stream.total_in;
    let total_out = stream.total_out;

    reset(stream);

    stream.total_in = total_in;
    stream.total_out = total_out;

    stream.state.gzip_flags = flags;
    stream.state.mode = Mode::Type;

    ReturnCode::Ok
}

/*
  Returns true if inflate is currently at the end of a block generated by
  Z_SYNC_FLUSH or Z_FULL_FLUSH. This function is used by one PPP
  implementation to provide an additional safety check. PPP uses
  Z_SYNC_FLUSH but removes the length bytes of the resulting empty stored
  block. When decompressing, PPP checks that at the end of input packet,
  inflate is waiting for these length bytes.
*/
pub fn sync_point(stream: &mut InflateStream) -> bool {
    matches!(stream.state.mode, Mode::Stored) && stream.state.bit_reader.bits_in_buffer() == 0
}

pub unsafe fn copy<'a>(
    dest: &mut MaybeUninit<InflateStream<'a>>,
    source: &InflateStream<'a>,
) -> ReturnCode {
    if source.next_out.is_null() || (source.next_in.is_null() && source.avail_in != 0) {
        return ReturnCode::StreamError;
    }

    // Safety: source and dest are both mutable references, so guaranteed not to overlap.
    // dest being a reference to maybe uninitialized memory makes a copy of 1 DeflateStream valid.
    unsafe { core::ptr::copy_nonoverlapping(source, dest.as_mut_ptr(), 1) };

    // Allocate space.
    let allocs = InflateAllocOffsets::new();
    debug_assert_eq!(allocs.total_size, source.state.total_allocation_size);

    let Some(allocation_start) = source.alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let address = allocation_start.as_ptr() as usize;
    let align_offset = address.next_multiple_of(64) - address;
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    let window_allocation = unsafe { buf.add(allocs.window_pos) };
    let window = unsafe {
        source
            .state
            .window
            .clone_to(window_allocation, (1 << MAX_WBITS) + 64)
    };

    let copy = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { core::ptr::copy_nonoverlapping(source.state, copy, 1) };

    let field_ptr = unsafe { core::ptr::addr_of_mut!((*copy).window) };
    unsafe { core::ptr::write(field_ptr, window) };

    let field_ptr = unsafe { core::ptr::addr_of_mut!((*copy).allocation_start) };
    unsafe { core::ptr::write(field_ptr, allocation_start.as_ptr()) };

    let field_ptr = unsafe { core::ptr::addr_of_mut!((*dest.as_mut_ptr()).state) };
    unsafe { core::ptr::write(field_ptr as *mut *mut State, copy) };

    ReturnCode::Ok
}

pub fn undermine(stream: &mut InflateStream, subvert: i32) -> ReturnCode {
    stream.state.flags.update(Flags::SANE, (!subvert) != 0);

    ReturnCode::Ok
}

/// Configures whether the checksum is calculated and checked.
pub fn validate(stream: &mut InflateStream, check: bool) -> ReturnCode {
    if check && stream.state.wrap != 0 {
        stream.state.wrap |= 0b100;
    } else {
        stream.state.wrap &= !0b100;
    }

    ReturnCode::Ok
}

pub fn mark(stream: &InflateStream) -> c_long {
    if stream.next_out.is_null() || (stream.next_in.is_null() && stream.avail_in != 0) {
        return c_long::MIN;
    }

    let state = &stream.state;

    let length = match state.mode {
        Mode::CopyBlock => state.length,
        Mode::Match => state.was - state.length,
        _ => 0,
    };

    (((state.back as c_long) as c_ulong) << 16) as c_long + length as c_long
}

pub fn set_dictionary(stream: &mut InflateStream, dictionary: &[u8]) -> ReturnCode {
    if stream.state.wrap != 0 && !matches!(stream.state.mode, Mode::Dict) {
        return ReturnCode::StreamError;
    }

    // check for correct dictionary identifier
    if matches!(stream.state.mode, Mode::Dict) {
        let dictid = adler32(1, dictionary);

        if dictid != stream.state.checksum {
            return ReturnCode::DataError;
        }
    }

    stream.state.window.extend(
        dictionary,
        stream.state.gzip_flags,
        false,
        &mut stream.state.checksum,
        &mut stream.state.crc_fold,
    );

    stream.state.flags.update(Flags::HAVE_DICT, true);

    ReturnCode::Ok
}

pub fn end<'a>(stream: &'a mut InflateStream<'_>) -> &'a mut z_stream {
    let alloc = stream.alloc;
    let allocation_start = stream.state.allocation_start;
    let total_allocation_size = stream.state.total_allocation_size;

    let mut window = Window::empty();
    core::mem::swap(&mut window, &mut stream.state.window);

    let stream = stream.as_z_stream_mut();
    let _ = core::mem::replace(&mut stream.state, core::ptr::null_mut());

    unsafe { alloc.deallocate(allocation_start, total_allocation_size) };

    stream
}

/// # Safety
///
/// The caller must guarantee:
///
/// * If `head` is `Some`:
///     - If `head.extra` is not NULL, it must be writable for at least `head.extra_max` bytes
///     - if `head.name` is not NULL, it must be writable for at least `head.name_max` bytes
///     - if `head.comment` is not NULL, it must be writable for at least `head.comm_max` bytes
pub unsafe fn get_header<'a>(
    stream: &mut InflateStream<'a>,
    head: Option<&'a mut gz_header>,
) -> ReturnCode {
    if (stream.state.wrap & 2) == 0 {
        return ReturnCode::StreamError;
    }

    stream.state.head = head.map(|head| {
        head.done = 0;
        head
    });
    ReturnCode::Ok
}

/// # Safety
///
/// The `dictionary` must have enough space for the dictionary.
pub unsafe fn get_dictionary(stream: &InflateStream<'_>, dictionary: *mut u8) -> usize {
    let whave = stream.state.window.have();
    let wnext = stream.state.window.next();

    if !dictionary.is_null() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                stream.state.window.as_ptr().add(wnext),
                dictionary,
                whave - wnext,
            );

            core::ptr::copy_nonoverlapping(
                stream.state.window.as_ptr(),
                dictionary.add(whave).sub(wnext).cast(),
                wnext,
            );
        }
    }

    stream.state.window.have()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncompress_buffer_overflow() {
        let mut output = [0; 1 << 13];
        let input = [
            72, 137, 58, 0, 3, 39, 255, 255, 255, 255, 255, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 184, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14,
            14, 14, 14, 14, 14, 63, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14, 255, 14, 103, 14,
            14, 14, 14, 14, 14, 61, 14, 255, 255, 63, 14, 14, 14, 14, 14, 14, 14, 14, 184, 14, 14,
            255, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 14, 14, 14, 14, 14, 14, 14, 14, 71,
            4, 137, 106,
        ];

        let config = InflateConfig { window_bits: 15 };

        let (_decompressed, err) = decompress_slice(&mut output, &input, config);
        assert_eq!(err, ReturnCode::DataError);
    }
}
