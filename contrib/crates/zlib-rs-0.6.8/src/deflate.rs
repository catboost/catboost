use core::{ffi::CStr, marker::PhantomData, mem::MaybeUninit, ops::ControlFlow, ptr::NonNull};

use crate::{
    adler32::adler32,
    allocate::Allocator,
    c_api::{gz_header, internal_state, z_checksum, z_stream},
    crc32::{crc32, Crc32Fold},
    trace,
    weak_slice::{WeakArrayMut, WeakSliceMut},
    DeflateFlush, ReturnCode, ADLER32_INITIAL_VALUE, CRC32_INITIAL_VALUE, MAX_WBITS, MIN_WBITS,
};

use self::{
    algorithm::CONFIGURATION_TABLE,
    hash_calc::{HashCalcVariant, RollHashCalc, StandardHashCalc},
    pending::Pending,
    sym_buf::SymBuf,
    trees_tbl::STATIC_LTREE,
    window::Window,
};

mod algorithm;
mod compare256;
mod hash_calc;
mod longest_match;
mod pending;
mod slide_hash;
mod sym_buf;
mod trees_tbl;
mod window;

// Position relative to the current window
pub(crate) type Pos = u16;

// SAFETY: This struct must have the same layout as [`z_stream`], so that casts and transmutations
// between the two can work without UB.
#[repr(C)]
pub struct DeflateStream<'a> {
    pub(crate) next_in: *mut crate::c_api::Bytef,
    pub(crate) avail_in: crate::c_api::uInt,
    pub(crate) total_in: crate::c_api::z_size,
    pub(crate) next_out: *mut crate::c_api::Bytef,
    pub(crate) avail_out: crate::c_api::uInt,
    pub(crate) total_out: crate::c_api::z_size,
    pub(crate) msg: *const core::ffi::c_char,
    pub(crate) state: &'a mut State<'a>,
    pub(crate) alloc: Allocator<'a>,
    pub(crate) data_type: core::ffi::c_int,
    pub(crate) adler: crate::c_api::z_checksum,
    pub(crate) reserved: crate::c_api::uLong,
}

unsafe impl Sync for DeflateStream<'_> {}
unsafe impl Send for DeflateStream<'_> {}

impl<'a> DeflateStream<'a> {
    // z_stream and DeflateStream must have the same layout. Do our best to check if this is true.
    // (imperfect check, but should catch most mistakes.)
    const _S: () = assert!(core::mem::size_of::<z_stream>() == core::mem::size_of::<Self>());
    const _A: () = assert!(core::mem::align_of::<z_stream>() == core::mem::align_of::<Self>());

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

        // SAFETY: DeflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<DeflateStream>().as_mut() }
    }

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

        // SAFETY: DeflateStream has an equivalent layout as z_stream
        unsafe { strm.cast::<DeflateStream>().as_ref() }
    }

    fn as_z_stream_mut(&mut self) -> &mut z_stream {
        // SAFETY: a valid &mut DeflateStream is also a valid &mut z_stream
        unsafe { &mut *(self as *mut DeflateStream as *mut z_stream) }
    }

    pub fn pending(&self) -> (usize, u8) {
        (
            self.state.bit_writer.pending.pending,
            self.state.bit_writer.bits_valid,
        )
    }

    /// Last number of used bits when going to a byte boundary.
    pub fn bits_used(&self) -> u8 {
        self.state.bit_writer.bits_used
    }

    pub fn new(config: DeflateConfig) -> Self {
        let mut inner = crate::c_api::z_stream::default();

        let ret = crate::deflate::init(&mut inner, config);
        assert_eq!(ret, ReturnCode::Ok);

        unsafe { core::mem::transmute(inner) }
    }
}

/// number of elements in hash table
pub(crate) const HASH_SIZE: usize = 65536;
/// log2(HASH_SIZE)
const HASH_BITS: usize = 16;

/// Maximum value for memLevel in deflateInit2
const MAX_MEM_LEVEL: i32 = 9;
pub const DEF_MEM_LEVEL: i32 = if MAX_MEM_LEVEL > 8 { 8 } else { MAX_MEM_LEVEL };

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub enum Method {
    #[default]
    Deflated = 8,
}

impl TryFrom<i32> for Method {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Deflated),
            _ => Err(()),
        }
    }
}

/// Configuration for compression.
///
/// Used with [`compress_slice`].
///
/// In most cases only the compression level is relevant. We provide three profiles:
///
/// - [`DeflateConfig::best_speed`] provides the fastest compression (at the cost of compression
///   quality)
/// - [`DeflateConfig::default`] tries to find a happy middle
/// - [`DeflateConfig::best_compression`] provides the best compression (at the cost of longer
///   runtime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub struct DeflateConfig {
    pub level: i32,
    pub method: Method,
    pub window_bits: i32,
    pub mem_level: i32,
    pub strategy: Strategy,
}

#[cfg(any(test, feature = "__internal-test"))]
impl quickcheck::Arbitrary for DeflateConfig {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let mem_levels: Vec<_> = (1..=9).collect();
        let levels: Vec<_> = (0..=9).collect();

        let mut window_bits = Vec::new();
        window_bits.extend(9..=15); // zlib
        window_bits.extend(9 + 16..=15 + 16); // gzip
        window_bits.extend(-15..=-9); // raw

        Self {
            level: *g.choose(&levels).unwrap(),
            method: Method::Deflated,
            window_bits: *g.choose(&window_bits).unwrap(),
            mem_level: *g.choose(&mem_levels).unwrap(),
            strategy: *g
                .choose(&[
                    Strategy::Default,
                    Strategy::Filtered,
                    Strategy::HuffmanOnly,
                    Strategy::Rle,
                    Strategy::Fixed,
                ])
                .unwrap(),
        }
    }
}

impl DeflateConfig {
    pub fn new(level: i32) -> Self {
        Self {
            level,
            ..Self::default()
        }
    }

    /// Configure for the best compression (takes longer).
    pub fn best_compression() -> Self {
        Self::new(crate::c_api::Z_BEST_COMPRESSION)
    }

    /// Configure for the fastest compression (compresses less well).
    pub fn best_speed() -> Self {
        Self::new(crate::c_api::Z_BEST_SPEED)
    }
}

impl Default for DeflateConfig {
    fn default() -> Self {
        Self {
            level: crate::c_api::Z_DEFAULT_COMPRESSION,
            method: Method::Deflated,
            window_bits: MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        }
    }
}

pub fn init(stream: &mut z_stream, config: DeflateConfig) -> ReturnCode {
    let DeflateConfig {
        mut level,
        method: _,
        mut window_bits,
        mem_level,
        strategy,
    } = config;

    /* Todo: ignore strm->next_in if we use it as window */
    stream.msg = core::ptr::null_mut();

    // for safety we  must really make sure that alloc and free are consistent
    // this is a (slight) deviation from stock zlib. In this crate we pick the rust
    // allocator as the default, but `libz-rs-sys` always explicitly sets an allocator,
    // and can configure the C allocator
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

    if level == crate::c_api::Z_DEFAULT_COMPRESSION {
        level = 6;
    }

    let wrap = if window_bits < 0 {
        if window_bits < -MAX_WBITS {
            return ReturnCode::StreamError;
        }
        window_bits = -window_bits;

        0
    } else if window_bits > MAX_WBITS {
        window_bits -= 16;
        2
    } else {
        1
    };

    if (!(1..=MAX_MEM_LEVEL).contains(&mem_level))
        || !(MIN_WBITS..=MAX_WBITS).contains(&window_bits)
        || !(0..=9).contains(&level)
        || (window_bits == 8 && wrap != 1)
    {
        return ReturnCode::StreamError;
    }

    let window_bits = if window_bits == 8 {
        9 /* until 256-byte window bug fixed */
    } else {
        window_bits as usize
    };

    let alloc = Allocator {
        zalloc: stream.zalloc.unwrap(),
        zfree: stream.zfree.unwrap(),
        opaque: stream.opaque,
        _marker: PhantomData,
    };

    let lit_bufsize = 1 << (mem_level + 6); // 16K elements by default
    let allocs = DeflateAllocOffsets::new(window_bits, lit_bufsize);

    // FIXME: pointer methods on NonNull are stable since 1.80.0
    let Some(allocation_start) = alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let w_size = 1 << window_bits;
    let align_offset = (allocation_start.as_ptr() as usize).next_multiple_of(64)
        - (allocation_start.as_ptr() as usize);
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    let (window, prev, head, pending, sym_buf) = unsafe {
        let window_ptr: *mut u8 = buf.add(allocs.window_pos);
        window_ptr.write_bytes(0u8, 2 * w_size);
        let window = Window::from_raw_parts(window_ptr, window_bits);

        // FIXME: write_bytes is stable for NonNull since 1.80.0
        let prev_ptr = buf.add(allocs.prev_pos).cast::<Pos>();
        prev_ptr.write_bytes(0, w_size);
        let prev = WeakSliceMut::from_raw_parts_mut(prev_ptr, w_size);

        let head_ptr = buf.add(allocs.head_pos).cast::<[Pos; HASH_SIZE]>();
        // Zero out the full array. `write_bytes` will write `1 * size_of<[Pos; HASH_SIZE]>` bytes.
        head_ptr.write_bytes(0, 1);
        let head = WeakArrayMut::<Pos, HASH_SIZE>::from_ptr(head_ptr);

        let pending_ptr = buf.add(allocs.pending_pos).cast::<MaybeUninit<u8>>();
        let pending = Pending::from_raw_parts(pending_ptr, 4 * lit_bufsize);

        let sym_buf_ptr = buf.add(allocs.sym_buf_pos);
        let sym_buf = SymBuf::from_raw_parts(sym_buf_ptr, lit_bufsize);

        (window, prev, head, pending, sym_buf)
    };

    let state = State {
        status: Status::Init,

        // window
        w_size,

        // allocated values
        window,
        prev,
        head,
        bit_writer: BitWriter::from_pending(pending),

        //
        lit_bufsize,

        //
        sym_buf,

        //
        level: level as i8, // set to zero again for testing?
        strategy,

        // these fields are not set explicitly at this point
        last_flush: 0,
        wrap,
        strstart: 0,
        block_start: 0,
        block_open: 0,
        window_size: 0,
        insert: 0,
        matches: 0,
        opt_len: 0,
        static_len: 0,
        lookahead: 0,
        ins_h: 0,
        max_chain_length: 0,
        max_lazy_match: 0,
        good_match: 0,
        nice_match: 0,

        //
        l_desc: TreeDesc::EMPTY,
        d_desc: TreeDesc::EMPTY,
        bl_desc: TreeDesc::EMPTY,

        //
        crc_fold: Crc32Fold::new(),
        gzhead: None,
        gzindex: 0,

        //
        match_start: 0,
        prev_match: 0,
        match_available: false,
        prev_length: 0,

        allocation_start,
        total_allocation_size: allocs.total_size,

        // just provide a valid default; gets set properly later
        hash_calc_variant: HashCalcVariant::Standard,
        _cache_line_0: (),
        _cache_line_1: (),
        _cache_line_2: (),
        _cache_line_3: (),
        _padding_0: [0; 16],
    };

    let state_allocation = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { state_allocation.write(state) };

    stream.state = state_allocation.cast::<internal_state>();

    let Some(stream) = (unsafe { DeflateStream::from_stream_mut(stream) }) else {
        if cfg!(debug_assertions) {
            unreachable!("we should have initialized the stream properly");
        }
        return ReturnCode::StreamError;
    };

    reset(stream)
}

pub fn params(stream: &mut DeflateStream, level: i32, strategy: Strategy) -> ReturnCode {
    let level = if level == crate::c_api::Z_DEFAULT_COMPRESSION {
        6
    } else {
        level
    };

    if !(0..=9).contains(&level) {
        return ReturnCode::StreamError;
    }

    let level = level as i8;

    let func = CONFIGURATION_TABLE[stream.state.level as usize].func;

    let state = &mut stream.state;

    // FIXME: use fn_addr_eq when it's available in our MSRV. The comparison returning false here
    // is not functionally incorrect, but would be inconsistent with zlib-ng.
    #[allow(unpredictable_function_pointer_comparisons)]
    if (strategy != state.strategy || func != CONFIGURATION_TABLE[level as usize].func)
        && state.last_flush != -2
    {
        // Flush the last buffer.
        let err = deflate(stream, DeflateFlush::Block);
        if err == ReturnCode::StreamError {
            return err;
        }

        let state = &mut stream.state;

        if stream.avail_in != 0
            || ((state.strstart as isize - state.block_start) + state.lookahead as isize) != 0
        {
            return ReturnCode::BufError;
        }
    }

    let state = &mut stream.state;

    if state.level != level {
        if state.level == 0 && state.matches != 0 {
            if state.matches == 1 {
                self::slide_hash::slide_hash(state);
            } else {
                state.head.as_mut_slice().fill(0);
            }
            state.matches = 0;
        }

        lm_set_level(state, level);
    }

    state.strategy = strategy;

    ReturnCode::Ok
}

pub fn set_dictionary(stream: &mut DeflateStream, mut dictionary: &[u8]) -> ReturnCode {
    let state = &mut stream.state;

    let wrap = state.wrap;

    if wrap == 2 || (wrap == 1 && state.status != Status::Init) || state.lookahead != 0 {
        return ReturnCode::StreamError;
    }

    // when using zlib wrappers, compute Adler-32 for provided dictionary
    if wrap == 1 {
        stream.adler = adler32(stream.adler as u32, dictionary) as z_checksum;
    }

    // avoid computing Adler-32 in read_buf
    state.wrap = 0;

    // if dictionary would fill window, just replace the history
    if dictionary.len() >= state.window.capacity() {
        if wrap == 0 {
            // clear the hash table
            state.head.as_mut_slice().fill(0);

            state.strstart = 0;
            state.block_start = 0;
            state.insert = 0;
        } else {
            /* already empty otherwise */
        }

        // use the tail
        dictionary = &dictionary[dictionary.len() - state.w_size..];
    }

    // insert dictionary into window and hash
    let avail = stream.avail_in;
    let next = stream.next_in;
    stream.avail_in = dictionary.len() as _;
    stream.next_in = dictionary.as_ptr() as *mut u8;
    fill_window(stream);

    while stream.state.lookahead >= STD_MIN_MATCH {
        let str = stream.state.strstart;
        let n = stream.state.lookahead - (STD_MIN_MATCH - 1);
        stream.state.insert_string(str, n);
        stream.state.strstart = str + n;
        stream.state.lookahead = STD_MIN_MATCH - 1;
        fill_window(stream);
    }

    let state = &mut stream.state;

    state.strstart += state.lookahead;
    state.block_start = state.strstart as _;
    state.insert = state.lookahead;
    state.lookahead = 0;
    state.prev_length = 0;
    state.match_available = false;

    // restore the state
    stream.next_in = next;
    stream.avail_in = avail;
    state.wrap = wrap;

    ReturnCode::Ok
}

pub fn prime(stream: &mut DeflateStream, mut bits: i32, value: i32) -> ReturnCode {
    // our logic actually supports up to 32 bits.
    debug_assert!(bits <= 16, "zlib only supports up to 16 bits here");

    let mut value64 = value as u64;

    let state = &mut stream.state;

    if bits < 0
        || bits > BitWriter::BIT_BUF_SIZE as i32
        || bits > (core::mem::size_of_val(&value) << 3) as i32
    {
        return ReturnCode::BufError;
    }

    let mut put;

    loop {
        put = BitWriter::BIT_BUF_SIZE - state.bit_writer.bits_valid;
        let put = Ord::min(put as i32, bits);

        if state.bit_writer.bits_valid == 0 {
            state.bit_writer.bit_buffer = value64;
        } else {
            state.bit_writer.bit_buffer |=
                (value64 & ((1 << put) - 1)) << state.bit_writer.bits_valid;
        }

        state.bit_writer.bits_valid += put as u8;
        state.bit_writer.flush_bits();
        value64 >>= put;
        bits -= put;

        if bits == 0 {
            break;
        }
    }

    ReturnCode::Ok
}

pub fn copy<'a>(
    dest: &mut MaybeUninit<DeflateStream<'a>>,
    source: &mut DeflateStream<'a>,
) -> ReturnCode {
    let w_size = source.state.w_size;
    let window_bits = source.state.w_bits() as usize;
    let lit_bufsize = source.state.lit_bufsize;

    // SAFETY: source and dest are both mutable references, so guaranteed not to overlap.
    // dest being a reference to maybe uninitialized memory makes a copy of 1 DeflateStream valid.
    unsafe { core::ptr::copy_nonoverlapping(source, dest.as_mut_ptr(), 1) };

    let source_state = &source.state;
    let alloc = &source.alloc;

    let allocs = DeflateAllocOffsets::new(window_bits, lit_bufsize);

    let Some(allocation_start) = alloc.allocate_slice_raw::<u8>(allocs.total_size) else {
        return ReturnCode::MemError;
    };

    let align_offset = (allocation_start.as_ptr() as usize).next_multiple_of(64)
        - (allocation_start.as_ptr() as usize);
    let buf = unsafe { allocation_start.as_ptr().add(align_offset) };

    let (window, prev, head, pending, sym_buf) = unsafe {
        let window_ptr: *mut u8 = buf.add(allocs.window_pos);
        window_ptr
            .copy_from_nonoverlapping(source_state.window.as_ptr(), source_state.window.capacity());
        let window = Window::from_raw_parts(window_ptr, window_bits);

        // FIXME: write_bytes is stable for NonNull since 1.80.0
        let prev_ptr = buf.add(allocs.prev_pos).cast::<Pos>();
        prev_ptr.copy_from_nonoverlapping(source_state.prev.as_ptr(), source_state.prev.len());
        let prev = WeakSliceMut::from_raw_parts_mut(prev_ptr, w_size);

        // zero out head's first element
        let head_ptr = buf.add(allocs.head_pos).cast::<[Pos; HASH_SIZE]>();
        head_ptr.copy_from_nonoverlapping(source_state.head.as_ptr(), 1);
        let head = WeakArrayMut::<Pos, HASH_SIZE>::from_ptr(head_ptr);

        let pending_ptr = buf.add(allocs.pending_pos);
        let pending = source_state.bit_writer.pending.clone_to(pending_ptr);

        let sym_buf_ptr = buf.add(allocs.sym_buf_pos);
        let sym_buf = source_state.sym_buf.clone_to(sym_buf_ptr);

        (window, prev, head, pending, sym_buf)
    };

    let mut bit_writer = BitWriter::from_pending(pending);
    bit_writer.bit_buffer = source_state.bit_writer.bit_buffer;
    bit_writer.bits_valid = source_state.bit_writer.bits_valid;
    bit_writer.bits_used = source_state.bit_writer.bits_used;

    let dest_state = State {
        status: source_state.status,
        bit_writer,
        last_flush: source_state.last_flush,
        wrap: source_state.wrap,
        strategy: source_state.strategy,
        level: source_state.level,
        good_match: source_state.good_match,
        nice_match: source_state.nice_match,
        l_desc: source_state.l_desc.clone(),
        d_desc: source_state.d_desc.clone(),
        bl_desc: source_state.bl_desc.clone(),
        prev_match: source_state.prev_match,
        match_available: source_state.match_available,
        strstart: source_state.strstart,
        match_start: source_state.match_start,
        prev_length: source_state.prev_length,
        max_chain_length: source_state.max_chain_length,
        max_lazy_match: source_state.max_lazy_match,
        block_start: source_state.block_start,
        block_open: source_state.block_open,
        window,
        sym_buf,
        lit_bufsize: source_state.lit_bufsize,
        window_size: source_state.window_size,
        matches: source_state.matches,
        opt_len: source_state.opt_len,
        static_len: source_state.static_len,
        insert: source_state.insert,
        w_size: source_state.w_size,
        lookahead: source_state.lookahead,
        prev,
        head,
        ins_h: source_state.ins_h,
        hash_calc_variant: source_state.hash_calc_variant,
        crc_fold: source_state.crc_fold,
        gzhead: None,
        gzindex: source_state.gzindex,
        allocation_start,
        total_allocation_size: allocs.total_size,
        _cache_line_0: (),
        _cache_line_1: (),
        _cache_line_2: (),
        _cache_line_3: (),
        _padding_0: source_state._padding_0,
    };

    // write the cloned state into state_ptr
    let state_allocation = unsafe { buf.add(allocs.state_pos).cast::<State>() };
    unsafe { state_allocation.write(dest_state) }; // FIXME: write is stable for NonNull since 1.80.0

    // insert the state_ptr into `dest`
    let field_ptr = unsafe { core::ptr::addr_of_mut!((*dest.as_mut_ptr()).state) };
    unsafe { core::ptr::write(field_ptr as *mut *mut State, state_allocation) };

    // update the gzhead field (it contains a mutable reference so we need to be careful
    let field_ptr = unsafe { core::ptr::addr_of_mut!((*dest.as_mut_ptr()).state.gzhead) };
    unsafe { core::ptr::copy(&source_state.gzhead, field_ptr, 1) };

    ReturnCode::Ok
}

/// # Returns
///
/// - Err when deflate is not done. A common cause is insufficient output space
/// - Ok otherwise
pub fn end<'a>(stream: &'a mut DeflateStream) -> Result<&'a mut z_stream, &'a mut z_stream> {
    let status = stream.state.status;
    let allocation_start = stream.state.allocation_start;
    let total_allocation_size = stream.state.total_allocation_size;
    let alloc = stream.alloc;

    let stream = stream.as_z_stream_mut();
    let _ = core::mem::replace(&mut stream.state, core::ptr::null_mut());

    unsafe { alloc.deallocate(allocation_start.as_ptr(), total_allocation_size) };

    match status {
        Status::Busy => Err(stream),
        _ => Ok(stream),
    }
}

pub fn reset(stream: &mut DeflateStream) -> ReturnCode {
    let ret = reset_keep(stream);

    if ret == ReturnCode::Ok {
        lm_init(stream.state);
    }

    ret
}

pub fn reset_keep(stream: &mut DeflateStream) -> ReturnCode {
    stream.total_in = 0;
    stream.total_out = 0;
    stream.msg = core::ptr::null_mut();
    stream.data_type = crate::c_api::Z_UNKNOWN;

    let state = &mut stream.state;

    state.bit_writer.pending.reset_keep();

    // can be made negative by deflate(..., Z_FINISH);
    state.wrap = state.wrap.abs();

    state.status = match state.wrap {
        2 => Status::GZip,
        _ => Status::Init,
    };

    stream.adler = match state.wrap {
        2 => {
            state.crc_fold = Crc32Fold::new();
            CRC32_INITIAL_VALUE as _
        }
        _ => ADLER32_INITIAL_VALUE as _,
    };

    state.last_flush = -2;

    state.zng_tr_init();

    ReturnCode::Ok
}

fn lm_init(state: &mut State) {
    state.window_size = 2 * state.w_size;

    // zlib uses CLEAR_HASH here
    state.head.as_mut_slice().fill(0);

    // Set the default configuration parameters:
    lm_set_level(state, state.level);

    state.strstart = 0;
    state.block_start = 0;
    state.lookahead = 0;
    state.insert = 0;
    state.prev_length = 0;
    state.match_available = false;
    state.match_start = 0;
    state.ins_h = 0;
}

fn lm_set_level(state: &mut State, level: i8) {
    state.max_lazy_match = CONFIGURATION_TABLE[level as usize].max_lazy;
    state.good_match = CONFIGURATION_TABLE[level as usize].good_length;
    state.nice_match = CONFIGURATION_TABLE[level as usize].nice_length;
    state.max_chain_length = CONFIGURATION_TABLE[level as usize].max_chain;

    state.hash_calc_variant = HashCalcVariant::for_max_chain_length(state.max_chain_length);
    state.level = level;
}

pub fn tune(
    stream: &mut DeflateStream,
    good_length: usize,
    max_lazy: usize,
    nice_length: usize,
    max_chain: usize,
) -> ReturnCode {
    stream.state.good_match = good_length as u16;
    stream.state.max_lazy_match = max_lazy as u16;
    stream.state.nice_match = nice_length as u16;
    stream.state.max_chain_length = max_chain as u16;

    ReturnCode::Ok
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Value {
    a: u16,
    b: u16,
}

impl Value {
    pub(crate) const fn new(a: u16, b: u16) -> Self {
        Self { a, b }
    }

    pub(crate) fn freq_mut(&mut self) -> &mut u16 {
        &mut self.a
    }

    pub(crate) fn code_mut(&mut self) -> &mut u16 {
        &mut self.a
    }

    pub(crate) fn dad_mut(&mut self) -> &mut u16 {
        &mut self.b
    }

    pub(crate) fn len_mut(&mut self) -> &mut u16 {
        &mut self.b
    }

    #[inline(always)]
    pub(crate) const fn freq(self) -> u16 {
        self.a
    }

    #[inline(always)]
    pub(crate) const fn code(self) -> u16 {
        self.a
    }

    #[inline(always)]
    pub(crate) const fn dad(self) -> u16 {
        self.b
    }

    #[inline(always)]
    pub(crate) const fn len(self) -> u16 {
        self.b
    }
}

/// number of length codes, not counting the special END_BLOCK code
pub(crate) const LENGTH_CODES: usize = 29;

/// number of literal bytes 0..255
const LITERALS: usize = 256;

/// number of Literal or Length codes, including the END_BLOCK code
pub(crate) const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;

/// number of distance codes
pub(crate) const D_CODES: usize = 30;

/// number of codes used to transfer the bit lengths
const BL_CODES: usize = 19;

/// maximum heap size
const HEAP_SIZE: usize = 2 * L_CODES + 1;

/// all codes must not exceed MAX_BITS bits
const MAX_BITS: usize = 15;

/// Bit length codes must not exceed MAX_BL_BITS bits
const MAX_BL_BITS: usize = 7;

pub(crate) const DIST_CODE_LEN: usize = 512;

struct BitWriter<'a> {
    pub(crate) pending: Pending<'a>, // output still pending

    /// Output buffer. bits are inserted starting at the bottom (least significant bits).
    pub(crate) bit_buffer: u64,

    /// Number of valid bits in bit_buffer. All bits above the last valid bit are always zero.
    pub(crate) bits_valid: u8,

    /// Last number of used bits when going to a byte boundary.
    pub(crate) bits_used: u8,

    /// total bit length of compressed file (NOTE: zlib-ng uses a 32-bit integer here)
    #[cfg(feature = "ZLIB_DEBUG")]
    compressed_len: usize,
    /// bit length of compressed data sent (NOTE: zlib-ng uses a 32-bit integer here)
    #[cfg(feature = "ZLIB_DEBUG")]
    bits_sent: usize,
}

#[inline]
const fn encode_len(ltree: &[Value], lc: u8) -> (u64, usize) {
    let mut lc = lc as usize;

    /* Send the length code, len is the match length - STD_MIN_MATCH */
    let code = self::trees_tbl::LENGTH_CODE[lc] as usize;
    let c = code + LITERALS + 1;
    assert!(c < L_CODES, "bad l_code");
    // send_code_trace(s, c);

    let lnode = ltree[c];
    let mut match_bits: u64 = lnode.code() as u64;
    let mut match_bits_len = lnode.len() as usize;
    let extra = StaticTreeDesc::EXTRA_LBITS[code] as usize;
    if extra != 0 {
        lc -= self::trees_tbl::BASE_LENGTH[code] as usize;
        match_bits |= (lc as u64) << match_bits_len;
        match_bits_len += extra;
    }

    (match_bits, match_bits_len)
}

#[inline]
const fn encode_dist(dtree: &[Value], mut dist: u16) -> (u64, usize) {
    dist -= 1; /* dist is now the match distance - 1 */
    let code = State::d_code(dist as usize) as usize;
    assert!(code < D_CODES, "bad d_code");
    // send_code_trace(s, code);

    /* Send the distance code */
    let dnode = dtree[code];
    let mut match_bits = dnode.code() as u64;
    let mut match_bits_len = dnode.len() as usize;
    let extra = StaticTreeDesc::EXTRA_DBITS[code] as usize;
    if extra != 0 {
        dist -= self::trees_tbl::BASE_DIST[code];
        match_bits |= (dist as u64) << match_bits_len;
        match_bits_len += extra;
    }

    (match_bits, match_bits_len)
}

impl<'a> BitWriter<'a> {
    pub(crate) const BIT_BUF_SIZE: u8 = 64;

    fn from_pending(pending: Pending<'a>) -> Self {
        Self {
            pending,
            bit_buffer: 0,
            bits_valid: 0,
            bits_used: 0,

            #[cfg(feature = "ZLIB_DEBUG")]
            compressed_len: 0,
            #[cfg(feature = "ZLIB_DEBUG")]
            bits_sent: 0,
        }
    }

    fn flush_bits(&mut self) {
        debug_assert!(self.bits_valid <= 64);
        let removed = self.bits_valid.saturating_sub(7).next_multiple_of(8);
        let keep_bytes = self.bits_valid / 8; // can never divide by zero

        let src = &self.bit_buffer.to_le_bytes();
        self.pending.extend(&src[..keep_bytes as usize]);

        self.bits_valid -= removed;
        self.bit_buffer = self.bit_buffer.checked_shr(removed as u32).unwrap_or(0);
    }

    fn emit_align(&mut self) {
        debug_assert!(self.bits_valid <= 64);
        let keep_bytes = self.bits_valid.div_ceil(8);
        let src = &self.bit_buffer.to_le_bytes();
        self.pending.extend(&src[..keep_bytes as usize]);

        self.bits_used = match self.bits_valid {
            0 => 8,
            _ => ((self.bits_valid - 1) & 7) + 1,
        };
        self.bit_buffer = 0;
        self.bits_valid = 0;

        self.sent_bits_align();
    }

    fn send_bits_trace(&self, _value: u64, _len: u8) {
        trace!(" l {:>2} v {:>4x} ", _len, _value);
    }

    fn cmpr_bits_add(&mut self, _len: usize) {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.compressed_len += _len;
        }
    }

    fn cmpr_bits_align(&mut self) {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.compressed_len = self.compressed_len.next_multiple_of(8);
        }
    }

    fn sent_bits_add(&mut self, _len: usize) {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.bits_sent += _len;
        }
    }

    fn sent_bits_align(&mut self) {
        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.bits_sent = self.bits_sent.next_multiple_of(8);
        }
    }

    #[inline(always)]
    fn send_bits(&mut self, val: u64, len: u8) {
        debug_assert!(len <= 64);
        debug_assert!(self.bits_valid <= 64);

        let total_bits = len + self.bits_valid;

        self.send_bits_trace(val, len);
        self.sent_bits_add(len as usize);

        if total_bits < Self::BIT_BUF_SIZE {
            self.bit_buffer |= val << self.bits_valid;
            self.bits_valid = total_bits;
        } else {
            self.send_bits_overflow(val, total_bits);
        }
    }

    fn send_bits_overflow(&mut self, val: u64, total_bits: u8) {
        if self.bits_valid == Self::BIT_BUF_SIZE {
            self.pending.extend(&self.bit_buffer.to_le_bytes());
            self.bit_buffer = val;
            self.bits_valid = total_bits - Self::BIT_BUF_SIZE;
        } else {
            self.bit_buffer |= val << self.bits_valid;
            self.pending.extend(&self.bit_buffer.to_le_bytes());
            self.bit_buffer = val >> (Self::BIT_BUF_SIZE - self.bits_valid);
            self.bits_valid = total_bits - Self::BIT_BUF_SIZE;
        }
    }

    fn send_code(&mut self, code: usize, tree: &[Value]) {
        let node = tree[code];
        self.send_bits(node.code() as u64, node.len() as u8)
    }

    /// Send one empty static block to give enough lookahead for inflate.
    /// This takes 10 bits, of which 7 may remain in the bit buffer.
    pub fn align(&mut self) {
        self.emit_tree(BlockType::StaticTrees, false);
        self.emit_end_block(&STATIC_LTREE, false);
        self.flush_bits();
    }

    pub(crate) fn emit_tree(&mut self, block_type: BlockType, is_last_block: bool) {
        let header_bits = ((block_type as u64) << 1) | (is_last_block as u64);
        self.send_bits(header_bits, 3);
        trace!("\n--- Emit Tree: Last: {}\n", is_last_block as u8);
    }

    pub(crate) fn emit_end_block_and_align(&mut self, ltree: &[Value], is_last_block: bool) {
        self.emit_end_block(ltree, is_last_block);

        if is_last_block {
            self.emit_align();
        }
    }

    fn emit_end_block(&mut self, ltree: &[Value], _is_last_block: bool) {
        const END_BLOCK: usize = 256;
        self.send_code(END_BLOCK, ltree);

        trace!(
            "\n+++ Emit End Block: Last: {} Pending: {} Total Out: {}\n",
            _is_last_block as u8,
            self.pending.pending().len(),
            "<unknown>"
        );
    }

    pub(crate) fn emit_lit(&mut self, ltree: &[Value], c: u8) -> u16 {
        self.send_code(c as usize, ltree);

        #[cfg(feature = "ZLIB_DEBUG")]
        if let Some(c) = char::from_u32(c as u32) {
            if isgraph(c as u8) {
                trace!(" '{}' ", c);
            }
        }

        ltree[c as usize].len()
    }

    pub(crate) fn emit_dist(
        &mut self,
        ltree: &[Value],
        dtree: &[Value],
        lc: u8,
        dist: u16,
    ) -> usize {
        let (mut match_bits, mut match_bits_len) = encode_len(ltree, lc);

        let (dist_match_bits, dist_match_bits_len) = encode_dist(dtree, dist);

        match_bits |= dist_match_bits << match_bits_len;
        match_bits_len += dist_match_bits_len;

        self.send_bits(match_bits, match_bits_len as u8);

        match_bits_len
    }

    pub(crate) fn emit_dist_static(&mut self, lc: u8, dist: u16) -> usize {
        let precomputed_len = trees_tbl::STATIC_LTREE_ENCODINGS[lc as usize];
        let mut match_bits = precomputed_len.code() as u64;
        let mut match_bits_len = precomputed_len.len() as usize;

        let dtree = self::trees_tbl::STATIC_DTREE.as_slice();
        let (dist_match_bits, dist_match_bits_len) = encode_dist(dtree, dist);

        match_bits |= dist_match_bits << match_bits_len;
        match_bits_len += dist_match_bits_len;

        self.send_bits(match_bits, match_bits_len as u8);

        match_bits_len
    }

    fn compress_block_help(&mut self, sym_buf: &SymBuf, ltree: &[Value], dtree: &[Value]) {
        for (dist, lc) in sym_buf.iter() {
            match dist {
                0 => self.emit_lit(ltree, lc) as usize,
                _ => self.emit_dist(ltree, dtree, lc, dist),
            };
        }

        self.emit_end_block(ltree, false)
    }

    fn send_tree(&mut self, tree: &[Value], bl_tree: &[Value], max_code: usize) {
        /* tree: the tree to be scanned */
        /* max_code and its largest code of non zero frequency */
        let mut prevlen: isize = -1; /* last emitted length */
        let mut curlen; /* length of current code */
        let mut nextlen = tree[0].len(); /* length of next code */
        let mut count = 0; /* repeat count of the current code */
        let mut max_count = 7; /* max repeat count */
        let mut min_count = 4; /* min repeat count */

        /* tree[max_code+1].Len = -1; */
        /* guard already set */
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }

        for n in 0..=max_code {
            curlen = nextlen;
            nextlen = tree[n + 1].len();
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                loop {
                    self.send_code(curlen as usize, bl_tree);

                    count -= 1;
                    if count == 0 {
                        break;
                    }
                }
            } else if curlen != 0 {
                if curlen as isize != prevlen {
                    self.send_code(curlen as usize, bl_tree);
                    count -= 1;
                }
                assert!((3..=6).contains(&count), " 3_6?");
                self.send_code(REP_3_6, bl_tree);
                self.send_bits(count - 3, 2);
            } else if count <= 10 {
                self.send_code(REPZ_3_10, bl_tree);
                self.send_bits(count - 3, 3);
            } else {
                self.send_code(REPZ_11_138, bl_tree);
                self.send_bits(count - 11, 7);
            }

            count = 0;
            prevlen = curlen as isize;

            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }
}

#[repr(C, align(64))]
pub(crate) struct State<'a> {
    status: Status,

    last_flush: i8, /* value of flush param for previous deflate call */

    pub(crate) wrap: i8, /* bit 0 true for zlib, bit 1 true for gzip */

    pub(crate) strategy: Strategy,
    pub(crate) level: i8,

    /// Whether or not a block is currently open for the QUICK deflation scheme.
    /// 0 if the block is closed, 1 if there is an active block, or 2 if there
    /// is an active block and it is the last block.
    pub(crate) block_open: u8,

    pub(crate) hash_calc_variant: HashCalcVariant,

    pub(crate) match_available: bool, /* set if previous match exists */

    /// Use a faster search when the previous match is longer than this
    pub(crate) good_match: u16,

    /// Stop searching when current match exceeds this
    pub(crate) nice_match: u16,

    pub(crate) match_start: Pos, /* start of matching string */
    pub(crate) prev_match: Pos,  /* previous match */
    pub(crate) strstart: usize,  /* start of string to insert */

    pub(crate) window: Window<'a>,
    pub(crate) w_size: usize, /* LZ77 window size (32K by default) */

    pub(crate) lookahead: usize, /* number of valid bytes ahead in window */

    _cache_line_0: (),

    /// prev[N], where N is an offset in the current window, contains the offset in the window
    /// of the previous 4-byte sequence that hashes to the same value as the 4-byte sequence
    /// starting at N. Together with head, prev forms a chained hash table that can be used
    /// to find earlier strings in the window that are potential matches for new input being
    /// deflated.
    pub(crate) prev: WeakSliceMut<'a, u16>,
    /// head[H] contains the offset of the last 4-character sequence seen so far in
    /// the current window that hashes to H (as calculated using the hash_calc_variant).
    pub(crate) head: WeakArrayMut<'a, u16, HASH_SIZE>,

    /// Length of the best match at previous step. Matches not greater than this
    /// are discarded. This is used in the lazy match evaluation.
    pub(crate) prev_length: u16,

    /// To speed up deflation, hash chains are never searched beyond this length.
    /// A higher limit improves compression ratio but degrades the speed.
    pub(crate) max_chain_length: u16,

    // TODO untangle this mess! zlib uses the same field differently based on compression level
    // we should just have 2 fields for clarity!
    //
    // Insert new strings in the hash table only if the match length is not
    // greater than this length. This saves time but degrades compression.
    // max_insert_length is used only for compression levels <= 6.
    // define max_insert_length  max_lazy_match
    /// Attempt to find a better match only when the current match is strictly smaller
    /// than this value. This mechanism is used only for compression levels >= 4.
    pub(crate) max_lazy_match: u16,

    /// number of string matches in current block
    /// NOTE: this is a saturating 8-bit counter, to help keep the struct compact. The code that
    /// makes decisions based on this field only cares whether the count is greater than 2, so
    /// an 8-bit counter is sufficient.
    pub(crate) matches: u8,

    /// Window position at the beginning of the current output block. Gets
    /// negative when the window is moved backwards.
    pub(crate) block_start: isize,

    pub(crate) sym_buf: SymBuf<'a>,

    _cache_line_1: (),

    /// Size of match buffer for literals/lengths.  There are 4 reasons for
    /// limiting lit_bufsize to 64K:
    ///   - frequencies can be kept in 16 bit counters
    ///   - if compression is not successful for the first block, all input
    ///     data is still in the window so we can still emit a stored block even
    ///     when input comes from standard input.  (This can also be done for
    ///     all blocks if lit_bufsize is not greater than 32K.)
    ///   - if compression is not successful for a file smaller than 64K, we can
    ///     even emit a stored file instead of a stored block (saving 5 bytes).
    ///     This is applicable only for zip (not gzip or zlib).
    ///   - creating new Huffman trees less frequently may not provide fast
    ///     adaptation to changes in the input data statistics. (Take for
    ///     example a binary file with poorly compressible code followed by
    ///     a highly compressible string table.) Smaller buffer sizes give
    ///     fast adaptation but have of course the overhead of transmitting
    ///     trees more frequently.
    ///   - I can't count above 4
    lit_bufsize: usize,

    /// Actual size of window: 2*w_size, except when the user input buffer is directly used as sliding window.
    pub(crate) window_size: usize,

    bit_writer: BitWriter<'a>,

    _cache_line_2: (),

    /// bit length of current block with optimal trees
    opt_len: usize,
    /// bit length of current block with static trees
    static_len: usize,

    /// bytes at end of window left to insert
    pub(crate) insert: usize,

    ///  hash index of string to be inserted
    pub(crate) ins_h: u32,

    gzhead: Option<&'a mut gz_header>,
    gzindex: usize,

    _padding_0: [u8; 16],

    _cache_line_3: (),

    crc_fold: crate::crc32::Crc32Fold,

    /// The (unaligned) address of the state allocation that can be passed to zfree.
    allocation_start: NonNull<u8>,
    /// Total size of the allocation in bytes.
    total_allocation_size: usize,

    l_desc: TreeDesc<HEAP_SIZE>,             /* literal and length tree */
    d_desc: TreeDesc<{ 2 * D_CODES + 1 }>,   /* distance tree */
    bl_desc: TreeDesc<{ 2 * BL_CODES + 1 }>, /* Huffman tree for bit lengths */
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "__internal-fuzz", derive(arbitrary::Arbitrary))]
pub enum Strategy {
    #[default]
    Default = 0,
    Filtered = 1,
    HuffmanOnly = 2,
    Rle = 3,
    Fixed = 4,
}

impl TryFrom<i32> for Strategy {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Strategy::Default),
            1 => Ok(Strategy::Filtered),
            2 => Ok(Strategy::HuffmanOnly),
            3 => Ok(Strategy::Rle),
            4 => Ok(Strategy::Fixed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DataType {
    Binary = 0,
    Text = 1,
    Unknown = 2,
}

impl<'a> State<'a> {
    pub const BIT_BUF_SIZE: u8 = BitWriter::BIT_BUF_SIZE;

    // log2(w_size)  (in the range MIN_WBITS..=MAX_WBITS)
    pub(crate) fn w_bits(&self) -> u32 {
        self.w_size.trailing_zeros()
    }

    pub(crate) fn w_mask(&self) -> usize {
        self.w_size - 1
    }

    pub(crate) fn max_dist(&self) -> usize {
        self.w_size - MIN_LOOKAHEAD
    }

    // TODO untangle this mess! zlib uses the same field differently based on compression level
    // we should just have 2 fields for clarity!
    pub(crate) fn max_insert_length(&self) -> usize {
        self.max_lazy_match as usize
    }

    /// Total size of the pending buf. But because `pending` shares memory with `sym_buf`, this is
    /// not the number of bytes that are actually in `pending`!
    pub(crate) fn pending_buf_size(&self) -> usize {
        self.lit_bufsize * 4
    }

    #[inline(always)]
    pub(crate) fn update_hash(&self, h: u32, val: u32) -> u32 {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::update_hash(h, val),
            HashCalcVariant::Roll => RollHashCalc::update_hash(h, val),
        }
    }

    #[inline(always)]
    pub(crate) fn quick_insert_string(&mut self, string: usize) -> u16 {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::quick_insert_string(self, string),
            HashCalcVariant::Roll => RollHashCalc::quick_insert_string(self, string),
        }
    }

    #[inline(always)]
    pub(crate) fn insert_string(&mut self, string: usize, count: usize) {
        match self.hash_calc_variant {
            HashCalcVariant::Standard => StandardHashCalc::insert_string(self, string, count),
            HashCalcVariant::Roll => RollHashCalc::insert_string(self, string, count),
        }
    }

    #[inline(always)]
    pub(crate) fn tally_lit(&mut self, unmatched: u8) -> bool {
        Self::tally_lit_help(&mut self.sym_buf, &mut self.l_desc, unmatched)
    }

    // This helper is to work around an ownership issue in algorithm/medium.
    pub(crate) fn tally_lit_help(
        sym_buf: &mut SymBuf,
        l_desc: &mut TreeDesc<HEAP_SIZE>,
        unmatched: u8,
    ) -> bool {
        const _VERIFY: () = {
            // Verify during compilation that even the largest possible value
            // of unmatched will fit within the expected range.
            assert!(
                u8::MAX as usize <= STD_MAX_MATCH - STD_MIN_MATCH,
                "tally_lit: bad literal"
            );
        };

        sym_buf.push_lit(unmatched);

        *l_desc.dyn_tree[unmatched as usize].freq_mut() += 1;

        // signal that the current block should be flushed
        sym_buf.should_flush_block()
    }

    const fn d_code(dist: usize) -> u8 {
        const _VERIFY: () = {
            // Verify during compilation that every DIST_CODE value is < D_CODES.
            let mut i = 0;
            while i < trees_tbl::DIST_CODE.len() {
                assert!(trees_tbl::DIST_CODE[i] < D_CODES as u8);
                i += 1;
            }
        };

        let index = if dist < 256 { dist } else { 256 + (dist >> 7) };
        self::trees_tbl::DIST_CODE[index]
    }

    #[inline(always)]
    pub(crate) fn tally_dist(&mut self, mut dist: usize, len: usize) -> bool {
        self.sym_buf.push_dist(dist as u16, len as u8);

        self.matches = self.matches.saturating_add(1);
        dist -= 1;

        assert!(dist < self.max_dist(), "tally_dist: bad match");

        let index = self::trees_tbl::LENGTH_CODE[len] as usize + LITERALS + 1;
        *self.l_desc.dyn_tree[index].freq_mut() += 1;

        *self.d_desc.dyn_tree[Self::d_code(dist) as usize].freq_mut() += 1;

        // signal that the current block should be flushed
        self.sym_buf.should_flush_block()
    }

    fn detect_data_type(dyn_tree: &[Value]) -> DataType {
        // set bits 0..6, 14..25, and 28..31
        // 0xf3ffc07f = binary 11110011111111111100000001111111
        const NON_TEXT: u64 = 0xf3ffc07f;
        let mut mask = NON_TEXT;

        /* Check for non-textual bytes. */
        for value in &dyn_tree[0..32] {
            if (mask & 1) != 0 && value.freq() != 0 {
                return DataType::Binary;
            }

            mask >>= 1;
        }

        /* Check for textual bytes. */
        if dyn_tree[9].freq() != 0 || dyn_tree[10].freq() != 0 || dyn_tree[13].freq() != 0 {
            return DataType::Text;
        }

        if dyn_tree[32..LITERALS].iter().any(|v| v.freq() != 0) {
            return DataType::Text;
        }

        // there are no explicit text or non-text bytes. The stream is either empty or has only
        // tolerated bytes
        DataType::Binary
    }

    fn compress_block_static_trees(&mut self) {
        let ltree = self::trees_tbl::STATIC_LTREE.as_slice();
        for (dist, lc) in self.sym_buf.iter() {
            match dist {
                0 => self.bit_writer.emit_lit(ltree, lc) as usize,
                _ => self.bit_writer.emit_dist_static(lc, dist),
            };
        }

        self.bit_writer.emit_end_block(ltree, false)
    }

    fn compress_block_dynamic_trees(&mut self) {
        self.bit_writer.compress_block_help(
            &self.sym_buf,
            &self.l_desc.dyn_tree,
            &self.d_desc.dyn_tree,
        );
    }

    fn header(&self) -> u16 {
        // preset dictionary flag in zlib header
        const PRESET_DICT: u16 = 0x20;

        // The deflate compression method (the only one supported in this version)
        const Z_DEFLATED: u16 = 8;

        let dict = match self.strstart {
            0 => 0,
            _ => PRESET_DICT,
        };

        let h = ((Z_DEFLATED + ((self.w_bits() as u16 - 8) << 4)) << 8)
            | (self.level_flags() << 6)
            | dict;

        h + 31 - (h % 31)
    }

    fn level_flags(&self) -> u16 {
        if self.strategy >= Strategy::HuffmanOnly || self.level < 2 {
            0
        } else if self.level < 6 {
            1
        } else if self.level == 6 {
            2
        } else {
            3
        }
    }

    fn zng_tr_init(&mut self) {
        self.l_desc.stat_desc = &StaticTreeDesc::L;

        self.d_desc.stat_desc = &StaticTreeDesc::D;

        self.bl_desc.stat_desc = &StaticTreeDesc::BL;

        self.bit_writer.bit_buffer = 0;
        self.bit_writer.bits_valid = 0;
        self.bit_writer.bits_used = 0;

        #[cfg(feature = "ZLIB_DEBUG")]
        {
            self.bit_writer.compressed_len = 0;
            self.bit_writer.bits_sent = 0;
        }

        // Initialize the first block of the first file:
        self.init_block();
    }

    /// initializes a new block
    fn init_block(&mut self) {
        // Initialize the trees.
        // TODO would a memset also work here?

        for value in &mut self.l_desc.dyn_tree[..L_CODES] {
            *value.freq_mut() = 0;
        }

        for value in &mut self.d_desc.dyn_tree[..D_CODES] {
            *value.freq_mut() = 0;
        }

        for value in &mut self.bl_desc.dyn_tree[..BL_CODES] {
            *value.freq_mut() = 0;
        }

        // end of block literal code
        const END_BLOCK: usize = 256;

        *self.l_desc.dyn_tree[END_BLOCK].freq_mut() = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.sym_buf.clear();
        self.matches = 0;
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Init = 1,

    GZip = 4,
    Extra = 5,
    Name = 6,
    Comment = 7,
    Hcrc = 8,

    Busy = 2,
    Finish = 3,
}

const fn rank_flush(f: i8) -> i8 {
    // rank Z_BLOCK between Z_NO_FLUSH and Z_PARTIAL_FLUSH
    ((f) * 2) - (if (f) > 4 { 9 } else { 0 })
}

#[derive(Debug)]
pub(crate) enum BlockState {
    /// block not completed, need more input or more output
    NeedMore = 0,
    /// block flush performed
    BlockDone = 1,
    /// finish started, need only more output at next deflate
    FinishStarted = 2,
    /// finish done, accept no more input or output
    FinishDone = 3,
}

// Maximum stored block length in deflate format (not including header).
pub(crate) const MAX_STORED: usize = 65535; // so u16::max

pub(crate) fn read_buf_window(stream: &mut DeflateStream, offset: usize, size: usize) -> usize {
    let len = Ord::min(stream.avail_in as usize, size);

    if len == 0 {
        return 0;
    }

    stream.avail_in -= len as u32;

    if stream.state.wrap == 2 {
        // we likely cannot fuse the crc32 and the copy here because the input can be changed by
        // a concurrent thread. Therefore it cannot be converted into a slice!
        let window = &mut stream.state.window;
        // SAFETY: len is bounded by avail_in, so this copy is in bounds.
        unsafe { window.copy_and_initialize(offset..offset + len, stream.next_in) };

        let data = &stream.state.window.filled()[offset..][..len];
        stream.state.crc_fold.fold(data, CRC32_INITIAL_VALUE);
    } else if stream.state.wrap == 1 {
        // we likely cannot fuse the adler32 and the copy here because the input can be changed by
        // a concurrent thread. Therefore it cannot be converted into a slice!
        let window = &mut stream.state.window;
        // SAFETY: len is bounded by avail_in, so this copy is in bounds.
        unsafe { window.copy_and_initialize(offset..offset + len, stream.next_in) };

        let data = &stream.state.window.filled()[offset..][..len];
        stream.adler = adler32(stream.adler as u32, data) as _;
    } else {
        let window = &mut stream.state.window;
        // SAFETY: len is bounded by avail_in, so this copy is in bounds.
        unsafe { window.copy_and_initialize(offset..offset + len, stream.next_in) };
    }

    // These can overflow, especially on windows where the integer type is u32 and input/output are
    // larger than 4GB.
    stream.next_in = stream.next_in.wrapping_add(len);
    stream.total_in = stream.total_in.wrapping_add(len as crate::c_api::z_size);

    len
}

pub(crate) enum BlockType {
    StoredBlock = 0,
    StaticTrees = 1,
    DynamicTrees = 2,
}

pub(crate) fn zng_tr_stored_block(
    state: &mut State,
    window_range: core::ops::Range<usize>,
    is_last: bool,
) {
    // send block type
    state.bit_writer.emit_tree(BlockType::StoredBlock, is_last);

    // align on byte boundary
    state.bit_writer.emit_align();

    state.bit_writer.cmpr_bits_align();

    let input_block: &[u8] = &state.window.filled()[window_range];
    let stored_len = input_block.len() as u16;

    state.bit_writer.pending.extend(&stored_len.to_le_bytes());
    state
        .bit_writer
        .pending
        .extend(&(!stored_len).to_le_bytes());

    state.bit_writer.cmpr_bits_add(32);
    state.bit_writer.sent_bits_add(32);
    if stored_len > 0 {
        state.bit_writer.pending.extend(input_block);
        state.bit_writer.cmpr_bits_add((stored_len << 3) as usize);
        state.bit_writer.sent_bits_add((stored_len << 3) as usize);
    }
}

/// The minimum match length mandated by the deflate standard
pub(crate) const STD_MIN_MATCH: usize = 3;
/// The maximum match length mandated by the deflate standard
pub(crate) const STD_MAX_MATCH: usize = 258;

/// The minimum wanted match length, affects deflate_quick, deflate_fast, deflate_medium and deflate_slow
pub(crate) const WANT_MIN_MATCH: usize = 4;

pub(crate) const MIN_LOOKAHEAD: usize = STD_MAX_MATCH + STD_MIN_MATCH + 1;

#[inline]
pub(crate) fn fill_window(stream: &mut DeflateStream) {
    debug_assert!(stream.state.lookahead < MIN_LOOKAHEAD);

    let wsize = stream.state.w_size;

    loop {
        let state = &mut *stream.state;
        let mut more = state.window_size - state.lookahead - state.strstart;

        // If the window is almost full and there is insufficient lookahead,
        // move the upper half to the lower one to make room in the upper half.
        if state.strstart >= wsize + state.max_dist() {
            // shift the window to the left
            let (old, new) = state.window.filled_mut()[..2 * wsize].split_at_mut(wsize);
            old.copy_from_slice(new);

            if state.match_start >= wsize as u16 {
                state.match_start -= wsize as u16;
            } else {
                state.match_start = 0;
                state.prev_length = 0;
            }

            state.strstart -= wsize; /* we now have strstart >= MAX_DIST */
            state.block_start = state.block_start.wrapping_sub_unsigned(wsize);
            state.insert = Ord::min(state.insert, state.strstart);

            self::slide_hash::slide_hash(state);

            more += wsize;
        }

        if stream.avail_in == 0 {
            break;
        }

        // If there was no sliding:
        //    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
        //    more == window_size - lookahead - strstart
        // => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
        // => more >= window_size - 2*WSIZE + 2
        // In the BIG_MEM or MMAP case (not yet supported),
        //   window_size == input_size + MIN_LOOKAHEAD  &&
        //   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
        // Otherwise, window_size == 2*WSIZE so more >= 2.
        // If there was sliding, more >= WSIZE. So in all cases, more >= 2.
        assert!(more >= 2, "more < 2");

        let n = read_buf_window(stream, stream.state.strstart + stream.state.lookahead, more);

        let state = &mut *stream.state;
        state.lookahead += n;

        // Initialize the hash value now that we have some input:
        if state.lookahead + state.insert >= STD_MIN_MATCH {
            let string = state.strstart - state.insert;
            if state.max_chain_length > 1024 {
                let v0 = state.window.filled()[string] as u32;
                let v1 = state.window.filled()[string + 1] as u32;
                state.ins_h = state.update_hash(v0, v1);
            } else if string >= 1 {
                state.quick_insert_string(string + 2 - STD_MIN_MATCH);
            }
            let mut count = state.insert;
            if state.lookahead == 1 {
                count -= 1;
            }
            if count > 0 {
                state.insert_string(string, count);
                state.insert -= count;
            }
        }

        // If the whole input has less than STD_MIN_MATCH bytes, ins_h is garbage,
        // but this is not important since only literal bytes will be emitted.

        if !(stream.state.lookahead < MIN_LOOKAHEAD && stream.avail_in != 0) {
            break;
        }
    }

    assert!(
        stream.state.strstart <= stream.state.window_size - MIN_LOOKAHEAD,
        "not enough room for search"
    );
}

pub(crate) struct StaticTreeDesc {
    /// static tree or NULL
    pub(crate) static_tree: &'static [Value],
    /// extra bits for each code or NULL
    extra_bits: &'static [u8],
    /// base index for extra_bits
    extra_base: usize,
    /// max number of elements in the tree
    elems: usize,
    /// max bit length for the codes
    max_length: u16,
}

impl StaticTreeDesc {
    const EMPTY: Self = Self {
        static_tree: &[],
        extra_bits: &[],
        extra_base: 0,
        elems: 0,
        max_length: 0,
    };

    /// extra bits for each length code
    const EXTRA_LBITS: [u8; LENGTH_CODES] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];

    /// extra bits for each distance code
    const EXTRA_DBITS: [u8; D_CODES] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    /// extra bits for each bit length code
    const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];

    /// The lengths of the bit length codes are sent in order of decreasing
    /// probability, to avoid transmitting the lengths for unused bit length codes.
    const BL_ORDER: [u8; BL_CODES] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    pub(crate) const L: Self = Self {
        static_tree: &self::trees_tbl::STATIC_LTREE,
        extra_bits: &Self::EXTRA_LBITS,
        extra_base: LITERALS + 1,
        elems: L_CODES,
        max_length: MAX_BITS as u16,
    };

    pub(crate) const D: Self = Self {
        static_tree: &self::trees_tbl::STATIC_DTREE,
        extra_bits: &Self::EXTRA_DBITS,
        extra_base: 0,
        elems: D_CODES,
        max_length: MAX_BITS as u16,
    };

    pub(crate) const BL: Self = Self {
        static_tree: &[],
        extra_bits: &Self::EXTRA_BLBITS,
        extra_base: 0,
        elems: BL_CODES,
        max_length: MAX_BL_BITS as u16,
    };
}

#[derive(Clone)]
pub(crate) struct TreeDesc<const N: usize> {
    dyn_tree: [Value; N],
    max_code: usize,
    stat_desc: &'static StaticTreeDesc,
}

impl<const N: usize> TreeDesc<N> {
    const EMPTY: Self = Self {
        dyn_tree: [Value::new(0, 0); N],
        max_code: 0,
        stat_desc: &StaticTreeDesc::EMPTY,
    };
}

fn build_tree<const N: usize>(state: &mut State, desc: &mut TreeDesc<N>) {
    let tree = &mut desc.dyn_tree;
    let stree = desc.stat_desc.static_tree;
    let elements = desc.stat_desc.elems;

    let mut heap = Heap::new();
    let mut max_code = heap.initialize(&mut tree[..elements]);

    // The pkzip format requires that at least one distance code exists,
    // and that at least one bit should be sent even if there is only one
    // possible code. So to avoid special checks later on we force at least
    // two codes of non zero frequency.
    while heap.heap_len < 2 {
        heap.heap_len += 1;
        let node = if max_code < 2 {
            max_code += 1;
            max_code
        } else {
            0
        };

        debug_assert!(node >= 0);
        let node = node as usize;

        heap.heap[heap.heap_len] = node as u32;
        *tree[node].freq_mut() = 1;
        heap.depth[node] = 0;
        state.opt_len -= 1;
        if !stree.is_empty() {
            state.static_len -= stree[node].len() as usize;
        }
        /* node is 0 or 1 so it does not have extra bits */
    }

    debug_assert!(max_code >= 0);
    let max_code = max_code as usize;
    desc.max_code = max_code;

    // The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
    // establish sub-heaps of increasing lengths:
    let mut n = heap.heap_len / 2;
    while n >= 1 {
        heap.pqdownheap(tree, n);
        n -= 1;
    }

    heap.construct_huffman_tree(tree, elements);

    // At this point, the fields freq and dad are set. We can now
    // generate the bit lengths.
    let bl_count = gen_bitlen(state, &mut heap, desc);

    // The field len is now set, we can generate the bit codes
    gen_codes(&mut desc.dyn_tree, max_code, &bl_count);
}

fn gen_bitlen<const N: usize>(
    state: &mut State,
    heap: &mut Heap,
    desc: &mut TreeDesc<N>,
) -> [u16; MAX_BITS + 1] {
    let tree = &mut desc.dyn_tree;
    let max_code = desc.max_code;
    let stree = desc.stat_desc.static_tree;
    let extra = desc.stat_desc.extra_bits;
    let base = desc.stat_desc.extra_base;
    let max_length = desc.stat_desc.max_length;

    let mut bl_count = [0u16; MAX_BITS + 1];

    // In a first pass, compute the optimal bit lengths (which may
    // overflow in the case of the bit length tree).
    *tree[heap.heap[heap.heap_max] as usize].len_mut() = 0; /* root of the heap */

    // number of elements with bit length too large
    let mut overflow: i32 = 0;

    for &n in &heap.heap[heap.heap_max + 1..HEAP_SIZE] {
        let n = n as usize;
        let mut bits = tree[usize::from(tree[n].dad())].len() + 1;

        if bits > max_length {
            bits = max_length;
            overflow += 1;
        }

        // We overwrite tree[n].Dad which is no longer needed
        *tree[n].len_mut() = bits;

        // not a leaf node
        if n > max_code {
            continue;
        }

        bl_count[usize::from(bits)] += 1;
        let mut xbits = 0;
        if n >= base {
            xbits = extra[n - base] as usize;
        }

        let f = tree[n].freq() as usize;
        state.opt_len += f * (bits as usize + xbits);

        if !stree.is_empty() {
            state.static_len += f * (stree[n].len() as usize + xbits);
        }
    }

    if overflow == 0 {
        return bl_count;
    }

    /* Find the first bit length which could increase: */
    loop {
        let mut bits = max_length as usize - 1;
        while bl_count[bits] == 0 {
            bits -= 1;
        }
        bl_count[bits] -= 1; /* move one leaf down the tree */
        bl_count[bits + 1] += 2; /* move one overflow item as its brother */
        bl_count[max_length as usize] -= 1;
        /* The brother of the overflow item also moves one step up,
         * but this does not affect bl_count[max_length]
         */
        overflow -= 2;

        if overflow <= 0 {
            break;
        }
    }

    // Now recompute all bit lengths, scanning in increasing frequency.
    // h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
    // lengths instead of fixing only the wrong ones. This idea is taken
    // from 'ar' written by Haruhiko Okumura.)
    let mut h = HEAP_SIZE;
    for bits in (1..=max_length).rev() {
        let mut n = bl_count[bits as usize];
        while n != 0 {
            h -= 1;
            let m = heap.heap[h] as usize;
            if m > max_code {
                continue;
            }

            if tree[m].len() != bits {
                // Tracev((stderr, "code %d bits %d->%u\n", m, tree[m].Len, bits));
                state.opt_len += (bits * tree[m].freq()) as usize;
                state.opt_len -= (tree[m].len() * tree[m].freq()) as usize;
                *tree[m].len_mut() = bits;
            }

            n -= 1;
        }
    }
    bl_count
}

/// Checks that symbol is a printing character (excluding space)
#[allow(unused)]
fn isgraph(c: u8) -> bool {
    (c > 0x20) && (c <= 0x7E)
}

fn gen_codes(tree: &mut [Value], max_code: usize, bl_count: &[u16]) {
    /* tree: the tree to decorate */
    /* max_code: largest code with non zero frequency */
    /* bl_count: number of codes at each bit length */
    let mut next_code = [0; MAX_BITS + 1]; /* next code value for each bit length */
    let mut code = 0; /* running code value */

    /* The distribution counts are first used to generate the code values
     * without bit reversal.
     */
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    /* Check that the bit counts in bl_count are consistent. The last code
     * must be all ones.
     */
    assert!(
        code + bl_count[MAX_BITS] - 1 == (1 << MAX_BITS) - 1,
        "inconsistent bit counts"
    );

    trace!("\ngen_codes: max_code {max_code} ");

    for n in 0..=max_code {
        let len = tree[n].len();
        if len == 0 {
            continue;
        }

        /* Now reverse the bits */
        assert!((1..=15).contains(&len), "code length must be 1-15");
        *tree[n].code_mut() = next_code[len as usize].reverse_bits() >> (16 - len);
        next_code[len as usize] += 1;

        if tree != self::trees_tbl::STATIC_LTREE.as_slice() {
            trace!(
                "\nn {:>3} {} l {:>2} c {:>4x} ({:x}) ",
                n,
                if isgraph(n as u8) {
                    char::from_u32(n as u32).unwrap()
                } else {
                    ' '
                },
                len,
                tree[n].code(),
                next_code[len as usize] - 1
            );
        }
    }
}

/// repeat previous bit length 3-6 times (2 bits of repeat count)
const REP_3_6: usize = 16;

/// repeat a zero length 3-10 times  (3 bits of repeat count)
const REPZ_3_10: usize = 17;

/// repeat a zero length 11-138 times  (7 bits of repeat count)
const REPZ_11_138: usize = 18;

fn scan_tree(bl_desc: &mut TreeDesc<{ 2 * BL_CODES + 1 }>, tree: &mut [Value], max_code: usize) {
    /* tree: the tree to be scanned */
    /* max_code: and its largest code of non zero frequency */
    let mut prevlen = -1isize; /* last emitted length */
    let mut curlen: isize; /* length of current code */
    let mut nextlen = tree[0].len(); /* length of next code */
    let mut count = 0; /* repeat count of the current code */
    let mut max_count = 7; /* max repeat count */
    let mut min_count = 4; /* min repeat count */

    if nextlen == 0 {
        max_count = 138;
        min_count = 3;
    }

    *tree[max_code + 1].len_mut() = 0xffff; /* guard */

    let bl_tree = &mut bl_desc.dyn_tree;

    for n in 0..=max_code {
        curlen = nextlen as isize;
        nextlen = tree[n + 1].len();
        count += 1;
        if count < max_count && curlen == nextlen as isize {
            continue;
        } else if count < min_count {
            *bl_tree[curlen as usize].freq_mut() += count;
        } else if curlen != 0 {
            if curlen != prevlen {
                *bl_tree[curlen as usize].freq_mut() += 1;
            }
            *bl_tree[REP_3_6].freq_mut() += 1;
        } else if count <= 10 {
            *bl_tree[REPZ_3_10].freq_mut() += 1;
        } else {
            *bl_tree[REPZ_11_138].freq_mut() += 1;
        }

        count = 0;
        prevlen = curlen;

        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        } else if curlen == nextlen as isize {
            max_count = 6;
            min_count = 3;
        } else {
            max_count = 7;
            min_count = 4;
        }
    }
}

fn send_all_trees(state: &mut State, lcodes: usize, dcodes: usize, blcodes: usize) {
    assert!(
        lcodes >= 257 && dcodes >= 1 && blcodes >= 4,
        "not enough codes"
    );
    assert!(
        lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES,
        "too many codes"
    );

    trace!("\nbl counts: ");
    state.bit_writer.send_bits(lcodes as u64 - 257, 5); /* not +255 as stated in appnote.txt */
    state.bit_writer.send_bits(dcodes as u64 - 1, 5);
    state.bit_writer.send_bits(blcodes as u64 - 4, 4); /* not -3 as stated in appnote.txt */

    for rank in 0..blcodes {
        trace!("\nbl code {:>2} ", StaticTreeDesc::BL_ORDER[rank]);
        state.bit_writer.send_bits(
            state.bl_desc.dyn_tree[StaticTreeDesc::BL_ORDER[rank] as usize].len() as u64,
            3,
        );
    }
    trace!("\nbl tree: sent {}", state.bit_writer.bits_sent);

    // literal tree
    state
        .bit_writer
        .send_tree(&state.l_desc.dyn_tree, &state.bl_desc.dyn_tree, lcodes - 1);
    trace!("\nlit tree: sent {}", state.bit_writer.bits_sent);

    // distance tree
    state
        .bit_writer
        .send_tree(&state.d_desc.dyn_tree, &state.bl_desc.dyn_tree, dcodes - 1);
    trace!("\ndist tree: sent {}", state.bit_writer.bits_sent);
}

/// Construct the Huffman tree for the bit lengths and return the index in
/// bl_order of the last bit length code to send.
fn build_bl_tree(state: &mut State) -> usize {
    /* Determine the bit length frequencies for literal and distance trees */

    scan_tree(
        &mut state.bl_desc,
        &mut state.l_desc.dyn_tree,
        state.l_desc.max_code,
    );

    scan_tree(
        &mut state.bl_desc,
        &mut state.d_desc.dyn_tree,
        state.d_desc.max_code,
    );

    /* Build the bit length tree: */
    {
        let mut tmp = TreeDesc::EMPTY;
        core::mem::swap(&mut tmp, &mut state.bl_desc);
        build_tree(state, &mut tmp);
        core::mem::swap(&mut tmp, &mut state.bl_desc);
    }

    /* opt_len now includes the length of the tree representations, except
     * the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
     */

    /* Determine the number of bit length codes to send. The pkzip format
     * requires that at least 4 bit length codes be sent. (appnote.txt says
     * 3 but the actual value used is 4.)
     */
    let mut max_blindex = BL_CODES - 1;
    while max_blindex >= 3 {
        let index = StaticTreeDesc::BL_ORDER[max_blindex] as usize;
        if state.bl_desc.dyn_tree[index].len() != 0 {
            break;
        }

        max_blindex -= 1;
    }

    /* Update opt_len to include the bit length tree and counts */
    state.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;
    trace!(
        "\ndyn trees: dyn {}, stat {}",
        state.opt_len,
        state.static_len
    );

    max_blindex
}

fn zng_tr_flush_block(
    stream: &mut DeflateStream,
    window_offset: Option<usize>,
    stored_len: u32,
    last: bool,
) {
    /* window_offset: offset of the input block into the window */
    /* stored_len: length of input block */
    /* last: one if this is the last block for a file */

    let mut opt_lenb;
    let static_lenb;
    let mut max_blindex = 0;

    let state = &mut stream.state;

    if state.sym_buf.is_empty() {
        opt_lenb = 0;
        static_lenb = 0;
        state.static_len = 7;
    } else if state.level > 0 {
        if stream.data_type == DataType::Unknown as i32 {
            stream.data_type = State::detect_data_type(&state.l_desc.dyn_tree) as i32;
        }

        {
            let mut tmp = TreeDesc::EMPTY;
            core::mem::swap(&mut tmp, &mut state.l_desc);

            build_tree(state, &mut tmp);
            core::mem::swap(&mut tmp, &mut state.l_desc);

            trace!(
                "\nlit data: dyn {}, stat {}",
                state.opt_len,
                state.static_len
            );
        }

        {
            let mut tmp = TreeDesc::EMPTY;
            core::mem::swap(&mut tmp, &mut state.d_desc);
            build_tree(state, &mut tmp);
            core::mem::swap(&mut tmp, &mut state.d_desc);

            trace!(
                "\ndist data: dyn {}, stat {}",
                state.opt_len,
                state.static_len
            );
        }

        // Build the bit length tree for the above two trees, and get the index
        // in bl_order of the last bit length code to send.
        max_blindex = build_bl_tree(state);

        // Determine the best encoding. Compute the block lengths in bytes.
        opt_lenb = (state.opt_len + 3 + 7) >> 3;
        static_lenb = (state.static_len + 3 + 7) >> 3;

        trace!(
            "\nopt {}({}) stat {}({}) stored {} lit {} ",
            opt_lenb,
            state.opt_len,
            static_lenb,
            state.static_len,
            stored_len,
            state.sym_buf.iter().count()
        );

        if static_lenb <= opt_lenb || state.strategy == Strategy::Fixed {
            opt_lenb = static_lenb;
        }
    } else {
        assert!(window_offset.is_some(), "lost buf");
        /* force a stored block */
        opt_lenb = stored_len as usize + 5;
        static_lenb = stored_len as usize + 5;
    }

    #[allow(clippy::unnecessary_unwrap)]
    if stored_len as usize + 4 <= opt_lenb && window_offset.is_some() {
        /* 4: two words for the lengths
         * The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
         * Otherwise we can't have processed more than WSIZE input bytes since
         * the last block flush, because compression would have been
         * successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
         * transform a block into a stored block.
         */
        let window_offset = window_offset.unwrap();
        let range = window_offset..window_offset + stored_len as usize;
        zng_tr_stored_block(state, range, last);
    } else if static_lenb == opt_lenb {
        state.bit_writer.emit_tree(BlockType::StaticTrees, last);
        state.compress_block_static_trees();
    // cmpr_bits_add(s, s.static_len);
    } else {
        state.bit_writer.emit_tree(BlockType::DynamicTrees, last);
        send_all_trees(
            state,
            state.l_desc.max_code + 1,
            state.d_desc.max_code + 1,
            max_blindex + 1,
        );

        state.compress_block_dynamic_trees();
    }

    // TODO
    // This check is made mod 2^32, for files larger than 512 MB and unsigned long implemented on 32 bits.
    // assert_eq!(state.compressed_len, state.bits_sent, "bad compressed size");

    state.init_block();
    if last {
        state.bit_writer.emit_align();
    }

    // Tracev((stderr, "\ncomprlen {}(%lu) ", s->compressed_len>>3, s->compressed_len-7*last));
}

pub(crate) fn flush_block_only(stream: &mut DeflateStream, is_last: bool) {
    zng_tr_flush_block(
        stream,
        (stream.state.block_start >= 0).then_some(stream.state.block_start as usize),
        (stream.state.strstart as isize - stream.state.block_start) as u32,
        is_last,
    );

    stream.state.block_start = stream.state.strstart as isize;
    flush_pending(stream)
}

fn flush_bytes(stream: &mut DeflateStream, mut bytes: &[u8]) -> ControlFlow<ReturnCode> {
    let mut state = &mut stream.state;

    // we'll be using the pending buffer as temporary storage
    let mut beg = state.bit_writer.pending.pending().len(); /* start of bytes to update crc */

    while state.bit_writer.pending.remaining() < bytes.len() {
        let copy = state.bit_writer.pending.remaining();

        state.bit_writer.pending.extend(&bytes[..copy]);

        stream.adler = crc32(
            stream.adler as u32,
            &state.bit_writer.pending.pending()[beg..],
        ) as z_checksum;

        state.gzindex += copy;
        flush_pending(stream);
        state = &mut stream.state;

        // could not flush all the pending output
        if !state.bit_writer.pending.pending().is_empty() {
            state.last_flush = -1;
            return ControlFlow::Break(ReturnCode::Ok);
        }

        beg = 0;
        bytes = &bytes[copy..];
    }

    state.bit_writer.pending.extend(bytes);

    stream.adler = crc32(
        stream.adler as u32,
        &state.bit_writer.pending.pending()[beg..],
    ) as z_checksum;
    state.gzindex = 0;

    ControlFlow::Continue(())
}

pub fn deflate(stream: &mut DeflateStream, flush: DeflateFlush) -> ReturnCode {
    if stream.next_out.is_null()
        || (stream.avail_in != 0 && stream.next_in.is_null())
        || (stream.state.status == Status::Finish && flush != DeflateFlush::Finish)
    {
        let err = ReturnCode::StreamError;
        stream.msg = err.error_message();
        return err;
    }

    if stream.avail_out == 0 {
        let err = ReturnCode::BufError;
        stream.msg = err.error_message();
        return err;
    }

    let old_flush = stream.state.last_flush;
    stream.state.last_flush = flush as i8;

    /* Flush as much pending output as possible */
    if !stream.state.bit_writer.pending.pending().is_empty() {
        flush_pending(stream);
        if stream.avail_out == 0 {
            /* Since avail_out is 0, deflate will be called again with
             * more output space, but possibly with both pending and
             * avail_in equal to zero. There won't be anything to do,
             * but this is not an error situation so make sure we
             * return OK instead of BUF_ERROR at next call of deflate:
             */
            stream.state.last_flush = -1;
            return ReturnCode::Ok;
        }

        /* Make sure there is something to do and avoid duplicate consecutive
         * flushes. For repeated and useless calls with Z_FINISH, we keep
         * returning Z_STREAM_END instead of Z_BUF_ERROR.
         */
    } else if stream.avail_in == 0
        && rank_flush(flush as i8) <= rank_flush(old_flush)
        && flush != DeflateFlush::Finish
    {
        let err = ReturnCode::BufError;
        stream.msg = err.error_message();
        return err;
    }

    /* User must not provide more input after the first FINISH: */
    if stream.state.status == Status::Finish && stream.avail_in != 0 {
        let err = ReturnCode::BufError;
        stream.msg = err.error_message();
        return err;
    }

    /* Write the header */
    if stream.state.status == Status::Init && stream.state.wrap == 0 {
        stream.state.status = Status::Busy;
    }

    if stream.state.status == Status::Init {
        let header = stream.state.header();
        stream
            .state
            .bit_writer
            .pending
            .extend(&header.to_be_bytes());

        /* Save the adler32 of the preset dictionary: */
        if stream.state.strstart != 0 {
            let adler = stream.adler as u32;
            stream.state.bit_writer.pending.extend(&adler.to_be_bytes());
        }

        stream.adler = ADLER32_INITIAL_VALUE as _;
        stream.state.status = Status::Busy;

        // compression must start with an empty pending buffer
        flush_pending(stream);

        if !stream.state.bit_writer.pending.pending().is_empty() {
            stream.state.last_flush = -1;

            return ReturnCode::Ok;
        }
    }

    if stream.state.status == Status::GZip {
        /* gzip header */
        stream.state.crc_fold = Crc32Fold::new();

        stream.state.bit_writer.pending.extend(&[31, 139, 8]);

        let extra_flags = if stream.state.level == 9 {
            2
        } else if stream.state.strategy >= Strategy::HuffmanOnly || stream.state.level < 2 {
            4
        } else {
            0
        };

        match &stream.state.gzhead {
            None => {
                let bytes = [0, 0, 0, 0, 0, extra_flags, gz_header::OS_CODE];
                stream.state.bit_writer.pending.extend(&bytes);
                stream.state.status = Status::Busy;

                /* Compression must start with an empty pending buffer */
                flush_pending(stream);
                if !stream.state.bit_writer.pending.pending().is_empty() {
                    stream.state.last_flush = -1;
                    return ReturnCode::Ok;
                }
            }
            Some(gzhead) => {
                stream.state.bit_writer.pending.extend(&[gzhead.flags()]);
                let bytes = (gzhead.time as u32).to_le_bytes();
                stream.state.bit_writer.pending.extend(&bytes);
                stream
                    .state
                    .bit_writer
                    .pending
                    .extend(&[extra_flags, gzhead.os as u8]);

                if !gzhead.extra.is_null() {
                    let bytes = (gzhead.extra_len as u16).to_le_bytes();
                    stream.state.bit_writer.pending.extend(&bytes);
                }

                if gzhead.hcrc != 0 {
                    stream.adler = crc32(
                        stream.adler as u32,
                        stream.state.bit_writer.pending.pending(),
                    ) as z_checksum
                }

                stream.state.gzindex = 0;
                stream.state.status = Status::Extra;
            }
        }
    }

    if stream.state.status == Status::Extra {
        if let Some(gzhead) = stream.state.gzhead.as_ref() {
            if !gzhead.extra.is_null() {
                let gzhead_extra = gzhead.extra;

                let extra = unsafe {
                    core::slice::from_raw_parts(
                        // SAFETY: gzindex is always less than extra_len, and the user
                        // guarantees the pointer is valid for extra_len.
                        gzhead_extra.add(stream.state.gzindex),
                        (gzhead.extra_len & 0xffff) as usize - stream.state.gzindex,
                    )
                };

                if let ControlFlow::Break(err) = flush_bytes(stream, extra) {
                    return err;
                }
            }
        }
        stream.state.status = Status::Name;
    }

    if stream.state.status == Status::Name {
        if let Some(gzhead) = stream.state.gzhead.as_ref() {
            if !gzhead.name.is_null() {
                // SAFETY: user satisfies precondition that gzhead.name is a C string.
                let gzhead_name = unsafe { CStr::from_ptr(gzhead.name.cast()) };
                let bytes = gzhead_name.to_bytes_with_nul();
                if let ControlFlow::Break(err) = flush_bytes(stream, bytes) {
                    return err;
                }
            }
            stream.state.status = Status::Comment;
        }
    }

    if stream.state.status == Status::Comment {
        if let Some(gzhead) = stream.state.gzhead.as_ref() {
            if !gzhead.comment.is_null() {
                // SAFETY: user satisfies precondition that gzhead.name is a C string.
                let gzhead_comment = unsafe { CStr::from_ptr(gzhead.comment.cast()) };
                let bytes = gzhead_comment.to_bytes_with_nul();
                if let ControlFlow::Break(err) = flush_bytes(stream, bytes) {
                    return err;
                }
            }
            stream.state.status = Status::Hcrc;
        }
    }

    if stream.state.status == Status::Hcrc {
        if let Some(gzhead) = stream.state.gzhead.as_ref() {
            if gzhead.hcrc != 0 {
                let bytes = (stream.adler as u16).to_le_bytes();
                if let ControlFlow::Break(err) = flush_bytes(stream, &bytes) {
                    return err;
                }
            }
        }

        stream.state.status = Status::Busy;

        // compression must start with an empty pending buffer
        flush_pending(stream);
        if !stream.state.bit_writer.pending.pending().is_empty() {
            stream.state.last_flush = -1;
            return ReturnCode::Ok;
        }
    }

    // Start a new block or continue the current one.
    let state = &mut stream.state;
    if stream.avail_in != 0
        || state.lookahead != 0
        || (flush != DeflateFlush::NoFlush && state.status != Status::Finish)
    {
        let bstate = self::algorithm::run(stream, flush);

        let state = &mut stream.state;

        if matches!(bstate, BlockState::FinishStarted | BlockState::FinishDone) {
            state.status = Status::Finish;
        }

        match bstate {
            BlockState::NeedMore | BlockState::FinishStarted => {
                if stream.avail_out == 0 {
                    state.last_flush = -1; /* avoid BUF_ERROR next call, see above */
                }
                return ReturnCode::Ok;
                /* If flush != Z_NO_FLUSH && avail_out == 0, the next call
                 * of deflate should use the same flush parameter to make sure
                 * that the flush is complete. So we don't have to output an
                 * empty block here, this will be done at next call. This also
                 * ensures that for a very small output buffer, we emit at most
                 * one empty block.
                 */
            }
            BlockState::BlockDone => {
                match flush {
                    DeflateFlush::NoFlush => unreachable!("condition of inner surrounding if"),
                    DeflateFlush::PartialFlush => {
                        state.bit_writer.align();
                    }
                    DeflateFlush::SyncFlush => {
                        // add an empty stored block that is marked as not final. This is useful for
                        // parallel deflate where we want to make sure the intermediate blocks are not
                        // marked as "last block".
                        zng_tr_stored_block(state, 0..0, false);
                    }
                    DeflateFlush::FullFlush => {
                        // add an empty stored block that is marked as not final. This is useful for
                        // parallel deflate where we want to make sure the intermediate blocks are not
                        // marked as "last block".
                        zng_tr_stored_block(state, 0..0, false);

                        state.head.as_mut_slice().fill(0); // forget history

                        if state.lookahead == 0 {
                            state.strstart = 0;
                            state.block_start = 0;
                            state.insert = 0;
                        }
                    }
                    DeflateFlush::Block => { /* fall through */ }
                    DeflateFlush::Finish => unreachable!("condition of outer surrounding if"),
                }

                flush_pending(stream);

                if stream.avail_out == 0 {
                    stream.state.last_flush = -1; /* avoid BUF_ERROR at next call, see above */
                    return ReturnCode::Ok;
                }
            }
            BlockState::FinishDone => { /* do nothing */ }
        }
    }

    if flush != DeflateFlush::Finish {
        return ReturnCode::Ok;
    }

    // write the trailer
    if stream.state.wrap == 2 {
        let crc_fold = core::mem::take(&mut stream.state.crc_fold);
        stream.adler = crc_fold.finish() as z_checksum;

        let adler = stream.adler as u32;
        stream.state.bit_writer.pending.extend(&adler.to_le_bytes());

        let total_in = stream.total_in as u32;
        stream
            .state
            .bit_writer
            .pending
            .extend(&total_in.to_le_bytes());
    } else if stream.state.wrap == 1 {
        let adler = stream.adler as u32;
        stream.state.bit_writer.pending.extend(&adler.to_be_bytes());
    }

    flush_pending(stream);

    // If avail_out is zero, the application will call deflate again to flush the rest.
    if stream.state.wrap > 0 {
        stream.state.wrap = -stream.state.wrap; /* write the trailer only once! */
    }

    if stream.state.bit_writer.pending.pending().is_empty() {
        assert_eq!(stream.state.bit_writer.bits_valid, 0, "bi_buf not flushed");
        return ReturnCode::StreamEnd;
    }
    ReturnCode::Ok
}

pub(crate) fn flush_pending(stream: &mut DeflateStream) {
    let state = &mut stream.state;

    state.bit_writer.flush_bits();

    let pending = state.bit_writer.pending.pending();
    let len = Ord::min(pending.len(), stream.avail_out as usize);

    if len == 0 {
        return;
    }

    trace!("\n[FLUSH {len} bytes]");
    // SAFETY: len is min(pending, stream.avail_out), so we won't overrun next_out.
    unsafe { core::ptr::copy_nonoverlapping(pending.as_ptr(), stream.next_out, len) };

    stream.next_out = stream.next_out.wrapping_add(len);
    stream.total_out += len as crate::c_api::z_size;
    stream.avail_out -= len as crate::c_api::uInt;

    state.bit_writer.pending.advance(len);
}

/// Compresses `input` into the provided `output` buffer.
///
/// Returns a subslice of `output` containing the compressed bytes and a
/// [`ReturnCode`] indicating the result of the operation. Returns [`ReturnCode::BufError`] if
/// there is insufficient output space.
///
/// Use [`compress_bound`] for an upper bound on how large the output buffer needs to be.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
/// # fn foo(input: &[u8]) {
/// let mut buf = vec![0u8; compress_bound(input.len())];
/// let (compressed, rc) = compress_slice(&mut buf, input, DeflateConfig::default());
/// # }
/// ```
pub fn compress_slice<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    // SAFETY: a [u8] is a valid [MaybeUninit<u8>].
    let output_uninit = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut MaybeUninit<u8>, output.len())
    };

    compress(output_uninit, input, config)
}

pub fn compress<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: DeflateConfig,
) -> (&'a mut [u8], ReturnCode) {
    compress_with_flush(output, input, config, DeflateFlush::Finish)
}

pub fn compress_slice_with_flush<'a>(
    output: &'a mut [u8],
    input: &[u8],
    config: DeflateConfig,
    flush: DeflateFlush,
) -> (&'a mut [u8], ReturnCode) {
    // SAFETY: a [u8] is a valid [MaybeUninit<u8>], and `compress_with_flush` never uninitializes previously initialized memory.
    let output_uninit = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut MaybeUninit<u8>, output.len())
    };

    compress_with_flush(output_uninit, input, config, flush)
}

pub fn compress_with_flush<'a>(
    output: &'a mut [MaybeUninit<u8>],
    input: &[u8],
    config: DeflateConfig,
    final_flush: DeflateFlush,
) -> (&'a mut [u8], ReturnCode) {
    let mut stream = z_stream {
        next_in: input.as_ptr() as *mut u8,
        avail_in: 0, // for special logic in the first  iteration
        total_in: 0,
        next_out: output.as_mut_ptr() as *mut u8,
        avail_out: 0, // for special logic on the first iteration
        total_out: 0,
        msg: core::ptr::null_mut(),
        state: core::ptr::null_mut(),
        zalloc: None,
        zfree: None,
        opaque: core::ptr::null_mut(),
        data_type: 0,
        adler: 0,
        reserved: 0,
    };

    let err = init(&mut stream, config);
    if err != ReturnCode::Ok {
        return (&mut [], err);
    }

    let max = core::ffi::c_uint::MAX as usize;

    let mut left = output.len();
    let mut source_len = input.len();

    let return_code = loop {
        if stream.avail_out == 0 {
            stream.avail_out = Ord::min(left, max) as _;
            left -= stream.avail_out as usize;
        }

        if stream.avail_in == 0 {
            stream.avail_in = Ord::min(source_len, max) as _;
            source_len -= stream.avail_in as usize;
        }

        let flush = if source_len > 0 {
            DeflateFlush::NoFlush
        } else {
            final_flush
        };

        let err = if let Some(stream) = unsafe { DeflateStream::from_stream_mut(&mut stream) } {
            deflate(stream, flush)
        } else {
            ReturnCode::StreamError
        };

        match err {
            ReturnCode::Ok => continue,
            ReturnCode::StreamEnd => break ReturnCode::Ok,
            _ => break err,
        }
    };

    // SAFETY: we have now initialized these bytes
    let output_slice = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, stream.total_out as usize)
    };

    // may DataError if insufficient output space
    if let Some(stream) = unsafe { DeflateStream::from_stream_mut(&mut stream) } {
        let _ = end(stream);
    }

    (output_slice, return_code)
}

/// Returns the upper bound on the compressed size for an input of `source_len` bytes.
///
/// When compression has this much space available, it will never fail because of insufficient
/// output space.
///
/// # Example
///
/// ```
/// # use zlib_rs::*;
///
/// assert_eq!(compress_bound(1024), 1161);
/// assert_eq!(compress_bound(4096), 4617);
/// assert_eq!(compress_bound(65536), 73737);
///
/// # fn foo(input: &[u8]) {
/// let mut buf = vec![0u8; compress_bound(input.len())];
/// let (compressed, rc) = compress_slice(&mut buf, input, DeflateConfig::default());
/// # }
/// ```
pub const fn compress_bound(source_len: usize) -> usize {
    compress_bound_help(source_len, ZLIB_WRAPLEN)
}

const fn compress_bound_help(source_len: usize, wrap_len: usize) -> usize {
    source_len // The source size itself */
        // Always at least one byte for any input
        .wrapping_add(if source_len == 0 { 1 } else { 0 })
        // One extra byte for lengths less than 9
        .wrapping_add(if source_len < 9 { 1 } else { 0 })
        // Source encoding overhead, padded to next full byte
        .wrapping_add(deflate_quick_overhead(source_len))
        // Deflate block overhead bytes
        .wrapping_add(DEFLATE_BLOCK_OVERHEAD)
        // none, zlib or gzip wrapper
        .wrapping_add(wrap_len)
}

///  heap used to build the Huffman trees
///
/// The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
/// The same heap array is used to build all trees.
#[derive(Clone)]
struct Heap {
    heap: [u32; 2 * L_CODES + 1],

    /// number of elements in the heap
    heap_len: usize,

    /// element of the largest frequency
    heap_max: usize,

    depth: [u8; 2 * L_CODES + 1],
}

impl Heap {
    // an empty heap
    fn new() -> Self {
        Self {
            heap: [0; 2 * L_CODES + 1],
            heap_len: 0,
            heap_max: 0,
            depth: [0; 2 * L_CODES + 1],
        }
    }

    /// Construct the initial heap, with least frequent element in
    /// heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
    fn initialize(&mut self, tree: &mut [Value]) -> isize {
        let mut max_code = -1;

        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;

        for (n, node) in tree.iter_mut().enumerate() {
            if node.freq() > 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as u32;
                max_code = n as isize;
                self.depth[n] = 0;
            } else {
                *node.len_mut() = 0;
            }
        }

        max_code
    }

    /// Index within the heap array of least frequent node in the Huffman tree
    const SMALLEST: usize = 1;

    fn pqdownheap(&mut self, tree: &[Value], mut k: usize) {
        /* tree: the tree to restore */
        /* k: node to move down */

        // Given the index $i of a node in the tree, pack the node's frequency and depth
        // into a single integer. The heap ordering logic uses a primary sort on frequency
        // and a secondary sort on depth, so packing both into one integer makes it
        // possible to sort with fewer comparison operations.
        macro_rules! freq_and_depth {
            ($i:expr) => {
                (tree[$i as usize].freq() as u32) << 8 | self.depth[$i as usize] as u32
            };
        }

        let v = self.heap[k];
        let v_val = freq_and_depth!(v);
        let mut j = k << 1; /* left son of k */

        while j <= self.heap_len {
            /* Set j to the smallest of the two sons: */
            let mut j_val = freq_and_depth!(self.heap[j]);
            if j < self.heap_len {
                let j1_val = freq_and_depth!(self.heap[j + 1]);
                if j1_val <= j_val {
                    j += 1;
                    j_val = j1_val;
                }
            }

            /* Exit if v is smaller than both sons */
            if v_val <= j_val {
                break;
            }

            /* Exchange v with the smallest son */
            self.heap[k] = self.heap[j];
            k = j;

            /* And continue down the tree, setting j to the left son of k */
            j <<= 1;
        }

        self.heap[k] = v;
    }

    /// Remove the smallest element from the heap and recreate the heap with
    /// one less element. Updates heap and heap_len.
    fn pqremove(&mut self, tree: &[Value]) -> u32 {
        let top = self.heap[Self::SMALLEST];
        self.heap[Self::SMALLEST] = self.heap[self.heap_len];
        self.heap_len -= 1;

        self.pqdownheap(tree, Self::SMALLEST);

        top
    }

    /// Construct the Huffman tree by repeatedly combining the least two frequent nodes.
    fn construct_huffman_tree(&mut self, tree: &mut [Value], mut node: usize) {
        loop {
            let n = self.pqremove(tree) as usize; /* n = node of least frequency */
            let m = self.heap[Heap::SMALLEST] as usize; /* m = node of next least frequency */

            self.heap_max -= 1;
            self.heap[self.heap_max] = n as u32; /* keep the nodes sorted by frequency */
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as u32;

            /* Create a new node father of n and m */
            *tree[node].freq_mut() = tree[n].freq() + tree[m].freq();
            self.depth[node] = Ord::max(self.depth[n], self.depth[m]) + 1;

            *tree[n].dad_mut() = node as u16;
            *tree[m].dad_mut() = node as u16;

            /* and insert the new node in the heap */
            self.heap[Heap::SMALLEST] = node as u32;
            node += 1;

            self.pqdownheap(tree, Heap::SMALLEST);

            if self.heap_len < 2 {
                break;
            }
        }

        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[Heap::SMALLEST];
    }
}

/// # Safety
///
/// The caller must guarantee:
///
/// * If `head` is `Some`
///     - `head.extra` is `NULL` or is readable for at least `head.extra_len` bytes
///     - `head.name` is `NULL` or satisfies the requirements of [`core::ffi::CStr::from_ptr`]
///     - `head.comment` is `NULL` or satisfies the requirements of [`core::ffi::CStr::from_ptr`]
pub unsafe fn set_header<'a>(
    stream: &mut DeflateStream<'a>,
    head: Option<&'a mut gz_header>,
) -> ReturnCode {
    if stream.state.wrap != 2 {
        ReturnCode::StreamError
    } else {
        stream.state.gzhead = head;
        ReturnCode::Ok
    }
}

// zlib format overhead
const ZLIB_WRAPLEN: usize = 6;
// gzip format overhead
const GZIP_WRAPLEN: usize = 18;

const DEFLATE_HEADER_BITS: usize = 3;
const DEFLATE_EOBS_BITS: usize = 15;
const DEFLATE_PAD_BITS: usize = 6;
const DEFLATE_BLOCK_OVERHEAD: usize =
    (DEFLATE_HEADER_BITS + DEFLATE_EOBS_BITS + DEFLATE_PAD_BITS) >> 3;

const DEFLATE_QUICK_LIT_MAX_BITS: usize = 9;
const fn deflate_quick_overhead(x: usize) -> usize {
    let sum = x
        .wrapping_mul(DEFLATE_QUICK_LIT_MAX_BITS - 8)
        .wrapping_add(7);

    // imitate zlib-ng rounding behavior (on windows, c_ulong is 32 bits)
    (sum as core::ffi::c_ulong >> 3) as usize
}

/// For the default windowBits of 15 and memLevel of 8, this function returns
/// a close to exact, as well as small, upper bound on the compressed size.
/// They are coded as constants here for a reason--if the #define's are
/// changed, then this function needs to be changed as well.  The return
/// value for 15 and 8 only works for those exact settings.
///
/// For any setting other than those defaults for windowBits and memLevel,
/// the value returned is a conservative worst case for the maximum expansion
/// resulting from using fixed blocks instead of stored blocks, which deflate
/// can emit on compressed data for some combinations of the parameters.
///
/// This function could be more sophisticated to provide closer upper bounds for
/// every combination of windowBits and memLevel.  But even the conservative
/// upper bound of about 14% expansion does not seem onerous for output buffer
/// allocation.
pub fn bound(stream: Option<&mut DeflateStream>, source_len: usize) -> usize {
    // on windows, c_ulong is only a 32-bit integer
    let mask = core::ffi::c_ulong::MAX as usize;

    // conservative upper bound for compressed data
    let comp_len = source_len
        .wrapping_add((source_len.wrapping_add(7) & mask) >> 3)
        .wrapping_add((source_len.wrapping_add(63) & mask) >> 6)
        .wrapping_add(5);

    let Some(stream) = stream else {
        // return conservative bound plus zlib wrapper
        return comp_len.wrapping_add(6);
    };

    /* compute wrapper length */
    let wrap_len = match stream.state.wrap {
        0 => {
            // raw deflate
            0
        }
        1 => {
            // zlib wrapper
            if stream.state.strstart != 0 {
                ZLIB_WRAPLEN + 4
            } else {
                ZLIB_WRAPLEN
            }
        }
        2 => {
            // gzip wrapper
            let mut gz_wrap_len = GZIP_WRAPLEN;

            if let Some(header) = &stream.state.gzhead {
                if !header.extra.is_null() {
                    gz_wrap_len += 2 + header.extra_len as usize;
                }

                let mut c_string = header.name;
                if !c_string.is_null() {
                    loop {
                        gz_wrap_len += 1;
                        // SAFETY: user guarantees header.name is a valid C string.
                        unsafe {
                            if *c_string == 0 {
                                break;
                            }
                            c_string = c_string.add(1);
                        }
                    }
                }

                let mut c_string = header.comment;
                if !c_string.is_null() {
                    loop {
                        gz_wrap_len += 1;
                        // SAFETY: user guarantees header.comment is a valid C string.
                        unsafe {
                            if *c_string == 0 {
                                break;
                            }
                            c_string = c_string.add(1);
                        }
                    }
                }

                if header.hcrc != 0 {
                    gz_wrap_len += 2;
                }
            }

            gz_wrap_len
        }
        _ => {
            // default
            ZLIB_WRAPLEN
        }
    };

    if stream.state.w_bits() != MAX_WBITS as u32 || HASH_BITS < 15 {
        if stream.state.level == 0 {
            /* upper bound for stored blocks with length 127 (memLevel == 1) ~4% overhead plus a small constant */
            source_len
                .wrapping_add(source_len >> 5)
                .wrapping_add(source_len >> 7)
                .wrapping_add(source_len >> 11)
                .wrapping_add(7)
                .wrapping_add(wrap_len)
        } else {
            comp_len.wrapping_add(wrap_len)
        }
    } else {
        compress_bound_help(source_len, wrap_len)
    }
}

/// # Safety
///
/// The `dictionary` must have enough space for the dictionary.
pub unsafe fn get_dictionary(stream: &DeflateStream<'_>, dictionary: *mut u8) -> usize {
    let s = &stream.state;
    let len = Ord::min(s.strstart + s.lookahead, s.w_size);

    if !dictionary.is_null() && len > 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                s.window.as_ptr().add(s.strstart + s.lookahead - len),
                dictionary,
                len,
            );
        }
    }

    len
}

struct DeflateAllocOffsets {
    total_size: usize,
    state_pos: usize,
    window_pos: usize,
    pending_pos: usize,
    sym_buf_pos: usize,
    prev_pos: usize,
    head_pos: usize,
}

impl DeflateAllocOffsets {
    fn new(window_bits: usize, lit_bufsize: usize) -> Self {
        use core::mem::size_of;

        // 64B alignment of individual items in the alloc.
        // Note that changing this also requires changes in 'init' and 'copy'.
        const ALIGN_SIZE: usize = 64;
        const LIT_BUFS: usize = 4;

        let mut curr_size = 0usize;

        /* Define sizes */
        let state_size = size_of::<State>();
        // Allocate a second window worth of space to avoid the need to shift the data constantly.
        let window_size = (1 << window_bits) * 2;
        let prev_size = (1 << window_bits) * size_of::<Pos>();
        let head_size = HASH_SIZE * size_of::<Pos>();
        let pending_size = lit_bufsize * LIT_BUFS;
        let sym_buf_size = lit_bufsize * (LIT_BUFS - 1);
        // let alloc_size = size_of::<DeflateAlloc>();

        /* Calculate relative buffer positions and paddings */
        let state_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = state_pos + state_size;

        let window_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = window_pos + window_size;

        let prev_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = prev_pos + prev_size;

        let head_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = head_pos + head_size;

        let pending_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = pending_pos + pending_size;

        let sym_buf_pos = curr_size.next_multiple_of(ALIGN_SIZE);
        curr_size = sym_buf_pos + sym_buf_size;

        /* Add ALIGN_SIZE-1 to allow alignment (done in the 'init' and 'copy' functions), and round
         * size of buffer up to next multiple of ALIGN_SIZE */
        let total_size = (curr_size + (ALIGN_SIZE - 1)).next_multiple_of(ALIGN_SIZE);

        Self {
            total_size,
            state_pos,
            window_pos,
            pending_pos,
            sym_buf_pos,
            prev_pos,
            head_pos,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        inflate::{decompress_slice, InflateConfig, InflateStream},
        InflateFlush,
    };

    use super::*;

    use core::{ffi::CStr, sync::atomic::AtomicUsize};

    #[test]
    fn detect_data_type_basic() {
        let empty = || [Value::new(0, 0); LITERALS];

        assert_eq!(State::detect_data_type(&empty()), DataType::Binary);

        let mut binary = empty();
        binary[0] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&binary), DataType::Binary);

        let mut text = empty();
        text[b'\r' as usize] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&text), DataType::Text);

        let mut text = empty();
        text[b'a' as usize] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&text), DataType::Text);

        let mut non_text = empty();
        non_text[7] = Value::new(1, 0);
        assert_eq!(State::detect_data_type(&non_text), DataType::Binary);
    }

    #[test]
    fn from_stream_mut() {
        unsafe {
            assert!(DeflateStream::from_stream_mut(core::ptr::null_mut()).is_none());

            let mut stream = z_stream::default();
            assert!(DeflateStream::from_stream_mut(&mut stream).is_none());

            // state is still NULL
            assert!(DeflateStream::from_stream_mut(&mut stream).is_none());

            init(&mut stream, DeflateConfig::default());
            let stream = DeflateStream::from_stream_mut(&mut stream);
            assert!(stream.is_some());

            assert!(end(stream.unwrap()).is_ok());
        }
    }

    #[cfg(feature = "c-allocator")]
    unsafe extern "C" fn fail_nth_allocation<const N: usize>(
        opaque: crate::c_api::voidpf,
        items: crate::c_api::uInt,
        size: crate::c_api::uInt,
    ) -> crate::c_api::voidpf {
        let count = unsafe { &*(opaque as *const AtomicUsize) };

        if count.fetch_add(1, core::sync::atomic::Ordering::Relaxed) != N {
            // must use the C allocator internally because (de)allocation is based on function
            // pointer values and because we don't use the rust allocator directly, the allocation
            // logic will store the pointer to the start at the start of the allocation.
            unsafe { (crate::allocate::C.zalloc)(opaque, items, size) }
        } else {
            core::ptr::null_mut()
        }
    }

    #[test]
    #[cfg(feature = "c-allocator")]
    fn init_invalid_allocator() {
        {
            let atomic = AtomicUsize::new(0);
            let mut stream = z_stream {
                zalloc: Some(fail_nth_allocation::<0>),
                zfree: Some(crate::allocate::C.zfree),
                opaque: &atomic as *const _ as *const core::ffi::c_void as *mut _,
                ..z_stream::default()
            };
            assert_eq!(
                init(&mut stream, DeflateConfig::default()),
                ReturnCode::MemError
            );
        }
    }

    #[test]
    #[cfg(feature = "c-allocator")]
    fn copy_invalid_allocator() {
        let mut stream = z_stream::default();

        let atomic = AtomicUsize::new(0);
        stream.opaque = &atomic as *const _ as *const core::ffi::c_void as *mut _;
        stream.zalloc = Some(fail_nth_allocation::<1>);
        stream.zfree = Some(crate::allocate::C.zfree);

        // init performs 6 allocations; we don't want those to fail
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&mut stream) }) else {
            unreachable!()
        };

        let mut stream_copy = MaybeUninit::<DeflateStream>::zeroed();

        assert_eq!(copy(&mut stream_copy, stream), ReturnCode::MemError);

        assert!(end(stream).is_ok());
    }

    mod invalid_deflate_config {
        use super::*;

        #[test]
        fn sanity_check() {
            let mut stream = z_stream::default();
            assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);

            assert!(stream.zalloc.is_some());
            assert!(stream.zfree.is_some());

            // this should be the default level
            let stream = unsafe { DeflateStream::from_stream_mut(&mut stream) }.unwrap();
            assert_eq!(stream.state.level, 6);

            assert!(end(stream).is_ok());
        }

        #[test]
        fn window_bits_correction() {
            // window_bits of 8 gets turned into 9 internally
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: 8,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::Ok);
            let stream = unsafe { DeflateStream::from_stream_mut(&mut stream) }.unwrap();
            assert_eq!(stream.state.w_bits(), 9);

            assert!(end(stream).is_ok());
        }

        #[test]
        fn window_bits_too_low() {
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: -16,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::StreamError);
        }

        #[test]
        fn window_bits_too_high() {
            // window bits too high
            let mut stream = z_stream::default();
            let config = DeflateConfig {
                window_bits: 42,
                ..Default::default()
            };
            assert_eq!(init(&mut stream, config), ReturnCode::StreamError);
        }
    }

    #[test]
    fn end_data_error() {
        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&mut stream) }.unwrap();

        // next deflate into too little space
        let input = b"Hello World\n";
        stream.next_in = input.as_ptr() as *mut u8;
        stream.avail_in = input.len() as _;
        let output = &mut [0, 0, 0];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        // the deflate is fine
        assert_eq!(deflate(stream, DeflateFlush::NoFlush), ReturnCode::Ok);

        // but end is not
        assert!(end(stream).is_err());
    }

    #[test]
    fn test_reset_keep() {
        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, DeflateConfig::default()), ReturnCode::Ok);
        let stream = unsafe { DeflateStream::from_stream_mut(&mut stream) }.unwrap();

        // next deflate into too little space
        let input = b"Hello World\n";
        stream.next_in = input.as_ptr() as *mut u8;
        stream.avail_in = input.len() as _;

        let output = &mut [0; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        assert_eq!(reset_keep(stream), ReturnCode::Ok);

        let output = &mut [0; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        assert!(end(stream).is_ok());
    }

    #[test]
    fn hello_world_huffman_only() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x08, 0xcf, 0x2f, 0xca, 0x49, 0x51,
            0xe4, 0x02, 0x00, 0x20, 0x91, 0x04, 0x48,
        ];

        let input = "Hello World!\n";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 6,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::HuffmanOnly,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    #[test]
    fn hello_world_quick() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x08, 0xcf, 0x2f, 0xca, 0x49, 0x51,
            0xe4, 0x02, 0x00, 0x20, 0x91, 0x04, 0x48,
        ];

        let input = "Hello World!\n";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 1,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    #[test]
    fn hello_world_quick_random() {
        const EXPECTED: &[u8] = &[
            0x78, 0x01, 0x53, 0xe1, 0x50, 0x51, 0xe1, 0x52, 0x51, 0x51, 0x01, 0x00, 0x03, 0xec,
            0x00, 0xeb,
        ];

        let input = "$\u{8}$$\n$$$";

        let mut output = vec![0; 128];

        let config = DeflateConfig {
            level: 1,
            method: Method::Deflated,
            window_bits: crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let (output, err) = compress_slice(&mut output, input.as_bytes(), config);

        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output.len(), EXPECTED.len());

        assert_eq!(EXPECTED, output);
    }

    fn fuzz_based_test(input: &[u8], config: DeflateConfig, expected: &[u8]) {
        let mut output_rs = [0; 1 << 17];
        let (output_rs, err) = compress_slice(&mut output_rs, input, config);
        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output_rs, expected);
    }

    #[test]
    fn simple_rle() {
        fuzz_based_test(
            "\0\0\0\0\u{6}".as_bytes(),
            DeflateConfig {
                level: -1,
                method: Method::Deflated,
                window_bits: 11,
                mem_level: 4,
                strategy: Strategy::Rle,
            },
            &[56, 17, 99, 0, 2, 54, 0, 0, 11, 0, 7],
        )
    }

    #[test]
    fn fill_window_out_of_bounds() {
        const INPUT: &[u8] = &[
            0x71, 0x71, 0x71, 0x71, 0x71, 0x6a, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x1d, 0x1d, 0x63,
            0x63, 0x63, 0x63, 0x63, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d, 0x1d,
            0x1d, 0x27, 0x0, 0x0, 0x0, 0x1d, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71,
            0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x31, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x0, 0x0, 0x0, 0x0, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50,
            0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x48, 0x50,
            0x50, 0x50, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2c, 0x0, 0x0, 0x0, 0x0, 0x4a,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x70, 0x71, 0x71, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x6a, 0x0, 0x0, 0x0, 0x0,
            0x71, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x31, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x0, 0x4a, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71,
            0x6a, 0x0, 0x0, 0x0, 0x0, 0x71, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x31, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1d, 0x1d, 0x0, 0x0, 0x0, 0x0,
            0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50,
            0x50, 0x50, 0x50, 0x50, 0x48, 0x50, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x3f, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x50, 0x50, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x2c, 0x0, 0x0, 0x0, 0x0, 0x4a, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x70, 0x71, 0x71, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x3f, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x3b, 0x3f, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x20, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x71, 0x75, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x10, 0x0, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x3b, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x76, 0x71, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x71, 0x71, 0x10, 0x0, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
            0x71, 0x3b, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x76, 0x71, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x0, 0x0, 0x0, 0x0, 0x0, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
            0x34, 0x34, 0x30, 0x34, 0x34, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x71, 0x0, 0x0, 0x0, 0x0, 0x6,
        ];

        fuzz_based_test(
            INPUT,
            DeflateConfig {
                level: -1,
                method: Method::Deflated,
                window_bits: 9,
                mem_level: 1,
                strategy: Strategy::HuffmanOnly,
            },
            &[
                0x18, 0x19, 0x4, 0xc1, 0x21, 0x1, 0xc4, 0x0, 0x10, 0x3, 0xb0, 0x18, 0x29, 0x1e,
                0x7e, 0x17, 0x83, 0xf5, 0x70, 0x6c, 0xac, 0xfe, 0xc9, 0x27, 0xdb, 0xb6, 0x6f, 0xdb,
                0xb6, 0x6d, 0xdb, 0x80, 0x24, 0xb9, 0xbb, 0xbb, 0x24, 0x49, 0x92, 0x24, 0xf, 0x2,
                0xd8, 0x36, 0x0, 0xf0, 0x3, 0x0, 0x0, 0x24, 0xd0, 0xb6, 0x6d, 0xdb, 0xb6, 0x6d,
                0xdb, 0xbe, 0x6d, 0xf9, 0x13, 0x4, 0xc7, 0x4, 0x0, 0x80, 0x30, 0x0, 0xc3, 0x22,
                0x68, 0xf, 0x36, 0x90, 0xc2, 0xb5, 0xfa, 0x7f, 0x48, 0x80, 0x81, 0xb, 0x40, 0x55,
                0x55, 0x55, 0xd5, 0x16, 0x80, 0xaa, 0x7, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                0xe, 0x7c, 0x82, 0xe0, 0x98, 0x0, 0x0, 0x0, 0x4, 0x60, 0x10, 0xf9, 0x8c, 0xe2,
                0xe5, 0xfa, 0x3f, 0x2, 0x54, 0x55, 0x55, 0x65, 0x0, 0xa8, 0xaa, 0xaa, 0xaa, 0xba,
                0x2, 0x50, 0xb5, 0x90, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x78, 0x82, 0xe0, 0xd0,
                0x8a, 0x41, 0x0, 0x0, 0xa2, 0x58, 0x54, 0xb7, 0x60, 0x83, 0x9a, 0x6a, 0x4, 0x96,
                0x87, 0xba, 0x51, 0xf8, 0xfb, 0x9b, 0x26, 0xfc, 0x0, 0x1c, 0x7, 0x6c, 0xdb, 0xb6,
                0x6d, 0xdb, 0xb6, 0x6d, 0xf7, 0xa8, 0x3a, 0xaf, 0xaa, 0x6a, 0x3, 0xf8, 0xc2, 0x3,
                0x40, 0x55, 0x55, 0x55, 0xd5, 0x5b, 0xf8, 0x80, 0xaa, 0x7a, 0xb, 0x0, 0x7f, 0x82,
                0xe0, 0x98, 0x0, 0x40, 0x18, 0x0, 0x82, 0xd8, 0x49, 0x40, 0x2, 0x22, 0x7e, 0xeb,
                0x80, 0xa6, 0xc, 0xa0, 0x9f, 0xa4, 0x2a, 0x38, 0xf, 0x0, 0x0, 0xe7, 0x1, 0xdc,
                0x55, 0x95, 0x17, 0x0, 0x0, 0xae, 0x0, 0x38, 0xc0, 0x67, 0xdb, 0x36, 0x80, 0x2b,
                0x0, 0xe, 0xf0, 0xd9, 0xf6, 0x13, 0x4, 0xc7, 0x4, 0x0, 0x0, 0x30, 0xc, 0x83, 0x22,
                0x69, 0x7, 0xc6, 0xea, 0xff, 0x19, 0x0, 0x0, 0x80, 0xaa, 0x0, 0x0, 0x0, 0x0, 0x0,
                0x0, 0x8e, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0x6a,
                0xf5, 0x63, 0x60, 0x60, 0x3, 0x0, 0xee, 0x8a, 0x88, 0x67,
            ],
        )
    }

    #[test]
    fn gzip_no_header() {
        let config = DeflateConfig {
            level: 9,
            method: Method::Deflated,
            window_bits: 31, // gzip
            ..Default::default()
        };

        let input = b"Hello World!";
        let os = gz_header::OS_CODE;

        fuzz_based_test(
            input,
            config,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 2, os, 243, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 4, 0, 163, 28, 41, 28, 12, 0, 0, 0,
            ],
        )
    }

    #[test]
    #[rustfmt::skip]
    fn gzip_stored_block_checksum() {
        fuzz_based_test(
            &[
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 9, 0,
            ],
            DeflateConfig {
                level: 0,
                method: Method::Deflated,
                window_bits: 26,
                mem_level: 6,
                strategy: Strategy::Default,
            },
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 4, gz_header::OS_CODE, 1, 18, 0, 237, 255, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 9, 0, 60, 101, 156, 55, 18, 0, 0, 0,
            ],
        )
    }

    #[test]
    fn gzip_header_pending_flush() {
        let extra = "aaaaaaaaaaaaaaaaaaaa\0";
        let name = "bbbbbbbbbbbbbbbbbbbb\0";
        let comment = "cccccccccccccccccccc\0";

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_ptr() as *mut _,
            extra_len: extra.len() as _,
            extra_max: 0,
            name: name.as_ptr() as *mut _,
            name_max: 0,
            comment: comment.as_ptr() as *mut _,
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            mem_level: 1,
            ..Default::default()
        };

        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, config), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&mut stream) }) else {
            unreachable!()
        };

        unsafe { set_header(stream, Some(&mut header)) };

        let input = b"Hello World\n";
        stream.next_in = input.as_ptr() as *mut _;
        stream.avail_in = input.len() as _;

        let mut output = [0u8; 1024];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = 100;

        assert_eq!(stream.state.bit_writer.pending.capacity(), 512);

        // only 12 bytes remain, so to write the name the pending buffer must be flushed.
        // but there is insufficient output space to flush (only 100 bytes)
        stream.state.bit_writer.pending.extend(&[0; 500]);

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::Ok);

        // now try that again but with sufficient output space
        stream.avail_out = output.len() as _;
        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let n = stream.total_out as usize;

        assert!(end(stream).is_ok());

        let output_rs = &mut output[..n];

        assert_eq!(output_rs.len(), 500 + 99);
    }

    #[test]
    fn gzip_with_header() {
        // this test is here mostly so we get some MIRI action on the gzip header. A test that
        // compares behavior with zlib-ng is in the libz-rs-sys test suite

        let extra = "some extra stuff\0";
        let name = "nomen est omen\0";
        let comment = "such comment\0";

        let mut header = gz_header {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: extra.as_ptr() as *mut _,
            extra_len: extra.len() as _,
            extra_max: 0,
            name: name.as_ptr() as *mut _,
            name_max: 0,
            comment: comment.as_ptr() as *mut _,
            comm_max: 0,
            hcrc: 1,
            done: 0,
        };

        let config = DeflateConfig {
            window_bits: 31,
            ..Default::default()
        };

        let mut stream = z_stream::default();
        assert_eq!(init(&mut stream, config), ReturnCode::Ok);

        let Some(stream) = (unsafe { DeflateStream::from_stream_mut(&mut stream) }) else {
            unreachable!()
        };

        unsafe { set_header(stream, Some(&mut header)) };

        let input = b"Hello World\n";
        stream.next_in = input.as_ptr() as *mut _;
        stream.avail_in = input.len() as _;

        let mut output = [0u8; 256];
        stream.next_out = output.as_mut_ptr();
        stream.avail_out = output.len() as _;

        assert_eq!(deflate(stream, DeflateFlush::Finish), ReturnCode::StreamEnd);

        let n = stream.total_out as usize;

        assert!(end(stream).is_ok());

        let output_rs = &mut output[..n];

        assert_eq!(output_rs.len(), 81);

        {
            let mut stream = z_stream::default();

            let config = InflateConfig {
                window_bits: config.window_bits,
            };

            assert_eq!(crate::inflate::init(&mut stream, config), ReturnCode::Ok);

            let Some(stream) = (unsafe { InflateStream::from_stream_mut(&mut stream) }) else {
                unreachable!();
            };

            stream.next_in = output_rs.as_mut_ptr() as _;
            stream.avail_in = output_rs.len() as _;

            let mut output = [0u8; 12];
            stream.next_out = output.as_mut_ptr();
            stream.avail_out = output.len() as _;

            let mut extra_buf = [0u8; 64];
            let mut name_buf = [0u8; 64];
            let mut comment_buf = [0u8; 64];

            let mut header = gz_header {
                text: 0,
                time: 0,
                xflags: 0,
                os: 0,
                extra: extra_buf.as_mut_ptr(),
                extra_len: 0,
                extra_max: extra_buf.len() as _,
                name: name_buf.as_mut_ptr(),
                name_max: name_buf.len() as _,
                comment: comment_buf.as_mut_ptr(),
                comm_max: comment_buf.len() as _,
                hcrc: 0,
                done: 0,
            };

            assert_eq!(
                unsafe { crate::inflate::get_header(stream, Some(&mut header)) },
                ReturnCode::Ok
            );

            assert_eq!(
                unsafe { crate::inflate::inflate(stream, InflateFlush::Finish) },
                ReturnCode::StreamEnd
            );

            crate::inflate::end(stream);

            assert!(!header.comment.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.comment.cast()) }
                    .to_str()
                    .unwrap(),
                comment.trim_end_matches('\0')
            );

            assert!(!header.name.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.name.cast()) }
                    .to_str()
                    .unwrap(),
                name.trim_end_matches('\0')
            );

            assert!(!header.extra.is_null());
            assert_eq!(
                unsafe { CStr::from_ptr(header.extra.cast()) }
                    .to_str()
                    .unwrap(),
                extra.trim_end_matches('\0')
            );
        }
    }

    #[test]
    fn insufficient_compress_space() {
        const DATA: &[u8] = include_bytes!("deflate/test-data/inflate_buf_error.dat");

        fn helper(deflate_buf: &mut [u8], deflate_err: ReturnCode) -> ReturnCode {
            let config = DeflateConfig {
                level: 0,
                method: Method::Deflated,
                window_bits: 10,
                mem_level: 6,
                strategy: Strategy::Default,
            };

            let (output, err) = compress_slice(deflate_buf, DATA, config);
            assert_eq!(err, deflate_err);

            let config = InflateConfig {
                window_bits: config.window_bits,
            };

            let mut uncompr = [0; 1 << 17];
            let (uncompr, err) = decompress_slice(&mut uncompr, output, config);

            if err == ReturnCode::Ok {
                assert_eq!(DATA, uncompr);
            }

            err
        }

        let mut output = [0; 1 << 17];

        // this is too little space
        assert_eq!(
            helper(&mut output[..1 << 16], ReturnCode::BufError),
            ReturnCode::DataError
        );

        // this is sufficient space
        assert_eq!(helper(&mut output, ReturnCode::Ok), ReturnCode::Ok);
    }

    fn test_flush(flush: DeflateFlush, expected: &[u8]) {
        let input = b"Hello World!\n";

        let config = DeflateConfig {
            level: 6, // use gzip
            method: Method::Deflated,
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        let mut output_rs = vec![0; 128];

        // with the flush modes that we test here, the deflate process still has `Status::Busy`,
        // and the `deflate` function will return `BufError` because more input is needed before
        // the flush can occur.
        let expected_err = ReturnCode::BufError;

        let (rs, err) = compress_slice_with_flush(&mut output_rs, input, config, flush);
        assert_eq!(expected_err, err);

        assert_eq!(rs, expected);
    }

    #[test]
    #[rustfmt::skip]
    fn sync_flush() {
        test_flush(
            DeflateFlush::SyncFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 0, 0, 0, 255, 255,
            ],
        )
    }

    #[test]
    #[rustfmt::skip]
    fn partial_flush() {
        test_flush(
            DeflateFlush::PartialFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 8,
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn full_flush() {
        test_flush(
            DeflateFlush::FullFlush,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2, 0, 0, 0, 255, 255,
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn block_flush() {
        test_flush(
            DeflateFlush::Block,
            &[
                31, 139, 8, 0, 0, 0, 0, 0, 0, gz_header::OS_CODE, 242, 72, 205, 201, 201, 87, 8, 207, 47, 202, 73,
                81, 228, 2,
            ],
        );
    }

    #[test]
    // splits the input into two, deflates them seperately and then joins the deflated byte streams
    // into something that can be correctly inflated again. This is the basic idea behind pigz, and
    // allows for parallel compression.
    fn split_deflate() {
        let input = "Hello World!\n";

        let (input1, input2) = input.split_at(6);

        let mut output1 = vec![0; 128];
        let mut output2 = vec![0; 128];

        let config = DeflateConfig {
            level: 6, // use gzip
            method: Method::Deflated,
            window_bits: 16 + crate::MAX_WBITS,
            mem_level: DEF_MEM_LEVEL,
            strategy: Strategy::Default,
        };

        // see also the docs on `SyncFlush`. it makes sure everything is flushed, ends on a byte
        // boundary, and that the final block does not have the "last block" bit set.
        let (prefix, err) = compress_slice_with_flush(
            &mut output1,
            input1.as_bytes(),
            config,
            DeflateFlush::SyncFlush,
        );
        assert_eq!(err, ReturnCode::BufError);

        let (output2, err) = compress_slice_with_flush(
            &mut output2,
            input2.as_bytes(),
            config,
            DeflateFlush::Finish,
        );
        assert_eq!(err, ReturnCode::Ok);

        let inflate_config = crate::inflate::InflateConfig {
            window_bits: 16 + 15,
        };

        // cuts off the length and crc
        let (suffix, end) = output2.split_at(output2.len() - 8);
        let (crc2, len2) = end.split_at(4);
        let crc2 = u32::from_le_bytes(crc2.try_into().unwrap());

        // cuts off the gzip header (10 bytes) from the front
        let suffix = &suffix[10..];

        let mut result: Vec<u8> = Vec::new();
        result.extend(prefix.iter());
        result.extend(suffix);

        // it would be more proper to use `stream.total_in` here, but the slice helpers hide the
        // stream so we're cheating a bit here
        let len1 = input1.len() as u32;
        let len2 = u32::from_le_bytes(len2.try_into().unwrap());
        assert_eq!(len2 as usize, input2.len());

        let crc1 = crate::crc32::crc32(0, input1.as_bytes());
        let crc = crate::crc32::crc32_combine(crc1, crc2, len2 as u64);

        // combined crc of the parts should be the crc of the whole
        let crc_cheating = crate::crc32::crc32(0, input.as_bytes());
        assert_eq!(crc, crc_cheating);

        // write the trailer
        result.extend(crc.to_le_bytes());
        result.extend((len1 + len2).to_le_bytes());

        let mut output = vec![0; 128];
        let (output, err) = crate::inflate::decompress_slice(&mut output, &result, inflate_config);
        assert_eq!(err, ReturnCode::Ok);

        assert_eq!(output, input.as_bytes());
    }

    #[test]
    fn inflate_window_copy_slice() {
        let uncompressed = [
            9, 126, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 76, 33, 8, 2, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12,
            10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 76, 33, 8, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 12, 10, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 9, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 12, 28, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 12, 10, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 14, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 9, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 12, 28, 0, 2, 0, 0, 0, 63, 1, 0, 12, 2,
            36, 0, 28, 0, 0, 0, 1, 0, 0, 63, 63, 13, 0, 0, 0, 0, 0, 0, 0, 63, 63, 63, 63, 0, 0, 0,
            0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 12, 33, 2, 0, 0, 8,
            0, 4, 0, 0, 0, 12, 10, 41, 12, 10, 47,
        ];

        let compressed = &[
            31, 139, 8, 0, 0, 0, 0, 0, 4, 3, 181, 193, 49, 14, 194, 32, 24, 128, 209, 175, 192, 0,
            228, 151, 232, 206, 66, 226, 226, 96, 60, 2, 113, 96, 235, 13, 188, 139, 103, 23, 106,
            104, 108, 100, 49, 169, 239, 185, 39, 11, 199, 7, 51, 39, 171, 248, 118, 226, 63, 52,
            157, 120, 86, 102, 78, 86, 209, 104, 58, 241, 84, 129, 166, 12, 4, 154, 178, 229, 202,
            30, 36, 130, 166, 19, 79, 21, 104, 202, 64, 160, 41, 91, 174, 236, 65, 34, 10, 200, 19,
            162, 206, 68, 96, 130, 156, 15, 188, 229, 138, 197, 157, 161, 35, 3, 87, 126, 245, 0,
            28, 224, 64, 146, 2, 139, 1, 196, 95, 196, 223, 94, 10, 96, 92, 33, 86, 2, 0, 0,
        ];

        let config = InflateConfig { window_bits: 25 };

        let mut dest_vec_rs = vec![0u8; uncompressed.len()];
        let (output_rs, error) =
            crate::inflate::decompress_slice(&mut dest_vec_rs, compressed, config);

        assert_eq!(ReturnCode::Ok, error);
        assert_eq!(output_rs, uncompressed);
    }

    #[test]
    fn hash_calc_difference() {
        let input = [
            0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0,
            0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 64, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 112, 102, 102, 102, 102, 102, 102, 102, 102,
            102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 0,
            0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 64, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0,
            50, 0,
        ];

        let config = DeflateConfig {
            level: 6,
            method: Method::Deflated,
            window_bits: 9,
            mem_level: 8,
            strategy: Strategy::Default,
        };

        let expected = [
            24, 149, 99, 96, 96, 96, 96, 208, 6, 17, 112, 138, 129, 193, 128, 1, 29, 24, 50, 208,
            1, 200, 146, 169, 79, 24, 74, 59, 96, 147, 52, 71, 22, 70, 246, 88, 26, 94, 80, 128,
            83, 6, 162, 219, 144, 76, 183, 210, 5, 8, 67, 105, 36, 159, 35, 128, 57, 118, 97, 100,
            160, 197, 192, 192, 96, 196, 0, 0, 3, 228, 25, 128,
        ];

        fuzz_based_test(&input, config, &expected);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    mod _cache_lines {
        use super::State;
        // FIXME: once zlib-rs Minimum Supported Rust Version >= 1.77, switch to core::mem::offset_of
        // and move this _cache_lines module from up a level from tests to super::
        use memoffset::offset_of;

        const _: () = assert!(offset_of!(State, status) == 0);
        const _: () = assert!(offset_of!(State, _cache_line_0) == 64);
        const _: () = assert!(offset_of!(State, _cache_line_1) == 128);
        #[cfg(not(feature = "ZLIB_DEBUG"))] // ZLIB_DEBUG adds 16 bytes
        const _: () = assert!(offset_of!(State, _cache_line_2) == 192);
        #[cfg(not(feature = "ZLIB_DEBUG"))] // ZLIB_DEBUG adds 16 bytes
        const _: () = assert!(offset_of!(State, _cache_line_3) == 256);
    }
}
