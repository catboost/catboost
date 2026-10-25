#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::ffi::{c_char, c_int, c_uchar, c_uint, c_ulong, c_void};

use crate::allocate::Allocator;

pub type alloc_func = unsafe extern "C" fn(voidpf, uInt, uInt) -> voidpf;
pub type free_func = unsafe extern "C" fn(voidpf, voidpf);

pub type Bytef = u8;
pub type in_func = unsafe extern "C" fn(*mut c_void, *mut *const c_uchar) -> c_uint;
pub type out_func = unsafe extern "C" fn(*mut c_void, *mut c_uchar, c_uint) -> c_int;
pub type uInt = c_uint;
pub type uLong = c_ulong;
pub type uLongf = c_ulong;
pub type voidp = *mut c_void;
pub type voidpc = *const c_void;
pub type voidpf = *mut c_void;

/// The current stream state
///
/// # Custom allocators
///
/// The low-level API supports passing in a custom allocator as part of the [`z_stream`]:
///
/// ```no_check
/// struct z_stream {
///     // ...
///     zalloc: Option<unsafe extern "C" fn(*mut c_void, c_uint, c_uint) -> *mut c_void>,
///     zfree: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
///     opaque: *mut c_void,
/// }
/// ```
///
/// When these fields are `None` (or `NULL` in C), the initialization functions use a default allocator,
/// based on feature flags:
///
/// - `"rust-allocator"` uses the rust global allocator
/// - `"c-allocator"` uses an allocator based on `malloc` and `free`
///
/// When both are configured, the `"rust-allocator"` is preferred. When no default allocator is configured,
/// and custom `zalloc` and `zfree` are provided, the initialization functions will return a [`Z_STREAM_ERROR`].
///
/// When custom `zalloc` and `zfree` functions are given, they must adhere to the following contract
/// to be safe:
///
/// - a call `zalloc(opaque, n, m)` must return a pointer `p` to `n * m` bytes of memory, or
///   `NULL` if out of memory
/// - a call `zfree(opaque, p)` must free that memory
///
/// The `strm.opaque` value is passed to as the first argument to all calls to `zalloc`
/// and `zfree`, but is otherwise ignored by the library.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct z_stream {
    pub next_in: *const Bytef,
    pub avail_in: uInt,
    pub total_in: z_size,
    pub next_out: *mut Bytef,
    pub avail_out: uInt,
    pub total_out: z_size,
    pub msg: *mut c_char,
    pub state: *mut internal_state,
    pub zalloc: Option<alloc_func>,
    pub zfree: Option<free_func>,
    pub opaque: voidpf,
    pub data_type: c_int,
    pub adler: z_checksum,
    pub reserved: uLong,
}
pub type z_streamp = *mut z_stream;

impl Default for z_stream {
    fn default() -> Self {
        let mut stream = Self {
            next_in: core::ptr::null_mut(),
            avail_in: 0,
            total_in: 0,
            next_out: core::ptr::null_mut(),
            avail_out: 0,
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

        #[cfg(feature = "rust-allocator")]
        if stream.zalloc.is_none() || stream.zfree.is_none() {
            stream.configure_default_rust_allocator()
        }

        #[cfg(feature = "c-allocator")]
        if stream.zalloc.is_none() || stream.zfree.is_none() {
            stream.configure_default_c_allocator()
        }

        stream
    }
}

impl z_stream {
    fn configure_allocator(&mut self, alloc: Allocator) {
        self.zalloc = Some(alloc.zalloc);
        self.zfree = Some(alloc.zfree);
        self.opaque = alloc.opaque;
    }

    #[cfg(feature = "rust-allocator")]
    pub fn configure_default_rust_allocator(&mut self) {
        self.configure_allocator(crate::allocate::RUST)
    }

    #[cfg(feature = "c-allocator")]
    pub fn configure_default_c_allocator(&mut self) {
        self.configure_allocator(crate::allocate::C)
    }
}

// // zlib stores Adler-32 and CRC-32 checksums in unsigned long; zlib-ng uses uint32_t.
pub(crate) type z_size = c_ulong;
pub(crate) type z_checksum = c_ulong;

// opaque to the user
pub enum internal_state {}

pub const Z_NO_FLUSH: c_int = 0;
pub const Z_PARTIAL_FLUSH: c_int = 1;
pub const Z_SYNC_FLUSH: c_int = 2;
pub const Z_FULL_FLUSH: c_int = 3;
pub const Z_FINISH: c_int = 4;
pub const Z_BLOCK: c_int = 5;
pub const Z_TREES: c_int = 6;

pub const Z_OK: c_int = 0;
pub const Z_STREAM_END: c_int = 1;
pub const Z_NEED_DICT: c_int = 2;
pub const Z_ERRNO: c_int = -1;
pub const Z_STREAM_ERROR: c_int = -2;
pub const Z_DATA_ERROR: c_int = -3;
pub const Z_MEM_ERROR: c_int = -4;
pub const Z_BUF_ERROR: c_int = -5;
pub const Z_VERSION_ERROR: c_int = -6;

pub const Z_NO_COMPRESSION: c_int = 0;
pub const Z_BEST_SPEED: c_int = 1;
pub const Z_BEST_COMPRESSION: c_int = 9;
pub const Z_DEFAULT_COMPRESSION: c_int = -1;

pub const Z_DEFLATED: c_int = 8;

pub const Z_BINARY: c_int = 0;
pub const Z_TEXT: c_int = 1;
pub const Z_ASCII: c_int = Z_TEXT; /* for compatibility with 1.2.2 and earlier */
pub const Z_UNKNOWN: c_int = 2;

pub const Z_FILTERED: c_int = 1;
pub const Z_HUFFMAN_ONLY: c_int = 2;
pub const Z_RLE: c_int = 3;
pub const Z_FIXED: c_int = 4;
pub const Z_DEFAULT_STRATEGY: c_int = 0;

pub type gz_headerp = *mut gz_header;

/// gzip header information passed to and from zlib routines.
/// See RFC 1952 for more details on the meanings of these fields.
#[derive(Debug)]
#[repr(C)]
pub struct gz_header {
    /// true if compressed data believed to be text
    pub text: i32,
    /// modification time
    pub time: c_ulong,
    /// extra flags (not used when writing a gzip file)
    pub xflags: i32,
    /// operating system
    pub os: i32,
    /// pointer to extra field or NULL if none
    pub extra: *mut u8,
    /// extra field length (valid if extra != NULL)
    pub extra_len: u32,
    /// space at extra (only when reading header)
    pub extra_max: u32,
    /// pointer to zero-terminated file name or NULL
    pub name: *mut u8,
    /// space at name (only when reading header)
    pub name_max: u32,
    /// pointer to zero-terminated comment or NULL
    pub comment: *mut u8,
    /// space at comment (only when reading header)
    pub comm_max: u32,
    /// true if there was or will be a header crc
    pub hcrc: i32,
    /// true when done reading gzip header (not used when writing a gzip file)
    pub done: i32,
}

impl Default for gz_header {
    fn default() -> Self {
        Self {
            text: 0,
            time: 0,
            xflags: 0,
            os: 0,
            extra: core::ptr::null_mut(),
            extra_len: 0,
            extra_max: 0,
            name: core::ptr::null_mut(),
            name_max: 0,
            comment: core::ptr::null_mut(),
            comm_max: 0,
            hcrc: 0,
            done: 0,
        }
    }
}

impl gz_header {
    // based on the spec https://www.ietf.org/rfc/rfc1952.txt
    //
    //   0 - FAT filesystem (MS-DOS, OS/2, NT/Win32)
    //   1 - Amiga
    //   2 - VMS (or OpenVMS)
    //   3 - Unix
    //   4 - VM/CMS
    //   5 - Atari TOS
    //   6 - HPFS filesystem (OS/2, NT)
    //   7 - Macintosh
    //   8 - Z-System
    //   9 - CP/M
    //  10 - TOPS-20
    //  11 - NTFS filesystem (NT)
    //  12 - QDOS
    //  13 - Acorn RISCOS
    // 255 - unknown
    #[allow(clippy::if_same_then_else)]
    pub const OS_CODE: u8 = {
        if cfg!(windows) {
            10
        } else if cfg!(target_os = "macos") {
            19
        } else if cfg!(unix) {
            3
        } else {
            3 // assume unix
        }
    };

    pub(crate) fn flags(&self) -> u8 {
        (if self.text != 0 { 1 } else { 0 })
            + (if self.hcrc != 0 { 2 } else { 0 })
            + (if self.extra.is_null() { 0 } else { 4 })
            + (if self.name.is_null() { 0 } else { 8 })
            + (if self.comment.is_null() { 0 } else { 16 })
    }
}
