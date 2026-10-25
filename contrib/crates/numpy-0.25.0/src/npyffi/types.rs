use pyo3::ffi::{Py_hash_t, Py_intptr_t, Py_uintptr_t};
use std::os::raw::*;

pub type npy_intp = Py_intptr_t;
pub type npy_uintp = Py_uintptr_t;
pub type npy_longlong = c_longlong;
pub type npy_ulonglong = c_ulonglong;
pub type npy_bool = c_uchar;
pub type npy_longdouble = f64;
pub type npy_byte = c_char;
pub type npy_ubyte = c_uchar;
pub type npy_ushort = c_ushort;
pub type npy_uint = c_uint;
pub type npy_ulong = c_ulong;
pub type npy_char = c_char;
pub type npy_short = c_short;
pub type npy_int = c_int;
pub type npy_long = c_long;
pub type npy_float = f32;
pub type npy_double = f64;
pub type npy_hash_t = Py_hash_t;
pub type npy_int64 = i64;
pub type npy_uint64 = u64;
pub type npy_int32 = i32;
pub type npy_uint32 = u32;
pub type npy_ucs4 = c_uint;
pub type npy_int16 = i16;
pub type npy_uint16 = u16;
pub type npy_int8 = i8;
pub type npy_uint8 = u8;
pub type npy_float64 = f64;
pub type npy_complex128 = npy_cdouble;
pub type npy_float32 = f32;
pub type npy_complex64 = npy_cfloat;
pub type npy_half = npy_uint16;
pub type npy_float16 = npy_half;
pub type npy_float128 = npy_longdouble;
pub type npy_complex256 = npy_clongdouble;
pub type npy_timedelta = npy_int64;
pub type npy_datetime = npy_int64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_cdouble {
    pub real: f64,
    pub imag: f64,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_cfloat {
    pub real: f32,
    pub imag: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_clongdouble {
    pub real: npy_longdouble,
    pub imag: npy_longdouble,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_ORDER {
    NPY_ANYORDER = -1,
    NPY_CORDER = 0,
    NPY_FORTRANORDER = 1,
    NPY_KEEPORDER = 2,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_SCALARKIND {
    NPY_NOSCALAR = -1,
    NPY_BOOL_SCALAR = 0,
    NPY_INTPOS_SCALAR = 1,
    NPY_INTNEG_SCALAR = 2,
    NPY_FLOAT_SCALAR = 3,
    NPY_COMPLEX_SCALAR = 4,
    NPY_OBJECT_SCALAR = 5,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_SORTKIND {
    NPY_QUICKSORT = 0,
    NPY_HEAPSORT = 1,
    NPY_MERGESORT = 2,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_SEARCHSIDE {
    NPY_SEARCHLEFT = 0,
    NPY_SEARCHRIGHT = 1,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_DATETIMEUNIT {
    NPY_FR_Y = 0,
    NPY_FR_M = 1,
    NPY_FR_W = 2,
    NPY_FR_D = 4,
    NPY_FR_h = 5,
    NPY_FR_m = 6,
    NPY_FR_s = 7,
    NPY_FR_ms = 8,
    NPY_FR_us = 9,
    NPY_FR_ns = 10,
    NPY_FR_ps = 11,
    NPY_FR_fs = 12,
    NPY_FR_as = 13,
    NPY_FR_GENERIC = 14,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NPY_TYPES {
    NPY_BOOL = 0,
    NPY_BYTE = 1,
    NPY_UBYTE = 2,
    NPY_SHORT = 3,
    NPY_USHORT = 4,
    NPY_INT = 5,
    NPY_UINT = 6,
    NPY_LONG = 7,
    NPY_ULONG = 8,
    NPY_LONGLONG = 9,
    NPY_ULONGLONG = 10,
    NPY_FLOAT = 11,
    NPY_DOUBLE = 12,
    NPY_LONGDOUBLE = 13,
    NPY_CFLOAT = 14,
    NPY_CDOUBLE = 15,
    NPY_CLONGDOUBLE = 16,
    NPY_OBJECT = 17,
    NPY_STRING = 18,
    NPY_UNICODE = 19,
    NPY_VOID = 20,
    NPY_DATETIME = 21,
    NPY_TIMEDELTA = 22,
    NPY_HALF = 23,
    NPY_NTYPES_LEGACY = 24,
    NPY_NOTYPE = 25,
    NPY_CHAR = 26,
    NPY_USERDEF = 256,
    NPY_VSTRING = 2056,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_SELECTKIND {
    NPY_INTROSELECT = 0,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_CASTING {
    NPY_NO_CASTING = 0,
    NPY_EQUIV_CASTING = 1,
    NPY_SAFE_CASTING = 2,
    NPY_SAME_KIND_CASTING = 3,
    NPY_UNSAFE_CASTING = 4,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_CLIPMODE {
    NPY_CLIP = 0,
    NPY_WRAP = 1,
    NPY_RAISE = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_datetimestruct {
    pub year: npy_int64,
    pub month: npy_int32,
    pub day: npy_int32,
    pub hour: npy_int32,
    pub min: npy_int32,
    pub sec: npy_int32,
    pub us: npy_int32,
    pub ps: npy_int32,
    pub as_: npy_int32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_timedeltastruct {
    pub day: npy_int64,
    pub sec: npy_int32,
    pub us: npy_int32,
    pub ps: npy_int32,
    pub as_: npy_int32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_stride_sort_item {
    pub perm: npy_intp,
    pub stride: npy_intp,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum NPY_TYPECHAR {
    NPY_BOOLLTR = b'?',
    NPY_BYTELTR = b'b',
    NPY_UBYTELTR = b'B',
    NPY_SHORTLTR = b'h',
    NPY_USHORTLTR = b'H',
    NPY_INTLTR = b'i',
    NPY_UINTLTR = b'I',
    NPY_LONGLTR = b'l',
    NPY_ULONGLTR = b'L',
    NPY_LONGLONGLTR = b'q',
    NPY_ULONGLONGLTR = b'Q',
    NPY_HALFLTR = b'e',
    NPY_FLOATLTR = b'f',
    NPY_DOUBLELTR = b'd',
    NPY_LONGDOUBLELTR = b'g',
    NPY_CFLOATLTR = b'F',
    NPY_CDOUBLELTR = b'D',
    NPY_CLONGDOUBLELTR = b'G',
    NPY_OBJECTLTR = b'O',
    NPY_STRINGLTR = b'S',
    NPY_STRINGLTR2 = b'a',
    NPY_UNICODELTR = b'U',
    NPY_VOIDLTR = b'V',
    NPY_DATETIMELTR = b'M',
    NPY_TIMEDELTALTR = b'm',
    NPY_CHARLTR = b'c',
    NPY_INTPLTR = b'p',
    NPY_UINTPLTR = b'P',
}

// Note: NPY_TYPEKINDCHAR doesn't exist in the header and has been created here artificially
// because the original C enum contained duplicate values - namely, those related to type kinds.
// There's also a comment in the C code preceding this block of values and stating that they are
// related to type kind chars and not type codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum NPY_TYPEKINDCHAR {
    NPY_GENBOOLLTR = b'b',
    NPY_SIGNEDLTR = b'i',
    NPY_UNSIGNEDLTR = b'u',
    NPY_FLOATINGLTR = b'f',
    NPY_COMPLEXLTR = b'c',
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_BYTEORDER_CHAR {
    NPY_LITTLE = b'<',
    NPY_BIG = b'>',
    NPY_NATIVE = b'=',
    NPY_SWAP = b's',
    NPY_IGNORE = b'|',
}

impl NPY_BYTEORDER_CHAR {
    #[cfg(target_endian = "little")]
    pub const NPY_NATBYTE: Self = Self::NPY_LITTLE;
    #[cfg(target_endian = "little")]
    pub const NPY_OPPBYTE: Self = Self::NPY_BIG;

    #[cfg(target_endian = "big")]
    pub const NPY_NATBYTE: Self = Self::NPY_BIG;
    #[cfg(target_endian = "big")]
    pub const NPY_OPPBYTE: Self = Self::NPY_LITTLE;
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_ARRAYMETHOD_FLAGS {
    NPY_METH_REQUIRES_PYAPI = 1 << 0,
    NPY_METH_NO_FLOATINGPOINT_ERRORS = 1 << 1,
    NPY_METH_SUPPORTS_UNALIGNED = 1 << 2,
    NPY_METH_IS_REORDERABLE = 1 << 3,
    _NPY_METH_FORCE_CAST_INPUTS = 1 << 17,
    NPY_METH_RUNTIME_FLAGS = (1 << 0) | (1 << 1), // NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS
}
