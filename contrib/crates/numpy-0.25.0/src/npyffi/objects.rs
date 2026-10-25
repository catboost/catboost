//! Low-Lebel binding for NumPy C API C-objects
//!
//! <https://numpy.org/doc/stable/reference/c-api/types-and-structures.html>
#![allow(non_camel_case_types)]

use libc::FILE;
use pyo3::ffi::*;
use std::os::raw::*;

use super::types::*;
use crate::npyffi::*;

#[repr(C)]
pub struct PyArrayObject {
    pub ob_base: PyObject,
    pub data: *mut c_char,
    pub nd: c_int,
    pub dimensions: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub base: *mut PyObject,
    pub descr: *mut PyArray_Descr,
    pub flags: c_int,
    pub weakreflist: *mut PyObject,
}

#[repr(C)]
pub struct PyArray_Descr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub _former_flags: c_char,
    pub type_num: c_int,
}

#[repr(C)]
pub struct PyArray_DescrProto {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub flags: c_char,
    pub type_num: c_int,
    pub elsize: c_int,
    pub alignment: c_int,
    pub subarray: *mut PyArray_ArrayDescr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub f: *mut PyArray_ArrFuncs,
    pub metadata: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
    pub hash: npy_hash_t,
}

#[repr(C)]
pub struct _PyArray_DescrNumPy2 {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub _former_flags: c_char,
    pub type_num: c_int,
    pub flags: npy_uint64,
    pub elsize: npy_intp,
    pub alignment: npy_intp,
    pub metadata: *mut PyObject,
    pub hash: npy_hash_t,
    pub reserved_null: [*mut std::ffi::c_void; 2],
}

#[repr(C)]
struct _PyArray_LegacyDescr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub _former_flags: c_char,
    pub type_num: c_int,
    pub flags: npy_uint64,
    pub elsize: npy_intp,
    pub alignment: npy_intp,
    pub metadata: *mut PyObject,
    pub hash: npy_hash_t,
    pub reserved_null: [*mut std::ffi::c_void; 2],
    pub subarray: *mut PyArray_ArrayDescr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
}

#[allow(non_snake_case)]
#[inline(always)]
pub unsafe fn PyDataType_ISLEGACY(dtype: *const PyArray_Descr) -> bool {
    (*dtype).type_num < NPY_TYPES::NPY_VSTRING as _ && (*dtype).type_num >= 0
}

#[allow(non_snake_case)]
#[inline(always)]
pub unsafe fn PyDataType_SET_ELSIZE<'py>(
    py: Python<'py>,
    dtype: *mut PyArray_Descr,
    size: npy_intp,
) {
    if is_numpy_2(py) {
        unsafe {
            (*(dtype as *mut _PyArray_DescrNumPy2)).elsize = size;
        }
    } else {
        unsafe {
            (*(dtype as *mut PyArray_DescrProto)).elsize = size as c_int;
        }
    }
}

#[allow(non_snake_case)]
#[inline(always)]
pub unsafe fn PyDataType_FLAGS<'py>(py: Python<'py>, dtype: *const PyArray_Descr) -> npy_uint64 {
    if is_numpy_2(py) {
        unsafe { (*(dtype as *mut _PyArray_DescrNumPy2)).flags }
    } else {
        unsafe { (*(dtype as *mut PyArray_DescrProto)).flags as c_uchar as npy_uint64 }
    }
}

macro_rules! define_descr_accessor {
    ($name:ident, $property:ident, $type:ty, $legacy_only:literal, $default:expr) => {
        #[allow(non_snake_case)]
        #[inline(always)]
        pub unsafe fn $name<'py>(py: Python<'py>, dtype: *const PyArray_Descr) -> $type {
            if $legacy_only && !PyDataType_ISLEGACY(dtype) {
                $default
            } else {
                if is_numpy_2(py) {
                    unsafe { (*(dtype as *const _PyArray_LegacyDescr)).$property }
                } else {
                    unsafe { (*(dtype as *mut PyArray_DescrProto)).$property as $type }
                }
            }
        }
    };
}

define_descr_accessor!(PyDataType_ELSIZE, elsize, npy_intp, false, 0);
define_descr_accessor!(PyDataType_ALIGNMENT, alignment, npy_intp, false, 0);
define_descr_accessor!(
    PyDataType_METADATA,
    metadata,
    *mut PyObject,
    true,
    std::ptr::null_mut()
);
define_descr_accessor!(
    PyDataType_SUBARRAY,
    subarray,
    *mut PyArray_ArrayDescr,
    true,
    std::ptr::null_mut()
);
define_descr_accessor!(
    PyDataType_NAMES,
    names,
    *mut PyObject,
    true,
    std::ptr::null_mut()
);
define_descr_accessor!(
    PyDataType_FIELDS,
    fields,
    *mut PyObject,
    true,
    std::ptr::null_mut()
);
define_descr_accessor!(
    PyDataType_C_METADATA,
    c_metadata,
    *mut NpyAuxData,
    true,
    std::ptr::null_mut()
);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_ArrayDescr {
    pub base: *mut PyArray_Descr,
    pub shape: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_ArrFuncs {
    pub cast: [PyArray_VectorUnaryFunc; 21usize],
    pub getitem: PyArray_GetItemFunc,
    pub setitem: PyArray_SetItemFunc,
    pub copyswapn: PyArray_CopySwapNFunc,
    pub copyswap: PyArray_CopySwapFunc,
    pub compare: PyArray_CompareFunc,
    pub argmax: PyArray_ArgFunc,
    pub dotfunc: PyArray_DotFunc,
    pub scanfunc: PyArray_ScanFunc,
    pub fromstr: PyArray_FromStrFunc,
    pub nonzero: PyArray_NonzeroFunc,
    pub fill: PyArray_FillFunc,
    pub fillwithscalar: PyArray_FillWithScalarFunc,
    pub sort: [PyArray_SortFunc; 3usize],
    pub argsort: [PyArray_ArgSortFunc; 3usize],
    pub castdict: *mut PyObject,
    pub scalarkind: PyArray_ScalarKindFunc,
    pub cancastscalarkindto: *mut *mut c_int,
    pub cancastto: *mut c_int,
    pub fastclip: PyArray_FastClipFunc,
    pub fastputmask: PyArray_FastPutmaskFunc,
    pub fasttake: PyArray_FastTakeFunc,
    pub argmin: PyArray_ArgFunc,
}

pub type PyArray_GetItemFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut PyObject>;
pub type PyArray_SetItemFunc =
    Option<unsafe extern "C" fn(*mut PyObject, *mut c_void, *mut c_void) -> c_int>;
pub type PyArray_CopySwapNFunc = Option<
    unsafe extern "C" fn(
        *mut c_void,
        npy_intp,
        *mut c_void,
        npy_intp,
        npy_intp,
        c_int,
        *mut c_void,
    ),
>;
pub type PyArray_CopySwapFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, *mut c_void)>;
pub type PyArray_NonzeroFunc = Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uchar>;
pub type PyArray_CompareFunc =
    Option<unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void) -> c_int>;
pub type PyArray_ArgFunc =
    Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut npy_intp, *mut c_void) -> c_int>;
pub type PyArray_DotFunc = Option<
    unsafe extern "C" fn(
        *mut c_void,
        npy_intp,
        *mut c_void,
        npy_intp,
        *mut c_void,
        npy_intp,
        *mut c_void,
    ),
>;
pub type PyArray_VectorUnaryFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void, npy_intp, *mut c_void, *mut c_void)>;
pub type PyArray_ScanFunc =
    Option<unsafe extern "C" fn(*mut FILE, *mut c_void, *mut c_char, *mut PyArray_Descr) -> c_int>;
pub type PyArray_FromStrFunc = Option<
    unsafe extern "C" fn(*mut c_char, *mut c_void, *mut *mut c_char, *mut PyArray_Descr) -> c_int,
>;
pub type PyArray_FillFunc =
    Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void) -> c_int>;
pub type PyArray_SortFunc =
    Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void) -> c_int>;
pub type PyArray_ArgSortFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut npy_intp, npy_intp, *mut c_void) -> c_int>;
pub type PyArray_PartitionFunc = Option<
    unsafe extern "C" fn(
        *mut c_void,
        npy_intp,
        npy_intp,
        *mut npy_intp,
        *mut npy_intp,
        *mut c_void,
    ) -> c_int,
>;
pub type PyArray_ArgPartitionFunc = Option<
    unsafe extern "C" fn(
        *mut c_void,
        *mut npy_intp,
        npy_intp,
        npy_intp,
        *mut npy_intp,
        *mut npy_intp,
        *mut c_void,
    ) -> c_int,
>;
pub type PyArray_FillWithScalarFunc =
    Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void, *mut c_void) -> c_int>;
pub type PyArray_ScalarKindFunc = Option<unsafe extern "C" fn(*mut c_void) -> c_int>;
pub type PyArray_FastClipFunc =
    Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void, *mut c_void, *mut c_void)>;
pub type PyArray_FastPutmaskFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void, npy_intp, *mut c_void, npy_intp)>;
pub type PyArray_FastTakeFunc = Option<
    unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut npy_intp,
        npy_intp,
        npy_intp,
        npy_intp,
        npy_intp,
        NPY_CLIPMODE,
    ) -> c_int,
>;

#[repr(C)]
pub struct PyArrayFlagsObject {
    pub ob_base: PyObject,
    pub arr: *mut PyObject,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Dims {
    pub ptr: *mut npy_intp,
    pub len: c_int,
}

#[repr(C)]
pub struct PyArray_Chunk {
    pub ob_base: PyObject,
    pub base: *mut PyObject,
    pub ptr: *mut c_void,
    pub len: npy_intp,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayInterface {
    pub two: c_int,
    pub nd: c_int,
    pub typekind: c_char,
    pub itemsize: c_int,
    pub flags: c_int,
    pub shape: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub data: *mut c_void,
    pub descr: *mut PyObject,
}

#[repr(C)]
pub struct PyUFuncObject {
    pub ob_base: PyObject,
    pub nin: c_int,
    pub nout: c_int,
    pub nargs: c_int,
    pub identity: c_int,
    pub functions: *mut PyUFuncGenericFunction,
    pub data: *mut *mut c_void,
    pub ntypes: c_int,
    pub reserved1: c_int,
    pub name: *const c_char,
    pub types: *mut c_char,
    pub doc: *const c_char,
    pub ptr: *mut c_void,
    pub obj: *mut PyObject,
    pub userloops: *mut PyObject,
    pub core_enabled: c_int,
    pub core_num_dim_ix: c_int,
    pub core_num_dims: *mut c_int,
    pub core_dim_ixs: *mut c_int,
    pub core_offsets: *mut c_int,
    pub core_signature: *mut c_char,
    pub type_resolver: PyUFunc_TypeResolutionFunc,
    pub legacy_inner_loop_selector: PyUFunc_LegacyInnerLoopSelectionFunc,
    pub reserved2: *mut c_void,
    pub masked_inner_loop_selector: PyUFunc_MaskedInnerLoopSelectionFunc,
    pub op_flags: *mut npy_uint32,
    pub iter_flags: npy_uint32,
}

pub type PyUFuncGenericFunction =
    Option<unsafe extern "C" fn(*mut *mut c_char, *mut npy_intp, *mut npy_intp, *mut c_void)>;
pub type PyUFunc_MaskedStridedInnerLoopFunc = Option<
    unsafe extern "C" fn(
        *mut *mut c_char,
        *mut npy_intp,
        *mut c_char,
        npy_intp,
        npy_intp,
        *mut NpyAuxData,
    ),
>;
pub type PyUFunc_TypeResolutionFunc = Option<
    unsafe extern "C" fn(
        *mut PyUFuncObject,
        NPY_CASTING,
        *mut *mut PyArrayObject,
        *mut PyObject,
        *mut *mut PyArray_Descr,
    ) -> c_int,
>;
pub type PyUFunc_LegacyInnerLoopSelectionFunc = Option<
    unsafe extern "C" fn(
        *mut PyUFuncObject,
        *mut *mut PyArray_Descr,
        *mut PyUFuncGenericFunction,
        *mut *mut c_void,
        *mut c_int,
    ) -> c_int,
>;
pub type PyUFunc_MaskedInnerLoopSelectionFunc = Option<
    unsafe extern "C" fn(
        *mut PyUFuncObject,
        *mut *mut PyArray_Descr,
        *mut PyArray_Descr,
        *mut npy_intp,
        npy_intp,
        *mut PyUFunc_MaskedStridedInnerLoopFunc,
        *mut *mut NpyAuxData,
        *mut c_int,
    ) -> c_int,
>;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NpyIter([u8; 0]);

#[repr(C)]
pub struct PyArrayIterObject {
    pub ob_base: PyObject,
    pub nd_m1: c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut c_char,
    pub contiguous: npy_bool,
    pub bounds: [[npy_intp; 2usize]; 32usize],
    pub limits: [[npy_intp; 2usize]; 32usize],
    pub limits_sizes: [npy_intp; 32usize],
    pub translate: npy_iter_get_dataptr_t,
}

#[repr(C)]
pub struct PyArrayMultiIterObject {
    pub ob_base: PyObject,
    pub numiter: c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: c_int,
    pub dimensions: [npy_intp; 32usize],
    pub iters: [*mut PyArrayIterObject; 32usize],
}

#[repr(C)]
pub struct PyArrayNeighborhoodIterObject {
    pub ob_base: PyObject,
    pub nd_m1: c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut c_char,
    pub contiguous: npy_bool,
    pub bounds: [[npy_intp; 2usize]; 32usize],
    pub limits: [[npy_intp; 2usize]; 32usize],
    pub limits_sizes: [npy_intp; 32usize],
    pub translate: npy_iter_get_dataptr_t,
    pub nd: npy_intp,
    pub dimensions: [npy_intp; 32usize],
    pub _internal_iter: *mut PyArrayIterObject,
    pub constant: *mut c_char,
    pub mode: c_int,
}

#[repr(C)]
pub struct PyArrayMapIterObject {
    pub ob_base: PyObject,
    pub numiter: c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: c_int,
    pub dimensions: [npy_intp; 32usize],
    pub outer: *mut NpyIter,
    pub unused: [*mut c_void; 30usize],
    pub array: *mut PyArrayObject,
    pub ait: *mut PyArrayIterObject,
    pub subspace: *mut PyArrayObject,
    pub iteraxes: [c_int; 32usize],
    pub fancy_strides: [npy_intp; 32usize],
    pub baseoffset: *mut c_char,
    pub consec: c_int,
    pub dataptr: *mut c_char,
    pub nd_fancy: c_int,
    pub fancy_dims: [npy_intp; 32usize],
    pub needs_api: c_int,
    pub extra_op: *mut PyArrayObject,
    pub extra_op_dtype: *mut PyArray_Descr,
    pub extra_op_flags: *mut npy_uint32,
    pub extra_op_iter: *mut NpyIter,
    pub extra_op_next: NpyIter_IterNextFunc,
    pub extra_op_ptrs: *mut *mut c_char,
    pub outer_next: NpyIter_IterNextFunc,
    pub outer_ptrs: *mut *mut c_char,
    pub outer_strides: *mut npy_intp,
    pub subspace_iter: *mut NpyIter,
    pub subspace_next: NpyIter_IterNextFunc,
    pub subspace_ptrs: *mut *mut c_char,
    pub subspace_strides: *mut npy_intp,
    pub iter_count: npy_intp,
}

pub type NpyIter_IterNextFunc = Option<unsafe extern "C" fn(*mut NpyIter) -> c_int>;
pub type NpyIter_GetMultiIndexFunc = Option<unsafe extern "C" fn(*mut NpyIter, *mut npy_intp)>;
pub type PyDataMem_EventHookFunc =
    Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut c_void)>;
pub type npy_iter_get_dataptr_t =
    Option<unsafe extern "C" fn(*mut PyArrayIterObject, *mut npy_intp) -> *mut c_char>;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut c_void; 2usize],
}

pub type NpyAuxData_FreeFunc = Option<unsafe extern "C" fn(*mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc = Option<unsafe extern "C" fn(*mut NpyAuxData) -> *mut NpyAuxData>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_DatetimeMetaData {
    pub base: NPY_DATETIMEUNIT,
    pub num: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_DatetimeDTypeMetaData {
    pub base: NpyAuxData,
    pub meta: PyArray_DatetimeMetaData,
}

// npy_packed_static_string and npy_string_allocator are opaque pointers.
// FIXME(adamreichold): Consider extern types when they are stabilized.
// https://github.com/rust-lang/rust/issues/43467
pub type npy_packed_static_string = c_void;
pub type npy_string_allocator = c_void;
pub type PyArray_DTypeMeta = PyTypeObject;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct npy_static_string {
    pub size: usize,
    pub buf: *const c_char,
}

#[repr(C)]
pub struct PyArray_StringDTypeObject {
    pub base: PyArray_Descr,
    pub na_object: *mut PyObject,
    pub coerce: c_char,
    pub has_nan_na: c_char,
    pub has_string_na: c_char,
    pub array_owned: c_char,
    pub default_string: npy_static_string,
    pub na_name: npy_static_string,
    pub allocator: *mut npy_string_allocator,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayMethod_Spec {
    pub name: *const c_char,
    pub nin: c_int,
    pub nout: c_int,
    pub casting: NPY_CASTING,
    pub flags: NPY_ARRAYMETHOD_FLAGS,
    pub dtypes: *mut *mut PyArray_DTypeMeta,
    pub slots: *mut PyType_Slot,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayDTypeMeta_Spec {
    pub typeobj: *mut PyTypeObject,
    pub flags: c_int,
    pub casts: *mut *mut PyArrayMethod_Spec,
    pub slots: *mut PyType_Slot,
    pub baseclass: *mut PyTypeObject,
}
