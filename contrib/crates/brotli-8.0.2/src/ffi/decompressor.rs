pub use brotli_decompressor::ffi::interface::{brotli_alloc_func, brotli_free_func, c_void};
pub use brotli_decompressor::{ffi, BrotliDecoderReturnInfo, HuffmanCode};

pub unsafe extern "C" fn CBrotliDecoderCreateInstance(
    alloc_func: brotli_alloc_func,
    free_func: brotli_free_func,
    opaque: *mut c_void,
) -> *mut ffi::BrotliDecoderState {
    ffi::BrotliDecoderCreateInstance(alloc_func, free_func, opaque)
}

pub unsafe extern "C" fn CBrotliDecoderSetParameter(
    state_ptr: *mut ffi::BrotliDecoderState,
    selector: ffi::interface::BrotliDecoderParameter,
    value: u32,
) {
    ffi::BrotliDecoderSetParameter(state_ptr, selector, value)
}

#[cfg(feature = "std")] // this requires a default allocator
pub unsafe extern "C" fn CBrotliDecoderDecompress(
    encoded_size: usize,
    encoded_buffer: *const u8,
    decoded_size: *mut usize,
    decoded_buffer: *mut u8,
) -> ffi::interface::BrotliDecoderResult {
    ffi::BrotliDecoderDecompress(encoded_size, encoded_buffer, decoded_size, decoded_buffer)
}

pub unsafe extern "C" fn CBrotliDecoderDecompressStream(
    state_ptr: *mut ffi::BrotliDecoderState,
    available_in: *mut usize,
    input_buf_ptr: *mut *const u8,
    available_out: *mut usize,
    output_buf_ptr: *mut *mut u8,
    total_out: *mut usize,
) -> ffi::interface::BrotliDecoderResult {
    ffi::BrotliDecoderDecompressStream(
        state_ptr,
        available_in,
        input_buf_ptr,
        available_out,
        output_buf_ptr,
        total_out,
    )
}

pub unsafe extern "C" fn CBrotliDecoderDecompressStreaming(
    state_ptr: *mut ffi::BrotliDecoderState,
    available_in: *mut usize,
    input_buf_ptr: *const u8,
    available_out: *mut usize,
    output_buf_ptr: *mut u8,
) -> ffi::interface::BrotliDecoderResult {
    ffi::BrotliDecoderDecompressStreaming(
        state_ptr,
        available_in,
        input_buf_ptr,
        available_out,
        output_buf_ptr,
    )
}

pub unsafe extern "C" fn CBrotliDecoderDecompressWithReturnInfo(
    available_in: usize,
    input_buf_ptr: *const u8,
    available_out_and_scratch: usize,
    output_buf_and_scratch: *mut u8,
) -> BrotliDecoderReturnInfo {
    ffi::BrotliDecoderDecompressWithReturnInfo(
        available_in,
        input_buf_ptr,
        available_out_and_scratch,
        output_buf_and_scratch,
    )
}

pub unsafe extern "C" fn CBrotliDecoderDecompressPrealloc(
    available_in: usize,
    input_buf_ptr: *const u8,
    available_out: usize,
    output_buf_ptr: *mut u8,
    available_u8: usize,
    u8_ptr: *mut u8,
    available_u32: usize,
    u32_ptr: *mut u32,
    available_hc: usize,
    hc_ptr: *mut HuffmanCode,
) -> BrotliDecoderReturnInfo {
    ffi::BrotliDecoderDecompressPrealloc(
        available_in,
        input_buf_ptr,
        available_out,
        output_buf_ptr,
        available_u8,
        u8_ptr,
        available_u32,
        u32_ptr,
        available_hc,
        hc_ptr,
    )
}

pub unsafe extern "C" fn CBrotliDecoderMallocU8(
    state_ptr: *mut ffi::BrotliDecoderState,
    size: usize,
) -> *mut u8 {
    ffi::BrotliDecoderMallocU8(state_ptr, size)
}

pub unsafe extern "C" fn CBrotliDecoderFreeU8(
    state_ptr: *mut ffi::BrotliDecoderState,
    data: *mut u8,
    size: usize,
) {
    ffi::BrotliDecoderFreeU8(state_ptr, data, size)
}

pub unsafe extern "C" fn CBrotliDecoderMallocUsize(
    state_ptr: *mut ffi::BrotliDecoderState,
    size: usize,
) -> *mut usize {
    ffi::BrotliDecoderMallocUsize(state_ptr, size)
}

pub unsafe extern "C" fn CBrotliDecoderFreeUsize(
    state_ptr: *mut ffi::BrotliDecoderState,
    data: *mut usize,
    size: usize,
) {
    ffi::BrotliDecoderFreeUsize(state_ptr, data, size)
}

pub unsafe extern "C" fn CBrotliDecoderDestroyInstance(state_ptr: *mut ffi::BrotliDecoderState) {
    ffi::BrotliDecoderDestroyInstance(state_ptr)
}

pub extern "C" fn CBrotliDecoderVersion() -> u32 {
    ffi::BrotliDecoderVersion()
}

#[no_mangle]
pub extern "C" fn CBrotliDecoderErrorString(c: ffi::BrotliDecoderErrorCode) -> *const u8 {
    ffi::BrotliDecoderErrorString(c)
}

#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderHasMoreOutput(
    state_ptr: *const ffi::BrotliDecoderState,
) -> i32 {
    ffi::BrotliDecoderHasMoreOutput(state_ptr)
}

#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderTakeOutput(
    state_ptr: *mut ffi::BrotliDecoderState,
    size: *mut usize,
) -> *const u8 {
    ffi::BrotliDecoderTakeOutput(state_ptr, size)
}

#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderIsUsed(state_ptr: *const ffi::BrotliDecoderState) -> i32 {
    ffi::BrotliDecoderIsUsed(state_ptr)
}
#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderIsFinished(
    state_ptr: *const ffi::BrotliDecoderState,
) -> i32 {
    ffi::BrotliDecoderIsFinished(state_ptr)
}
#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderGetErrorCode(
    state_ptr: *const ffi::BrotliDecoderState,
) -> ffi::BrotliDecoderErrorCode {
    ffi::BrotliDecoderGetErrorCode(state_ptr)
}
#[no_mangle]
pub unsafe extern "C" fn CBrotliDecoderGetErrorString(
    state_ptr: *const ffi::BrotliDecoderState,
) -> *const u8 {
    ffi::BrotliDecoderGetErrorString(state_ptr)
}
