#[cfg(feature="std")]
use std::{thread,panic, io, boxed, any, string};
#[cfg(feature="std")]
use std::io::Write;
use core;
use core::slice;
use core::ops;
pub mod interface;
pub mod alloc_util;
use self::alloc_util::SubclassableAllocator;
use alloc::{Allocator, SliceWrapper, SliceWrapperMut, StackAllocator, AllocatedStackMemory, bzero};
use self::interface::{CAllocator, c_void, BrotliDecoderParameter, BrotliDecoderResult, brotli_alloc_func, brotli_free_func};
use ::BrotliResult;
use ::BrotliDecoderReturnInfo;
use ::brotli_decode;
pub use ::HuffmanCode;
pub use super::state::{BrotliDecoderErrorCode, BrotliState};

pub unsafe fn slice_from_raw_parts_or_nil<'a, T>(data: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    slice::from_raw_parts(data, len)
}

pub unsafe fn slice_from_raw_parts_or_nil_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 {
        return &mut [];
    }
    slice::from_raw_parts_mut(data, len)
}

#[cfg(feature="std")]
type BrotliAdditionalErrorData = boxed::Box<dyn any::Any + Send + 'static>;
#[cfg(not(feature="std"))]
type BrotliAdditionalErrorData = ();

#[repr(C)]
pub struct BrotliDecoderState {
    pub custom_allocator: CAllocator,
    pub decompressor: ::BrotliState<SubclassableAllocator,
                                    SubclassableAllocator,
                                    SubclassableAllocator>,
}

#[cfg(not(feature="std"))]
fn brotli_new_decompressor_without_custom_alloc(_to_box: BrotliDecoderState) -> *mut BrotliDecoderState{
    panic!("Must supply allocators if calling divans when compiled without features=std");
}

#[cfg(feature="std")]
fn brotli_new_decompressor_without_custom_alloc(to_box: BrotliDecoderState) -> *mut BrotliDecoderState{
    alloc_util::Box::<BrotliDecoderState>::into_raw(
        alloc_util::Box::<BrotliDecoderState>::new(to_box))
}


#[no_mangle]
pub unsafe extern fn BrotliDecoderCreateInstance(
    alloc_func: brotli_alloc_func,
    free_func: brotli_free_func,
    opaque: *mut c_void,
) -> *mut BrotliDecoderState {
    match catch_panic_state(|| {
      let allocators = CAllocator {
        alloc_func:alloc_func,
        free_func:free_func,
        opaque:opaque,
      };
      let custom_dictionary = <SubclassableAllocator as Allocator<u8>>::AllocatedMemory::default();
      let to_box = BrotliDecoderState {
        custom_allocator: allocators.clone(),
        decompressor: ::BrotliState::new_with_custom_dictionary(
          SubclassableAllocator::new(allocators.clone()),
          SubclassableAllocator::new(allocators.clone()),
          SubclassableAllocator::new(allocators.clone()),
          custom_dictionary,
        ),
      };
      if let Some(alloc) = alloc_func {
        if free_func.is_none() {
            panic!("either both alloc and free must exist or neither");
        }
        let ptr = alloc(allocators.opaque, core::mem::size_of::<BrotliDecoderState>());
        let brotli_decoder_state_ptr = core::mem::transmute::<*mut c_void, *mut BrotliDecoderState>(ptr);
        core::ptr::write(brotli_decoder_state_ptr, to_box);
        brotli_decoder_state_ptr
      } else {
        brotli_new_decompressor_without_custom_alloc(to_box)
      }
    }) {
        Ok(ret) => ret,
        Err(mut e) => {
            error_print(core::ptr::null_mut(), &mut e);
            core::ptr::null_mut()
        },
    }
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderSetParameter(_state_ptr: *mut BrotliDecoderState,
                                       _selector: BrotliDecoderParameter,
                                       _value: u32) {
  // not implemented
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderDecompressPrealloc(
  encoded_size: usize,
  encoded_buffer: *const u8,
  decoded_size: usize,
  decoded_buffer: *mut u8,
  scratch_u8_size: usize,
  scratch_u8_buffer: *mut u8,
  scratch_u32_size: usize,
  scratch_u32_buffer: *mut u32,
  scratch_hc_size: usize,
  scratch_hc_buffer: *mut HuffmanCode,
) -> BrotliDecoderReturnInfo {
  let input = slice_from_raw_parts_or_nil(encoded_buffer, encoded_size);
  let output = slice_from_raw_parts_or_nil_mut(decoded_buffer, decoded_size);
  let scratch_u8 = slice_from_raw_parts_or_nil_mut(scratch_u8_buffer, scratch_u8_size);
  let scratch_u32 = slice_from_raw_parts_or_nil_mut(scratch_u32_buffer, scratch_u32_size);
  let scratch_hc = slice_from_raw_parts_or_nil_mut(scratch_hc_buffer, scratch_hc_size);
  ::brotli_decode_prealloc(input, output, scratch_u8, scratch_u32, scratch_hc)
}


#[no_mangle]
pub unsafe extern fn BrotliDecoderDecompressWithReturnInfo(
  encoded_size: usize,
  encoded_buffer: *const u8,
  decoded_size: usize,
  decoded_buffer: *mut u8,
) -> BrotliDecoderReturnInfo {
  let input = slice_from_raw_parts_or_nil(encoded_buffer, encoded_size);
  let output_scratch = slice_from_raw_parts_or_nil_mut(decoded_buffer, decoded_size);
  ::brotli_decode(input, output_scratch)
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderDecompress(
  encoded_size: usize,
  encoded_buffer: *const u8,
  decoded_size: *mut usize,
  decoded_buffer: *mut u8,
) -> BrotliDecoderResult {
  let res = BrotliDecoderDecompressWithReturnInfo(encoded_size, encoded_buffer, *decoded_size, decoded_buffer);
  *decoded_size = res.decoded_size;  
  match res.result {
      BrotliResult::ResultSuccess => BrotliDecoderResult::BROTLI_DECODER_RESULT_SUCCESS,
      _ => BrotliDecoderResult::BROTLI_DECODER_RESULT_ERROR
  }
}

#[cfg(all(feature="std", not(feature="pass-through-ffi-panics")))]
fn catch_panic<F:FnOnce()->BrotliDecoderResult+panic::UnwindSafe>(f: F) -> thread::Result<BrotliDecoderResult> {
    panic::catch_unwind(f)
}

#[cfg(all(feature="std", not(feature="pass-through-ffi-panics")))]
fn catch_panic_state<F:FnOnce()->*mut BrotliDecoderState+panic::UnwindSafe>(f: F) -> thread::Result<*mut BrotliDecoderState> {
    panic::catch_unwind(f)
}

#[cfg(all(feature="std", not(feature="pass-through-ffi-panics")))]
unsafe fn error_print(state_ptr: *mut BrotliDecoderState, err: &mut BrotliAdditionalErrorData) {
    if let Some(st) = err.downcast_ref::<&str>() {
        if !state_ptr.is_null() {
          let mut str_cpy = [0u8;256];
          let src:&[u8] = st.as_ref();
          let xlen = core::cmp::min(src.len(), str_cpy.len() - 1);
          str_cpy.split_at_mut(xlen).0.clone_from_slice(
                src.split_at(xlen).0);
          str_cpy[xlen] = 0; // null terminate
          (*state_ptr).decompressor.mtf_or_error_string = Err(str_cpy);
        }
        let _ign = writeln!(&mut io::stderr(), "panic: {}", st);
    } else {
        if let Some(st) = err.downcast_ref::<string::String>() {

          if !state_ptr.is_null() {
            let mut str_cpy = [0u8;256];
            let src: &[u8] = st.as_ref();
            let xlen = core::cmp::min(src.len(), str_cpy.len() - 1);
            str_cpy.split_at_mut(xlen).0.clone_from_slice(
                src.split_at(xlen).0);
            str_cpy[xlen] = 0; // null terminate
            (*state_ptr).decompressor.mtf_or_error_string = Err(str_cpy);
          }
          let _ign = writeln!(&mut io::stderr(), "Internal Error {:?}", st);
        } else {
            let _ign = writeln!(&mut io::stderr(), "Internal Error {:?}", err);
        }
    }
}

// can't catch panics in a reliable way without std:: configure with panic=abort. These shouldn't happen
#[cfg(any(not(feature="std"), feature="pass-through-ffi-panics"))]
fn catch_panic<F:FnOnce()->BrotliDecoderResult>(f: F) -> Result<BrotliDecoderResult, BrotliAdditionalErrorData> {
    Ok(f())
}

#[cfg(any(not(feature="std"), feature="pass-through-ffi-panics"))]
fn catch_panic_state<F:FnOnce()->*mut BrotliDecoderState>(f: F) -> Result<*mut BrotliDecoderState, BrotliAdditionalErrorData> {
    Ok(f())
}

#[cfg(any(not(feature="std"), feature="pass-through-ffi-panics"))]
fn error_print(_state_ptr: *mut BrotliDecoderState, _err: &mut BrotliAdditionalErrorData) {
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderDecompressStream(
    state_ptr: *mut BrotliDecoderState,
    available_in: *mut usize,
    input_buf_ptr: *mut*const u8,
    available_out: *mut usize,
    output_buf_ptr: *mut*mut u8,
    mut total_out: *mut usize) -> BrotliDecoderResult {
    match catch_panic(move || {
    let mut input_offset = 0usize;
    let mut output_offset = 0usize;
    let mut fallback_total_out = 0usize;
    if total_out.is_null() {
        total_out = &mut fallback_total_out;
    }
    let result: BrotliDecoderResult;
    {
        let input_buf = slice_from_raw_parts_or_nil(*input_buf_ptr, *available_in);
        let output_buf = slice_from_raw_parts_or_nil_mut(*output_buf_ptr, *available_out);
            result = super::decode::BrotliDecompressStream(
                &mut *available_in,
                &mut input_offset,
                input_buf,
                &mut *available_out,
                &mut output_offset,
                output_buf,
                &mut *total_out,
                &mut (*state_ptr).decompressor,
            ).into();
    }
    *input_buf_ptr = (*input_buf_ptr).offset(input_offset as isize);
    *output_buf_ptr = (*output_buf_ptr).offset(output_offset as isize);
                                           result
    }) {
        Ok(ret) => ret,
        Err(mut readable_err) => { // if we panic (completely unexpected) then we should report it back to C and print
            error_print(state_ptr, &mut readable_err);
            (*state_ptr).decompressor.error_code = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE;
            BrotliDecoderResult::BROTLI_DECODER_RESULT_ERROR
        }
    }
}

/// Equivalent to BrotliDecoderDecompressStream but with no optional arg and no double indirect ptrs
#[no_mangle]
pub unsafe extern fn BrotliDecoderDecompressStreaming(
    state_ptr: *mut BrotliDecoderState,
    available_in: *mut usize,
    mut input_buf_ptr: *const u8,
    available_out: *mut usize,
    mut output_buf_ptr: *mut u8) -> BrotliDecoderResult {
    BrotliDecoderDecompressStream(state_ptr,
                                  available_in,
                                  &mut input_buf_ptr,
                                  available_out,
                                  &mut output_buf_ptr,
                                  core::ptr::null_mut())
}

#[cfg(feature="std")]
unsafe fn free_decompressor_no_custom_alloc(state_ptr: *mut BrotliDecoderState) {
    let _state = alloc_util::Box::from_raw(state_ptr);
}

#[cfg(not(feature="std"))]
unsafe fn free_decompressor_no_custom_alloc(_state_ptr: *mut BrotliDecoderState) {
    unreachable!();
}


#[no_mangle]
pub unsafe extern fn BrotliDecoderMallocU8(state_ptr: *mut BrotliDecoderState, size: usize) -> *mut u8 {
    if let Some(alloc_fn) = (*state_ptr).custom_allocator.alloc_func {
        return core::mem::transmute::<*mut c_void, *mut u8>(alloc_fn((*state_ptr).custom_allocator.opaque, size));
    } else {
        return alloc_util::alloc_stdlib(size);
    }
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderFreeU8(state_ptr: *mut BrotliDecoderState, data: *mut u8, size: usize) {
    if let Some(free_fn) = (*state_ptr).custom_allocator.free_func {
        free_fn((*state_ptr).custom_allocator.opaque, core::mem::transmute::<*mut u8, *mut c_void>(data));
    } else {
        alloc_util::free_stdlib(data, size);
    }
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderMallocUsize(state_ptr: *mut BrotliDecoderState, size: usize) -> *mut usize {
    if let Some(alloc_fn) = (*state_ptr).custom_allocator.alloc_func {
        return core::mem::transmute::<*mut c_void, *mut usize>(alloc_fn((*state_ptr).custom_allocator.opaque,
                                                                         size * core::mem::size_of::<usize>()));
    } else {
        return alloc_util::alloc_stdlib(size);
    }
}
#[no_mangle]
pub unsafe extern fn BrotliDecoderFreeUsize(state_ptr: *mut BrotliDecoderState, data: *mut usize, size: usize) {
    if let Some(free_fn) = (*state_ptr).custom_allocator.free_func {
        free_fn((*state_ptr).custom_allocator.opaque, core::mem::transmute::<*mut usize, *mut c_void>(data));
    } else {
        alloc_util::free_stdlib(data, size);
    }
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderDestroyInstance(state_ptr: *mut BrotliDecoderState) {
    if let Some(_) = (*state_ptr).custom_allocator.alloc_func {
        if let Some(free_fn) = (*state_ptr).custom_allocator.free_func {
            let _to_free = core::ptr::read(state_ptr);
            let ptr = core::mem::transmute::<*mut BrotliDecoderState, *mut c_void>(state_ptr);
            free_fn((*state_ptr).custom_allocator.opaque, ptr);
        }
    } else {
        free_decompressor_no_custom_alloc(state_ptr);
    }
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderHasMoreOutput(state_ptr: *const BrotliDecoderState) -> i32 {
  if super::decode::BrotliDecoderHasMoreOutput(&(*state_ptr).decompressor) {1} else {0}
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderTakeOutput(state_ptr: *mut BrotliDecoderState, size: *mut usize) -> *const u8 {
  super::decode::BrotliDecoderTakeOutput(&mut (*state_ptr).decompressor, &mut *size).as_ptr()
}



#[no_mangle]
pub unsafe extern fn BrotliDecoderIsUsed(state_ptr: *const BrotliDecoderState) -> i32 {
  if super::decode::BrotliDecoderIsUsed(&(*state_ptr).decompressor) {1} else {0}
}
#[no_mangle]
pub unsafe extern fn BrotliDecoderIsFinished(state_ptr: *const BrotliDecoderState) -> i32 {
  if super::decode::BrotliDecoderIsFinished(&(*state_ptr).decompressor) {1} else {0}
}
#[no_mangle]
pub unsafe extern fn BrotliDecoderGetErrorCode(state_ptr: *const BrotliDecoderState) -> BrotliDecoderErrorCode {
  super::decode::BrotliDecoderGetErrorCode(&(*state_ptr).decompressor)
}

#[no_mangle]
pub unsafe extern fn BrotliDecoderGetErrorString(state_ptr: *const BrotliDecoderState) -> *const u8 {
  if !state_ptr.is_null() {
    if let &Err(ref msg) = &(*state_ptr).decompressor.mtf_or_error_string {
      // important: this must be a ref
      // so stack memory is not returned
      return msg.as_ptr();
    }
  }
  BrotliDecoderErrorString(super::decode::BrotliDecoderGetErrorCode(&(*state_ptr).decompressor))
}
#[no_mangle]
pub extern fn BrotliDecoderErrorString(c: BrotliDecoderErrorCode) -> *const u8 {
    ::state::BrotliDecoderErrorStr(c).as_ptr()
}


#[no_mangle]
pub extern fn BrotliDecoderVersion() -> u32 {
  0x1000f00
}
