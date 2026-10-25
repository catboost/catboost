use ::BrotliResult;
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum c_void{
    _Nothing = 0,
}

#[repr(C)]
#[allow(dead_code)]
pub enum BrotliDecoderParameter {
    BROTLI_DECODER_PARAM_DISABLE_RING_BUFFER_REALLOCATION = 0,
    BROTLI_DECODER_PARAM_LARGE_WINDOW = 1,
}


#[repr(C)]
pub enum BrotliDecoderResult {
    BROTLI_DECODER_RESULT_ERROR = 0,
    BROTLI_DECODER_RESULT_SUCCESS = 1,
    BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT = 2,
    BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT = 3,
}


impl From<BrotliResult> for BrotliDecoderResult {
  fn from(r: BrotliResult) -> Self {
    match r {
      BrotliResult::ResultSuccess => BrotliDecoderResult::BROTLI_DECODER_RESULT_SUCCESS,
      BrotliResult::ResultFailure => BrotliDecoderResult::BROTLI_DECODER_RESULT_ERROR,
      BrotliResult::NeedsMoreInput => BrotliDecoderResult::BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT ,
      BrotliResult::NeedsMoreOutput => BrotliDecoderResult::BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT ,
    }
  }
}
pub type brotli_alloc_func = Option<extern "C" fn(data: *mut c_void, size: usize) -> *mut c_void>;

pub type brotli_free_func = Option<extern "C" fn(data: *mut c_void, ptr: *mut c_void) -> ()>;


#[repr(C)]
#[derive(Clone)]
pub struct CAllocator {
    pub alloc_func: brotli_alloc_func,
    pub free_func: brotli_free_func,
    pub opaque: *mut c_void,
}

unsafe impl Send for CAllocator {
}
