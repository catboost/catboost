use core;

pub use brotli_decompressor::ffi::interface::c_void;
use brotli_decompressor::ffi::{slice_from_raw_parts_or_nil, slice_from_raw_parts_or_nil_mut};

use crate::concat::BroCatli;
pub use crate::concat::BroCatliResult;
pub type BroccoliResult = BroCatliResult;
// a tool to concatenate brotli files together

#[repr(C)]
pub struct BroccoliState {
    more_data: *mut c_void,
    current_data: [u8; 120],
}

impl Clone for BroccoliState {
    fn clone(&self) -> BroccoliState {
        let mut cd = [0u8; 120];
        cd.clone_from_slice(&self.current_data[..]);
        BroccoliState {
            more_data: self.more_data,
            current_data: cd,
        }
    }
}

impl Copy for BroccoliState {}

impl Default for BroccoliState {
    fn default() -> BroccoliState {
        BroCatli::new().into()
    }
}
impl From<BroCatli> for BroccoliState {
    fn from(data: BroCatli) -> BroccoliState {
        let mut buffer = [0u8; 120];
        data.serialize_to_buffer(&mut buffer[..]).unwrap();
        BroccoliState {
            more_data: core::ptr::null_mut(),
            current_data: buffer,
        }
    }
}
impl From<BroccoliState> for BroCatli {
    fn from(val: BroccoliState) -> Self {
        BroCatli::deserialize_from_buffer(&val.current_data[..]).unwrap()
    }
}

#[no_mangle]
pub extern "C" fn BroccoliCreateInstance() -> BroccoliState {
    BroCatli::new().into()
}
#[no_mangle]
pub extern "C" fn BroccoliCreateInstanceWithWindowSize(window_size: u8) -> BroccoliState {
    BroCatli::new_with_window_size(window_size).into()
}
#[no_mangle]
pub extern "C" fn BroccoliDestroyInstance(_state: BroccoliState) {}

#[no_mangle]
pub unsafe extern "C" fn BroccoliNewBrotliFile(state: *mut BroccoliState) {
    let mut bro_catli: BroCatli = (*state).into();
    bro_catli.new_brotli_file();
    *state = BroccoliState::from(bro_catli);
}

#[no_mangle]
pub unsafe extern "C" fn BroccoliConcatStream(
    state: *mut BroccoliState,
    available_in: *mut usize,
    input_buf_ptr: *mut *const u8,
    available_out: *mut usize,
    output_buf_ptr: *mut *mut u8,
) -> BroccoliResult {
    let input_buf = slice_from_raw_parts_or_nil(*input_buf_ptr, *available_in);
    let output_buf = slice_from_raw_parts_or_nil_mut(*output_buf_ptr, *available_out);
    let mut input_offset = 0usize;
    let mut output_offset = 0usize;
    let mut bro_catli: BroCatli = (*state).into();
    let ret = bro_catli.stream(input_buf, &mut input_offset, output_buf, &mut output_offset);
    *input_buf_ptr = (*input_buf_ptr).add(input_offset);
    *output_buf_ptr = (*output_buf_ptr).add(output_offset);
    *available_in -= input_offset;
    *available_out -= output_offset;
    *state = BroccoliState::from(bro_catli);
    ret
}

#[no_mangle]
pub unsafe extern "C" fn BroccoliConcatStreaming(
    state: *mut BroccoliState,
    available_in: *mut usize,
    mut input_buf: *const u8,
    available_out: *mut usize,
    mut output_buf: *mut u8,
) -> BroccoliResult {
    BroccoliConcatStream(
        state,
        available_in,
        &mut input_buf,
        available_out,
        &mut output_buf,
    )
}

#[no_mangle]
pub unsafe extern "C" fn BroccoliConcatFinish(
    state: *mut BroccoliState,
    available_out: *mut usize,
    output_buf_ptr: *mut *mut u8,
) -> BroCatliResult {
    let output_buf = slice_from_raw_parts_or_nil_mut(*output_buf_ptr, *available_out);
    let mut output_offset = 0usize;
    let mut bro_catli: BroCatli = (*state).into();
    let ret = bro_catli.finish(output_buf, &mut output_offset);
    *output_buf_ptr = (*output_buf_ptr).add(output_offset);
    *available_out -= output_offset;
    *state = BroccoliState::from(bro_catli);
    ret
}

// exactly the same as BrotliConcatFinish but without the indirect
#[no_mangle]
pub unsafe extern "C" fn BroccoliConcatFinished(
    state: *mut BroccoliState,
    available_out: *mut usize,
    mut output_buf: *mut u8,
) -> BroCatliResult {
    BroccoliConcatFinish(state, available_out, &mut output_buf)
}
