use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use {parser, select, select_as_str};

const INVALID_PATH: &str = "invalid path";
const INVALID_JSON: &str = "invalud json";

fn to_str(v: *const c_char, err_msg: &str) -> &str {
    unsafe { CStr::from_ptr(v) }.to_str().expect(err_msg)
}

fn to_char_ptr(v: &str) -> *const c_char {
    let s = CString::new(v).unwrap_or_else(|_| panic!("invalid string: {}", v));
    let ptr = s.as_ptr();
    std::mem::forget(s);
    ptr
}

#[no_mangle]
pub extern "C" fn ffi_select(json_str: *const c_char, path: *const c_char) -> *const c_char {
    let json_str = to_str(json_str, INVALID_JSON);
    let path = to_str(path, INVALID_PATH);
    match select_as_str(json_str, path) {
        Ok(v) => to_char_ptr(v.as_str()),
        Err(e) => {
            panic!("{:?}", e);
        }
    }
}

#[no_mangle]
#[allow(clippy::forget_copy)]
pub extern "C" fn ffi_path_compile(path: *const c_char) -> *mut c_void {
    let path = to_str(path, INVALID_PATH);
    #[allow(deprecated)]
    let ref_node = Box::into_raw(Box::new(parser::Parser::compile(path).unwrap()));
    let ptr = ref_node as *mut c_void;
    std::mem::forget(ref_node);
    ptr
}

#[no_mangle]
pub extern "C" fn ffi_select_with_compiled_path(
    path_ptr: *mut c_void,
    json_ptr: *const c_char,
) -> *const c_char {
    #[allow(deprecated)]
    let node = unsafe { Box::from_raw(path_ptr as *mut parser::Node) };
    let json_str = to_str(json_ptr, INVALID_JSON);
    let json = serde_json::from_str(json_str)
        .unwrap_or_else(|_| panic!("invalid json string: {}", json_str));

    #[allow(deprecated)]
    let mut selector = select::Selector::default();
    let found = selector.compiled_path(&node).value(&json).select().unwrap();
    std::mem::forget(node);

    let result = serde_json::to_string(&found)
        .unwrap_or_else(|_| panic!("json serialize error: {:?}", found));
    to_char_ptr(result.as_str())
}
