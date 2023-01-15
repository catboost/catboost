#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn it_works() {
        let model_handle = unsafe { ModelCalcerCreate() };
        let ret_val = unsafe {
            LoadFullModelFromFile(
                model_handle,
                std::ffi::CString::new("../tmp/model.bin").unwrap().as_ptr(),
            )
        };
        if !ret_val {
            let c_str = unsafe { CStr::from_ptr(GetErrorString()) };
            let str_slice = c_str.to_str().unwrap();
            panic!(str_slice);
        }

        let tree_count = unsafe { GetTreeCount(model_handle) };
        assert_eq!(tree_count, 1000);

        let float_features_count = unsafe { GetFloatFeaturesCount(model_handle) };
        assert_eq!(float_features_count, 3);
        let cat_features_count = unsafe { GetCatFeaturesCount(model_handle) };
        assert_eq!(cat_features_count, 1);

        unsafe { ModelCalcerDelete(model_handle) };
    }
}
