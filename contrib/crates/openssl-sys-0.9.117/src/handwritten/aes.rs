use super::super::*;
use libc::size_t;
use std::ffi::{c_int, c_uchar, c_uint};

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
#[repr(C)]
pub struct AES_KEY {
    // There is some business with AES_LONG which is there to ensure the values here are 32 bits
    rd_key: [u32; 4 * (AES_MAXNR as usize + 1)],
    rounds: c_int,
}

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_AES_set_encrypt_key")]
    pub fn AES_set_encrypt_key(userKey: *const c_uchar, bits: c_int, key: *mut AES_KEY) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_AES_set_decrypt_key")]
    pub fn AES_set_decrypt_key(userKey: *const c_uchar, bits: c_int, key: *mut AES_KEY) -> c_int;

    #[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_AES_ige_encrypt")]
    pub fn AES_ige_encrypt(
        in_: *const c_uchar,
        out: *mut c_uchar,
        length: size_t,
        key: *const AES_KEY,
        ivec: *mut c_uchar,
        enc: c_int,
    );

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_AES_wrap_key")]

    pub fn AES_wrap_key(
        key: *mut AES_KEY,
        iv: *const c_uchar,
        out: *mut c_uchar,
        in_: *const c_uchar,
        inlen: c_uint,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_AES_unwrap_key")]

    pub fn AES_unwrap_key(
        key: *mut AES_KEY,
        iv: *const c_uchar,
        out: *mut c_uchar,
        in_: *const c_uchar,
        inlen: c_uint,
    ) -> c_int;
}
