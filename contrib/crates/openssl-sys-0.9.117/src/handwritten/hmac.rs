use libc::size_t;
use std::ffi::{c_int, c_uchar, c_uint, c_void};

use super::super::*;

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_CTX_new")]
    pub fn HMAC_CTX_new() -> *mut HMAC_CTX;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_CTX_free")]
    pub fn HMAC_CTX_free(ctx: *mut HMAC_CTX);
}

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_Init_ex")]
    pub fn HMAC_Init_ex(
        ctx: *mut HMAC_CTX,
        key: *const c_void,
        len: c_int,
        md: *const EVP_MD,
        impl_: *mut ENGINE,
    ) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_Update")]
    pub fn HMAC_Update(ctx: *mut HMAC_CTX, data: *const c_uchar, len: size_t) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_Final")]
    pub fn HMAC_Final(ctx: *mut HMAC_CTX, md: *mut c_uchar, len: *mut c_uint) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_HMAC_CTX_copy")]
    pub fn HMAC_CTX_copy(dst: *mut HMAC_CTX, src: *mut HMAC_CTX) -> c_int;
}
