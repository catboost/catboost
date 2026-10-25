use libc::size_t;
use std::ffi::{c_int, c_uchar, c_void};

use super::super::*;

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_CTX_new")]
    pub fn CMAC_CTX_new() -> *mut CMAC_CTX;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_CTX_free")]
    pub fn CMAC_CTX_free(ctx: *mut CMAC_CTX);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_Init")]
    pub fn CMAC_Init(
        ctx: *mut CMAC_CTX,
        key: *const c_void,
        len: size_t,
        cipher: *const EVP_CIPHER,
        impl_: *mut ENGINE,
    ) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_Update")]
    pub fn CMAC_Update(ctx: *mut CMAC_CTX, data: *const c_void, len: size_t) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_Final")]
    pub fn CMAC_Final(ctx: *mut CMAC_CTX, out: *mut c_uchar, len: *mut size_t) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_CMAC_CTX_copy")]
    pub fn CMAC_CTX_copy(dst: *mut CMAC_CTX, src: *const CMAC_CTX) -> c_int;
}
