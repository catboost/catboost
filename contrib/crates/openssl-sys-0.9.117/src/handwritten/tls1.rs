use super::super::*;
use libc::size_t;
use std::ffi::{c_char, c_int, c_uchar};

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_get_servername")]
    pub fn SSL_get_servername(ssl: *const SSL, name_type: c_int) -> *const c_char;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_export_keying_material")]

    pub fn SSL_export_keying_material(
        s: *mut SSL,
        out: *mut c_uchar,
        olen: size_t,
        label: *const c_char,
        llen: size_t,
        context: *const c_uchar,
        contextlen: size_t,
        use_context: c_int,
    ) -> c_int;

    #[cfg(ossl111)]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_export_keying_material_early")]
    pub fn SSL_export_keying_material_early(
        s: *mut SSL,
        out: *mut c_uchar,
        olen: size_t,
        label: *const c_char,
        llen: size_t,
        context: *const c_uchar,
        contextlen: size_t,
    ) -> c_int;
}
