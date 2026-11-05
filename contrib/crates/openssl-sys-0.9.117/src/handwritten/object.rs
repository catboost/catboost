use libc::size_t;
use std::ffi::{c_char, c_int, c_uchar};

use super::super::*;

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_nid2ln")]
    pub fn OBJ_nid2ln(nid: c_int) -> *const c_char;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_nid2sn")]
    pub fn OBJ_nid2sn(nid: c_int) -> *const c_char;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_nid2obj")]
    pub fn OBJ_nid2obj(n: c_int) -> *mut ASN1_OBJECT;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_obj2nid")]
    pub fn OBJ_obj2nid(o: *const ASN1_OBJECT) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_obj2txt")]
    pub fn OBJ_obj2txt(
        buf: *mut c_char,
        buf_len: c_int,
        a: *const ASN1_OBJECT,
        no_name: c_int,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_find_sigid_algs")]

    pub fn OBJ_find_sigid_algs(signid: c_int, pdig_nid: *mut c_int, ppkey_nid: *mut c_int)
        -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_sn2nid")]
    pub fn OBJ_sn2nid(sn: *const c_char) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_txt2obj")]
    pub fn OBJ_txt2obj(s: *const c_char, no_name: c_int) -> *mut ASN1_OBJECT;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_create")]
    pub fn OBJ_create(oid: *const c_char, sn: *const c_char, ln: *const c_char) -> c_int;
    #[cfg(ossl111)]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_length")]
    pub fn OBJ_length(obj: *const ASN1_OBJECT) -> size_t;
    #[cfg(ossl111)]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_get0_data")]
    pub fn OBJ_get0_data(obj: *const ASN1_OBJECT) -> *const c_uchar;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OBJ_cmp")]
    pub fn OBJ_cmp(a: *const ASN1_OBJECT, b: *const ASN1_OBJECT) -> c_int;
}
