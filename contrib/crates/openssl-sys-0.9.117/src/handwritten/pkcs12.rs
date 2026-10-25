use std::ffi::{c_char, c_int, c_long, c_uchar};

use super::super::*;

pub enum PKCS12 {}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_PKCS12_free")]
    pub fn PKCS12_free(p12: *mut PKCS12);
}
const_ptr_api! {
    extern "C" {
        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_PKCS12")]
        pub fn i2d_PKCS12(a: #[const_ptr_if(ossl300)] PKCS12, buf: *mut *mut u8) -> c_int;
    }
}
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_PKCS12")]
    pub fn d2i_PKCS12(a: *mut *mut PKCS12, pp: *mut *const u8, length: c_long) -> *mut PKCS12;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_PKCS12_parse")]

    pub fn PKCS12_parse(
        p12: *mut PKCS12,
        pass: *const c_char,
        pkey: *mut *mut EVP_PKEY,
        cert: *mut *mut X509,
        ca: *mut *mut stack_st_X509,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_PKCS12_set_mac")]

    pub fn PKCS12_set_mac(
        p12: *mut PKCS12,
        pass: *const c_char,
        passlen: c_int,
        salt: *mut c_uchar,
        saltlen: c_int,
        iter: c_int,
        md_type: *const EVP_MD,
    ) -> c_int;
}
const_ptr_api! {
    extern "C" {
        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_PKCS12_create")]
        pub fn PKCS12_create(
            pass: *const c_char,
            friendly_name: *const c_char,
            pkey: *mut EVP_PKEY,
            cert: *mut X509,
            ca: *mut stack_st_X509,
            nid_key: c_int,
            nid_cert: c_int,
            iter: c_int,
            mac_iter: c_int,
            keytype: c_int,
        ) -> *mut PKCS12;

        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_PKCS12_bio")]

        pub fn i2d_PKCS12_bio(b: *mut BIO, a: #[const_ptr_if(ossl300)] PKCS12) -> c_int;
    }
}
