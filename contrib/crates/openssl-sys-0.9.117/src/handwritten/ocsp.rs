use super::super::*;
use std::ffi::{c_int, c_long, c_uchar, c_ulong};

pub enum OCSP_CERTID {}

pub enum OCSP_ONEREQ {}

pub enum OCSP_REQUEST {}

pub enum OCSP_BASICRESP {}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_cert_to_id")]
    pub fn OCSP_cert_to_id(
        dgst: *const EVP_MD,
        subject: *const X509,
        issuer: *const X509,
    ) -> *mut OCSP_CERTID;
}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_request_add0_id")]
    pub fn OCSP_request_add0_id(r: *mut OCSP_REQUEST, id: *mut OCSP_CERTID) -> *mut OCSP_ONEREQ;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_resp_find_status")]

    pub fn OCSP_resp_find_status(
        bs: *mut OCSP_BASICRESP,
        id: *mut OCSP_CERTID,
        status: *mut c_int,
        reason: *mut c_int,
        revtime: *mut *mut ASN1_GENERALIZEDTIME,
        thisupd: *mut *mut ASN1_GENERALIZEDTIME,
        nextupd: *mut *mut ASN1_GENERALIZEDTIME,
    ) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_check_validity")]
    pub fn OCSP_check_validity(
        thisupd: *mut ASN1_GENERALIZEDTIME,
        nextupd: *mut ASN1_GENERALIZEDTIME,
        sec: c_long,
        maxsec: c_long,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_response_status")]

    pub fn OCSP_response_status(resp: *mut OCSP_RESPONSE) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_response_get1_basic")]
    pub fn OCSP_response_get1_basic(resp: *mut OCSP_RESPONSE) -> *mut OCSP_BASICRESP;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_response_create")]

    pub fn OCSP_response_create(status: c_int, bs: *mut OCSP_BASICRESP) -> *mut OCSP_RESPONSE;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_BASICRESP_new")]

    pub fn OCSP_BASICRESP_new() -> *mut OCSP_BASICRESP;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_BASICRESP_free")]
    pub fn OCSP_BASICRESP_free(r: *mut OCSP_BASICRESP);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_RESPONSE_new")]
    pub fn OCSP_RESPONSE_new() -> *mut OCSP_RESPONSE;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_RESPONSE_free")]
    pub fn OCSP_RESPONSE_free(r: *mut OCSP_RESPONSE);
}

const_ptr_api! {
    extern "C" {
        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_OCSP_RESPONSE")]
        pub fn i2d_OCSP_RESPONSE(a: #[const_ptr_if(ossl300)] OCSP_RESPONSE, pp: *mut *mut c_uchar) -> c_int;
    }
}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_OCSP_RESPONSE")]
    pub fn d2i_OCSP_RESPONSE(
        a: *mut *mut OCSP_RESPONSE,
        pp: *mut *const c_uchar,
        length: c_long,
    ) -> *mut OCSP_RESPONSE;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_ONEREQ_free")]
    pub fn OCSP_ONEREQ_free(r: *mut OCSP_ONEREQ);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_CERTID_free")]
    pub fn OCSP_CERTID_free(id: *mut OCSP_CERTID);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_REQUEST_new")]
    pub fn OCSP_REQUEST_new() -> *mut OCSP_REQUEST;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_REQUEST_free")]
    pub fn OCSP_REQUEST_free(r: *mut OCSP_REQUEST);
}

const_ptr_api! {
    extern "C" {
        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_OCSP_REQUEST")]
        pub fn i2d_OCSP_REQUEST(a: #[const_ptr_if(ossl300)] OCSP_REQUEST, pp: *mut *mut c_uchar) -> c_int;
    }
}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_OCSP_REQUEST")]
    pub fn d2i_OCSP_REQUEST(
        a: *mut *mut OCSP_REQUEST,
        pp: *mut *const c_uchar,
        length: c_long,
    ) -> *mut OCSP_REQUEST;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_OCSP_basic_verify")]

    pub fn OCSP_basic_verify(
        bs: *mut OCSP_BASICRESP,
        certs: *mut stack_st_X509,
        st: *mut X509_STORE,
        flags: c_ulong,
    ) -> c_int;
}
