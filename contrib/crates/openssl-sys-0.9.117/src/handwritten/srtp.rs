use super::super::*;
use std::ffi::{c_char, c_int};

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_CTX_set_tlsext_use_srtp")]
    pub fn SSL_CTX_set_tlsext_use_srtp(ctx: *mut SSL_CTX, profiles: *const c_char) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_set_tlsext_use_srtp")]
    pub fn SSL_set_tlsext_use_srtp(ssl: *mut SSL, profiles: *const c_char) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_get_srtp_profiles")]

    pub fn SSL_get_srtp_profiles(ssl: *mut SSL) -> *mut stack_st_SRTP_PROTECTION_PROFILE;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_SSL_get_selected_srtp_profile")]
    pub fn SSL_get_selected_srtp_profile(ssl: *mut SSL) -> *mut SRTP_PROTECTION_PROFILE;
}
