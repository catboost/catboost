use std::ffi::{c_double, c_int, c_void};

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_RAND_bytes")]
    pub fn RAND_bytes(buf: *mut u8, num: c_int) -> c_int;

    #[cfg(ossl111)]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_RAND_priv_bytes")]
    pub fn RAND_priv_bytes(buf: *mut u8, num: c_int) -> c_int;

    #[cfg(ossl111)]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_RAND_keep_random_devices_open")]
    pub fn RAND_keep_random_devices_open(keep: c_int);

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_RAND_status")]

    pub fn RAND_status() -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_RAND_add")]

    pub fn RAND_add(buf: *const c_void, num: c_int, randomness: c_double);
}
