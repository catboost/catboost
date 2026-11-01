use std::ffi::{c_int, c_long, c_uchar, c_uint, c_ulong};

use super::super::*;

#[cfg(ossl300)]
extern "C" {
    pub fn EVP_PKEY_CTX_set_dsa_paramgen_bits(ctx: *mut EVP_PKEY_CTX, nbits: c_int) -> c_int;
}

pub enum DSA_SIG {}

#[cfg(not(osslconf = "OPENSSL_NO_DEPRECATED_3_0"))]
extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_new")]
    pub fn DSA_new() -> *mut DSA;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_free")]
    pub fn DSA_free(dsa: *mut DSA);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_up_ref")]
    pub fn DSA_up_ref(dsa: *mut DSA) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_size")]
    pub fn DSA_size(dsa: *const DSA) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_sign")]
    pub fn DSA_sign(
        dummy: c_int,
        dgst: *const c_uchar,
        len: c_int,
        sigret: *mut c_uchar,
        siglen: *mut c_uint,
        dsa: *mut DSA,
    ) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_verify")]
    pub fn DSA_verify(
        dummy: c_int,
        dgst: *const c_uchar,
        len: c_int,
        sigbuf: *const c_uchar,
        siglen: c_int,
        dsa: *mut DSA,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_DSAPublicKey")]

    pub fn d2i_DSAPublicKey(a: *mut *mut DSA, pp: *mut *const c_uchar, length: c_long) -> *mut DSA;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_DSAPrivateKey")]
    pub fn d2i_DSAPrivateKey(a: *mut *mut DSA, pp: *mut *const c_uchar, length: c_long)
        -> *mut DSA;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_generate_parameters_ex")]

    pub fn DSA_generate_parameters_ex(
        dsa: *mut DSA,
        bits: c_int,
        seed: *const c_uchar,
        seed_len: c_int,
        counter_ref: *mut c_int,
        h_ret: *mut c_ulong,
        cb: *mut BN_GENCB,
    ) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_generate_key")]

    pub fn DSA_generate_key(dsa: *mut DSA) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_DSAPublicKey")]
    pub fn i2d_DSAPublicKey(a: *const DSA, pp: *mut *mut c_uchar) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_DSAPrivateKey")]
    pub fn i2d_DSAPrivateKey(a: *const DSA, pp: *mut *mut c_uchar) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_get0_pqg")]

    pub fn DSA_get0_pqg(
        d: *const DSA,
        p: *mut *const BIGNUM,
        q: *mut *const BIGNUM,
        q: *mut *const BIGNUM,
    );
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_set0_pqg")]
    pub fn DSA_set0_pqg(d: *mut DSA, p: *mut BIGNUM, q: *mut BIGNUM, q: *mut BIGNUM) -> c_int;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_get0_key")]
    pub fn DSA_get0_key(d: *const DSA, pub_key: *mut *const BIGNUM, priv_key: *mut *const BIGNUM);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_set0_key")]
    pub fn DSA_set0_key(d: *mut DSA, pub_key: *mut BIGNUM, priv_key: *mut BIGNUM) -> c_int;
}

extern "C" {
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_d2i_DSA_SIG")]
    pub fn d2i_DSA_SIG(
        sig: *mut *mut DSA_SIG,
        pp: *mut *const c_uchar,
        length: c_long,
    ) -> *mut DSA_SIG;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_i2d_DSA_SIG")]
    pub fn i2d_DSA_SIG(a: *const DSA_SIG, pp: *mut *mut c_uchar) -> c_int;

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_SIG_new")]

    pub fn DSA_SIG_new() -> *mut DSA_SIG;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_SIG_free")]
    pub fn DSA_SIG_free(sig: *mut DSA_SIG);

    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_SIG_get0")]

    pub fn DSA_SIG_get0(sig: *const DSA_SIG, pr: *mut *const BIGNUM, ps: *mut *const BIGNUM);
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_DSA_SIG_set0")]
    pub fn DSA_SIG_set0(sig: *mut DSA_SIG, pr: *mut BIGNUM, ps: *mut BIGNUM) -> c_int;
}
