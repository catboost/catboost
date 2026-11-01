use super::super::*;

const_ptr_api! {
    extern "C" {
        #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_NCONF_new")]
        pub fn NCONF_new(meth: #[const_ptr_if(libressl400)] CONF_METHOD) -> *mut CONF;
    }
}

extern "C" {
    #[cfg(not(libressl400))]
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_NCONF_default")]
    pub fn NCONF_default() -> *mut CONF_METHOD;
    #[cfg_attr(arcadia_openssl_renamed, link_name = "v1_NCONF_free")]
    pub fn NCONF_free(conf: *mut CONF);
}
