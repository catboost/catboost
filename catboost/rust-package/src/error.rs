use catboost_sys;
use std::ffi::CStr;
use std::fmt;

pub type CatBoostResult<T> = std::result::Result<T, CatBoostError>;

#[derive(Debug, Eq, PartialEq)]
pub struct CatBoostError {
    pub description: String,
}

impl CatBoostError {
    /// Check the return value from an CatBoost FFI call, and return the last error message on error.
    /// Return values of true are treated as success, returns values of false are treated as errors.
    pub fn check_return_value(ret_val: bool) -> CatBoostResult<()> {
        if ret_val {
            Ok(())
        } else {
            Err(CatBoostError::fetch_catboost_error())
        }
    }

    /// Fetch current error message from CatBoost.
    fn fetch_catboost_error() -> Self {
        let c_str = unsafe { CStr::from_ptr(catboost_sys::GetErrorString()) };
        let str_slice = c_str.to_str().unwrap();
        CatBoostError {
            description: str_slice.to_owned(),
        }
    }
}

impl fmt::Display for CatBoostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl std::error::Error for CatBoostError {}
