use std::ffi::CStr;
use std::fmt::Display;
use std::fmt::Formatter;
use std::io::Error;
use std::io::ErrorKind;
use std::str;

pub use lz4_sys::*;

#[derive(Debug)]
pub struct LZ4Error(String);

impl Display for LZ4Error {
    fn fmt(&self, f: &mut Formatter) -> Result<(), ::std::fmt::Error> {
        write!(f, "LZ4 error: {}", &self.0)
    }
}

impl ::std::error::Error for LZ4Error {
    fn description(&self) -> &str {
        &self.0
    }

    fn cause(&self) -> Option<&dyn (::std::error::Error)> {
        None
    }
}

pub fn check_error(code: LZ4FErrorCode) -> Result<usize, Error> {
    unsafe {
        if LZ4F_isError(code) != 0 {
            let error_name = LZ4F_getErrorName(code);
            return Err(Error::new(
                ErrorKind::Other,
                LZ4Error(
                    str::from_utf8(CStr::from_ptr(error_name).to_bytes())
                        .unwrap()
                        .to_string(),
                ),
            ));
        }
    }
    Ok(code as usize)
}

pub fn version() -> i32 {
    unsafe { LZ4_versionNumber() }
}

#[test]
fn test_version_number() {
    version();
}
