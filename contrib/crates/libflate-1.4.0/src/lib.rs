//! A Rust implementation of DEFLATE algorithm and related formats (ZLIB, GZIP).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(no_std, feature = "no_std")]

pub use finish::Finish;

#[cfg(feature = "no_std")]
extern crate alloc;

#[cfg(not(feature = "no_std"))]
macro_rules! invalid_data_error {
    ($fmt:expr) => { invalid_data_error!("{}", $fmt) };
    ($fmt:expr, $($arg:tt)*) => {
        ::std::io::Error::new(::std::io::ErrorKind::InvalidData, format!($fmt, $($arg)*))
    }
}

#[cfg(feature = "no_std")]
macro_rules! invalid_data_error {
    ($fmt:expr) => {
        invalid_data_error!($fmt, "")
    };
    ($fmt:expr, $($arg:tt)*) => {
        ::core2::io::Error::new(::core2::io::ErrorKind::InvalidData, $fmt)
    };
}

macro_rules! finish_try {
    ($e:expr) => {
        match $e.unwrap() {
            (inner, None) => inner,
            (inner, error) => return crate::finish::Finish::new(inner, error),
        }
    };
}

pub mod deflate;
pub mod finish;
pub mod gzip;
pub mod lz77;
pub mod non_blocking;
pub mod zlib;

mod bit;
mod checksum;
mod huffman;
mod util;
