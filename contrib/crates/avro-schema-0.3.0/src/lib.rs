#![doc = include_str!("lib.md")]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]

pub mod error;
pub mod file;
pub mod schema;

pub mod read;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod read_async;

pub mod write;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod write_async;
