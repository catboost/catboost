//! Implementations that can handle non-blocking I/O.
//!
//! The implementations in this module can handle non-blocking
//! `Reader`s and `Writer`s which will return `ErrorKind::WouldBlock` error
//! when I/O operations would block.
//!
//! If inner `Reader`s and `Writer`s return `ErrorKind::WouldBlock` error,
//! `Decoder`s and `Encoder`s in this module will also return `ErrorKind::WouldBlock`.
//!
//! If retrying the operation after the inner I/O become available, it will proceed successfully.
//!
//! # NOTICE
//!
//! There is some performance penalty for non-blocking implementations
//! against those that do not consider nonblocking I / O.
//! So, it is recommended to use the latter if you are not need to handle non-blocking I/O.
pub mod deflate;
pub mod gzip;
pub mod zlib;

mod transaction;
