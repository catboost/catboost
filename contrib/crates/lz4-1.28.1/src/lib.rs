#![doc = include_str!("../README.md")]

extern crate lz4_sys;

pub mod liblz4;

mod decoder;
mod encoder;

pub mod block;

pub use crate::decoder::Decoder;
pub use crate::encoder::Encoder;
pub use crate::encoder::EncoderBuilder;
pub use crate::liblz4::version;
pub use crate::liblz4::BlockMode;
pub use crate::liblz4::BlockSize;
pub use crate::liblz4::ContentChecksum;

use lz4_sys::{c_char, size_t};
