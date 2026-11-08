//! Types which operate over [`AsyncWrite`](futures_io::AsyncWrite) streams, both encoders and
//! decoders for various formats.

#[macro_use]
mod macros;
mod generic;

mod buf_writer;

use self::{
    buf_writer::BufWriter,
    generic::{Decoder, Encoder},
};
use crate::generic::write::AsyncBufWrite;

algos!(futures::write<W>);
