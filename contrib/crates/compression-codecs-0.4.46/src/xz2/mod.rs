mod decoder;
mod encoder;

#[derive(Debug)]
pub enum Xz2FileFormat {
    Xz,
    Lzma,
}

pub use self::{decoder::Xz2Decoder, encoder::Xz2Encoder};

use compression_core::util::{PartialBuffer, WriteBuffer};
use liblzma::stream::{Action, Status, Stream};
use std::io;

/// Return `Ok(true)` if stream ends.
fn process_stream(
    stream: &mut Stream,
    input: &mut PartialBuffer<&[u8]>,
    output: &mut WriteBuffer<'_>,
    action: Action,
) -> io::Result<bool> {
    let previous_in = stream.total_in() as usize;
    let previous_out = stream.total_out() as usize;

    // Safety: We **trust** liblzma to not write uninitialized bytes into the buffer
    let status =
        stream.process_uninit(input.unwritten(), unsafe { output.unwritten_mut() }, action)?;

    input.advance(stream.total_in() as usize - previous_in);
    // Safety: We **trust** liblzma to write bytes into the buffer properly
    unsafe { output.assume_init_and_advance(stream.total_out() as usize - previous_out) };

    match status {
        Status::Ok => Ok(false),
        Status::StreamEnd => Ok(true),
        Status::GetCheck => Err(io::Error::other("Unexpected lzma integrity check")),
        Status::MemNeeded => Err(io::ErrorKind::OutOfMemory.into()),
    }
}
