mod decoder;
mod encoder;

pub use self::{decoder::Decoder, encoder::Encoder};

use crate::core::util::WriteBuffer;
use std::{io::Result, task::Poll};
use tokio::io::ReadBuf;

fn poll_read(
    buf: &mut ReadBuf<'_>,
    do_poll_read: impl FnOnce(&mut WriteBuffer<'_>) -> Poll<Result<()>>,
) -> Poll<Result<()>> {
    if buf.remaining() == 0 {
        return Poll::Ready(Ok(()));
    }

    let initialized = buf.initialized().len() - buf.filled().len();
    // Safety: `WriteBuffer` has the same safety invariant as `ReadBuf`
    let mut output = WriteBuffer::new_uninitialized(unsafe { buf.unfilled_mut() });
    // Safety: `ReadBuf` ensures that it is initialized
    unsafe { output.assume_init(initialized) };

    let res = do_poll_read(&mut output);

    let initialized = output.initialized_len();
    let written = output.written_len();

    // Safety: We trust our implementation to have properly initialized it
    unsafe { buf.assume_init(initialized) };
    buf.advance(written);

    res
}
