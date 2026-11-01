use super::Buffer;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};

pub(crate) trait AsyncBufWrite {
    /// Attempt to return an internal buffer to write to, flushing data out to the inner reader if
    /// it is full.
    ///
    /// On success, returns `Poll::Ready(Ok(buf))`.
    ///
    /// If the buffer is full and cannot be flushed, the method returns `Poll::Pending` and
    /// arranges for the current task context (`cx`) to receive a notification when the object
    /// becomes readable or is closed.
    fn poll_partial_flush_buf(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<Buffer<'_>>>;
}
