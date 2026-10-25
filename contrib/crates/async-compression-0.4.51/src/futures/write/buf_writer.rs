// Originally sourced from `futures_util::io::buf_writer`, needs to be redefined locally so that
// the `AsyncBufWrite` impl can access its internals, and changed a bit to make it more efficient
// with those methods.

use crate::generic::write::impl_buf_writer;
use futures_io::{AsyncSeek, AsyncWrite, SeekFrom};
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};

impl_buf_writer!(poll_close);

impl<W: AsyncWrite + AsyncSeek> AsyncSeek for BufWriter<W> {
    /// Seek to the offset, in bytes, in the underlying writer.
    ///
    /// Seeking always writes out the internal buffer before seeking.
    fn poll_seek(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        pos: SeekFrom,
    ) -> Poll<io::Result<u64>> {
        ready!(self.as_mut().flush_buf(cx))?;
        self.project().writer.poll_seek(cx, pos)
    }
}
