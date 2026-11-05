use crate::{
    codecs::EncodeV2,
    core::util::{PartialBuffer, WriteBuffer},
    generic::bufread::impl_encoder,
};
use futures_io::{AsyncBufRead, AsyncRead, AsyncWrite, IoSlice};
use std::{
    io::Result,
    pin::Pin,
    task::{Context, Poll},
};

impl_encoder!();

impl<R: AsyncBufRead, E: EncodeV2> AsyncRead for Encoder<R, E> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }

        let mut output = WriteBuffer::new_initialized(buf);
        self.do_poll_read(cx, &mut output)
            .map_ok(|()| output.written_len())
    }
}

impl<R: AsyncWrite, E> AsyncWrite for Encoder<R, E> {
    fn poll_write(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<Result<usize>> {
        self.get_pin_mut().poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.get_pin_mut().poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.get_pin_mut().poll_close(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<Result<usize>> {
        self.get_pin_mut().poll_write_vectored(cx, bufs)
    }
}
