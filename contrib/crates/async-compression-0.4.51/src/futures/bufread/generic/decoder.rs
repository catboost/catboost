use crate::{
    codecs::DecodeV2,
    core::util::{PartialBuffer, WriteBuffer},
    generic::bufread::impl_decoder,
};
use core::{
    pin::Pin,
    task::{Context, Poll},
};
use futures_io::{AsyncBufRead, AsyncRead, AsyncWrite};
use std::io::{IoSlice, Result};

impl_decoder!();

impl<R: AsyncBufRead, D: DecodeV2> AsyncRead for Decoder<R, D> {
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

impl<R: AsyncWrite, D> AsyncWrite for Decoder<R, D> {
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
