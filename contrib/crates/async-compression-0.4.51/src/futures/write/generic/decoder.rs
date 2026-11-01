use crate::{futures::write::BufWriter, generic::write::impl_decoder};
use futures_io::{AsyncBufRead, AsyncRead, AsyncWrite, IoSliceMut};
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};

impl_decoder!(poll_close);

impl<W: AsyncRead, D> AsyncRead for Decoder<W, D> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        self.get_pin_mut().poll_read(cx, buf)
    }

    fn poll_read_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &mut [IoSliceMut<'_>],
    ) -> Poll<io::Result<usize>> {
        self.get_pin_mut().poll_read_vectored(cx, bufs)
    }
}
