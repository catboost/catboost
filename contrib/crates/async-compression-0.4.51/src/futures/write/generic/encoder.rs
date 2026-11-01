use crate::{futures::write::BufWriter, generic::write::impl_encoder};
use futures_io::{AsyncBufRead, AsyncRead, AsyncWrite, IoSliceMut};
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};

impl_encoder!(poll_close);

impl<W: AsyncRead, E> AsyncRead for Encoder<W, E> {
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
