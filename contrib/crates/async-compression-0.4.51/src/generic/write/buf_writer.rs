// Originally sourced from `futures_util::io::buf_writer`, needs to be redefined locally so that
// the `AsyncBufWrite` impl can access its internals, and changed a bit to make it more efficient
// with those methods.

use super::AsyncBufWrite;
use compression_core::util::WriteBuffer;
use std::{
    fmt, io,
    pin::Pin,
    task::{ready, Context, Poll},
};

const DEFAULT_BUF_SIZE: usize = 8192;

pub struct BufWriter {
    buf: Box<[u8]>,
    written: usize,
    buffered: usize,
}

impl fmt::Debug for BufWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenericBufWriter")
            .field(
                "buffer",
                &format_args!("{}/{}", self.buffered, self.buf.len()),
            )
            .field("written", &self.written)
            .finish()
    }
}

impl BufWriter {
    /// Creates a new `BufWriter` with a default buffer capacity. The default is currently 8 KB,
    /// but may change in the future.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUF_SIZE)
    }

    /// Creates a new `BufWriter` with the specified buffer capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: vec![0; cap].into(),
            written: 0,
            buffered: 0,
        }
    }

    /// Remove the already written data
    fn remove_written(&mut self) {
        self.buf.copy_within(self.written..self.buffered, 0);
        self.buffered -= self.written;
        self.written = 0;
    }

    fn do_flush(
        &mut self,
        poll_write: &mut dyn FnMut(&[u8]) -> Poll<io::Result<usize>>,
    ) -> Poll<io::Result<()>> {
        while self.written < self.buffered {
            let bytes_written = ready!(poll_write(&self.buf[self.written..self.buffered]))?;
            if bytes_written == 0 {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write the buffered data",
                )));
            }

            self.written += bytes_written;
        }

        Poll::Ready(Ok(()))
    }

    fn partial_flush_buf(
        &mut self,
        poll_write: &mut dyn FnMut(&[u8]) -> Poll<io::Result<usize>>,
    ) -> Poll<io::Result<()>> {
        let ret = if let Poll::Ready(res) = self.do_flush(poll_write) {
            res
        } else {
            Ok(())
        };

        if self.written > 0 || self.buffered < self.buf.len() {
            Poll::Ready(ret)
        } else {
            ret?;
            Poll::Pending
        }
    }

    pub fn flush_buf(
        &mut self,
        poll_write: &mut dyn FnMut(&[u8]) -> Poll<io::Result<usize>>,
    ) -> Poll<io::Result<()>> {
        ready!(self.do_flush(poll_write))?;

        debug_assert_eq!(self.buffered, self.written);
        self.buffered = 0;
        self.written = 0;

        Poll::Ready(Ok(()))
    }

    pub fn poll_write(
        &mut self,
        buf: &[u8],
        poll_write: &mut dyn FnMut(&[u8]) -> Poll<io::Result<usize>>,
    ) -> Poll<io::Result<usize>> {
        if buf.len() >= self.buf.len() {
            ready!(self.flush_buf(poll_write))?;
            poll_write(buf)
        } else if (self.buf.len() - self.buffered) >= buf.len() {
            self.buf[self.buffered..].copy_from_slice(buf);
            self.buffered += buf.len();

            Poll::Ready(Ok(buf.len()))
        } else {
            ready!(self.partial_flush_buf(poll_write))?;
            if self.written > 0 {
                self.remove_written();
            }

            let len = buf.len().min(self.buf.len() - self.buffered);
            self.buf[self.buffered..self.buffered + len].copy_from_slice(&buf[..len]);
            self.buffered += len;

            Poll::Ready(Ok(len))
        }
    }

    pub fn poll_partial_flush_buf(
        &mut self,
        poll_write: &mut dyn FnMut(&[u8]) -> Poll<io::Result<usize>>,
    ) -> Poll<io::Result<Buffer<'_>>> {
        ready!(self.partial_flush_buf(poll_write))?;

        // when the flushed data is larger than or equal to half of yet-to-be-flushed data,
        // the copyback could use version of memcpy that do copies from the head of the buffer.
        // Anything smaller than that, an overlap would happen that forces use of memmove.
        if self.written >= (self.buffered / 3)
            || self.written >= 512
            || self.buffered == self.buf.len()
        {
            self.remove_written();
        }

        Poll::Ready(Ok(Buffer {
            write_buffer: WriteBuffer::new_initialized(&mut self.buf[self.buffered..]),
            buffered: &mut self.buffered,
        }))
    }
}

pub struct Buffer<'a> {
    buffered: &'a mut usize,
    pub write_buffer: WriteBuffer<'a>,
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        *self.buffered += self.write_buffer.written_len();
    }
}

macro_rules! impl_buf_writer {
    ($poll_close: tt) => {
        use crate::generic::write::{AsyncBufWrite, BufWriter as GenericBufWriter, Buffer};
        use pin_project_lite::pin_project;
        use std::task::ready;

        pin_project! {
            #[derive(Debug)]
            pub struct BufWriter<W> {
                #[pin]
                writer: W,
                inner: GenericBufWriter,
            }
        }

        impl<W> BufWriter<W> {
            /// Creates a new `BufWriter` with a default buffer capacity. The default is currently 8 KB,
            /// but may change in the future.
            pub fn new(writer: W) -> Self {
                Self {
                    writer,
                    inner: GenericBufWriter::new(),
                }
            }

            /// Creates a new `BufWriter` with the specified buffer capacity.
            pub fn with_capacity(cap: usize, writer: W) -> Self {
                Self {
                    writer,
                    inner: GenericBufWriter::with_capacity(cap),
                }
            }

            /// Gets a reference to the underlying writer.
            pub fn get_ref(&self) -> &W {
                &self.writer
            }

            /// Gets a mutable reference to the underlying writer.
            ///
            /// It is inadvisable to directly write to the underlying writer.
            pub fn get_mut(&mut self) -> &mut W {
                &mut self.writer
            }

            /// Gets a pinned mutable reference to the underlying writer.
            ///
            /// It is inadvisable to directly write to the underlying writer.
            pub fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut W> {
                self.project().writer
            }

            /// Consumes this `BufWriter`, returning the underlying writer.
            ///
            /// Note that any leftover data in the internal buffer is lost.
            pub fn into_inner(self) -> W {
                self.writer
            }
        }

        fn get_poll_write<'a, 'b, W: AsyncWrite>(
            mut writer: Pin<&'a mut W>,
            cx: &'a mut Context<'b>,
        ) -> impl for<'buf> FnMut(&'buf [u8]) -> Poll<io::Result<usize>> + use<'a, 'b, W> {
            move |buf| writer.as_mut().poll_write(cx, buf)
        }

        impl<W: AsyncWrite> BufWriter<W> {
            fn flush_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                let this = self.project();
                this.inner.flush_buf(&mut get_poll_write(this.writer, cx))
            }
        }

        impl<W: AsyncWrite> AsyncWrite for BufWriter<W> {
            fn poll_write(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<io::Result<usize>> {
                let this = self.project();
                this.inner
                    .poll_write(buf, &mut get_poll_write(this.writer, cx))
            }

            fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                ready!(self.as_mut().flush_buf(cx))?;
                self.project().writer.poll_flush(cx)
            }

            fn $poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                ready!(self.as_mut().flush_buf(cx))?;
                self.project().writer.$poll_close(cx)
            }
        }

        impl<W: AsyncWrite> AsyncBufWrite for BufWriter<W> {
            fn poll_partial_flush_buf(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<io::Result<Buffer<'_>>> {
                let this = self.project();
                this.inner
                    .poll_partial_flush_buf(&mut get_poll_write(this.writer, cx))
            }
        }
    };
}
pub(crate) use impl_buf_writer;
