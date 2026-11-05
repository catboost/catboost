use crate::{
    codecs::EncodeV2,
    core::util::{PartialBuffer, WriteBuffer},
    generic::write::AsyncBufWrite,
};
use std::{
    io,
    pin::Pin,
    task::{ready, Context, Poll},
};

#[derive(Debug)]
enum State {
    Encoding,
    Finishing,
    Done,
}

#[derive(Debug)]
pub struct Encoder {
    state: State,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            state: State::Encoding,
        }
    }
}

impl Encoder {
    fn do_poll_write(
        &mut self,
        cx: &mut Context<'_>,
        input: &mut PartialBuffer<&[u8]>,
        mut writer: Pin<&mut dyn AsyncBufWrite>,
        encoder: &mut dyn EncodeV2,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut output = ready!(writer.as_mut().poll_partial_flush_buf(cx))?;
            let output = &mut output.write_buffer;

            self.state = match self.state {
                State::Encoding => {
                    encoder.encode(input, output)?;
                    State::Encoding
                }

                State::Finishing | State::Done => {
                    break Poll::Ready(Err(io::Error::other("Write after close")));
                }
            };

            if input.unwritten().is_empty() {
                break Poll::Ready(Ok(()));
            }
        }
    }

    pub fn poll_write(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
        writer: Pin<&mut dyn AsyncBufWrite>,
        encoder: &mut dyn EncodeV2,
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }

        let mut input = PartialBuffer::new(buf);

        match self.do_poll_write(cx, &mut input, writer, encoder)? {
            Poll::Pending if input.written().is_empty() => Poll::Pending,
            _ => Poll::Ready(Ok(input.written().len())),
        }
    }

    pub fn do_poll_flush(
        &mut self,
        cx: &mut Context<'_>,
        mut writer: Pin<&mut dyn AsyncBufWrite>,
        encoder: &mut dyn EncodeV2,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut output = ready!(writer.as_mut().poll_partial_flush_buf(cx))?;
            let output = &mut output.write_buffer;

            let done = match self.state {
                State::Encoding => encoder.flush(output)?,

                State::Finishing | State::Done => {
                    break Poll::Ready(Err(io::Error::other("Flush after close")));
                }
            };

            if done {
                break Poll::Ready(Ok(()));
            }
        }
    }

    pub fn do_poll_close(
        &mut self,
        cx: &mut Context<'_>,
        mut writer: Pin<&mut dyn AsyncBufWrite>,
        encoder: &mut dyn EncodeV2,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut output = ready!(writer.as_mut().poll_partial_flush_buf(cx))?;
            let output = &mut output.write_buffer;

            self.state = match self.state {
                State::Encoding | State::Finishing => {
                    if encoder.finish(output)? {
                        State::Done
                    } else {
                        State::Finishing
                    }
                }

                State::Done => State::Done,
            };

            if let State::Done = self.state {
                break Poll::Ready(Ok(()));
            }
        }
    }
}

macro_rules! impl_encoder {
    ($poll_close: tt) => {
        use crate::{codecs::EncodeV2, generic::write::Encoder as GenericEncoder};
        use pin_project_lite::pin_project;
        use std::task::ready;

        pin_project! {
            #[derive(Debug)]
            pub struct Encoder<W, E> {
                #[pin]
                writer: BufWriter<W>,
                encoder: E,
                inner: GenericEncoder,
            }
        }

        impl<W, E> Encoder<W, E> {
            pub fn get_ref(&self) -> &W {
                self.writer.get_ref()
            }

            pub fn get_mut(&mut self) -> &mut W {
                self.writer.get_mut()
            }

            pub fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut W> {
                self.project().writer.get_pin_mut()
            }

            pub(crate) fn get_encoder_ref(&self) -> &E {
                &self.encoder
            }

            pub fn into_inner(self) -> W {
                self.writer.into_inner()
            }

            pub fn new(writer: W, encoder: E) -> Self {
                Self {
                    writer: BufWriter::new(writer),
                    encoder,
                    inner: Default::default(),
                }
            }

            pub fn with_capacity(writer: W, encoder: E, cap: usize) -> Self {
                Self {
                    writer: BufWriter::with_capacity(cap, writer),
                    encoder,
                    inner: Default::default(),
                }
            }
        }

        impl<W: AsyncWrite, E: EncodeV2> AsyncWrite for Encoder<W, E> {
            fn poll_write(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<io::Result<usize>> {
                let this = self.project();
                this.inner.poll_write(cx, buf, this.writer, this.encoder)
            }

            fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                let mut this = self.project();

                ready!(this
                    .inner
                    .do_poll_flush(cx, this.writer.as_mut(), this.encoder))?;
                this.writer.poll_flush(cx)
            }

            fn $poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                let mut this = self.project();

                ready!(this
                    .inner
                    .do_poll_close(cx, this.writer.as_mut(), this.encoder))?;
                this.writer.$poll_close(cx)
            }
        }

        impl<W: AsyncBufRead, E> AsyncBufRead for Encoder<W, E> {
            fn poll_fill_buf(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<io::Result<&[u8]>> {
                self.get_pin_mut().poll_fill_buf(cx)
            }

            fn consume(self: Pin<&mut Self>, amt: usize) {
                self.get_pin_mut().consume(amt)
            }
        }
    };
}
pub(crate) use impl_encoder;
