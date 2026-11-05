use crate::{
    codecs::DecodeV2,
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
    Decoding,
    Finishing,
    Done,
}

#[derive(Debug)]
pub struct Decoder {
    state: State,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            state: State::Decoding,
        }
    }
}

impl Decoder {
    fn do_poll_write(
        &mut self,
        cx: &mut Context<'_>,
        input: &mut PartialBuffer<&[u8]>,
        mut writer: Pin<&mut dyn AsyncBufWrite>,
        decoder: &mut dyn DecodeV2,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut output = ready!(writer.as_mut().poll_partial_flush_buf(cx))?;
            let output = &mut output.write_buffer;

            self.state = match self.state {
                State::Decoding => {
                    if decoder.decode(input, output)? {
                        State::Finishing
                    } else {
                        State::Decoding
                    }
                }

                State::Finishing => {
                    if decoder.finish(output)? {
                        State::Done
                    } else {
                        State::Finishing
                    }
                }

                State::Done => {
                    return Poll::Ready(Err(io::Error::other("Write after end of stream")));
                }
            };

            if let State::Done = self.state {
                return Poll::Ready(Ok(()));
            }

            if input.unwritten().is_empty() {
                return Poll::Ready(Ok(()));
            }
        }
    }

    pub fn poll_write(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
        writer: Pin<&mut dyn AsyncBufWrite>,
        decoder: &mut dyn DecodeV2,
    ) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }

        let mut input = PartialBuffer::new(buf);

        match self.do_poll_write(cx, &mut input, writer, decoder)? {
            Poll::Pending if input.written().is_empty() => Poll::Pending,
            _ => Poll::Ready(Ok(input.written().len())),
        }
    }

    pub fn do_poll_flush(
        &mut self,
        cx: &mut Context<'_>,
        mut writer: Pin<&mut dyn AsyncBufWrite>,
        decoder: &mut dyn DecodeV2,
    ) -> Poll<io::Result<()>> {
        loop {
            let mut output = ready!(writer.as_mut().poll_partial_flush_buf(cx))?;
            let output = &mut output.write_buffer;

            let (state, done) = match self.state {
                State::Decoding => {
                    let done = decoder.flush(output)?;
                    (State::Decoding, done)
                }

                State::Finishing => {
                    let before = output.written_len();
                    if decoder.finish(output)? {
                        (State::Done, false)
                    } else if output.written_len() == before && !output.has_no_spare_space() {
                        return Poll::Ready(Err(io::ErrorKind::UnexpectedEof.into()));
                    } else {
                        (State::Finishing, false)
                    }
                }

                State::Done => (State::Done, true),
            };

            self.state = state;

            if done {
                break Poll::Ready(Ok(()));
            }
        }
    }

    pub fn do_close(&mut self) {
        if let State::Decoding = self.state {
            self.state = State::Finishing;
        }
    }

    pub fn is_done(&self) -> bool {
        matches!(self.state, State::Done)
    }
}

macro_rules! impl_decoder {
    ($poll_close: tt) => {
        use crate::{
            codecs::DecodeV2, core::util::PartialBuffer, generic::write::Decoder as GenericDecoder,
        };
        use pin_project_lite::pin_project;
        use std::task::ready;

        pin_project! {
            #[derive(Debug)]
            pub struct Decoder<W, D> {
                #[pin]
                writer: BufWriter<W>,
                decoder: D,
                inner: GenericDecoder,
            }
        }

        impl<W: AsyncWrite, D: DecodeV2> Decoder<W, D> {
            pub fn new(writer: W, decoder: D) -> Self {
                Self {
                    writer: BufWriter::new(writer),
                    decoder,
                    inner: Default::default(),
                }
            }
        }

        impl<W, D> Decoder<W, D> {
            pub fn get_ref(&self) -> &W {
                self.writer.get_ref()
            }

            pub fn get_mut(&mut self) -> &mut W {
                self.writer.get_mut()
            }

            pub fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut W> {
                self.project().writer.get_pin_mut()
            }

            pub fn into_inner(self) -> W {
                self.writer.into_inner()
            }
        }

        impl<W: AsyncWrite, D: DecodeV2> Decoder<W, D> {
            fn do_poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                let mut this = self.project();

                this.inner.do_poll_flush(cx, this.writer, this.decoder)
            }
        }

        impl<W: AsyncWrite, D: DecodeV2> AsyncWrite for Decoder<W, D> {
            fn poll_write(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<io::Result<usize>> {
                let mut this = self.project();

                this.inner.poll_write(cx, buf, this.writer, this.decoder)
            }

            fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                ready!(self.as_mut().do_poll_flush(cx))?;
                self.project().writer.poll_flush(cx)
            }

            fn $poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
                self.as_mut().project().inner.do_close();

                ready!(self.as_mut().do_poll_flush(cx))?;

                let this = self.project();
                if this.inner.is_done() {
                    this.writer.$poll_close(cx)
                } else {
                    Poll::Ready(Err(io::Error::other(
                        "Attempt to close before finishing input",
                    )))
                }
            }
        }

        impl<W: AsyncBufRead, D> AsyncBufRead for Decoder<W, D> {
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
pub(crate) use impl_decoder;
