use crate::{
    codecs::DecodeV2,
    core::util::{PartialBuffer, WriteBuffer},
};
use std::{io::Result, ops::ControlFlow, panic::AssertUnwindSafe};

#[derive(Debug)]
enum State {
    Decoding,
    Flushing,
    Done,
    Next,
    Error(AssertUnwindSafe<std::io::Error>),
}

#[derive(Debug)]
pub struct Decoder {
    state: State,
    multiple_members: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            state: State::Decoding,
            multiple_members: false,
        }
    }
}

impl Decoder {
    pub fn multiple_members(&mut self, enabled: bool) {
        self.multiple_members = enabled;
    }

    pub fn do_poll_read(
        &mut self,
        output: &mut WriteBuffer<'_>,
        decoder: &mut dyn DecodeV2,
        input: &mut PartialBuffer<&[u8]>,
        mut first: bool,
    ) -> ControlFlow<Result<()>> {
        loop {
            self.state = match self.state {
                State::Decoding => {
                    if input.unwritten().is_empty() && !first {
                        // Avoid attempting to reinitialise the decoder if the
                        // reader has returned EOF.
                        self.multiple_members = false;

                        State::Flushing
                    } else {
                        match decoder.decode(input, output) {
                            Ok(true) => State::Flushing,
                            // ignore the first error, occurs when input is empty
                            // but we need to run decode to flush
                            Err(err) if !first => {
                                self.state = State::Error(AssertUnwindSafe(err));
                                if output.written_len() > 0 {
                                    return ControlFlow::Break(Ok(()));
                                } else {
                                    continue;
                                }
                            }
                            // poll for more data for the next decode
                            _ => break,
                        }
                    }
                }

                State::Flushing => {
                    let before = output.written_len();
                    let result = decoder.finish(output).and_then(|finished| {
                        if !finished
                            && output.written_len() == before
                            && !output.has_no_spare_space()
                        {
                            Err(std::io::ErrorKind::UnexpectedEof.into())
                        } else {
                            Ok(finished)
                        }
                    });
                    match result {
                        Ok(true) => {
                            if self.multiple_members {
                                if let Err(err) = decoder.reinit() {
                                    self.state = State::Error(AssertUnwindSafe(err));
                                    if output.written_len() > 0 {
                                        return ControlFlow::Break(Ok(()));
                                    } else {
                                        continue;
                                    }
                                }

                                // Poll again only if the decode stage consumed all the input.
                                first = input.unwritten().is_empty();
                                State::Next
                            } else {
                                State::Done
                            }
                        }
                        Ok(false) => State::Flushing,
                        Err(err) => {
                            self.state = State::Error(AssertUnwindSafe(err));
                            if output.written_len() > 0 {
                                return ControlFlow::Break(Ok(()));
                            } else {
                                continue;
                            }
                        }
                    }
                }

                State::Done => return ControlFlow::Break(Ok(())),

                State::Next => {
                    if input.unwritten().is_empty() {
                        if first {
                            // poll for more data to check if there's another stream
                            break;
                        }
                        State::Done
                    } else {
                        State::Decoding
                    }
                }

                State::Error(_) => {
                    let State::Error(err) = std::mem::replace(&mut self.state, State::Done) else {
                        unreachable!()
                    };
                    return ControlFlow::Break(Err(err.0));
                }
            };

            if output.has_no_spare_space() {
                return ControlFlow::Break(Ok(()));
            }
        }

        if output.has_no_spare_space() {
            ControlFlow::Break(Ok(()))
        } else {
            ControlFlow::Continue(())
        }
    }
}

macro_rules! impl_decoder {
    () => {
        use crate::generic::bufread::Decoder as GenericDecoder;

        use std::{ops::ControlFlow, task::ready};

        use pin_project_lite::pin_project;

        pin_project! {
            #[derive(Debug)]
            pub struct Decoder<R, D> {
                #[pin]
                reader: R,
                decoder: D,
                inner: GenericDecoder,
            }
        }

        impl<R: AsyncBufRead, D: DecodeV2> Decoder<R, D> {
            pub fn new(reader: R, decoder: D) -> Self {
                Self {
                    reader,
                    decoder,
                    inner: GenericDecoder::default(),
                }
            }
        }

        impl<R, D> Decoder<R, D> {
            pub fn get_ref(&self) -> &R {
                &self.reader
            }

            pub fn get_mut(&mut self) -> &mut R {
                &mut self.reader
            }

            pub fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut R> {
                self.project().reader
            }

            pub fn into_inner(self) -> R {
                self.reader
            }

            pub fn multiple_members(&mut self, enabled: bool) {
                self.inner.multiple_members(enabled);
            }
        }

        fn do_poll_read(
            inner: &mut GenericDecoder,
            decoder: &mut dyn DecodeV2,
            mut reader: Pin<&mut dyn AsyncBufRead>,
            cx: &mut Context<'_>,
            output: &mut WriteBuffer<'_>,
        ) -> Poll<Result<()>> {
            if let ControlFlow::Break(res) =
                inner.do_poll_read(output, decoder, &mut PartialBuffer::new(&[][..]), true)
            {
                return Poll::Ready(res);
            }

            loop {
                let mut input = PartialBuffer::new(match reader.as_mut().poll_fill_buf(cx)? {
                    Poll::Ready(input) => input,
                    Poll::Pending if output.written().is_empty() => return Poll::Pending,
                    _ => return Poll::Ready(Ok(())),
                });

                let control_flow = inner.do_poll_read(output, decoder, &mut input, false);

                let bytes_read = input.written().len();
                reader.as_mut().consume(bytes_read);

                if let ControlFlow::Break(res) = control_flow {
                    break Poll::Ready(res);
                }
            }
        }

        impl<R: AsyncBufRead, D: DecodeV2> Decoder<R, D> {
            fn do_poll_read(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                output: &mut WriteBuffer<'_>,
            ) -> Poll<Result<()>> {
                let this = self.project();

                do_poll_read(this.inner, this.decoder, this.reader, cx, output)
            }
        }
    };
}
pub(crate) use impl_decoder;
