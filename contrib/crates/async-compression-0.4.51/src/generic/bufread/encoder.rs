use crate::{
    codecs::EncodeV2,
    core::util::{PartialBuffer, WriteBuffer},
};
use std::{io::Result, ops::ControlFlow, panic::AssertUnwindSafe};

#[derive(Debug)]
enum State {
    Encoding(usize),
    Flushing,
    Finishing,
    Done,
    Error(AssertUnwindSafe<std::io::Error>),
}

#[derive(Debug)]
pub struct Encoder {
    state: State,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            state: State::Encoding(0),
        }
    }
}

impl Encoder {
    /// `input` - should be `None` if `Poll::Pending`.
    pub fn do_poll_read(
        &mut self,
        output: &mut WriteBuffer<'_>,
        encoder: &mut dyn EncodeV2,
        mut input: Option<&mut PartialBuffer<&[u8]>>,
    ) -> ControlFlow<Result<()>> {
        loop {
            self.state = match &mut self.state {
                State::Encoding(read) => match input.as_mut() {
                    None => {
                        if *read == 0 {
                            if output.written().is_empty() {
                                // Poll for more data
                                break;
                            } else {
                                return ControlFlow::Break(Ok(()));
                            }
                        } else {
                            State::Flushing
                        }
                    }
                    Some(input) => {
                        if input.unwritten().is_empty() {
                            State::Finishing
                        } else {
                            if let Err(err) = encoder.encode(input, output) {
                                self.state = State::Error(AssertUnwindSafe(err));
                                if output.written_len() > 0 {
                                    return ControlFlow::Break(Ok(()));
                                } else {
                                    continue;
                                }
                            }

                            *read += input.written().len();

                            // Poll for more data
                            break;
                        }
                    }
                },

                State::Flushing => match encoder.flush(output) {
                    Ok(true) => {
                        self.state = State::Encoding(0);

                        // Poll for more data
                        break;
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
                },

                State::Finishing => match encoder.finish(output) {
                    Ok(true) => State::Done,
                    Ok(false) => State::Finishing,
                    Err(err) => {
                        self.state = State::Error(AssertUnwindSafe(err));
                        if output.written_len() > 0 {
                            return ControlFlow::Break(Ok(()));
                        } else {
                            continue;
                        }
                    }
                },

                State::Done => return ControlFlow::Break(Ok(())),

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

macro_rules! impl_encoder {
    () => {
        use crate::generic::bufread::Encoder as GenericEncoder;

        use std::{ops::ControlFlow, task::ready};

        use pin_project_lite::pin_project;

        pin_project! {
            #[derive(Debug)]
            pub struct Encoder<R, E> {
                #[pin]
                reader: R,
                encoder: E,
                inner: GenericEncoder,
            }
        }

        impl<R: AsyncBufRead, E: EncodeV2> Encoder<R, E> {
            pub fn new(reader: R, encoder: E) -> Self {
                Self {
                    reader,
                    encoder,
                    inner: Default::default(),
                }
            }

            pub fn with_capacity(reader: R, encoder: E, _cap: usize) -> Self {
                Self::new(reader, encoder)
            }
        }

        impl<R, E> Encoder<R, E> {
            pub fn get_ref(&self) -> &R {
                &self.reader
            }

            pub fn get_mut(&mut self) -> &mut R {
                &mut self.reader
            }

            pub fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut R> {
                self.project().reader
            }

            pub(crate) fn get_encoder_ref(&self) -> &E {
                &self.encoder
            }

            pub fn into_inner(self) -> R {
                self.reader
            }
        }

        fn do_poll_read(
            inner: &mut GenericEncoder,
            encoder: &mut dyn EncodeV2,
            mut reader: Pin<&mut dyn AsyncBufRead>,
            cx: &mut Context<'_>,
            output: &mut WriteBuffer<'_>,
        ) -> Poll<Result<()>> {
            if let ControlFlow::Break(res) = inner.do_poll_read(output, encoder, None) {
                return Poll::Ready(res);
            }

            loop {
                let mut input = match reader.as_mut().poll_fill_buf(cx) {
                    Poll::Pending => None,
                    Poll::Ready(res) => Some(PartialBuffer::new(res?)),
                };

                let control_flow = inner.do_poll_read(output, encoder, input.as_mut());

                let is_pending = input.is_none();
                if let Some(input) = input {
                    let len = input.written().len();
                    reader.as_mut().consume(len);
                }

                if let ControlFlow::Break(res) = control_flow {
                    break Poll::Ready(res);
                }

                if is_pending {
                    if output.written().is_empty() {
                        return Poll::Pending;
                    } else {
                        return Poll::Ready(Ok(()));
                    }
                }
            }
        }

        impl<R: AsyncBufRead, E: EncodeV2> Encoder<R, E> {
            fn do_poll_read(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                output: &mut WriteBuffer<'_>,
            ) -> Poll<Result<()>> {
                let this = self.project();

                do_poll_read(this.inner, this.encoder, this.reader, cx, output)
            }
        }
    };
}
pub(crate) use impl_encoder;
