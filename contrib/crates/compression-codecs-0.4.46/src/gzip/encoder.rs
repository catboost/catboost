use crate::{flate::params::FlateEncoderParams, EncodeV2, FlateEncoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use flate2::{Compression, Crc};
use std::io;

#[derive(Debug)]
enum State {
    Header(PartialBuffer<[u8; 10]>),
    Encoding,
    Footer(PartialBuffer<[u8; 8]>),
    Done,
}

#[derive(Debug)]
pub struct GzipEncoder {
    inner: FlateEncoder,
    crc: Crc,
    state: State,
}

fn header(level: Compression) -> [u8; 10] {
    let level_byte = if level.level() >= Compression::best().level() {
        0x02
    } else if level.level() <= Compression::fast().level() {
        0x04
    } else {
        0x00
    };

    [0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, level_byte, 0xff]
}

impl GzipEncoder {
    pub fn new(level: FlateEncoderParams) -> Self {
        Self {
            inner: FlateEncoder::new(level.clone(), false),
            crc: Crc::new(),
            state: State::Header(header(Compression::from(level)).into()),
        }
    }

    fn footer(&mut self) -> [u8; 8] {
        let mut output = [0; 8];

        output[..4].copy_from_slice(&self.crc.sum().to_le_bytes());
        output[4..].copy_from_slice(&self.crc.amount().to_le_bytes());

        output
    }
}

impl EncodeV2 for GzipEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<()> {
        loop {
            match &mut self.state {
                State::Header(header) => {
                    output.copy_unwritten_from(&mut *header);

                    if header.unwritten().is_empty() {
                        self.state = State::Encoding;
                    }
                }

                State::Encoding => {
                    let prior_written = input.written().len();
                    self.inner.encode(input, output)?;
                    self.crc.update(&input.written()[prior_written..]);
                }

                State::Footer(_) | State::Done => {
                    return Err(io::Error::other("encode after complete"));
                }
            };

            if input.unwritten().is_empty() || output.has_no_spare_space() {
                return Ok(());
            }
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        loop {
            let done = match &mut self.state {
                State::Header(header) => {
                    output.copy_unwritten_from(&mut *header);

                    if header.unwritten().is_empty() {
                        self.state = State::Encoding;
                    }
                    false
                }

                State::Encoding => self.inner.flush(output)?,

                State::Footer(footer) => {
                    output.copy_unwritten_from(&mut *footer);

                    if footer.unwritten().is_empty() {
                        self.state = State::Done;
                        true
                    } else {
                        false
                    }
                }

                State::Done => true,
            };

            if done {
                return Ok(true);
            }

            if output.has_no_spare_space() {
                return Ok(false);
            }
        }
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        loop {
            match &mut self.state {
                State::Header(header) => {
                    output.copy_unwritten_from(&mut *header);

                    if header.unwritten().is_empty() {
                        self.state = State::Encoding;
                    }
                }

                State::Encoding => {
                    if self.inner.finish(output)? {
                        self.state = State::Footer(self.footer().into());
                    }
                }

                State::Footer(footer) => {
                    output.copy_unwritten_from(&mut *footer);

                    if footer.unwritten().is_empty() {
                        self.state = State::Done;
                    }
                }

                State::Done => {}
            };

            if let State::Done = self.state {
                return Ok(true);
            }

            if output.has_no_spare_space() {
                return Ok(false);
            }
        }
    }
}
