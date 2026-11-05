use compression_core::util::PartialBuffer;
use flate2::Crc;
use std::io;

#[derive(Debug, Default)]
struct Flags {
    _ascii: bool,
    crc: bool,
    extra: bool,
    filename: bool,
    comment: bool,
}

#[derive(Debug, Default)]
pub(super) struct Header {
    flags: Flags,
}

#[derive(Debug)]
enum State {
    Fixed(PartialBuffer<[u8; 10]>),
    ExtraLen(PartialBuffer<[u8; 2]>),
    Extra(usize),
    Filename,
    Comment,
    Crc(PartialBuffer<[u8; 2]>),
    Done,
}

impl Default for State {
    fn default() -> Self {
        State::Fixed(<_>::default())
    }
}

#[derive(Debug, Default)]
pub(super) struct Parser {
    state: State,
    header: Header,
}

/// The fixed prefix every gzip member starts with: magic bytes plus the deflate method.
const MAGIC: [u8; 3] = [0x1f, 0x8b, 0x08];

impl Header {
    fn parse(input: &[u8; 10]) -> io::Result<Self> {
        if input[0..3] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid gzip header",
            ));
        }

        let flag = input[3];

        let flags = Flags {
            _ascii: (flag & 0b0000_0001) != 0,
            crc: (flag & 0b0000_0010) != 0,
            extra: (flag & 0b0000_0100) != 0,
            filename: (flag & 0b0000_1000) != 0,
            comment: (flag & 0b0001_0000) != 0,
        };

        Ok(Header { flags })
    }
}

fn consume_input(crc: &mut Crc, n: usize, input: &mut PartialBuffer<&[u8]>) {
    crc.update(&input.unwritten()[..n]);
    input.advance(n);
}

fn consume_cstr(crc: &mut Crc, input: &mut PartialBuffer<&[u8]>) -> Option<()> {
    if let Some(len) = memchr::memchr(0, input.unwritten()) {
        consume_input(crc, len + 1, input);
        Some(())
    } else {
        consume_input(crc, input.unwritten().len(), input);
        None
    }
}

impl Parser {
    pub(super) fn input(
        &mut self,
        crc: &mut Crc,
        input: &mut PartialBuffer<&[u8]>,
    ) -> io::Result<Option<Header>> {
        loop {
            match &mut self.state {
                State::Fixed(data) => {
                    data.copy_unwritten_from(input);

                    // `MAGIC` decides validity, so reject as soon as the bytes seen so far
                    // contradict it instead of waiting for all 10. A reader that stalls part
                    // way through the header would otherwise never resolve, which matters for
                    // `multiple_members` over a live stream.
                    let seen = data.written();
                    let checked = seen.len().min(MAGIC.len());
                    if seen[..checked] != MAGIC[..checked] {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Invalid gzip header",
                        ));
                    }

                    if data.unwritten().is_empty() {
                        let data = data.get_mut();
                        crc.update(data);
                        self.header = Header::parse(data)?;
                        self.state = State::ExtraLen(<_>::default());
                    } else {
                        break Ok(None);
                    }
                }

                State::ExtraLen(data) => {
                    if !self.header.flags.extra {
                        self.state = State::Filename;
                        continue;
                    }

                    data.copy_unwritten_from(input);

                    if data.unwritten().is_empty() {
                        let data = data.get_mut();
                        crc.update(data);
                        let len = u16::from_le_bytes(*data);
                        self.state = State::Extra(len.into());
                    } else {
                        break Ok(None);
                    }
                }

                State::Extra(bytes_to_consume) => {
                    let n = input.unwritten().len().min(*bytes_to_consume);
                    *bytes_to_consume -= n;
                    consume_input(crc, n, input);

                    if *bytes_to_consume == 0 {
                        self.state = State::Filename;
                    } else {
                        break Ok(None);
                    }
                }

                State::Filename => {
                    if !self.header.flags.filename {
                        self.state = State::Comment;
                        continue;
                    }

                    if consume_cstr(crc, input).is_none() {
                        break Ok(None);
                    }

                    self.state = State::Comment;
                }

                State::Comment => {
                    if !self.header.flags.comment {
                        self.state = State::Crc(<_>::default());
                        continue;
                    }

                    if consume_cstr(crc, input).is_none() {
                        break Ok(None);
                    }

                    self.state = State::Crc(<_>::default());
                }

                State::Crc(data) => {
                    let header = std::mem::take(&mut self.header);

                    if !self.header.flags.crc {
                        self.state = State::Done;
                        break Ok(Some(header));
                    }

                    data.copy_unwritten_from(input);

                    break if data.unwritten().is_empty() {
                        let data = data.take().into_inner();
                        self.state = State::Done;
                        let checksum = crc.sum().to_le_bytes();

                        if data == checksum[..2] {
                            Ok(Some(header))
                        } else {
                            Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "CRC computed for header does not match",
                            ))
                        }
                    } else {
                        Ok(None)
                    };
                }

                State::Done => break Err(io::Error::other("parser used after done")),
            }
        }
    }
}
