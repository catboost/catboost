use crate::DecodeV2;
use brotli::{enc::StandardAlloc, BrotliDecompressStream, BrotliResult};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::{fmt, io};

type BrotliState = brotli::BrotliState<StandardAlloc, StandardAlloc, StandardAlloc>;

pub struct BrotliDecoder {
    // `BrotliState` is very large (over 2kb) which is why we're boxing it.
    state: Box<BrotliState>,
}

impl Default for BrotliDecoder {
    fn default() -> Self {
        Self {
            state: Box::new(Self::new_brotli_state()),
        }
    }
}

impl BrotliDecoder {
    fn new_brotli_state() -> BrotliState {
        BrotliState::new(
            StandardAlloc::default(),
            StandardAlloc::default(),
            StandardAlloc::default(),
        )
    }

    pub fn new() -> Self {
        Self::default()
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<BrotliResult> {
        let in_buf = input.unwritten();
        let out_buf = output.initialize_unwritten();

        let mut input_len = 0;
        let mut output_len = 0;

        let status = match BrotliDecompressStream(
            &mut in_buf.len(),
            &mut input_len,
            in_buf,
            &mut out_buf.len(),
            &mut output_len,
            out_buf,
            &mut 0,
            &mut self.state,
        ) {
            BrotliResult::ResultFailure => return Err(io::Error::other("brotli error")),
            status => status,
        };

        input.advance(input_len);
        output.advance(output_len);

        Ok(status)
    }
}

impl DecodeV2 for BrotliDecoder {
    fn reinit(&mut self) -> io::Result<()> {
        *self.state = Self::new_brotli_state();
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<bool> {
        match self.decode(input, output)? {
            BrotliResult::ResultSuccess => Ok(true),
            BrotliResult::NeedsMoreOutput | BrotliResult::NeedsMoreInput => Ok(false),
            BrotliResult::ResultFailure => unreachable!(),
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        match self.decode(&mut PartialBuffer::new(&[][..]), output)? {
            BrotliResult::ResultSuccess | BrotliResult::NeedsMoreInput => Ok(true),
            BrotliResult::NeedsMoreOutput => Ok(false),
            BrotliResult::ResultFailure => unreachable!(),
        }
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        match self.decode(&mut PartialBuffer::new(&[][..]), output)? {
            BrotliResult::ResultSuccess => Ok(true),
            BrotliResult::NeedsMoreOutput => Ok(false),
            BrotliResult::NeedsMoreInput => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "reached unexpected EOF",
            )),
            BrotliResult::ResultFailure => unreachable!(),
        }
    }
}

impl fmt::Debug for BrotliDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BrotliDecoder")
            .field("decompress", &"<no debug>")
            .finish()
    }
}
