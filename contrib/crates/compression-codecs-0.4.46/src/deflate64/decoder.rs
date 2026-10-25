use crate::DecodeV2;
use compression_core::util::{PartialBuffer, WriteBuffer};
use deflate64::InflaterManaged;
use std::io::{Error, ErrorKind, Result};

#[derive(Debug)]
pub struct Deflate64Decoder {
    inflater: Box<InflaterManaged>,
}

impl Default for Deflate64Decoder {
    fn default() -> Self {
        Self {
            inflater: Box::new(InflaterManaged::new()),
        }
    }
}

impl Deflate64Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool> {
        let result = self
            .inflater
            // Safety: We **trust** deflate64 to not write uninitialized bytes
            .inflate_uninit(input.unwritten(), unsafe { output.unwritten_mut() });

        if result.data_error {
            Err(Error::new(ErrorKind::InvalidData, "invalid data"))
        } else {
            input.advance(result.bytes_consumed);
            // Safety: We **trust** deflate64 to properly write bytes into buffer
            unsafe { output.assume_init_and_advance(result.bytes_written) };

            Ok(self.inflater.finished() && self.inflater.available_output() == 0)
        }
    }
}

impl DecodeV2 for Deflate64Decoder {
    fn reinit(&mut self) -> Result<()> {
        *self.inflater = InflaterManaged::new();
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool> {
        self.decode(input, output)
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.decode(&mut PartialBuffer::new(&[]), output)?;

        loop {
            let old_len = output.written_len();
            self.decode(&mut PartialBuffer::new(&[]), output)?;
            if output.written_len() == old_len {
                break;
            }
        }

        Ok(!output.has_no_spare_space())
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        let before = output.written_len();
        let finished = self.decode(&mut PartialBuffer::new(&[]), output)?;
        if !finished && output.written_len() == before && !output.has_no_spare_space() {
            Err(ErrorKind::UnexpectedEof.into())
        } else {
            Ok(finished)
        }
    }
}
