use crate::{DecodeV2, FlateDecoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::io::Result;

#[derive(Debug)]
pub struct DeflateDecoder {
    inner: FlateDecoder,
}

impl Default for DeflateDecoder {
    fn default() -> Self {
        Self {
            inner: FlateDecoder::new(false),
        }
    }
}

impl DeflateDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DecodeV2 for DeflateDecoder {
    fn reinit(&mut self) -> Result<()> {
        self.inner.reinit()?;
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool> {
        self.inner.decode(input, output)
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.inner.flush(output)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.inner.finish(output)
    }
}
