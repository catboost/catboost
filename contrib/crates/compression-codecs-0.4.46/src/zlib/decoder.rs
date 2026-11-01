use crate::{DecodeV2, FlateDecoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::io::Result;

#[derive(Debug)]
pub struct ZlibDecoder {
    inner: FlateDecoder,
}
impl Default for ZlibDecoder {
    fn default() -> Self {
        Self {
            inner: FlateDecoder::new(true),
        }
    }
}
impl ZlibDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DecodeV2 for ZlibDecoder {
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
