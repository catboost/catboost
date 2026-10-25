use crate::{EncodeV2, Xz2Encoder, Xz2FileFormat};
use compression_core::{
    util::{PartialBuffer, WriteBuffer},
    Level,
};
use std::io::Result;

/// Lzma encoding stream
#[derive(Debug)]
pub struct LzmaEncoder {
    inner: Xz2Encoder,
}

impl LzmaEncoder {
    pub fn new(level: Level) -> Self {
        Self {
            inner: Xz2Encoder::new(Xz2FileFormat::Lzma, level),
        }
    }
}

impl From<Xz2Encoder> for LzmaEncoder {
    fn from(inner: Xz2Encoder) -> Self {
        Self { inner }
    }
}

impl EncodeV2 for LzmaEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<()> {
        self.inner.encode(input, output)
    }

    fn flush(&mut self, _output: &mut WriteBuffer<'_>) -> Result<bool> {
        // Flush on LZMA 1 is not supported
        Ok(true)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.inner.finish(output)
    }
}
