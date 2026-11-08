use crate::{flate::params::FlateEncoderParams, EncodeV2, FlateEncoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::io::Result;

#[derive(Debug)]
pub struct DeflateEncoder {
    inner: FlateEncoder,
}

impl DeflateEncoder {
    pub fn new(level: FlateEncoderParams) -> Self {
        Self {
            inner: FlateEncoder::new(level, false),
        }
    }

    pub fn get_ref(&self) -> &FlateEncoder {
        &self.inner
    }
}

impl EncodeV2 for DeflateEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<()> {
        self.inner.encode(input, output)
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.inner.flush(output)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.inner.finish(output)
    }
}
