use crate::{DecodeV2, DecodedSize, Xz2Decoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::{convert::TryInto, io::Result};

/// Lzma decoding stream
#[derive(Debug)]
pub struct LzmaDecoder {
    inner: Xz2Decoder,
}

impl From<Xz2Decoder> for LzmaDecoder {
    fn from(inner: Xz2Decoder) -> Self {
        Self { inner }
    }
}

impl Default for LzmaDecoder {
    fn default() -> Self {
        Self {
            inner: Xz2Decoder::new(usize::MAX.try_into().unwrap()),
        }
    }
}

impl LzmaDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_memlimit(memlimit: u64) -> Self {
        Self {
            inner: Xz2Decoder::new(memlimit),
        }
    }
}

impl DecodeV2 for LzmaDecoder {
    fn reinit(&mut self) -> Result<()> {
        self.inner.reinit()
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

impl DecodedSize for LzmaDecoder {
    fn decoded_size(input: &[u8]) -> Result<u64> {
        Xz2Decoder::decoded_size(input)
    }
}
