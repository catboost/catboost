use crate::{DecodeV2, DecodedSize, Xz2Decoder};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::{
    convert::TryInto,
    io::{Error, ErrorKind, Result},
};

/// Xz decoding stream
#[derive(Debug)]
pub struct XzDecoder {
    inner: Xz2Decoder,
    skip_padding: Option<u8>,
}

impl Default for XzDecoder {
    fn default() -> Self {
        Self {
            inner: Xz2Decoder::new(usize::MAX.try_into().unwrap()),
            skip_padding: None,
        }
    }
}

impl XzDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_memlimit(memlimit: u64) -> Self {
        Self {
            inner: Xz2Decoder::new(memlimit),
            skip_padding: None,
        }
    }

    #[cfg(feature = "xz-parallel")]
    pub fn parallel(threads: std::num::NonZeroU32, memlimit: u64) -> Self {
        Self {
            inner: Xz2Decoder::parallel(threads, memlimit),
            skip_padding: None,
        }
    }
}

impl DecodeV2 for XzDecoder {
    fn reinit(&mut self) -> Result<()> {
        self.skip_padding = Some(4);
        self.inner.reinit()
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool> {
        if let Some(ref mut count) = self.skip_padding {
            while input.unwritten().first() == Some(&0) {
                input.advance(1);
                *count -= 1;
                if *count == 0 {
                    *count = 4;
                }
            }
            if input.unwritten().is_empty() {
                return Ok(true);
            }
            // If this is non-padding then it cannot start with null bytes, so it must be invalid
            // padding
            if *count != 4 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "stream padding was not a multiple of 4 bytes",
                ));
            }
            self.skip_padding = None;
        }
        self.inner.decode(input, output)
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        if self.skip_padding.is_some() {
            return Ok(true);
        }
        self.inner.flush(output)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        if self.skip_padding.is_some() {
            return Ok(true);
        }
        self.inner.finish(output)
    }
}

impl DecodedSize for XzDecoder {
    fn decoded_size(input: &[u8]) -> Result<u64> {
        Xz2Decoder::decoded_size(input)
    }
}
