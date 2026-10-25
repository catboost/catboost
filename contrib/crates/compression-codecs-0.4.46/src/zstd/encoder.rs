use crate::{
    zstd::{params::CParameter, OperationExt},
    EncodeV2,
};
use compression_core::{
    unshared::Unshared,
    util::{PartialBuffer, WriteBuffer},
};
use libzstd::stream::raw::Encoder;
use std::io::{self, Result};

#[derive(Debug)]
pub struct ZstdEncoder {
    encoder: Unshared<Encoder<'static>>,
}

impl ZstdEncoder {
    pub fn new(level: i32) -> Self {
        Self {
            encoder: Unshared::new(Encoder::new(level).unwrap()),
        }
    }

    pub fn new_with_params(level: i32, params: &[CParameter]) -> Self {
        let mut encoder = Encoder::new(level).unwrap();
        for param in params {
            encoder.set_parameter(param.as_zstd()).unwrap();
        }
        Self {
            encoder: Unshared::new(encoder),
        }
    }

    pub fn new_with_dict(level: i32, dictionary: &[u8]) -> io::Result<Self> {
        let encoder = Encoder::with_dictionary(level, dictionary)?;
        Ok(Self {
            encoder: Unshared::new(encoder),
        })
    }
}

impl EncodeV2 for ZstdEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<()> {
        self.encoder.run(input, output)?;
        Ok(())
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.encoder.flush(output)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.encoder.finish(output)
    }
}
