use compression_core::{
    util::{PartialBuffer, WriteBuffer},
    Level,
};
use liblzma::stream::{Action, Check, Stream};
use std::{
    convert::{TryFrom, TryInto},
    fmt, io,
};

use crate::{
    lzma::params::{LzmaEncoderParams, LzmaOptions},
    xz2::process_stream,
    EncodeV2, Xz2FileFormat,
};

/// Xz2 encoding stream
pub struct Xz2Encoder {
    stream: Stream,
    params: LzmaEncoderParams,
}

impl fmt::Debug for Xz2Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Xz2Encoder").finish_non_exhaustive()
    }
}

impl TryFrom<LzmaEncoderParams> for Xz2Encoder {
    type Error = liblzma::stream::Error;

    fn try_from(params: LzmaEncoderParams) -> Result<Self, Self::Error> {
        let stream = Stream::try_from(&params)?;
        Ok(Self {
            stream,
            params: params.clone(),
        })
    }
}

fn xz2_level(level: Level) -> u32 {
    match level {
        Level::Fastest => 0,
        Level::Best => 9,
        Level::Precise(quality) => quality.try_into().unwrap_or(0).clamp(0, 9),
        _ => 5,
    }
}

impl Xz2Encoder {
    pub fn new(format: Xz2FileFormat, level: Level) -> Self {
        let preset = xz2_level(level);
        let params = match format {
            Xz2FileFormat::Xz => LzmaEncoderParams::Easy {
                preset,
                check: Check::Crc64,
            },
            Xz2FileFormat::Lzma => {
                let options = LzmaOptions::default().preset(preset);
                LzmaEncoderParams::Lzma { options }
            }
        };

        Self::try_from(params).unwrap()
    }

    #[cfg(feature = "xz-parallel")]
    pub fn xz_parallel(level: Level, threads: std::num::NonZeroU32) -> Self {
        use crate::lzma::params::MtStreamBuilder;

        let preset = xz2_level(level);
        let mut builder = MtStreamBuilder::default();
        builder
            .threads(threads)
            .timeout_ms(300)
            .preset(preset)
            .check(Check::Crc64);
        let params = LzmaEncoderParams::MultiThread { builder };
        Self::try_from(params).unwrap()
    }
}

impl EncodeV2 for Xz2Encoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<()> {
        process_stream(&mut self.stream, input, output, Action::Run).map(|_| ())
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        let action = match &self.params {
            // Multi-threaded streams don't support SyncFlush, use FullFlush instead
            #[cfg(feature = "xz-parallel")]
            LzmaEncoderParams::MultiThread { builder: _ } => Action::FullFlush,
            _ => Action::SyncFlush,
        };

        process_stream(
            &mut self.stream,
            &mut PartialBuffer::new(&[]),
            output,
            action,
        )
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        process_stream(
            &mut self.stream,
            &mut PartialBuffer::new(&[]),
            output,
            Action::Finish,
        )
    }
}
