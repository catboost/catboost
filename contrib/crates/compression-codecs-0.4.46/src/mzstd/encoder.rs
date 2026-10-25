use crate::{
    mzstd::params::{encoder_config, CParameter},
    EncodeV2,
};
use compression_core::util::{PartialBuffer, WriteBuffer};
use mzstd::{
    dictionary::DictionaryBuilder, Compressor, EncoderSessionOwned, EncoderStatus, Operation,
    StreamConfig,
};
use std::io;

/// Feeds `input` to the session, writing straight into the uninitialized tail of `output`.
///
/// Returns how much input was consumed and what the session needs next.
fn process(
    session: &mut EncoderSessionOwned,
    input: &[u8],
    output: &mut WriteBuffer<'_>,
    operation: Operation,
) -> io::Result<(usize, EncoderStatus)> {
    // SAFETY: `process_uninit` never de-initializes bytes of `output`.
    let progress = session
        .process_uninit(input, unsafe { output.unwritten_mut() }, operation)
        .map_err(io::Error::other)?;
    // SAFETY: `process_uninit` initializes exactly `produced` leading bytes of `output`.
    unsafe { output.assume_init_and_advance(progress.produced) };
    Ok((progress.consumed, progress.status))
}

#[derive(Debug)]
pub struct MzstdEncoder {
    session: EncoderSessionOwned,
}

impl MzstdEncoder {
    pub fn new(level: i32) -> Self {
        Self::new_with_params(level, &[])
    }

    /// # Panics
    ///
    /// Panics if a parameter is out of range, as the `zstd` backend does.
    pub fn new_with_params(level: i32, params: &[CParameter]) -> Self {
        let compressor = Compressor::new(encoder_config(level, params)).unwrap();
        Self {
            session: compressor.into_session(StreamConfig::default()),
        }
    }

    pub fn new_with_dict(level: i32, dictionary: &[u8]) -> io::Result<Self> {
        let config = encoder_config(level, &[]);
        let dictionary = DictionaryBuilder::new(dictionary)
            .with_config(config)
            .build()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        // `encoder_config` clamps the level and sets no other parameters, so it is valid.
        let session = Compressor::new(config)
            .unwrap()
            .into_session_with_dictionary(dictionary, StreamConfig::default())
            .map_err(io::Error::other)?;
        Ok(Self { session })
    }
}

impl EncodeV2 for MzstdEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<()> {
        let (consumed, _) = process(
            &mut self.session,
            input.unwritten(),
            output,
            Operation::Continue,
        )?;
        input.advance(consumed);

        Ok(())
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        let (_, status) = process(&mut self.session, &[], output, Operation::Flush)?;

        match status {
            EncoderStatus::NeedsInput | EncoderStatus::Finished => Ok(true),
            EncoderStatus::NeedsOutput => Ok(false),
        }
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        let (_, status) = process(&mut self.session, &[], output, Operation::End)?;

        match status {
            EncoderStatus::Finished => Ok(true),
            EncoderStatus::NeedsInput | EncoderStatus::NeedsOutput => Ok(false),
        }
    }
}
