use crate::{mbrotli::params::EncoderParams, EncodeV2};
use compression_core::util::{PartialBuffer, WriteBuffer};
use mbrotli::{Compressor, EncoderSessionOwned, EncoderStatus, Operation};
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
    let progress = session.process_uninit(input, unsafe { output.unwritten_mut() }, operation)?;
    // SAFETY: `process_uninit` initializes exactly `produced` leading bytes of `output`.
    unsafe { output.assume_init_and_advance(progress.produced) };
    Ok((progress.consumed, progress.status))
}

#[derive(Debug)]
pub struct MbrotliEncoder {
    session: EncoderSessionOwned,
}

impl MbrotliEncoder {
    pub fn new(params: EncoderParams) -> Self {
        // `EncoderParams` only builds standard windows, the one combination mbrotli rejects at
        // configuration time is a large window at a low quality.
        let compressor = Compressor::new(params.config()).unwrap();
        // A fresh compressor with a zero stream offset has no way to refuse a session other than
        // failing to allocate its workspace.
        let session = compressor.into_session(params.stream()).unwrap();
        Self { session }
    }
}

impl EncodeV2 for MbrotliEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<()> {
        let (consumed, _) = process(
            &mut self.session,
            input.unwritten(),
            output,
            Operation::Process,
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
        let (_, status) = process(&mut self.session, &[], output, Operation::Finish)?;

        match status {
            EncoderStatus::Finished => Ok(true),
            EncoderStatus::NeedsInput | EncoderStatus::NeedsOutput => Ok(false),
        }
    }
}
