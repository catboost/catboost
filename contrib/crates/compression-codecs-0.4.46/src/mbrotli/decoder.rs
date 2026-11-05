use crate::DecodeV2;
use compression_core::util::{PartialBuffer, WriteBuffer};
use mbrotli::{
    DecodeOperation, DecodeStreamConfig, DecoderConfig, DecoderSessionOwned, DecoderStatus,
    Decompressor, WindowLimit,
};
use std::io;

/// Only accept RFC 7932 windows, as the `brotli` backend does. Large windows (RFC 9841) are not
/// valid in the `br` content coding.
const DECODER_CONFIG: DecoderConfig = DecoderConfig::new().with_window_limit(
    match WindowLimit::standard(mbrotli::Window::MAX_STANDARD_BITS) {
        Ok(limit) => limit,
        Err(_) => panic!("the standard window limit is valid"),
    },
);

#[derive(Debug)]
pub struct MbrotliDecoder {
    session: DecoderSessionOwned,
}

impl Default for MbrotliDecoder {
    fn default() -> Self {
        let decompressor = Decompressor::new(DECODER_CONFIG).unwrap();
        // A fresh decompressor without an exact output size has no way to refuse a session.
        let session = decompressor
            .into_session(DecodeStreamConfig::default())
            .unwrap();
        Self { session }
    }
}

impl MbrotliDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Feeds `input` to the session, writing straight into the uninitialized tail of `output`.
///
/// Returns how much input was consumed and what the session needs next.
fn process(
    session: &mut DecoderSessionOwned,
    input: &[u8],
    output: &mut WriteBuffer<'_>,
    operation: DecodeOperation,
) -> (usize, io::Result<DecoderStatus>) {
    // SAFETY: `process_uninit` never de-initializes bytes of `output`.
    let result = session.process_uninit(input, unsafe { output.unwritten_mut() }, operation);
    // `process_uninit` initializes exactly `produced` leading bytes of `output`, both when it
    // succeeds and when it fails.
    match result {
        Ok(progress) => {
            // SAFETY: see above.
            unsafe { output.assume_init_and_advance(progress.produced) };
            (progress.consumed, Ok(progress.status))
        }
        Err(failure) => {
            // SAFETY: see above.
            unsafe { output.assume_init_and_advance(failure.produced) };
            (failure.consumed, Err(failure.into_error().into()))
        }
    }
}

impl DecodeV2 for MbrotliDecoder {
    fn reinit(&mut self) -> io::Result<()> {
        self.session.reinit(DecodeStreamConfig::default())?;
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<bool> {
        let (consumed, status) = process(
            &mut self.session,
            input.unwritten(),
            output,
            DecodeOperation::Process,
        );
        input.advance(consumed);

        match status? {
            DecoderStatus::Finished => Ok(true),
            DecoderStatus::NeedsInput | DecoderStatus::NeedsOutput => Ok(false),
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        // Empty input with `Process` delivers pending output without declaring EOF.
        let (_, status) = process(&mut self.session, &[], output, DecodeOperation::Process);

        match status? {
            DecoderStatus::Finished | DecoderStatus::NeedsInput => Ok(true),
            DecoderStatus::NeedsOutput => Ok(false),
        }
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        // Declaring EOF on a member that is still open fails with `UnexpectedEof`.
        let (_, status) = process(&mut self.session, &[], output, DecodeOperation::Finish);

        match status? {
            DecoderStatus::Finished => Ok(true),
            DecoderStatus::NeedsInput | DecoderStatus::NeedsOutput => Ok(false),
        }
    }
}
