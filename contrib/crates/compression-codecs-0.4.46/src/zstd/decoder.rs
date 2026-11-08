use crate::{
    zstd::{params::DParameter, OperationExt},
    {DecodeV2, DecodedSize},
};
use compression_core::{
    unshared::Unshared,
    util::{PartialBuffer, WriteBuffer},
};
use libzstd::stream::raw::Decoder;
use std::{
    convert::TryInto,
    io::{self, Result},
};
use zstd_safe::get_error_name;

#[derive(Debug)]
pub struct ZstdDecoder {
    decoder: Unshared<Decoder<'static>>,
    stream_ended: bool,
    needs_input: bool,
}

impl Default for ZstdDecoder {
    fn default() -> Self {
        Self {
            decoder: Unshared::new(Decoder::new().unwrap()),
            stream_ended: false,
            needs_input: false,
        }
    }
}

impl ZstdDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_params(params: &[DParameter]) -> Self {
        let mut decoder = Decoder::new().unwrap();
        for param in params {
            decoder.set_parameter(param.as_zstd()).unwrap();
        }
        Self {
            decoder: Unshared::new(decoder),
            stream_ended: false,
            needs_input: false,
        }
    }

    pub fn new_with_dict(dictionary: &[u8]) -> io::Result<Self> {
        let decoder = Decoder::with_dictionary(dictionary)?;
        Ok(Self {
            decoder: Unshared::new(decoder),
            stream_ended: false,
            needs_input: false,
        })
    }
}

impl DecodeV2 for ZstdDecoder {
    fn reinit(&mut self) -> Result<()> {
        self.decoder.reinit()?;
        self.stream_ended = false;
        self.needs_input = false;
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> Result<bool> {
        let empty_input = input.unwritten().is_empty();
        // Once buffered output is drained, wait for input before calling zstd
        // again. Repeated empty-input calls (e.g. while a reader is Pending)
        // otherwise trigger its no-forward-progress error and poison the context.
        if empty_input && self.needs_input {
            return Ok(false);
        }
        let before = output.written_len();
        let finished = self.decoder.run(input, output)?;
        self.needs_input =
            empty_input && output.written_len() == before && !output.has_no_spare_space();
        if finished {
            self.stream_ended = true;
        }
        Ok(finished)
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        // Note: stream_ended is not updated here because zstd's flush only flushes
        // buffered output and doesn't indicate stream completion. Stream completion
        // is detected in decode() when status.remaining == 0.
        self.decoder.flush(output)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> Result<bool> {
        self.decoder.finish(output)?;

        if self.stream_ended {
            Ok(true)
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "zstd stream did not finish",
            ))
        }
    }
}

impl DecodedSize for ZstdDecoder {
    fn decoded_size(input: &[u8]) -> Result<u64> {
        zstd_safe::find_frame_compressed_size(input)
            .map_err(|error_code| io::Error::other(get_error_name(error_code)))
            .and_then(|size| {
                size.try_into()
                    .map_err(|_| io::Error::from(io::ErrorKind::FileTooLarge))
            })
    }
}
