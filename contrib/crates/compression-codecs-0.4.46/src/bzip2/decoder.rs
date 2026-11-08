use crate::DecodeV2;
use bzip2::{Decompress, Status};
use compression_core::util::{PartialBuffer, WriteBuffer};
use std::{fmt, io};

pub struct BzDecoder {
    decompress: Decompress,
    stream_ended: bool,
}

impl fmt::Debug for BzDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BzDecoder {{total_in: {}, total_out: {}}}",
            self.decompress.total_in(),
            self.decompress.total_out()
        )
    }
}

impl Default for BzDecoder {
    fn default() -> Self {
        Self {
            decompress: Decompress::new(false),
            stream_ended: false,
        }
    }
}

impl BzDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<Status> {
        let prior_in = self.decompress.total_in();
        let prior_out = self.decompress.total_out();

        let status = self
            .decompress
            // Safety: We **trust** bzip2 to only write initialized data to it
            .decompress_uninit(input.unwritten(), unsafe { output.unwritten_mut() })
            .map_err(io::Error::other)?;

        input.advance((self.decompress.total_in() - prior_in) as usize);
        // Safety: We **trust** bzip2 to write bytes properly
        unsafe {
            output.assume_init_and_advance((self.decompress.total_out() - prior_out) as usize)
        };

        // Track when stream has properly ended
        if status == Status::StreamEnd {
            self.stream_ended = true;
        }

        Ok(status)
    }
}

impl DecodeV2 for BzDecoder {
    fn reinit(&mut self) -> io::Result<()> {
        self.decompress = Decompress::new(false);
        self.stream_ended = false;
        Ok(())
    }

    fn decode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<bool> {
        match self.decode(input, output)? {
            // Decompression went fine, nothing much to report.
            Status::Ok => Ok(false),

            // The Flush action on a compression went ok.
            Status::FlushOk => unreachable!(),

            // THe Run action on compression went ok.
            Status::RunOk => unreachable!(),

            // The Finish action on compression went ok.
            Status::FinishOk => unreachable!(),

            // The stream's end has been met, meaning that no more data can be input.
            Status::StreamEnd => Ok(true),

            // There was insufficient memory in the input or output buffer to complete
            // the request, but otherwise everything went normally.
            Status::MemNeeded => Err(io::ErrorKind::OutOfMemory.into()),
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        self.decode(&mut PartialBuffer::new(&[][..]), output)?;

        loop {
            let old_len = output.written_len();
            self.decode(&mut PartialBuffer::new(&[][..]), output)?;
            if output.written_len() == old_len {
                break;
            }
        }

        Ok(!output.has_no_spare_space())
    }

    fn finish(&mut self, _output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        if self.stream_ended {
            Ok(true)
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "bzip2 stream did not finish",
            ))
        }
    }
}
