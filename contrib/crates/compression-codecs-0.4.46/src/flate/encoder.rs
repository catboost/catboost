use crate::{flate::params::FlateEncoderParams, EncodeV2};
use compression_core::util::{PartialBuffer, WriteBuffer};
use flate2::{Compress, FlushCompress, Status};
use std::io;

#[derive(Debug)]
pub struct FlateEncoder {
    compress: Compress,
    flushed: bool,
}

impl FlateEncoder {
    pub fn new(level: FlateEncoderParams, zlib_header: bool) -> Self {
        let level = flate2::Compression::from(level);
        Self {
            compress: Compress::new(level, zlib_header),
            flushed: true,
        }
    }

    pub fn get_ref(&self) -> &Compress {
        &self.compress
    }

    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
        flush: FlushCompress,
    ) -> io::Result<Status> {
        let prior_in = self.compress.total_in();
        let prior_out = self.compress.total_out();

        let status = self
            .compress
            // Safety: We **trust** flate2 to not write uninitialized bytes into buffer
            .compress_uninit(input.unwritten(), unsafe { output.unwritten_mut() }, flush)?;

        input.advance((self.compress.total_in() - prior_in) as usize);
        // Safety: We **trust** flate2 to write bytes properly into buffer
        unsafe { output.assume_init_and_advance((self.compress.total_out() - prior_out) as usize) };

        Ok(status)
    }
}

impl EncodeV2 for FlateEncoder {
    fn encode(
        &mut self,
        input: &mut PartialBuffer<&[u8]>,
        output: &mut WriteBuffer<'_>,
    ) -> io::Result<()> {
        self.flushed = false;
        match self.encode(input, output, FlushCompress::None)? {
            Status::Ok => Ok(()),
            Status::StreamEnd => unreachable!(),
            Status::BufError => Err(io::Error::other("unexpected BufError")),
        }
    }

    fn flush(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        // We need to keep track of whether we've already flushed otherwise we'll just keep writing
        // out sync blocks continuously and probably never complete flushing.
        if self.flushed {
            return Ok(true);
        }

        self.encode(
            &mut PartialBuffer::new(&[][..]),
            output,
            FlushCompress::Sync,
        )?;

        loop {
            let old_len = output.written_len();
            self.encode(
                &mut PartialBuffer::new(&[][..]),
                output,
                FlushCompress::None,
            )?;
            if output.written_len() == old_len {
                break;
            }
        }

        let internal_flushed = !output.has_no_spare_space();
        self.flushed = internal_flushed;
        Ok(internal_flushed)
    }

    fn finish(&mut self, output: &mut WriteBuffer<'_>) -> io::Result<bool> {
        self.flushed = false;
        match self.encode(
            &mut PartialBuffer::new(&[][..]),
            output,
            FlushCompress::Finish,
        )? {
            Status::Ok => Ok(false),
            Status::StreamEnd => Ok(true),
            Status::BufError => Err(io::Error::other("unexpected BufError")),
        }
    }
}
