//! Raw in-memory stream compression/decompression.
//!
//! This module defines a `Decoder` and an `Encoder` to decode/encode streams
//! of data using buffers.
//!
//! They are mostly thin wrappers around `zstd_safe::{DCtx, CCtx}`.
use std::io;

pub use zstd_safe::{CParameter, DParameter, InBuffer, OutBuffer, WriteBuf};

use crate::dict::{DecoderDictionary, EncoderDictionary};
use crate::map_error_code;

/// Represents an abstract compression/decompression operation.
///
/// This trait covers both `Encoder` and `Decoder`.
pub trait Operation {
    /// Performs a single step of this operation.
    ///
    /// Should return a hint for the next input size.
    ///
    /// If the result is `Ok(0)`, it may indicate that a frame was just
    /// finished.
    fn run<C: WriteBuf + ?Sized>(
        &mut self,
        input: &mut InBuffer<'_>,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize>;

    /// Performs a single step of this operation.
    ///
    /// This is a comvenience wrapper around `Operation::run` if you don't
    /// want to deal with `InBuffer`/`OutBuffer`.
    fn run_on_buffers(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> io::Result<Status> {
        let mut input = InBuffer::around(input);
        let mut output = OutBuffer::around(output);

        let remaining = self.run(&mut input, &mut output)?;

        Ok(Status {
            remaining,
            bytes_read: input.pos(),
            bytes_written: output.pos(),
        })
    }

    /// Flushes any internal buffer, if any.
    ///
    /// Returns the number of bytes still in the buffer.
    /// To flush entirely, keep calling until it returns `Ok(0)`.
    fn flush<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        let _ = output;
        Ok(0)
    }

    /// Prepares the operation for a new frame.
    ///
    /// This is hopefully cheaper than creating a new operation.
    fn reinit(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Finishes the operation, writing any footer if necessary.
    ///
    /// Returns the number of bytes still to write.
    ///
    /// Keep calling this method until it returns `Ok(0)`,
    /// and then don't ever call this method.
    fn finish<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        finished_frame: bool,
    ) -> io::Result<usize> {
        let _ = output;
        let _ = finished_frame;
        Ok(0)
    }
}

/// Dummy operation that just copies its input to the output.
pub struct NoOp;

impl Operation for NoOp {
    fn run<C: WriteBuf + ?Sized>(
        &mut self,
        input: &mut InBuffer<'_>,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        // Skip the prelude
        let src = &input.src[input.pos..];
        // Safe because `output.pos() <= output.capacity()`.
        let output_pos = output.pos();
        let dst = unsafe { output.as_mut_ptr().add(output_pos) };

        // Ignore anything past the end
        let len = usize::min(src.len(), output.capacity() - output_pos);
        let src = &src[..len];

        // Safe because:
        // * `len` is less than either of the two lengths
        // * `src` and `dst` do not overlap because we have `&mut` to each.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len) };
        input.set_pos(input.pos() + len);
        unsafe { output.set_pos(output_pos + len) };

        Ok(0)
    }
}

/// Describes the result of an operation.
pub struct Status {
    /// Number of bytes expected for next input.
    ///
    /// * If `remaining = 0`, then we are at the end of a frame.
    /// * If `remaining > 0`, then it's just a hint for how much there is still
    ///   to read.
    pub remaining: usize,

    /// Number of bytes read from the input.
    pub bytes_read: usize,

    /// Number of bytes written to the output.
    pub bytes_written: usize,
}

/// An in-memory decoder for streams of data.
pub struct Decoder<'a> {
    context: MaybeOwnedDCtx<'a>,
}

impl Decoder<'static> {
    /// Creates a new decoder.
    pub fn new() -> io::Result<Self> {
        Self::with_dictionary(&[])
    }

    /// Creates a new decoder initialized with the given dictionary.
    pub fn with_dictionary(dictionary: &[u8]) -> io::Result<Self> {
        let mut context = zstd_safe::DCtx::create();
        context.init().map_err(map_error_code)?;
        context
            .load_dictionary(dictionary)
            .map_err(map_error_code)?;
        Ok(Decoder {
            context: MaybeOwnedDCtx::Owned(context),
        })
    }
}

impl<'a> Decoder<'a> {
    /// Creates a new decoder which employs the provided context for deserialization.
    pub fn with_context(context: &'a mut zstd_safe::DCtx<'static>) -> Self {
        Self {
            context: MaybeOwnedDCtx::Borrowed(context),
        }
    }

    /// Creates a new decoder, using an existing `DecoderDictionary`.
    pub fn with_prepared_dictionary<'b>(
        dictionary: &DecoderDictionary<'b>,
    ) -> io::Result<Self>
    where
        'b: 'a,
    {
        let mut context = zstd_safe::DCtx::create();
        context
            .ref_ddict(dictionary.as_ddict())
            .map_err(map_error_code)?;
        Ok(Decoder {
            context: MaybeOwnedDCtx::Owned(context),
        })
    }

    /// Creates a new decoder, using a ref prefix
    pub fn with_ref_prefix<'b>(ref_prefix: &'b [u8]) -> io::Result<Self>
    where
        'b: 'a,
    {
        let mut context = zstd_safe::DCtx::create();
        context.ref_prefix(ref_prefix).map_err(map_error_code)?;
        Ok(Decoder {
            context: MaybeOwnedDCtx::Owned(context),
        })
    }

    /// Sets a decompression parameter for this decoder.
    pub fn set_parameter(&mut self, parameter: DParameter) -> io::Result<()> {
        match &mut self.context {
            MaybeOwnedDCtx::Owned(x) => x.set_parameter(parameter),
            MaybeOwnedDCtx::Borrowed(x) => x.set_parameter(parameter),
        }
        .map_err(map_error_code)?;
        Ok(())
    }
}

impl Operation for Decoder<'_> {
    fn run<C: WriteBuf + ?Sized>(
        &mut self,
        input: &mut InBuffer<'_>,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        match &mut self.context {
            MaybeOwnedDCtx::Owned(x) => x.decompress_stream(output, input),
            MaybeOwnedDCtx::Borrowed(x) => x.decompress_stream(output, input),
        }
        .map_err(map_error_code)
    }

    fn flush<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        // To flush, we just offer no additional input.
        self.run(&mut InBuffer::around(&[]), output)?;

        // We don't _know_ how much (decompressed data) there is still in buffer.
        if output.pos() < output.capacity() {
            // We only know when there's none (the output buffer is not full).
            Ok(0)
        } else {
            // Otherwise, pretend there's still "1 byte" remaining.
            Ok(1)
        }
    }

    fn reinit(&mut self) -> io::Result<()> {
        match &mut self.context {
            MaybeOwnedDCtx::Owned(x) => {
                x.reset(zstd_safe::ResetDirective::SessionOnly)
            }
            MaybeOwnedDCtx::Borrowed(x) => {
                x.reset(zstd_safe::ResetDirective::SessionOnly)
            }
        }
        .map_err(map_error_code)?;
        Ok(())
    }

    fn finish<C: WriteBuf + ?Sized>(
        &mut self,
        _output: &mut OutBuffer<'_, C>,
        finished_frame: bool,
    ) -> io::Result<usize> {
        if finished_frame {
            Ok(0)
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete frame",
            ))
        }
    }
}

/// An in-memory encoder for streams of data.
pub struct Encoder<'a> {
    context: MaybeOwnedCCtx<'a>,
}

impl Encoder<'static> {
    /// Creates a new encoder.
    pub fn new(level: i32) -> io::Result<Self> {
        Self::with_dictionary(level, &[])
    }

    /// Creates a new encoder initialized with the given dictionary.
    pub fn with_dictionary(level: i32, dictionary: &[u8]) -> io::Result<Self> {
        let mut context = zstd_safe::CCtx::create();

        context
            .set_parameter(CParameter::CompressionLevel(level))
            .map_err(map_error_code)?;

        context
            .load_dictionary(dictionary)
            .map_err(map_error_code)?;

        Ok(Encoder {
            context: MaybeOwnedCCtx::Owned(context),
        })
    }
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder that uses the provided context for serialization.
    pub fn with_context(context: &'a mut zstd_safe::CCtx<'static>) -> Self {
        Self {
            context: MaybeOwnedCCtx::Borrowed(context),
        }
    }

    /// Creates a new encoder using an existing `EncoderDictionary`.
    pub fn with_prepared_dictionary<'b>(
        dictionary: &EncoderDictionary<'b>,
    ) -> io::Result<Self>
    where
        'b: 'a,
    {
        let mut context = zstd_safe::CCtx::create();
        context
            .ref_cdict(dictionary.as_cdict())
            .map_err(map_error_code)?;
        Ok(Encoder {
            context: MaybeOwnedCCtx::Owned(context),
        })
    }

    /// Creates a new encoder initialized with the given ref prefix.
    pub fn with_ref_prefix<'b>(
        level: i32,
        ref_prefix: &'b [u8],
    ) -> io::Result<Self>
    where
        'b: 'a,
    {
        let mut context = zstd_safe::CCtx::create();

        context
            .set_parameter(CParameter::CompressionLevel(level))
            .map_err(map_error_code)?;

        context.ref_prefix(ref_prefix).map_err(map_error_code)?;

        Ok(Encoder {
            context: MaybeOwnedCCtx::Owned(context),
        })
    }

    /// Sets a compression parameter for this encoder.
    pub fn set_parameter(&mut self, parameter: CParameter) -> io::Result<()> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => x.set_parameter(parameter),
            MaybeOwnedCCtx::Borrowed(x) => x.set_parameter(parameter),
        }
        .map_err(map_error_code)?;
        Ok(())
    }

    /// Sets the size of the input expected by zstd.
    ///
    /// May affect compression ratio.
    ///
    /// It is an error to give an incorrect size (an error _will_ be returned when closing the
    /// stream).
    ///
    /// If `None` is given, it assume the size is not known (default behaviour).
    pub fn set_pledged_src_size(
        &mut self,
        pledged_src_size: Option<u64>,
    ) -> io::Result<()> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => {
                x.set_pledged_src_size(pledged_src_size)
            }
            MaybeOwnedCCtx::Borrowed(x) => {
                x.set_pledged_src_size(pledged_src_size)
            }
        }
        .map_err(map_error_code)?;
        Ok(())
    }
}

impl<'a> Operation for Encoder<'a> {
    fn run<C: WriteBuf + ?Sized>(
        &mut self,
        input: &mut InBuffer<'_>,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => x.compress_stream(output, input),
            MaybeOwnedCCtx::Borrowed(x) => x.compress_stream(output, input),
        }
        .map_err(map_error_code)
    }

    fn flush<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> io::Result<usize> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => x.flush_stream(output),
            MaybeOwnedCCtx::Borrowed(x) => x.flush_stream(output),
        }
        .map_err(map_error_code)
    }

    fn finish<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        _finished_frame: bool,
    ) -> io::Result<usize> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => x.end_stream(output),
            MaybeOwnedCCtx::Borrowed(x) => x.end_stream(output),
        }
        .map_err(map_error_code)
    }

    fn reinit(&mut self) -> io::Result<()> {
        match &mut self.context {
            MaybeOwnedCCtx::Owned(x) => {
                x.reset(zstd_safe::ResetDirective::SessionOnly)
            }
            MaybeOwnedCCtx::Borrowed(x) => {
                x.reset(zstd_safe::ResetDirective::SessionOnly)
            }
        }
        .map_err(map_error_code)?;
        Ok(())
    }
}

enum MaybeOwnedCCtx<'a> {
    Owned(zstd_safe::CCtx<'a>),
    Borrowed(&'a mut zstd_safe::CCtx<'static>),
}

enum MaybeOwnedDCtx<'a> {
    Owned(zstd_safe::DCtx<'a>),
    Borrowed(&'a mut zstd_safe::DCtx<'static>),
}

#[cfg(test)]
mod tests {

    // This requires impl for [u8; N] which is currently behind a feature.
    #[cfg(feature = "arrays")]
    #[test]
    fn test_cycle() {
        use super::{Decoder, Encoder, InBuffer, Operation, OutBuffer};

        let mut encoder = Encoder::new(1).unwrap();
        let mut decoder = Decoder::new().unwrap();

        // Step 1: compress
        let mut input = InBuffer::around(b"AbcdefAbcdefabcdef");

        let mut output = [0u8; 128];
        let mut output = OutBuffer::around(&mut output);

        loop {
            encoder.run(&mut input, &mut output).unwrap();

            if input.pos == input.src.len() {
                break;
            }
        }
        encoder.finish(&mut output, true).unwrap();

        let initial_data = input.src;

        // Step 2: decompress
        let mut input = InBuffer::around(output.as_slice());
        let mut output = [0u8; 128];
        let mut output = OutBuffer::around(&mut output);

        loop {
            decoder.run(&mut input, &mut output).unwrap();

            if input.pos == input.src.len() {
                break;
            }
        }

        assert_eq!(initial_data, output.as_slice());
    }
}
