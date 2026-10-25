use alloc::{Allocator, SliceWrapperMut};
#[cfg(feature = "std")]
use std::io;
#[cfg(feature = "std")]
use std::io::{Error, ErrorKind, Read};

#[cfg(feature = "std")]
pub use alloc_stdlib::StandardAlloc;
use brotli_decompressor::CustomRead;
#[cfg(feature = "std")]
pub use brotli_decompressor::{IntoIoReader, IoReaderWrapper, IoWriterWrapper};

use super::backward_references::BrotliEncoderParams;
use super::combined_alloc::BrotliAlloc;
use super::encode::{
    BrotliEncoderDestroyInstance, BrotliEncoderOperation, BrotliEncoderParameter,
    BrotliEncoderStateStruct,
};
use super::interface;
use crate::enc::combined_alloc::allocate;

#[cfg(feature = "std")]
pub struct CompressorReaderCustomAlloc<R: Read, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>(
    CompressorReaderCustomIo<io::Error, IntoIoReader<R>, BufferType, Alloc>,
);

#[cfg(feature = "std")]
impl<R: Read, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CompressorReaderCustomAlloc<R, BufferType, Alloc>
{
    pub fn new(r: R, buffer: BufferType, alloc: Alloc, q: u32, lgwin: u32) -> Self {
        CompressorReaderCustomAlloc::<R, BufferType, Alloc>(CompressorReaderCustomIo::<
            Error,
            IntoIoReader<R>,
            BufferType,
            Alloc,
        >::new(
            IntoIoReader::<R>(r),
            buffer,
            alloc,
            Error::new(ErrorKind::InvalidData, "Invalid Data"),
            q,
            lgwin,
        ))
    }

    pub fn get_ref(&self) -> &R {
        &self.0.get_ref().0
    }
    pub fn into_inner(self) -> R {
        self.0.into_inner().0
    }
}

#[cfg(feature = "std")]
impl<R: Read, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc> Read
    for CompressorReaderCustomAlloc<R, BufferType, Alloc>
{
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Error> {
        self.0.read(buf)
    }
}

#[cfg(feature = "std")]
pub struct CompressorReader<R: Read>(
    CompressorReaderCustomAlloc<
        R,
        <StandardAlloc as Allocator<u8>>::AllocatedMemory,
        StandardAlloc,
    >,
);

#[cfg(feature = "std")]
impl<R: Read> CompressorReader<R> {
    pub fn new(r: R, buffer_size: usize, q: u32, lgwin: u32) -> Self {
        let mut alloc = StandardAlloc::default();
        let buffer = allocate::<u8, _>(
            &mut alloc,
            if buffer_size == 0 { 4096 } else { buffer_size },
        );
        CompressorReader::<R>(CompressorReaderCustomAlloc::new(r, buffer, alloc, q, lgwin))
    }

    pub fn with_params(r: R, buffer_size: usize, params: &BrotliEncoderParams) -> Self {
        let mut reader = Self::new(r, buffer_size, params.quality as u32, params.lgwin as u32);
        (reader.0).0.state.0.params = params.clone();
        reader
    }

    pub fn get_ref(&self) -> &R {
        self.0.get_ref()
    }
    pub fn into_inner(self) -> R {
        self.0.into_inner()
    }
}

#[cfg(feature = "std")]
impl<R: Read> Read for CompressorReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Error> {
        self.0.read(buf)
    }
}

pub struct CompressorReaderCustomIo<
    ErrType,
    R: CustomRead<ErrType>,
    BufferType: SliceWrapperMut<u8>,
    Alloc: BrotliAlloc,
> {
    input_buffer: BufferType,
    total_out: Option<usize>,
    input_offset: usize,
    input_len: usize,
    input: R,
    input_eof: bool,
    error_if_invalid_data: Option<ErrType>,
    state: StateWrapper<Alloc>,
}
struct StateWrapper<Alloc: BrotliAlloc>(BrotliEncoderStateStruct<Alloc>);

impl<Alloc: BrotliAlloc> Drop for StateWrapper<Alloc> {
    fn drop(&mut self) {
        BrotliEncoderDestroyInstance(&mut self.0);
    }
}

impl<ErrType, R: CustomRead<ErrType>, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CompressorReaderCustomIo<ErrType, R, BufferType, Alloc>
{
    pub fn new(
        r: R,
        buffer: BufferType,
        alloc: Alloc,
        invalid_data_error_type: ErrType,
        q: u32,
        lgwin: u32,
    ) -> Self {
        let mut ret = CompressorReaderCustomIo {
            input_buffer: buffer,
            total_out: Some(0),
            input_offset: 0,
            input_len: 0,
            input_eof: false,
            input: r,
            state: StateWrapper(BrotliEncoderStateStruct::new(alloc)),
            error_if_invalid_data: Some(invalid_data_error_type),
        };
        ret.state
            .0
            .set_parameter(BrotliEncoderParameter::BROTLI_PARAM_QUALITY, q);
        ret.state
            .0
            .set_parameter(BrotliEncoderParameter::BROTLI_PARAM_LGWIN, lgwin);

        ret
    }
    pub fn copy_to_front(&mut self) {
        let avail_in = self.input_len - self.input_offset;
        if self.input_offset == self.input_buffer.slice_mut().len() {
            self.input_offset = 0;
            self.input_len = 0;
        } else if self.input_offset + 256 > self.input_buffer.slice_mut().len()
            && avail_in < self.input_offset
        {
            let (first, second) = self
                .input_buffer
                .slice_mut()
                .split_at_mut(self.input_offset);
            first[0..avail_in].clone_from_slice(&second[0..avail_in]);
            self.input_len -= self.input_offset;
            self.input_offset = 0;
        }
    }
    pub fn into_inner(self) -> R {
        match self {
            CompressorReaderCustomIo {
                input_buffer: _ib,
                total_out: _to,
                input_offset: _io,
                input_len: _len,
                input,
                input_eof: _ieof,
                error_if_invalid_data: _eiid,
                state: _state,
            } => input,
        }
    }
    pub fn get_ref(&self) -> &R {
        &self.input
    }
}
impl<ErrType, R: CustomRead<ErrType>, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CustomRead<ErrType> for CompressorReaderCustomIo<ErrType, R, BufferType, Alloc>
{
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ErrType> {
        let mut nop_callback =
            |_data: &mut interface::PredictionModeContextMap<interface::InputReferenceMut>,
             _cmds: &mut [interface::StaticCommand],
             _mb: interface::InputPair,
             _mfv: &mut Alloc| ();
        let mut output_offset: usize = 0;
        let mut avail_out = buf.len();
        let mut avail_in = self.input_len - self.input_offset;
        while output_offset == 0 {
            if self.input_len < self.input_buffer.slice_mut().len() && !self.input_eof {
                match self
                    .input
                    .read(&mut self.input_buffer.slice_mut()[self.input_len..])
                {
                    Err(e) => return Err(e),
                    Ok(size) => {
                        if size == 0 {
                            self.input_eof = true;
                        } else {
                            self.input_len += size;
                            avail_in = self.input_len - self.input_offset;
                        }
                    }
                }
            }
            let op: BrotliEncoderOperation;
            if avail_in == 0 {
                op = BrotliEncoderOperation::BROTLI_OPERATION_FINISH;
            } else {
                op = BrotliEncoderOperation::BROTLI_OPERATION_PROCESS;
            }
            let ret = self.state.0.compress_stream(
                op,
                &mut avail_in,
                self.input_buffer.slice_mut(),
                &mut self.input_offset,
                &mut avail_out,
                buf,
                &mut output_offset,
                &mut self.total_out,
                &mut nop_callback,
            );
            if avail_in == 0 {
                self.copy_to_front();
            }
            if !ret {
                return Err(self.error_if_invalid_data.take().unwrap());
            }
            if self.state.0.is_finished() {
                break;
            }
        }
        Ok(output_offset)
    }
}
