#[cfg(feature = "validation")]
use core;
use std::io::{self, Error, ErrorKind, Read, Write};

use alloc_no_stdlib::{Allocator, SliceWrapper};
use brotli::enc::BrotliEncoderParams;
use brotli::{CustomWrite, DecompressorWriterCustomIo};
#[cfg(feature = "validation")]
use sha2::{Digest, Sha256};

use super::{HeapAllocator, IoWriterWrapper, Rebox};
#[cfg(feature = "validation")]
type Checksum = Sha256;

struct Tee<OutputA: Write, OutputB: Write>(OutputA, OutputB);
impl<OutputA: Write, OutputB: Write> Write for Tee<OutputA, OutputB> {
    fn write(&mut self, data: &[u8]) -> Result<usize, io::Error> {
        match self.0.write(data) {
            Err(err) => Err(err),
            Ok(size) => match self.1.write_all(&data[..size]) {
                Ok(_) => Ok(size),
                Err(err) => Err(err),
            },
        }
    }
    fn flush(&mut self) -> Result<(), io::Error> {
        match self.0.flush() {
            Err(err) => Err(err),
            Ok(_) => loop {
                match self.1.flush() {
                    Err(e) => match e.kind() {
                        ErrorKind::Interrupted => continue,
                        _ => return Err(e),
                    },
                    Ok(e) => return Ok(e),
                }
            },
        }
    }
}

struct DecompressAndValidate<'a, OutputType: Write + 'a>(
    DecompressorWriterCustomIo<
        io::Error,
        IoWriterWrapper<'a, OutputType>,
        Rebox<u8>, // buffer type
        HeapAllocator,
        HeapAllocator,
        HeapAllocator,
    >,
);

impl<'a, OutputType: Write> Write for DecompressAndValidate<'a, OutputType> {
    fn write(&mut self, data: &[u8]) -> Result<usize, io::Error> {
        self.0.write(data)
    }
    fn flush(&mut self) -> Result<(), io::Error> {
        self.0.flush()
    }
}

#[cfg(not(feature = "validation"))]
fn make_sha_writer() -> io::Sink {
    io::sink()
}
#[cfg(not(feature = "validation"))]
fn make_sha_reader<InputType: Read>(r: &mut InputType) -> &mut InputType {
    r
}

#[cfg(not(feature = "validation"))]
fn sha_ok<InputType: Read>(_writer: &mut io::Sink, _reader: &mut InputType) -> bool {
    false
}

#[cfg(feature = "validation")]
struct ShaReader<'a, InputType: Read + 'a> {
    reader: &'a mut InputType,
    checksum: Checksum,
}
#[cfg(feature = "validation")]
impl<'a, InputType: Read + 'a> Read for ShaReader<'a, InputType> {
    fn read(&mut self, data: &mut [u8]) -> Result<usize, io::Error> {
        match self.reader.read(data) {
            Err(e) => Err(e),
            Ok(size) => {
                self.checksum.update(&data[..size]);
                Ok(size)
            }
        }
    }
}
#[cfg(feature = "validation")]
fn make_sha_reader<InputType: Read>(r: &mut InputType) -> ShaReader<InputType> {
    ShaReader {
        reader: r,
        checksum: Checksum::default(),
    }
}

#[cfg(feature = "validation")]
fn sha_ok<InputType: Read>(writer: &mut ShaWriter, reader: &mut ShaReader<InputType>) -> bool {
    core::mem::replace(&mut writer.0, Checksum::default()).finalize()
        == core::mem::replace(&mut reader.checksum, Checksum::default()).finalize()
}
#[cfg(feature = "validation")]
#[derive(Default)]
struct ShaWriter(Checksum);
#[cfg(feature = "validation")]
impl Write for ShaWriter {
    fn write(&mut self, data: &[u8]) -> Result<usize, io::Error> {
        self.0.update(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> Result<(), io::Error> {
        Ok(())
    }
}
#[cfg(feature = "validation")]
fn make_sha_writer() -> ShaWriter {
    ShaWriter::default()
}
#[cfg(feature = "validation")]
const VALIDATION_FAILED: &'static str = "Validation failed";
#[cfg(not(feature = "validation"))]
const VALIDATION_FAILED: &str =
    "Validation module not enabled: build with cargo build --features=validation";

pub fn compress_validate<InputType: Read, OutputType: Write>(
    r: &mut InputType,
    w: &mut OutputType,
    buffer_size: usize,
    params: &BrotliEncoderParams,
    custom_dictionary: Rebox<u8>,
    num_threads: usize,
) -> Result<(), io::Error> {
    let mut m8 = HeapAllocator::default();
    let buffer = m8.alloc_cell(buffer_size);
    // FIXME: could reuse the dictionary to seed the compressor, but that violates the abstraction right now
    // also dictionaries are not very popular since they are mostly an internal concept, given their deprecation in
    // the standard brotli spec
    let mut dict = Vec::<u8>::new();

    dict.extend_from_slice(custom_dictionary.slice());
    let mut sha_writer = make_sha_writer();
    let mut sha_reader = make_sha_reader(r);
    let ret;
    {
        let validate_writer =
            DecompressAndValidate(DecompressorWriterCustomIo::new_with_custom_dictionary(
                IoWriterWrapper(&mut sha_writer),
                buffer,
                m8,
                HeapAllocator::default(),
                HeapAllocator::default(),
                custom_dictionary,
                Error::new(ErrorKind::InvalidData, "Invalid Data"),
            ));
        let mut overarching_writer = Tee(validate_writer, w);
        ret = super::compress(
            &mut sha_reader,
            &mut overarching_writer,
            buffer_size,
            params,
            &dict[..],
            num_threads,
        );
    }
    match ret {
        Ok(_ret) => {
            if sha_ok(&mut sha_writer, &mut sha_reader) {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::InvalidData, VALIDATION_FAILED))
            }
        }
        Err(e) => Err(e),
    }
}
