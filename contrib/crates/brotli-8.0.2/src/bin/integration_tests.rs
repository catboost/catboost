#![cfg(test)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

extern crate brotli_decompressor;
extern crate core;

use core::cmp::min;
use std::io;
#[cfg(feature = "std")]
use std::io::{Read, Write};
use std::time::Duration;
#[cfg(not(feature = "disable-timer"))]
use std::time::SystemTime;

use brotli::BrotliDecompressStream;

use super::alloc_no_stdlib::{Allocator, SliceWrapper, SliceWrapperMut};
use super::brotli::{BrotliResult, BrotliState};
#[cfg(feature = "std")]
use super::brotli::{CompressorReader, CompressorWriter};
#[cfg(feature = "std")]
use super::brotli_decompressor::{Decompressor, DecompressorWriter};
use super::{HeapAllocator, Rebox};

#[cfg(feature = "benchmark")]
extern crate test;
#[cfg(feature = "benchmark")]
use self::test::Bencher;

pub struct Buffer {
    data: Vec<u8>,
    read_offset: usize,
}

#[derive(Debug)]
pub struct UnlimitedBuffer {
    pub data: Vec<u8>,
    read_offset: usize,
}

#[cfg(feature = "disable-timer")]
fn now() -> Duration {
    return Duration::new(0, 0);
}
#[cfg(not(feature = "disable-timer"))]
fn now() -> SystemTime {
    SystemTime::now()
}

#[cfg(not(feature = "disable-timer"))]
fn elapsed(start: SystemTime) -> (Duration, bool) {
    match start.elapsed() {
        Ok(delta) => (delta, false),
        _ => (Duration::new(0, 0), true),
    }
}

#[cfg(feature = "disable-timer")]
fn elapsed(_start: Duration) -> (Duration, bool) {
    return (Duration::new(0, 0), true);
}

fn _write_all<OutputType>(w: &mut OutputType, buf: &[u8]) -> Result<(), io::Error>
where
    OutputType: io::Write,
{
    let mut total_written: usize = 0;
    while total_written < buf.len() {
        match w.write(&buf[total_written..]) {
            Err(e) => match e.kind() {
                io::ErrorKind::Interrupted => continue,
                _ => return Err(e),
            },
            Ok(cur_written) => {
                if cur_written == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Write EOF"));
                }
                total_written += cur_written;
            }
        }
    }
    Ok(())
}
trait Runner {
    fn iter<Fn: FnMut()>(&mut self, cb: &mut Fn);
}

struct Passthrough {}
impl Runner for Passthrough {
    fn iter<Fn: FnMut()>(&mut self, cb: &mut Fn) {
        cb()
    }
}

#[cfg(feature = "benchmark")]
struct BenchmarkPassthrough<'a>(pub &'a mut Bencher);
#[cfg(feature = "benchmark")]
impl<'a> Runner for BenchmarkPassthrough<'a> {
    fn iter<Fn: FnMut()>(&mut self, cb: &mut Fn) {
        self.0.iter(cb)
    }
}
// option_env!("BENCHMARK_MODE").is_some()

fn decompress_internal<InputType, OutputType, Run: Runner>(
    r: &mut InputType,
    mut w: &mut OutputType,
    input_buffer_limit: usize,
    output_buffer_limit: usize,
    runner: &mut Run,
) -> Result<(), io::Error>
where
    InputType: io::Read,
    OutputType: io::Write,
{
    let mut total = Duration::new(0, 0);
    let mut range: usize = 0;
    let mut timing_error: bool = false;
    runner.iter(&mut || {
        range += 1;
        let mut brotli_state = BrotliState::new(
            HeapAllocator::default(),
            HeapAllocator::default(),
            HeapAllocator::default(),
        );
        let mut input = brotli_state.alloc_u8.alloc_cell(input_buffer_limit);
        let mut output = brotli_state.alloc_u8.alloc_cell(output_buffer_limit);
        let mut available_out: usize = output.slice().len();

        // let amount = try!(r.read(&mut buf));
        let mut available_in: usize = 0;
        let mut input_offset: usize = 0;
        let mut output_offset: usize = 0;
        let mut result: BrotliResult = BrotliResult::NeedsMoreInput;
        loop {
            match result {
                BrotliResult::NeedsMoreInput => {
                    input_offset = 0;
                    match r.read(input.slice_mut()) {
                        Err(e) => match e.kind() {
                            io::ErrorKind::Interrupted => continue,
                            _ => panic!("{}", e),
                        },
                        Ok(size) => {
                            if size == 0 {
                                panic!(
                                    "{:?}",
                                    io::Error::new(io::ErrorKind::UnexpectedEof, "Read EOF")
                                );
                            }
                            available_in = size;
                        }
                    }
                }
                BrotliResult::NeedsMoreOutput => {
                    _write_all(&mut w, &output.slice()[..output_offset]).unwrap();
                    output_offset = 0;
                }
                BrotliResult::ResultSuccess => break,
                BrotliResult::ResultFailure => panic!("FAILURE"),
            }
            let mut written: usize = 0;
            let start = now();
            result = BrotliDecompressStream(
                &mut available_in,
                &mut input_offset,
                input.slice(),
                &mut available_out,
                &mut output_offset,
                output.slice_mut(),
                &mut written,
                &mut brotli_state,
            );

            let (delta, err) = elapsed(start);
            if err {
                timing_error = true;
            }
            total += delta;
            if output_offset != 0 {
                _write_all(&mut w, &output.slice()[..output_offset]).unwrap();
                output_offset = 0;
                available_out = output.slice().len()
            }
        }
    });
    if timing_error {
        let _r = super::writeln0(&mut io::stderr(), "Timing error");
    } else {
        let _r = super::writeln_time(
            &mut io::stderr(),
            "Iterations; Time",
            range as u64,
            total.as_secs(),
            total.subsec_nanos(),
        );
    }
    Ok(())
}

impl Buffer {
    pub fn new(buf: &[u8]) -> Buffer {
        let mut ret = Buffer {
            data: Vec::<u8>::new(),
            read_offset: 0,
        };
        ret.data.extend(buf);
        ret
    }
}
impl UnlimitedBuffer {
    pub fn new(buf: &[u8]) -> Self {
        let mut ret = UnlimitedBuffer {
            data: Vec::<u8>::new(),
            read_offset: 0,
        };
        ret.data.extend(buf);
        ret
    }
    pub fn reset_read(&mut self) {
        self.read_offset = 0;
    }
    pub fn data(&self) -> &[u8] {
        &self.data[..]
    }
}
impl io::Read for Buffer {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_offset == self.data.len() {
            self.read_offset = 0;
        }
        let bytes_to_read = min(buf.len(), self.data.len() - self.read_offset);
        if bytes_to_read > 0 {
            buf[0..bytes_to_read]
                .clone_from_slice(&self.data[self.read_offset..self.read_offset + bytes_to_read]);
        }
        self.read_offset += bytes_to_read;
        Ok(bytes_to_read)
    }
}
impl io::Write for Buffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.read_offset == self.data.len() {
            return Ok(buf.len());
        }
        self.data.extend(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl io::Read for UnlimitedBuffer {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_to_read = min(buf.len(), self.data.len() - self.read_offset);
        if bytes_to_read > 0 {
            buf[0..bytes_to_read]
                .clone_from_slice(&self.data[self.read_offset..self.read_offset + bytes_to_read]);
        }
        self.read_offset += bytes_to_read;
        Ok(bytes_to_read)
    }
}

impl io::Write for UnlimitedBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.extend(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[allow(non_snake_case)]
#[test]
fn test_roundtrip_64x() {
    let X = b'X';
    let in_buf: [u8; 64] = [
        X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
        X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X,
        X, X, X, X,
    ];

    let mut input = UnlimitedBuffer::new(&in_buf);
    let mut compressed = UnlimitedBuffer::new(&[]);
    let mut output = UnlimitedBuffer::new(&[]);
    let q: i32 = 9;
    let lgwin: i32 = 16;
    let mut params = super::brotli::enc::BrotliEncoderInitParams();
    params.quality = q;
    params.lgwin = lgwin;
    match super::compress(&mut input, &mut compressed, 65536, &params, &[], 1) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut compressed_in = UnlimitedBuffer::new(&compressed.data[..]);
    match super::decompress(&mut compressed_in, &mut output, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    for i in 0..input.data.len() {
        assert_eq!(output.data[i], input.data[i]);
    }
    assert_eq!(output.data.len(), input.data.len());
    assert_eq!(input.read_offset, in_buf.len());
}

fn roundtrip_helper(in_buf: &[u8], q: i32, lgwin: i32, q9_5: bool) -> usize {
    let mut params = super::brotli::enc::BrotliEncoderInitParams();
    params.quality = q;
    params.q9_5 = q9_5;
    params.lgwin = lgwin;
    params.size_hint = if in_buf.len() > 100000 {
        2048 * 1024
    } else {
        in_buf.len()
    };
    let mut input = UnlimitedBuffer::new(in_buf);
    let mut compressed = UnlimitedBuffer::new(&[]);
    let mut output = UnlimitedBuffer::new(&[]);
    match super::compress(&mut input, &mut compressed, 4096, &params, &[], 1) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut compressed_in = UnlimitedBuffer::new(&compressed.data[..]);
    match super::decompress(&mut compressed_in, &mut output, 4096, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    for i in 0..input.data.len() {
        assert_eq!(output.data[i], input.data[i]);
    }
    assert_eq!(output.data.len(), input.data.len());
    assert_eq!(input.read_offset, in_buf.len());
    compressed.data[..].len()
}

fn total_roundtrip_helper(data: &[u8]) {
    for q in 0..10 {
        roundtrip_helper(data, q, q + 13, false);
    }
    roundtrip_helper(data, 10, 23, true);
}
static RANDOM_THEN_UNICODE: &[u8] = include_bytes!("../../testdata/random_then_unicode");
#[test]
fn test_random_then_unicode_0() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 0, 13, false);
}

#[test]
fn test_random_then_unicode_1() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 1, 14, false);
}

#[test]
fn test_random_then_unicode_2() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 2, 15, false);
}

#[test]
fn test_random_then_unicode_3() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 3, 16, false);
}

#[test]
fn test_random_then_unicode_4() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 4, 17, false);
}

#[test]
fn test_random_then_unicode_5() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 5, 18, false);
}

#[test]
fn test_random_then_unicode_6() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 6, 19, false);
}

#[test]
fn test_random_then_unicode_7() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 7, 20, false);
}

#[test]
fn test_random_then_unicode_8() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 8, 21, false);
}

#[test]
fn test_random_then_unicode_9() {
    roundtrip_helper(RANDOM_THEN_UNICODE, 9, 22, false);
}
#[cfg(feature = "std")]
const random_then_unicode_compressed_size_9_5: usize = 130036;
#[cfg(feature = "std")]
const random_then_unicode_compressed_size_9_5x: usize = 129715;

#[cfg(not(feature = "std"))]
const alice_compressed_size_10: usize = 47490;
#[cfg(not(feature = "std"))]
const alice_compressed_size_11: usize = 46496;

#[cfg(feature = "std")]
#[cfg(not(feature = "float64"))]
const alice_compressed_size_10: usize = 47488;
#[cfg(feature = "std")]
#[cfg(not(feature = "float64"))]
const alice_compressed_size_11: usize = 46493;

#[cfg(feature = "std")]
#[cfg(feature = "float64")]
const alice_compressed_size_10: usize = 47515;
#[cfg(feature = "std")]
#[cfg(feature = "float64")]
const alice_compressed_size_11: usize = 46510;

#[cfg(not(feature = "std"))] // approx log
const random_then_unicode_compressed_size_9_5: usize = 130105;
#[cfg(not(feature = "std"))] // approx log
const random_then_unicode_compressed_size_9_5x: usize = 129873;

#[test]
fn test_random_then_unicode_9_5() {
    let c_size = roundtrip_helper(RANDOM_THEN_UNICODE, 10, 28, true);
    assert_eq!(c_size, random_then_unicode_compressed_size_9_5);
}

#[test]
fn test_random_then_unicode_9x5() {
    let c_size = roundtrip_helper(RANDOM_THEN_UNICODE, 11, 22, true);
    assert_eq!(c_size, random_then_unicode_compressed_size_9_5x);
}

#[test]
fn test_alice29_11() {
    let c_size = roundtrip_helper(include_bytes!("../../testdata/alice29.txt"), 11, 22, false);
    if c_size != 46492 {
        // depends on log2 impl
        assert_eq!(c_size, alice_compressed_size_11);
    }
}

#[test]
fn test_alice29_10() {
    let c_size = roundtrip_helper(include_bytes!("../../testdata/alice29.txt"), 10, 22, false);
    assert_eq!(c_size, alice_compressed_size_10);
}

#[test]
fn test_roundtrip_quickfox_repeated() {
    total_roundtrip_helper(include_bytes!("../../testdata/quickfox_repeated"));
}

#[test]
fn test_roundtrip_alice29() {
    total_roundtrip_helper(include_bytes!("../../testdata/alice29.txt"));
}

#[test]
fn test_roundtrip_as_you_lik() {
    total_roundtrip_helper(include_bytes!("../../testdata/asyoulik.txt"));
}

#[cfg(feature = "std")]
fn reader_helper(mut in_buf: &[u8], q: u32, lgwin: u32) {
    let original_buf = in_buf;
    let mut cmp = [0u8; 259];
    let mut input = UnlimitedBuffer::new(in_buf);
    {
        let renc = CompressorReader::new(&mut input, 255, q, lgwin);
        let mut rdec = Decompressor::new(renc, 257);
        loop {
            match rdec.read(&mut cmp[..]) {
                Ok(size) => {
                    if size == 0 {
                        break;
                    }
                    assert_eq!(cmp[..size], in_buf[..size]);
                    in_buf = &in_buf[size..];
                }
                Err(e) => panic!("Error {:?}", e),
            }
        }
        let inner_item = rdec.into_inner();
        let _inner_inner_item = inner_item.into_inner();
    }
    in_buf = original_buf;
    input = UnlimitedBuffer::new(in_buf);
    let mut r2enc = CompressorReader::new(&mut input, 255, q, lgwin);
    let mut compressed_size = 0usize;
    loop {
        match r2enc.read(&mut cmp[..]) {
            Ok(size) => {
                if size == 0 {
                    break;
                }
                compressed_size += size;
            }
            Err(e) => panic!("Error {:?}", e),
        }
    }
    let pct_ratio = 90usize;
    assert!(compressed_size < original_buf.len() * pct_ratio / 100);
}

/*

#[cfg(feature="std")]
fn simple_reader_helper(mut in_buf: &[u8], q: u32, lgwin: u32) {
    let mut xinput = UnlimitedBuffer::new(&in_buf);
    let xenc;
        let mut buf = HeapAllocator::default().alloc_cell(1024);
    xenc = SimpleReader::new(xinput, buf, 1, 16);
    return;

  let original_buf = in_buf;
    let mut cmp = [0u8; 259];
    let mut buf = HeapAllocator::default().alloc_cell(1024);
  let mut input = UnlimitedBuffer::new(&in_buf);
  {
      let renc = SimpleReader::new(&mut input, buf, q, lgwin).unwrap();
  let mut rdec = Decompressor::new(renc, 257);
  loop {
    match rdec.read(&mut cmp[..]) {
      Ok(size) => {
        if size == 0 {
          break;
        }
        assert_eq!(cmp[..size], in_buf[..size]);
        in_buf = &in_buf[size..];
      }
      Err(e) => panic!("Error {:?}", e),
    }
  }
  }
  in_buf = original_buf;
  input = UnlimitedBuffer::new(&in_buf);
  let mut r2enc = CompressorReader::new(&mut input, 255, q, lgwin);
  let mut compressed_size = 0usize;
  loop {
    match r2enc.read(&mut cmp[..]) {
      Ok(size) => {
        if size == 0 {
          break;
        }
        compressed_size += size;
      }
      Err(e) => panic!("Error {:?}", e),
    }
  }
  let pct_ratio = 90usize;
  assert!(compressed_size < original_buf.len() * pct_ratio / 100);
}
*/
#[cfg(feature = "std")]
#[test]
fn test_reader_64x() {
    reader_helper(include_bytes!("../../testdata/64x"), 9, 20);
}
/*
#[cfg(feature="std")]
#[test]
fn test_simple_64x() {
    bogus_reader_helper(include_bytes!("../../testdata/64x"), 9, 20);
}*/

#[cfg(feature = "std")]
#[test]
fn test_reader_as_you_lik() {
    reader_helper(include_bytes!("../../testdata/asyoulik.txt"), 9, 20);
}
#[cfg(feature = "std")]
#[test]
fn test_reader_quickfox_repeated() {
    reader_helper(include_bytes!("../../testdata/quickfox_repeated"), 9, 20);
}
#[cfg(feature = "std")]
#[test]
fn test_reader_random_then_unicode() {
    reader_helper(include_bytes!("../../testdata/random_then_unicode"), 9, 20);
}

#[cfg(feature = "std")]
#[test]
fn test_reader_alice() {
    reader_helper(include_bytes!("../../testdata/alice29.txt"), 9, 22);
}

#[cfg(feature = "std")]
fn writer_helper(mut in_buf: &[u8], buf_size: usize, q: u32, lgwin: u32, do_flush: bool) {
    let original_buf = in_buf;
    let mut output = UnlimitedBuffer::new(&[]);
    {
        {
            let wdec = DecompressorWriter::new(&mut output, 257);
            {
                let mut wenc = CompressorWriter::new(wdec, 255, q, lgwin);
                while !in_buf.is_empty() {
                    match wenc.write(&in_buf[..min(in_buf.len(), buf_size)]) {
                        Ok(size) => {
                            if size == 0 {
                                break;
                            }
                            in_buf = &in_buf[size..];
                        }
                        Err(e) => panic!("Error {:?}", e),
                    }
                    if do_flush {
                        if let Err(e) = wenc.flush() {
                            panic!("Error {:?}", e);
                        }
                    }
                }
            }
        }
        assert_eq!(output.data.len(), original_buf.len());
        for i in 0..min(original_buf.len(), output.data.len()) {
            assert_eq!(output.data[i], original_buf[i]);
        }
        in_buf = original_buf;
        let mut compressed = UnlimitedBuffer::new(&[]);
        {
            let mut wenc = CompressorWriter::new(&mut compressed, 255, q, lgwin);
            while !in_buf.is_empty() {
                match wenc.write(&in_buf[..min(in_buf.len(), buf_size)]) {
                    Ok(size) => {
                        if size == 0 {
                            break;
                        }
                        in_buf = &in_buf[size..];
                    }
                    Err(e) => panic!("Error {:?}", e),
                }
            }
        }
        let pct_ratio = 90usize;
        assert!(compressed.data.len() < original_buf.len() * pct_ratio / 100);
    }
}

#[cfg(feature = "std")]
fn into_inner_writer_helper(mut in_buf: &[u8], buf_size: usize, q: u32, lgwin: u32) {
    let orig_buf = in_buf;
    let mut compressed = UnlimitedBuffer::new(&[]);
    let mut wenc = CompressorWriter::new(&mut compressed, 255, q, lgwin);
    while !in_buf.is_empty() {
        match wenc.write(&in_buf[..min(in_buf.len(), buf_size)]) {
            Ok(size) => {
                if size == 0 {
                    break;
                }
                in_buf = &in_buf[size..];
            }
            Err(e) => panic!("Error {:?}", e),
        }
    }
    let c2 = wenc.into_inner();

    let pct_ratio = 95usize;
    assert!(!c2.data.is_empty());
    assert!(c2.data.len() < orig_buf.len() * pct_ratio / 100);
    let mut compressed_in = UnlimitedBuffer::new(&c2.data[..]);
    let mut output = UnlimitedBuffer::new(&[]);
    match super::decompress(&mut compressed_in, &mut output, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    for i in 0..orig_buf.len() {
        assert_eq!(output.data[i], orig_buf[i]);
    }
    assert_eq!(output.data.len(), orig_buf.len());
}

#[cfg(feature = "std")]
#[test]
fn test_writer_as_you_lik() {
    writer_helper(
        include_bytes!("../../testdata/asyoulik.txt"),
        17,
        9,
        20,
        false,
    );
}
#[cfg(feature = "std")]
#[test]
fn test_writer_as_you_lik_into_inner() {
    into_inner_writer_helper(include_bytes!("../../testdata/asyoulik.txt"), 256, 4, 16);
}
#[cfg(feature = "std")]
#[test]
fn test_writer_64x() {
    writer_helper(include_bytes!("../../testdata/64x"), 17, 9, 20, false);
}
#[cfg(feature = "std")]
#[test]
fn test_writer_quickfox_repeated() {
    writer_helper(
        include_bytes!("../../testdata/quickfox_repeated"),
        251,
        9,
        20,
        false,
    );
}
#[cfg(feature = "std")]
#[test]
fn test_writer_random_then_unicode() {
    writer_helper(
        include_bytes!("../../testdata/random_then_unicode"),
        277,
        9,
        20,
        false,
    );
}

#[cfg(feature = "std")]
#[test]
fn test_writer_alice() {
    writer_helper(
        include_bytes!("../../testdata/alice29.txt"),
        299,
        9,
        22,
        true,
    );
}

#[test]
fn test_10x_10y() {
    let in_buf: [u8; 12] = [
        0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a,
    ];
    let mut input = Buffer::new(&in_buf);
    let mut output = Buffer::new(&[]);
    output.read_offset = 20;
    match super::decompress(&mut input, &mut output, 65536, Rebox::default()) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut i: usize = 0;
    while i < 10 {
        assert_eq!(output.data[i], b'X');
        assert_eq!(output.data[i + 10], b'Y');
        i += 1;
    }
    assert_eq!(output.data.len(), 20);
    assert_eq!(input.read_offset, in_buf.len());
}

#[test]
fn test_10x_10y_one_out_byte() {
    let in_buf: [u8; 12] = [
        0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a,
    ];
    let mut input = Buffer::new(&in_buf);
    let mut output = Buffer::new(&[]);
    output.read_offset = 20;
    match decompress_internal(&mut input, &mut output, 12, 1, &mut Passthrough {}) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut i: usize = 0;
    while i < 10 {
        assert_eq!(output.data[i], b'X');
        assert_eq!(output.data[i + 10], b'Y');
        i += 1;
    }
    assert_eq!(output.data.len(), 20);
    assert_eq!(input.read_offset, in_buf.len());
}

#[test]
fn test_10x_10y_byte_by_byte() {
    let in_buf: [u8; 12] = [
        0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a,
    ];
    let mut input = Buffer::new(&in_buf);
    let mut output = Buffer::new(&[]);
    output.read_offset = 20;
    match decompress_internal(&mut input, &mut output, 1, 1, &mut Passthrough {}) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut i: usize = 0;
    while i < 10 {
        assert_eq!(output.data[i], b'X');
        assert_eq!(output.data[i + 10], b'Y');
        i += 1;
    }
    assert_eq!(output.data.len(), 20);
    assert_eq!(input.read_offset, in_buf.len());
}

fn assert_decompressed_input_matches_output(
    input_slice: &[u8],
    output_slice: &[u8],
    input_buffer_size: usize,
    output_buffer_size: usize,
) {
    let mut input = Buffer::new(input_slice);
    let mut output = Buffer::new(&[]);
    output.read_offset = output_slice.len();
    if input_buffer_size == output_buffer_size {
        match super::decompress(&mut input, &mut output, input_buffer_size, Rebox::default()) {
            Ok(_) => {}
            Err(e) => panic!("Error {:?}", e),
        }
    } else {
        match decompress_internal(
            &mut input,
            &mut output,
            input_buffer_size,
            output_buffer_size,
            &mut Passthrough {},
        ) {
            Ok(_) => {}
            Err(e) => panic!("Error {:?}", e),
        }
    }
    assert_eq!(output.data.len(), output_slice.len());
    assert_eq!(output.data, output_slice)
}

#[cfg(feature = "benchmark")]
fn benchmark_decompressed_input(
    input_slice: &[u8],
    output_slice: &[u8],
    input_buffer_size: usize,
    output_buffer_size: usize,
    bench: &mut Bencher,
) {
    let mut input = Buffer::new(input_slice);
    let mut output = Buffer::new(&[]);
    output.read_offset = output_slice.len();
    match decompress_internal(
        &mut input,
        &mut output,
        input_buffer_size,
        output_buffer_size,
        &mut BenchmarkPassthrough(bench),
    ) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    assert_eq!(output.data.len(), output_slice.len());
    assert_eq!(output.data, output_slice)
}
pub struct LimitedBuffer<'a> {
    pub data: &'a mut [u8],
    pub write_offset: usize,
    pub read_offset: usize,
}

impl<'a> LimitedBuffer<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        LimitedBuffer {
            data: buf,
            write_offset: 0,
            read_offset: 0,
        }
    }
}
impl<'a> LimitedBuffer<'a> {
    fn reset(&mut self) {
        self.write_offset = 0;
        self.read_offset = 0;
        self.data.split_at_mut(32).0.clone_from_slice(&[0u8; 32]); // clear the first 256 bits
    }
    fn reset_read(&mut self) {
        self.read_offset = 0;
    }
    fn written(&self) -> &[u8] {
        &self.data[..self.write_offset]
    }
}
impl<'a> io::Read for LimitedBuffer<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_to_read = min(buf.len(), self.data.len() - self.read_offset);
        if bytes_to_read > 0 {
            buf[0..bytes_to_read]
                .clone_from_slice(&self.data[self.read_offset..self.read_offset + bytes_to_read]);
        }
        self.read_offset += bytes_to_read;
        Ok(bytes_to_read)
    }
}

impl<'a> io::Write for LimitedBuffer<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_to_write = min(buf.len(), self.data.len() - self.write_offset);
        if bytes_to_write > 0 {
            self.data[self.write_offset..self.write_offset + bytes_to_write]
                .clone_from_slice(&buf[..bytes_to_write]);
        } else {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "OutOfBufferSpace"));
        }
        self.write_offset += bytes_to_write;
        Ok(bytes_to_write)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn benchmark_helper<Run: Runner>(
    input_slice: &[u8],
    compress_buffer_size: usize,
    decompress_buffer_size: usize,
    bench_compress: bool,
    bench_decompress: bool,
    bench: &mut Run,
    quality: i32,
    q9_5: bool,
) {
    let mut params = super::brotli::enc::BrotliEncoderInitParams();
    params.quality = quality;
    params.q9_5 = q9_5;
    params.large_window = true;
    let mut input = UnlimitedBuffer::new(input_slice);
    let mut compressed_array = vec![0; input_slice.len() * 100 / 99];
    let mut rt_array = vec![0; input_slice.len() + 1];
    let mut compressed = LimitedBuffer::new(&mut compressed_array[..]);
    let mut rt = LimitedBuffer::new(&mut rt_array[..]);
    if !bench_compress {
        match super::compress(
            &mut input,
            &mut compressed,
            compress_buffer_size,
            &params,
            &[],
            1,
        ) {
            Ok(_) => {}
            Err(e) => panic!("Error {:?}", e),
        }
    }
    bench.iter(&mut || {
        input.reset_read();
        if bench_compress {
            compressed.reset();
            match super::compress(
                &mut input,
                &mut compressed,
                compress_buffer_size,
                &params,
                &[],
                1,
            ) {
                Ok(_) => {}
                Err(e) => panic!("Error {:?}", e),
            }
        }
        if bench_decompress {
            compressed.reset_read();
            rt.reset();
            match super::decompress(
                &mut compressed,
                &mut rt,
                decompress_buffer_size,
                Rebox::default(),
            ) {
                Ok(_) => {}
                Err(e) => panic!("Error {:?}", e),
            }
        }
    });
    if !bench_decompress {
        compressed.reset_read();
        rt.reset();
        match super::decompress(
            &mut compressed,
            &mut rt,
            decompress_buffer_size,
            Rebox::default(),
        ) {
            Ok(_) => {}
            Err(e) => panic!("Error {:?}", e),
        }
    }
    assert_eq!(rt.write_offset, input_slice.len());
    assert_eq!(&rt.data[..input_slice.len()], input_slice);
}

fn expand_test_data(size: usize) -> Vec<u8> {
    let mut ret = vec![0u8; size];
    let original_data = include_bytes!("../../testdata/random_then_unicode");
    let mut count = 0;
    let mut iter = 0usize;
    while count < size {
        let to_copy = min(size - count, original_data.len());
        let target = &mut ret[count..(count + to_copy)];
        for (dst, src) in target.iter_mut().zip(original_data[..to_copy].iter()) {
            *dst = src.wrapping_add(iter as u8);
        }
        count += to_copy;
        iter += 1;
    }
    ret
}

#[test]
fn test_1024k() {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        true,
        true,
        &mut Passthrough {},
        2,
        false,
    );
}

static UKKONOOA: &[u8] = include_bytes!("../../testdata/ukkonooa");
#[test]
fn test_ukkonooa() {
    let td = UKKONOOA;
    benchmark_helper(td, 65536, 65536, true, true, &mut Passthrough {}, 11, false);
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_decode_q9_5_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        false,
        true,
        &mut BenchmarkPassthrough(bench),
        11,
        true,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_decode_q11_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        false,
        true,
        &mut BenchmarkPassthrough(bench),
        11,
        false,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_decode_q5_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        false,
        true,
        &mut BenchmarkPassthrough(bench),
        5,
        false,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_rt_q9_5_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        true,
        true,
        &mut BenchmarkPassthrough(bench),
        11,
        true,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_rt_q11_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        true,
        true,
        &mut BenchmarkPassthrough(bench),
        11,
        false,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_rt_q9_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        true,
        true,
        &mut BenchmarkPassthrough(bench),
        9,
        false,
    );
}

#[cfg(feature = "benchmark")]
#[bench]
fn bench_e2e_rt_q5_1024k(bench: &mut Bencher) {
    let td = expand_test_data(1024 * 1024);
    benchmark_helper(
        &td[..],
        65536,
        65536,
        true,
        true,
        &mut BenchmarkPassthrough(bench),
        5,
        false,
    );
}

#[test]
fn test_64x() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/64x.compressed"),
        include_bytes!("../../testdata/64x"),
        3,
        3,
    );
}

#[test]
fn test_as_you_like_it() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/asyoulik.txt.compressed"),
        include_bytes!("../../testdata/asyoulik.txt"),
        65536,
        65536,
    );
}

#[test]
#[should_panic]
fn test_negative_hypothesis() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/64x"),
        include_bytes!("../../testdata/64x"),
        3,
        3,
    );
}
static ALICE29_BR: &[u8] = include_bytes!("../../testdata/alice29.txt.compressed");
static ALICE29: &[u8] = include_bytes!("../../testdata/alice29.txt");
#[test]
fn test_alice29() {
    assert_decompressed_input_matches_output(ALICE29_BR, ALICE29, 65536, 65536);
}

#[cfg(feature = "benchmark")]
#[bench]
fn benchmark_alice29(bench: &mut Bencher) {
    benchmark_decompressed_input(ALICE29_BR, ALICE29, 65536, 65536, bench);
}

#[test]
fn test_alice1() {
    assert_decompressed_input_matches_output(ALICE29_BR, ALICE29, 1, 65536);
}

#[test]
fn test_backward65536() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/backward65536.compressed"),
        include_bytes!("../../testdata/backward65536"),
        65536,
        65536,
    );
}

#[test]
fn test_compressed_file() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/compressed_file.compressed"),
        include_bytes!("../../testdata/compressed_file"),
        65536,
        65536,
    );
}

#[test]
fn test_compressed_repeated() {
    assert_decompressed_input_matches_output(
        include_bytes!(
            "../../testdata/compressed_repeated.\
                                                           compressed"
        ),
        include_bytes!("../../testdata/compressed_repeated"),
        65536,
        65536,
    );
}

#[test]
fn test_empty() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty0() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.00"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty1() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.01"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty2() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.02"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty3() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.03"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty4() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.04"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty5() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.05"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty6() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.06"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty7() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.07"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty8() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.08"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty9() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.09"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty10() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.10"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty11() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.11"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty12() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.12"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty13() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.13"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty14() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.14"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty15() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.15"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty16() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.16"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty17() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.17"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}
#[test]
fn test_empty18() {
    assert_decompressed_input_matches_output(
        include_bytes!("../../testdata/empty.compressed.18"),
        include_bytes!("../../testdata/empty"),
        65536,
        65536,
    );
}

pub struct SoonErrorReader(&'static [u8], bool);
impl io::Read for SoonErrorReader {
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        let first = self.1;
        self.1 = false;
        if first {
            let len = min(self.0.len(), data.len());
            data[..len].clone_from_slice(&self.0[..len]);
            return Ok(len);
        }
        Err(io::Error::new(io::ErrorKind::PermissionDenied, "err"))
    }
}
#[test]
fn test_error_returned() {
    static onetwothreefourfive: [u8; 5] = [1, 2, 3, 4, 5];
    let params = super::brotli::enc::BrotliEncoderParams::default();
    let mut erroring = SoonErrorReader(&onetwothreefourfive[..], true);
    let mut br = UnlimitedBuffer::new(&[]);
    let dict = &[];
    let err = super::compress(&mut erroring, &mut br, 4096, &params, dict, 1);
    if let Ok(_x) = err {
        panic!("Should have errored {:?}\n", err);
    }
    let mut output = UnlimitedBuffer::new(&[]);
    let mut br_copy = Buffer::new(&br.data[..]);
    assert_eq!(br.data.len(), 9);
    match decompress_internal(&mut br_copy, &mut output, 1, 1, &mut Passthrough {}) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    assert_eq!(output.data, &onetwothreefourfive[..]);
}
