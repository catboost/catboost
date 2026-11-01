#![cfg(test)]
extern crate core;
use std::io;
#[cfg(feature="std")]
use std::io::{Read,Write};
use core::cmp;
use super::brotli_decompressor::BrotliResult;
use super::brotli_decompressor::BrotliDecompressStream;
#[cfg(feature="std")]
use super::brotli_decompressor::{Decompressor, DecompressorWriter};
use super::brotli_decompressor::BrotliState;
use super::brotli_decompressor::HuffmanCode;
use super::HeapAllocator;

#[allow(unused_imports)]
use super::alloc_no_stdlib::{Allocator, SliceWrapper, SliceWrapperMut};
use std::time::Duration;
#[cfg(not(feature="disable-timer"))]
use std::time::SystemTime;

struct Buffer {
  data: Vec<u8>,
  read_offset: usize,
}
#[cfg(feature="std")]
struct UnlimitedBuffer {
  data: Vec<u8>,
  read_offset: usize,
}

#[cfg(feature="std")]
impl UnlimitedBuffer {
  pub fn new(buf: &[u8]) -> Self {
    let mut ret = UnlimitedBuffer {
      data: Vec::<u8>::new(),
      read_offset: 0,
    };
    ret.data.extend(buf);
    return ret;
  }
}

#[cfg(feature="std")]
impl io::Read for UnlimitedBuffer {
  fn read(self: &mut Self, buf: &mut [u8]) -> io::Result<usize> {
    let bytes_to_read = cmp::min(buf.len(), self.data.len() - self.read_offset);
    if bytes_to_read > 0 {
      buf[0..bytes_to_read].clone_from_slice(&self.data[self.read_offset..
                                              self.read_offset + bytes_to_read]);
    }
    self.read_offset += bytes_to_read;
    return Ok(bytes_to_read);
  }
}

#[cfg(feature="std")]
impl io::Write for UnlimitedBuffer {
  fn write(self: &mut Self, buf: &[u8]) -> io::Result<usize> {
    self.data.extend(buf);
    return Ok(buf.len());
  }
  fn flush(self: &mut Self) -> io::Result<()> {
    return Ok(());
  }
}


#[cfg(feature="disable-timer")]
fn now() -> Duration {
  return Duration::new(0, 0);
}
#[cfg(not(feature="disable-timer"))]
fn now() -> SystemTime {
  return SystemTime::now();
}

#[cfg(not(feature="disable-timer"))]
fn elapsed(start: SystemTime) -> (Duration, bool) {
  match start.elapsed() {
    Ok(delta) => return (delta, false),
    _ => return (Duration::new(0, 0), true),
  }
}

#[cfg(feature="disable-timer")]
fn elapsed(_start: Duration) -> (Duration, bool) {
  return (Duration::new(0, 0), true);
}


fn _write_all<OutputType>(w: &mut OutputType, buf: &[u8]) -> Result<(), io::Error>
  where OutputType: io::Write
{
  let mut total_written: usize = 0;
  while total_written < buf.len() {
    match w.write(&buf[total_written..]) {
      Err(e) => {
        match e.kind() {
          io::ErrorKind::Interrupted => continue,
          _ => return Err(e),
        }
      }
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


#[cfg(feature="benchmark")]
const NUM_BENCHMARK_ITERATIONS: usize = 1000;
#[cfg(not(feature="benchmark"))]
const NUM_BENCHMARK_ITERATIONS: usize = 2;

// option_env!("BENCHMARK_MODE").is_some()

pub fn decompress_internal<InputType, OutputType>(r: &mut InputType,
                                                  mut w: &mut OutputType,
                                                  input_buffer_limit: usize,
                                                  output_buffer_limit: usize,
                                                  benchmark_mode: bool)
                                                  -> Result<(), io::Error>
  where InputType: io::Read,
        OutputType: io::Write
{
  let mut total = Duration::new(0, 0);
  let range: usize;
  let mut timing_error: bool = false;
  if benchmark_mode {
    range = NUM_BENCHMARK_ITERATIONS;
  } else {
    range = 1;
  }
  for _i in 0..range {
    let mut brotli_state =
      BrotliState::new(HeapAllocator::<u8> { default_value: 0 },
                       HeapAllocator::<u32> { default_value: 0 },
                       HeapAllocator::<HuffmanCode> { default_value: HuffmanCode::default() });
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
            Err(e) => {
              match e.kind() {
                io::ErrorKind::Interrupted => continue,
                _ => return Err(e),
              }
            }
            Ok(size) => {
              if size == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Read EOF"));
              }
              available_in = size;
            }
          }
        }
        BrotliResult::NeedsMoreOutput => {
            if let Err(e) = _write_all(&mut w, &output.slice()[..output_offset]) {
                return Err(e)
            }
            output_offset = 0;
        }
        BrotliResult::ResultSuccess => break,
        BrotliResult::ResultFailure => panic!("FAILURE"),
      }
      let mut written: usize = 0;
      let start = now();
      result = BrotliDecompressStream(&mut available_in,
                                      &mut input_offset,
                                      &input.slice(),
                                      &mut available_out,
                                      &mut output_offset,
                                      &mut output.slice_mut(),
                                      &mut written,
                                      &mut brotli_state);

      let (delta, err) = elapsed(start);
      if err {
        timing_error = true;
      }
      total = total + delta;
      if output_offset != 0 {
        if let Err(e) = _write_all(&mut w, &output.slice()[..output_offset]) {
          return Err(e)
        }
        output_offset = 0;
        available_out = output.slice().len()
      }
    }
  }
  if timing_error {
    let _r = super::writeln0(&mut io::stderr(), "Timing error");
  } else {
    let _r = super::writeln_time(&mut io::stderr(),
                                 "Iterations; Time",
                                 range as u64,
                                 total.as_secs(),
                                 total.subsec_nanos());
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
    return ret;
  }
}
impl io::Read for Buffer {
  fn read(self: &mut Self, buf: &mut [u8]) -> io::Result<usize> {
    if self.read_offset == self.data.len() {
      self.read_offset = 0;
    }
    let bytes_to_read = cmp::min(buf.len(), self.data.len() - self.read_offset);
    if bytes_to_read > 0 {
      buf[0..bytes_to_read]
        .clone_from_slice(&self.data[self.read_offset..self.read_offset + bytes_to_read]);
    }
    self.read_offset += bytes_to_read;
    return Ok(bytes_to_read);
  }
}
impl io::Write for Buffer {
  fn write(self: &mut Self, buf: &[u8]) -> io::Result<usize> {
    if self.read_offset == self.data.len() {
      return Ok(buf.len());
    }
    self.data.extend(buf);
    return Ok(buf.len());
  }
  fn flush(self: &mut Self) -> io::Result<()> {
    return Ok(());
  }
}
#[test]
fn test_10x_10y() {
  let in_buf: [u8; 12] = [0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a];
  let mut input = Buffer::new(&in_buf);
  let mut output = Buffer::new(&[]);
  output.read_offset = 20;
  match super::decompress(&mut input, &mut output, 65536, Vec::new()) {
    Ok(_) => {}
    Err(e) => panic!("Error {:?}", e),
  }
  let mut i: usize = 0;
  while i < 10 {
    assert_eq!(output.data[i], 'X' as u8);
    assert_eq!(output.data[i + 10], 'Y' as u8);
    i += 1;
  }
  assert_eq!(output.data.len(), 20);
  assert_eq!(input.read_offset, in_buf.len());
}

#[test]
fn test_10x_10y_one_out_byte() {
  let in_buf: [u8; 12] = [0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a];
  let mut input = Buffer::new(&in_buf);
  let mut output = Buffer::new(&[]);
  output.read_offset = 20;
  match decompress_internal(&mut input, &mut output, 12, 1, false) {
    Ok(_) => {}
    Err(e) => panic!("Error {:?}", e),
  }
  let mut i: usize = 0;
  while i < 10 {
    assert_eq!(output.data[i], 'X' as u8);
    assert_eq!(output.data[i + 10], 'Y' as u8);
    i += 1;
  }
  assert_eq!(output.data.len(), 20);
  assert_eq!(input.read_offset, in_buf.len());
}
#[cfg(feature="std")]
fn reader_helper(in_buf: &[u8], mut desired_buf: &[u8], bufsize : usize) {
  let mut cmp = [0u8; 178];
  let mut input = UnlimitedBuffer::new(&in_buf);
  {
  let mut rdec = Decompressor::new(&mut input, bufsize);
  loop {
    match rdec.read(&mut cmp[..]) {
      Ok(size) => {
        if size == 0 {
          break;
        }
        assert_eq!(cmp[..size], desired_buf[..size]);
        desired_buf = &desired_buf[size..];
      }
      Err(e) => panic!("Error {:?}", e),
    }
  }
  }
  assert_eq!(desired_buf.len(), 0);
}

#[test]
#[cfg(feature="std")]
fn test_reader_64x() {
  reader_helper(include_bytes!("../../testdata/64x.compressed"),
                                           include_bytes!("../../testdata/64x"), 181)

}

#[test]
#[cfg(feature="std")]
fn test_streaming_leftover_buffer() {
  reader_helper(include_bytes!("../../testdata/reducetostream.map.compressed"),
                                           include_bytes!("../../testdata/reducetostream.map"), 8192)
}

#[test]
#[cfg(feature="std")]
fn test_reader_uni() {
  reader_helper(include_bytes!("../../testdata/random_then_unicode.compressed"),
                                           include_bytes!("../../testdata/random_then_unicode"), 121)

}


#[cfg(feature="std")]
fn writer_helper(mut in_buf: &[u8], desired_out_buf: &[u8], buf_size: usize) {
  let output = UnlimitedBuffer::new(&[]);

  {let mut wdec = DecompressorWriter::new(output, 517);
  while in_buf.len() > 0 {
    match wdec.write(&in_buf[..cmp::min(in_buf.len(), buf_size)]) {
      Ok(size) => {
        if size == 0 {
          break;
        }
        in_buf = &in_buf[size..];
      }
      Err(e) => panic!("Error {:?}", e),
    }
  }
   let ub = match wdec.into_inner() {
     Ok(w) => w,
     Err(_) => panic!("error with into_inner"),
   };
     assert_eq!(ub.data.len(), desired_out_buf.len());
  for i in 0..cmp::min(desired_out_buf.len(), ub.data.len()) {
    assert_eq!(ub.data[i], desired_out_buf[i]);
  }

  }
}

#[cfg(feature="std")]
fn writer_early_out_helper(in_buf: &[u8], desired_out_buf: &[u8], buf_size: usize, ibuf: usize) {
  let output = UnlimitedBuffer::new(&[]);

  {let mut wdec = DecompressorWriter::new(output, ibuf);
  if in_buf.len() > 0 {
    match wdec.write(&in_buf[..cmp::min(in_buf.len(), buf_size)]) {
      Ok(_size) => {
      }
      Err(e) => panic!("Error {:?}", e),
    }
  }
   match wdec.into_inner() {
     Err(ub) => {
       assert!(ub.data.len() != 0);
       for i in 0..cmp::min(desired_out_buf.len(), ub.data.len()) {
         assert_eq!(ub.data[i], desired_out_buf[i]);
       }
     },
     Ok(_) => panic!("unreachable"),
   }
  }
}

#[test]
#[cfg(feature="std")]
fn test_writer_64x() {
  writer_helper(include_bytes!("../../testdata/64x.compressed"),
                                           include_bytes!("../../testdata/64x"), 1)

}



#[test]
#[cfg(feature="std")]
fn test_writer_mapsdatazrh() {
  writer_helper(include_bytes!("../../testdata/mapsdatazrh.compressed"),
                include_bytes!("../../testdata/mapsdatazrh"), 512)
    
}

#[test]
#[cfg(feature="std")]
fn test_writer_mapsdatazrh_truncated() {
  writer_early_out_helper(include_bytes!("../../testdata/mapsdatazrh.compressed"),
                                           include_bytes!("../../testdata/mapsdatazrh"), 65536, 65536)

}


#[test]
fn test_10x_10y_byte_by_byte() {
  let in_buf: [u8; 12] = [0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a];
  let mut input = Buffer::new(&in_buf);
  let mut output = Buffer::new(&[]);
  output.read_offset = 20;
  match decompress_internal(&mut input, &mut output, 1, 1, false) {
    Ok(_) => {}
    Err(e) => panic!("Error {:?}", e),
  }
  let mut i: usize = 0;
  while i < 10 {
    assert_eq!(output.data[i], 'X' as u8);
    assert_eq!(output.data[i + 10], 'Y' as u8);
    i += 1;
  }
  assert_eq!(output.data.len(), 20);
  assert_eq!(input.read_offset, in_buf.len());
}


fn assert_decompressed_input_matches_output(input_slice: &[u8],
                                            output_slice: &[u8],
                                            input_buffer_size: usize,
                                            output_buffer_size: usize) {
  let mut input = Buffer::new(input_slice);
  let mut output = Buffer::new(&[]);
  output.read_offset = output_slice.len();
  if input_buffer_size == output_buffer_size {
    match super::decompress(&mut input, &mut output, input_buffer_size, Vec::new()) {
      Ok(_) => {}
      Err(e) => panic!("Error {:?}", e),
    }
  } else {
    match decompress_internal(&mut input,
                              &mut output,
                              input_buffer_size,
                              output_buffer_size,
                              false) {
      Ok(_) => {}
      Err(e) => panic!("Error {:?}", e),
    }
  }
  assert_eq!(output.data.len(), output_slice.len());
  assert_eq!(output.data, output_slice)
}

fn assert_huge_file_input_matches_output(input_slice: &[u8],
                                         output_prefix: &[u8],
                                         output_postfix: &[u8],
                                         size: usize,
                                         rep_gap: usize,
                                         input_buffer_size: usize,
                                         output_buffer_size: usize) {
  let mut input = Buffer::new(input_slice);
  let mut output = Buffer::new(&[]);
  output.read_offset = size;
  if input_buffer_size == output_buffer_size {
    match super::decompress(&mut input, &mut output, input_buffer_size, Vec::new()) {
      Ok(_) => {}
      Err(e) => panic!("Error {:?}", e),
    }
  } else {
    match decompress_internal(&mut input,
                              &mut output,
                              input_buffer_size,
                              output_buffer_size,
                              false) {
      Ok(_) => {}
      Err(e) => panic!("Error {:?}", e),
    }
  }
  assert_eq!(output.data.len(), size);
  assert_eq!(output.data.split_at(output_prefix.len()).0, output_prefix);
  assert_eq!(output.data.split_at(output.data.len() - output_postfix.len()).1, output_postfix);
  assert_eq!(output.data.split_at(output_prefix.len() + rep_gap).1.split_at(output_prefix.len()).0, output_prefix);
  let mut zero_count: usize = 0;
  for item in output.data {
      if item == 0 {
          zero_count += 1;
      }
  }
  let mut nulls_in_input: usize = 0;
  for item in output_prefix.iter().chain(output_prefix.iter().chain(output_postfix.iter())) {
      if *item == 0 {
          nulls_in_input += 1;
      }
  }
  assert_eq!(zero_count - nulls_in_input, size - output_prefix.len() * 2 - output_postfix.len());
}

fn benchmark_decompressed_input(input_slice: &[u8],
                                output_slice: &[u8],
                                input_buffer_size: usize,
                                output_buffer_size: usize) {
  let mut input = Buffer::new(input_slice);
  let mut output = Buffer::new(&[]);
  output.read_offset = output_slice.len();
  match decompress_internal(&mut input,
                            &mut output,
                            input_buffer_size,
                            output_buffer_size,
                            true) {
    Ok(_) => {}
    Err(e) => panic!("Error {:?}", e),
  }
  assert_eq!(output.data.len(), output_slice.len());
  assert_eq!(output.data, output_slice)
}

#[test]
fn test_64x() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/64x.compressed"),
                                           include_bytes!("../../testdata/64x"),
                                           3,
                                           3);
}

#[test]
fn test_random1024() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/random1024.br"),
                                           include_bytes!("../../testdata/random1024"),
                                           65536,
                                           65536);
}
#[test]
fn test_random1024small_buffer() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/random1024.br"),
                                           include_bytes!("../../testdata/random1024"),
                                           1024,
                                           1024);
}

#[test]
fn test_as_you_like_it() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/asyoulik.txt.compressed"),
                                           include_bytes!("../../testdata/asyoulik.txt"),
                                           65536,
                                           65536);
}


#[test]
#[should_panic]
fn test_negative_hypothesis() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/64x"),
                                           include_bytes!("../../testdata/64x"),
                                           3,
                                           3);
}
static ALICE29_BR: &'static [u8] = include_bytes!("../../testdata/alice29.txt.compressed");
static ALICE29: &'static [u8] = include_bytes!("../../testdata/alice29.txt");
#[test]
fn test_alice29() {
  assert_decompressed_input_matches_output(ALICE29_BR, ALICE29, 65536, 65536);
}

#[test]
fn benchmark_alice29() {
  benchmark_decompressed_input(ALICE29_BR, ALICE29, 65536, 65536);
}

#[test]
fn test_alice1() {
  assert_decompressed_input_matches_output(ALICE29_BR, ALICE29, 1, 65536);
}

#[test]
fn test_backward65536() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/backward65536.compressed"),
                                           include_bytes!("../../testdata/backward65536"),
                                           65536,
                                           65536);
}


#[test]
fn test_compressed_file() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/compressed_file.compressed"),
                                           include_bytes!("../../testdata/compressed_file"),
                                           65536,
                                           65536);
}

#[test]
fn test_compressed_repeated() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/compressed_repeated.\
                                                           compressed"),
                                           include_bytes!("../../testdata/compressed_repeated"),
                                           65536,
                                           65536);
}

#[test]
fn test_empty() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty0() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.00"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.01"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty2() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.02"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty3() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.03"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty4() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.04"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty5() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.05"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty6() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.06"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty7() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.07"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty8() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.08"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty9() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.09"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty10() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.10"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty11() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.11"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty12() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.12"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty13() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.13"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty14() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.14"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty15() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.15"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty16() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.16"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty17() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.17"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}
#[test]
fn test_empty18() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/empty.compressed.18"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}

#[test]
fn lcet10() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/lcet10.txt.compressed"),
                                           include_bytes!("../../testdata/lcet10.txt"),
                                           65536,
                                           65536);
}

#[test]
fn test_mapsdatazrh() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/mapsdatazrh.compressed"),
                                           include_bytes!("../../testdata/mapsdatazrh"),
                                           65536,
                                           65536);
}

#[test]
fn test_monkey() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/monkey.compressed"),
                                           include_bytes!("../../testdata/monkey"),
                                           65536,
                                           65536);
}

#[test]
fn test_monkey1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/monkey.compressed"),
                                           include_bytes!("../../testdata/monkey"),
                                           1,
                                           1);
}

#[test]
fn test_monkey3() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/monkey.compressed"),
                                           include_bytes!("../../testdata/monkey"),
                                           3,
                                           65536);
}

#[test]
fn test_plrabn12() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/plrabn12.txt.compressed"),
                                           include_bytes!("../../testdata/plrabn12.txt"),
                                           65536,
                                           65536);
}

#[test]
fn test_random_org_10k() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/random_org_10k.bin.\
                                                           compressed"),
                                           include_bytes!("../../testdata/random_org_10k.bin"),
                                           65536,
                                           65536);
}

#[test]
fn test_ukkonooa() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/ukkonooa.compressed"),
                                           include_bytes!("../../testdata/ukkonooa"),
                                           65536,
                                           65536);
}

#[test]
fn test_ukkonooa3() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/ukkonooa.compressed"),
                                           include_bytes!("../../testdata/ukkonooa"),
                                           3,
                                           3);
}

#[test]
fn test_ukkonooa1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/ukkonooa.compressed"),
                                           include_bytes!("../../testdata/ukkonooa"),
                                           1,
                                           1);
}

#[test]
fn test_x() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/x.compressed"),
                                           include_bytes!("../../testdata/x"),
                                           65536,
                                           65536);
}
#[test]
fn test_x_0() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/x.compressed.00"),
                                           include_bytes!("../../testdata/x"),
                                           65536,
                                           65536);
}
#[test]
fn test_x_1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/x.compressed.01"),
                                           include_bytes!("../../testdata/x"),
                                           65536,
                                           65536);
}
#[test]
fn test_x_2() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/x.compressed.02"),
                                           include_bytes!("../../testdata/x"),
                                           65536,
                                           65536);
}
#[test]
fn test_x_3() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/x.compressed.03"),
                                           include_bytes!("../../testdata/x"),
                                           65536,
                                           65536);
}

#[test]
fn test_xyzzy() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/xyzzy.compressed"),
                                           include_bytes!("../../testdata/xyzzy"),
                                           65536,
                                           65536);
}

#[test]
fn test_zeros() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/zeros.compressed"),
                                           include_bytes!("../../testdata/zeros"),
                                           65536,
                                           65536);
}


#[test]
fn test_metablock_reset() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/metablock_reset.compressed"),
                                           include_bytes!("../../testdata/metablock_reset"),
                                           65536,
                                           65536);
}

#[test]
fn test_metablock_section_4_distance_symbol_0() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/fuzz502.compressed"),
                                           include_bytes!("../../testdata/fuzz502"),
                                           65536,
                                           65536);
}

#[test]
fn test_intact_distance_ring_buffer0() {
  static BR:&'static[u8] = &[0x1b, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x80, 0xe3, 0xb4, 0x0d, 0x00, 0x00,
                            0x07, 0x5b, 0x26, 0x31, 0x40, 0x02, 0x00, 0xe0, 0x4e, 0x1b, 0xa1, 0x80,
                            0x20, 0x00];
  static OUT:&'static[u8] = b"himselfself";
  assert_decompressed_input_matches_output(BR,
                                           OUT,
                                           65536,
                                           65536);
}

#[test]
fn test_intact_distance_ring_buffer1() {
  static BR:&'static[u8] = &[0x1b, 0x09, 0x00, 0x00, 0x00, 0x00, 0x80, 0xe3, 0xb4, 0x0d, 0x00, 0x00,
    0x07, 0x5b, 0x26, 0x31, 0x40, 0x02, 0x00, 0xe0, 0x4e, 0x1b, 0x21, 0xa0,
    0x20, 0x00
  ];
  static OUT:&'static[u8] = b"scrollroll";
  assert_decompressed_input_matches_output(BR,
                                           OUT,
                                           65536,
                                           65536);
}

#[test]
fn test_intact_distance_ring_buffer2() {
  static BR:&'static[u8] = &[0x1b, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x80, 0xe3, 0xb4, 0x0d, 0x00, 0x00,
    0x07, 0x5b, 0x26, 0x31, 0x40, 0x02, 0x00, 0xe0, 0x4e, 0x1b, 0x41, 0x80,
    0x20, 0x50, 0x10, 0x24, 0x08, 0x06];
  static OUT:&'static[u8] = b"leftdatadataleft";
  assert_decompressed_input_matches_output(BR,
                                           OUT,
                                           65536,
                                           65536);
}

#[test]
fn test_metablock_reset1_65536() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/metablock_reset.compressed"),
                                           include_bytes!("../../testdata/metablock_reset"),
                                           1,
                                           65536);
}

#[test]
fn test_metablock_reset65536_1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/metablock_reset.compressed"),
                                           include_bytes!("../../testdata/metablock_reset"),
                                           65536,
                                           1);
}

#[test]
fn test_metablock_reset1() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/metablock_reset.compressed"),
                                           include_bytes!("../../testdata/metablock_reset"),
                                           1,
                                           1);
}

#[test]
fn test_metablock_reset3() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/metablock_reset.compressed"),
                                           include_bytes!("../../testdata/metablock_reset"),
                                           3,
                                           3);
}

#[test]
#[should_panic]
fn test_broken_file() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/borked.compressed"),
                                           include_bytes!("../../testdata/empty"),
                                           65536,
                                           65536);
}

#[test]
fn test_ends_with_truncated_dictionary() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/ends_with_truncated_dictionary.\
                                                           compressed"),
                                           include_bytes!("../../testdata/ends_with_truncated_dictionary"),
                                           65536,
                                           65536);
}

#[test]
fn test_random_then_unicode() {
  assert_decompressed_input_matches_output(include_bytes!("../../testdata/random_then_unicode.\
                                                           compressed"),
                                           include_bytes!("../../testdata/random_then_unicode"),
                                           65536,
                                           65536);
}
#[test]
fn test_large_window() {
  assert_huge_file_input_matches_output(include_bytes!("../../testdata/rnd_chunk.br"),
                                        include_bytes!("../../testdata/rnd_prefix"),
                                        include_bytes!("../../testdata/rnd_postfix"),
                                        100011280,
                                        100000000,
                                        1,
                                        16384);
}
