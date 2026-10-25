#![cfg(test)]

extern crate alloc_no_stdlib as alloc;
use alloc::{AllocatedStackMemory, Allocator, SliceWrapper, SliceWrapperMut, StackAllocator, bzero};
#[cfg(feature="std")]
use std::vec::Vec;
#[cfg(feature="std")]
use std::io;

use core::ops;


pub use super::{BrotliDecompressStream, BrotliResult, BrotliState, HuffmanCode};

declare_stack_allocator_struct!(MemPool, 4096, stack);



fn oneshot(input: &mut [u8], mut output: &mut [u8]) -> (BrotliResult, usize, usize) {
  let mut available_out: usize = output.len();
  let mut stack_u8_buffer = define_allocator_memory_pool!(4096, u8, [0; 300 * 1024], stack);
  let mut stack_u32_buffer = define_allocator_memory_pool!(4096, u32, [0; 12 * 1024], stack);
  let mut stack_hc_buffer = define_allocator_memory_pool!(4096,
                                                          super::HuffmanCode,
                                                          [HuffmanCode::default(); 18 * 1024],
                                                          stack);
  let stack_u8_allocator = MemPool::<u8>::new_allocator(&mut stack_u8_buffer, bzero);
  let stack_u32_allocator = MemPool::<u32>::new_allocator(&mut stack_u32_buffer, bzero);
  let stack_hc_allocator = MemPool::<HuffmanCode>::new_allocator(&mut stack_hc_buffer, bzero);
  let mut available_in: usize = input.len();
  let mut input_offset: usize = 0;
  let mut output_offset: usize = 0;
  let mut written: usize = 0;
  let mut brotli_state =
    BrotliState::new(stack_u8_allocator, stack_u32_allocator, stack_hc_allocator);
  let result = BrotliDecompressStream(&mut available_in,
                                      &mut input_offset,
                                      &input[..],
                                      &mut available_out,
                                      &mut output_offset,
                                      &mut output,
                                      &mut written,
                                      &mut brotli_state);
  return (result, input_offset, output_offset);
}

#[test]
fn test_10x10y() {
  const BUFFER_SIZE: usize = 2048;
  let mut input: [u8; 12] = [0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02,
                             0x8a];
  let mut output = [0u8; BUFFER_SIZE];
  let (result, input_offset, output_offset) = oneshot(&mut input[..], &mut output[..]);
  match result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  let mut i: usize = 0;
  while i < 10 {
    assert_eq!(output[i], 'X' as u8);
    assert_eq!(output[i + 10], 'Y' as u8);
    i += 1;
  }
  assert_eq!(output_offset, 20);
  assert_eq!(input_offset, input.len());
}



#[test]
fn test_x() {
  const BUFFER_SIZE: usize = 128;
  let mut input: [u8; 5] = [0x0b, 0x00, 0x80, 0x58, 0x03];
  let mut output = [0u8; BUFFER_SIZE];
  let (result, input_offset, output_offset) = oneshot(&mut input[..], &mut output[..]);
  match result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  assert_eq!(output[0], 'X' as u8);
  assert_eq!(output_offset, 1);
  assert_eq!(input_offset, input.len());
}

#[test]
fn test_corrupt_input_large_distance_code() {
  const BUFFER_SIZE: usize = 128;
  let mut input: [u8; 46] = [17, 139, 32, 255, 8, 0, 136, 255, 32, 46, 146, 32, 255, 255, 255, 255, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32];
  let mut output = [0u8; BUFFER_SIZE];
  let (result, _, _) = oneshot(&mut input[..], &mut output[..]);
  match result {
    BrotliResult::ResultFailure => {}
    _ => assert!(false),
  }
}

#[test]
fn test_empty() {
  const BUFFER_SIZE: usize = 128;
  let mut input: [u8; 1] = [0x06];
  let mut output = [0u8; BUFFER_SIZE];
  let (result, input_offset, output_offset) = oneshot(&mut input[..], &mut output[..]);
  match result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  assert_eq!(output_offset, 0);
  assert_eq!(input_offset, input.len());
}
const QF_BUFFER_SIZE: usize = 180 * 1024;
static mut quick_fox_output: [u8; QF_BUFFER_SIZE] = [0u8; QF_BUFFER_SIZE];

#[test]
fn test_quickfox_repeated_custom() {
  let mut input: [u8; 58] =
    [0x5B, 0xFF, 0xAF, 0x02, 0xC0, 0x22, 0x79, 0x5C, 0xFB, 0x5A, 0x8C, 0x42, 0x3B, 0xF4, 0x25,
     0x55, 0x19, 0x5A, 0x92, 0x99, 0xB1, 0x35, 0xC8, 0x19, 0x9E, 0x9E, 0x0A, 0x7B, 0x4B, 0x90,
     0xB9, 0x3C, 0x98, 0xC8, 0x09, 0x40, 0xF3, 0xE6, 0xD9, 0x4D, 0xE4, 0x6D, 0x65, 0x1B, 0x27,
     0x87, 0x13, 0x5F, 0xA6, 0xE9, 0x30, 0x96, 0x7B, 0x3C, 0x15, 0xD8, 0x53, 0x1C];

  let (result, input_offset, output_offset) = oneshot(&mut input[..], &mut unsafe{&mut quick_fox_output[..]});
  match result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  assert_eq!(output_offset, 176128);
  assert_eq!(input_offset, input.len());
  const fox: [u8; 0x2b] = [0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6B, 0x20, 0x62, 0x72,
                           0x6F, 0x77, 0x6E, 0x20, 0x66, 0x6F, 0x78, 0x20, 0x6A, 0x75, 0x6D, 0x70,
                           0x73, 0x20, 0x6F, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6C,
                           0x61, 0x7A, 0x79, 0x20, 0x64, 0x6F, 0x67];
  let mut index: usize = 0;
  for item in unsafe{quick_fox_output[0..176128].iter()} {
    assert_eq!(*item, fox[index]);
    index += 1;
    if index == 0x2b {
      index = 0;
    }
  }
}

static mut quick_fox_exported_output: [u8; QF_BUFFER_SIZE * 3] = [0u8; QF_BUFFER_SIZE * 3];
#[test]
fn test_quickfox_repeated_exported() {
  let input: [u8; 58] =
    [0x5B, 0xFF, 0xAF, 0x02, 0xC0, 0x22, 0x79, 0x5C, 0xFB, 0x5A, 0x8C, 0x42, 0x3B, 0xF4, 0x25,
     0x55, 0x19, 0x5A, 0x92, 0x99, 0xB1, 0x35, 0xC8, 0x19, 0x9E, 0x9E, 0x0A, 0x7B, 0x4B, 0x90,
     0xB9, 0x3C, 0x98, 0xC8, 0x09, 0x40, 0xF3, 0xE6, 0xD9, 0x4D, 0xE4, 0x6D, 0x65, 0x1B, 0x27,
     0x87, 0x13, 0x5F, 0xA6, 0xE9, 0x30, 0x96, 0x7B, 0x3C, 0x15, 0xD8, 0x53, 0x1C];
  let res = ::brotli_decode(&input[..], unsafe{&mut quick_fox_exported_output[..]});
  match res.result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  assert_eq!(res.decoded_size, 176128);
  const fox: [u8; 0x2b] = [0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6B, 0x20, 0x62, 0x72,
                           0x6F, 0x77, 0x6E, 0x20, 0x66, 0x6F, 0x78, 0x20, 0x6A, 0x75, 0x6D, 0x70,
                           0x73, 0x20, 0x6F, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6C,
                           0x61, 0x7A, 0x79, 0x20, 0x64, 0x6F, 0x67];
  let mut index: usize = 0;
  for item in unsafe{quick_fox_exported_output[0..176128].iter()} {
    assert_eq!(*item, fox[index]);
    index += 1;
    if index == 0x2b {
      index = 0;
    }
  }
}

static mut quick_fox_prealloc_output: [u8; QF_BUFFER_SIZE * 3] = [0u8; QF_BUFFER_SIZE * 3];
#[test]
fn test_quickfox_repeated_exported_prealloc() {
  let input: [u8; 58] =
    [0x5B, 0xFF, 0xAF, 0x02, 0xC0, 0x22, 0x79, 0x5C, 0xFB, 0x5A, 0x8C, 0x42, 0x3B, 0xF4, 0x25,
     0x55, 0x19, 0x5A, 0x92, 0x99, 0xB1, 0x35, 0xC8, 0x19, 0x9E, 0x9E, 0x0A, 0x7B, 0x4B, 0x90,
     0xB9, 0x3C, 0x98, 0xC8, 0x09, 0x40, 0xF3, 0xE6, 0xD9, 0x4D, 0xE4, 0x6D, 0x65, 0x1B, 0x27,
     0x87, 0x13, 0x5F, 0xA6, 0xE9, 0x30, 0x96, 0x7B, 0x3C, 0x15, 0xD8, 0x53, 0x1C];
  let (qf, scratch) = unsafe{quick_fox_prealloc_output.split_at_mut(QF_BUFFER_SIZE)};
  let res = ::brotli_decode_prealloc(&input[..], qf, scratch, &mut[0u32;65536][..], &mut[HuffmanCode::default();65536][..]);
  match res.result {
    BrotliResult::ResultSuccess => {}
    _ => assert!(false),
  }
  assert_eq!(res.decoded_size, 176128);
  const fox: [u8; 0x2b] = [0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6B, 0x20, 0x62, 0x72,
                           0x6F, 0x77, 0x6E, 0x20, 0x66, 0x6F, 0x78, 0x20, 0x6A, 0x75, 0x6D, 0x70,
                           0x73, 0x20, 0x6F, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6C,
                           0x61, 0x7A, 0x79, 0x20, 0x64, 0x6F, 0x67];
  let mut index: usize = 0;
  for item in qf[0..176128].iter() {
    assert_eq!(*item, fox[index]);
    index += 1;
    if index == 0x2b {
      index = 0;
    }
  }
}


#[cfg(feature="std")]
struct Buffer {
  data: Vec<u8>,
  read_offset: usize,
}
#[cfg(feature="std")]
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
#[cfg(feature="std")]
impl io::Read for Buffer {
  fn read(self: &mut Self, buf: &mut [u8]) -> io::Result<usize> {
    let bytes_to_read = ::core::cmp::min(buf.len(), self.data.len() - self.read_offset);
    if bytes_to_read > 0 {
      buf[0..bytes_to_read]
        .clone_from_slice(&self.data[self.read_offset..self.read_offset + bytes_to_read]);
    }
    self.read_offset += bytes_to_read;
    return Ok(bytes_to_read);
  }
}
#[cfg(feature="std")]
impl io::Write for Buffer {
  fn write(self: &mut Self, buf: &[u8]) -> io::Result<usize> {
    self.data.extend(buf);
    return Ok(buf.len());
  }
  fn flush(self: &mut Self) -> io::Result<()> {
    return Ok(());
  }
}


#[test]
#[cfg(feature="std")]
fn test_reader_quickfox_repeated() {
  let in_buf: [u8; 58] = [0x5B, 0xFF, 0xAF, 0x02, 0xC0, 0x22, 0x79, 0x5C, 0xFB, 0x5A, 0x8C, 0x42,
                          0x3B, 0xF4, 0x25, 0x55, 0x19, 0x5A, 0x92, 0x99, 0xB1, 0x35, 0xC8, 0x19,
                          0x9E, 0x9E, 0x0A, 0x7B, 0x4B, 0x90, 0xB9, 0x3C, 0x98, 0xC8, 0x09, 0x40,
                          0xF3, 0xE6, 0xD9, 0x4D, 0xE4, 0x6D, 0x65, 0x1B, 0x27, 0x87, 0x13, 0x5F,
                          0xA6, 0xE9, 0x30, 0x96, 0x7B, 0x3C, 0x15, 0xD8, 0x53, 0x1C];

  let mut output = Buffer::new(&[]);
  let mut input = super::Decompressor::new(Buffer::new(&in_buf), 4096);
  match super::copy_from_to(&mut input, &mut output) {
    Ok(_) => {}
    Err(e) => panic!("Error {:?}", e),
  }

  assert_eq!(output.data.len(), 176128);
  const fox: [u8; 0x2b] = [0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6B, 0x20, 0x62, 0x72,
                           0x6F, 0x77, 0x6E, 0x20, 0x66, 0x6F, 0x78, 0x20, 0x6A, 0x75, 0x6D, 0x70,
                           0x73, 0x20, 0x6F, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6C,
                           0x61, 0x7A, 0x79, 0x20, 0x64, 0x6F, 0x67];
  let mut index: usize = 0;
  for item in output.data[0..176128].iter_mut() {
    assert_eq!(*item, fox[index]);
    index += 1;
    if index == 0x2b {
      index = 0;
    }
  }
}

#[test]
fn test_early_eof() {
  const BUFFER_SIZE: usize = 128;
  let mut input: [u8; 47] = [17, 17, 32, 32, 109, 109, 32, 32, 32, 181, 2, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 151, 32, 42, 181, 32, 149, 59, 0, 0, 0, 0, 42, 42, 42, 42, 42, 5, 255, 255, 255, 255, 255];
  let mut output = [0u8; BUFFER_SIZE];
  let (result, input_offset, _output_offset) = oneshot(&mut input[..], &mut output[..]);
  match result {
    BrotliResult::ResultFailure => {}
    _ => assert!(false),
  }
  assert_eq!(input_offset, input.len());
}

#[test]
#[cfg(feature="std")]
fn test_run_out_of_writer_space() {
  // this is a valid compression of [0u8; 2048];
  let compression = [27, 255, 7, 0, 36, 0, 194, 177, 64, 114, 7];
  // output buffer doesn't have enough space
  let mut output_buffer = [0u8; 2047];

  super::BrotliDecompress(
    &mut io::Cursor::new(compression),
    &mut io::Cursor::new(&mut output_buffer[..]),
  )
  .unwrap_err();
}

#[test]
fn test_dict() {
  let patch: &[u8] = &[
    27, 103, 0, 96, 47, 14, 120, 211, 142, 228, 22, 15, 167, 193, 55, 28, 228, 226, 254, 54, 10,
    36, 226, 192, 19, 76, 50, 8, 169, 92, 9, 197, 47, 12, 211, 114, 34, 175, 18, 241, 122, 134,
    170, 32, 189, 4, 112, 153, 119, 12, 237, 23, 120, 130, 2,
  ];

  let dict: Vec<u8> = vec![
    2, 0, 0, 0, 0, 213, 195, 31, 121, 231, 225, 250, 238, 34, 174, 158, 246, 208, 145, 187, 92, 2,
    0, 0, 4, 0, 0, 0, 46, 0, 0, 0, 0, 0, 11, 123, 105, 100, 125, 46, 105, 102, 116, 95, 116, 107,
    20, 0, 0, 52, 40, 103, 221, 215, 223, 255, 95, 54, 15, 13, 85, 53, 206, 115, 249, 165, 159,
    159, 16, 29, 37, 17, 114, 1, 163, 2, 16, 33, 51, 4, 32, 0, 226, 29, 19, 88, 254, 195, 129, 23,
    25, 22, 8, 19, 21, 41, 130, 136, 51, 8, 67, 209, 52, 204, 204, 70, 199, 130, 252, 47, 16, 40,
    186, 251, 62, 63, 19, 236, 147, 240, 211, 215, 59,
  ];

  let mut input_buffer: [u8; 4096] = [0; 4096];
  let mut output_buffer: [u8; 4096] = [0; 4096];

  let mut cursor = io::Cursor::new(patch);
  let mut output: Vec<u8> = vec![];

  let res = super::BrotliDecompressCustomDict(
    &mut cursor,
    &mut output,
    &mut input_buffer,
    &mut output_buffer,
    dict,
  );

  assert!(res.is_ok(), "Unexpected error {:?}", res);
  assert_eq!(
    output,
    vec![
      0x02, 0x00, 0x00, 0x00, 0x00, 0x8c, 0x16, 0xa6, 0x25, 0x18, 0xf8, 0x68, 0x63, 0x4e, 0xe4,
      0x09, 0x2b, 0xa1, 0xe2, 0x4b, 0xba, 0x02, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2e, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x0b, 0x7b, 0x69, 0x64, 0x7d, 0x2e, 0x69, 0x66, 0x74, 0x5f, 0x74,
      0x6b, 0x14, 0x00, 0x00, 0x38, 0x1d, 0x25, 0x11, 0x72, 0x01, 0xa3, 0x02, 0x10, 0x21, 0x33,
      0x04, 0x20, 0x00, 0xe2, 0x1d, 0x13, 0x58, 0xfe, 0xc3, 0x81, 0x17, 0x19, 0x16, 0x08, 0x13,
      0x15, 0x29, 0x82, 0x88, 0x33, 0x08, 0x43, 0xd1, 0x34, 0xcc, 0xcc, 0x46, 0xc7, 0x82, 0xfc,
      0x2f, 0x10, 0x28, 0xba, 0xfb, 0x3e, 0x3f, 0x13, 0xec, 0x93, 0xf0, 0xd3, 0xd7, 0x3b,
    ]
  );
}



#[test]
fn test_dict_medium() {
  let br: &[u8] = &[
      27, 250, 0, 64, 44, 11, 108, 247, 52, 24, 142, 163, 204, 142, 80, 252, 182, 120, 165, 79, 250, 13, 158, 3, 30, 234, 40, 250, 29, 20, 77, 120, 230, 200, 129, 115, 130, 14, 64, 215, 56, 237, 122, 86, 138, 52, 110, 119, 30, 215, 82, 74, 30, 171, 105, 88, 99, 31, 14, 167, 214, 226, 231, 246, 248, 42, 94, 190, 205, 223, 231, 243, 213, 253, 63, 108, 192, 137, 120, 140, 143, 190, 202, 64, 147, 222, 31, 143, 132, 147, 173, 58, 1, 126, 218, 171, 171, 199, 239, 64, 16, 232, 46, 13, 155, 237, 189, 161, 186, 4, 147, 245, 53, 148, 218, 183, 80, 50, 59, 249, 130, 113, 103, 219, 228, 206, 36, 150, 127, 93, 210, 225, 40, 54, 247, 51, 28, 139, 149, 194, 210, 171, 62, 190, 158, 203, 35, 87, 91, 43, 9, 5, 0, 28, 217, 82, 157, 50, 63, 118, 229, 72, 167, 108, 155, 216, 214, 2, 116, 200, 103, 42, 194, 63, 159, 85, 202, 72, 167, 142, 139, 27, 106, 104, 251, 151, 64, 122, 231, 226, 114, 39, 28, 49, 117, 70, 13, 65, 119, 69, 181, 42, 87, 152, 223, 0
  ];

  let mut dict: Vec<u8> = vec![0u8; 256];
  for (index, val) in dict[..].iter_mut().enumerate()
  {
    *val = index as u8;
  }

  let mut input_buffer: [u8; 4096] = [0; 4096];
  let mut output_buffer: [u8; 4096] = [0; 4096];

  let mut cursor = io::Cursor::new(br);
  let mut output: Vec<u8> = vec![];

  let res = super::BrotliDecompressCustomDict(
    &mut cursor,
    &mut output,
    &mut input_buffer,
    &mut output_buffer,
    dict,
  );

  assert!(res.is_ok(), "Unexpected error {:?}", res);
  assert_eq!(
    output,
    vec![148, 100, 52, 4, 5, 6, 7, 214, 165, 116, 67, 18, 225, 176, 127, 78, 29, 235, 185, 135, 85, 35, 241, 191, 141, 91, 41, 247, 196, 145, 94, 43, 248, 197, 146, 95, 44, 249, 198, 146, 94, 42, 246, 194, 142, 90, 38, 242, 190, 138, 85, 32, 235, 182, 129, 76, 23, 226, 173, 120, 67, 13, 215, 161, 107, 53, 255, 201, 147, 93, 39, 241, 186, 131, 76, 21, 222, 167, 112, 57, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 215, 158, 101, 44, 243, 186, 129, 72, 15, 16, 17, 215, 157, 99, 41, 239, 181, 123, 65, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 215, 155, 95, 35, 231, 171, 111, 51, 247, 187, 127, 66, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 216, 154, 92, 29, 222, 159, 96, 33, 226, 163, 100, 37, 230, 167, 103, 39, 231, 167, 103, 39, 231, 167, 103, 39, 231, 166, 101, 36, 227, 162, 97, 32, 223, 158, 93, 28, 218, 152, 86, 20, 21, 22, 23, 24, 25, 26, 27, 216, 149, 82, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 217, 149, 81, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 217, 147, 77, 7, 8, 9, 10]
  );
}
