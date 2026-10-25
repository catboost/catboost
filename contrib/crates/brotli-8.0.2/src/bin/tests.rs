#![cfg(test)]

extern crate core;

use core::cmp::min;
use std::io;

struct Buffer {
    data: Vec<u8>,
    read_offset: usize,
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
impl io::Read for Buffer {
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
impl io::Write for Buffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.extend(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn copy_from_to<R: io::Read, W: io::Write>(mut r: R, mut w: W) -> io::Result<usize> {
    let mut buffer: [u8; 65536] = [0; 65536];
    let mut out_size: usize = 0;
    loop {
        match r.read(&mut buffer[..]) {
            Err(e) => {
                match e.kind() {
                    io::ErrorKind::Interrupted => continue,
                    _ => {}
                }
                return Err(e);
            }
            Ok(size) => {
                if size == 0 {
                    break;
                } else {
                    match w.write_all(&buffer[..size]) {
                        Err(e) => {
                            match e.kind() {
                                io::ErrorKind::Interrupted => continue,
                                _ => {}
                            }
                            return Err(e);
                        }
                        Ok(_) => out_size += size,
                    }
                }
            }
        }
    }
    Ok(out_size)
}

#[test]
fn test_10x_10y() {
    let in_buf: [u8; 12] = [
        0x1b, 0x13, 0x00, 0x00, 0xa4, 0xb0, 0xb2, 0xea, 0x81, 0x47, 0x02, 0x8a,
    ];

    let mut output = Buffer::new(&[]);
    let mut input = super::BrotliDecompressor::new(Buffer::new(&in_buf), 4096);
    match copy_from_to(&mut input, &mut output) {
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
}

#[test]
fn test_alice() {
    let in_buf = include_bytes!("../../testdata/alice29.txt.compressed");

    let mut output = Buffer::new(&[]);
    let mut input = super::BrotliDecompressor::new(Buffer::new(in_buf), 1);
    match copy_from_to(&mut input, &mut output) {
        Ok(_) => {}
        Err(e) => panic!("Error {:?}", e),
    }
    let mut i: usize = 0;
    let truth = include_bytes!("../../testdata/alice29.txt");
    while i < truth.len() {
        assert_eq!(output.data[i], truth[i]);
        i += 1;
    }
    assert_eq!(truth.len(), output.data.len());
}
