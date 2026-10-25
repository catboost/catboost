#![cfg(test)]
#![cfg(feature="std")]
use std::io::{self, Read, Cursor};

extern crate brotli_decompressor;

static ENCODED: &'static [u8] = b"\x1b\x03)\x00\xa4\xcc\xde\xe2\xb3 vA\x00\x0c";

enum State { First, Second, Third, Fourth }
struct R(State);
impl Read for R {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.0 {
            State::First => {
                buf[0] = ENCODED[0];
                self.0 = State::Second;
                return Ok(1);
            }
            State::Second => {
                self.0 = State::Third;
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "foo"));
            }
            State::Third => {
                self.0 = State::Fourth;
                buf[..ENCODED.len() - 1].copy_from_slice(&ENCODED[1..]);
                return Ok(ENCODED[1..].len());
            }
            State::Fourth => {
                return Ok(0);
            }
        }
    }
}
#[test]
fn test_would_block() {

    let mut d = brotli_decompressor::Decompressor::new(R(State::First), 8192);
    let mut b = [0; 8192];
    assert_eq!(d.read(&mut b).unwrap_err().kind(), io::ErrorKind::WouldBlock);
    assert!(d.read(&mut b).unwrap() != 0);
    println!("{}", String::from_utf8(b.to_vec()).unwrap());
    assert!(d.read(&mut b).unwrap() != 0);
    assert_eq!(d.read(&mut b).unwrap(), 0);
}

static ENCODED2: &'static [u8] = include_bytes!("ipsum.brotli");
static DECODED: &'static str = include_str!("ipsum.raw");

enum State2 {
    First,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    Seventh,
    Eighth,
}

struct R2 {
    offset: usize,
    len: usize,
    state: State2,
}

impl R2 {
    fn new() -> R2 {
        R2 {
            offset: 0,
            len: 1,
            state: State2::First,
        }
    }
}
impl Read for R2 {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.state {
            State2::First => {
                self.state = State2::Second;
                let len = self.len;
                buf[..len].copy_from_slice(&ENCODED2[self.offset..self.offset+len]);
                self.offset += len;
                self.len = 100;
                return Ok(len);
            }
            State2::Second => {
                self.state = State2::Third;
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "foo"));
            }
            State2::Third => {
                self.state = State2::Fourth;
                let len = self.len;
                buf[..len].copy_from_slice(&ENCODED2[self.offset..self.offset+len]);
                self.offset += len;
                self.len = 100;
                return Ok(len);
            }
            State2::Fourth => {
                self.state = State2::Fifth;
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "foo"));
            }
            State2::Fifth => {
                self.state = State2::Sixth;
                let len = self.len;
                buf[..len].copy_from_slice(&ENCODED2[self.offset..self.offset+len]);
                self.offset += len;
                self.len = 100;
                return Ok(len);
            }
            State2::Sixth => {
                self.state = State2::Seventh;
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "foo"));
            }
            State2::Seventh => {
                self.state = State2::Eighth;
                buf[..ENCODED2.len() - self.offset].copy_from_slice(&ENCODED2[self.offset..]);
                return Ok(ENCODED2.len() - self.offset);
            }
            State2::Eighth => {
                return Ok(0);
            }
        }
    }
}

#[test]
fn would_block_more() {
    // Reference synchronous decoding.
    let mut b = [0; 8192];
    let mut bytes = vec![];
    let mut d = brotli_decompressor::Decompressor::new(Cursor::new(ENCODED2), 8192);
    let read = d.read(&mut b).unwrap();
    assert!(read != 0);
    bytes.extend_from_slice(&b[0..read]);

    assert_eq!(d.read(&mut b).unwrap(), 0);
    let reference_decoded = String::from_utf8(bytes).unwrap();
    // Ensure synchronous decoding matches original input.
    assert_eq!(reference_decoded, DECODED);

    // Incremental decoding.
    let r = R2::new();
    let mut d = brotli_decompressor::Decompressor::new(r, 8192);
    let mut bytes = vec![];
    let mut b = [0; 8192];

    assert_eq!(d.read(&mut b).unwrap_err().kind(), io::ErrorKind::WouldBlock);

    assert_eq!(d.read(&mut b).unwrap_err().kind(), io::ErrorKind::WouldBlock);

    let read = d.read(&mut b).unwrap();
    assert!(read != 0);
    bytes.extend_from_slice(&b[0..read]);

    assert_eq!(d.read(&mut b).unwrap_err().kind(), io::ErrorKind::WouldBlock);

    let read = d.read(&mut b).unwrap();
    assert!(read != 0);
    bytes.extend_from_slice(&b[0..read]);

    assert_eq!(d.read(&mut b).unwrap(), 0);

    let decoded = String::from_utf8(bytes).unwrap();

    // Ensure incremental decoding matches original input after brotli decompressor is finished.
    assert_eq!(decoded, reference_decoded);
}
