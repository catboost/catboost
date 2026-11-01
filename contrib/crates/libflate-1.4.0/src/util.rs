#[cfg(all(test, feature = "no_std"))]
use core2::io::{self, Read};
#[cfg(all(test, not(feature = "no_std")))]
use std::io::{self, Read};

#[cfg(test)]
pub struct WouldBlockReader<R> {
    inner: R,
    do_block: bool,
}
#[cfg(test)]
impl<R: Read> WouldBlockReader<R> {
    pub fn new(inner: R) -> Self {
        WouldBlockReader {
            inner,
            do_block: false,
        }
    }
}
#[cfg(test)]
impl<R: Read> Read for WouldBlockReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.do_block = !self.do_block;
        if self.do_block {
            Err(io::Error::new(io::ErrorKind::WouldBlock, "Would block"))
        } else if buf.is_empty() {
            Ok(0)
        } else {
            let mut byte = [0; 1];
            if self.inner.read(&mut byte[..])? == 1 {
                buf[0] = byte[0];
                Ok(1)
            } else {
                Ok(0)
            }
        }
    }
}

#[cfg(test)]
pub fn nb_read_to_end<R: Read>(mut reader: R) -> io::Result<Vec<u8>> {
    let mut buf = vec![0; 1024];
    let mut offset = 0;
    loop {
        match reader.read(&mut buf[offset..]) {
            Err(e) => {
                if e.kind() != io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
            Ok(0) => {
                buf.truncate(offset);
                break;
            }
            Ok(size) => {
                offset += size;
                if offset == buf.len() {
                    buf.resize(offset * 2, 0);
                }
            }
        }
    }
    Ok(buf)
}
