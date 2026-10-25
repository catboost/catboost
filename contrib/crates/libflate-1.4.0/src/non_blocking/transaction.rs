use crate::bit;
#[cfg(feature = "no_std")]
use alloc::vec::Vec;
#[cfg(feature = "no_std")]
use core::cmp;
#[cfg(feature = "no_std")]
use core2::io::{self, Read};
#[cfg(not(feature = "no_std"))]
use std::{
    cmp,
    io::{self, Read},
};

#[derive(Debug)]
pub struct TransactionalBitReader<R> {
    inner: bit::BitReader<TransactionalReader<R>>,
    savepoint: bit::BitReaderState,
}
impl<R: Read> TransactionalBitReader<R> {
    pub fn new(inner: R) -> Self {
        let inner = bit::BitReader::new(TransactionalReader::new(inner));
        let savepoint = inner.state();
        TransactionalBitReader { inner, savepoint }
    }
    #[inline]
    pub fn transaction<F, T>(&mut self, f: F) -> io::Result<T>
    where
        F: FnOnce(&mut bit::BitReader<TransactionalReader<R>>) -> io::Result<T>,
    {
        self.start_transaction();
        let result = f(&mut self.inner);
        if result.is_ok() {
            self.commit_transaction();
        } else {
            self.abort_transaction();
        }
        result
    }
    #[inline]
    pub fn start_transaction(&mut self) {
        self.inner.as_inner_mut().start_transaction();
        self.savepoint = self.inner.state();
    }
    #[inline]
    pub fn abort_transaction(&mut self) {
        self.inner.as_inner_mut().abort_transaction();
        self.inner.restore_state(self.savepoint);
    }
    #[inline]
    pub fn commit_transaction(&mut self) {
        self.inner.as_inner_mut().commit_transaction();
    }
}
impl<R> TransactionalBitReader<R> {
    pub fn as_inner_ref(&self) -> &R {
        &self.inner.as_inner_ref().inner
    }
    pub fn as_inner_mut(&mut self) -> &mut R {
        &mut self.inner.as_inner_mut().inner
    }
    pub fn into_inner(self) -> R {
        self.inner.into_inner().inner
    }
}

#[derive(Debug)]
pub struct TransactionalReader<R> {
    inner: R,
    in_transaction: bool,
    buffer: Vec<u8>,
    offset: usize,
}
impl<R> TransactionalReader<R> {
    pub fn new(inner: R) -> Self {
        TransactionalReader {
            inner,
            buffer: Vec::new(),
            in_transaction: false,
            offset: 0,
        }
    }
    #[inline]
    pub fn start_transaction(&mut self) {
        assert!(!self.in_transaction);
        self.in_transaction = true;
    }
    #[inline]
    pub fn commit_transaction(&mut self) {
        self.in_transaction = false;
        self.offset = 0;
        self.buffer.clear();
    }
    #[inline]
    pub fn abort_transaction(&mut self) {
        self.in_transaction = false;
        self.offset = 0;
    }
}
impl<R: Read> Read for TransactionalReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.offset < self.buffer.len() {
            let unread_buf_size = self.buffer.len() - self.offset;
            let size = cmp::min(buf.len(), unread_buf_size);
            buf[0..size].copy_from_slice(&self.buffer[self.offset..self.offset + size]);
            self.offset += size;
            return Ok(size);
        }

        let size = self.inner.read(buf)?;
        if self.in_transaction {
            self.buffer.extend_from_slice(&buf[0..size]);
            self.offset += size;
        }
        Ok(size)
    }
}
