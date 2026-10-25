use core::{marker::PhantomData, mem::MaybeUninit};

use crate::weak_slice::WeakSliceMut;

pub struct Pending<'a> {
    /// start of the allocation
    buf: WeakSliceMut<'a, MaybeUninit<u8>>,
    /// next pending byte to output to the stream
    out: usize,
    /// number of bytes in the pending buffer
    pub(crate) pending: usize,
    /// semantically we're storing a mutable slice of bytes
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> Pending<'a> {
    pub fn reset_keep(&mut self) {
        // keep the buffer as it is
        self.pending = 0;
    }

    pub fn pending(&self) -> &[u8] {
        let slice = &self.buf.as_slice()[self.out..][..self.pending];
        // SAFETY: the slice contains initialized bytes.
        unsafe { &*(slice as *const [MaybeUninit<u8>] as *const [u8]) }
    }

    /// Number of bytes that can be added to the pending buffer until it is full
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - (self.out + self.pending)
    }

    /// Total number of bytes that can be stored in the pending buffer
    pub(crate) fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    #[track_caller]
    /// Mark a number of pending bytes as no longer pending
    pub fn advance(&mut self, number_of_bytes: usize) {
        debug_assert!(self.pending >= number_of_bytes);

        self.out = self.out.wrapping_add(number_of_bytes);
        self.pending -= number_of_bytes;

        if self.pending == 0 {
            self.out = 0;
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn rewind(&mut self, n: usize) {
        assert!(n <= self.pending, "rewinding past then start");

        self.pending -= n;

        if self.pending == 0 {
            self.out = 0;
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn extend(&mut self, buf: &[u8]) {
        assert!(
            self.remaining() >= buf.len(),
            "buf.len() must fit in remaining()"
        );

        // SAFETY: [u8] is valid [MaybeUninit<u8>]
        let buf = unsafe { &*(buf as *const [u8] as *const [MaybeUninit<u8>]) };

        self.buf.as_mut_slice()[self.out + self.pending..][..buf.len()].copy_from_slice(buf);

        self.pending += buf.len();
    }

    pub(crate) unsafe fn from_raw_parts(ptr: *mut MaybeUninit<u8>, len: usize) -> Self {
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) };

        Self {
            buf,
            out: 0,
            pending: 0,
            _marker: PhantomData,
        }
    }

    pub(crate) unsafe fn clone_to(&self, ptr: *mut u8) -> Self {
        let ptr = ptr.cast::<MaybeUninit<u8>>();
        unsafe { ptr.copy_from_nonoverlapping(self.buf.as_ptr(), self.buf.len()) };
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, self.buf.len()) },
            out: self.out,
            pending: self.pending,
            _marker: PhantomData,
        }
    }
}
