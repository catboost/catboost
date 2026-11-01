use crate::weak_slice::WeakSliceMut;

#[derive(Debug)]
pub struct Window<'a> {
    // the full window allocation. This is longer than w_size so that operations don't need to
    // perform bounds checks.
    buf: WeakSliceMut<'a, u8>,

    window_bits: usize,
}

impl<'a> Window<'a> {
    pub unsafe fn from_raw_parts(ptr: *mut u8, window_bits: usize) -> Self {
        let len = (1 << window_bits) * 2;
        let buf = unsafe { WeakSliceMut::from_raw_parts_mut(ptr, len) };

        Self { buf, window_bits }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    pub fn capacity(&self) -> usize {
        2 * (1 << self.window_bits)
    }

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        // SAFETY: `self.buf` has been initialized for at least `filled` elements
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr().cast(), self.buf.len()) }
    }

    /// Returns a mutable reference to the filled portion of the buffer.
    #[inline]
    pub fn filled_mut(&mut self) -> &mut [u8] {
        // SAFETY: `self.buf` has been initialized for at least `filled` elements
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr().cast(), self.buf.len()) }
    }

    /// # Safety
    ///
    /// `src` must point to `range.end - range.start` valid (initialized!) bytes
    pub unsafe fn copy_and_initialize(&mut self, range: core::ops::Range<usize>, src: *const u8) {
        let (start, end) = (range.start, range.end);

        let dst = self.buf.as_mut_slice()[range].as_mut_ptr();
        unsafe { core::ptr::copy_nonoverlapping(src, dst, end - start) };
    }
}
