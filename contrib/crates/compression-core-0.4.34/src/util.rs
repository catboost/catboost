use core::mem::MaybeUninit;

pub const fn _assert_send<T: Send>() {}
pub const fn _assert_sync<T: Sync>() {}

#[derive(Debug, Default)]
pub struct PartialBuffer<B> {
    buffer: B,
    index: usize,
}

impl<B: AsRef<[u8]>> PartialBuffer<B> {
    pub fn new(buffer: B) -> Self {
        Self { buffer, index: 0 }
    }

    pub fn written(&self) -> &[u8] {
        &self.buffer.as_ref()[..self.index]
    }

    /// Convenient method for `.writen().len()`
    pub fn written_len(&self) -> usize {
        self.index
    }

    pub fn unwritten(&self) -> &[u8] {
        &self.buffer.as_ref()[self.index..]
    }

    pub fn advance(&mut self, amount: usize) {
        self.index += amount;
        debug_assert!(self.index <= self.buffer.as_ref().len());
    }

    pub fn get_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    pub fn into_inner(self) -> B {
        self.buffer
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> PartialBuffer<B> {
    pub fn unwritten_mut(&mut self) -> &mut [u8] {
        &mut self.buffer.as_mut()[self.index..]
    }

    pub fn copy_unwritten_from<C: AsRef<[u8]>>(&mut self, other: &mut PartialBuffer<C>) -> usize {
        let len = self.unwritten().len().min(other.unwritten().len());

        self.unwritten_mut()[..len].copy_from_slice(&other.unwritten()[..len]);

        self.advance(len);
        other.advance(len);
        len
    }
}

impl<B: AsRef<[u8]> + Default> PartialBuffer<B> {
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> From<B> for PartialBuffer<B> {
    fn from(buffer: B) -> Self {
        Self::new(buffer)
    }
}

/// Write buffer for compression-codecs.
///
/// Currently it only supports initialized buffer, but will support uninitialized
/// buffer soon.
///
/// # Layout
///
/// ```text
/// |                                       buffer                                    |
/// | written and initialized | unwritten but initialized | unwritten and uninitialized
/// ```
#[derive(Debug)]
pub struct WriteBuffer<'a> {
    buffer: &'a mut [MaybeUninit<u8>],
    index: usize,
    initialized: usize,
}

impl<'a> WriteBuffer<'a> {
    pub fn new_initialized(buffer: &'a mut [u8]) -> Self {
        Self {
            initialized: buffer.len(),
            // Safety: with initialized set to len of the buffer,
            // `WriteBuffer` would treat it as a `&mut [u8]`.
            buffer: unsafe { &mut *(buffer as *mut [u8] as *mut _) },
            index: 0,
        }
    }

    pub fn new_uninitialized(buffer: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            buffer,
            index: 0,
            initialized: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr() as *mut _
    }

    pub fn initialized_len(&self) -> usize {
        self.initialized
    }

    pub fn written(&self) -> &[u8] {
        assert!(self.index <= self.initialized);

        // Safety: All bytes in the returned slice are initialized.
        unsafe { &*(&self.buffer[..self.index] as *const _ as *const [u8]) }
    }

    /// Convenient method for `.writen().len()`
    pub fn written_len(&self) -> usize {
        self.index
    }

    /// Buffer has no spare space to write any data
    pub fn has_no_spare_space(&self) -> bool {
        self.index == self.buffer.len()
    }

    /// Initialize all uninitialized, unwritten part to initialized, unwritten part
    /// Return all unwritten part
    pub fn initialize_unwritten(&mut self) -> &mut [u8] {
        self.buffer[self.initialized..]
            .iter_mut()
            .for_each(|maybe_uninit| {
                maybe_uninit.write(0);
            });
        self.initialized = self.buffer.len();

        unsafe { &mut *(&mut self.buffer[self.index..] as *mut _ as *mut [u8]) }
    }

    /// Advance written index within initialized part.
    ///
    /// # Panics
    ///
    /// Panics if `amount` exceeds the number of initialized, unwritten bytes.
    pub fn advance(&mut self, amount: usize) {
        // Check the remaining lengths to avoid overflowing `self.index + amount`.
        assert!(amount <= self.buffer.len() - self.index);
        assert!(amount <= self.initialized - self.index);

        self.index += amount;
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Returns a mutable reference to the unwritten part of the buffer without
    /// ensuring that it has been fully initialized.
    ///
    /// # Safety
    ///
    /// The caller must not de-initialize portions of the buffer that have already
    /// been initialized.
    ///
    /// This includes any bytes in the region returned by this function.
    pub unsafe fn unwritten_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buffer[self.index..]
    }

    /// Asserts that the first `n` unwritten bytes of the buffer are initialized,
    /// starting at [`WriteBuffer::written_len`].
    ///
    /// [`WriteBuffer`] assumes that bytes are never de-initialized, so this method
    /// does nothing when called with fewer bytes than are already known to be initialized.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds the number of unwritten bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unwritten bytes of the buffer have already been initialized.
    pub unsafe fn assume_init(&mut self, n: usize) {
        // Check the remaining length to avoid overflowing `self.index + n`.
        assert!(n <= self.buffer.len() - self.index);

        self.initialized = self.initialized.max(self.index + n);
    }

    /// Convenient function combining [`WriteBuffer::assume_init`] and [`WriteBuffer::advance`].
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds the number of unwritten bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unwritten bytes of the buffer have already been initialized.
    pub unsafe fn assume_init_and_advance(&mut self, n: usize) {
        assert!(n <= self.buffer.len() - self.index);

        self.index += n;
        self.initialized = self.initialized.max(self.index);
    }

    /// Convenient function combining [`WriteBuffer::assume_init`] and [`WriteBuffer::advance`],
    /// works similar to [`Vec::set_len`].
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds the buffer's capacity.
    ///
    /// # Safety
    ///
    /// The caller must ensure that first `n` bytes of the buffer have already been initialized.
    pub unsafe fn set_written_and_initialized_len(&mut self, n: usize) {
        assert!(n <= self.buffer.len());

        self.index = n;
        self.initialized = self.initialized.max(n);
    }

    pub fn copy_unwritten_from<C: AsRef<[u8]>>(&mut self, other: &mut PartialBuffer<C>) -> usize {
        fn inner(this: &mut WriteBuffer<'_>, input: &[u8]) -> usize {
            // Safety: We will never ever write uninitialized bytes into it
            let out = unsafe { this.unwritten_mut() };

            let len = out.len().min(input.len());

            out[..len]
                .iter_mut()
                .zip(&input[..len])
                .for_each(|(maybe_uninit, byte)| {
                    maybe_uninit.write(*byte);
                });

            // Safety: We have written `len` bytes of initialized data into it
            unsafe { this.assume_init_and_advance(len) };
            len
        }

        let len = inner(self, other.unwritten());
        other.advance(len);

        len
    }
}

#[cfg(test)]
mod tests {
    use super::{PartialBuffer, WriteBuffer};
    use std::{
        mem::MaybeUninit,
        panic::{catch_unwind, AssertUnwindSafe},
    };

    #[test]
    fn advance_within_initialized_buffer() {
        let mut storage = [1, 2, 3, 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);

        output.advance(0);
        assert!(output.written().is_empty());
        output.advance(2);
        assert_eq!(output.written(), &[1, 2]);
        output.advance(2);
        assert_eq!(output.written(), &[1, 2, 3, 4]);
        output.advance(0);
        assert_eq!(output.written(), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn advance_into_uninitialized_buffer_panics() {
        let mut allocation = Vec::<u8>::with_capacity(8);
        let mut output = WriteBuffer::new_uninitialized(&mut allocation.spare_capacity_mut()[..8]);

        output.advance(1);
    }

    #[test]
    fn advance_past_initialized_preserves_written_data() {
        let mut storage = [MaybeUninit::uninit(); 4];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);
        output.copy_unwritten_from(&mut PartialBuffer::new(&[1, 2][..]));
        output.reset();
        output.advance(1);

        let result = catch_unwind(AssertUnwindSafe(|| output.advance(2)));

        assert!(result.is_err());
        assert_eq!(output.written(), &[1]);
        output.advance(1);
        assert_eq!(output.written(), &[1, 2]);
    }

    #[test]
    #[should_panic]
    fn advance_past_capacity_panics() {
        let mut storage = [0; 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);
        output.advance(4);

        output.advance(1);
    }

    #[test]
    #[should_panic]
    fn advance_overflow_panics() {
        let mut storage = [0; 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);
        output.advance(1);

        output.advance(usize::MAX);
    }

    #[test]
    fn assume_init_is_not_additive() {
        let mut storage = [MaybeUninit::new(1); 6];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);

        for n in [4, 2, 4, 0] {
            // Safety: All bytes in storage are initialized.
            unsafe { output.assume_init(n) };
            assert_eq!(output.initialized_len(), 4);
            assert_eq!(output.written_len(), 0);
        }
    }

    #[test]
    fn assume_init_starts_at_written_len() {
        let mut storage = [MaybeUninit::new(1); 6];
        let mut output = WriteBuffer::new_uninitialized(&mut storage);

        // Safety: All bytes in storage are initialized.
        unsafe { output.assume_init(4) };
        output.advance(2);
        // Safety: All bytes in storage are initialized.
        unsafe { output.assume_init(3) };
        assert_eq!(output.initialized_len(), 5);
        assert_eq!(output.written_len(), 2);
    }

    #[test]
    #[should_panic]
    fn assume_init_overflow_panics() {
        let mut storage = [1; 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);
        output.advance(1);

        // Safety: Initialized storage; the invalid length must panic before use.
        unsafe { output.assume_init(usize::MAX) };
    }

    #[test]
    #[should_panic]
    fn assume_init_and_advance_overflow_panics() {
        let mut storage = [1; 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);
        output.advance(1);

        // Safety: Initialized storage; the invalid length must panic before use.
        unsafe { output.assume_init_and_advance(usize::MAX) };
    }

    #[test]
    #[should_panic]
    fn set_written_and_initialized_len_past_capacity_panics() {
        let mut storage = [1; 4];
        let mut output = WriteBuffer::new_initialized(&mut storage);

        // Safety: Initialized storage; the invalid length must panic before use.
        unsafe { output.set_written_and_initialized_len(5) };
    }
}
