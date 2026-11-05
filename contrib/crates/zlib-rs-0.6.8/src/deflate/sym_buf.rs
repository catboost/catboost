// taken from https://docs.rs/tokio/latest/src/tokio/io/read_buf.rs.html#23-27
// based on https://rust-lang.github.io/rfcs/2930-read-buf.html

use crate::weak_slice::WeakSliceMut;

pub(crate) struct SymBuf<'a> {
    buf: WeakSliceMut<'a, u8>,
    filled: usize,
}

impl<'a> SymBuf<'a> {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (u16, u8)> + '_ {
        self.buf.as_slice()[..self.filled]
            .chunks_exact(3)
            .map(|chunk| match *chunk {
                [dist_low, dist_high, lc] => (u16::from_le_bytes([dist_low, dist_high]), lc),
                _ => unreachable!("chunks are exactly 3 elements wide"),
            })
    }

    #[inline]
    pub fn should_flush_block(&self) -> bool {
        self.filled == self.buf.len() - 3
    }

    /// Returns true if there are no bytes in this ReadBuf
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) {
        self.buf.as_mut_slice().fill(0);
        self.filled = 0;
    }

    #[inline(always)]
    pub fn push_lit(&mut self, byte: u8) {
        // NOTE: we rely on the buffer being zeroed here!
        self.buf.as_mut_slice()[self.filled + 2] = byte;

        self.filled += 3;
    }

    #[inline(always)]
    pub fn push_dist(&mut self, dist: u16, len: u8) {
        let buf = &mut self.buf.as_mut_slice()[self.filled..][..3];

        // Write both dist bytes as a single 2-byte store. This avoids the
        // `movb %ch, [mem]` instruction pattern (store from high-byte register
        // alias) that LLVM otherwise emits when dist arrives as a wide register.
        // That pattern triggers the Intel Raptor Lake CPU errata, causing silent
        // 2-byte stores that corrupt the adjacent `len` byte.
        // SAFETY: buf has at least 3 bytes (guaranteed above), so the unaligned
        // 2-byte write at offset 0 stays within bounds.
        unsafe {
            core::ptr::write_unaligned(buf.as_mut_ptr().cast::<[u8; 2]>(), dist.to_le_bytes())
        };
        buf[2] = len;

        self.filled += 3;
    }

    pub(crate) unsafe fn from_raw_parts(ptr: *mut u8, lit_bufsize: usize) -> Self {
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, lit_bufsize * 3) },
            filled: 0,
        }
    }

    pub(crate) unsafe fn clone_to(&self, ptr: *mut u8) -> Self {
        unsafe { ptr.copy_from_nonoverlapping(self.buf.as_ptr(), self.buf.len()) };
        Self {
            buf: unsafe { WeakSliceMut::from_raw_parts_mut(ptr, self.buf.len()) },
            filled: self.filled,
        }
    }
}
