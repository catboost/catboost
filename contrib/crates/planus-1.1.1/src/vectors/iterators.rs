use core::num::NonZeroUsize;

use super::Vector;
use crate::VectorRead;

fn div_ceil(lhs: usize, rhs: usize) -> usize {
    let d = lhs / rhs;
    let r = lhs % rhs;
    if r > 0 && rhs > 0 {
        d + 1
    } else {
        d
    }
}

/// An iterator over the elements of a `Vector`.
///
/// This struct is created by the [`iter`][`Vector::iter`] method on [`Vector`]s.
pub struct Iter<'buf, T> {
    v: Vector<'buf, T>,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for Iter<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Iter").field("v", &self.v).finish()
    }
}

impl<T> Clone for Iter<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { v: self.v }
    }
}

impl<'buf, T: VectorRead<'buf>> Iter<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>) -> Self {
        Self { v }
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for Iter<'buf, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if let Some((first, remaining)) = self.v.split_first() {
            self.v = remaining;
            Some(first)
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.v = self.v.get(n..).unwrap_or_else(|| Vector::new_empty());
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.v.last()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for Iter<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((last, remaining)) = self.v.split_last() {
            self.v = remaining;
            Some(last)
        } else {
            None
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.v = unsafe { self.v.get_unchecked(..self.v.len().saturating_sub(n)) };
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for Iter<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        self.v.len()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for Iter<'buf, T> {}

/// An iterator over a [`Vector`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the [`Vector`].
///
/// When the [`Vector`] len is not evenly divided by the chunk size, the last
/// [`Vector`] of the iteration will be the remainder.
///
/// This struct is created by the [`chunks`][`Vector::chunks`] method on [`Vector`]s.
pub struct Chunks<'buf, T> {
    v: Vector<'buf, T>,
    chunk_size: NonZeroUsize,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for Chunks<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chunks")
            .field("v", &self.v)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl<T> Clone for Chunks<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            v: self.v,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> Chunks<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>, chunk_size: NonZeroUsize) -> Self {
        Self { v, chunk_size }
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for Chunks<'buf, T> {
    type Item = Vector<'buf, T>;

    #[inline]
    fn next(&mut self) -> Option<Vector<'buf, T>> {
        if self.v.is_empty() {
            None
        } else if let Some((first, remaining)) = self.v.split_at(self.chunk_size.get()) {
            self.v = remaining;
            Some(first)
        } else {
            Some(core::mem::replace(&mut self.v, Vector::new_empty()))
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size.get());
        self.v = self.v.get(skip..).unwrap_or_else(|| Vector::new_empty());
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for Chunks<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let split_point = (self.v.len() - 1) / self.chunk_size * self.chunk_size.get();
            let (remaining, last) = unsafe { self.v.split_at_unchecked(split_point) };
            self.v = remaining;
            Some(last)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n > 0 {
            // This will be strictly lower than the len
            let new_chunk_count = self.len().saturating_sub(n);
            // Note that all of the remaining chunks will be full chunks
            // This makes it easy to calculate the new size
            self.v = unsafe {
                self.v
                    .get_unchecked(..new_chunk_count * self.chunk_size.get())
            };
        }
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for Chunks<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        let len = self.v.len();
        div_ceil(len, self.chunk_size.get())
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for Chunks<'buf, T> {}

/// An iterator over a [`Vector`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the end of the [`Vector`].
///
/// When the [`Vector`] len is not evenly divided by the chunk size, the last [`Vector`]
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks`][`Vector::rchunks`] method on [`Vector`]s.
pub struct RChunks<'buf, T> {
    v: Vector<'buf, T>,
    chunk_size: NonZeroUsize,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for RChunks<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RChunks")
            .field("v", &self.v)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl<T> Clone for RChunks<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            v: self.v,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> RChunks<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>, chunk_size: NonZeroUsize) -> Self {
        Self { v, chunk_size }
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for RChunks<'buf, T> {
    type Item = Vector<'buf, T>;

    #[inline]
    fn next(&mut self) -> Option<Vector<'buf, T>> {
        if self.v.is_empty() {
            None
        } else {
            let (remaining, cur) = unsafe {
                self.v
                    .split_at_unchecked(self.v.len().saturating_sub(self.chunk_size.get()))
            };
            self.v = remaining;
            Some(cur)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.v = unsafe {
            self.v.get_unchecked(
                ..self
                    .v
                    .len()
                    .saturating_sub(n.saturating_mul(self.chunk_size.get())),
            )
        };
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for RChunks<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let mid = ((self.v.len() - 1) % self.chunk_size) + 1;
            let (start, remaining) = unsafe { self.v.split_at_unchecked(mid) };
            self.v = remaining;
            Some(start)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n > 0 {
            // This will be strictly lower than the len
            let new_chunk_count = self.len().saturating_sub(n);
            // Note that all of the remaining chunks will be full chunks
            // This makes it easy to calculate the new size
            self.v = unsafe {
                self.v
                    .get_unchecked(self.v.len() - (new_chunk_count * self.chunk_size.get())..)
            };
        }
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for RChunks<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        div_ceil(self.v.len(), self.chunk_size.get())
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for RChunks<'buf, T> {}

/// An iterator over a [`Vector`] in (non-overlapping) chunks (`chunk_size` elements
/// at a time), starting at the beginning of the slice.
///
/// When the [`Vector`] len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact`] method on [`Vector`]s.
///
/// [`chunks_exact`]: Vector::chunks_exact
/// [`remainder`]: ChunksExact::remainder
pub struct ChunksExact<'buf, T> {
    v: Vector<'buf, T>,
    rem: Vector<'buf, T>,
    chunk_size: NonZeroUsize,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for ChunksExact<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChunksExact")
            .field("v", &self.v)
            .field("rem", &self.rem)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl<T> Clone for ChunksExact<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            v: self.v,
            rem: self.rem,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> ChunksExact<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>, chunk_size: NonZeroUsize) -> Self {
        let len = v.len() / chunk_size.get() * chunk_size.get();
        let (v, rem) = unsafe { v.split_at_unchecked(len) };
        Self { v, rem, chunk_size }
    }

    /// Returns the remainder of the original vector that is not going to be
    /// returned by the iterator. The returned vector has at most `chunk_size-1`
    /// elements.
    #[inline]
    #[must_use]
    pub fn remainder(&self) -> Vector<'buf, T> {
        self.rem
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for ChunksExact<'buf, T> {
    type Item = Vector<'buf, T>;

    #[inline]
    fn next(&mut self) -> Option<Vector<'buf, T>> {
        debug_assert!(self.v.len() % self.chunk_size == 0);
        if self.v.is_empty() {
            None
        } else {
            let (first, remaining) = unsafe { self.v.split_at_unchecked(self.chunk_size.get()) };
            self.v = remaining;
            Some(first)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size.get());
        self.v = self.v.get(skip..).unwrap_or_else(|| Vector::new_empty());
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for ChunksExact<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.v.len() % self.chunk_size == 0);
        if self.v.is_empty() {
            None
        } else {
            let (remaining, last) = unsafe {
                self.v
                    .split_at_unchecked(self.v.len() - self.chunk_size.get())
            };
            self.v = remaining;
            Some(last)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.v = unsafe {
            self.v.get_unchecked(
                ..self
                    .v
                    .len()
                    .saturating_sub(n.saturating_mul(self.chunk_size.get())),
            )
        };
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for ChunksExact<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        self.v.len() / self.chunk_size
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for ChunksExact<'buf, T> {}

/// An iterator over a [`Vector`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the [`Vector`] len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact`] method on [`Vector`]s.
///
/// [`remainder`]: RChunksExact::remainder
/// [`rchunks_exact`]: Vector::rchunks_exact
pub struct RChunksExact<'buf, T> {
    v: Vector<'buf, T>,
    rem: Vector<'buf, T>,
    chunk_size: NonZeroUsize,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for RChunksExact<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RChunksExact")
            .field("v", &self.v)
            .field("rem", &self.rem)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl<T> Clone for RChunksExact<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            v: self.v,
            rem: self.rem,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> RChunksExact<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>, chunk_size: NonZeroUsize) -> Self {
        let rem_size = v.len() % chunk_size;
        let (rem, v) = unsafe { v.split_at_unchecked(rem_size) };
        Self { v, rem, chunk_size }
    }

    /// Returns the remainder of the original vector that is not going to be
    /// returned by the iterator. The returned vector has at most `chunk_size-1`
    /// elements.
    #[inline]
    #[must_use]
    pub fn remainder(&self) -> Vector<'buf, T> {
        self.rem
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for RChunksExact<'buf, T> {
    type Item = Vector<'buf, T>;

    #[inline]
    fn next(&mut self) -> Option<Vector<'buf, T>> {
        debug_assert!(self.v.len() % self.chunk_size == 0);
        if self.v.is_empty() {
            None
        } else {
            let (remaining, last) = unsafe {
                self.v
                    .split_at_unchecked(self.v.len() - self.chunk_size.get())
            };
            self.v = remaining;
            Some(last)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.v = unsafe {
            self.v.get_unchecked(
                ..self
                    .v
                    .len()
                    .saturating_sub(n.saturating_mul(self.chunk_size.get())),
            )
        };
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for RChunksExact<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.v.len() % self.chunk_size == 0);
        if self.v.is_empty() {
            None
        } else {
            let (first, remaining) = unsafe { self.v.split_at_unchecked(self.chunk_size.get()) };
            self.v = remaining;
            Some(first)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size.get());
        self.v = self.v.get(skip..).unwrap_or_else(|| Vector::new_empty());
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for RChunksExact<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        self.v.len() / self.chunk_size
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for RChunksExact<'buf, T> {}

/// An iterator over overlapping sub-vectors of length `size`.
///
/// This struct is created by the [`windows`] method on [`Vector`]s.
///
/// [`windows`]: Vector::windows
pub struct Windows<'buf, T> {
    v: Vector<'buf, T>,
    size: NonZeroUsize,
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for Windows<'buf, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Windows")
            .field("v", &self.v)
            .field("size", &self.size)
            .finish()
    }
}

impl<T> Clone for Windows<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            v: self.v,
            size: self.size,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> Windows<'buf, T> {
    #[inline]
    pub(crate) fn new(v: Vector<'buf, T>, size: NonZeroUsize) -> Self {
        Self { v, size }
    }
}

impl<'buf, T: VectorRead<'buf>> Iterator for Windows<'buf, T> {
    type Item = Vector<'buf, T>;

    #[inline]
    fn next(&mut self) -> Option<Vector<'buf, T>> {
        let window = self.v.get(..self.size.get())?;
        self.v = unsafe { self.v.get_unchecked(1..) };
        Some(window)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.v = self.v.get(n..).unwrap_or_else(|| Vector::new_empty());
        self.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::DoubleEndedIterator for Windows<'buf, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = self.v.len().checked_sub(self.size.get())?;
        let window = unsafe { self.v.get_unchecked(index..) };
        self.v = unsafe { self.v.get_unchecked(..self.v.len() - 1) };
        Some(window)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.v = unsafe { self.v.get_unchecked(..self.v.len().saturating_sub(n)) };
        self.next_back()
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::ExactSizeIterator for Windows<'buf, T> {
    #[inline]
    fn len(&self) -> usize {
        self.v.len().saturating_sub(self.size.get() - 1)
    }
}

impl<'buf, T: VectorRead<'buf>> core::iter::FusedIterator for Windows<'buf, T> {}
