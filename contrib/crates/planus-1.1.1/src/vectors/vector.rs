use core::{marker::PhantomData, num::NonZeroUsize};

use crate::{
    errors::{self, ErrorKind},
    impls::array_from_buffer,
    slice_helpers::SliceWithStartOffset,
    traits::VectorRead,
    TableRead,
};

/// A [`slice`]-like view into a serialized flatbuffer that deserializes on demand.
pub struct Vector<'buf, T: ?Sized> {
    buffer: SliceWithStartOffset<'buf>,
    len: usize,
    _marker: PhantomData<&'buf T>,
}

impl<T: ?Sized> Copy for Vector<'_, T> {}
impl<T: ?Sized> Clone for Vector<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'buf, T: ?Sized> Vector<'buf, T> {
    /// Returns an empty `Vector`
    ///
    /// This is typically not very useful, since the vector is read-only, but
    /// has uses for instance as a default value.
    #[inline]
    #[must_use]
    pub const fn new_empty() -> Vector<'buf, T> {
        Self {
            buffer: SliceWithStartOffset {
                buffer: &[],
                offset_from_start: 0,
            },
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Checks if the vector is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements in the vector.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.len
    }
}

impl<'buf, T: VectorRead<'buf>> Vector<'buf, T> {
    /// Returns the first element of the `Vector`, or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn first(self) -> Option<T> {
        self.get(0)
    }

    /// Returns the last element of the `Vector`, or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn last(self) -> Option<T> {
        self.get(self.len().checked_sub(1)?)
    }

    /// Returns an element or sub-vector depending on the type of
    /// index.
    ///
    /// - If given a position, returns the element at that
    ///   position or `None` if out of bounds.
    /// - If given a range, returns the sub-vector corresponding to that range,
    ///   or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get<I>(self, index: I) -> Option<I::Output>
    where
        I: VectorIndex<'buf, T>,
    {
        index.get(self)
    }

    /// Returns an element or sub-vector, without doing bounds checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting output is not used.
    ///
    /// [`get`]: Vector::get
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked<I>(self, index: I) -> I::Output
    where
        I: VectorIndex<'buf, T>,
    {
        index.get_unchecked(self)
    }

    /// Returns an iterator over the vector.
    #[inline]
    #[must_use]
    pub fn iter(self) -> super::Iter<'buf, T> {
        super::Iter::new(self)
    }

    /// Returns an iterator over `chunk_size` elements of the [`Vector`] at a time, starting at the
    /// beginning of the vector.
    ///
    /// The chunks are [`Vector`]s themselves and do not overlap. If `chunk_size` does not
    /// divide the length of the [`Vector`], then the last chunk will not have length `chunk_size`.
    ///
    /// See [`chunks_exact`] for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements, and [`rchunks`] for the same iterator but starting at the end of the
    /// vector.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`chunks_exact`]: Vector::chunks_exact
    /// [`rchunks`]: Vector::rchunks
    #[inline]
    #[must_use]
    pub fn chunks(self, chunk_size: usize) -> super::Chunks<'buf, T> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("chunks cannot have a size of zero");
        super::Chunks::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the [`Vector`] at a time, starting at the end
    /// of the vector.
    ///
    /// The chunks are [`Vector`]s themselves and do not overlap. If `chunk_size` does not
    /// divide the length of the [`Vector`], then the last chunk will not have length `chunk_size`.
    ///
    /// See [`rchunks_exact`] for a variant of this iterator that returns chunks of always exactly
    /// `chunk_size` elements, and [`chunks`] for the same iterator but starting at the beginning
    /// of the vector.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`rchunks_exact`]: Vector::rchunks_exact
    /// [`chunks`]: Vector::chunks
    #[inline]
    #[must_use]
    pub fn rchunks(self, chunk_size: usize) -> super::RChunks<'buf, T> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("chunks cannot have a size of zero");
        super::RChunks::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the [`Vector`] at a time, starting at the
    /// beginning of the vector.
    ///
    /// The chunks are [`Vector`]s themselves and do not overlap. If `chunk_size` does not
    /// divide the length of the vector, then the last up to `chunk_size-1` elements will
    /// be omitted and can be retrieved from the `remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`chunks`].
    ///
    /// See [`chunks`] for a variant of this iterator that also returns the remainder as a smaller
    /// chunk, and [`rchunks_exact`] for the same iterator but starting at the end of the vector.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`chunks`]: Vector::chunks
    /// [`rchunks_exact`]: Vector::rchunks_exact
    #[inline]
    #[must_use]
    pub fn chunks_exact(self, chunk_size: usize) -> super::ChunksExact<'buf, T> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("chunks cannot have a size of zero");
        super::ChunksExact::new(self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the [`Vector`] at a time, starting at the
    /// end of the vector.
    ///
    /// The chunks are [`Vector`]s themselves and do not overlap. If `chunk_size` does not
    /// divide the length of the vector, then the last up to `chunk_size-1` elements will
    /// be omitted and can be retrieved from the `remainder` function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can often optimize the
    /// resulting code better than in the case of [`rchunks`].
    ///
    /// See [`rchunks`] for a variant of this iterator that also returns the remainder as a smaller
    /// chunk, and [`chunks_exact`] for the same iterator but starting at the beginning of the
    /// vector.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`rchunks`]: Vector::rchunks
    /// [`chunks_exact`]: Vector::chunks_exact
    #[inline]
    #[must_use]
    pub fn rchunks_exact(self, chunk_size: usize) -> super::RChunksExact<'buf, T> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("chunks cannot have a size of zero");
        super::RChunksExact::new(self, chunk_size)
    }

    /// Returns an iterator over all contiguous windows of length
    /// `size`. The windows overlap. If the vector is shorter than
    /// `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[inline]
    #[must_use]
    pub fn windows(self, size: usize) -> super::Windows<'buf, T> {
        let size = NonZeroUsize::new(size).expect("windows cannot have a size of zero");
        super::Windows::new(self, size)
    }

    /// Returns the first and all the rest of the elements of the `Vector`, or `None` if it is empty
    #[inline]
    #[must_use]
    pub fn split_first(self) -> Option<(T, Vector<'buf, T>)> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { (self.get_unchecked(0), self.get_unchecked(1..)) })
        }
    }

    /// Returns the last and all the rest of the elements of the `Vector`, or `None` if it is empty
    #[inline]
    #[must_use]
    pub fn split_last(self) -> Option<(T, Vector<'buf, T>)> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe {
                (
                    self.get_unchecked(self.len - 1),
                    self.get_unchecked(..self.len - 1),
                )
            })
        }
    }

    /// Divides one `Vector` into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    #[inline]
    #[must_use]
    pub fn split_at(self, mid: usize) -> Option<(Vector<'buf, T>, Vector<'buf, T>)> {
        if mid <= self.len {
            Some(unsafe { self.split_at_unchecked(mid) })
        } else {
            None
        }
    }

    /// Divides one [`Vector`] into two at an index, without doing bounds checking.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// For a safe alternative see [`split_at`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting output is not used. The caller has to ensure that
    /// `0 <= mid <= self.len()`.
    ///
    /// [`split_at`]: Vector::split_at
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    #[must_use]
    pub unsafe fn split_at_unchecked(self, mid: usize) -> (Vector<'buf, T>, Vector<'buf, T>) {
        (self.get_unchecked(..mid), self.get_unchecked(mid..))
    }
}

impl<'buf, T: VectorRead<'buf>> Vector<'buf, T> {
    /// Copies self into a new `Vec`.
    pub fn to_vec<O>(self) -> crate::Result<alloc::vec::Vec<O>>
    where
        O: core::convert::TryFrom<T>,
        crate::errors::Error: From<O::Error>,
    {
        self.iter()
            .map(|v| O::try_from(v).map_err(crate::errors::Error::from))
            .collect()
    }
}

impl<'buf, T, E> Vector<'buf, core::result::Result<T, E>> {
    /// Copies self into a new `Vec`.
    pub fn to_vec_result<O>(self) -> crate::Result<alloc::vec::Vec<O>>
    where
        T: crate::traits::VectorReadInner<'buf>,
        E: core::convert::From<T::Error>,
        O: core::convert::TryFrom<T>,
        crate::errors::Error: From<E> + From<O::Error>,
    {
        self.iter()
            .map(|v| O::try_from(v?).map_err(|e| e.into()))
            .collect()
    }
}

impl<'buf, T: VectorRead<'buf>> IntoIterator for Vector<'buf, T> {
    type Item = T;
    type IntoIter = super::Iter<'buf, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'buf, T: VectorRead<'buf> + core::fmt::Debug> core::fmt::Debug for Vector<'buf, T>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(*self).finish()
    }
}

/// A helper trait used for indexing operations.
pub trait VectorIndex<'buf, T: VectorRead<'buf>>: private::Sealed {
    /// The output type returned by methods.
    type Output;

    /// Returns a the output at this location, if in bounds.
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output>;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting output is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output;
}

/// Convert pair of `ops::Bound`s into `core::ops::Range` without performing any bounds checking and (in debug) overflow checking
fn into_range_unchecked(
    len: usize,
    (start, end): (core::ops::Bound<usize>, core::ops::Bound<usize>),
) -> core::ops::Range<usize> {
    use core::ops::Bound;
    let start = match start {
        Bound::Included(i) => i,
        Bound::Excluded(i) => i + 1,
        Bound::Unbounded => 0,
    };
    let end = match end {
        Bound::Included(i) => i + 1,
        Bound::Excluded(i) => i,
        Bound::Unbounded => len,
    };
    start..end
}

/// Convert pair of `core::ops::Bound`s into `core::ops::Range`.
/// Returns `None` on overflowing indices.
fn into_range(
    len: usize,
    (start, end): (core::ops::Bound<usize>, core::ops::Bound<usize>),
) -> Option<core::ops::Range<usize>> {
    use core::ops::Bound;
    let start = match start {
        Bound::Included(start) => start,
        Bound::Excluded(start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };

    let end = match end {
        Bound::Included(end) => end.checked_add(1)?,
        Bound::Excluded(end) => end,
        Bound::Unbounded => len,
    };

    // Don't bother with checking `start < end` and `end <= len`
    // since these checks are handled by `Range` impls

    Some(start..end)
}
impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T>
    for (core::ops::Bound<usize>, core::ops::Bound<usize>)
{
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        into_range(vector.len, self)?.get(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        into_range_unchecked(vector.len, self).get_unchecked(vector)
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for usize {
    type Output = T;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        if self < vector.len {
            Some(unsafe { self.get_unchecked(vector) })
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        debug_assert!(self < vector.len);
        debug_assert!(vector.len.checked_mul(T::STRIDE).unwrap() <= vector.buffer.len());
        T::from_buffer(vector.buffer, T::STRIDE * self)
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::Range<usize> {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        if self.start > self.end || self.end > vector.len {
            None
        } else {
            // SAFETY: `self` is checked to be valid and in bounds above.
            unsafe { Some(self.get_unchecked(vector)) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        debug_assert!(self.start <= self.end);
        debug_assert!(self.end <= vector.len);
        Vector {
            buffer: vector
                .buffer
                .advance(T::STRIDE * self.start)
                .expect("IMPOSSIBLE: the length was checked on creation"),
            len: self.end - self.start,
            _marker: PhantomData,
        }
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::RangeFrom<usize> {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        (self.start..vector.len).get(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        (self.start..vector.len).get_unchecked(vector)
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::RangeFull {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        Some(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        vector
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::RangeInclusive<usize> {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        (*self.start()..self.end().checked_add(1)?).get(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        (*self.start()..self.end() + 1).get_unchecked(vector)
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::RangeTo<usize> {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        (0..self.end).get(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        (0..self.end).get_unchecked(vector)
    }
}

impl<'buf, T: VectorRead<'buf>> VectorIndex<'buf, T> for core::ops::RangeToInclusive<usize> {
    type Output = Vector<'buf, T>;

    #[inline]
    fn get(self, vector: Vector<'buf, T>) -> Option<Self::Output> {
        (0..=self.end).get(vector)
    }

    #[inline]
    unsafe fn get_unchecked(self, vector: Vector<'buf, T>) -> Self::Output {
        (0..=self.end).get_unchecked(vector)
    }
}

impl<'buf, T: VectorRead<'buf>, O> TryFrom<Vector<'buf, T>> for alloc::vec::Vec<O>
where
    O: core::convert::TryFrom<T>,
    errors::Error: From<O::Error>,
{
    type Error = crate::errors::Error;

    fn try_from(value: Vector<'buf, T>) -> Result<Self, Self::Error> {
        value
            .iter()
            .map(|v| O::try_from(v).map_err(errors::Error::from))
            .collect()
    }
}

impl<'buf, T: ?Sized + VectorRead<'buf>> TableRead<'buf> for Vector<'buf, T> {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> core::result::Result<Self, ErrorKind> {
        let (buffer, len) = array_from_buffer(buffer, offset)?;
        if len.checked_mul(T::STRIDE).ok_or(ErrorKind::InvalidLength)? <= buffer.len() {
            Ok(Vector {
                buffer,
                len,
                _marker: PhantomData,
            })
        } else {
            Err(ErrorKind::InvalidLength)
        }
    }
}

mod private {
    pub trait Sealed {}

    // Implement for those same types, but no others.
    impl Sealed for (core::ops::Bound<usize>, core::ops::Bound<usize>) {}
    impl Sealed for usize {}
    impl Sealed for core::ops::Range<usize> {}
    impl Sealed for core::ops::RangeFrom<usize> {}
    impl Sealed for core::ops::RangeFull {}
    impl Sealed for core::ops::RangeInclusive<usize> {}
    impl Sealed for core::ops::RangeTo<usize> {}
    impl Sealed for core::ops::RangeToInclusive<usize> {}
}
