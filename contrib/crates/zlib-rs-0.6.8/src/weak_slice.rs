use core::marker::PhantomData;

/// a mutable "slice" (bundle of pointer and length). The main goal of this type is passing MIRI
/// with stacked borrows. In particular, storing a standard slice in data structures violates the
/// stacked borrows rule when that slice is deallocated. By only materializing the slice when
/// needed for data access, hence bounding the lifetime more tightly, this restriction is circumvented.
#[derive(Debug)]
pub(crate) struct WeakSliceMut<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> WeakSliceMut<'a, T> {
    /// # Safety
    ///
    /// The arguments must satisfy the requirements of [`core::slice::from_raw_parts_mut`]. The
    /// difference versus a slice is that the slice requirements are only enforced when a slice is
    /// needed, so in practice we mostly get the bounds checking and other convenient slice APIs,
    /// without the exact correctness constraints of a rust core/std slice.
    pub(crate) unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_raw_parts(self) -> (*mut T, usize) {
        (self.ptr, self.len)
    }

    pub(crate) fn as_slice(&self) -> &'a [T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn empty() -> Self {
        let buf = &mut [];
        Self {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct WeakArrayMut<'a, T, const N: usize> {
    ptr: *mut [T; N],
    _marker: PhantomData<&'a mut [T; N]>,
}

impl<'a, T, const N: usize> WeakArrayMut<'a, T, N> {
    /// # Safety
    ///
    /// The pointer must be [convertable to a reference](https://doc.rust-lang.org/std/ptr/index.html#pointer-to-reference-conversion).
    pub(crate) unsafe fn from_ptr(ptr: *mut [T; N]) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    pub(crate) fn as_slice(&self) -> &'a [T] {
        unsafe { core::slice::from_raw_parts(self.ptr.cast(), N) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.cast(), N) }
    }

    pub(crate) fn as_ptr(&self) -> *const [T; N] {
        self.ptr
    }
}
