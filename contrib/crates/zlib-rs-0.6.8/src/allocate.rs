#![allow(unpredictable_function_pointer_comparisons)]

#[cfg(unix)]
use core::ffi::c_int;
use core::{
    alloc::Layout,
    ffi::{c_uint, c_void},
    marker::PhantomData,
    mem,
    ptr::NonNull,
};

#[allow(non_camel_case_types)]
type size_t = usize;

const ALIGN: u8 = 64;
// posix_memalign requires that the alignment be a power of two and a multiple of sizeof(void*).
const _: () = assert!(ALIGN.count_ones() == 1);
const _: () = assert!(ALIGN as usize % mem::size_of::<*mut c_void>() == 0);

/// # Safety
///
/// This function is safe, but must have this type signature to be used elsewhere in the library
#[cfg(unix)]
unsafe extern "C" fn zalloc_c(opaque: *mut c_void, items: c_uint, size: c_uint) -> *mut c_void {
    let _ = opaque;

    extern "C" {
        fn posix_memalign(memptr: *mut *mut c_void, align: size_t, size: size_t) -> c_int;
    }

    let mut ptr = core::ptr::null_mut();
    let size = items as size_t * size as size_t;
    if size == 0 {
        return ptr;
    }
    // SAFETY: ALIGN is a power of 2 and multiple of sizeof(void*), as required by posix_memalign.
    // In addition, since posix_memalign is allowed to return a unique but non-null pointer when
    // called with a size of zero, we returned above if the size was zero.
    match unsafe { posix_memalign(&mut ptr, ALIGN.into(), size) } {
        0 => ptr,
        _ => core::ptr::null_mut(),
    }
}

/// # Safety
///
/// This function is safe, but must have this type signature to be used elsewhere in the library
#[cfg(not(unix))]
unsafe extern "C" fn zalloc_c(opaque: *mut c_void, items: c_uint, size: c_uint) -> *mut c_void {
    let _ = opaque;

    let size = items as size_t * size as size_t;
    if size == 0 {
        return core::ptr::null_mut();
    }

    extern "C" {
        fn malloc(size: size_t) -> *mut c_void;
    }

    // SAFETY: malloc is allowed to return a unique but non-null pointer when given a size
    // of zero. To prevent potentially undefined behavior from such a pointer reaching Rust code
    // and being dereferenced, we handled the zero-size case separately above.
    unsafe { malloc(size) }
}

/// # Safety
///
/// This function is safe, but must have this type signature to be used elsewhere in the library
unsafe extern "C" fn zalloc_c_calloc(
    opaque: *mut c_void,
    items: c_uint,
    size: c_uint,
) -> *mut c_void {
    let _ = opaque;

    extern "C" {
        fn calloc(nitems: size_t, size: size_t) -> *mut c_void;
    }

    if items as size_t * size as size_t == 0 {
        return core::ptr::null_mut();
    }

    // SAFETY: When the item count or size is zero, calloc is allowed to return either
    // null or some non-null value that is safe to free but not safe to dereference.
    // To avoid the possibility of exposing such a pointer to Rust code, we check for
    // zero above and avoid calling calloc.
    unsafe { calloc(items as size_t, size as size_t) }
}

/// # Safety
///
/// The `ptr` must be allocated with the allocator that is used internally by `zcfree`
unsafe extern "C" fn zfree_c(opaque: *mut c_void, ptr: *mut c_void) {
    let _ = opaque;

    extern "C" {
        fn free(p: *mut c_void);
    }

    // SAFETY: The caller ensured that ptr was obtained from the same allocator. Also,
    // free properly handles the case where ptr == NULL.
    unsafe { free(ptr) }
}

/// # Safety
///
/// This function is safe to call.
#[cfg(feature = "rust-allocator")]
unsafe extern "C" fn zalloc_rust(_opaque: *mut c_void, count: c_uint, size: c_uint) -> *mut c_void {
    let size = count as usize * size as usize;
    if size == 0 {
        return core::ptr::null_mut();
    }

    // internally, we want to align allocations to 64 bytes (in part for SIMD reasons)
    let layout = Layout::from_size_align(size, ALIGN.into()).unwrap();

    // SAFETY: alloc requires that the layout have a nonzero size, so we return null
    // above (and never reach this call) if the requested count * size is zero.
    let ptr = unsafe { std::alloc::alloc(layout) };

    ptr as *mut c_void
}

/// # Safety
///
/// This function is safe to call.
#[cfg(feature = "rust-allocator")]
unsafe extern "C" fn zalloc_rust_calloc(
    _opaque: *mut c_void,
    count: c_uint,
    size: c_uint,
) -> *mut c_void {
    let size = count as usize * size as usize;
    if size == 0 {
        return core::ptr::null_mut();
    }

    // internally, we want to align allocations to 64 bytes (in part for SIMD reasons)
    let layout = Layout::from_size_align(size, ALIGN.into()).unwrap();

    // SAFETY: alloc_zeroed requires that the layout have a nonzero size, so we return
    // null above (and never reach this call) if the requested count * size is zero.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

    ptr as *mut c_void
}

/// # Safety
///
/// - `ptr` must be allocated with the rust `alloc::alloc` allocator
/// - `opaque` is a `&usize` that represents the size of the allocation
#[cfg(feature = "rust-allocator")]
unsafe extern "C" fn zfree_rust(opaque: *mut c_void, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    // we can't really do much else. Deallocating with an invalid layout is UB.
    debug_assert!(!opaque.is_null());
    if opaque.is_null() {
        return;
    }

    // SAFETY: The caller ensured that *opaque is valid to dereference.
    let size = unsafe { *(opaque as *mut usize) };

    // zalloc_rust and zalloc_rust_calloc bypass the Rust allocator and just return
    // null when asked to allocate something of zero size. So if a caller tries to
    // free something of zero size, we return here rather than trying to call the
    // Rust deallocator.
    if size == 0 {
        return;
    }

    let layout = Layout::from_size_align(size, ALIGN.into());
    let layout = layout.unwrap();

    // SAFETY: The caller ensured that ptr was allocated with the `alloc` allocator,
    // and the size check above ensures that we are not trying to use a zero-size layout
    // that would produce undefined behavior in the allocator.
    unsafe { std::alloc::dealloc(ptr.cast(), layout) };
}

#[cfg(test)]
unsafe extern "C" fn zalloc_fail(_: *mut c_void, _: c_uint, _: c_uint) -> *mut c_void {
    core::ptr::null_mut()
}

#[cfg(test)]
unsafe extern "C" fn zfree_fail(_: *mut c_void, _: *mut c_void) {
    // do nothing
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Allocator<'a> {
    pub zalloc: crate::c_api::alloc_func,
    pub zfree: crate::c_api::free_func,
    pub opaque: crate::c_api::voidpf,
    pub _marker: PhantomData<&'a ()>,
}

unsafe impl Sync for Allocator<'static> {}

#[cfg(feature = "rust-allocator")]
pub static RUST: Allocator<'static> = Allocator {
    zalloc: zalloc_rust,
    zfree: zfree_rust,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

#[cfg(feature = "c-allocator")]
pub static C: Allocator<'static> = Allocator {
    zalloc: zalloc_c,
    zfree: zfree_c,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

#[cfg(test)]
static FAIL: Allocator<'static> = Allocator {
    zalloc: zalloc_fail,
    zfree: zfree_fail,
    opaque: core::ptr::null_mut(),
    _marker: PhantomData,
};

impl Allocator<'_> {
    fn allocate_layout(&self, layout: Layout) -> *mut c_void {
        assert!(layout.align() <= ALIGN.into());

        // Special case for the Rust `alloc` backed allocator
        #[cfg(feature = "rust-allocator")]
        if self.zalloc == RUST.zalloc {
            let ptr = unsafe { (RUST.zalloc)(self.opaque, layout.size() as _, 1) };

            debug_assert_eq!(ptr as usize % layout.align(), 0);

            return ptr;
        }

        // General case for c-style allocation

        // We cannot rely on the allocator giving properly aligned allocations and have to fix that ourselves.
        //
        // The general approach is to allocate a bit more than the layout needs, so that we can
        // give the application a properly aligned address and also store the real allocation
        // pointer in the allocation so that `free` can free the real allocation pointer.
        //
        //
        // Example: The layout represents `(u32, u32)`, with an alignment of 4 bytes and a
        // total size of 8 bytes.
        //
        // Assume that the allocator will give us address `0x07`. We need that to be a multiple
        // of the alignment, so that shifts the starting position to `0x08`. Then we also need
        // to store the pointer to the start of the allocation so that `free` can free that
        // pointer, bumping to `0x10`. The `0x10` pointer is then the pointer that the application
        // deals with. When free'ing, the original allocation pointer can be read from `0x10 - size_of::<*const c_void>()`.
        //
        // Of course there does need to be enough space in the allocation such that when we
        // shift the start forwards, the end is still within the allocation. Hence we allocate
        // `extra_space` bytes: enough for a full alignment plus a pointer.

        // we need at least
        //
        // - `align` extra space so that no matter what pointer we get from zalloc, we can shift the start of the
        //      allocation by at most `align - 1` so that `ptr as usize % align == 0
        // - `size_of::<*mut _>` extra space so that after aligning to `align`,
        //      there is `size_of::<*mut _>` space to store the pointer to the allocation.
        //      This pointer is then retrieved in `free`
        let extra_space = core::mem::size_of::<*mut c_void>() + layout.align();

        // Safety: we assume allocating works correctly in the safety assumptions on
        // `DeflateStream` and `InflateStream`.
        let Ok(allocation_size) = c_uint::try_from(layout.size() + extra_space) else {
            return core::ptr::null_mut();
        };
        let ptr = unsafe { (self.zalloc)(self.opaque, allocation_size, 1) };

        if ptr.is_null() {
            return ptr;
        }

        // Calculate return pointer address with space enough to store original pointer
        let align_diff = (ptr as usize).next_multiple_of(layout.align()) - (ptr as usize);

        // Safety: offset is smaller than 64, and we allocated 64 extra bytes in the allocation
        let mut return_ptr = unsafe { ptr.cast::<u8>().add(align_diff) };

        // if there is not enough space to store a pointer we need to make more
        if align_diff < core::mem::size_of::<*mut c_void>() {
            // # Safety
            //
            // - `return_ptr` is well-aligned, therefore `return_ptr + align` is also well-aligned
            // - we reserve `size_of::<*mut _> + align` extra space in the allocation, so
            //      `ptr + align_diff + align` is still valid for (at least) `layout.size` bytes
            let offset = Ord::max(core::mem::size_of::<*mut c_void>(), layout.align());
            return_ptr = unsafe { return_ptr.add(offset) };
        }

        // Store the original pointer for free()
        //
        // Safety: `align >= size_of::<*mut _>`, so there is now space for a pointer before `return_ptr`
        // in the allocation
        unsafe {
            let original_ptr = return_ptr.sub(core::mem::size_of::<*mut c_void>());
            core::ptr::write_unaligned(original_ptr.cast::<*mut c_void>(), ptr);
        };

        // Return properly aligned pointer in allocation
        let ptr = return_ptr.cast::<c_void>();

        debug_assert_eq!(ptr as usize % layout.align(), 0);

        ptr
    }

    fn allocate_layout_zeroed(&self, layout: Layout) -> *mut c_void {
        assert!(layout.align() <= ALIGN.into());

        #[cfg(feature = "rust-allocator")]
        if self.zalloc == RUST.zalloc {
            let ptr = unsafe { zalloc_rust_calloc(self.opaque, layout.size() as _, 1) };

            debug_assert_eq!(ptr as usize % layout.align(), 0);

            return ptr;
        }

        #[cfg(feature = "c-allocator")]
        if self.zalloc == C.zalloc {
            let alloc = Allocator {
                zalloc: zalloc_c_calloc,
                zfree: zfree_c,
                opaque: core::ptr::null_mut(),
                _marker: PhantomData,
            };

            return alloc.allocate_layout(layout);
        }

        // create the allocation (contents are uninitialized)
        let ptr = self.allocate_layout(layout);

        if !ptr.is_null() {
            // zero all contents (thus initializing the buffer)
            unsafe { core::ptr::write_bytes(ptr, 0u8, layout.size()) };
        }

        ptr
    }

    pub fn allocate_raw<T>(&self) -> Option<NonNull<T>> {
        NonNull::new(self.allocate_layout(Layout::new::<T>()).cast())
    }

    pub fn allocate_slice_raw<T>(&self, len: usize) -> Option<NonNull<T>> {
        NonNull::new(self.allocate_layout(Layout::array::<T>(len).ok()?).cast())
    }

    pub fn allocate_zeroed_raw<T>(&self) -> Option<NonNull<T>> {
        NonNull::new(self.allocate_layout_zeroed(Layout::new::<T>()).cast())
    }

    pub fn allocate_zeroed_buffer(&self, len: usize) -> Option<NonNull<u8>> {
        let layout = Layout::array::<u8>(len).ok()?;
        NonNull::new(self.allocate_layout_zeroed(layout).cast())
    }

    /// # Panics
    ///
    /// - when `len` is 0
    ///
    /// # Safety
    ///
    /// - `ptr` must be allocated with this allocator
    /// - `len` must be the number of `T`s that are in this allocation
    #[allow(unused)] // Rust needs `len` for deallocation
    pub unsafe fn deallocate<T>(&self, ptr: *mut T, len: usize) {
        if !ptr.is_null() {
            // Special case for the Rust `alloc` backed allocator
            #[cfg(feature = "rust-allocator")]
            if self.zfree == RUST.zfree {
                assert_ne!(len, 0, "invalid size for {ptr:?}");
                let mut size = core::mem::size_of::<T>() * len;
                // SAFETY: The caller ensured that ptr was allocated with this allocator, and
                // we initialized size above.
                return unsafe { (RUST.zfree)(&mut size as *mut usize as *mut c_void, ptr.cast()) };
            }

            // General case for c-style allocation
            // SAFETY: allocate_layout allocates extra space at the start so that *ptr is preceded
            // by a pointer holding the pointer to the actual allocation. Therefore, it is safe to
            // subtract size_of::<*const c_void> from pointer, dereference the resulting address,
            // and use that as the argument to the low-level free function.
            unsafe {
                let original_ptr = (ptr as *mut u8).sub(core::mem::size_of::<*const c_void>());
                let free_ptr = core::ptr::read_unaligned(original_ptr as *mut *mut c_void);

                (self.zfree)(self.opaque, free_ptr)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicPtr, Ordering};
    use std::ptr;
    use std::sync::Mutex;

    use super::*;

    static PTR: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    static MUTEX: Mutex<()> = Mutex::new(());

    unsafe extern "C" fn unaligned_alloc(
        _opaque: *mut c_void,
        _items: c_uint,
        _size: c_uint,
    ) -> *mut c_void {
        PTR.load(Ordering::Relaxed)
    }

    unsafe extern "C" fn unaligned_free(_opaque: *mut c_void, ptr: *mut c_void) {
        let expected = PTR.load(Ordering::Relaxed);
        assert_eq!(expected, ptr)
    }

    fn unaligned_allocator_help<T>() {
        let mut buf = [0u8; 1024];

        // we don't want anyone else messing with the PTR static
        let _guard = MUTEX.lock().unwrap();

        for i in 0..64 {
            let ptr = unsafe { buf.as_mut_ptr().add(i).cast() };
            PTR.store(ptr, Ordering::Relaxed);

            let allocator = Allocator {
                zalloc: unaligned_alloc,
                zfree: unaligned_free,
                opaque: core::ptr::null_mut(),
                _marker: PhantomData,
            };

            let ptr = allocator.allocate_raw::<T>().unwrap().as_ptr();
            assert_eq!(ptr as usize % core::mem::align_of::<T>(), 0);
            unsafe { allocator.deallocate(ptr, 1) }

            let ptr = allocator.allocate_slice_raw::<T>(10).unwrap().as_ptr();
            assert_eq!(ptr as usize % core::mem::align_of::<T>(), 0);
            unsafe { allocator.deallocate(ptr, 10) }
        }
    }

    #[test]
    fn unaligned_allocator_0() {
        unaligned_allocator_help::<()>()
    }

    #[test]
    fn unaligned_allocator_1() {
        unaligned_allocator_help::<u8>()
    }

    #[test]
    fn unaligned_allocator_2() {
        unaligned_allocator_help::<u16>()
    }
    #[test]
    fn unaligned_allocator_4() {
        unaligned_allocator_help::<u32>()
    }
    #[test]
    fn unaligned_allocator_8() {
        unaligned_allocator_help::<u64>()
    }
    #[test]
    fn unaligned_allocator_16() {
        unaligned_allocator_help::<u128>()
    }

    #[test]
    fn unaligned_allocator_32() {
        #[repr(C, align(32))]
        struct Align32(u8);

        unaligned_allocator_help::<Align32>()
    }

    #[test]
    fn unaligned_allocator_64() {
        #[repr(C, align(64))]
        struct Align64(u8);

        unaligned_allocator_help::<Align64>()
    }

    fn test_allocate_zeroed_help(allocator: Allocator) {
        #[repr(C, align(64))]
        struct Align64(u8);

        let ptr = allocator.allocate_raw::<Align64>();
        assert!(ptr.is_some());
        unsafe { allocator.deallocate(ptr.unwrap().as_ptr(), 1) };
    }

    #[test]
    fn test_allocate_zeroed() {
        #[cfg(feature = "rust-allocator")]
        test_allocate_zeroed_help(RUST);

        #[cfg(feature = "c-allocator")]
        test_allocate_zeroed_help(C);

        assert!(FAIL.allocate_raw::<u128>().is_none());
    }

    fn test_allocate_zeroed_buffer_help(allocator: Allocator) {
        let len = 42;
        let Some(buf) = allocator.allocate_zeroed_buffer(len) else {
            return;
        };

        let slice = unsafe { core::slice::from_raw_parts_mut(buf.as_ptr(), len) };

        assert_eq!(slice.iter().sum::<u8>(), 0);

        unsafe { allocator.deallocate(buf.as_ptr(), len) };
    }

    #[test]
    fn test_allocate_buffer_zeroed() {
        #[cfg(feature = "rust-allocator")]
        test_allocate_zeroed_buffer_help(RUST);

        #[cfg(feature = "c-allocator")]
        test_allocate_zeroed_buffer_help(C);

        test_allocate_zeroed_buffer_help(FAIL);
    }

    #[test]
    fn test_deallocate_null() {
        unsafe {
            #[cfg(feature = "rust-allocator")]
            (RUST.zfree)(core::ptr::null_mut(), core::ptr::null_mut());

            #[cfg(feature = "c-allocator")]
            (C.zfree)(core::ptr::null_mut(), core::ptr::null_mut());

            (FAIL.zfree)(core::ptr::null_mut(), core::ptr::null_mut());
        }
    }

    #[test]
    fn test_allocate_zero_size() {
        // Verify that zero-size allocation requests return a null pointer.
        unsafe {
            assert!(zalloc_c(ptr::null_mut(), 1, 0).is_null());
            assert!(zalloc_c(ptr::null_mut(), 0, 1).is_null());
            assert!(zalloc_c_calloc(ptr::null_mut(), 1, 0).is_null());
            assert!(zalloc_c_calloc(ptr::null_mut(), 0, 1).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust(ptr::null_mut(), 1, 0).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust(ptr::null_mut(), 0, 1).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust_calloc(ptr::null_mut(), 1, 0).is_null());
            #[cfg(feature = "rust-allocator")]
            assert!(zalloc_rust_calloc(ptr::null_mut(), 0, 1).is_null());
        }
    }

    #[test]
    fn allocate_more_than_c_uint_max() {
        unsafe extern "C" fn small_alloc(
            _opaque: *mut c_void,
            _items: c_uint,
            _size: c_uint,
        ) -> *mut c_void {
            unreachable!("we should return NULL before this");
        }

        unsafe extern "C" fn small_free(_opaque: *mut c_void, _ptr: *mut c_void) {}

        let mut storage = [0u8; 64];
        let allocator = Allocator {
            zalloc: small_alloc,
            zfree: small_free,
            opaque: storage.as_mut_ptr().cast(),
            _marker: PhantomData,
        };

        let Some(len) = (c_uint::MAX as usize).checked_add(1) else {
            return;
        };
        assert!(allocator.allocate_slice_raw::<u8>(len).is_none());
    }
}
