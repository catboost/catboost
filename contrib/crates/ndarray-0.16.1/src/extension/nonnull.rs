#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::ptr::NonNull;

/// Return a NonNull<T> pointer to the vector's data
pub(crate) fn nonnull_from_vec_data<T>(v: &mut Vec<T>) -> NonNull<T>
{
    // this pointer is guaranteed to be non-null
    unsafe { NonNull::new_unchecked(v.as_mut_ptr()) }
}

/// Converts `ptr` to `NonNull<T>`
///
/// Safety: `ptr` *must* be non-null.
/// This is checked with a debug assertion, and will panic if this is not true,
/// but treat this as an unconditional conversion.
#[inline]
pub(crate) unsafe fn nonnull_debug_checked_from_ptr<T>(ptr: *mut T) -> NonNull<T>
{
    debug_assert!(!ptr.is_null());
    NonNull::new_unchecked(ptr)
}
