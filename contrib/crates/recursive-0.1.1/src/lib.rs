//! Rust has a problematic relationship with recursive functions, because functions that recurse
//! deeply can overflow the stack, crashing your program. This crate makes it easy to remedy
//! this problem by marking (indirectly) recursive functions as such:
//! ```rust
//! use recursive::recursive;
//!
//! #[recursive]
//! fn sum(nums: &[u64]) -> u64 {
//!     if let Some((head, tail)) = nums.split_first() {
//!         head + sum(tail)
//!     } else {
//!         0
//!     }
//! }
//! ```
//!
//! The way this prevents stack overflows is by checking the size of the remaining stack at the
//! start of each call to your function. If this size is under a boundary set by
//! [`set_minimum_stack_size`] (by default 128 KiB), a new stack is allocated and execution
//! continues on that stack. This new stack's size is set using [`set_stack_allocation_size`], which
//! is 2 MiB by default.
//!
//! This crate works by wrapping your function body in a call to [`stacker::maybe_grow`]. If this
//! crate is not flexible enough for your needs consider using [`stacker`] directly yourself.
//!
//! ## What are the downsides?
//!
//! This crate is **not** zero cost, but it is also not limited to simple tail recursion or direct
//! recursion. However, in most cases the stack size test is very fast and almost always succeeds
//! without needing to allocate. If your recursive algorithm is very performance-sensitive I would
//! suggest rewriting it to an iterative version regardless.
//!
//! This crate only supports those platforms that [`stacker`] supports. The Rust compiler itself
//! uses [`stacker`], so the platform you're compiling on should always be fine, but for more
//! obscure targets see its documentation.
//!
//! ## Which functions should I mark as `#[recursive]`?
//!
//! Any function that directly calls itself should be marked as `#[recursive]`, unless you know for
//! certain that the stack is sufficiently large for any inputs that function will be called with.
//! If you are feeding untrusted input into a recursive function you should always mark it as
//! `#[recursive]`.
//!
//! It is not necessary to mark every single function that can indirectly recurse as `#[recursive]`.
//! As long as every possible cycle of function calls includes at least one function marked
//! `#[recursive]` you will be protected against stack overflows due to recursion.

use std::sync::atomic::{AtomicUsize, Ordering};

// These are referred to only by the procedural macro.
#[doc(hidden)]
pub mod __impl {
    #[allow(unused)]
    pub use stacker;
}

/// Marks a function to use an automatically growing segmented stack.
///
/// See the [crate docs](crate) for details.
pub use recursive_proc_macro_impl::recursive;

static MINIMUM_STACK_SIZE: AtomicUsize = AtomicUsize::new(128 * 1024);
static STACK_ALLOC_SIZE: AtomicUsize = AtomicUsize::new(2 * 1024 * 1024);

/// This sets the minimum stack size that [`recursive`] requires.
///
/// If a function tagged with `#[recursive]` is called when the remaining stack size is less than
/// this value a new stack gets allocated.
///
/// The default value if this function is never called is 128 KiB.
pub fn set_minimum_stack_size(bytes: usize) {
    MINIMUM_STACK_SIZE.store(bytes, Ordering::Relaxed);
}

/// Returns the value set by [`set_minimum_stack_size`].
pub fn get_minimum_stack_size() -> usize {
    MINIMUM_STACK_SIZE.load(Ordering::Relaxed)
}

/// When a new stack gets allocated it will get allocated with this size.
///
/// The default value if this function is never called is 2 MiB.
pub fn set_stack_allocation_size(bytes: usize) {
    STACK_ALLOC_SIZE.store(bytes, Ordering::Relaxed);
}

/// Returns the value set by [`set_stack_allocation_size`].
pub fn get_stack_allocation_size() -> usize {
    STACK_ALLOC_SIZE.load(Ordering::Relaxed)
}
