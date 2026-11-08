//! This crate provides a `no_std` version of std's [`is_x86_feature_detected!`]
//! macro.
//!
//! This is possible because x86 chips can just use the `cpuid` instruction to
//! detect CPU features, whereas most other architectures require either reading
//! files or querying the OS.
//!
//! # Usage
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//! # fn main() {
//! if core_detect::is_x86_feature_detected!("ssse3") {
//!     println!("SSSE3 is available");
//! }
//! # }
//! # #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
//! # fn main() {}
//! ```
//!
//! Note that like the [equivalent macro in `std`][stddetect], this will error
//! on architectures other than x86/x86_64, so you should put the code behind a
//! `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]` check.
//!
//! [stddetect]:
//! https://doc.rust-lang.org/nightly/std/macro.is_x86_feature_detected.html
//!
//! (In the future, this crate may provide another macro which returns false in
//! these cases instead, and supports testing multiple features simultaneously).
//!
//! # Caveats
//! The `cpuid` instruction doesn't exist on all x86 machines, it was added
//! around 1994. (It's also not available on SGX, but this doesn't cause any
//! issues since we can check that with `cfg(target_env = "sgx")`).
//!
//! If you run `cpuid` on a machine older than that, it causes an illegal
//! instruction fault (SIGILL). Unfortunately, there's no good stable way to
//! reliably determine if `cpuid` will fault in stable rust: A
//! [`core::arch::x86::has_cpuid`](https://doc.rust-lang.org/nightly/core/arch/x86/fn.has_cpuid.html)
//! function exists, but didn't stabilize with the rest of `core::arch::x86`,
//! and the only way to implement it ourselves is with inline asm, which... is
//! also still unstable.
//!
//! For what it's worth, it's actually pretty uncommon that we'd need to call
//! `has_cpuid` on common rust targets, since we perform the following compile
//! time checks:
//! - We never have cpuid on `target_env = "sgx"` (as mentioned).
//! - We always have cpuid on `target_arch = "x86_64"`.
//! - And we always have cpuid if `target_feature = "sse"` (which covers the
//!   `i686-*` targets).
//!
//! Unfortunately, if none of those applies... we don't know if calling CPUID
//! will crash the process. This library has a few ways it can handle this:
//!
//! 1. Cautiously (the default): In this mode, we conceptually swap `has_cpuid`
//!    out with a function that always returns false. That is: we never call it
//!    unless we're sure it wont crash the process.
//!
//! 2. Recklessly (`feature = "assume_has_cpuid"`): This is essentially the
//!    opposite of the last one — assume `has_cpuid` would have returned true,
//!    and call `cpuid` anyway.
//!
//!     In practice, this should be fine. These machines are rare now (they're
//!     over 30 years old...), and pretty only are common through QEMU, and even
//!     then, usually after a misconfiguration.
//!
//!     If you do happen to run the instruction, the process crashes, but in a
//!     controlled manner — Executing an illegal instruction to tringger a
//!     SIGILL is what `core::intrinsics::abort` does on x86, so it's not
//!     dangerous or anything.
//!
//! 3. Using unstable nightly features (`feature = "unstable_has_cpuid"`): This
//!    approach requires a nightly compiler, but has no other major downsides,
//!    besides the fact that the `has_cpuid` function could vanish at any time.
//!
//! Eventually, inline asm will stabilize and we can solve this problem more
//! cleanly.
#![no_std]
#![allow(dead_code)]
#![cfg_attr(
    all(
        target_arch = "x86",
        not(target_env = "sgx"),
        not(target_feature = "sse"),
        feature = "unstable_has_cpuid",
    ),
    feature(stdsimd)
)]

#[macro_use]
mod macros;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "arch/x86.rs"]
#[macro_use]
mod arch;

// Unimplemented architecture:
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod arch {
    #[doc(hidden)]
    pub(crate) enum Feature {
        Null,
    }

    #[doc(hidden)]
    pub mod __is_feature_detected {}

    impl Feature {
        #[doc(hidden)]
        pub(crate) fn from_str(_s: &str) -> Result<Feature, ()> {
            Err(())
        }
        #[doc(hidden)]
        pub(crate) fn to_str(self) -> &'static str {
            ""
        }
    }
}

pub(crate) use crate::arch::Feature;
#[doc(hidden)]
pub use crate::arch::__is_feature_detected;

/// Performs run-time feature detection.
#[inline]
#[allow(dead_code)]
fn check_for(x: Feature) -> bool {
    cache::test(x as u32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "os/x86.rs"]
mod os;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod os {
    #[inline]
    pub(crate) fn detect_features() -> crate::cache::Initializer {
        Default::default()
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_export]
macro_rules! is_x86_feature_detected {
    ($t: tt) => {
        compile_error!(
            r#"
        is_x86_feature_detected can only be used on x86 and x86_64 targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected(...) { ... }
            }
        "#
        )
    };
}

mod cache;
