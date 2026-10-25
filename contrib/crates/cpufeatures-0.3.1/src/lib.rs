#![no_std]
#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg"
)]

#[cfg(not(miri))]
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub mod aarch64;

#[cfg(not(miri))]
#[cfg(target_arch = "loongarch64")]
#[doc(hidden)]
pub mod loongarch64;

#[cfg(not(miri))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(miri)]
mod miri;

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "loongarch64",
    target_arch = "x86",
    target_arch = "x86_64"
)))]
compile_error!("This crate works only on `aarch64`, `loongarch64`, `x86`, and `x86-64` targets.");

/// Create module with CPU feature detection code.
#[macro_export]
macro_rules! new {
    ($mod_name:ident, $($tf:tt),+ $(,)?) => {
        mod $mod_name {
            use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

            const UNINIT: u8 = u8::MAX;
            static STORAGE: AtomicU8 = AtomicU8::new(UNINIT);

            /// Initialization token
            #[derive(Copy, Clone, Debug)]
            pub struct InitToken(());

            impl InitToken {
                /// Initialize token, performing CPU feature detection.
                pub fn init() -> Self {
                    init()
                }

                /// Initialize token and return a `bool` indicating if the feature is supported.
                pub fn init_get() -> (Self, bool) {
                    init_get()
                }

                /// Get initialized value.
                #[inline(always)]
                pub fn get(&self) -> bool {
                    $crate::__unless_target_features! {
                        $($tf),+ => {
                            STORAGE.load(Relaxed) == 1
                        }
                    }
                }
            }

            /// Get stored value and initialization token,
            /// initializing underlying storage if needed.
            #[inline]
            pub fn init_get() -> (InitToken, bool) {
                let res = $crate::__unless_target_features! {
                    $($tf),+ => {
                        #[cold]
                        fn init_inner() -> bool {
                            let res = $crate::__detect_target_features!($($tf),+);
                            STORAGE.store(res as u8, Relaxed);
                            res
                        }

                        // Relaxed ordering is fine, as we only have a single atomic variable.
                        let val = STORAGE.load(Relaxed);

                        if val == UNINIT {
                            init_inner()
                        } else {
                            val == 1
                        }
                    }
                };

                (InitToken(()), res)
            }

            /// Initialize underlying storage if needed and get initialization token.
            #[inline]
            pub fn init() -> InitToken {
                init_get().0
            }

            /// Initialize underlying storage if needed and get stored value.
            #[inline]
            pub fn get() -> bool {
                init_get().1
            }
        }
    };
}
