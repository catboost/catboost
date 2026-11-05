//! # Bindings to the `CoreGraphics` framework
//!
//! See [Apple's docs][apple-doc] and [the general docs on framework crates][framework-crates] for more information.
//!
//! [apple-doc]: https://developer.apple.com/documentation/coregraphics/
//! [framework-crates]: https://docs.rs/objc2/latest/objc2/topics/about_generated/index.html
#![no_std]
#![cfg_attr(feature = "unstable-darwin-objc", feature(darwin_objc))]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Update in Cargo.toml as well.
#![doc(html_root_url = "https://docs.rs/objc2-core-graphics/0.3.2")]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "CGBitmapContext")]
mod bitmap_context;
mod generated;
#[cfg(feature = "CGImage")]
mod image;
mod thread_safety;
#[cfg(feature = "CGBitmapContext")]
#[allow(unused_imports, unreachable_pub)]
pub use self::bitmap_context::*;
#[allow(unused_imports, unreachable_pub)]
pub use self::generated::*;

#[allow(dead_code)]
pub(crate) type UniCharCount = core::ffi::c_ulong;
#[allow(dead_code)]
pub(crate) type UniChar = u16;

#[allow(non_upper_case_globals)]
#[cfg(feature = "CGWindowLevel")]
/// [Apple's documentation](https://developer.apple.com/documentation/coregraphics/kcgnumreservedwindowlevels?language=objc)
pub const kCGNumReservedWindowLevels: i32 = 16;

#[allow(non_upper_case_globals)]
#[cfg(feature = "CGWindowLevel")]
/// [Apple's documentation](https://developer.apple.com/documentation/coregraphics/kcgnumreservedbasewindowlevels?language=objc)
pub const kCGNumReservedBaseWindowLevels: i32 = 5;

#[allow(non_upper_case_globals)]
#[cfg(feature = "CGWindowLevel")]
/// [Apple's documentation](https://developer.apple.com/documentation/coregraphics/kcgbasewindowlevel?language=objc)
pub const kCGBaseWindowLevel: CGWindowLevel = i32::MIN;

#[allow(non_upper_case_globals)]
#[cfg(feature = "CGWindowLevel")]
/// [Apple's documentation](https://developer.apple.com/documentation/coregraphics/kcgminimumwindowlevel?language=objc)
pub const kCGMinimumWindowLevel: CGWindowLevel =
    kCGBaseWindowLevel + kCGNumReservedBaseWindowLevels;

#[allow(non_upper_case_globals)]
#[cfg(feature = "CGWindowLevel")]
/// [Apple's documentation](https://developer.apple.com/documentation/coregraphics/kcgmaximumwindowlevel?language=objc)
pub const kCGMaximumWindowLevel: CGWindowLevel = i32::MAX - kCGNumReservedWindowLevels;
