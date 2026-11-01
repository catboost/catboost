#[allow(unused_imports)]
use crate::*;

// SAFETY: Marked as @unchecked Swift.Sendable under the following conditions:
// @available(macOS 10.9, iOS 7.0, tvOS 9.0, watchOS 2.0, visionOS 1.0, *)
// Which are all newer than Rust's minimum supported versions
#[cfg(feature = "CGColor")]
unsafe impl Send for CGColor {}
#[cfg(feature = "CGColor")]
unsafe impl Sync for CGColor {}
#[cfg(feature = "CGImage")]
unsafe impl Send for CGImage {}
#[cfg(feature = "CGImage")]
unsafe impl Sync for CGImage {}
#[cfg(feature = "CGColorSpace")]
unsafe impl Send for CGColorSpace {}
#[cfg(feature = "CGColorSpace")]
unsafe impl Sync for CGColorSpace {}

// TODO: CGDisplayMode?
