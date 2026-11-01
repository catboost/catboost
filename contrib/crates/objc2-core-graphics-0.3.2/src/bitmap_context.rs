#![allow(warnings, clippy)]
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2_core_foundation::*;

use crate::*;

/// # Safety
///
/// - `data` must be a valid pointer or null.
/// - `release_callback` must be implemented correctly.
/// - `release_info` must be a valid pointer or null.
#[cfg(all(feature = "CGColorSpace", feature = "CGContext", feature = "CGImage"))]
#[inline]
pub unsafe extern "C-unwind" fn CGBitmapContextCreateWithData(
    data: *mut c_void,
    width: usize,
    height: usize,
    bits_per_component: usize,
    bytes_per_row: usize,
    space: Option<&CGColorSpace>,
    bitmap_info: u32,
    release_callback: CGBitmapContextReleaseDataCallback,
    release_info: *mut c_void,
) -> Option<CFRetained<CGContext>> {
    extern "C-unwind" {
        fn CGBitmapContextCreateWithData(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: Option<&CGColorSpace>,
            bitmap_info: u32,
            release_callback: CGBitmapContextReleaseDataCallback,
            release_info: *mut c_void,
        ) -> Option<NonNull<CGContext>>;
    }
    let ret = unsafe {
        CGBitmapContextCreateWithData(
            data,
            width,
            height,
            bits_per_component,
            bytes_per_row,
            space,
            bitmap_info,
            release_callback,
            release_info,
        )
    };
    ret.map(|ret| unsafe { CFRetained::from_raw(ret) })
}

/// # Safety
///
/// `data` must be a valid pointer or null.
#[cfg(all(feature = "CGColorSpace", feature = "CGContext", feature = "CGImage"))]
#[inline]
pub unsafe extern "C-unwind" fn CGBitmapContextCreate(
    data: *mut c_void,
    width: usize,
    height: usize,
    bits_per_component: usize,
    bytes_per_row: usize,
    space: Option<&CGColorSpace>,
    bitmap_info: u32,
) -> Option<CFRetained<CGContext>> {
    extern "C-unwind" {
        fn CGBitmapContextCreate(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: Option<&CGColorSpace>,
            bitmap_info: u32,
        ) -> Option<NonNull<CGContext>>;
    }
    let ret = unsafe {
        CGBitmapContextCreate(
            data,
            width,
            height,
            bits_per_component,
            bytes_per_row,
            space,
            bitmap_info,
        )
    };
    ret.map(|ret| unsafe { CFRetained::from_raw(ret) })
}
