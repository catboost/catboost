use crate::{CGBitmapInfo, CGImageByteOrderInfo};

#[allow(non_upper_case_globals, deprecated)]
impl CGBitmapInfo {
    #[doc(alias = "kCGBitmapByteOrder16Host")]
    pub const ByteOrder16Host: Self = if cfg!(target_endian = "big") {
        Self::ByteOrder16Big
    } else {
        Self::ByteOrder16Little
    };

    #[doc(alias = "kCGBitmapByteOrder32Host")]
    pub const ByteOrder32Host: Self = if cfg!(target_endian = "big") {
        Self::ByteOrder32Big
    } else {
        Self::ByteOrder32Little
    };
}

#[allow(non_upper_case_globals, deprecated)]
impl CGImageByteOrderInfo {
    #[doc(alias = "kCGImageByteOrder16Host")]
    pub const Order16Host: Self = if cfg!(target_endian = "big") {
        Self::Order16Big
    } else {
        Self::Order16Little
    };

    #[doc(alias = "kCGImageByteOrder32Host")]
    pub const Order32Host: Self = if cfg!(target_endian = "big") {
        Self::Order32Big
    } else {
        Self::Order32Little
    };
}
