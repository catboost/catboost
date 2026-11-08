use core::mem::MaybeUninit;

use crate::{builder::Builder, traits::*, Cursor, Offset};

impl WriteAsOffset<str> for alloc::string::String {
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<str> {
        WriteAsOffset::prepare(self.as_str(), builder)
    }
}

impl WriteAs<Offset<str>> for alloc::string::String {
    type Prepared = Offset<str>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Offset<str> {
        WriteAsOffset::prepare(self.as_str(), builder)
    }
}

impl WriteAsDefault<Offset<str>, str> for alloc::string::String {
    type Prepared = Offset<str>;

    #[inline]
    fn prepare(&self, builder: &mut Builder, default: &str) -> Option<Offset<str>> {
        if self == default {
            None
        } else {
            Some(WriteAsOffset::prepare(self.as_str(), builder))
        }
    }
}

impl WriteAsOptional<Offset<str>> for alloc::string::String {
    type Prepared = Offset<str>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<Offset<str>> {
        Some(WriteAsOffset::prepare(self.as_str(), builder))
    }
}

/// # Safety
/// The implementation of `write_values` initializes all the bytes.
unsafe impl VectorWrite<Offset<str>> for alloc::string::String {
    type Value = Offset<str>;

    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Self::Value {
        WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[Offset<str>],
        bytes: *mut MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [MaybeUninit<u8>; 4];
        for (i, v) in values.iter().enumerate() {
            v.write(
                Cursor::new(&mut *bytes.add(i)),
                buffer_position - (4 * i) as u32,
            );
        }
    }
}
