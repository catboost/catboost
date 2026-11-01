use core::mem::MaybeUninit;

use crate::{builder::Builder, traits::*, Cursor, Offset};

/// # Safety
/// An offset is a 32-bit unsigned integer + a zero-sized type.
unsafe impl<T: ?Sized> Primitive for Offset<T> {
    const ALIGNMENT: usize = 4;
    const SIZE: usize = 4;
}

impl<T: ?Sized> WriteAsPrimitive<Offset<T>> for Offset<T> {
    #[inline]
    fn write<const N: usize>(&self, cursor: Cursor<'_, N>, buffer_position: u32) {
        cursor
            .assert_size()
            .finish(u32::to_le_bytes(buffer_position - self.offset));
    }
}

impl<T: ?Sized> WriteAsOffset<T> for Offset<T> {
    fn prepare(&self, _builder: &mut Builder) -> Offset<T> {
        *self
    }
}

impl<T: ?Sized> WriteAs<Offset<T>> for Offset<T> {
    type Prepared = Self;
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Self {
        *self
    }
}

impl<T: ?Sized> WriteAsOptional<Offset<T>> for Offset<T> {
    type Prepared = Self;
    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Option<Self> {
        Some(*self)
    }
}

impl<T: ?Sized, D> WriteAsDefault<Offset<T>, D> for Offset<T> {
    type Prepared = Self;
    #[inline]
    fn prepare(&self, _builder: &mut Builder, _default: &D) -> Option<Self> {
        Some(*self)
    }
}

// # Safety
// The implementation of `write_values` initializes all the bytes.
unsafe impl<T: ?Sized> VectorWrite<Offset<T>> for Offset<T> {
    const STRIDE: usize = 4;
    type Value = Offset<T>;

    #[inline]
    fn prepare(&self, _builder: &mut Builder) -> Self::Value {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Offset<T>],
        bytes: *mut MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [MaybeUninit<u8>; 4];
        for (i, v) in values.iter().enumerate() {
            v.write(
                Cursor::new(&mut *bytes.add(i)),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}
