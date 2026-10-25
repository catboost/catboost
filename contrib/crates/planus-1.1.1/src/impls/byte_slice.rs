use core::mem::MaybeUninit;

use crate::{
    errors::ErrorKind, slice_helpers::SliceWithStartOffset, traits::*, Builder, Cursor, Offset,
};

impl<'buf> TableRead<'buf> for &'buf [u8] {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> core::result::Result<Self, ErrorKind> {
        let (buffer, len) = super::array_from_buffer(buffer, offset)?;
        buffer.as_slice().get(..len).ok_or(ErrorKind::InvalidLength)
    }
}

impl<'buf> TableRead<'buf> for &'buf [i8] {
    fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> core::result::Result<Self, ErrorKind> {
        let (buffer, len) = super::array_from_buffer(buffer, offset)?;
        let slice = buffer
            .as_slice()
            .get(..len)
            .ok_or(ErrorKind::InvalidLength)?;
        Ok(unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const i8, slice.len()) })
    }
}

impl WriteAsOffset<[u8]> for [u8] {
    #[allow(clippy::let_and_return)]
    fn prepare(&self, builder: &mut Builder) -> Offset<[u8]> {
        #[cfg(feature = "bytes-cache")]
        let hash = {
            let hash = builder.bytes_cache.hash(self);
            if let Some(offset) = builder
                .bytes_cache
                .get(builder.inner.as_slice(), hash, self)
            {
                return offset.into();
            }
            hash
        };

        // SAFETY: We make sure to write the 4+len bytes inside the closure
        unsafe {
            builder.write_with(
                self.len().checked_add(4).unwrap(),
                u32::ALIGNMENT_MASK,
                |buffer_position, bytes| {
                    let bytes = bytes.as_mut_ptr();

                    (self.len() as u32).write(
                        Cursor::new(&mut *(bytes as *mut [MaybeUninit<u8>; 4])),
                        buffer_position,
                    );

                    core::ptr::copy_nonoverlapping(
                        self.as_ptr(),
                        bytes.add(4) as *mut u8,
                        self.len(),
                    );
                },
            )
        }
        let offset = builder.current_offset();

        #[cfg(feature = "bytes-cache")]
        builder
            .bytes_cache
            .insert(hash, offset.into(), builder.inner.as_slice());

        offset
    }
}

impl WriteAsOffset<[i8]> for [i8] {
    #[allow(clippy::let_and_return)]
    fn prepare(&self, builder: &mut Builder) -> Offset<[i8]> {
        #[cfg(feature = "bytes-cache")]
        let hash = {
            let v: &[u8] =
                unsafe { core::slice::from_raw_parts(self.as_ptr() as *const u8, self.len()) };
            let hash = builder.bytes_cache.hash(v);
            if let Some(offset) = builder.bytes_cache.get(builder.inner.as_slice(), hash, v) {
                return offset.into();
            }
            hash
        };

        // SAFETY: We make sure to write the 4+len bytes inside the closure
        unsafe {
            builder.write_with(
                self.len().checked_add(4).unwrap(),
                u32::ALIGNMENT_MASK,
                |buffer_position, bytes| {
                    let bytes = bytes.as_mut_ptr();

                    (self.len() as u32).write(
                        Cursor::new(&mut *(bytes as *mut [MaybeUninit<u8>; 4])),
                        buffer_position,
                    );

                    core::ptr::copy_nonoverlapping(
                        self.as_ptr(),
                        bytes.add(4) as *mut i8,
                        self.len(),
                    );
                },
            )
        }
        let offset = builder.current_offset();

        #[cfg(feature = "bytes-cache")]
        builder
            .bytes_cache
            .insert(hash, offset.into(), builder.inner.as_slice());

        offset
    }
}

impl<const N: usize> WriteAsOffset<[u8]> for [u8; N] {
    fn prepare(&self, builder: &mut Builder) -> Offset<[u8]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<const N: usize> WriteAs<Offset<[u8]>> for [u8; N] {
    type Prepared = Offset<[u8]>;

    fn prepare(&self, builder: &mut Builder) -> Offset<[u8]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<const N: usize> WriteAsOptional<Offset<[u8]>> for [u8; N] {
    type Prepared = Offset<[u8]>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<Offset<[u8]>> {
        Some(WriteAsOffset::prepare(self.as_slice(), builder))
    }
}

impl<const N: usize> WriteAsOffset<[i8]> for [i8; N] {
    fn prepare(&self, builder: &mut Builder) -> Offset<[i8]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<const N: usize> WriteAs<Offset<[i8]>> for [i8; N] {
    type Prepared = Offset<[i8]>;

    fn prepare(&self, builder: &mut Builder) -> Offset<[i8]> {
        WriteAsOffset::prepare(self.as_slice(), builder)
    }
}

impl<const N: usize> WriteAsOptional<Offset<[i8]>> for [i8; N] {
    type Prepared = Offset<[i8]>;

    #[inline]
    fn prepare(&self, builder: &mut Builder) -> Option<Offset<[i8]>> {
        Some(WriteAsOffset::prepare(self.as_slice(), builder))
    }
}
