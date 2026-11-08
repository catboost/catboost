use crate::traits::{VectorRead, VectorReadInner};

impl<'buf, T: VectorReadInner<'buf>, E: 'buf> VectorRead<'buf> for Result<T, E>
where
    E: core::convert::From<T::Error>,
{
    const STRIDE: usize = T::STRIDE;

    unsafe fn from_buffer(buffer: crate::SliceWithStartOffset<'buf>, offset: usize) -> Self {
        Ok(T::from_buffer(buffer, offset)?)
    }
}
