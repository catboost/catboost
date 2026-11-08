mod array;
mod bool_;
mod box_;
mod byte_slice;
mod offset;
mod option;
mod primitives;
mod ref_;
mod result;
mod slice;
mod str;
mod string;
mod union_offset;
mod union_vector_offset;
mod unit;
mod vec;

pub(crate) fn array_from_buffer(
    buffer: crate::slice_helpers::SliceWithStartOffset<'_>,
    offset: usize,
) -> core::result::Result<
    (crate::slice_helpers::SliceWithStartOffset<'_>, usize),
    crate::errors::ErrorKind,
> {
    let value: u32 = crate::traits::TableRead::from_buffer(buffer, offset)?;
    let vector_offset = offset
        .checked_add(value as usize)
        .ok_or(crate::errors::ErrorKind::InvalidOffset)?;
    let buffer = buffer.advance(vector_offset)?;
    let len: u32 = crate::traits::TableRead::from_buffer(buffer, 0)?;
    Ok((buffer.advance(4)?, len as usize))
}
