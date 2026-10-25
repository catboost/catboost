use crate::{
    errors::ErrorKind, slice_helpers::SliceWithStartOffset, TableRead, TableReadUnion,
    TableReadUnionVector,
};

#[derive(Copy, Clone, Debug)]
pub struct Table<'buf> {
    object: SliceWithStartOffset<'buf>,
    vtable: &'buf [u8],
}

impl<'buf> Table<'buf> {
    pub fn from_buffer(
        buffer: SliceWithStartOffset<'buf>,
        field_offset: usize,
    ) -> Result<Self, ErrorKind> {
        let field_value = u32::from_buffer(buffer, field_offset)?;
        let object_offset = field_offset
            .checked_add(field_value as usize)
            .ok_or(ErrorKind::InvalidOffset)?;
        let object = buffer.advance(object_offset)?;

        let vtable_offset_relative = i32::from_buffer(buffer, object_offset)?;
        let vtable_offset: usize = (object_offset as i64)
            .checked_sub(vtable_offset_relative as i64)
            .ok_or(ErrorKind::InvalidOffset)?
            .try_into()
            .map_err(|_| ErrorKind::InvalidOffset)?;

        let vtable_size = u16::from_buffer(buffer, vtable_offset)?;
        if vtable_size < 4 || vtable_size % 2 != 0 {
            return Err(ErrorKind::InvalidVtableLength {
                length: vtable_size,
            });
        }
        let vtable_full = buffer
            .advance(vtable_offset)?
            .as_slice()
            .get(..vtable_size as usize)
            .ok_or(ErrorKind::InvalidLength)?;
        let vtable = vtable_full.get(4..).ok_or(ErrorKind::InvalidOffset)?;

        Ok(Self { object, vtable })
    }

    pub fn access<T: TableRead<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<Option<T>> {
        let offset = self
            .vtable
            .get(2 * vtable_offset..2 * (vtable_offset + 1))
            .unwrap_or(&[0, 0]);
        let offset = u16::from_le_bytes(offset.try_into().unwrap()) as usize;
        if offset != 0 {
            T::from_buffer(self.object, offset)
                .map(Some)
                .map_err(|error_kind| crate::errors::Error {
                    source_location: crate::errors::ErrorLocation {
                        type_,
                        method,
                        byte_offset: self.object.offset_from_start,
                    },
                    error_kind,
                })
        } else {
            Ok(None)
        }
    }

    pub fn access_required<T: TableRead<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<T> {
        self.access(vtable_offset, type_, method)?
            .ok_or(crate::errors::Error {
                source_location: crate::errors::ErrorLocation {
                    type_,
                    method,
                    byte_offset: self.object.offset_from_start,
                },
                error_kind: ErrorKind::MissingRequired,
            })
    }

    pub fn access_union<T: TableReadUnion<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<Option<T>> {
        let source_location = crate::errors::ErrorLocation {
            type_,
            method,
            byte_offset: self.object.offset_from_start,
        };
        let make_error = |error_kind| crate::errors::Error {
            source_location,
            error_kind,
        };

        if let Some(offset) = self.vtable.get(2 * vtable_offset..2 * (vtable_offset + 2)) {
            let tag_offset = u16::from_le_bytes(offset[..2].try_into().unwrap()) as usize;
            let value_offset = u16::from_le_bytes(offset[2..].try_into().unwrap()) as usize;
            let tag = u8::from_buffer(self.object, tag_offset).map_err(make_error)?;
            if tag_offset != 0 && value_offset != 0 && tag != 0 {
                T::from_buffer(self.object, tag, value_offset)
                    .map(Some)
                    .map_err(make_error)
            } else {
                Ok(None)
            }
        } else if self.vtable.len() <= 2 * vtable_offset {
            Ok(None)
        } else {
            Err(make_error(ErrorKind::InvalidVtableLength {
                // The slice does not contain the vtable or objects
                // size, but they are included for the error message
                length: self.vtable.len() as u16 + 4,
            }))
        }
    }

    pub fn access_union_required<T: TableReadUnion<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<T> {
        self.access_union(vtable_offset, type_, method)?
            .ok_or(crate::errors::Error {
                source_location: crate::errors::ErrorLocation {
                    type_,
                    method,
                    byte_offset: self.object.offset_from_start,
                },
                error_kind: ErrorKind::MissingRequired,
            })
    }

    pub fn access_union_vector<T: TableReadUnionVector<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<Option<T>> {
        let source_location = crate::errors::ErrorLocation {
            type_,
            method,
            byte_offset: self.object.offset_from_start,
        };
        let make_error = |error_kind| crate::errors::Error {
            source_location,
            error_kind,
        };

        if let Some(offset) = self.vtable.get(2 * vtable_offset..2 * (vtable_offset + 2)) {
            let tags_offset = u16::from_le_bytes(offset[..2].try_into().unwrap()) as usize;
            let values_offset = u16::from_le_bytes(offset[2..].try_into().unwrap()) as usize;
            match (tags_offset, values_offset) {
                (0, 0) => Ok(None),
                (0, _) | (_, 0) => todo!(),
                (tags_offset, values_offset) => Ok(Some(
                    T::from_buffer(self.object, tags_offset, values_offset).map_err(make_error)?,
                )),
            }
        } else {
            Ok(None)
        }
    }

    pub fn access_union_vector_required<T: TableReadUnionVector<'buf>>(
        &self,
        vtable_offset: usize,
        type_: &'static str,
        method: &'static str,
    ) -> crate::Result<T> {
        self.access_union_vector(vtable_offset, type_, method)?
            .ok_or(crate::errors::Error {
                source_location: crate::errors::ErrorLocation {
                    type_,
                    method,
                    byte_offset: self.object.offset_from_start,
                },
                error_kind: ErrorKind::MissingRequired,
            })
    }
}
