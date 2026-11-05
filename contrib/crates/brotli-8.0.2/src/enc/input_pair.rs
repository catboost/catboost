use core;
use core::cmp::min;

use super::super::alloc::{SliceWrapper, SliceWrapperMut};
use super::interface::Freezable;
#[derive(Copy, Clone, Default, Debug)]
pub struct InputReference<'a> {
    pub data: &'a [u8],
    pub orig_offset: usize, // offset into the original slice of data
}
impl<'a> SliceWrapper<u8> for InputReference<'a> {
    fn slice(&self) -> &[u8] {
        self.data
    }
}

impl<'a> Freezable for InputReference<'a> {
    fn freeze(&self) -> super::interface::SliceOffset {
        debug_assert!(self.data.len() <= 0xffff_ffff);
        super::interface::SliceOffset(self.orig_offset, self.data.len() as u32)
    }
}

#[derive(Default)]
pub struct InputReferenceMut<'a> {
    pub data: &'a mut [u8],
    pub orig_offset: usize, // offset into the original slice of data
}

impl<'a> SliceWrapper<u8> for InputReferenceMut<'a> {
    fn slice(&self) -> &[u8] {
        self.data
    }
}
impl<'a> SliceWrapperMut<u8> for InputReferenceMut<'a> {
    fn slice_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

impl<'a> From<InputReferenceMut<'a>> for InputReference<'a> {
    fn from(val: InputReferenceMut<'a>) -> InputReference<'a> {
        InputReference {
            data: val.data,
            orig_offset: val.orig_offset,
        }
    }
}

impl<'a> From<&'a InputReferenceMut<'a>> for InputReference<'a> {
    fn from(val: &'a InputReferenceMut<'a>) -> InputReference<'a> {
        InputReference {
            data: val.data,
            orig_offset: val.orig_offset,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct InputPair<'a>(pub InputReference<'a>, pub InputReference<'a>);

impl<'a> PartialEq for InputPair<'a> {
    fn eq(&self, other: &InputPair<'_>) -> bool {
        if self.0.len() + self.1.len() != other.0.len() + other.1.len() {
            return false;
        }
        for (a_iter, b_iter) in self
            .0
            .data
            .iter()
            .chain(self.1.data.iter())
            .zip(other.0.data.iter().chain(other.1.data.iter()))
        {
            if *a_iter != *b_iter {
                return false;
            }
        }
        true
    }
}
impl<'a> core::ops::Index<usize> for InputPair<'a> {
    type Output = u8;
    fn index(&self, index: usize) -> &u8 {
        if index >= self.0.len() {
            &self.1.data[index - self.0.len()]
        } else {
            &self.0.data[index]
        }
    }
}
impl<'a> core::fmt::LowerHex for InputPair<'a> {
    fn fmt(&self, fmtr: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        for item in self.0.data {
            fmtr.write_fmt(format_args!("{:02x}", item))?
        }
        for item in self.1.data {
            fmtr.write_fmt(format_args!("{:02x}", item))?
        }
        Ok(())
    }
}

impl<'a> InputPair<'a> {
    pub fn split_at(&self, loc: usize) -> (InputPair<'a>, InputPair<'a>) {
        if loc >= self.0.len() {
            let offset_from_self_1 = loc - self.0.len();
            let (first, second) = self.1.data.split_at(min(offset_from_self_1, self.1.len()));
            return (
                InputPair::<'a>(
                    self.0,
                    InputReference::<'a> {
                        data: first,
                        orig_offset: self.1.orig_offset,
                    },
                ),
                InputPair::<'a>(
                    InputReference::<'a>::default(),
                    InputReference::<'a> {
                        data: second,
                        orig_offset: offset_from_self_1 + self.1.orig_offset,
                    },
                ),
            );
        }
        let (first, second) = self.0.data.split_at(min(loc, self.0.len()));
        (
            InputPair::<'a>(
                InputReference::<'a> {
                    data: first,
                    orig_offset: self.0.orig_offset,
                },
                InputReference::<'a>::default(),
            ),
            InputPair::<'a>(
                InputReference::<'a> {
                    data: second,
                    orig_offset: self.0.orig_offset + loc,
                },
                self.1,
            ),
        )
    }
    pub fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }
}
