use std::ops::Index;
use std::ops::IndexMut;

use crate::dimension::Dimension;

pub(in crate::dimension) struct Forward<D>(pub(crate) D);
pub(in crate::dimension) struct Reverse<D>(pub(crate) D);

impl<D> Index<usize> for Forward<&D>
where D: Dimension
{
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize
    {
        &self.0[index]
    }
}

impl<D> Index<usize> for Forward<&mut D>
where D: Dimension
{
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize
    {
        &self.0[index]
    }
}

impl<D> IndexMut<usize> for Forward<&mut D>
where D: Dimension
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut usize
    {
        &mut self.0[index]
    }
}

impl<D> Index<usize> for Reverse<&D>
where D: Dimension
{
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize
    {
        &self.0[self.len() - index - 1]
    }
}

impl<D> Index<usize> for Reverse<&mut D>
where D: Dimension
{
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize
    {
        &self.0[self.len() - index - 1]
    }
}

impl<D> IndexMut<usize> for Reverse<&mut D>
where D: Dimension
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut usize
    {
        let len = self.len();
        &mut self.0[len - index - 1]
    }
}

/// Indexable sequence with length
pub(in crate::dimension) trait Sequence: Index<usize>
{
    fn len(&self) -> usize;
}

/// Indexable sequence with length (mut)
pub(in crate::dimension) trait SequenceMut: Sequence + IndexMut<usize> {}

impl<D> Sequence for Forward<&D>
where D: Dimension
{
    #[inline]
    fn len(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<D> Sequence for Forward<&mut D>
where D: Dimension
{
    #[inline]
    fn len(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<D> SequenceMut for Forward<&mut D> where D: Dimension {}

impl<D> Sequence for Reverse<&D>
where D: Dimension
{
    #[inline]
    fn len(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<D> Sequence for Reverse<&mut D>
where D: Dimension
{
    #[inline]
    fn len(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<D> SequenceMut for Reverse<&mut D> where D: Dimension {}
