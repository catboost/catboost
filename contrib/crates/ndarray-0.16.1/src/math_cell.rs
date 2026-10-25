use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;

use std::ops::{Deref, DerefMut};

/// A transparent wrapper of [`Cell<T>`](std::cell::Cell) which is identical in every way, except
/// it will implement arithmetic operators as well.
///
/// The purpose of `MathCell` is to be used from [.cell_view()](crate::ArrayBase::cell_view).
/// The `MathCell` derefs to `Cell`, so all the cell's methods are available.
#[repr(transparent)]
#[derive(Default)]
pub struct MathCell<T>(Cell<T>);

impl<T> MathCell<T>
{
    /// Create a new cell with the given value
    #[inline(always)]
    pub const fn new(value: T) -> Self
    {
        MathCell(Cell::new(value))
    }

    /// Return the inner value
    pub fn into_inner(self) -> T
    {
        Cell::into_inner(self.0)
    }

    /// Swap value with another cell
    pub fn swap(&self, other: &Self)
    {
        Cell::swap(&self.0, &other.0)
    }
}

impl<T> Deref for MathCell<T>
{
    type Target = Cell<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl<T> DerefMut for MathCell<T>
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        &mut self.0
    }
}

impl<T> Clone for MathCell<T>
where T: Copy
{
    fn clone(&self) -> Self
    {
        MathCell::new(self.get())
    }
}

impl<T> PartialEq for MathCell<T>
where T: Copy + PartialEq
{
    fn eq(&self, rhs: &Self) -> bool
    {
        self.get() == rhs.get()
    }
}

impl<T> Eq for MathCell<T> where T: Copy + Eq {}

impl<T> PartialOrd for MathCell<T>
where T: Copy + PartialOrd
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering>
    {
        self.get().partial_cmp(&rhs.get())
    }

    fn lt(&self, rhs: &Self) -> bool
    {
        self.get().lt(&rhs.get())
    }
    fn le(&self, rhs: &Self) -> bool
    {
        self.get().le(&rhs.get())
    }
    fn gt(&self, rhs: &Self) -> bool
    {
        self.get().gt(&rhs.get())
    }
    fn ge(&self, rhs: &Self) -> bool
    {
        self.get().ge(&rhs.get())
    }
}

impl<T> Ord for MathCell<T>
where T: Copy + Ord
{
    fn cmp(&self, rhs: &Self) -> Ordering
    {
        self.get().cmp(&rhs.get())
    }
}

impl<T> fmt::Debug for MathCell<T>
where T: Copy + fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        self.get().fmt(f)
    }
}

#[cfg(test)]
mod tests
{
    use super::MathCell;

    #[test]
    fn test_basic()
    {
        let c = &MathCell::new(0);
        c.set(1);
        assert_eq!(c.get(), 1);
    }
}
