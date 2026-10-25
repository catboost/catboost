use std::slice::SliceIndex;

/// Trait for indexing into an array
pub trait Indexed<T> {
    /// Elements of the array
    type Element: ?Sized;

    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    fn get(&self, i: T) -> Option<&Self::Element>;
}
/// A trait for the minimal common functionality of a vale array
pub trait Array {
    /// Elements of the array
    type Element;

    /// Iterates over the values paris
    #[must_use]
    fn iter(&self) -> Box<dyn Iterator<Item = &Self::Element> + '_>;

    /// Number of key/value pairs
    #[must_use]
    fn len(&self) -> usize;

    /// Returns if the array is empty
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for indexing into an array
pub trait IndexedMut<T> {
    /// Elements of the array
    type Element: ?Sized;

    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    fn get_mut(&mut self, i: T) -> Option<&mut Self::Element>;
}
/// Mutability functions for a value array
pub trait ArrayMut {
    /// Elements of the array
    type Element;

    /// Returns the last element of the array or `None`
    #[must_use]
    fn pop(&mut self) -> Option<Self::Element>;

    /// Appends e to the end of the `Array`
    fn push(&mut self, e: Self::Element);
}

impl<T, I> Indexed<I> for Vec<T>
where
    I: SliceIndex<[T]>,
{
    type Element = <I as SliceIndex<[T]>>::Output;
    #[inline]
    fn get(&self, i: I) -> Option<&Self::Element> {
        <[T]>::get(self, i)
    }
}

impl<T> Array for Vec<T> {
    type Element = T;

    fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        Box::new(<[T]>::iter(self))
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, I> IndexedMut<I> for Vec<T>
where
    I: SliceIndex<[T]>,
{
    type Element = <I as SliceIndex<[T]>>::Output;
    #[inline]
    fn get_mut(&mut self, i: I) -> Option<&mut Self::Element> {
        <[T]>::get_mut(self, i)
    }
}

impl<T> ArrayMut for Vec<T> {
    type Element = T;

    #[inline]
    fn pop(&mut self) -> Option<T> {
        Vec::pop(self)
    }

    #[inline]
    fn push(&mut self, e: T) {
        Vec::push(self, e);
    }
}

#[cfg(feature = "c-abi")]
mod cabi {
    use super::{Array, ArrayMut, Indexed, IndexedMut, SliceIndex};
    use abi_stable::std_types::RVec;

    impl<T, I> Indexed<I> for RVec<T>
    where
        I: SliceIndex<[T]>,
    {
        type Element = <I as SliceIndex<[T]>>::Output;
        #[inline]
        fn get(&self, i: I) -> Option<&Self::Element> {
            <[T]>::get(self, i)
        }
    }

    impl<T> Array for RVec<T> {
        type Element = T;

        fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = &T> + 'i> {
            Box::new(<[T]>::iter(self))
        }

        #[inline]
        fn len(&self) -> usize {
            self.len()
        }
    }

    impl<T, I> IndexedMut<I> for RVec<T>
    where
        I: SliceIndex<[T]>,
    {
        type Element = <I as SliceIndex<[T]>>::Output;
        #[inline]
        fn get_mut(&mut self, i: I) -> Option<&mut Self::Element> {
            <[T]>::get_mut(self, i)
        }
    }

    #[cfg(feature = "c-abi")]
    impl<T> ArrayMut for RVec<T> {
        type Element = T;

        #[inline]
        fn pop(&mut self) -> Option<T> {
            abi_stable::std_types::RVec::pop(self)
        }

        #[inline]
        fn push(&mut self, e: T) {
            abi_stable::std_types::RVec::push(self, e);
        }
    }
}
