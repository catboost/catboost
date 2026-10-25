//! Fallible, streaming iteration.
//!
//! `FallibleStreamingIterator` differs from the standard library's `Iterator` trait in two ways:
//! iteration can fail, resulting in an error, and only one element of the iteration is available at
//! any time.
//!
//! While these iterators cannot be used with Rust `for` loops, `while let` loops offer a similar
//! level of ergonomics:
//!
//! ```ignore
//! while let Some(value) = it.next()? {
//!     // use value
//! }
//! ```
#![doc(html_root_url = "https://docs.rs/fallible-streaming-iterator/0.1")]
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

use core::cmp;
use core::marker::PhantomData;

/// A fallible, streaming iterator.
pub trait FallibleStreamingIterator {
    /// The type being iterated over.
    type Item: ?Sized;

    /// The error type of iteration.
    type Error;

    /// Advances the iterator to the next position.
    ///
    /// Iterators start just before the first item, so this method should be called before `get`
    /// when iterating.
    ///
    /// The behavior of calling this method after `get` has returned `None`, or after this method
    /// has returned an error is unspecified.
    fn advance(&mut self) -> Result<(), Self::Error>;

    /// Returns the current element.
    ///
    /// The behavior of calling this method before any calls to `advance` is unspecified.
    fn get(&self) -> Option<&Self::Item>;

    /// Advances the iterator, returning the next element.
    ///
    /// The default implementation simply calls `advance` followed by `get`.
    #[inline]
    fn next(&mut self) -> Result<Option<&Self::Item>, Self::Error> {
        self.advance()?;
        Ok((*self).get())
    }

    /// Returns bounds on the number of remaining elements in the iterator.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    /// Determines if all elements of the iterator satisfy a predicate.
    #[inline]
    fn all<F>(&mut self, mut f: F) -> Result<bool, Self::Error>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        while let Some(e) = self.next()? {
            if !f(e) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Determines if any elements of the iterator satisfy a predicate.
    #[inline]
    fn any<F>(&mut self, mut f: F) -> Result<bool, Self::Error>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        self.all(|e| !f(e)).map(|r| !r)
    }

    /// Borrows an iterator, rather than consuming it.
    ///
    /// This is useful to allow the application of iterator adaptors while still retaining ownership
    /// of the original adaptor.
    #[inline]
    fn by_ref(&mut self) -> &mut Self
    where
        Self: Sized,
    {
        self
    }

    /// Returns the number of remaining elements in the iterator.
    #[inline]
    fn count(mut self) -> Result<usize, Self::Error>
    where
        Self: Sized,
    {
        let mut count = 0;
        while let Some(_) = self.next()? {
            count += 1;
        }
        Ok(count)
    }

    /// Returns an iterator which filters elements by a predicate.
    #[inline]
    fn filter<F>(self, f: F) -> Filter<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        Filter { it: self, f: f }
    }

    /// Returns the first element of the iterator which satisfies a predicate.
    #[inline]
    fn find<F>(&mut self, mut f: F) -> Result<Option<&Self::Item>, Self::Error>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        loop {
            self.advance()?;
            match self.get() {
                Some(v) => {
                    if f(v) {
                        break;
                    }
                }
                None => break,
            }
        }
        Ok((*self).get())
    }

    /// Calls a closure on each element of an iterator.
    #[inline]
    fn for_each<F>(mut self, mut f: F) -> Result<(), Self::Error>
    where
        Self: Sized,
        F: FnMut(&Self::Item),
    {
        while let Some(value) = self.next()? {
            f(value);
        }
        Ok(())
    }

    /// Returns an iterator which is well-behaved at the beginning and end of iteration.
    #[inline]
    fn fuse(self) -> Fuse<Self>
    where
        Self: Sized,
    {
        Fuse {
            it: self,
            state: FuseState::Start,
        }
    }

    /// Returns an iterator which applies a transform to elements.
    #[inline]
    fn map<F, B>(self, f: F) -> Map<Self, F, B>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        Map {
            it: self,
            f: f,
            value: None,
        }
    }

    /// Returns an iterator which applies a transform to elements.
    ///
    /// Unlike `map`, the the closure provided to this method returns a reference into the original
    /// value.
    #[inline]
    fn map_ref<F, B: ?Sized>(self, f: F) -> MapRef<Self, F>
    where
        Self: Sized,
        F: Fn(&Self::Item) -> &B,
    {
        MapRef { it: self, f: f }
    }

    /// Returns an iterator that applies a transform to errors.
    #[inline]
    fn map_err<F, B>(self, f: F) -> MapErr<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Error) -> B,
    {
        MapErr { it: self, f: f }
    }

    /// Returns the `nth` element of the iterator.
    #[inline]
    fn nth(&mut self, n: usize) -> Result<Option<&Self::Item>, Self::Error> {
        for _ in 0..n {
            self.advance()?;
            if let None = self.get() {
                return Ok(None);
            }
        }
        self.next()
    }

    /// Returns the position of the first element matching a predicate.
    #[inline]
    fn position<F>(&mut self, mut f: F) -> Result<Option<usize>, Self::Error>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        let mut pos = 0;
        while let Some(v) = self.next()? {
            if f(v) {
                return Ok(Some(pos));
            }
            pos += 1;
        }
        Ok(None)
    }

    /// Returns an iterator which skips the first `n` elements.
    #[inline]
    fn skip(self, n: usize) -> Skip<Self>
    where
        Self: Sized,
    {
        Skip { it: self, n: n }
    }

    /// Returns an iterator which skips the first sequence of elements matching a predicate.
    #[inline]
    fn skip_while<F>(self, f: F) -> SkipWhile<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        SkipWhile {
            it: self,
            f: f,
            done: false,
        }
    }

    /// Returns an iterator which only returns the first `n` elements.
    #[inline]
    fn take(self, n: usize) -> Take<Self>
    where
        Self: Sized,
    {
        Take {
            it: self,
            n: n,
            done: false,
        }
    }

    /// Returns an iterator which only returns the first sequence of elements matching a predicate.
    #[inline]
    fn take_while<F>(self, f: F) -> TakeWhile<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> bool,
    {
        TakeWhile {
            it: self,
            f: f,
            done: false,
        }
    }
}

/// A fallible, streaming iterator which can be advanced from either end.
pub trait DoubleEndedFallibleStreamingIterator: FallibleStreamingIterator {
    /// Advances the state of the iterator to the next item from the end.
    ///
    /// Iterators start just after the last item, so this method should be called before `get`
    /// when iterating.
    ///
    /// The behavior of calling this method after `get` has returned `None`, or after this method
    /// or `advance` has returned an error is unspecified.
    fn advance_back(&mut self) -> Result<(), Self::Error>;

    /// Advances the back of the iterator, returning the last element.
    ///
    /// The default implementation simply calls `advance_back` followed by `get`.
    #[inline]
    fn next_back(&mut self) -> Result<Option<&Self::Item>, Self::Error> {
        self.advance_back()?;
        Ok((*self).get())
    }
}

impl<'a, I: ?Sized> FallibleStreamingIterator for &'a mut I
where
    I: FallibleStreamingIterator,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        (**self).advance()
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        (**self).get()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    #[inline]
    fn next(&mut self) -> Result<Option<&I::Item>, I::Error> {
        (**self).next()
    }
}

#[cfg(feature = "std")]
impl<I: ?Sized> FallibleStreamingIterator for Box<I>
where
    I: FallibleStreamingIterator,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        (**self).advance()
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        (**self).get()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    #[inline]
    fn next(&mut self) -> Result<Option<&I::Item>, I::Error> {
        (**self).next()
    }
}

/// Converts a normal `Iterator` over `Results` of references into a
/// `FallibleStreamingIterator`.
pub fn convert<'a, I, T, E>(it: I) -> Convert<'a, I, T>
where
    I: Iterator<Item = Result<&'a T, E>>,
{
    Convert { it: it, item: None }
}

/// An iterator which wraps a normal `Iterator`.
pub struct Convert<'a, I, T: 'a> {
    it: I,
    item: Option<&'a T>,
}

impl<'a, I, T, E> FallibleStreamingIterator for Convert<'a, I, T>
where
    I: Iterator<Item = Result<&'a T, E>>,
{
    type Item = T;
    type Error = E;

    #[inline]
    fn advance(&mut self) -> Result<(), E> {
        self.item = match self.it.next() {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => return Err(e),
            None => None,
        };
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&T> {
        self.item
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, I, T, E> DoubleEndedFallibleStreamingIterator for Convert<'a, I, T>
where
    I: DoubleEndedIterator<Item = Result<&'a T, E>>,
{
    #[inline]
    fn advance_back(&mut self) -> Result<(), E> {
        self.item = match self.it.next_back() {
            Some(Ok(v)) => Some(v),
            Some(Err(e)) => return Err(e),
            None => None,
        };
        Ok(())
    }
}

/// Returns an iterator over no items.
pub fn empty<T, E>() -> Empty<T, E> {
    Empty(PhantomData)
}

/// An iterator over no items.
pub struct Empty<T, E>(PhantomData<(T, E)>);

impl<T, E> FallibleStreamingIterator for Empty<T, E> {
    type Item = T;
    type Error = E;

    #[inline]
    fn advance(&mut self) -> Result<(), E> {
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&T> {
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl<T, E> DoubleEndedFallibleStreamingIterator for Empty<T, E> {
    #[inline]
    fn advance_back(&mut self) -> Result<(), E> {
        Ok(())
    }
}

/// An iterator which filters elements with a predicate.
pub struct Filter<I, F> {
    it: I,
    f: F,
}

impl<I, F> FallibleStreamingIterator for Filter<I, F>
where
    I: FallibleStreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        while let Some(i) = self.it.next()? {
            if (self.f)(i) {
                break;
            }
        }
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.it.size_hint().1)
    }
}

#[derive(Copy, Clone)]
enum FuseState {
    Start,
    Middle,
    End,
}

/// An iterator which is well-behaved at the beginning and end of iteration.
pub struct Fuse<I> {
    it: I,
    state: FuseState,
}

impl<I> FallibleStreamingIterator for Fuse<I>
where
    I: FallibleStreamingIterator,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        match self.state {
            FuseState::Start => {
                match self.it.next() {
                    Ok(Some(_)) => self.state = FuseState::Middle,
                    Ok(None) => self.state = FuseState::End,
                    Err(e) => {
                        self.state = FuseState::End;
                        return Err(e);
                    }
                };
            }
            FuseState::Middle => match self.it.next() {
                Ok(Some(_)) => {}
                Ok(None) => self.state = FuseState::End,
                Err(e) => {
                    self.state = FuseState::End;
                    return Err(e);
                }
            },
            FuseState::End => {}
        }
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        match self.state {
            FuseState::Middle => self.it.get(),
            FuseState::Start | FuseState::End => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    #[inline]
    fn next(&mut self) -> Result<Option<&I::Item>, I::Error> {
        match self.state {
            FuseState::Start => match self.it.next() {
                Ok(Some(v)) => {
                    self.state = FuseState::Middle;
                    Ok(Some(v))
                }
                Ok(None) => {
                    self.state = FuseState::End;
                    Ok(None)
                }
                Err(e) => {
                    self.state = FuseState::End;
                    Err(e)
                }
            },
            FuseState::Middle => match self.it.next() {
                Ok(Some(v)) => Ok(Some(v)),
                Ok(None) => {
                    self.state = FuseState::End;
                    Ok(None)
                }
                Err(e) => {
                    self.state = FuseState::End;
                    Err(e)
                }
            },
            FuseState::End => Ok(None),
        }
    }
}

/// An iterator which applies a transform to elements.
pub struct Map<I, F, B> {
    it: I,
    f: F,
    value: Option<B>,
}

impl<I, F, B> FallibleStreamingIterator for Map<I, F, B>
where
    I: FallibleStreamingIterator,
    F: FnMut(&I::Item) -> B,
{
    type Item = B;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        self.value = self.it.next()?.map(&mut self.f);
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&B> {
        self.value.as_ref()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I, F, B> DoubleEndedFallibleStreamingIterator for Map<I, F, B>
where
    I: DoubleEndedFallibleStreamingIterator,
    F: FnMut(&I::Item) -> B,
{
    #[inline]
    fn advance_back(&mut self) -> Result<(), I::Error> {
        self.value = self.it.next_back()?.map(&mut self.f);
        Ok(())
    }
}

/// An iterator which applies a transform to elements.
pub struct MapRef<I, F> {
    it: I,
    f: F,
}

impl<I, F, B: ?Sized> FallibleStreamingIterator for MapRef<I, F>
where
    I: FallibleStreamingIterator,
    F: Fn(&I::Item) -> &B,
{
    type Item = B;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        self.it.advance()
    }

    #[inline]
    fn get(&self) -> Option<&B> {
        self.it.get().map(&self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I, F, B: ?Sized> DoubleEndedFallibleStreamingIterator for MapRef<I, F>
where
    I: DoubleEndedFallibleStreamingIterator,
    F: Fn(&I::Item) -> &B,
{
    #[inline]
    fn advance_back(&mut self) -> Result<(), I::Error> {
        self.it.advance_back()
    }
}

/// An iterator which applies a transform to errors.
pub struct MapErr<I, F> {
    it: I,
    f: F,
}

impl<I, F, B> FallibleStreamingIterator for MapErr<I, F>
where
    I: FallibleStreamingIterator,
    F: Fn(I::Error) -> B,
{
    type Item = I::Item;
    type Error = B;

    #[inline]
    fn advance(&mut self) -> Result<(), B> {
        self.it.advance().map_err(&mut self.f)
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }

    #[inline]
    fn next(&mut self) -> Result<Option<&I::Item>, B> {
        self.it.next().map_err(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I, F, B> DoubleEndedFallibleStreamingIterator for MapErr<I, F>
where
    I: DoubleEndedFallibleStreamingIterator,
    F: Fn(I::Error) -> B,
{
    #[inline]
    fn advance_back(&mut self) -> Result<(), B> {
        self.it.advance_back().map_err(&mut self.f)
    }

    #[inline]
    fn next_back(&mut self) -> Result<Option<&I::Item>, B> {
        self.it.next_back().map_err(&mut self.f)
    }
}

/// An iterator which skips a number of initial elements.
pub struct Skip<I> {
    it: I,
    n: usize,
}

impl<I> FallibleStreamingIterator for Skip<I>
where
    I: FallibleStreamingIterator,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        for _ in 0..self.n {
            if let None = self.it.next()? {
                return Ok(());
            }
        }
        self.n = 0;
        self.advance()
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.it.size_hint();
        (
            hint.0.saturating_sub(self.n),
            hint.1.map(|h| h.saturating_sub(self.n)),
        )
    }
}

/// An iterator which skips initial elements matching a predicate.
pub struct SkipWhile<I, F> {
    it: I,
    f: F,
    done: bool,
}

impl<I, F> FallibleStreamingIterator for SkipWhile<I, F>
where
    I: FallibleStreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        if !self.done {
            self.done = true;
            let f = &mut self.f;
            self.it.find(|i| !f(i)).map(|_| ())
        } else {
            self.it.advance()
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.it.size_hint();
        if self.done {
            hint
        } else {
            (0, hint.1)
        }
    }
}

/// An iterator which only returns a number of initial elements.
pub struct Take<I> {
    it: I,
    n: usize,
    done: bool,
}

impl<I> FallibleStreamingIterator for Take<I>
where
    I: FallibleStreamingIterator,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        if self.n != 0 {
            self.it.advance()?;
            self.n -= 1;
        } else {
            self.done = true;
        }
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        if self.done {
            None
        } else {
            self.it.get()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.it.size_hint();

        let lower = cmp::min(lower, self.n);

        let upper = match upper {
            Some(x) if x < self.n => Some(x),
            _ => Some(self.n)
        };

        (lower, upper)
    }
}

/// An iterator which only returns initial elements matching a predicate.
pub struct TakeWhile<I, F> {
    it: I,
    f: F,
    done: bool,
}

impl<I, F> FallibleStreamingIterator for TakeWhile<I, F>
where
    I: FallibleStreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;
    type Error = I::Error;

    #[inline]
    fn advance(&mut self) -> Result<(), I::Error> {
        if let Some(v) = self.it.next()? {
            if !(self.f)(v) {
                self.done = true;
            }
        }
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        if self.done {
            None
        } else {
            self.it.get()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            (0, self.it.size_hint().1)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn _is_object_safe(_: &FallibleStreamingIterator<Item = (), Error = ()>) {}
    fn _is_object_safe_double(_: &DoubleEndedFallibleStreamingIterator<Item = (), Error = ()>) {}
}
