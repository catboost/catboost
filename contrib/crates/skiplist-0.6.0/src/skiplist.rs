//! A skiplist implementation which allows faster random access than a standard linked list.

#![allow(warnings)]
#![allow(clippy)]
#![allow(unknown_lints)]

use std::{
    cmp, cmp::Ordering, default, fmt, hash, hash::Hash, iter, ops, ops::Bound, ptr::NonNull,
};

pub use crate::skipnode::{IntoIter, Iter, IterMut};
use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    skipnode::SkipNode,
};

// ////////////////////////////////////////////////////////////////////////////
// SkipList
// ////////////////////////////////////////////////////////////////////////////

/// SkipList provides a way of storing elements and provides efficient way to
/// access, insert and remove nodes.
///
/// Unlike a standard linked list, the skiplist can skip ahead when trying to
/// find a particular index.
pub struct SkipList<T, Gen = Geometric> {
    // Storage, this is not sorted
    head: Box<SkipNode<T>>,
    len: usize,
    level_generator: Gen,
}

// ///////////////////////////////////////////////
// Inherent methods
// ///////////////////////////////////////////////

impl<T> SkipList<T> {
    /// Create a new skiplist with the default number of 16 levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist: SkipList<i64> = SkipList::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        // Parameters are fixed and will produce a valid level generator.
        let lg = Geometric::new(16, 1.0 / 2.0).expect("Failed to create level generator.");
        Self::new_from_gen(lg)
    }

    /// Constructs a new, empty skiplist with the optimal number of levels for
    /// the intended capacity.  Specifically, it uses `floor(log2(capacity))`
    /// number of levels, ensuring that only *a few* nodes occupy the highest
    /// level.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::with_capacity(100);
    /// skiplist.extend(0..100);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let levels = cmp::max(1, (capacity as f64).log2().floor() as usize);
        // Parameters are safe as levels >= 1 and p is in (0, 1).
        let lg = Geometric::new(levels, 1.0 / 2.0).expect("Failed to create level generator.");
        Self::new_from_gen(lg)
    }
}

impl<T, G> SkipList<T, G>
where
    G: LevelGenerator,
{
    /// Constructs a new, empty skip list using a given [LevelGenerator]
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::{SkipList, level_generator::GeometricalLevelGenerator};
    ///
    /// let lg = GeometricalLevelGenerator::new(10, 0.5);
    /// let mut skiplist: SkipList<i64> = SkipList::new_from_gen(lg);
    /// ```
    #[inline]
    pub fn new_from_gen(lg: G) -> Self {
        SkipList {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
        }
    }

    /// Clears the skiplist, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// skiplist.clear();
    /// assert!(skiplist.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        *self.head = SkipNode::head(self.level_generator.total());
    }

    /// Returns the number of elements in the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.len(), 10);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the skiplist contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.is_empty());
    ///
    /// skiplist.push_back(1);
    /// assert!(!skiplist.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert the element into the skiplist at the given index, shifting all
    /// subsequent nodes down.
    ///
    /// # Panics
    ///
    /// Panics if the insert index is greater than the length of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    ///
    /// skiplist.insert(0, 0);
    /// skiplist.insert(5, 1);
    /// assert_eq!(skiplist.len(), 2);
    /// assert!(!skiplist.is_empty());
    /// ```
    pub fn insert(&mut self, value: T, index: usize) {
        if index > self.len() {
            panic!("Index out of bounds.");
        }
        self.len += 1;
        let new_node = Box::new(SkipNode::new(value, self.level_generator.level()));
        self.head
            .insert_at(new_node, index)
            .unwrap_or_else(|_| panic!("No insertion position is found!"));
    }

    /// Insert the element into the front of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.push_front(1);
    /// skiplist.push_front(2);
    /// ```
    pub fn push_front(&mut self, value: T) {
        self.insert(value, 0);
    }

    /// Insert the element into the back of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    /// ```
    pub fn push_back(&mut self, value: T) {
        let len = self.len();
        self.insert(value, len);
    }

    /// Provides a reference to the front element, or `None` if the skiplist is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.front().is_none());
    ///
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    /// assert_eq!(skiplist.front(), Some(&1));
    /// ```
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() { None } else { self.get(0) }
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// skiplist is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.front().is_none());
    ///
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    /// assert_eq!(skiplist.front_mut(), Some(&mut 1));
    /// ```
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            None
        } else {
            self.get_mut(0)
        }
    }

    /// Provides a reference to the back element, or `None` if the skiplist is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.back().is_none());
    ///
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    /// assert_eq!(skiplist.back(), Some(&2));
    /// ```
    #[inline]
    pub fn back(&self) -> Option<&T> {
        let len = self.len();
        if len > 0 { self.get(len - 1) } else { None }
    }

    /// Provides a reference to the back element, or `None` if the skiplist is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.back().is_none());
    ///
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    /// assert_eq!(skiplist.back_mut(), Some(&mut 2));
    /// ```
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        let len = self.len();
        if len > 0 { self.get_mut(len - 1) } else { None }
    }

    /// Provides a reference to the element at the given index, or `None` if the
    /// skiplist is empty or the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.get(0).is_none());
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.get(0), Some(&0));
    /// assert!(skiplist.get(10).is_none());
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.get_index(index).and_then(|node| node.item.as_ref())
    }

    /// Provides a mutable reference to the element at the given index, or
    /// `None` if the skiplist is empty or the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// assert!(skiplist.get_mut(0).is_none());
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.get_mut(0), Some(&mut 0));
    /// assert!(skiplist.get_mut(10).is_none());
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.get_index_mut(index)
            .and_then(|node| node.item.as_mut())
    }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    ///
    /// assert_eq!(skiplist.pop_front(), Some(1));
    /// assert_eq!(skiplist.pop_front(), Some(2));
    /// assert!(skiplist.pop_front().is_none());
    /// ```
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(0))
        }
    }

    /// Removes the last element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.push_back(1);
    /// skiplist.push_back(2);
    ///
    /// assert_eq!(skiplist.pop_back(), Some(2));
    /// assert_eq!(skiplist.pop_back(), Some(1));
    /// assert!(skiplist.pop_back().is_none());
    /// ```
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        let len = self.len();
        if len > 0 {
            Some(self.remove(len - 1))
        } else {
            None
        }
    }

    /// Removes and returns an element with the given index.
    ///
    /// # Panics
    ///
    /// Panics is the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.remove(4), 4);
    /// assert_eq!(skiplist.remove(4), 5);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len() {
            panic!("Index out of bounds.");
        } else {
            let node = self.head.remove_at(index).unwrap();
            self.len -= 1;
            node.into_inner().unwrap()
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns false.
    /// This method operates in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// skiplist.retain(|&x| x%2 == 0);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.len -= self.head.retain(move |_, x| f(x));
    }

    /// Get an owning iterator over the entries of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.into_iter() {
    ///     println!("Value: {}", i);
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(mut self) -> IntoIter<T> {
        let len = self.len();
        unsafe { IntoIter::from_head(&mut self.head, len) }
    }

    /// Creates an iterator over the entries of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.iter() {
    ///     println!("Value: {}", i);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<T> {
        unsafe { Iter::from_head(&self.head, self.len()) }
    }

    /// Creates an mutable iterator over the entries of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.iter_mut() {
    ///     println!("Value: {}", i);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let len = self.len();
        unsafe { IterMut::from_head(&mut self.head, len) }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the
    /// skiplist, starting at min, and ending at max. If min is `Unbounded`,
    /// then it will be treated as "negative infinity", and if max is
    /// `Unbounded`, then it will be treated as "positive infinity".  Thus
    /// range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    /// use std::ops::Bound::{Included, Unbounded};
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.range(Included(3), Included(7)) {
    ///     println!("Value: {}", i);
    /// }
    /// assert_eq!(Some(&4), skiplist.range(Included(4), Unbounded).next());
    /// ```
    pub fn range(&self, min: Bound<usize>, max: Bound<usize>) -> Iter<T> {
        let first = Self::_lower_bound(min);
        let last = self._upper_bound(max);
        self.iter_range(first, last)
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements
    /// in the skiplist, starting at min, and ending at max. If min is
    /// `Unbounded`, then it will be treated as "negative infinity", and if max
    /// is `Unbounded`, then it will be treated as "positive infinity".  Thus
    /// range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    /// use std::ops::Bound::{Included, Unbounded};
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.range_mut(Included(3), Included(7)) {
    ///     println!("Value: {}", i);
    /// }
    /// assert_eq!(Some(&mut 4), skiplist.range_mut(Included(4), Unbounded).next());
    /// ```
    pub fn range_mut(&mut self, min: Bound<usize>, max: Bound<usize>) -> IterMut<T> {
        let first = match min {
            Bound::Included(i) => i,
            Bound::Excluded(i) => i + 1,
            Bound::Unbounded => 0,
        };
        let last = match max {
            Bound::Included(i) => i,
            Bound::Excluded(i) => i - 1,
            Bound::Unbounded => self.len() - 1,
        };
        self.iter_range_mut(first, last)
    }

    /// Returns an `Option<&T>` pointing to the lowest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    ///
    /// assert_eq!(skiplist.lower_bound(Unbounded), Some(&0));
    /// assert_eq!(skiplist.lower_bound(Excluded(0)), Some(&1));
    /// assert_eq!(skiplist.lower_bound(Included(0)), Some(&0));
    /// assert_eq!(skiplist.lower_bound(Excluded(10)), None);
    /// ```
    pub fn lower_bound(&self, min: Bound<usize>) -> Option<&T> {
        self.get_index(Self::_lower_bound(min))
            .and_then(|node| node.item.as_ref())
    }

    /// Returns an `Option<&T>` pointing to the highest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    ///
    /// assert_eq!(skiplist.upper_bound(Unbounded), Some(&9));
    /// assert_eq!(skiplist.upper_bound(Excluded(9)), Some(&8));
    /// assert_eq!(skiplist.upper_bound(Included(9)), Some(&9));
    /// ```
    pub fn upper_bound(&self, max: Bound<usize>) -> Option<&T> {
        self.get_index(self._upper_bound(max))
            .and_then(|node| node.item.as_ref())
    }

    fn _lower_bound(min: Bound<usize>) -> usize {
        match min {
            Bound::Included(i) => i,
            Bound::Excluded(i) => i + 1,
            Bound::Unbounded => 0,
        }
    }

    fn _upper_bound(&self, max: Bound<usize>) -> usize {
        match max {
            Bound::Included(i) => i,
            Bound::Excluded(i) => i - 1,
            Bound::Unbounded => self.len() - 1,
        }
    }
}

impl<T, G> SkipList<T, G>
where
    T: PartialEq,
    G: LevelGenerator,
{
    /// Returns true if the value is contained in the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// assert!(skiplist.contains(&4));
    /// assert!(!skiplist.contains(&15));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        self.iter().any(|val| val.eq(value))
    }

    /// Returns the index of the value if it's contained in the skiplist or None otherwise.
    ///
    /// Since this skip list is not ordered, this method has a complexity of O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.index_of(&4), 4);
    /// ```
    #[inline]
    pub fn index_of(&self, item: &T) -> Option<usize> {
        self.iter()
            .enumerate()
            .find_map(|(idx, it)| if item.eq(it) { Some(idx) } else { None })
    }

    /// Removes all consecutive repeated elements in the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipList;
    ///
    /// let mut skiplist = SkipList::new();
    /// skiplist.push_back(0);
    /// skiplist.push_back(0);
    /// assert_eq!(skiplist.len(), 2);
    /// skiplist.dedup();
    /// assert_eq!(skiplist.len(), 1);
    /// ```
    pub fn dedup(&mut self) {
        let removed = self
            .head
            .retain(|prev, current| prev.is_none_or(|prev| !prev.eq(current)));
        self.len -= removed;
    }
}

// ///////////////////////////////////////////////
// Internal methods
// ///////////////////////////////////////////////

impl<T, G> SkipList<T, G>
where
    G: LevelGenerator,
{
    /// Checks the integrity of the skiplist.
    #[allow(dead_code)]
    fn check(&self) {
        self.head.check();
    }

    /// Makes an iterator between [begin, end]
    fn iter_range(&self, first_idx: usize, last_idx: usize) -> Iter<T> {
        if first_idx > last_idx {
            return Iter {
                first: None,
                last: None,
                size: 0,
            };
        }
        let first = self.get_index(first_idx);
        let last = self.get_index(last_idx);
        if first.is_some() && last.is_some() {
            Iter {
                first,
                last,
                size: last_idx - first_idx + 1,
            }
        } else {
            Iter {
                first: None,
                last: None,
                size: 0,
            }
        }
    }

    /// Makes an iterator between [begin, end]
    fn iter_range_mut(&mut self, first_idx: usize, last_idx: usize) -> IterMut<T> {
        if first_idx > last_idx {
            return IterMut {
                first: None,
                last: None,
                size: 0,
            };
        }
        let last = self.get_index_mut(last_idx).and_then(|p| NonNull::new(p));
        let first = self.get_index_mut(first_idx);
        if first.is_some() && last.is_some() {
            IterMut {
                first,
                last,
                size: last_idx - first_idx + 1,
            }
        } else {
            IterMut {
                first: None,
                last: None,
                size: 0,
            }
        }
    }

    /// Gets a pointer to the node with the given index.
    fn get_index(&self, index: usize) -> Option<&SkipNode<T>> {
        if self.len() <= index {
            None
        } else {
            self.head.advance(index + 1)
        }
    }

    fn get_index_mut(&mut self, index: usize) -> Option<&mut SkipNode<T>> {
        if self.len() <= index {
            None
        } else {
            self.head.advance_mut(index + 1)
        }
    }
}

impl<T> SkipList<T>
where
    T: fmt::Debug,
{
    /// Prints out the internal structure of the skiplist (for debugging
    /// purposes).
    #[allow(dead_code)]
    fn debug_structure(&self) {
        unsafe {
            let mut node = self.head.as_ref();
            let mut rows: Vec<_> = iter::repeat(String::new())
                .take(self.level_generator.total())
                .collect();

            loop {
                let value = match node.item {
                    Some(ref v) => {
                        format!("> [{:?}]", v)
                    }
                    _ => "> []".to_string(),
                };

                let max_str_len = format!("{} -{}-", value, node.links_len[node.level]).len() + 1;

                let mut lvl = self.level_generator.total();
                while lvl > 0 {
                    lvl -= 1;

                    let mut value_len = if lvl <= node.level {
                        format!("{} -{}-", value, node.links_len[lvl])
                    } else {
                        format!("{} -", value)
                    };
                    for _ in 0..(max_str_len - value_len.len()) {
                        value_len.push('-');
                    }

                    let mut dashes = String::new();
                    for _ in 0..value_len.len() {
                        dashes.push('-');
                    }

                    if lvl <= node.level {
                        rows[lvl].push_str(value_len.as_ref());
                    } else {
                        rows[lvl].push_str(dashes.as_ref());
                    }
                }

                if let Some(next) = node.links[0].and_then(|p| p.as_ptr().as_ref()) {
                    node = next;
                } else {
                    break;
                }
            }

            for row in rows.iter().rev() {
                println!("{}", row);
            }
        }
    }
}

// ///////////////////////////////////////////////
// Trait implementation
// ///////////////////////////////////////////////

unsafe impl<T: Send> Send for SkipList<T> {}
unsafe impl<T: Sync> Sync for SkipList<T> {}

impl<T> default::Default for SkipList<T> {
    fn default() -> SkipList<T> {
        SkipList::new()
    }
}

/// This implementation of PartialEq only checks that the *values* are equal; it
/// does not check for equivalence of other features (such as the ordering
/// function and the node levels). Furthermore, this uses `T`'s implementation
/// of PartialEq and *does not* use the owning skiplist's comparison function.
impl<A, B> cmp::PartialEq<SkipList<B>> for SkipList<A>
where
    A: cmp::PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &SkipList<B>) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
    #[allow(clippy::partialeq_ne_impl)]
    #[inline]
    fn ne(&self, other: &SkipList<B>) -> bool {
        self.len != other.len || self.iter().ne(other)
    }
}

impl<T> cmp::Eq for SkipList<T> where T: cmp::Eq {}

impl<A, B> cmp::PartialOrd<SkipList<B>> for SkipList<A>
where
    A: cmp::PartialOrd<B>,
{
    #[inline]
    fn partial_cmp(&self, other: &SkipList<B>) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<T> Ord for SkipList<T>
where
    T: cmp::Ord,
{
    #[inline]
    fn cmp(&self, other: &SkipList<T>) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<T> Extend<T> for SkipList<T> {
    #[inline]
    fn extend<I: iter::IntoIterator<Item = T>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        for element in iterator {
            self.push_back(element);
        }
    }
}

impl<T> ops::Index<usize> for SkipList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).expect("Index out of range")
    }
}

impl<T> ops::IndexMut<usize> for SkipList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of range")
    }
}

impl<T> fmt::Debug for SkipList<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;

        for (i, entry) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", entry)?;
        }
        write!(f, "]")
    }
}

impl<T> fmt::Display for SkipList<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;

        for (i, entry) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", entry)?;
        }
        write!(f, "]")
    }
}

impl<T> iter::IntoIterator for SkipList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}
impl<'a, T> iter::IntoIterator for &'a SkipList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}
impl<'a, T> iter::IntoIterator for &'a mut SkipList<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> iter::FromIterator<T> for SkipList<T> {
    #[inline]
    fn from_iter<I>(iter: I) -> SkipList<T>
    where
        I: iter::IntoIterator<Item = T>,
    {
        let mut skiplist = SkipList::new();
        skiplist.extend(iter);
        skiplist
    }
}

impl<T: Hash> Hash for SkipList<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for elt in self {
            elt.hash(state);
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////
// Tests
// ////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use std::ops::Bound::{self, Excluded, Included, Unbounded};

    use anyhow::Result;
    use rand::{
        Rng,
        distr::{Uniform, uniform::UniformUsize},
    };

    use super::SkipList;

    #[test]
    fn push_front() {
        let mut sl = SkipList::new();
        for i in (1..100).rev() {
            sl.push_front(i);
        }

        assert!(sl.into_iter().eq(1..100));
    }

    #[test]
    fn push_back() {
        let mut sl = SkipList::new();
        for i in 1..100 {
            sl.push_back(i);
        }
        assert!(sl.into_iter().eq(1..100));
    }

    #[test]
    fn insert_rand() -> Result<()> {
        let mut rng = rand::thread_rng();
        let mut sl: SkipList<usize> = SkipList::new();
        let mut vec: Vec<usize> = Vec::new();
        for i in 0..100 {
            let idx = rng.sample(Uniform::new_inclusive(0, i)?);
            sl.insert(i, idx);
            vec.insert(idx, i);
        }
        assert_eq!(sl.into_iter().collect::<Vec<_>>(), vec);
        Ok(())
    }

    #[test]
    fn insert_repeat() {
        let mut sl = SkipList::new();
        let repeat = 10;
        for val in 0..10 {
            for i in 0..repeat {
                sl.insert(val * 10 + i, val * 10);
                sl.check();
            }
        }
    }

    #[test]
    fn remove_rand() -> Result<()> {
        let mut rng = rand::thread_rng();
        let mut v: Vec<i32> = (0..1000).collect();
        let mut sl: SkipList<i32> = (0..1000).collect();
        for i in (0..1000).rev() {
            let idx = rng.sample(Uniform::new_inclusive(0, i)?);
            assert_eq!(sl.remove(idx), v.remove(idx));
        }
        Ok(())
    }

    #[test]
    fn append_test() {}

    #[test]
    fn basic_small() {
        let mut sl: SkipList<i64> = SkipList::new();
        sl.check();
        sl.insert(1, 0);
        sl.check();
        assert_eq!(sl.remove(0), 1);
        sl.check();
        sl.insert(1, 0);
        sl.check();
        sl.insert(2, 1);
        sl.check();
        assert_eq!(sl.remove(0), 1);
        sl.check();
        assert_eq!(sl.remove(0), 2);
        sl.check();
    }

    #[test]
    fn basic_large() {
        let size = 500;
        let mut sl = SkipList::with_capacity(500);
        assert!(sl.is_empty());

        for i in 0..size {
            sl.insert(i, i);
            assert_eq!(sl.len(), i + 1);
        }
        sl.check();

        for i in 0..size {
            assert_eq!(sl.remove(0), i);
            assert_eq!(sl.len(), size - i - 1);
        }
        sl.check();

        for i in 0..size {
            sl = (0..size).collect();
            assert_eq!(sl.remove(i), i);
        }
    }

    #[test]
    fn clear() {
        let mut sl: SkipList<i64> = (0..100).collect();
        assert_eq!(sl.len(), 100);
        sl.clear();
        sl.check();
        assert!(sl.is_empty());
    }

    #[test]
    fn last() {
        let mut sl = SkipList::new();
        assert_eq!(sl.iter().next_back(), None);
        for i in 0..100 {
            sl.push_back(i);
            assert_eq!(sl.iter().next_back(), Some(&i));
        }
    }

    #[test]
    fn iter() {
        let size = 10000;

        let mut sl: SkipList<_> = (0..size).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = usize>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), i);
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sl.iter().copied());
        #[allow(clippy::map_clone)]
        test(size, sl.iter_mut().map(|&mut i| i));
        test(size, sl.into_iter());
    }

    #[test]
    fn iter_rev() {
        let size = 10000;

        let mut sl: SkipList<_> = (0..size).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = usize>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), size - i - 1);
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sl.iter().rev().copied());
        #[allow(clippy::map_clone)]
        test(size, sl.iter_mut().rev().map(|&mut i| i));
        test(size, sl.into_iter().rev());
    }

    #[test]
    fn iter_mixed() {
        let size = 10000;

        let mut sl: SkipList<_> = (0..size).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = usize> + DoubleEndedIterator,
        {
            for i in 0..size / 4 {
                assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
                assert_eq!(iter.next().unwrap(), i);
                assert_eq!(iter.next_back().unwrap(), size - i - 1);
            }
            for i in size / 4..size * 3 / 4 {
                assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
                assert_eq!(iter.next().unwrap(), i);
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sl.iter().copied());
        #[allow(clippy::map_clone)]
        test(size, sl.iter_mut().map(|&mut i| i));
        test(size, sl.into_iter());
    }

    #[test]
    fn range_small() {
        let size = 5;

        let sl: SkipList<_> = (0..size).collect();

        let mut j = 0;
        for (&v, i) in sl.range(Included(2), Unbounded).zip(2..size) {
            assert_eq!(v, i);
            j += 1;
        }
        assert_eq!(j, size - 2);
    }

    #[test]
    fn range_1000() {
        let size = 1000;
        let sl: SkipList<_> = (0..size).collect();

        fn test(sl: &SkipList<usize>, min: Bound<usize>, max: Bound<usize>) {
            let mut values = sl.range(min, max);
            #[allow(clippy::range_plus_one)]
            let mut expects = match (min, max) {
                (Excluded(a), Excluded(b)) => (a + 1)..b,
                (Included(a), Excluded(b)) => a..b,
                (Unbounded, Excluded(b)) => 0..b,
                (Excluded(a), Included(b)) => (a + 1)..(b + 1),
                (Included(a), Included(b)) => a..(b + 1),
                (Unbounded, Included(b)) => 0..(b + 1),
                (Excluded(a), Unbounded) => (a + 1)..1000,
                (Included(a), Unbounded) => a..1000,
                (Unbounded, Unbounded) => 0..1000,
            };

            assert_eq!(values.size_hint(), expects.size_hint());

            for (&v, e) in values.by_ref().zip(expects.by_ref()) {
                assert_eq!(v, e);
            }
            assert!(values.next().is_none());
            assert!(expects.next().is_none());
        }

        test(&sl, Excluded(200), Excluded(800));
        test(&sl, Included(200), Excluded(800));
        test(&sl, Unbounded, Excluded(800));
        test(&sl, Excluded(200), Included(800));
        test(&sl, Included(200), Included(800));
        test(&sl, Unbounded, Included(800));
        test(&sl, Excluded(200), Unbounded);
        test(&sl, Included(200), Unbounded);
        test(&sl, Unbounded, Unbounded);
    }

    #[test]
    fn range_mut_1000() {
        let size = 1000;
        let mut sl: SkipList<_> = (0..size).collect();

        fn test(sl: &mut SkipList<usize>, min: Bound<usize>, max: Bound<usize>) {
            let mut values = sl.range(min, max);
            #[allow(clippy::range_plus_one)]
            let mut expects = match (min, max) {
                (Excluded(a), Excluded(b)) => (a + 1)..b,
                (Included(a), Excluded(b)) => a..b,
                (Unbounded, Excluded(b)) => 0..b,
                (Excluded(a), Included(b)) => (a + 1)..(b + 1),
                (Included(a), Included(b)) => a..(b + 1),
                (Unbounded, Included(b)) => 0..(b + 1),
                (Excluded(a), Unbounded) => (a + 1)..1000,
                (Included(a), Unbounded) => a..1000,
                (Unbounded, Unbounded) => 0..1000,
            };
            assert_eq!(values.size_hint(), expects.size_hint());

            for (&v, e) in values.by_ref().zip(expects.by_ref()) {
                assert_eq!(v, e);
            }
            assert!(values.next().is_none());
            assert!(expects.next().is_none());
        }

        test(&mut sl, Excluded(200), Excluded(800));
        test(&mut sl, Included(200), Excluded(800));
        test(&mut sl, Unbounded, Excluded(800));
        test(&mut sl, Excluded(200), Included(800));
        test(&mut sl, Included(200), Included(800));
        test(&mut sl, Unbounded, Included(800));
        test(&mut sl, Excluded(200), Unbounded);
        test(&mut sl, Included(200), Unbounded);
        test(&mut sl, Unbounded, Unbounded);
    }

    #[test]
    fn range() {
        let size = 200;
        let sl: SkipList<_> = (0..size).collect();

        for i in 0..size {
            for j in 0..size {
                let mut values = sl.range(Included(i), Included(j));
                let mut expects = i..=j;

                for (&v, e) in values.by_ref().zip(expects.by_ref()) {
                    assert_eq!(v, e);
                }
                assert!(values.next().is_none());
                assert!(expects.next().is_none());
            }
        }

        for i in 0..size {
            for j in 0..size {
                let mut values = sl.range(Included(i), Included(j)).rev();
                let mut expects = (i..=j).rev();

                assert_eq!(values.size_hint(), expects.size_hint());

                for (&v, e) in values.by_ref().zip(expects.by_ref()) {
                    assert_eq!(v, e);
                }
                assert!(values.next().is_none());
                assert!(expects.next().is_none());
            }
        }
    }

    #[test]
    fn index_of() {
        let size = 200;
        let sl: SkipList<_> = (0..size).collect();
        for val in 0..size {
            let i = sl.index_of(&val).expect("Index can't be None");
            let item_at_index = sl.get(i).expect("Item can't be None");
            assert_eq!(&val, item_at_index);
        }
    }

    #[test]
    fn bound() {
        let sl: SkipList<_> = (0..2).collect();

        assert_eq!(sl.lower_bound(Unbounded), Some(&0));
        assert_eq!(sl.lower_bound(Excluded(0)), Some(&1));
        assert_eq!(sl.lower_bound(Included(0)), Some(&0));

        assert_eq!(sl.lower_bound(Excluded(2)), None);
        assert_eq!(sl.lower_bound(Included(2)), None);

        assert_eq!(sl.upper_bound(Unbounded), Some(&1));
        assert_eq!(sl.upper_bound(Excluded(1)), Some(&0));
        assert_eq!(sl.upper_bound(Included(1)), Some(&1));
    }

    #[test]
    fn index_pop() {
        let size = 1000;
        let mut sl: SkipList<_> = (0..size).collect();
        assert_eq!(sl.front(), Some(&0));
        assert_eq!(sl.front_mut(), Some(&mut 0));
        assert_eq!(sl.back(), Some(&(size - 1)));
        assert_eq!(sl.back_mut(), Some(&mut (size - 1)));
        for mut i in 0..size {
            assert_eq!(sl[i], i);
            assert_eq!(sl.get(i), Some(&i));
            assert_eq!(sl.get_mut(i), Some(&mut i))
        }

        let mut sl: SkipList<_> = (0..size).collect();
        for i in 0..size {
            assert_eq!(sl.pop_front(), Some(i));
            assert_eq!(sl.len(), size - i - 1);
        }
        assert!(sl.pop_front().is_none());
        assert!(sl.front().is_none());
        assert!(sl.is_empty());

        let mut sl: SkipList<_> = (0..size).collect();
        for i in 0..size {
            assert_eq!(sl.pop_back(), Some(size - i - 1));
            assert_eq!(sl.len(), size - i - 1);
        }
        assert!(sl.pop_back().is_none());
        assert!(sl.back().is_none());
        assert!(sl.is_empty());
    }

    #[test]
    fn contains() {
        let (min, max) = (25, 75);
        let sl: SkipList<_> = (min..max).collect();

        for i in 0..100 {
            if i < min || i >= max {
                assert!(!sl.contains(&i));
            } else {
                assert!(sl.contains(&i));
            }
        }
    }

    #[test]
    fn inplace_mut() {
        let size = 1000;
        let mut sl: SkipList<_> = (0..size).collect();

        for i in 0..size {
            let v = sl.get_mut(i).unwrap();
            *v *= 2;
        }

        for i in 0..size {
            assert_eq!(sl.get(i), Some(&(2 * i)));
        }
    }

    #[test]
    fn dedup() {
        let size = 1000;
        let repeats = 10;

        let mut sl: SkipList<usize> = SkipList::new();
        for i in 0..size {
            for _ in 0..repeats {
                sl.insert(i, i * repeats);
            }
        }
        {
            let mut iter = sl.iter();
            for i in 0..size {
                for _ in 0..repeats {
                    assert_eq!(iter.next(), Some(&i));
                }
            }
        }
        sl.dedup();
        sl.check();
        let mut iter = sl.iter();
        for i in 0..size {
            assert_eq!(iter.next(), Some(&i));
        }
    }

    #[test]
    fn retain() {
        let repeats = 10;
        let size = 100;

        let mut sl: SkipList<usize> = SkipList::new();
        for i in 0..size {
            for _ in 0..repeats {
                sl.insert(i, i * repeats);
                sl.check();
            }
        }
        {
            let mut iter = sl.iter();
            for i in 0..size {
                for _ in 0..repeats {
                    assert_eq!(iter.next(), Some(&i));
                }
            }
        }
        sl.retain(|&x| x % 5 == 0);
        sl.debug_structure();
        sl.check();
        assert_eq!(sl.len(), repeats * size / 5);

        {
            let mut iter = sl.iter();
            for i in 0..size / 5 {
                for _ in 0..repeats {
                    assert_eq!(iter.next(), Some(&(i * 5)));
                }
            }
        }
        sl.retain(|&_| false);
        sl.check();
        assert!(sl.is_empty());
    }

    #[test]
    fn debug_display() {
        let sl: SkipList<_> = (0..10).collect();
        sl.debug_structure();
        println!("{:?}", sl);
        println!("{}", sl);
    }

    #[test]
    #[allow(clippy::eq_op, clippy::many_single_char_names)]
    fn equality() {
        let a: SkipList<i64> = (0..100).collect();
        let b: SkipList<i64> = (0..100).collect();
        let c: SkipList<i64> = (0..10).collect();
        let d: SkipList<i64> = (100..200).collect();
        let e: SkipList<i64> = (0..100).chain(0..1).collect();

        assert_eq!(a, a);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(a, e);
        assert_eq!(b, b);
        assert_ne!(b, c);
        assert_ne!(b, d);
        assert_ne!(b, e);
        assert_eq!(c, c);
        assert_ne!(c, d);
        assert_ne!(c, e);
        assert_eq!(d, d);
        assert_ne!(d, e);
        assert_eq!(e, e);
    }
}
