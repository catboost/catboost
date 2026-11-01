//! An always-ordered skiplist.

#![allow(warnings)]
#![allow(clippy)]
#![allow(unknown_lints)]

use std::{
    cmp,
    cmp::Ordering,
    default, fmt, hash,
    hash::Hash,
    iter, mem, ops,
    ops::{Bound, Range},
    ptr,
};

pub use crate::skipnode::{IntoIter, Iter, IterMut};
use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    skipnode::{SkipListAction, SkipNode, insertion_fixup, removal_fixup},
};

// ////////////////////////////////////////////////////////////////////////////
// OrderedSkipList
// ////////////////////////////////////////////////////////////////////////////

/// The ordered skiplist provides a way of storing elements such that they are
/// always sorted and at the same time provides efficient way to access, insert
/// and remove nodes. Just like `SkipList`, it also provides access to indices.
///
/// By default, the OrderedSkipList uses the comparison function
/// `a.partial_cmp(b).expect("Value cannot be ordered")`.  This allows the list
/// to handles all types which implement `Ord` and `PartialOrd`, though it will
/// panic if value which cannot be ordered is inserted (such as `Float::nan()`).
///
/// The ordered skiplist has an associated sorting function which **must** be
/// well-behaved. Specifically, given some ordering function `f(a, b)`, it must
/// satisfy the following properties:
///
/// - Be well defined: `f(a, b)` should always return the same value
/// - Be anti-symmetric: `f(a, b) == Greater` iff `f(b, a) == Less` and `f(a, b)
///   == Equal == f(b, a)`.
/// - By transitive: If `f(a, b) == Greater` and `f(b, c) == Greater` then `f(a,
///   c) == Greater`.
///
/// **Failure to satisfy these properties can result in unexpected behavior at
/// best, and at worst will cause a segfault, null deref, or some other bad
/// behavior.**
pub struct OrderedSkipList<T> {
    // Storage, this is not sorted
    head: Box<SkipNode<T>>,
    len: usize,
    level_generator: Geometric,
    #[allow(clippy::type_complexity)]
    compare: Box<dyn Fn(&T, &T) -> Ordering>,
}

// ///////////////////////////////////////////////
// Inherent methods
// ///////////////////////////////////////////////

impl<T> OrderedSkipList<T>
where
    T: cmp::PartialOrd,
{
    /// Create a new skiplist with the default default comparison function of
    /// `|&a, &b| a.cmp(b).unwrap()` and the default number of 16 levels.  As a
    /// result, any element which cannot be ordered will cause insertion to
    /// panic.
    ///
    /// The comparison function can always be changed with `sort_by`, which has
    /// essentially no cost if done before inserting any elements.
    ///
    /// # Panic
    ///
    /// The default comparison function will cause a panic if an element is
    /// inserted which cannot be ordered (such as `Float::nan()`).
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist: OrderedSkipList<i64> = OrderedSkipList::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        // Parameters are fixed and will produce a valid level generator.
        let lg = Geometric::new(16, 1.0 / 2.0).expect("Failed to create level generator.");
        OrderedSkipList {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
            compare: (Box::new(|a: &T, b: &T| {
                a.partial_cmp(b).expect("Element cannot be ordered.")
            })) as Box<dyn Fn(&T, &T) -> Ordering>,
        }
    }

    /// Constructs a new, empty skiplist with the optimal number of levels for
    /// the intended capacity.  Specifically, it uses `floor(log2(capacity))`
    /// number of levels, ensuring that only *a few* nodes occupy the highest
    /// level.
    ///
    /// It uses the default comparison function of `|&a, &b| a.cmp(b).unwrap()`
    /// and can be changed with `sort_by`.
    ///
    /// # Panic
    ///
    /// The default comparison function will cause a panic if an element is
    /// inserted which cannot be ordered (such as `Float::nan()`).
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::with_capacity(100);
    /// skiplist.extend(0..100);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let levels = cmp::max(1, (capacity as f64).log2().floor() as usize);
        // Parameters are safe as levels >= 1 and p is in (0, 1).
        let lg = Geometric::new(levels, 1.0 / 2.0).expect("Failed to create level generator.");
        OrderedSkipList {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
            compare: (Box::new(|a: &T, b: &T| {
                a.partial_cmp(b).expect("Element cannot be ordered.")
            })) as Box<dyn Fn(&T, &T) -> Ordering>,
        }
    }
}

impl<T> OrderedSkipList<T>
where
    T: PartialEq,
{
    /// Returns the index of the value if it's contained in the skiplist or None otherwise.
    ///
    /// Since this skip list is ordered, this method has an average complexity of O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.index_of(&4), 4);
    /// ```
    #[inline]
    pub fn index_of(&self, item: &T) -> Option<usize> {
        let cmp = self.compare.as_ref();
        let (node, idx) = self
            .head
            .find_last_le_with(|node_value, target| cmp(node_value, target), item);
        if let Some(node) = &node.item {
            if node.eq(item) {
                return Some(idx - 1);
            }
        }
        None
    }
}

impl<T> OrderedSkipList<T> {
    /// Create a new skiplist using the provided function in order to determine
    /// the ordering of elements within the list.  It will be generated with 16
    /// levels.
    ///
    /// # Safety
    ///
    /// The ordered skiplist relies on a well-behaved comparison function.
    /// Specifically, given some ordering function `f(a, b)`, it **must**
    /// satisfy the following properties:
    ///
    /// - Be well defined: `f(a, b)` should always return the same value
    /// - Be anti-symmetric: `f(a, b) == Greater` if and only if `f(b, a) ==
    ///   Less`, and `f(a, b) == Equal == f(b, a)`.
    /// - By transitive: If `f(a, b) == Greater` and `f(b, c) == Greater` then
    ///   `f(a, c) == Greater`.
    ///
    /// **Failure to satisfy these properties can result in unexpected behavior
    /// at best, and at worst will cause a segfault, null deref, or some other
    /// bad behavior.**
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    /// use std::cmp::Ordering;
    ///
    /// // Store even number before odd ones and sort as usual within same parity group.
    /// let mut skiplist = unsafe { OrderedSkipList::with_comp(
    ///     |a: &u64, b: &u64|
    ///     if a%2 == b%2 {
    ///         a.cmp(b)
    ///     } else if a%2 == 0 {
    ///         Ordering::Less
    ///     } else {
    ///         Ordering::Greater
    ///     })};
    /// ```
    #[inline]
    pub unsafe fn with_comp<F>(f: F) -> Self
    where
        F: 'static + Fn(&T, &T) -> Ordering,
    {
        // Parameters are fixed and will produce a valid level generator.
        let lg = Geometric::new(16, 1.0 / 2.0).expect("Failed to create level generator.");
        OrderedSkipList {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
            compare: Box::new(f),
        }
    }

    /// Change the method which determines the ordering of the elements in the
    /// skiplist.
    ///
    /// # Panics
    ///
    /// This call will panic if the ordering of the elements will be changed as
    /// a result of this new comparison method.
    ///
    /// As a result, `sort_by` is best to call if the skiplist is empty or has
    /// just a single element and may panic with 2 or more elements.
    ///
    /// # Safety
    ///
    /// The ordered skiplist relies on a well-behaved comparison function.
    /// Specifically, given some ordering function `f(a, b)`, it **must**
    /// satisfy the following properties:
    ///
    /// - Be well defined: `f(a, b)` should always return the same value
    /// - Be anti-symmetric: `f(a, b) == Greater` if and only if `f(b, a) ==
    ///   Less`, and `f(a, b) == Equal == f(b, a)`.
    /// - By transitive: If `f(a, b) == Greater` and `f(b, c) == Greater` then
    ///   `f(a, c) == Greater`.
    ///
    /// **Failure to satisfy these properties can result in unexpected behavior
    /// at best, and at worst will cause a segfault, null deref, or some other
    /// bad behavior.**
    ///
    /// # Examples
    ///
    /// ```should_fail
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// unsafe { skiplist.sort_by(|a: &i64, b: &i64| b.cmp(a)) } // All good; skiplist empty.
    /// skiplist.insert(0);                                      // Would still be good here.
    /// skiplist.insert(10);
    /// unsafe { skiplist.sort_by(|a: &i64, b: &i64| a.cmp(b)) } // Panics; order would change.
    /// ```
    pub unsafe fn sort_by<F>(&mut self, cmp: F)
    where
        F: 'static + Fn(&T, &T) -> Ordering,
    {
        assert!(
            self.is_sort(&cmp),
            "The new ordering does not respect the ordering in the list!"
        );
        self.compare = Box::new(cmp);
    }

    /// Clears the skiplist, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
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
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
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
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// assert!(skiplist.is_empty());
    ///
    /// skiplist.insert(1);
    /// assert!(!skiplist.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert the element into the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    ///
    /// skiplist.insert(0);
    /// skiplist.insert(5);
    /// assert_eq!(skiplist.len(), 2);
    /// assert!(!skiplist.is_empty());
    /// ```
    pub fn insert(&mut self, value: T) {
        let new_node = Box::new(SkipNode::new(value, self.level_generator.level()));
        let inserter = OrdInserter::new(self.compare.as_ref(), new_node);
        let _ = inserter.act(self.head.as_mut());
        self.len += 1;
    }

    /// Provides a reference to the front element, or `None` if the skiplist is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// assert!(skiplist.front().is_none());
    ///
    /// skiplist.insert(1);
    /// skiplist.insert(2);
    /// assert_eq!(skiplist.front(), Some(&1));
    /// ```
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self[0])
        }
    }

    /// Provides a reference to the back element, or `None` if the skiplist is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// assert!(skiplist.back().is_none());
    ///
    /// skiplist.insert(1);
    /// skiplist.insert(2);
    /// assert_eq!(skiplist.back(), Some(&2));
    /// ```
    #[inline]
    pub fn back(&self) -> Option<&T> {
        let len = self.len();
        if len > 0 { Some(&self[len - 1]) } else { None }
    }

    /// Provides a reference to the element at the given index, or `None` if the
    /// skiplist is empty or the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// assert!(skiplist.get(0).is_none());
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.get(0), Some(&0));
    /// assert!(skiplist.get(10).is_none());
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        let len = self.len();
        if index < len {
            Some(&self[index])
        } else {
            None
        }
    }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.insert(1);
    /// skiplist.insert(2);
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
            Some(self.remove_index(0))
        }
    }

    /// Removes the last element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.insert(1);
    /// skiplist.insert(2);
    ///
    /// assert_eq!(skiplist.pop_back(), Some(2));
    /// assert_eq!(skiplist.pop_back(), Some(1));
    /// assert!(skiplist.pop_back().is_none());
    /// ```
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        let len = self.len();
        if len > 0 {
            Some(self.remove_index(len - 1))
        } else {
            None
        }
    }

    /// Returns true if the value is contained in the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// assert!(skiplist.contains(&4));
    /// assert!(!skiplist.contains(&15));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        let (last_lt, _) = self.head.find_last_lt_with(&self.compare, value);
        last_lt
            .next_ref()
            .and_then(|node| node.item.as_ref())
            .map_or(false, |node_value| {
                (self.compare)(value, node_value) == Ordering::Equal
            })
    }

    /// Removes and returns an element with the same value or None if there are
    /// no such values in the skiplist.
    ///
    /// If the skiplist contains multiple values with the desired value, the
    /// highest level one will be removed.  This will results in a deterioration
    /// in the skiplist's performance if the skiplist contains *many* duplicated
    /// values which are very frequently inserted and removed. In such
    /// circumstances, the slower `remove_first` method is preferred.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.remove(&4), Some(4)); // Removes the last one
    /// assert!(skiplist.remove(&4).is_none()); // No more '4' left
    /// ```
    pub fn remove(&mut self, value: &T) -> Option<T> {
        let remover = OrdEagerRemover::new(self.compare.as_ref(), value);
        remover.act(self.head.as_mut()).ok().and_then(|node| {
            self.len -= 1;
            node.into_inner()
        })
    }

    /// Removes and returns an element with the same value or None if there are
    /// no such values in the skiplist.
    ///
    /// If the skiplist contains multiple values with the desired value, the
    /// first one in the skiplist will be returned.  If the skiplist contains
    /// *many* duplicated values which are frequently inserted and removed, this
    /// method should be preferred over `remove`.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// for _ in 0..10 {
    ///     skiplist.extend(0..10);
    /// }
    /// assert!(skiplist.remove(&15).is_none());
    /// for _ in 0..9 {
    ///     for i in 0..10 {
    ///         skiplist.remove_first(&i);
    ///     }
    /// }
    /// assert_eq!(skiplist.remove_first(&4), Some(4)); // Removes the last one
    /// assert!(skiplist.remove_first(&4).is_none()); // No more '4' left
    /// ```
    pub fn remove_first(&mut self, value: &T) -> Option<T> {
        let remover = OrdRemover::new(self.compare.as_ref(), value);
        remover.act(self.head.as_mut()).ok().and_then(|node| {
            self.len -= 1;
            node.into_inner()
        })
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
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// assert_eq!(skiplist.remove_index(4), 4);
    /// assert_eq!(skiplist.remove_index(4), 5);
    /// ```
    pub fn remove_index(&mut self, index: usize) -> T {
        let removed_node = self
            .head
            .remove_at(index)
            .unwrap_or_else(|| panic!("Index out of range"));
        self.len -= 1;
        removed_node.into_inner().unwrap()
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns false.
    /// This method operates in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// skiplist.retain(|&x| x%2 == 0);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.len -= self.head.retain(move |_, x| f(x));
    }

    /// Removes all repeated elements in the skiplist using the skiplist's
    /// comparison function.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..5);
    /// skiplist.insert(3);
    /// skiplist.dedup();
    /// ```
    pub fn dedup(&mut self) {
        let cmp = self.compare.as_ref();
        let removed = self.head.retain(|prev, current| {
            prev.map_or(true, |prev| (cmp)(prev, current) != Ordering::Equal)
        });
        self.len -= removed;
    }

    /// Get an owning iterator over the entries of the skiplist.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
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
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.iter() {
    ///     println!("Value: {}", i);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<T> {
        let len = self.len();
        unsafe { Iter::from_head(&self.head, len) }
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
    /// use skiplist::OrderedSkipList;
    /// use std::ops::Bound::{Included, Unbounded};
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.range(Included(&3), Included(&7)) {
    ///     println!("Value: {}", i);
    /// }
    /// assert_eq!(Some(&4), skiplist.range(Included(&4), Unbounded).next());
    /// ```
    pub fn range(&self, min: Bound<&T>, max: Bound<&T>) -> Iter<T> {
        self._range(min, max).unwrap_or(Iter {
            first: None,
            last: None,
            size: 0,
        })
    }

    /// A helper function to ease error handling.
    fn _range(&self, min: Bound<&T>, max: Bound<&T>) -> Option<Iter<T>> {
        let (first, first_distance_from_head) = self._lower_bound(min)?;
        let (last, last_distance_from_head) = self._upper_bound(max);
        let size = last_distance_from_head.checked_sub(first_distance_from_head)? + 1;
        Some(Iter {
            first: Some(first),
            last: Some(last),
            size,
        })
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the
    /// skiplist, starting at min, and ending at max.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    /// for i in skiplist.index_range(3..8) {
    ///     println!("Value: {}", i);
    /// }
    /// assert_eq!(Some(&4), skiplist.index_range(4..5).next());
    /// ```
    pub fn index_range(&self, range: Range<usize>) -> Iter<T> {
        self.iter_range(range.start, range.end)
    }

    /// Makes an iterator between [begin, end]
    fn iter_range(&self, first_idx: usize, last_idx: usize) -> Iter<T> {
        if first_idx >= last_idx {
            return Iter {
                first: None,
                last: None,
                size: 0,
            };
        }
        let first = self.get_index(first_idx);
        let last = self.get_index(last_idx - 1);
        if first.is_some() && last.is_some() {
            Iter {
                first,
                last,
                size: last_idx - first_idx,
            }
        } else {
            Iter {
                first: None,
                last: None,
                size: 0,
            }
        }
    }

    /// Returns an `Option<&T>` pointing to the lowest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    ///
    /// assert_eq!(skiplist.lower_bound(Unbounded), Some(&0));
    /// assert_eq!(skiplist.lower_bound(Excluded(&0)), Some(&1));
    /// assert_eq!(skiplist.lower_bound(Included(&0)), Some(&0));
    /// assert_eq!(skiplist.lower_bound(Excluded(&10)), None);
    /// ```
    pub fn lower_bound(&self, min: Bound<&T>) -> Option<&T> {
        self._lower_bound(min)
            .and_then(|(node, _)| node.item.as_ref())
    }

    /// Returns an `Option<&T>` pointing to the highest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::OrderedSkipList;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skiplist = OrderedSkipList::new();
    /// skiplist.extend(0..10);
    ///
    /// assert_eq!(skiplist.upper_bound(Unbounded), Some(&9));
    /// assert_eq!(skiplist.upper_bound(Excluded(&9)), Some(&8));
    /// assert_eq!(skiplist.upper_bound(Included(&9)), Some(&9));
    /// ```
    pub fn upper_bound(&self, max: Bound<&T>) -> Option<&T> {
        self._upper_bound(max).0.item.as_ref()
    }

    fn _lower_bound(&self, min: Bound<&T>) -> Option<(&SkipNode<T>, usize)> {
        Some(match min {
            Bound::Unbounded => (self.head.next_ref()?, 1usize),
            Bound::Included(min) => {
                let (last_lt, last_lt_from_head) = self.head.find_last_lt_with(&self.compare, min);
                let first_ge = last_lt.next_ref()?;
                (first_ge, last_lt_from_head + 1)
            }
            Bound::Excluded(min) => {
                let (last_le, last_le_from_head) = self.head.find_last_le_with(&self.compare, min);
                let first_gt = last_le.next_ref()?;
                (first_gt, last_le_from_head + 1)
            }
        })
    }

    fn _upper_bound(&self, max: Bound<&T>) -> (&SkipNode<T>, usize) {
        match max {
            Bound::Unbounded => (self.head.last(), self.len()),
            Bound::Included(max) => self.head.find_last_le_with(&self.compare, max),
            Bound::Excluded(max) => self.head.find_last_lt_with(&self.compare, max),
        }
    }
}

// ///////////////////////////////////////////////
// List Actions
// ///////////////////////////////////////////////

struct OrdInserter<F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    cmp: F,
    new_node: Box<SkipNode<T>>,
}

impl<F, T> OrdInserter<F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn new(cmp: F, new_node: Box<SkipNode<T>>) -> Self {
        Self { cmp, new_node }
    }
}

impl<'a, F, T: 'a> SkipListAction<'a, T> for OrdInserter<F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    type Ok = &'a mut SkipNode<T>;

    type Err = Box<SkipNode<T>>;

    fn fail(self) -> Self::Err {
        self.new_node
    }

    fn seek(
        &mut self,
        node: &'a mut SkipNode<T>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<T>, usize)> {
        let value = self.new_node.item.as_ref().unwrap();
        let (node, distance) = node.advance_while_at_level_mut(level, |_current, next| {
            let next_value = next.item.as_ref().unwrap();
            (self.cmp)(value, next_value) == Ordering::Greater
        });
        Some((node, distance))
    }

    // SAEFTY: The new node may never alias with the old nodes.
    unsafe fn act_on_node(self, node: &'a mut SkipNode<T>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            // SAFETY: links will be fixed by the caller.
            Ok(node.insert_next(self.new_node))
        }
    }

    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<T>,
        distance_to_target: usize,
        action_result: &mut Self::Ok,
    ) {
        insertion_fixup(level, level_head, distance_to_target, action_result)
    }
}

struct OrdRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    cmp: F,
    target_value: &'a T,
}

impl<'a, F, T> OrdRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn new(cmp: F, target_value: &'a T) -> Self {
        Self { cmp, target_value }
    }
}

impl<'a, F, T: 'a> SkipListAction<'a, T> for OrdRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    type Ok = Box<SkipNode<T>>;

    type Err = ();

    #[allow(clippy::unused_unit)]
    fn fail(self) -> Self::Err {
        ()
    }

    fn seek(
        &mut self,
        node: &'a mut SkipNode<T>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<T>, usize)> {
        let (target_parent, distance) = node.advance_while_at_level_mut(level, |_, next_node| {
            (self.cmp)(self.target_value, next_node.item.as_ref().unwrap()) == Ordering::Greater
        });
        if level == 0 {
            let next_value = target_parent
                .next_ref()
                .and_then(|node| node.item.as_ref())?;
            if (self.cmp)(self.target_value, next_value) != Ordering::Equal {
                return None;
            }
        }
        Some((target_parent, distance))
    }

    // SAFETY: The removed node will never alias with nodes in the list.
    unsafe fn act_on_node(self, node: &'a mut SkipNode<T>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            // SAFETY: Links will be fixed by the caller.
            node.take_next().ok_or(())
        }
    }

    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<T>,
        _distance_to_target: usize,
        action_result: &mut Self::Ok,
    ) {
        removal_fixup(level, level_head, action_result)
    }
}

struct OrdEagerRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    target_node: *const SkipNode<T>,
    target_value: &'a T,
    cmp: F,
}

impl<'a, F, T> OrdEagerRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    fn new(cmp: F, target_value: &'a T) -> Self {
        Self {
            target_node: ptr::null(),
            target_value,
            cmp,
        }
    }
}

impl<'a, F, T: 'a> SkipListAction<'a, T> for OrdEagerRemover<'a, F, T>
where
    F: Fn(&T, &T) -> Ordering,
{
    type Ok = Box<SkipNode<T>>;

    type Err = ();

    #[allow(clippy::unused_unit)]
    fn fail(self) -> Self::Err {
        ()
    }

    fn seek(
        &mut self,
        node: &'a mut SkipNode<T>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<T>, usize)> {
        if self.target_node.is_null() {
            let (target_parent, distance) =
                node.advance_while_at_level_mut(level, |_, next_node| {
                    (self.cmp)(self.target_value, next_node.item.as_ref().unwrap())
                        == Ordering::Greater
                });
            if level == 0 {
                if let Some(target_value) =
                    target_parent.next_ref().and_then(|node| node.item.as_ref())
                {
                    if (self.cmp)(self.target_value, target_value) == Ordering::Equal {
                        return Some((target_parent, distance));
                    }
                }
                None
            } else {
                if let Some(target_node) =
                    unsafe { target_parent.links[level].and_then(|p| p.as_ptr().as_ref()) }
                {
                    let target_value = target_node.item.as_ref().unwrap();
                    if (self.cmp)(self.target_value, target_value) == Ordering::Equal {
                        self.target_node = target_node as *const _;
                    }
                }
                Some((target_parent, distance))
            }
        } else {
            let (target_parent, distance) = node
                .advance_while_at_level_mut(level, |_, next_node| {
                    !ptr::eq(next_node, self.target_node)
                });
            Some((target_parent, distance))
        }
    }

    // SAFETY: The removed node will never alias with nodes in the list.
    unsafe fn act_on_node(self, node: &'a mut SkipNode<T>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            // SAFETY: Links will be fixed by the caller.
            node.take_next().ok_or(())
        }
    }

    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<T>,
        _distance_to_target: usize,
        action_result: &mut Self::Ok,
    ) {
        removal_fixup(level, level_head, action_result)
    }
}

// ///////////////////////////////////////////////
// Internal methods
// ///////////////////////////////////////////////

impl<T> OrderedSkipList<T> {
    fn is_sort(&self, cmp: impl Fn(&T, &T) -> Ordering) -> bool {
        let mut current_node = self.head.as_ref();
        while let Some(next_node) = current_node.next_ref() {
            if let (Some(current_value), Some(next_value)) =
                (current_node.item.as_ref(), next_node.item.as_ref())
            {
                if (cmp)(current_value, next_value) == Ordering::Greater {
                    return false;
                }
            }
            current_node = next_node;
        }
        true
    }
    /// Checks the integrity of the skiplist.
    #[allow(dead_code)]
    fn check(&self) {
        self.head.check();
        assert!(
            self.is_sort(self.compare.as_ref()),
            "The list isn't properly sorted!"
        );
    }

    /// Gets a pointer to the node with the given index.
    fn get_index(&self, index: usize) -> Option<&SkipNode<T>> {
        if self.len() <= index {
            None
        } else {
            self.head.advance(index + 1)
        }
    }
}

impl<T> OrderedSkipList<T>
where
    T: fmt::Debug,
{
    /// Prints out the internal structure of the skiplist (for debugging
    /// purposes).
    #[allow(dead_code)]
    fn debug_structure(&self) {
        unsafe {
            let mut node: *const SkipNode<T> = mem::transmute_copy(&self.head);
            let mut rows: Vec<_> = iter::repeat(String::new())
                .take(self.level_generator.total())
                .collect();

            loop {
                let value = match (*node).item {
                    Some(ref v) => {
                        format!("> [{:?}]", v)
                    }
                    _ => "> []".to_string(),
                };

                let max_str_len = format!("{} -{}-", value, (*node).links_len[(*node).level]).len();

                let mut lvl = self.level_generator.total();
                while lvl > 0 {
                    lvl -= 1;

                    let mut value_len = if lvl <= (*node).level {
                        format!("{} -{}-", value, (*node).links_len[lvl])
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

                    if lvl <= (*node).level {
                        rows[lvl].push_str(value_len.as_ref());
                    } else {
                        rows[lvl].push_str(dashes.as_ref());
                    }
                }

                match (*node).links[0].and_then(|p| p.as_ptr().as_ref()) {
                    Some(next) => {
                        node = next;
                    }
                    _ => {
                        break;
                    }
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

unsafe impl<T: Send> Send for OrderedSkipList<T> {}
unsafe impl<T: Sync> Sync for OrderedSkipList<T> {}

impl<T: PartialOrd> default::Default for OrderedSkipList<T> {
    fn default() -> OrderedSkipList<T> {
        OrderedSkipList::new()
    }
}

/// This implementation of PartialEq only checks that the *values* are equal; it
/// does not check for equivalence of other features (such as the ordering
/// function and the node levels). Furthermore, this uses `T`'s implementation
/// of PartialEq and *does not* use the owning skiplist's comparison function.
impl<A, B> cmp::PartialEq<OrderedSkipList<B>> for OrderedSkipList<A>
where
    A: cmp::PartialEq<B>,
{
    #[inline]
    fn eq(&self, other: &OrderedSkipList<B>) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
    #[allow(clippy::partialeq_ne_impl)]
    #[inline]
    fn ne(&self, other: &OrderedSkipList<B>) -> bool {
        self.len != other.len || self.iter().ne(other)
    }
}

impl<T> cmp::Eq for OrderedSkipList<T> where T: cmp::Eq {}

impl<A, B> cmp::PartialOrd<OrderedSkipList<B>> for OrderedSkipList<A>
where
    A: cmp::PartialOrd<B>,
{
    #[inline]
    fn partial_cmp(&self, other: &OrderedSkipList<B>) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<T> Ord for OrderedSkipList<T>
where
    T: cmp::Ord,
{
    #[inline]
    fn cmp(&self, other: &OrderedSkipList<T>) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<T> Extend<T> for OrderedSkipList<T> {
    #[inline]
    fn extend<I: iter::IntoIterator<Item = T>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        for element in iterator {
            self.insert(element);
        }
    }
}

impl<T> ops::Index<usize> for OrderedSkipList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get_index(index)
            .and_then(|node| node.item.as_ref())
            .expect("Index out of range")
    }
}

impl<T> fmt::Debug for OrderedSkipList<T>
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

impl<T> fmt::Display for OrderedSkipList<T>
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

impl<T> iter::IntoIterator for OrderedSkipList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}
impl<'a, T> iter::IntoIterator for &'a OrderedSkipList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}
impl<'a, T> iter::IntoIterator for &'a mut OrderedSkipList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T> iter::FromIterator<T> for OrderedSkipList<T>
where
    T: PartialOrd,
{
    #[inline]
    fn from_iter<I>(iter: I) -> OrderedSkipList<T>
    where
        I: iter::IntoIterator<Item = T>,
    {
        let mut skiplist = OrderedSkipList::new();
        skiplist.extend(iter);
        skiplist
    }
}

impl<T: Hash> Hash for OrderedSkipList<T> {
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
    use std::{
        cmp::Ordering,
        ops::Bound::{self, Excluded, Included, Unbounded},
    };

    use super::OrderedSkipList;

    #[test]
    fn basic_small() {
        let mut sl: OrderedSkipList<i64> = OrderedSkipList::new();
        sl.check();
        assert!(sl.remove(&1).is_none());
        sl.check();
        sl.insert(1);
        sl.check();
        assert_eq!(sl.remove(&1), Some(1));
        sl.check();
        sl.insert(1);
        sl.check();
        sl.insert(2);
        sl.check();
        assert_eq!(sl.remove(&1), Some(1));
        sl.check();
        assert_eq!(sl.remove(&2), Some(2));
        sl.check();
        assert!(sl.remove(&1).is_none());
        sl.check();
    }

    #[test]
    fn basic_large() {
        let size = 10_000;
        let mut sl = OrderedSkipList::with_capacity(size);
        assert!(sl.is_empty());

        for i in 0..size {
            sl.insert(i);
            assert_eq!(sl.len(), i + 1);
        }
        sl.check();

        for i in 0..size {
            assert_eq!(sl.remove(&i), Some(i));
            assert_eq!(sl.len(), size - i - 1);
        }
        sl.check();
    }

    #[test]
    fn iter() {
        let size = 10000;

        let sl: OrderedSkipList<_> = (0..size).collect();

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
        test(size, sl.into_iter());
    }

    #[test]
    fn iter_rev() {
        let size = 10000;

        let sl: OrderedSkipList<_> = (0..size).collect();

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
        test(size, sl.into_iter().rev());
    }

    #[test]
    fn iter_mixed() {
        let size = 10000;

        let sl: OrderedSkipList<_> = (0..size).collect();

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
        test(size, sl.into_iter());
    }

    #[test]
    fn with_comp() {
        let mut sl = unsafe {
            OrderedSkipList::with_comp(|a: &u64, b: &u64| {
                if a % 2 == b % 2 {
                    a.cmp(b)
                } else if a % 2 == 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
        };

        for i in 0..100 {
            sl.insert(i);
        }
        sl.check();

        let expect = (0..100)
            .filter(|i| i % 2 == 0)
            .chain((0..100).filter(|i| i % 2 == 1));

        for (&v, e) in sl.iter().zip(expect) {
            assert_eq!(v, e);
        }
    }

    #[test]
    fn sort_by() {
        // Change sort_by when empty
        let mut sl = OrderedSkipList::new();
        unsafe {
            sl.sort_by(|a: &u64, b: &u64| {
                if a % 2 == b % 2 {
                    a.cmp(b)
                } else if a % 2 == 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });
        };

        for i in 0..100 {
            sl.insert(i);
        }
        sl.check();

        let expect = (0..100)
            .filter(|i| i % 2 == 0)
            .chain((0..100).filter(|i| i % 2 == 1));

        for (&v, e) in sl.iter().zip(expect) {
            assert_eq!(v, e);
        }

        // Change sort_by in non-empty (but valid) skiplist
        let mut sl = OrderedSkipList::new();
        sl.insert(0);
        sl.insert(2);
        sl.insert(5);
        sl.insert(7);
        unsafe {
            sl.sort_by(|a: &u64, b: &u64| {
                if a % 2 == b % 2 {
                    a.cmp(b)
                } else if a % 2 == 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });
        };
        sl.check();
        sl.insert(4);
        sl.insert(6);
        sl.insert(3);

        for (v, e) in sl.iter().zip(&[0, 2, 4, 6, 3, 5, 7]) {
            assert_eq!(v, e);
        }
    }

    #[test]
    #[should_panic]
    fn sort_by_panic() {
        let mut sl = OrderedSkipList::new();
        sl.insert(0);
        sl.insert(1);
        sl.insert(2);
        unsafe {
            sl.sort_by(|a: &u64, b: &u64| {
                if a % 2 == b % 2 {
                    a.cmp(b)
                } else if a % 2 == 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });
        };
    }

    #[test]
    fn clear() {
        let mut sl: OrderedSkipList<i64> = (0..100).collect();
        assert_eq!(sl.len(), 100);
        sl.clear();
        sl.check();
        assert!(sl.is_empty());
    }

    #[test]
    fn range_small() {
        let size = 5;
        let sl: OrderedSkipList<_> = (0..size).collect();

        let mut j = 0;
        for (&v, i) in sl.range(Included(&2), Unbounded).zip(2..size) {
            assert_eq!(v, i);
            j += 1;
        }
        assert_eq!(j, size - 2);
    }

    #[test]
    fn range_1000() {
        let size = 1000;
        let sl: OrderedSkipList<_> = (0..size).collect();

        fn test(sl: &OrderedSkipList<u32>, min: Bound<&u32>, max: Bound<&u32>) {
            let mut values = sl.range(min, max);
            #[allow(clippy::range_plus_one)]
            let mut expects = match (min, max) {
                (Excluded(&a), Excluded(&b)) => (a + 1)..b,
                (Included(&a), Excluded(&b)) => a..b,
                (Unbounded, Excluded(&b)) => 0..b,
                (Excluded(&a), Included(&b)) => (a + 1)..(b + 1),
                (Included(&a), Included(&b)) => a..(b + 1),
                (Unbounded, Included(&b)) => 0..(b + 1),
                (Excluded(&a), Unbounded) => (a + 1)..1000,
                (Included(&a), Unbounded) => a..1000,
                (Unbounded, Unbounded) => 0..1000,
            };

            assert_eq!(values.size_hint(), expects.size_hint());

            for (&v, e) in values.by_ref().zip(expects.by_ref()) {
                assert_eq!(v, e);
            }
            assert!(values.next().is_none());
            assert!(expects.next().is_none());
        }

        test(&sl, Excluded(&200), Excluded(&800));
        test(&sl, Included(&200), Excluded(&800));
        test(&sl, Unbounded, Excluded(&800));
        test(&sl, Excluded(&200), Included(&800));
        test(&sl, Included(&200), Included(&800));
        test(&sl, Unbounded, Included(&800));
        test(&sl, Excluded(&200), Unbounded);
        test(&sl, Included(&200), Unbounded);
        test(&sl, Unbounded, Unbounded);
    }

    #[test]
    fn range() {
        let size = 200;
        let sl: OrderedSkipList<_> = (0..size).collect();

        for i in 0..size {
            for j in 0..size {
                let mut values = sl.range(Included(&i), Included(&j));
                let mut expects = i..=j;

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
    fn index_range() {
        let size = 200;
        let sl: OrderedSkipList<_> = (0..size).collect();
        for i in 0..size {
            for j in 0..size {
                let mut values = sl.index_range(i..j);
                let mut expects = i..j;

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
        let sl: OrderedSkipList<_> = (0..size).collect();
        for val in 0..size {
            let i = sl.index_of(&val).expect("Index can't be None");
            let item_at_index = sl.get(i).expect("Item can't be None");
            assert_eq!(&val, item_at_index);
        }
    }

    #[test]
    fn bound() {
        let sl: OrderedSkipList<_> = (1..3).collect();

        assert_eq!(sl.lower_bound(Unbounded), Some(&1));
        assert_eq!(sl.lower_bound(Excluded(&1)), Some(&2));
        assert_eq!(sl.lower_bound(Included(&1)), Some(&1));

        assert_eq!(sl.lower_bound(Excluded(&3)), None);
        assert_eq!(sl.lower_bound(Included(&3)), None);

        assert_eq!(sl.upper_bound(Unbounded), Some(&2));
        assert_eq!(sl.upper_bound(Excluded(&2)), Some(&1));
        assert_eq!(sl.upper_bound(Included(&2)), Some(&2));

        assert_eq!(sl.upper_bound(Excluded(&0)), None);
        assert_eq!(sl.upper_bound(Included(&0)), None);
    }

    #[test]
    fn index_pop() {
        let size = 1000;
        let sl: OrderedSkipList<_> = (0..size).collect();
        assert_eq!(sl.front(), Some(&0));
        assert_eq!(sl.back(), Some(&(size - 1)));
        for i in 0..size {
            assert_eq!(sl[i], i);
            assert_eq!(sl.get(i), Some(&i));
        }

        let mut sl: OrderedSkipList<_> = (0..size).collect();
        for i in 0..size {
            assert_eq!(sl.pop_front(), Some(i));
            assert_eq!(sl.len(), size - i - 1);
        }
        assert!(sl.pop_front().is_none());
        assert!(sl.front().is_none());
        assert!(sl.is_empty());

        let mut sl: OrderedSkipList<_> = (0..size).collect();
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
        let sl: OrderedSkipList<_> = (min..max).collect();

        for i in 0..100 {
            if i < min || i >= max {
                assert!(!sl.contains(&i));
            } else {
                assert!(sl.contains(&i));
            }
        }
    }

    #[test]
    fn remove() {
        let size = 100;
        let repeats = 5;
        let mut sl = OrderedSkipList::new();

        for _ in 0..repeats {
            sl.extend(0..size);
        }

        for _ in 0..repeats {
            for i in 0..size {
                assert_eq!(sl.remove(&i), Some(i));
            }
        }
        for i in 0..size {
            assert!(sl.remove(&i).is_none());
        }

        for _ in 0..repeats {
            sl.extend(0..size);
        }
        for _ in 0..repeats {
            for i in 0..size {
                assert_eq!(sl.remove_first(&i), Some(i));
            }
        }
        for i in 0..size {
            assert!(sl.remove_first(&i).is_none());
        }
    }

    #[test]
    fn dedup() {
        let size = 1000;
        let repeats = 10;

        let mut sl: OrderedSkipList<usize> = OrderedSkipList::new();
        for _ in 0..repeats {
            sl.extend(0..size);
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
        let size = 1000;
        let mut sl: OrderedSkipList<usize> = OrderedSkipList::new();
        for _ in 0..repeats {
            sl.extend(0..size);
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
    fn remove_index() {
        let size = 100;

        for i in 0..size {
            let mut sl: OrderedSkipList<_> = (0..size).collect();
            assert_eq!(sl.remove_index(i), i);
            assert_eq!(sl.len(), size - 1);
        }

        let mut sl: OrderedSkipList<_> = (0..size).collect();
        for i in 0..size {
            assert_eq!(sl.remove_index(0), i);
            assert_eq!(sl.len(), size - i - 1);
            sl.check();
        }
        assert!(sl.is_empty());
    }

    #[test]
    fn pop() {
        let size = 1000;
        let mut sl: OrderedSkipList<_> = (0..size).collect();
        for i in 0..size / 2 {
            assert_eq!(sl.pop_front(), Some(i));
            assert_eq!(sl.pop_back(), Some(size - i - 1));
            assert_eq!(sl.len(), size - 2 * (i + 1));
            sl.check();
        }
        assert!(sl.is_empty());
    }

    #[test]
    fn debug_display() {
        let sl: OrderedSkipList<_> = (0..10).collect();
        sl.debug_structure();
        println!("{:?}", sl);
        println!("{}", sl);
    }

    #[test]
    #[allow(clippy::eq_op, clippy::many_single_char_names)]
    fn equality() {
        let a: OrderedSkipList<i64> = (0..100).collect();
        let b: OrderedSkipList<i64> = (0..100).collect();
        let c: OrderedSkipList<i64> = (0..10).collect();
        let d: OrderedSkipList<i64> = (100..200).collect();
        let e: OrderedSkipList<i64> = (0..100).chain(0..1).collect();

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
