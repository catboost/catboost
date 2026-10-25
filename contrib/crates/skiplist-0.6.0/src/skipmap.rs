//! SkipMap stores key-value pairs, with the keys being unique and always
//! sorted.

#![allow(warnings)]
#![allow(clippy)]
#![allow(unknown_lints)]

use std::{
    borrow::Borrow, cmp, cmp::Ordering, default, fmt, hash, hash::Hash, iter, mem, ops, ops::Bound,
};

pub use crate::skipnode::IntoIter;
use crate::{
    level_generator::{LevelGenerator, geometric::Geometric},
    skipnode::{self, SkipListAction, insertion_fixup},
};

type SkipNode<K, V> = skipnode::SkipNode<(K, V)>;

impl<K, V> SkipNode<K, V> {
    fn key_ref(&self) -> Option<&K> {
        self.item.as_ref().map(|item| &item.0)
    }

    fn value_ref(&self) -> Option<&V> {
        self.item.as_ref().map(|item| &item.1)
    }

    fn value_mut(&mut self) -> Option<&mut V> {
        self.item.as_mut().map(|item| &mut item.1)
    }

    fn item_ref(&self) -> Option<(&K, &V)> {
        self.item.as_ref().map(|item| (&item.0, &item.1))
    }

    fn item_mut(&mut self) -> Option<(&K, &mut V)> {
        self.item.as_mut().map(|item| (&item.0, &mut item.1))
    }
}

// ////////////////////////////////////////////////////////////////////////////
// SkipMap
// ////////////////////////////////////////////////////////////////////////////

/// The skipmap provides a way of storing element pairs such that they keys are
/// always sorted whilst at the same time providing efficient way to access,
/// insert and removes nodes.
///
/// A particular node can be accessed through the matching key, and since the
/// keys are always sorted, it is also possible to access key-value pairs by
/// index.
///
/// Note that mutable references to keys are not available at all as this could
/// result in a node being left out of the proper ordering.
pub struct SkipMap<K, V> {
    // Storage, this is not sorted
    head: Box<SkipNode<K, V>>,
    len: usize,
    level_generator: Geometric,
}

// ///////////////////////////////////////////////
// Inherent methods
// ///////////////////////////////////////////////

impl<K, V> SkipMap<K, V>
where
    K: cmp::Ord,
{
    /// Create a new skipmap with the default number of 16 levels.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap: SkipMap<i64, String> = SkipMap::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        // Parameters are fixed and will produce a valid level generator.
        let lg = Geometric::new(16, 1.0 / 2.0).expect("Failed to create level generator.");
        SkipMap {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
        }
    }

    /// Constructs a new, empty skipmap with the optimal number of levels for
    /// the intended capacity.  Specifically, it uses `floor(log2(capacity))`
    /// number of levels, ensuring that only *a few* nodes occupy the highest
    /// level.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::with_capacity(100);
    /// skipmap.extend((0..100).map(|x| (x, x)));
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let levels = cmp::max(1, (capacity as f64).log2().floor() as usize);
        // Parameters are safe as levels >= 1 and p is in (0, 1).
        let lg = Geometric::new(levels, 1.0 / 2.0).expect("Failed to create level generator.");
        SkipMap {
            head: Box::new(SkipNode::head(lg.total())),
            len: 0,
            level_generator: lg,
        }
    }

    /// Insert the element into the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    ///
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    /// assert_eq!(skipmap.len(), 2);
    /// assert!(!skipmap.is_empty());
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let level_gen = &mut self.level_generator;
        let inserter = InsertOrReplace::new(key, value, |k, v| {
            Box::new(SkipNode::new((k, v), level_gen.level()))
        });
        let insert_res = inserter.act(self.head.as_mut());
        match insert_res {
            Ok(_) => {
                self.len += 1;
                None
            }
            Err(old_val) => Some(old_val),
        }
    }
}

impl<K, V> SkipMap<K, V> {
    /// Clears the skipmap, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// skipmap.clear();
    /// assert!(skipmap.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        *self.head = SkipNode::head(self.level_generator.total());
    }

    /// Returns the number of elements in the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert_eq!(skipmap.len(), 10);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the skipmap contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.is_empty());
    ///
    /// skipmap.insert(1, "Rust");
    /// assert!(!skipmap.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Provides a reference to the front element, or `None` if the skipmap is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.front().is_none());
    ///
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    /// assert_eq!(skipmap.front(), Some((&1, &"Hello")));
    /// ```
    #[inline]
    pub fn front(&self) -> Option<(&K, &V)> {
        self.get_index(0).and_then(|node| node.item_ref())
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// skipmap is empty.
    ///
    /// The reference to the key remains immutable as the keys must remain
    /// sorted.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.front().is_none());
    ///
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    /// assert_eq!(skipmap.front_mut(), Some((&1, &mut "Hello")));
    /// ```
    #[inline]
    pub fn front_mut(&mut self) -> Option<(&K, &mut V)> {
        self.get_index_mut(0).and_then(|node| node.item_mut())
    }

    /// Provides a reference to the back element, or `None` if the skipmap is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.back().is_none());
    ///
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    /// assert_eq!(skipmap.back(), Some((&2, &"World")));
    /// ```
    #[inline]
    pub fn back(&self) -> Option<(&K, &V)> {
        self.head.last().item_ref()
    }

    /// Provides a reference to the back element, or `None` if the skipmap is
    /// empty.
    ///
    /// The reference to the key remains immutable as the keys must remain
    /// sorted.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.back().is_none());
    ///
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    /// assert_eq!(skipmap.back_mut(), Some((&2, &mut "World")));
    /// ```
    #[inline]
    pub fn back_mut(&mut self) -> Option<(&K, &mut V)> {
        self.head.last_mut().item_mut()
    }

    /// Provides a reference to the element at the given index, or `None` if the
    /// skipmap is empty or the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.get(&0).is_none());
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert_eq!(skipmap.get(&0), Some(&0));
    /// assert!(skipmap.get(&10).is_none());
    /// ```
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_key(key).and_then(|node| node.value_ref())
    }

    /// Provides a reference to the element at the given index, or `None` if the
    /// skipmap is empty or the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// assert!(skipmap.get(&0).is_none());
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert_eq!(skipmap.get_mut(&0), Some(&mut 0));
    /// assert!(skipmap.get_mut(&10).is_none());
    ///
    /// match skipmap.get_mut(&0) {
    ///     Some(x) => *x = 100,
    ///     None => (),
    /// }
    /// assert_eq!(skipmap.get(&0), Some(&100));
    /// ```
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_key_mut(key).and_then(|node| node.value_mut())
    }

    /// Removes the first element and returns it, or `None` if the sequence is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    ///
    /// assert_eq!(skipmap.pop_front(), Some((1, "Hello")));
    /// assert_eq!(skipmap.pop_front(), Some((2, "World")));
    /// assert!(skipmap.pop_front().is_none());
    /// ```
    #[inline]
    pub fn pop_front(&mut self) -> Option<(K, V)> {
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
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.insert(1, "Hello");
    /// skipmap.insert(2, "World");
    ///
    /// assert_eq!(skipmap.pop_back(), Some((2, "World")));
    /// assert_eq!(skipmap.pop_back(), Some((1, "Hello")));
    /// assert!(skipmap.pop_back().is_none());
    /// ```
    #[inline]
    pub fn pop_back(&mut self) -> Option<(K, V)> {
        let len = self.len();
        if len > 0 {
            Some(self.remove_index(len - 1))
        } else {
            None
        }
    }

    /// Returns true if the value is contained in the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert!(skipmap.contains_key(&4));
    /// assert!(!skipmap.contains_key(&15));
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.find_key(key).is_some()
    }

    /// Removes and returns an element with the same value or None if there are
    /// no such values in the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert_eq!(skipmap.remove(&4), Some(4)); // Removes the last one
    /// assert!(skipmap.remove(&4).is_none());    // No more '4' left
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let remover = Remover(key);
        match remover.act(self.head.as_mut()) {
            Ok(node) => {
                self.len -= 1;
                node.into_inner().map(|(_key, val)| val)
            }
            Err(_) => None,
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
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// assert_eq!(skipmap.remove_index(4), (4, 4));
    /// assert_eq!(skipmap.remove_index(4), (5, 5));
    /// ```
    pub fn remove_index(&mut self, index: usize) -> (K, V) {
        if index >= self.len() {
            panic!("Index out of bounds.");
        } else {
            let node = self.head.remove_at(index).unwrap();
            self.len -= 1;
            node.into_inner().unwrap()
        }
    }

    /// Get an owning iterator over the entries of the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for (k, v) in skipmap.into_iter() {
    ///     println!("Key {}, Value: {}", k, v);
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(mut self) -> IntoIter<(K, V)> {
        let len = self.len();
        unsafe { IntoIter::from_head(&mut self.head, len) }
    }

    /// Creates an iterator over the entries of the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for (k, v) in skipmap.iter() {
    ///     println!("Key: {}, Value: {}", k, v);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V> {
        let len = self.len();
        unsafe { Iter::from_head(&self.head, len) }
    }

    /// Creates an mutable iterator over the entries of the skipmap.
    ///
    /// The keys cannot be modified as they must remain in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for (k, v) in skipmap.iter_mut() {
    ///     println!("Key: {}, Value: {}", k, v);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        let len = self.len();
        unsafe { IterMut::from_head(&mut self.head, len) }
    }

    /// Creates an iterator over the keys of the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for k in skipmap.keys() {
    ///     println!("Key: {}", k);
    /// }
    /// ```
    pub fn keys(&self) -> Keys<K, V> {
        Keys(self.iter())
    }

    /// Creates an iterator over the values of the skipmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for v in skipmap.values() {
    ///     println!("Value: {}", v);
    /// }
    /// ```
    pub fn values(&self) -> Values<K, V> {
        Values(self.iter())
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the
    /// skipmap, starting at min, and ending at max. If min is `Unbounded`, then
    /// it will be treated as "negative infinity", and if max is `Unbounded`,
    /// then it will be treated as "positive infinity".  Thus range(Unbounded,
    /// Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    /// use std::ops::Bound::{Included, Unbounded};
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    /// for (k, v) in skipmap.range(Included(&3), Included(&7)) {
    ///     println!("Key: {}, Value: {}", k, v);
    /// }
    /// assert_eq!(Some((&4, &4)), skipmap.range(Included(&4), Unbounded).next());
    /// ```
    pub fn range<Q>(&self, min: Bound<&Q>, max: Bound<&Q>) -> Iter<K, V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let iter_inner = self._range(min, max).unwrap_or(skipnode::Iter {
            first: None,
            last: None,
            size: 0,
        });
        Iter(iter_inner)
    }

    fn _range<Q>(&self, min: Bound<&Q>, max: Bound<&Q>) -> Option<skipnode::Iter<(K, V)>>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        let (first, first_distance_from_head) = self._lower_bound(min)?;
        let (last, last_distance_from_head) = self._upper_bound(max);
        let size = last_distance_from_head.checked_sub(first_distance_from_head)? + 1;
        Some(skipnode::Iter {
            first: Some(first),
            last: Some(last),
            size,
        })
    }

    /// Returns an `Option<&V>` pointing to the lowest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    ///
    /// assert_eq!(skipmap.lower_bound(Unbounded), Some((&0, &0)));
    /// assert_eq!(skipmap.lower_bound(Excluded(&0)), Some((&1, &1)));
    /// assert_eq!(skipmap.lower_bound(Included(&0)), Some((&0, &0)));
    /// assert_eq!(skipmap.lower_bound(Included(&10)), None);
    /// ```
    pub fn lower_bound<Q>(&self, min: Bound<&Q>) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self._lower_bound(min).and_then(|(node, _)| node.item_ref())
    }

    /// Returns an `Option<&V>` pointing to the highest element whose key is above
    /// the given bound. If no such element is found then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use skiplist::SkipMap;
    /// use std::ops::Bound::{Included, Excluded, Unbounded};
    ///
    /// let mut skipmap = SkipMap::new();
    /// skipmap.extend((0..10).map(|x| (x, x)));
    ///
    /// assert_eq!(skipmap.upper_bound(Unbounded), Some((&9, &9)));
    /// assert_eq!(skipmap.upper_bound(Excluded(&9)), Some((&8, &8)));
    /// assert_eq!(skipmap.upper_bound(Included(&9)), Some((&9, &9)));
    /// assert_eq!(skipmap.upper_bound(Excluded(&0)), None);
    /// ```
    pub fn upper_bound<Q>(&self, max: Bound<&Q>) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self._upper_bound(max).0.item_ref()
    }

    fn _lower_bound<Q>(&self, min: Bound<&Q>) -> Option<(&SkipNode<K, V>, usize)>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        Some(match min {
            Bound::Unbounded => (self.head.next_ref()?, 1usize),
            Bound::Included(min) => {
                let (last_lt, last_lt_from_head) = self.head.find_last_lt_with(cmp, min);
                let first_ge = last_lt.next_ref()?;
                (first_ge, last_lt_from_head + 1)
            }
            Bound::Excluded(min) => {
                let (last_le, last_le_from_head) = self.head.find_last_le_with(cmp, min);
                let first_gt = last_le.next_ref()?;
                (first_gt, last_le_from_head + 1)
            }
        })
    }

    fn _upper_bound<Q>(&self, max: Bound<&Q>) -> (&SkipNode<K, V>, usize)
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        match max {
            Bound::Unbounded => (self.head.last(), self.len()),
            Bound::Included(max) => self.head.find_last_le_with(cmp, max),
            Bound::Excluded(max) => self.head.find_last_lt_with(cmp, max),
        }
    }
}

fn cmp<Q: Ord, K: Borrow<Q>, V>(node_item: &(K, V), target: &Q) -> Ordering {
    node_item.0.borrow().cmp(target)
}

// ///////////////////////////////////////////////
// Internal methods
// ///////////////////////////////////////////////

impl<K: Ord, V> SkipMap<K, V> {
    /// Checks the integrity of the skipmap.
    #[allow(dead_code)]
    fn check(&self) {
        self.head.check();
        if let Some(mut node) = self.head.next_ref() {
            let mut key = node.key_ref().unwrap();
            while let Some(next_node) = node.next_ref() {
                let next_key = next_node.key_ref().unwrap();
                assert!(key <= next_key);
                node = next_node;
                key = next_key;
            }
        }
    }
}

impl<K, V> SkipMap<K, V> {
    /// Find the reference to the node equal to the given key.
    fn find_key<Q>(&self, key: &Q) -> Option<&SkipNode<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (last_le, _) = self.head.find_last_le_with(
            |(node_key, _), target| Ord::cmp(node_key.borrow(), target),
            key,
        );
        let node_key = last_le.key_ref()?;
        if node_key.borrow() == key {
            Some(last_le)
        } else {
            None
        }
    }

    /// Find the mutable reference to the node equal to the given key.
    fn find_key_mut<Q>(&mut self, key: &Q) -> Option<&mut SkipNode<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (last_le, _) = self.head.find_last_le_with_mut(
            |(node_key, _), target| Ord::cmp(node_key.borrow(), target),
            key,
        );
        let node_key = last_le.key_ref()?;
        if node_key.borrow() == key {
            Some(last_le)
        } else {
            None
        }
    }

    /// Gets a reference to the node with the given index.
    fn get_index(&self, index: usize) -> Option<&SkipNode<K, V>> {
        self.head.advance(index + 1)
    }

    /// Gets a mutable reference to the node with the given index.
    fn get_index_mut(&mut self, index: usize) -> Option<&mut SkipNode<K, V>> {
        self.head.advance_mut(index + 1)
    }
}

impl<K, V> SkipMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    /// Prints out the internal structure of the skipmap (for debugging
    /// purposes).
    #[allow(dead_code)]
    fn debug_structure(&self) {
        unsafe {
            let mut node: *const SkipNode<K, V> = mem::transmute_copy(&self.head);
            let mut rows: Vec<_> = iter::repeat(String::new())
                .take(self.level_generator.total())
                .collect();

            loop {
                let value = match ((*node).key_ref(), (*node).value_ref()) {
                    (Some(k), Some(v)) => {
                        format!("> ({:?}, {:?})", k, v)
                    }
                    _ => "> ()".to_string(),
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
// List Actions
// ///////////////////////////////////////////////

struct InsertOrReplace<K, V, MakeNode>
where
    K: Ord,
    MakeNode: FnOnce(K, V) -> Box<SkipNode<K, V>>,
{
    key: K,
    value: V,
    make_node: MakeNode,
}

impl<K, V, MakeNode> InsertOrReplace<K, V, MakeNode>
where
    K: Ord,
    MakeNode: FnOnce(K, V) -> Box<SkipNode<K, V>>,
{
    fn new(key: K, value: V, make_node: MakeNode) -> Self {
        Self {
            key,
            value,
            make_node,
        }
    }
}

impl<'a, K: 'a, V: 'a, MakeNode> SkipListAction<'a, (K, V)> for InsertOrReplace<K, V, MakeNode>
where
    K: Ord,
    MakeNode: FnOnce(K, V) -> Box<SkipNode<K, V>>,
{
    type Ok = &'a mut SkipNode<K, V>;
    type Err = V;
    fn fail(self) -> Self::Err {
        panic!("This action should never fail")
    }
    fn seek(
        &mut self,
        node: &'a mut SkipNode<K, V>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<K, V>, usize)> {
        Some(node.advance_while_at_level_mut(level, |_, next_node| {
            let next_key = next_node.key_ref().unwrap();
            let target_key = &self.key;
            next_key < target_key
        }))
    }

    unsafe fn act_on_node(self, node: &'a mut SkipNode<K, V>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            let target_key = &self.key;
            if let Some(target_node) = node.next_mut() {
                if let Some(node_key) = target_node.key_ref() {
                    if target_key == node_key {
                        let old_value = mem::replace(target_node.value_mut().unwrap(), self.value);
                        return Err(old_value);
                    }
                }
            }
            let new_node = (self.make_node)(self.key, self.value);
            node.insert_next(new_node);
            Ok(node.next_mut().unwrap())
        }
    }
    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<K, V>,
        distance_to_target: usize,
        action_result: &mut Self::Ok,
    ) {
        insertion_fixup(level, level_head, distance_to_target, action_result)
    }
}

struct Remover<'a, Q: ?Sized>(&'a Q);

impl<'a, Q: Ord + ?Sized, K: Borrow<Q>, V> SkipListAction<'a, (K, V)> for Remover<'a, Q> {
    type Ok = Box<SkipNode<K, V>>;
    type Err = ();
    #[allow(clippy::unused_unit)]
    fn fail(self) -> Self::Err {
        ()
    }

    fn seek(
        &mut self,
        node: &'a mut SkipNode<K, V>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<K, V>, usize)> {
        Some(node.advance_while_at_level_mut(level, |_, next_node| {
            let next_key = next_node.key_ref().unwrap().borrow();
            let target_key = self.0;
            next_key < target_key
        }))
    }

    unsafe fn act_on_node(
        self,
        target_parent: &'a mut SkipNode<K, V>,
    ) -> Result<Self::Ok, Self::Err> {
        unsafe {
            let node_key = target_parent
                .next_mut()
                .and_then(|node| node.key_ref())
                .ok_or(())?
                .borrow();
            let target_key = self.0;
            if node_key == target_key {
                Ok(target_parent.take_next().unwrap())
            } else {
                Err(())
            }
        }
    }
    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<K, V>,
        _distance: usize,
        action_result: &mut Self::Ok,
    ) {
        skipnode::removal_fixup(level, level_head, action_result)
    }
}

// ///////////////////////////////////////////////
// Trait implementation
// ///////////////////////////////////////////////

unsafe impl<K: Send, V: Send> Send for SkipMap<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for SkipMap<K, V> {}

impl<K: Ord, V> default::Default for SkipMap<K, V> {
    fn default() -> SkipMap<K, V> {
        SkipMap::new()
    }
}

/// This implementation of PartialEq only checks that the *values* are equal; it
/// does not check for equivalence of other features (such as the ordering
/// function and the node levels). Furthermore, this uses `T`'s implementation
/// of PartialEq and *does not* use the owning skipmap's comparison function.
impl<AK, AV, BK, BV> cmp::PartialEq<SkipMap<BK, BV>> for SkipMap<AK, AV>
where
    AK: cmp::PartialEq<BK>,
    AV: cmp::PartialEq<BV>,
{
    #[inline]
    fn eq(&self, other: &SkipMap<BK, BV>) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(x, y)| x.0 == y.0 && x.1 == y.1)
    }
    #[allow(clippy::partialeq_ne_impl)]
    #[inline]
    fn ne(&self, other: &SkipMap<BK, BV>) -> bool {
        self.len() != other.len()
            || self
                .iter()
                .zip(other.iter())
                .any(|(x, y)| x.0 != y.0 || x.1 != y.1)
    }
}

impl<K, V> cmp::Eq for SkipMap<K, V>
where
    K: cmp::Eq,
    V: cmp::Eq,
{
}

impl<AK, AV, BK, BV> cmp::PartialOrd<SkipMap<BK, BV>> for SkipMap<AK, AV>
where
    AK: cmp::PartialOrd<BK>,
    AV: cmp::PartialOrd<BV>,
{
    #[inline]
    fn partial_cmp(&self, other: &SkipMap<BK, BV>) -> Option<Ordering> {
        match self
            .iter()
            .map(|x| x.0)
            .partial_cmp(other.iter().map(|x| x.0))
        {
            None => None,
            Some(Ordering::Less) => Some(Ordering::Less),
            Some(Ordering::Greater) => Some(Ordering::Greater),
            Some(Ordering::Equal) => self
                .iter()
                .map(|x| x.1)
                .partial_cmp(other.iter().map(|x| x.1)),
        }
    }
}

impl<K, V> Ord for SkipMap<K, V>
where
    K: cmp::Ord,
    V: cmp::Ord,
{
    #[inline]
    fn cmp(&self, other: &SkipMap<K, V>) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<K, V> Extend<(K, V)> for SkipMap<K, V>
where
    K: Ord,
{
    #[inline]
    fn extend<I: iter::IntoIterator<Item = (K, V)>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        for element in iterator {
            self.insert(element.0, element.1);
        }
    }
}

impl<K, V> ops::Index<usize> for SkipMap<K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &V {
        self.get_index(index)
            .and_then(|node| node.value_ref())
            .expect("Index out of bounds")
    }
}

impl<K, V> ops::IndexMut<usize> for SkipMap<K, V> {
    fn index_mut(&mut self, index: usize) -> &mut V {
        self.get_index_mut(index)
            .and_then(|node| node.value_mut())
            .expect("Index out of bounds")
    }
}

impl<K, V> fmt::Debug for SkipMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "({:?}, {:?})", k, v)?;
        }
        write!(f, "]")
    }
}

impl<K, V> fmt::Display for SkipMap<K, V>
where
    K: fmt::Display,
    V: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {})", k, v)?;
        }
        write!(f, "]")
    }
}

impl<K, V> iter::IntoIterator for SkipMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}
impl<'a, K, V> iter::IntoIterator for &'a SkipMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, K, V> iter::IntoIterator for &'a mut SkipMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V> iter::FromIterator<(K, V)> for SkipMap<K, V>
where
    K: Ord,
{
    #[inline]
    fn from_iter<I>(iter: I) -> SkipMap<K, V>
    where
        I: iter::IntoIterator<Item = (K, V)>,
    {
        let mut skipmap = SkipMap::new();
        skipmap.extend(iter);
        skipmap
    }
}

impl<K: Hash, V: Hash> Hash for SkipMap<K, V> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for elt in self {
            elt.hash(state);
        }
    }
}

// ///////////////////////////////////////////////
// Extra structs
// ///////////////////////////////////////////////
//

/// An iterator for [SkipMap]
pub struct Iter<'a, K: 'a, V: 'a>(skipnode::Iter<'a, (K, V)>);

impl<'a, K: 'a, V: 'a> Iter<'a, K, V> {
    /// SAFETY: There must be `len` nodes after head.
    unsafe fn from_head(head: &'a SkipNode<K, V>, len: usize) -> Self {
        unsafe { Self(skipnode::Iter::from_head(head, len)) }
    }
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.0, &x.1))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|x| (&x.0, &x.1))
    }
}

/// A mutable iterator for [SkipMap]
pub struct IterMut<'a, K: 'a, V: 'a>(skipnode::IterMut<'a, (K, V)>);

impl<'a, K: 'a, V: 'a> IterMut<'a, K, V> {
    /// SAFETY: There must be `len` nodes after head.
    unsafe fn from_head(head: &'a mut SkipNode<K, V>, len: usize) -> Self {
        unsafe { Self(skipnode::IterMut::from_head(head, len)) }
    }
}

impl<'a, K: 'a, V: 'a> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.0, &mut x.1))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K: 'a, V: 'a> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|x| (&x.0, &mut x.1))
    }
}

/// Iterator over a [`SkipMap`]'s keys.
pub struct Keys<'a, K: 'a, V>(Iter<'a, K, V>);

impl<'a, K: 'a, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K: 'a, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|x| x.0)
    }
}

/// Iterator over a [`SkipMap`]'s values.
pub struct Values<'a, K, V: 'a>(Iter<'a, K, V>);

impl<'a, K, V: 'a> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K, V: 'a> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|x| x.1)
    }
}

// ////////////////////////////////////////////////////////////////////////////
// Tests
// ////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use std::ops::Bound::{self, Excluded, Included, Unbounded};

    use super::SkipMap;

    #[test]
    fn basic_small() {
        let mut sm: SkipMap<i64, i64> = SkipMap::new();
        sm.check();
        assert!(sm.remove(&1).is_none());
        sm.check();
        assert!(sm.insert(1, 0).is_none());
        sm.check();
        assert_eq!(sm.insert(1, 5), Some(0));
        sm.check();
        assert_eq!(sm.remove(&1), Some(5));
        sm.check();
        assert!(sm.insert(1, 10).is_none());
        sm.check();
        assert!(sm.insert(2, 20).is_none());
        sm.check();
        assert_eq!(sm.remove(&1), Some(10));
        sm.check();
        assert_eq!(sm.remove(&2), Some(20));
        sm.check();
        assert!(sm.remove(&1).is_none());
        sm.check();
    }

    #[test]
    fn basic_large() {
        let size = 10_000;
        let mut sm = SkipMap::with_capacity(size);
        assert!(sm.is_empty());

        for i in 0..size {
            sm.insert(i, i * 10);
            assert_eq!(sm.len(), i + 1);
        }
        sm.check();

        for i in 0..size {
            assert_eq!(sm.remove(&i), Some(i * 10));
            assert_eq!(sm.len(), size - i - 1);
        }
        sm.check();
    }

    #[test]
    fn insert_existing() {
        let size = 100;
        let mut sm = SkipMap::new();

        for i in 0..size {
            assert!(sm.insert(i, format!("{}", i)).is_none());
        }

        for i in 0..size {
            assert_eq!(sm.insert(i, format!("{}", i)), Some(format!("{}", i)));
        }
        for i in (0..size).rev() {
            assert_eq!(sm.insert(i, format!("{}", i)), Some(format!("{}", i)));
        }
    }

    #[test]
    fn clear() {
        let mut sm: SkipMap<_, _> = (0..100).map(|x| (x, x)).collect();
        assert_eq!(sm.len(), 100);
        sm.clear();
        sm.check();
        assert!(sm.is_empty());
    }

    #[test]
    fn iter() {
        let size = 10000;

        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sm.iter().map(|(&a, &b)| (a, b)));
        test(size, sm.iter_mut().map(|(&a, &mut b)| (a, b)));
        test(size, sm.into_iter());
    }

    #[test]
    fn iter_rev() {
        let size = 1000;

        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sm.iter().rev().map(|(&a, &b)| (a, b)));
        test(size, sm.iter_mut().rev().map(|(&a, &mut b)| (a, b)));
        test(size, sm.into_iter().rev());
    }

    #[test]
    fn iter_mixed() {
        let size = 1000;
        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)> + DoubleEndedIterator,
        {
            for i in 0..size / 4 {
                assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
                assert_eq!(iter.next().unwrap(), (i, i));
                assert_eq!(iter.next_back().unwrap(), (size - i - 1, size - i - 1));
            }
            for i in size / 4..size * 3 / 4 {
                assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }
        test(size, sm.iter().map(|(&a, &b)| (a, b)));
        test(size, sm.iter_mut().map(|(&a, &mut b)| (a, b)));
        test(size, sm.into_iter());
    }

    #[test]
    fn iter_key_val() {
        let size = 1000;
        let sm: SkipMap<_, _> = (0..size).map(|x| (x, 2 * x)).collect();

        let mut keys = sm.keys();
        for i in 0..size / 2 {
            assert_eq!(keys.next(), Some(&i));
        }
        for i in 0..size / 2 {
            assert_eq!(keys.next_back(), Some(&(size - i - 1)))
        }
        assert!(keys.next().is_none());

        let mut vals = sm.values();
        for i in 0..size / 2 {
            assert_eq!(vals.next(), Some(&(2 * i)));
        }
        for i in 0..size / 2 {
            assert_eq!(vals.next_back(), Some(&(2 * (size - i) - 2)))
        }
        assert!(vals.next().is_none());
    }

    #[test]
    fn range_small() {
        let size = 5;

        let sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        let mut j = 0;
        for ((&k, &v), i) in sm.range(Included(&2), Unbounded).zip(2..size) {
            assert_eq!(k, i);
            assert_eq!(v, i);
            j += 1;
        }
        assert_eq!(j, size - 2);
    }

    #[test]
    fn range_1000() {
        let size = 1000;
        let sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        fn test(sm: &SkipMap<usize, usize>, min: Bound<&usize>, max: Bound<&usize>) {
            let mut values = sm.range(min, max);
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

            for ((&k, &v), e) in values.by_ref().zip(expects.by_ref()) {
                assert_eq!(k, e);
                assert_eq!(v, e);
            }
            assert!(values.next().is_none());
            assert!(expects.next().is_none());
        }

        test(&sm, Excluded(&200), Excluded(&800));
        test(&sm, Included(&200), Excluded(&800));
        test(&sm, Unbounded, Excluded(&800));
        test(&sm, Excluded(&200), Included(&800));
        test(&sm, Included(&200), Included(&800));
        test(&sm, Unbounded, Included(&800));
        test(&sm, Excluded(&200), Unbounded);
        test(&sm, Included(&200), Unbounded);
        test(&sm, Unbounded, Unbounded);
    }

    #[test]
    fn range() {
        let size = 200;
        let sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();

        for i in 0..size {
            for j in 0..size {
                let mut values = sm.range(Included(&i), Included(&j)).map(|(&a, &b)| (a, b));
                let mut expects = i..=j;

                assert_eq!(values.size_hint(), expects.size_hint());

                for ((k, v), e) in values.by_ref().zip(expects.by_ref()) {
                    assert_eq!(k, e);
                    assert_eq!(v, e);
                }
                assert!(values.next().is_none());
                assert!(expects.next().is_none());
            }
        }

        // let mut values = sm.range(Included(&10), Included(&5)).map(|(&a, &b)| (a, b));
        // assert!(values.next().is_none());
    }

    #[test]
    fn bound() {
        let sm: SkipMap<_, _> = (1..3).map(|x| (x, x)).collect();

        assert_eq!(sm.lower_bound(Unbounded), Some((&1, &1)));
        assert_eq!(sm.lower_bound(Excluded(&1)), Some((&2, &2)));
        assert_eq!(sm.lower_bound(Included(&1)), Some((&1, &1)));

        assert_eq!(sm.lower_bound(Excluded(&3)), None);
        assert_eq!(sm.lower_bound(Included(&3)), None);

        assert_eq!(sm.upper_bound(Unbounded), Some((&2, &2)));
        assert_eq!(sm.upper_bound(Excluded(&2)), Some((&1, &1)));
        assert_eq!(sm.upper_bound(Included(&2)), Some((&2, &2)));

        assert_eq!(sm.upper_bound(Excluded(&0)), None);
        assert_eq!(sm.upper_bound(Included(&0)), None);
    }

    #[test]
    fn index_pop() {
        let size = 1000;
        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, 2 * x)).collect();
        assert_eq!(sm.front(), Some((&0, &0)));
        assert_eq!(sm.front_mut(), Some((&0, &mut 0)));
        assert_eq!(sm.back(), Some((&(size - 1), &(2 * size - 2))));
        assert_eq!(sm.back_mut(), Some((&(size - 1), &mut (2 * size - 2))));
        for i in 0..size {
            assert_eq!(sm[i], 2 * i);
            assert_eq!(sm.get(&i), Some(&(2 * i)));
            assert_eq!(sm.get_mut(&i), Some(&mut (2 * i)));
        }

        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, 2 * x)).collect();
        for i in 0..size {
            assert_eq!(sm.pop_front(), Some((i, 2 * i)));
            assert_eq!(sm.len(), size - i - 1);
        }
        assert!(sm.pop_front().is_none());
        assert!(sm.front().is_none());
        assert!(sm.is_empty());

        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, 2 * x)).collect();
        for i in 0..size {
            assert_eq!(sm.pop_back(), Some((size - i - 1, 2 * (size - i - 1))));
            assert_eq!(sm.len(), size - i - 1);
        }
        assert!(sm.pop_back().is_none());
        assert!(sm.back().is_none());
        assert!(sm.is_empty());
    }

    #[test]
    fn remove_index() {
        let size = 100;

        for i in 0..size {
            let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();
            assert_eq!(sm.remove_index(i), (i, i));
            assert_eq!(sm.len(), size - 1);
        }

        let mut sm: SkipMap<_, _> = (0..size).map(|x| (x, x)).collect();
        for i in 0..size {
            assert_eq!(sm.remove_index(0), (i, i));
            assert_eq!(sm.len(), size - i - 1);
            sm.check();
        }
        assert!(sm.is_empty());
    }

    #[test]
    fn contains() {
        let (min, max) = (25, 75);
        let sm: SkipMap<_, _> = (min..max).map(|x| (x, x)).collect();

        for i in 0..100 {
            println!("i = {} (contained: {})", i, sm.contains_key(&i));
            if i < min || i >= max {
                assert!(!sm.contains_key(&i));
            } else {
                assert!(sm.contains_key(&i));
            }
        }
    }

    #[test]
    fn debug_display() {
        let sl: SkipMap<_, _> = (0..10).map(|x| (x, x)).collect();
        sl.debug_structure();
        println!("{:?}", sl);
        println!("{}", sl);
    }

    #[test]
    #[allow(clippy::eq_op, clippy::many_single_char_names)]
    fn equality() {
        let a: SkipMap<i64, i64> = (0..100).map(|x| (x, x)).collect();
        let b: SkipMap<i64, i64> = (0..100).map(|x| (x, x)).collect();
        let c: SkipMap<i64, i64> = (0..10).map(|x| (x, x)).collect();
        let d: SkipMap<i64, i64> = (100..200).map(|x| (x, x)).collect();
        let e: SkipMap<i64, i64> = (0..100).chain(0..1).map(|x| (x, x)).collect();

        assert_eq!(a, a);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a, e);
        assert_eq!(b, b);
        assert_ne!(b, c);
        assert_ne!(b, d);
        assert_eq!(b, e);
        assert_eq!(c, c);
        assert_ne!(c, d);
        assert_ne!(c, e);
        assert_eq!(d, d);
        assert_ne!(d, e);
        assert_eq!(e, e);
    }
}
