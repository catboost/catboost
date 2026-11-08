#![allow(warnings)]
#![allow(clippy)]
#![allow(unknown_lints)]

use std::{
    cmp::Ordering,
    fmt, iter,
    ptr::{self, NonNull},
};

// ////////////////////////////////////////////////////////////////////////////
// SkipNode
// ////////////////////////////////////////////////////////////////////////////

/// A covariant pointer to a SkipNode.
///
/// SkipNode<V> should contain mutable pointers to other nodes,
/// but mutable pointers are not covariant in Rust.
/// The appropriate pointer type is std::ptr::NonNull.
///
/// See [`std::ptr::NonNull`] and Rustonomicon for details on covariance.
/// https://doc.rust-lang.org/nomicon/subtyping.html
type Link<T> = Option<NonNull<SkipNode<T>>>;

/// SkipNodes are make up the SkipList.  The SkipList owns the first head-node
/// (which has no value) and each node has ownership of the next node through
/// `next`.
///
/// The node has a `level` which corresponds to how 'high' the node reaches.
///
/// A node of `level` n has (n + 1) links to next nodes, which are stored in
/// a vector.
///
/// The node linked by level 0 should be considered owned by this node.
///
/// There is a corresponding vector of link lengths which contains the distance
/// between current node and the next node. If there's no next node, the distance
/// is distance between current node and last reachable node.
///
/// Lastly, each node contains a link to the immediately previous node in case
/// one needs to parse the list backwards.
#[derive(Clone, Debug)]
pub struct SkipNode<V> {
    // item should never be None, unless the node is a head.
    pub item: Option<V>,
    // how high the node reaches.
    pub level: usize,
    // The immediately previous element.
    pub prev: Link<V>,
    // Vector of links to the next node at the respective level.  This vector
    // *must* be of length `self.level + 1`.  links[0] stores a pointer to the
    // next node, which will have to be dropped.
    pub links: Vec<Link<V>>,
    // The corresponding length of each link
    pub links_len: Vec<usize>,
}

// ///////////////////////////////////////////////
// Inherent methods
// ///////////////////////////////////////////////

impl<V> SkipNode<V> {
    /// Create a new head node.
    pub fn head(total_levels: usize) -> Self {
        SkipNode {
            item: None,
            level: total_levels - 1,
            prev: None,
            links: iter::repeat(None).take(total_levels).collect(),
            links_len: iter::repeat(0).take(total_levels).collect(),
        }
    }

    /// Create a new SkipNode with the given item..
    /// All pointers default to null.
    pub fn new(item: V, level: usize) -> Self {
        SkipNode {
            item: Some(item),
            level,
            prev: None,
            links: iter::repeat(None).take(level + 1).collect(),
            links_len: iter::repeat(0).take(level + 1).collect(),
        }
    }

    /// Consumes the node returning the item it contains.
    pub fn into_inner(mut self) -> Option<V> {
        self.item.take()
    }

    /// Returns `true` is the node is a head-node.
    pub fn is_head(&self) -> bool {
        self.prev.is_none()
    }

    pub fn next_ref(&self) -> Option<&Self> {
        // SAFETY: all links either points to something or is null.
        unsafe { self.links[0].as_ref().map(|p| p.as_ref()) }
    }

    pub fn next_mut(&mut self) -> Option<&mut Self> {
        // SAFETY: all links either points to something or is null.
        unsafe { self.links[0].as_mut().map(|p| p.as_mut()) }
    }

    /// Takes the next node and set next_node.prev as null.
    ///
    /// SAFETY: please make sure no link at level 1 or greater becomes dangling.
    pub unsafe fn take_tail(&mut self) -> Option<Box<Self>> {
        unsafe {
            self.links[0].take().map(|p| {
                let mut next = Box::from_raw(p.as_ptr());
                next.prev = None;
                self.links_len[0] = 0;
                next
            })
        }
    }

    /// Replace the next node.
    /// Return the old node.
    ///
    /// SAFETY: please makes sure all links are fixed.
    pub unsafe fn replace_tail(&mut self, mut new_next: Box<Self>) -> Option<Box<Self>> {
        unsafe {
            let mut old_next = self.take_tail();
            if let Some(old_next) = old_next.as_mut() {
                old_next.prev = None;
            }
            new_next.prev = Some(NonNull::new_unchecked(self as *mut _));
            self.links[0] = Some(NonNull::new_unchecked(Box::into_raw(new_next)));
            self.links_len[0] = 1;
            old_next
        }
    }
    // /////////////////////////////
    // Value Manipulation
    // /////////////////////////////
    //
    // Methods that care about items carried by the nodes.

    /// Retain all nodes who satisfies `pred`. Return the number removed nodes..
    ///
    /// Requires `self` being the head of the skiplist.
    ///
    /// `pred` is a function that takes two parameters, `Option<&V>` and  `&V`.
    /// `Option<&V>` is the value of current node (`None` if the current node is the head),
    /// `&V` is the value of the next node.
    /// If the `pred` returns `false`, then the next node is dropped.
    #[must_use]
    pub fn retain<F>(&mut self, mut pred: F) -> usize
    where
        F: FnMut(Option<&V>, &V) -> bool,
    {
        assert!(self.is_head());
        let mut removed_count = 0;
        // Aliasing mutable references is undefined behavior.
        // However if you create a pointer from a mutable reference,
        // it essentially borrows from it, we are free to alias it until
        // the next time we use that reference.
        let mut current_node = self as *mut Self;
        // `level_heads` records every head of the linked list.
        // A head is the last node of a given level that is not after current_node.
        let mut level_heads: Vec<_> = iter::repeat(current_node).take(self.level + 1).collect();
        // SAFETY: a huge block of pointer manipulation.
        unsafe {
            while let Some(mut next_node) = (*current_node).take_tail() {
                // next_node is removed from the list, so we can refer it by value.
                if pred(
                    (*current_node).item.as_ref(),
                    next_node.item.as_ref().unwrap(),
                ) {
                    // Keeping next_node.
                    // First we should update level_heads, then we put next_node back to the list.
                    for x in &mut level_heads[0..=next_node.level] {
                        *x = next_node.as_mut() as *mut _;
                    }
                    (*current_node).replace_tail(next_node);
                    current_node = (*current_node).next_mut().unwrap();
                } else {
                    // Remove next_node.
                    removed_count += 1;
                    // Fixes links above level 0.
                    for (level, head) in level_heads
                        .iter_mut()
                        .map(|&mut node_p| &mut *node_p)
                        .enumerate()
                        .skip(1)
                    {
                        if level <= next_node.level {
                            assert!(ptr::eq(
                                head.links[level].unwrap().as_ptr(),
                                next_node.as_mut()
                            ));
                            head.links_len[level] += next_node.links_len[level];
                            head.links_len[level] -= 1;
                            head.links[level] = next_node.links[level];
                        } else {
                            head.links_len[level] -= 1;
                        }
                    }
                    // Fix the link at level 0.
                    if let Some(new_next) = next_node.take_tail() {
                        (*current_node).replace_tail(new_next);
                    }
                }
            }
        }
        removed_count
    }

    // /////////////////////////////
    // Pointer Manipulations
    // /////////////////////////////
    //
    // Methods that care about the whole node.
    //

    /// Distance between current node and the given node at specified level.
    /// If no node is given, then return distance between current node and the
    /// last possible node.
    /// If the node is not reachable on given level, return Err(()).
    pub fn distance_at_level(&self, level: usize, target: Option<&Self>) -> Result<usize, ()> {
        let distance = match target {
            Some(target) => {
                let (dest, distance) =
                    self.advance_while_at_level(level, |current, _| !ptr::eq(current, target));
                if !ptr::eq(dest, target) {
                    return Err(());
                }
                distance
            }
            None => {
                let (dest, distance) = self.advance_while_at_level(level, |_, _| true);
                dest.links_len[level] + distance
            }
        };
        Ok(distance)
    }

    /// Move for max_distance units.
    /// Returns None if it's not possible.
    pub fn advance(&self, max_distance: usize) -> Option<&Self> {
        let level = self.level;
        let mut node = self;
        let mut distance_left = max_distance;
        for level in (0..=level).rev() {
            let (new_node, steps) = node.advance_at_level(level, distance_left);
            distance_left -= steps;
            node = new_node;
        }
        if distance_left == 0 { Some(node) } else { None }
    }

    /// Move for max_distance units.
    /// Returns None if it's not possible.
    pub fn advance_mut(&mut self, max_distance: usize) -> Option<&mut Self> {
        let level = self.level;
        let mut node = self;
        let mut distance_left = max_distance;
        for level in (0..=level).rev() {
            let (new_node, steps) = node.advance_at_level_mut(level, distance_left);
            distance_left -= steps;
            node = new_node;
        }
        if distance_left == 0 { Some(node) } else { None }
    }

    /// Move to the last node reachable from this node.
    pub fn last(&self) -> &Self {
        (0..=self.level).rev().fold(self, |node, level| {
            node.advance_while_at_level(level, |_, _| true).0
        })
    }

    /// Move to the last node reachable from this node.
    pub fn last_mut(&mut self) -> &mut Self {
        (0..=self.level).rev().fold(self, |node, level| {
            node.advance_while_at_level_mut(level, |_, _| true).0
        })
    }

    /// Try to move for the given distance, only using links at the specified level.
    /// If it's impossible, then move as far as possible.
    ///
    /// Returns a reference to the new node and the distance travelled.
    pub fn advance_at_level(&self, level: usize, mut max_distance: usize) -> (&Self, usize) {
        self.advance_while_at_level(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                true
            } else {
                false
            }
        })
    }

    /// Try to move for the given distance, only using links at the specified level.
    /// If it's impossible, then move as far as possible.
    ///
    /// Returns a mutable reference to the new node and the distance travelled.
    pub fn advance_at_level_mut(
        &mut self,
        level: usize,
        mut max_distance: usize,
    ) -> (&mut Self, usize) {
        self.advance_while_at_level_mut(level, move |current_node, _| {
            let travelled = current_node.links_len[level];
            if travelled <= max_distance {
                max_distance -= travelled;
                true
            } else {
                false
            }
        })
    }

    /// Keep moving at the specified level as long as pred is true.
    /// pred takes reference to current node and next node.
    pub fn advance_while_at_level(
        &self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.next_if_at_level(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    /// Keep moving at the specified level as long as pred is true.
    /// pred takes reference to current node and next node.
    pub fn advance_while_at_level_mut(
        &mut self,
        level: usize,
        mut pred: impl FnMut(&Self, &Self) -> bool,
    ) -> (&mut Self, usize) {
        let mut current = self;
        let mut travelled = 0;
        loop {
            match current.next_if_at_level_mut(level, &mut pred) {
                Ok((node, steps)) => {
                    current = node;
                    travelled += steps;
                }
                Err(node) => return (node, travelled),
            }
        }
    }

    // The following methods return `Err(self)` if they fail.
    //
    // In Rust, the lifetime of returned value is the same as `self`.
    // Therefore if you return something that's borrowed from `self` in a branch,
    // `self` is considered borrowed in other branches.
    //
    // e.g.
    // ```
    // fn some_method(&mut self) -> Option<&mut Self>;
    //
    // fn caller(&mut self) {
    //     match self.some_method(){
    //         Some(x) => return x, // oops now `self` is borrowed until the function returns...
    //         None => return self, // Now you cannot use `self` in other branches..
    //     }                        // including returning it!
    // }
    // ```
    // While in this example you can restructure the code to fix that,
    // it's much more difficult when loops are involved.
    // The following methods are usually used in loops, so they return `Err(self)`
    // when they fail, to ease the pain.

    /// Move to the next node at given level if the given predicate is true.
    /// The predicate takes reference to the current node and the next node.
    pub fn next_if_at_level_mut(
        &mut self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&mut Self, usize), &mut Self> {
        // SAFETY: If a link contains Some(p), then p always points to something.
        let next = unsafe { self.links[level].and_then(|p| p.as_ptr().as_mut()) };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Move to the next node at given level if the given predicate is true.
    /// The predicate takes reference to the current node and the next node.
    pub fn next_if_at_level(
        &self,
        level: usize,
        predicate: impl FnOnce(&Self, &Self) -> bool,
    ) -> Result<(&Self, usize), &Self> {
        // SAFETY: If a link contains Some(p), then p always points to something.
        let next = unsafe { self.links[level].as_ref().map(|p| p.as_ref()) };
        match next {
            Some(next) if predicate(self, next) => Ok((next, self.links_len[level])),
            _ => Err(self),
        }
    }

    /// Insert a node after given distance after the list head.
    ///
    /// Requires that there's nothing before the node and the new node can't be at a higher level.
    ///
    /// Return the reference to the new node if successful.
    /// Give back the input node if not successful.
    pub fn insert_at(
        &mut self,
        new_node: Box<Self>,
        distance_to_parent: usize,
    ) -> Result<&mut Self, Box<Self>> {
        assert!(self.prev.is_none(), "Only the head may insert nodes!");
        assert!(
            self.level >= new_node.level,
            "You may not insert nodes with level higher than the head!"
        );
        let inserter = IndexInserter::new(distance_to_parent, new_node);
        inserter.act(self)
    }

    /// Move for distance units, and remove the node after it.
    ///
    /// Requires that there's nothing before the node and the new node can't be at a higher level.
    ///
    /// If that node exists, remove that node and return it.
    pub fn remove_at(&mut self, distance_to_parent: usize) -> Option<Box<Self>> {
        assert!(self.prev.is_none(), "Only the head may remove nodes!");
        let remover = IndexRemover::new(distance_to_parent);
        remover.act(self).ok()
    }

    /// Check the integrity of the list.
    ///
    pub fn check(&self) {
        assert!(self.is_head());
        assert!(self.item.is_none());
        let mut current_node = Some(self);
        let mut len = 0;
        while let Some(node) = current_node {
            // Check the integrity of node.
            assert_eq!(node.level + 1, node.links.len());
            assert_eq!(node.level + 1, node.links_len.len());
            if !node.is_head() {
                assert!(node.item.is_some());
            }
            // Check link at level 0
            if let Some(next_node) = node.next_ref() {
                len += 1;
                assert!(ptr::eq(next_node.prev.unwrap().as_ptr(), node));
            }
            current_node = node.next_ref();
        }

        let len = len; // no mutation

        for lvl in 1..=self.level {
            let mut length_sum = 0;
            let mut current_node = Some(self);
            while let Some(node) = current_node {
                length_sum += node.links_len[lvl];
                // SAFETY: all links are either None or should points to something.
                let next_node = unsafe { node.links[lvl].as_ref().map(|p| p.as_ref()) };
                assert_eq!(
                    node.links_len[lvl],
                    node.distance_at_level(lvl - 1, next_node).unwrap(),
                    "Node gives different distance at level {} and level {}!",
                    lvl,
                    lvl - 1
                );

                current_node = next_node;
            }

            assert_eq!(length_sum, len);
        }
    }
}

impl<V> Drop for SkipNode<V> {
    fn drop(&mut self) {
        // SAFETY: all nodes are going to be dropped; its okay that its links (except those at
        // level 0) become dangling.
        unsafe {
            let mut node = self.take_tail();
            while let Some(mut node_inner) = node {
                node = node_inner.take_tail();
            }
        }
    }
}

// ///////////////////////////////////////////////
// Trait implementation
// ///////////////////////////////////////////////

impl<V> fmt::Display for SkipNode<V>
where
    V: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.item {
            Some(ref v) => {
                write!(f, "{}", v)
            }
            _ => Ok(()),
        }
    }
}

// ///////////////////////////////////////////////
// Actions
// ///////////////////////////////////////////////

/// A SeekListAction seeks a node on the list and do something, e.g. insertion, deletion,
/// replacement, on it, often mutating the links in the process.
///
/// Actions on skiplists usually consist of three phases: for each level, seek the node on which we modify the links,
/// actually modify the link on the 0th level, and fixup the links at upper levels.
///
/// Between the phases there are some bookkeeping, and this trait abstract that away.
///
/// To use this trait, just implement the 3 phases (`seek`, `act_on_node`, `fixup`),
/// and call `act()` on the given list.
///
/// For examples, see one of the types that implements this trait, such as [IndexInserter] or [IndexRemover].
pub trait SkipListAction<'a, T>: Sized {
    /// Return type when this action succeeds.
    type Ok;
    /// Return type when this action fails.
    type Err;

    /// This is called when the list action has failed.
    ///
    /// This is only called when seek() finds nothing.
    fn fail(self) -> Self::Err;

    /// Find the target node at the given level.
    ///
    /// Return some node and distance travelled.
    /// Return `None` when target node does not exist anywhere in the list.
    /// In this case, the action is considered to have failed, and `Self::fail()` will be called.
    ///
    /// Target node may not exist at a higher level.
    /// You should return some node before the target node in this case.
    /// At level 0 it always finds the target or return `None`.
    fn seek(
        &mut self,
        node: &'a mut SkipNode<T>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<T>, usize)>;

    /// Do something on the node.
    ///
    /// If this action fails, then `Self::fail()` will not be called.
    /// This makes error handling more flexible.
    /// # SAFETY
    /// If `Self::Ok` is a reference, it shall not alias with any node that needs fixup.
    unsafe fn act_on_node(self, node: &'a mut SkipNode<T>) -> Result<Self::Ok, Self::Err>;

    /// Usually SkipListAction breaks links between nodes,  this method should fix that up.
    ///
    /// It should never fail.
    ///
    /// `level_head` is the node whose links may needs to be fixed.
    /// `action_result` is a mutable reference to the return value of act_on_node().
    /// `distance_to_target` is distance from `level_head` to the node that has been acted on.
    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<T>,
        distance_to_target: usize,
        action_result: &mut Self::Ok,
    );

    /// List traversal logic.
    ///
    /// This handles bookkeeping required for all skiplist mutations.
    ///
    /// It's unlikely one will need to override this.
    /// Override act() instead.
    unsafe fn _traverse(
        mut self,
        node: &'a mut SkipNode<T>,
        level: usize,
    ) -> Result<(Self::Ok, usize), Self::Err> {
        unsafe {
            let (level_head, distance_this_level) = match self.seek(node, level) {
                Some(res) => res,
                None => return Err(self.fail()),
            };
            let level_head_p = level_head as *mut SkipNode<T>;
            if level == 0 {
                let mut res = self.act_on_node(level_head)?;
                Self::fixup(0, &mut *level_head_p, 0, &mut res);
                Ok((res, distance_this_level))
            } else {
                let (mut res, distance_after_head) = self._traverse(level_head, level - 1)?;
                let level_head = &mut *level_head_p;
                Self::fixup(level, level_head, distance_after_head, &mut res);
                Ok((res, distance_this_level + distance_after_head))
            }
        }
    }

    /// Perform the action.
    fn act(self, list_head: &'a mut SkipNode<T>) -> Result<Self::Ok, Self::Err> {
        let (res, _distance) = unsafe { self._traverse(list_head, list_head.level)? };
        Ok(res)
    }
}

// helpers for ListActions.
impl<T> SkipNode<T> {
    /// Insert the new node immediately after this node.
    ///
    /// SAFETY: This doesn't fix links at level 1 or higher.
    pub unsafe fn insert_next(&mut self, mut new_node: Box<SkipNode<T>>) -> &mut SkipNode<T> {
        unsafe {
            if let Some(tail) = self.take_tail() {
                new_node.replace_tail(tail);
            }
            self.replace_tail(new_node);
            self.next_mut().unwrap()
        }
    }

    /// Take the node immediately after this node.
    ///
    /// SAFETY: This doesn't fix links at level 1 or higher.
    pub unsafe fn take_next(&mut self) -> Option<Box<SkipNode<T>>> {
        unsafe {
            let mut ret = self.take_tail()?;
            if let Some(new_tail) = ret.take_tail() {
                self.replace_tail(new_tail);
            }
            Some(ret)
        }
    }
}

/// Helper to seek the node at specific index.
/// self.0 is the distance between current node and target.
struct DistanceSeeker(usize);

impl DistanceSeeker {
    /// Find the last node reachable from the given level
    /// whose distance from `node` is no greater than `self.0`.
    ///
    /// Return target node and distance travelled if succeeds.
    fn seek<'a, V>(
        &mut self,
        node: &'a mut SkipNode<V>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<V>, usize)> {
        let (node, distance) = node.advance_at_level_mut(level, self.0);
        if level == 0 && distance != self.0 {
            None
        } else {
            self.0 -= distance;
            Some((node, distance))
        }
    }
}

/// Insert a new node at the given index.
///
/// See [SkipNode::insert_at] for examples on how to use.
struct IndexInserter<V> {
    index_seek: DistanceSeeker,
    new_node: Box<SkipNode<V>>,
}

impl<V> IndexInserter<V> {
    fn new(distance: usize, new_node: Box<SkipNode<V>>) -> Self {
        IndexInserter {
            index_seek: DistanceSeeker(distance),
            new_node,
        }
    }
}

impl<'a, V: 'a> SkipListAction<'a, V> for IndexInserter<V> {
    type Ok = &'a mut SkipNode<V>;

    type Err = Box<SkipNode<V>>;

    /// Return the would-be-inserted node if fails.
    fn fail(self) -> Self::Err {
        self.new_node
    }

    /// Finds the parent of the new node.
    fn seek(
        &mut self,
        node: &'a mut SkipNode<V>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<V>, usize)> {
        self.index_seek.seek(node, level)
    }

    /// SAFETY: This returns a new node, which should never alias with any old nodes.
    unsafe fn act_on_node(self, node: &'a mut SkipNode<V>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            // SAFETY: Links will be fixed by the caller.
            Ok(node.insert_next(self.new_node))
        }
    }

    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<V>,
        distance_to_parent: usize,
        new_node: &mut Self::Ok,
    ) {
        insertion_fixup(level, level_head, distance_to_parent, new_node)
    }
}

/// Remove the node at the given index.
///
/// See [SkipNode::remove_at] for examples on how to use.
struct IndexRemover {
    seeker: DistanceSeeker,
}

impl IndexRemover {
    fn new(distance: usize) -> Self {
        IndexRemover {
            seeker: DistanceSeeker(distance),
        }
    }
}

impl<'a, V> SkipListAction<'a, V> for IndexRemover {
    type Ok = Box<SkipNode<V>>;

    type Err = ();

    /// The only way to fail is when `seek()` does not find an appropriate node,
    /// so we just do nothing.
    #[allow(clippy::unused_unit)]
    fn fail(self) -> Self::Err {
        ()
    }

    fn seek(
        &mut self,
        node: &'a mut SkipNode<V>,
        level: usize,
    ) -> Option<(&'a mut SkipNode<V>, usize)> {
        self.seeker.seek(node, level)
    }

    // SAFETY: Self::Ok is not a reference type
    unsafe fn act_on_node(self, node: &'a mut SkipNode<V>) -> Result<Self::Ok, Self::Err> {
        unsafe {
            // SAFETY: links will be fixed by the caller.
            node.take_next().ok_or(())
        }
    }

    fn fixup(
        level: usize,
        level_head: &'a mut SkipNode<V>,
        _distance_to_parent: usize,
        removed_node: &mut Self::Ok,
    ) {
        removal_fixup(level, level_head, removed_node)
    }
}

/// Fixes links at `level` after insertion.
///
/// Put the new_node after level_head if applicable, and adjust link_len.
/// `distance_to_parent` is the distance from `level_head` to the parent of `new_node`.
pub fn insertion_fixup<T>(
    level: usize,
    level_head: &mut SkipNode<T>,
    distance_to_parent: usize,
    new_node: &mut &mut SkipNode<T>,
) {
    if level == 0 {
        // Already handled by insertion.
        return;
    }
    if level <= new_node.level {
        new_node.links[level] = level_head.links[level];
        level_head.links[level] = NonNull::new(*new_node);
        let old_len = level_head.links_len[level];
        new_node.links_len[level] = old_len - distance_to_parent;
        level_head.links_len[level] = distance_to_parent + 1;
    } else {
        level_head.links_len[level] += 1;
    }
}

/// Fix links at the given level after removal.
pub fn removal_fixup<T>(
    level: usize,
    level_head: &mut SkipNode<T>,
    removed_node: &mut Box<SkipNode<T>>,
) {
    if level == 0 {
        return;
    }
    if level <= removed_node.level {
        assert!(ptr::eq(
            level_head.links[level].unwrap().as_ptr(),
            removed_node.as_ref()
        ));
        level_head.links[level] = removed_node.links[level];
        level_head.links_len[level] += removed_node.links_len[level];
    }
    level_head.links_len[level] -= 1;
}

// helpers for ordered types.
impl<V> SkipNode<V> {
    /// Find the last node such that f(node.item) returns true.
    /// Return a reference to the node and distance travelled.
    fn find_ordering_impl<F>(&self, f: F) -> (&Self, usize)
    where
        F: Fn(&V) -> bool,
    {
        (0..=self.level)
            .rev()
            .fold((self, 0), |(node, distance), level| {
                let (node, steps) = node.advance_while_at_level(level, |_, next_node| {
                    let value = next_node.item.as_ref().unwrap();
                    f(value)
                });
                (node, distance + steps)
            })
    }

    /// Find the last node such that f(node.item) returns true.
    /// Return a mutable reference to the node and distance travelled.
    fn find_ordering_mut_impl<F>(&mut self, f: F) -> (&mut Self, usize)
    where
        F: Fn(&V) -> bool,
    {
        (0..=self.level)
            .rev()
            .fold((self, 0), |(node, distance), level| {
                let (node, steps) = node.advance_while_at_level_mut(level, |_, next_node| {
                    let value = next_node.item.as_ref().unwrap();
                    f(value)
                });
                (node, distance + steps)
            })
    }

    /// Given a list head, a comparison function and a target,
    /// return a reference to the last node whose item compares less than the target,
    /// and the distance to that node.
    pub fn find_last_le_with<F, T: ?Sized>(&self, cmp: F, target: &T) -> (&Self, usize)
    where
        F: Fn(&V, &T) -> Ordering,
    {
        self.find_ordering_impl(|node_value| cmp(node_value, target) != Ordering::Greater)
    }

    /// Given a list head, a comparison function and a target,
    /// return a mutable reference to the last node whose item compares less than the target.
    /// and the distance to that node.
    pub fn find_last_le_with_mut<F, T: ?Sized>(&mut self, cmp: F, target: &T) -> (&mut Self, usize)
    where
        F: Fn(&V, &T) -> Ordering,
    {
        self.find_ordering_mut_impl(|node_value| cmp(node_value, target) != Ordering::Greater)
    }

    /// Given a list head, a comparison function and a target,
    /// return a reference to the last node whose item compares less than or equal to the target.
    /// and the distance to that node.
    pub fn find_last_lt_with<F, T: ?Sized>(&self, cmp: F, target: &T) -> (&Self, usize)
    where
        F: Fn(&V, &T) -> Ordering,
    {
        assert!(self.is_head());
        self.find_ordering_impl(|node_value| cmp(node_value, target) == Ordering::Less)
    }

    /// Given a list head, a comparison function and a target,
    /// return a mutable refeerence to the last node whose item compares less than or equal to the target.
    /// and the distance to that node.
    #[allow(dead_code)]
    pub fn find_last_lt_with_mut<F, T: ?Sized>(&mut self, cmp: F, target: &T) -> (&mut Self, usize)
    where
        F: Fn(&V, &T) -> Ordering,
    {
        assert!(self.is_head());
        self.find_ordering_mut_impl(|node_value| cmp(node_value, target) == Ordering::Less)
    }
}

// ///////////////////////////////////////////////
// Helper Traits
// ///////////////////////////////////////////////

// Converting Option<&T> to *_ T becomes more and more annoying...
trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}

trait AsPtrMut<T> {
    fn as_ptr_mut(&mut self) -> *mut T;
}

impl<T> AsPtr<T> for Option<&T> {
    fn as_ptr(&self) -> *const T {
        self.map_or(ptr::null(), |inner_ref| inner_ref)
    }
}

impl<T> AsPtr<T> for Option<&mut T> {
    fn as_ptr(&self) -> *const T {
        self.as_ref().map_or(ptr::null(), |inner: &&mut T| &**inner)
    }
}

impl<T> AsPtrMut<T> for Option<&mut T> {
    fn as_ptr_mut(&mut self) -> *mut T {
        self.as_mut()
            .map_or(ptr::null_mut(), |inner: &mut &mut T| *inner)
    }
}

// /////////////////////////////////
// Iterators
// /////////////////////////////////
// Since Iterators (currently) only pop from front and back,
// they can be shared by some data structures.
// There's no need for a dummy head (that contains no item) in the iterator.
// so the members are named first and last instead of head/end to avoid confusion.

/// An iterator for [SkipList](super::SkipList) and [OrderedSkipList](super::OrderedSkipList).
pub struct Iter<'a, T> {
    pub(crate) first: Option<&'a SkipNode<T>>,
    pub(crate) last: Option<&'a SkipNode<T>>,
    pub(crate) size: usize,
}
impl<'a, T> Iter<'a, T> {
    /// SAFETY: There must be `len` nodes after head.
    pub(crate) unsafe fn from_head(head: &'a SkipNode<T>, len: usize) -> Self {
        if len == 0 {
            Iter {
                first: None,
                last: None,
                size: 0,
            }
        } else {
            let first = head.next_ref();
            let last = first.as_ref().map(|n| n.last());
            Iter {
                first,
                last,
                size: len,
            }
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_node = self.first?;
        if ptr::eq(current_node, self.last.as_ptr()) {
            self.first = None;
            self.last = None;
        } else {
            self.first = current_node.next_ref();
        }
        self.size -= 1;
        current_node.item.as_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let last_node = self.last?;

        if ptr::eq(self.first.as_ptr(), last_node) {
            self.first = None;
            self.last = None;
        } else {
            // SAFETY: The iterator is not empty yet.
            unsafe {
                self.last = last_node.prev.as_ref().map(|p| p.as_ref());
            }
        }
        self.size -= 1;
        last_node.item.as_ref()
    }
}

/// A mutable iterator for [SkipList](super::SkipList) and [OrderedSkipList](super::OrderedSkipList).
pub struct IterMut<'a, T> {
    pub(crate) first: Option<&'a mut SkipNode<T>>,
    pub(crate) last: Option<NonNull<SkipNode<T>>>,
    pub(crate) size: usize,
}

impl<'a, T> IterMut<'a, T> {
    /// SAFETY: There must be `len` nodes after head.
    pub(crate) unsafe fn from_head(head: &'a mut SkipNode<T>, len: usize) -> Self {
        if len == 0 {
            IterMut {
                first: None,
                last: None,
                size: 0,
            }
        } else {
            let last = NonNull::new(head.last_mut());
            let first = head.next_mut();
            IterMut {
                first,
                last,
                size: len,
            }
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_node = self.first.take()?;
        if ptr::eq(current_node, self.last.unwrap().as_ptr()) {
            self.first = None;
            self.last = None;
        } else {
            // calling current_node.next_mut() borrows it, transforming the reference to a pointer
            // unborrows that.
            let p = current_node.next_mut().unwrap() as *mut SkipNode<T>;
            // SAFETY: p.as_mut() is safe because it points to a valid object.
            // There's no aliasing issue since nobody else holds a reference to current_node
            // until this function returns, and the returned reference does not points to a node.
            unsafe {
                self.first = p.as_mut();
            }
        }
        self.size -= 1;
        current_node.item.as_mut()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.last?;
        debug_assert!(self.last.is_some());
        // There can be at most one mutable reference to the first node.
        // We need to take it from self.first before doing anything,
        // including simple comparison.
        let first = self.first.take().unwrap();
        let popped = if ptr::eq(first, self.last.unwrap().as_ptr()) {
            self.last = None;
            first
        } else {
            // SAFETY: self.last isn't null and doesn't alias first
            let new_last = unsafe { self.last.unwrap().as_mut().prev };
            if ptr::eq(first, new_last.unwrap().as_ptr()) {
                self.last = new_last;
                let popped_p = first.next_mut().unwrap() as *mut SkipNode<T>;
                self.first.replace(first);
                unsafe { &mut (*popped_p) }
            } else {
                self.first.replace(first);
                let last = self.last;
                self.last = new_last;
                unsafe { last.unwrap().as_ptr().as_mut().unwrap() }
            }
        };
        self.size -= 1;
        popped.item.as_mut()
    }
}

/// Consuming iterator for [SkipList](super::SkipList), [OrderedSkipList](super::OrderedSkipList) and [SkipMap](super::SkipMap).
pub struct IntoIter<T> {
    pub(crate) first: Option<Box<SkipNode<T>>>,
    pub(crate) last: Option<NonNull<SkipNode<T>>>,
    pub(crate) size: usize,
}

impl<T> IntoIter<T> {
    /// SAFETY: There must be `len` nodes after head.
    pub(crate) unsafe fn from_head(head: &mut SkipNode<T>, len: usize) -> Self {
        unsafe {
            if len == 0 {
                IntoIter {
                    first: None,
                    last: None,
                    size: 0,
                }
            } else {
                let last = NonNull::new(head.last_mut());
                let first = head.take_tail();
                IntoIter {
                    first,
                    last,
                    size: len,
                }
            }
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut popped_node = self.first.take()?;
        self.size -= 1;
        // SAFETY: no need to fix links at upper levels inside iterators.
        self.first = unsafe { popped_node.take_tail() };
        if self.first.is_none() {
            self.last = None;
        }
        popped_node.into_inner()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        #[allow(clippy::question_mark)]
        if self.first.is_none() {
            return None;
        }
        assert!(
            self.last.is_some(),
            "The IntoIter should be empty but IntoIter.last somehow still contains something"
        );
        let popped_node = if ptr::eq(self.first.as_deref().as_ptr(), self.last.unwrap().as_ptr()) {
            self.last = None;
            self.first.take()?
        } else {
            // SAFETY: we checked that self.last points to somewhere and does not alias to self.first
            let new_last = unsafe { self.last.unwrap().as_mut().prev };
            if ptr::eq(self.first.as_deref().as_ptr(), new_last.unwrap().as_ptr()) {
                // SAFETY: take_tail() is always safe in IntoIter.
                let popped = unsafe {
                    self.first
                        .as_mut()
                        .and_then(|node| node.take_tail())
                        .unwrap()
                };
                self.last = new_last;
                popped
            } else {
                // SAFETY: we checked new_last points to somewhere and do not alias to self.first.
                let popped = unsafe { new_last.unwrap().as_mut().take_tail().unwrap() };
                self.last = new_last;
                popped
            }
        };

        self.size -= 1;
        popped_node.into_inner()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Minimum levels required for a list of size n.
    fn levels_required(n: usize) -> usize {
        if n == 0 {
            1
        } else {
            let num_bits = std::mem::size_of::<usize>() * 8;
            num_bits - n.leading_zeros() as usize
        }
    }

    /// Test test_covariance for SkipNode.
    /// Those functions should compile if our data structures is covariant.
    /// Read Rustonomicon for details.
    #[test]
    fn test_covariance() {
        #[allow(dead_code)]
        fn shorten_lifetime<'min, 'max: 'min>(v: SkipNode<&'max ()>) -> SkipNode<&'min ()> {
            v
        }

        #[allow(dead_code)]
        fn shorten_lifetime_into_iter<'min, 'max: 'min>(
            v: IntoIter<&'max ()>,
        ) -> IntoIter<&'min ()> {
            v
        }

        // IterMut is covariant on the value type.
        // This is consistent with Rust reference &'a T.
        #[allow(dead_code)]
        fn shorten_lifetime_iter<'min, 'max: 'min>(
            v: Iter<'max, &'max ()>,
        ) -> Iter<'min, &'min ()> {
            v
        }

        // IterMut is not covariant on the value type.
        // This is consistent with Rust mutable reference type &mut T.
        // TODO: write a test that can't compile
        #[allow(dead_code)]
        fn shorten_lifetime_iter_mut<'min, 'max: 'min>(v: Iter<'max, ()>) -> Iter<'min, ()> {
            v
        }
    }

    #[test]
    fn test_level_required() {
        assert_eq!(levels_required(0), 1);
        assert_eq!(levels_required(1), 1);
        assert_eq!(levels_required(2), 2);
        assert_eq!(levels_required(3), 2);
        assert_eq!(levels_required(1023), 10);
        assert_eq!(levels_required(1024), 11);
    }

    fn level_for_index(mut n: usize) -> usize {
        let mut cnt = 0;
        while n & 0x1 == 1 {
            cnt += 1;
            n /= 2;
        }
        cnt
    }

    #[test]
    fn test_level_index() {
        assert_eq!(level_for_index(0), 0);
        assert_eq!(level_for_index(1), 1);
        assert_eq!(level_for_index(2), 0);
        assert_eq!(level_for_index(3), 2);
        assert_eq!(level_for_index(4), 0);
        assert_eq!(level_for_index(5), 1);
        assert_eq!(level_for_index(6), 0);
        assert_eq!(level_for_index(7), 3);
        assert_eq!(level_for_index(8), 0);
        assert_eq!(level_for_index(9), 1);
        assert_eq!(level_for_index(10), 0);
        assert_eq!(level_for_index(11), 2);
    }

    /// Make a list of size n
    /// levels are evenly spread out
    fn new_list_for_test(n: usize) -> Box<SkipNode<usize>> {
        let max_level = levels_required(n);
        let mut head = Box::new(SkipNode::<usize>::head(max_level));
        assert_eq!(head.links.len(), max_level);
        let mut nodes: Vec<_> = (0..n)
            .map(|n| {
                let new_node = Box::new(SkipNode::new(n, level_for_index(n)));
                Box::into_raw(new_node)
            })
            .collect();
        unsafe {
            let node_max_level = nodes.iter().map(|&node| (*node).level).max();
            if let Some(node_max_level) = node_max_level {
                assert_eq!(node_max_level + 1, max_level);
            }
            for level in 0..max_level {
                let mut last_node = head.as_mut() as *mut SkipNode<usize>;
                let mut len_left = n;
                for &mut node_ptr in nodes
                    .iter_mut()
                    .filter(|&&mut node_ptr| level <= (*node_ptr).level)
                {
                    if level == 0 {
                        (*node_ptr).prev = NonNull::new(last_node);
                    }
                    (*last_node).links[level] = NonNull::new(node_ptr);
                    (*last_node).links_len[level] = 1 << level;
                    last_node = node_ptr;
                    len_left -= 1 << level;
                }
                (*last_node).links_len[level] = len_left;
            }
        }
        head
    }

    /////////////////////////////////////////////////////////
    // Those tests are supposed to be run using Miri to detect UB.
    // The size of those test are limited since Miri doesn't run very fast.
    /////////////////////////////////////////////////////////

    #[test]
    fn miri_test_insert() {
        let mut list = new_list_for_test(50);
        list.insert_at(Box::new(SkipNode::new(100, 0)), 25).unwrap();
        list.insert_at(Box::new(SkipNode::new(101, 1)), 25).unwrap();
        list.insert_at(Box::new(SkipNode::new(102, 2)), 25).unwrap();
        list.insert_at(Box::new(SkipNode::new(103, 3)), 25).unwrap();
        list.insert_at(Box::new(SkipNode::new(104, 4)), 25).unwrap();
    }

    #[test]
    fn miri_test_remove() {
        let mut list = new_list_for_test(50);
        for i in (0..50).rev() {
            list.remove_at(i).unwrap();
        }
    }

    #[test]
    fn miri_test_distance() {
        let list = new_list_for_test(50);
        for i in 0..=list.level {
            let _ = list.distance_at_level(i, None);
        }
    }

    #[test]
    fn miri_test_iter() {
        fn test_iter(size: usize) {
            let list = new_list_for_test(size);
            let first = list.next_ref();
            let last = Some(list.last());
            let mut iter = Iter { first, last, size };
            for _ in 0..(size + 1) / 2 {
                let _ = iter.next();
                let _ = iter.next_back();
            }
            assert!(iter.next().is_none());
        }
        test_iter(9);
        test_iter(10);
    }

    #[test]
    fn miri_test_iter_mut() {
        fn test_iter_mut(size: usize) {
            let mut list = new_list_for_test(size);
            let mut first = list.next_mut();
            let last = first.as_mut().unwrap().last_mut();
            let last = NonNull::new(last);
            let mut iter = IterMut { first, last, size };
            for _ in 0..(size + 1) / 2 {
                let _ = iter.next();
                let _ = iter.next_back();
            }
            assert!(iter.next().is_none());
        }
        test_iter_mut(9);
        test_iter_mut(10);
    }

    #[test]
    fn miri_test_into_iter() {
        fn test_into_iter(size: usize) {
            let mut list = new_list_for_test(size);
            let mut first = unsafe { Some(list.take_tail().unwrap()) };
            let last = first.as_mut().unwrap().last_mut();
            let last = NonNull::new(last);
            let mut iter = IntoIter { first, last, size };
            for _ in 0..(size + 1) / 2 {
                let _ = iter.next();
                let _ = iter.next_back();
            }
            assert!(iter.next().is_none());
        }

        test_into_iter(9);
        test_into_iter(10);
    }

    #[test]
    fn miri_test_retain() {
        let mut list = new_list_for_test(50);
        let _ = list.retain(|_, val| val % 2 == 0);
    }

    #[test]
    fn miri_test_check() {
        let list = new_list_for_test(100);
        list.check();
    }
}
