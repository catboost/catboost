//! Visitor traits over skip list nodes.
//!
//! This module defines the traits for visitor structs, allowing for nodes to be
//! found quickly.  If we suppose we have a list of the form
//!
//! ```text
//! [3] head --------------------------> 6
//! [2] head ---------------------> 5 -> 6
//! [1] head ------> 2 -----------> 5 -> 6 ------> 8
//! [*] head -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
//! ```
//!
//! Starting from the head node, the paths to targets will be:
//!
//! - If trying to reach node 9, the links traversed would be `head -> 6 -> 8 ->
//!   9`.
//! - If trying to reach node 3, the algorithm would try the `head -> 6` and
//!   `head -> 5` links first, realise they overshoot the target, and then take
//!   the path `head -> 2 -> 3`.
//!
//! If the only reason for visiting the list is to find a node, the visitor
//! implementation should not worry about keeping a record of the links. More
//! generally though, if the intent is to insert or remove a node, the visitor
//! needs to keep track of the links that will need to be modified.
//!
//! In the `head -> 9` example above, the links that need to be modified at the
//! `6[3]`, `6[2]` and `8[1]` nodes, as well as the immediately previous node
//! `8` (even though most links are `None`); and in the `head -> 3` example, the
//! links that need to be modified are `head[3]`, `head[2]` and `2[1]`

mod index;
// mod index_mut;

/// Basic interface for a visitor.
///
/// This trait defines the interface for visiting nodes.
trait Visitor {
    /// Node reference associated type.
    type NodeRef;

    /// The current node being traversed.
    fn current(&self) -> Self::NodeRef;

    /// The current level of the visitor
    ///
    /// The visitor will always start at the highest level, and work its way
    /// down to the lowest level.
    fn level(&self) -> usize;

    /// Whether the traverser has reached its destination.
    fn found(&self) -> bool;

    /// Step towards the target.
    ///
    /// This attempts to take a single step towards the target. This may be
    /// either through a link, or by moving through the node's `next` pointer.
    ///
    /// If successful, this will return `Ok` with a reference to the new node
    /// that the traverser is on. On the other hand, if the traverser cannot
    /// take an additional step, this will return `Err` with a boolean
    /// indicating whether the traverser has reached the target.
    ///
    /// # Errors
    ///
    /// If the traverser can not take any further steps down the skiplist, this
    /// function should return a `Err(bool)` with `true` indicating that the
    /// traverser has reached the target
    fn step(&mut self) -> Result<Self::NodeRef, bool>;

    /// Traverse until we reach the target, or we can no longer step.
    ///
    /// This will continue to step until we reach the target, or we can no
    /// longer step. If we reach the target, this will return the target node
    /// reference, otherwise it will return `None`.
    fn traverse(&mut self) -> Option<Self::NodeRef> {
        let mut current = self.current();
        loop {
            match self.step() {
                Ok(node) => current = node,
                Err(true) => return Some(current),
                Err(false) => return None,
            }
        }
    }
}

/// Extension to the [`Visitor`] trait for mutation.
///
/// This trait extends the [`Visitor`] trait to allow for mutation of the
/// current node and the links around it.
trait VisitorMut: Visitor {
    /// Mutable node reference associated type.
    type NodeMut;
    /// Link reference associated type.
    type LinkRef;
    /// Mutable link reference associated type.
    type LinkMut;

    /// Get a mutable reference to the current node.
    fn current_mut(&mut self) -> Self::NodeMut;

    /// Get the links that may need to be modified.
    ///
    /// As the visitor finds the target node, it will need to keep track of
    /// links which may need to be modified. Specifically, these fall into three
    /// categories:
    ///
    /// 1.  Missing links. If a new node is being inserted, new links may need
    ///     to be created.
    /// 2.  Links that reach the target.
    ///
    ///     For example, if the target node is removed, these links need to be
    ///     updated, either by removing them or merging them with links from the
    ///     node being removed.
    /// 3.  Links that overshoot the target.
    ///
    ///     For example, if the target node is removed, these links need to be
    ///     updated to be shorter.
    fn links(&self) -> &[Option<Self::LinkRef>];

    /// Get a mutable reference to the links.
    fn links_mut(&mut self) -> &[Option<Self::LinkMut>];
}
