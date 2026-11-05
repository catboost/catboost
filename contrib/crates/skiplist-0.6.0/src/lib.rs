//! A skiplist is a way of storing sorted elements in such a way that they can
//! be accessed, inserted and removed, all in `O(log(n))` on average.
//!
//! Conceptually, a skiplist is arranged as follows:
//!
//! ```text
//! <head> ----------> [2] --------------------------------------------------> [9] ---------->
//! <head> ----------> [2] ------------------------------------[7] ----------> [9] ---------->
//! <head> ----------> [2] ----------> [4] ------------------> [7] ----------> [9] --> [10] ->
//! <head> --> [1] --> [2] --> [3] --> [4] --> [5] --> [6] --> [7] --> [8] --> [9] --> [10] ->
//! ```
//!
//! Each node contains at the very least a link to the next element in the list
//! (corresponding to the lowest level in the above diagram), but it can
//! randomly contain more links which skip further down the list (the *towers*
//! in the above diagram).  This allows for the algorithm to move down the list
//! faster than having to visit every element.
//!
//! Conceptually, the skiplist can be thought of as a stack of linked lists.  At
//! the very bottom is the full linked list with every element, and each layer
//! above corresponds to a linked list containing a random subset of the
//! elements from the layer immediately below it.  The probability distribution
//! that determines this random subset can be customized, but typically a layer
//! will contain half the nodes from the layer below.
//!
//! # Safety
//!
//! The ordered skiplist relies on a well-behaved comparison function.
//! Specifically, given some ordering function `f(a, b)`, it **must** satisfy
//! the following properties:
//!
//! - Be well defined: `f(a, b)` should always return the same value
//! - Be anti-symmetric: `f(a, b) == Greater` if and only if `f(b, a) == Less`,
//!   and `f(a, b) == Equal == f(b, a)`.
//! - By transitive: If `f(a, b) == Greater` and `f(b, c) == Greater` then `f(a,
//!   c) == Greater`.
//!
//! **Failure to satisfy these properties can result in unexpected behavior at
//! best, and at worst will cause a segfault, null deref, or some other bad
//! behavior.**

// In this library, the notion of 'height' of a node refers to how many links a
// node has (as a result, the minimum height is 1).  The 'levels' refer to the
// layers in the above diagram, with level 0 being the bottom-most layer, level
// 1 being the one above level 0, etc.

extern crate rand;
pub mod level_generator;
pub mod ordered_skiplist;
pub mod skiplist;
pub mod skipmap;
mod skipnode;

#[expect(
    clippy::pub_use,
    clippy::useless_attribute,
    reason = "Temporary silencing clippy"
)]
pub use crate::{ordered_skiplist::OrderedSkipList, skiplist::SkipList, skipmap::SkipMap};
