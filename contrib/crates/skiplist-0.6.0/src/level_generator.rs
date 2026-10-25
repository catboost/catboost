//! Level generator for skip lists.
//!
//! Skiplists use a probabilistic distribution to determine the links. If
//! `$S^{(0)} = \{n_1, n_2, \dots, n_p\}$` is the set of all the nodes, then
//! `$S^{(i)} \subseteq S^{(i-1)}$` is the set of nodes at level `$i$`.
//!
//! At each level, the nodes are linked to the next node in the list, and
//! therefore as the level increase and the number of nodes decrease, it is
//! expected that the distance between the nodes also increases making the
//! traversal of the list faster.
//!
//! The simplest way to implement this is to have the same probability `$p$`
//! that a node is present in the next level, with `$p$` being constant. This is
//! known as a geometric distribution and is implemented in
//! [`Geometric`][geometric::Geometric].
//!
//! It is very unlikely that this will need to be changed as the default should
//! suffice, but if need be custom level generators can be implemented through
//! the [`LevelGenerator`] trait.

pub mod geometric;

/// Trait for generating the level of a new node to be inserted into the
/// skiplist.
pub trait LevelGenerator {
    /// The total number of levels that are assumed to exist.
    #[must_use]
    fn total(&self) -> usize;
    /// Generate a random level for a new node in the range `[0, total)`.
    ///
    /// This function must not return a level greater or equal to
    /// [`total`][LevelGenerator::total].
    #[must_use]
    fn level(&mut self) -> usize;
}
