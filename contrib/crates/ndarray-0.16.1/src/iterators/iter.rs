//! Producers, iterables and iterators.
//!
//! This module collects all concrete producer, iterable and iterator
//! implementation structs.
//!
//!
//! See also [`NdProducer`](crate::NdProducer).

pub use crate::dimension::Axes;
pub use crate::indexes::{Indices, IndicesIter};
pub use crate::iterators::{
    AxisChunksIter,
    AxisChunksIterMut,
    AxisIter,
    AxisIterMut,
    AxisWindows,
    ExactChunks,
    ExactChunksIter,
    ExactChunksIterMut,
    ExactChunksMut,
    IndexedIter,
    IndexedIterMut,
    IntoIter,
    Iter,
    IterMut,
    Lanes,
    LanesIter,
    LanesIterMut,
    LanesMut,
    Windows,
};
