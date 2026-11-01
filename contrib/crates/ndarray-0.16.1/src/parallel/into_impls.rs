use crate::{ArcArray, Array, ArrayView, ArrayViewMut, Dimension};

use super::prelude::IntoParallelIterator;
use super::Parallel;

/// Requires crate feature `rayon`.
impl<'a, A, D> IntoParallelIterator for &'a Array<A, D>
where
    D: Dimension,
    A: Sync,
{
    type Item = &'a A;
    type Iter = Parallel<ArrayView<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter
    {
        self.view().into_par_iter()
    }
}

// This is allowed: goes through `.view()`
/// Requires crate feature `rayon`.
impl<'a, A, D> IntoParallelIterator for &'a ArcArray<A, D>
where
    D: Dimension,
    A: Sync,
{
    type Item = &'a A;
    type Iter = Parallel<ArrayView<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter
    {
        self.view().into_par_iter()
    }
}

/// Requires crate feature `rayon`.
impl<'a, A, D> IntoParallelIterator for &'a mut Array<A, D>
where
    D: Dimension,
    A: Sync + Send,
{
    type Item = &'a mut A;
    type Iter = Parallel<ArrayViewMut<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter
    {
        self.view_mut().into_par_iter()
    }
}

// This is allowed: goes through `.view_mut()`, which is unique access
/// Requires crate feature `rayon`.
impl<'a, A, D> IntoParallelIterator for &'a mut ArcArray<A, D>
where
    D: Dimension,
    A: Sync + Send + Clone,
{
    type Item = &'a mut A;
    type Iter = Parallel<ArrayViewMut<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter
    {
        self.view_mut().into_par_iter()
    }
}
