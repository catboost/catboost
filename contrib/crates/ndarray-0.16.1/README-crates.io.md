
`ndarray` implements an *n*-dimensional container for general elements and for
numerics.

In *n*-dimensional we include for example 1-dimensional rows or columns,
2-dimensional matrices, and higher dimensional arrays. If the array has *n*
dimensions, then an element in the array is accessed by using that many indices.
Each dimension is also called an *axis*.

## Highlights

- Generic *n*-dimensional array
- Slicing, also with arbitrary step size, and negative indices to mean
  elements from the end of the axis.
- Views and subviews of arrays; iterators that yield subviews.
- Higher order operations and arithmetic are performant
- Array views can be used to slice and mutate any `[T]` data using
  `ArrayView::from` and `ArrayViewMut::from`.
- `Zip` for lock step function application across two or more arrays or other
  item producers (`NdProducer` trait).
