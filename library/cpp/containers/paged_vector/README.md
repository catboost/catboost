# TPagedVector

`NPagedVector::TPagedVector<T, PageSize>` is a dynamic sequence container implemented as a 2-level radix tree: elements are stored in fixed-size, individually heap-allocated pages, and a top-level vector holds pointers to those pages.

```cpp
#include <library/cpp/containers/paged_vector/paged_vector.h>

namespace NPagedVector {
    template <class T, ui32 PageSize = 1u << 20u>
    class TPagedVector;
}
```

- `T` — element type.
- `PageSize` — number of elements per page (default: `1u << 20u` = 1,048,576 elements).

## Why use it instead of TVector / std::vector?

- **No reallocation of elements.** Growth allocates a new page instead of reallocating and moving the entire buffer. Elements are never moved on `push_back`/`emplace_back`, so references and pointers to existing elements remain valid when appending (iterators are offset-based and also stay usable).
- **No large contiguous allocations.** Memory is requested in page-size chunks, which is friendlier to the allocator for very large containers.
- **Cheaper worst-case append.** `push_back` never triggers an O(n) copy; the cost is at most one page allocation.

The trade-off is that storage is not contiguous (no `data()`), and indexing does one extra pointer dereference (`idx / PageSize`, `idx % PageSize`).

## API overview

The interface mirrors a subset of `std::vector`:

| Category | Members |
|---|---|
| Construction | default, copy, move, `TPagedVector(TIter b, TIter e)` |
| Assignment | copy, move, `swap()` |
| Element access | `operator[]`, `at()` (throws `std::out_of_range`), `front()`, `back()` |
| Iterators | `begin()/end()`, `rbegin()/rend()` + const versions; random-access iterators |
| Capacity | `size()`, `empty()`, `explicit operator bool()` (true when non-empty) |
| Modifiers | `push_back()`, `emplace_back()` (returns a reference), `pop_back()`, `append(b, e)`, `erase(it)`, `erase(b, e)`, `resize()`, `clear()` |
| Iteration helpers | `ForEach(fn)`, `ForEachReverse(fn)` |
| Comparison | `operator==`, `operator<` (lexicographical) |

Notable differences from `std::vector`:

- No `reserve()`/`capacity()`/`shrink_to_fit()` and no `data()` — storage is paged, not contiguous.

## Iterators

Iterators are random-access and are implemented as an *(owner pointer, offset)* pair. Consequences:

- Iterators are not invalidated by `push_back`/`emplace_back` (an `end()` iterator taken earlier keeps pointing to the same logical position).
- Dereferencing goes through the vector, so an iterator is only valid while its source container is alive.
- To get the current index of an element from an iterator, call `it.GetIndex()` — it returns the offset of the pointed-to element within the container (equivalent to `it - begin()`).

## Iteration helpers

```cpp
template <class Function>
void ForEach(Function fn) const;

template <class Function>
void ForEachReverse(Function fn) const;
```

`ForEach` applies `fn` to every element **from the first to the last**; `ForEachReverse` applies `fn` **from the last to the first**.

These are faster than iterating with `begin()/end()` or `rbegin()/rend()`: they walk the pages directly through raw pointers, avoiding the two levels of indirection that the offset-based iterators go through on each dereference. This matters for containers with a large `PageSize` (the default is 1M elements per page), where the inner per-page loop is tight.

```cpp
TPagedVector<int, 1024> v;
// ... fill v ...

long long sum = 0;
v.ForEach([&](int x) { sum += x; });

// process elements back-to-front, e.g. for a stack-like traversal
v.ForEachReverse([&](int x) {
    // ...
});
```

Notes:

- The order is well-defined and contiguous: `ForEach` visits element `0, 1, ..., size()-1`; `ForEachReverse` visits `size()-1, ..., 1, 0`.
- Both are O(n) and do not allocate.

## Complexity

| Operation | Complexity |
|---|---|
| `operator[]` / `at()` | O(1) |
| `push_back` / `emplace_back` | O(1) amortized (page allocation at most every `PageSize` appends) |
| `pop_back` | O(1) |
| `erase` | O(n) — shifts all following elements |
| `clear` | O(n) for non-trivially destructible `T`, O(pages) otherwise |
| `ForEach` / `ForEachReverse` | O(n), no allocations |

## Notes

- Pages are allocated as raw storage; elements are constructed in place and destroyed explicitly, so non-trivially destructible types are handled correctly.
- Destruction of trivially destructible types is skipped entirely, making `clear()` and the destructor fast for POD-like types.
- The copy constructor is exception-safe: on a throw during copying, already-constructed elements are destroyed.
