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
| Comparison | `operator==`, `operator<` (lexicographical) |

Notable differences from `std::vector`:

- No `reserve()`/`capacity()`/`shrink_to_fit()` and no `data()` — storage is paged, not contiguous.

## Iterators

Iterators are random-access and are implemented as an *(owner pointer, offset)* pair. Consequences:

- Iterators are not invalidated by `push_back`/`emplace_back` (an `end()` iterator taken earlier keeps pointing to the same logical position).
- Dereferencing goes through the vector, so an iterator is only valid while its source container is alive.
- To get the current index of an element from an iterator, call `it.GetIndex()` — it returns the offset of the pointed-to element within the container (equivalent to `it - begin()`).


## Complexity

| Operation | Complexity |
|---|---|
| `operator[]` / `at()` | O(1) |
| `push_back` / `emplace_back` | O(1) amortized (page allocation at most every `PageSize` appends) |
| `pop_back` | O(1) |
| `erase` | O(n) — shifts all following elements |
| `clear` | O(n) for non-trivially destructible `T`, O(pages) otherwise |

## Notes

- Pages are allocated as raw storage; elements are constructed in place and destroyed explicitly, so non-trivially destructible types are handled correctly.
- Destruction of trivially destructible types is skipped entirely, making `clear()` and the destructor fast for POD-like types.
- The copy constructor is exception-safe: on a throw during copying, already-constructed elements are destroyed.
