#pragma once

#include "compact_vector.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A heap structure optimized for storing elements inline
//! and with little memory overhead. See TCompactVector.
/*!
 *  When inline, uses linear search for selecting minimum
 *  instead of storing heap.
 */
template <class T, size_t N, class C = std::less<T>>
class TCompactHeap
{
public:
    static_assert(N <= 8, "TCompactHeap is not optimal for large N");

    explicit TCompactHeap(C comparator = C()) noexcept;

    using value_type = T;

    using const_reference = const T&;

    using const_iterator = const T*;

    using difference_type = ptrdiff_t;
    using size_type = size_t;

    void push(T value);
    void pop();

    const_reference get_min() const;
    value_type extract_min();

    const_iterator begin() const;
    const_iterator end() const;

    void swap(TCompactHeap<T, N, C>& other);

    size_t size() const;
    size_t capacity() const;
    size_t max_size() const;

    bool empty() const;

    void shrink_to_small();

private:
    TCompactVector<T, N> Heap_;

    class TReverseComparator
    {
    public:
        explicit TReverseComparator(C underlying);

        bool operator()(const T& lhs, const T& rhs) const;

    private:
        C Underlying_;
    };
    TReverseComparator Comparator_;

    bool IsInline() const;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPACT_HEAP_INL_H_
#include "compact_heap-inl.h"
#undef COMPACT_HEAP_INL_H_
