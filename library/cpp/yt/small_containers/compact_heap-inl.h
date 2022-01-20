#ifndef COMPACT_HEAP_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_heap.h"
// For the sake of sane code completion.
#include "compact_heap.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <algorithm>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N, class C>
TCompactHeap<T, N, C>::TCompactHeap(C comparator) noexcept
    : Comparator_(TReverseComparator(std::move(comparator)))
{ }

template <class T, size_t N, class C>
void TCompactHeap<T, N, C>::push(T value)
{
    bool wasInline = IsInline();
    Heap_.push_back(std::move(value));
    if (Y_UNLIKELY(!IsInline())) {
        if (wasInline) {
            std::make_heap(Heap_.begin(), Heap_.end(), Comparator_);
        } else {
            std::push_heap(Heap_.begin(), Heap_.end(), Comparator_);
        }
    }
}

template <class T, size_t N, class C>
void TCompactHeap<T, N, C>::pop()
{
    YT_ASSERT(!empty());

    if (Y_LIKELY(IsInline())) {
        auto minIt = std::max_element(Heap_.begin(), Heap_.end(), Comparator_);
        std::swap(*minIt, Heap_.back());
        Heap_.pop_back();
    } else {
        std::pop_heap(Heap_.begin(), Heap_.end(), Comparator_);
        Heap_.pop_back();
    }
}

template <class T, size_t N, class C>
auto TCompactHeap<T, N, C>::get_min() const -> const_reference
{
    YT_ASSERT(!empty());

    if (Y_LIKELY(IsInline())) {
        return *std::max_element(Heap_.begin(), Heap_.end(), Comparator_);
    } else {
        return Heap_.front();
    }
}

template <class T, size_t N, class C>
auto TCompactHeap<T, N, C>::extract_min() -> value_type
{
    YT_ASSERT(!empty());

    if (Y_LIKELY(IsInline())) {
        auto minIt = std::max_element(Heap_.begin(), Heap_.end(), Comparator_);
        std::swap(*minIt, Heap_.back());
        auto value = Heap_.back();
        Heap_.pop_back();

        return value;
    } else {
        std::pop_heap(Heap_.begin(), Heap_.end(), Comparator_);
        auto value = std::move(Heap_.back());
        Heap_.pop_back();

        return value;
    }
}

template <class T, size_t N, class C>
auto TCompactHeap<T, N, C>::begin() const -> const_iterator
{
    return Heap_.begin();
}

template <class T, size_t N, class C>
auto TCompactHeap<T, N, C>::end() const -> const_iterator
{
    return Heap_.end();
}

template <class T, size_t N, class C>
void TCompactHeap<T, N, C>::swap(TCompactHeap<T, N, C>& other)
{
    Heap_.swap(other.Heap_);
    std::swap(Comparator_, other.Comparator_);
}

template <class T, size_t N, class C>
size_t TCompactHeap<T, N, C>::size() const
{
    return Heap_.size();
}

template <class T, size_t N, class C>
size_t TCompactHeap<T, N, C>::capacity() const
{
    return Heap_.capacity();
}

template <class T, size_t N, class C>
size_t TCompactHeap<T, N, C>::max_size() const
{
    return Heap_.max_size();
}

template <class T, size_t N, class C>
bool TCompactHeap<T, N, C>::empty() const
{
    return Heap_.empty();
}

template <class T, size_t N, class C>
void TCompactHeap<T, N, C>::shrink_to_small()
{
    Heap_.shrink_to_small();
}

template <class T, size_t N, class C>
bool TCompactHeap<T, N, C>::IsInline() const
{
    return Heap_.capacity() == N;
}

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N, class C>
TCompactHeap<T, N, C>::TReverseComparator::TReverseComparator(C underlying)
    : Underlying_(std::move(underlying))
{ }

template <class T, size_t N, class C>
bool TCompactHeap<T, N, C>::TReverseComparator::operator()(const T& lhs, const T& rhs) const
{
    return Underlying_(rhs, lhs);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
