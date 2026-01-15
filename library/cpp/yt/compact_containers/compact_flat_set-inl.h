#ifndef COMPACT_FLAT_SET_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_flat_set.h"
// For the sake of sane code completion.
#include "compact_flat_set.h"
#endif

namespace NYT {

///////////////////////////////////////////////////////////////////////////////

template <class TValue, size_t N>
template <class TInputIterator>
TCompactFlatSet<TValue, N>::TCompactFlatSet(TInputIterator begin, TInputIterator end)
{
    insert(begin, end);
}

template <class TValue, size_t N>
TCompactFlatSet<TValue, N>::TCompactFlatSet(std::initializer_list<TValue> values)
    : TCompactFlatSet<TValue, N>(values.begin(), values.end())
{ }

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::iterator TCompactFlatSet<TValue, N>::begin()
{
    return Storage_.begin();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::begin() const
{
    return Storage_.begin();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::cbegin() const
{
    return Storage_.begin();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::iterator TCompactFlatSet<TValue, N>::end()
{
    return Storage_.end();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::end() const
{
    return Storage_.end();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::cend() const
{
    return Storage_.end();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::reverse_iterator TCompactFlatSet<TValue, N>::rbegin()
{
    return Storage_.rbegin();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_reverse_iterator TCompactFlatSet<TValue, N>::rbegin() const
{
    return Storage_.rbegin();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::reverse_iterator TCompactFlatSet<TValue, N>::rend()
{
    return Storage_.rend();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_reverse_iterator TCompactFlatSet<TValue, N>::rend() const
{
    return Storage_.rend();
}

template <class TValue, size_t N>
void TCompactFlatSet<TValue, N>::reserve(size_type size)
{
    Storage_.reserve(size);
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::size_type TCompactFlatSet<TValue, N>::size() const
{
    return Storage_.size();
}

template <class TValue, size_t N>
int TCompactFlatSet<TValue, N>::ssize() const
{
    return static_cast<int>(Storage_.size());
}

template <class TValue, size_t N>
bool TCompactFlatSet<TValue, N>::empty() const
{
    return Storage_.empty();
}

template <class TValue, size_t N>
void TCompactFlatSet<TValue, N>::clear()
{
    Storage_.clear();
}

template <class TValue, size_t N>
void TCompactFlatSet<TValue, N>::shrink_to_small()
{
    Storage_.shrink_to_small();
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::iterator TCompactFlatSet<TValue, N>::find(const TValue& value)
{
    auto [rangeBegin, rangeEnd] = equal_range(value);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::find(const TValue& value) const
{
    auto [rangeBegin, rangeEnd] = equal_range(value);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::size_type TCompactFlatSet<TValue, N>::count(const TValue& value) const
{
    auto [rangeBegin, rangeEnd] = equal_range(value);
    return rangeEnd - rangeBegin;
}


template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::lower_bound(const TValue& value) const
{
    return std::ranges::lower_bound(Storage_, value);
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::iterator TCompactFlatSet<TValue, N>::lower_bound(const TValue& value)
{
    return std::ranges::lower_bound(Storage_, value);
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::const_iterator TCompactFlatSet<TValue, N>::upper_bound(const TValue& value) const
{
    return std::ranges::upper_bound(Storage_, value);
}

template <class TValue, size_t N>
typename TCompactFlatSet<TValue, N>::iterator TCompactFlatSet<TValue, N>::upper_bound(const TValue& value)
{
    return std::ranges::upper_bound(Storage_, value);
}

template <class TValue, size_t N>
std::pair<typename TCompactFlatSet<TValue, N>::iterator, typename TCompactFlatSet<TValue, N>::iterator>
TCompactFlatSet<TValue, N>::equal_range(const TValue& value)
{
    auto result = std::ranges::equal_range(Storage_.begin(), Storage_.end(), value);
    YT_ASSERT(result.size() <= 1);
    return result;
}

template <class TValue, size_t N>
std::pair<typename TCompactFlatSet<TValue, N>::const_iterator, typename TCompactFlatSet<TValue, N>::const_iterator>
TCompactFlatSet<TValue, N>::equal_range(const TValue& value) const
{
    auto result = std::ranges::equal_range(Storage_.begin(), Storage_.end(), value);
    YT_ASSERT(result.size() <= 1);
    return result;
}

template <class TValue, size_t N>
bool TCompactFlatSet<TValue, N>::contains(const TValue& value) const
{
    return find(value) != end();
}

template <class TValue, size_t N>
auto TCompactFlatSet<TValue, N>::insert(const TValue& value) -> std::pair<iterator, bool>
{
    return DoInsert(value);
}

template <class TValue, size_t N>
auto TCompactFlatSet<TValue, N>::insert(TValue&& value) -> std::pair<iterator, bool>
{
    return DoInsert(std::move(value));
}

template <class TValue, size_t N>
template <class TInputIterator>
void TCompactFlatSet<TValue, N>::insert(TInputIterator begin, TInputIterator end)
{
    for (auto it = begin; it != end; ++it) {
        insert(*it);
    }
}

template <class TValue, size_t N>
bool TCompactFlatSet<TValue, N>::erase(const TValue& value)
{
    auto [rangeBegin, rangeEnd] = equal_range(value);
    erase(rangeBegin, rangeEnd);
    return rangeBegin != rangeEnd;
}

template <class TValue, size_t N>
void TCompactFlatSet<TValue, N>::erase(iterator pos)
{
    Storage_.erase(pos);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class TValue, size_t N>
void TCompactFlatSet<TValue, N>::erase(iterator begin, iterator end)
{
    Storage_.erase(begin, end);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class TValue, size_t N>
template <class TArg>
auto TCompactFlatSet<TValue, N>::DoInsert(TArg&& value) -> std::pair<iterator, bool>
{
    auto [rangeBegin, rangeEnd] = equal_range(value);
    if (rangeBegin != rangeEnd) {
        return {rangeBegin, false};
    } else {
        auto it = Storage_.insert(rangeBegin, std::forward<TArg>(value));
        return {it, true};
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
