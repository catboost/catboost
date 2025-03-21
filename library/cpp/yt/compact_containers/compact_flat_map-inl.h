#ifndef COMPACT_FLAT_MAP_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_flat_map.h"
// For the sake of sane code completion.
#include "compact_flat_map.h"
#endif

namespace NYT {

///////////////////////////////////////////////////////////////////////////////

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <class TInputIterator>
TCompactFlatMap<TKey, TValue, N, TKeyCompare>::TCompactFlatMap(TInputIterator begin, TInputIterator end)
{
    insert(begin, end);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
TCompactFlatMap<TKey, TValue, N, TKeyCompare>::TCompactFlatMap(std::initializer_list<value_type> values)
    : TCompactFlatMap<TKey, TValue, N, TKeyCompare>(values.begin(), values.end())
{ }

template <class TKey, class TValue, size_t N, class TKeyCompare>
bool TCompactFlatMap<TKey, TValue, N, TKeyCompare>::operator==(const TCompactFlatMap& rhs) const
{
    return Storage_ == rhs.Storage_;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
bool TCompactFlatMap<TKey, TValue, N, TKeyCompare>::operator!=(const TCompactFlatMap& rhs) const
{
    return !(*this == rhs);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::begin()
{
    return Storage_.begin();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::begin() const
{
    return Storage_.begin();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::cbegin() const
{
    return Storage_.begin();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::end()
{
    return Storage_.end();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::end() const
{
    return Storage_.end();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::cend() const
{
    return Storage_.end();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::reserve(size_type n)
{
    Storage_.reserve(n);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::size_type TCompactFlatMap<TKey, TValue, N, TKeyCompare>::size() const
{
    return Storage_.size();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
int TCompactFlatMap<TKey, TValue, N, TKeyCompare>::ssize() const
{
    return static_cast<int>(Storage_.size());
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
bool TCompactFlatMap<TKey, TValue, N, TKeyCompare>::empty() const
{
    return Storage_.empty();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::clear()
{
    Storage_.clear();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::shrink_to_small()
{
    Storage_.shrink_to_small();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::find(const TOtherKey& k)
{
    auto [rangeBegin, rangeEnd] = equal_range(k);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::find(const TOtherKey& k) const
{
    auto [rangeBegin, rangeEnd] = equal_range(k);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
bool TCompactFlatMap<TKey, TValue, N, TKeyCompare>::contains(const TOtherKey& k) const
{
    return find(k) != end();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
auto TCompactFlatMap<TKey, TValue, N, TKeyCompare>::insert(const value_type& value) -> std::pair<iterator, bool>
{
    return do_insert(value);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
auto TCompactFlatMap<TKey, TValue, N, TKeyCompare>::insert(value_type&& value) -> std::pair<iterator, bool>
{
    return do_insert(std::move(value));
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <class TArg>
auto TCompactFlatMap<TKey, TValue, N, TKeyCompare>::do_insert(TArg&& value) -> std::pair<iterator, bool>
{
    auto [rangeBegin, rangeEnd] = equal_range(value.first);
    if (rangeBegin != rangeEnd) {
        return {rangeBegin, false};
    } else {
        auto it = Storage_.insert(rangeBegin, std::forward<TArg>(value));
        return {it, true};
    }
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <class TInputIterator>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::insert(TInputIterator begin, TInputIterator end)
{
    for (auto it = begin; it != end; ++it) {
        insert(*it);
    }
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <class... TArgs>
auto TCompactFlatMap<TKey, TValue, N, TKeyCompare>::emplace(TArgs&&... args) -> std::pair<iterator, bool>
{
    return insert(value_type(std::forward<TArgs>(args)...));
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
TValue& TCompactFlatMap<TKey, TValue, N, TKeyCompare>::operator[](const TKey& k)
{
    auto [it, inserted] = insert({k, TValue()});
    return it->second;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::erase(const TKey& k)
{
    auto [rangeBegin, rangeEnd] = equal_range(k);
    erase(rangeBegin, rangeEnd);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::erase(iterator pos)
{
    Storage_.erase(pos);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
void TCompactFlatMap<TKey, TValue, N, TKeyCompare>::erase(iterator b, iterator e)
{
    Storage_.erase(b, e);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
std::pair<typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator, typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator>
TCompactFlatMap<TKey, TValue, N, TKeyCompare>::equal_range(const TOtherKey& k)
{
    auto result = std::ranges::equal_range(Storage_, k, {}, &value_type::first);
    YT_ASSERT(result.size() <= 1);
    return result;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
std::pair<typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator, typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator>
TCompactFlatMap<TKey, TValue, N, TKeyCompare>::equal_range(const TOtherKey& k) const
{
    auto result = std::ranges::equal_range(Storage_, k, {}, &value_type::first);
    YT_ASSERT(result.size() <= 1);
    return result;
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::lower_bound(const TOtherKey& k) const
{
    return std::ranges::lower_bound(Storage_, k, {}, &value_type::first);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::lower_bound(const TOtherKey& k)
{
    return std::ranges::lower_bound(Storage_, k, {}, &value_type::first);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::const_iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::upper_bound(const TOtherKey& k) const
{
    return std::ranges::upper_bound(Storage_, k, {}, &value_type::first);
}

template <class TKey, class TValue, size_t N, class TKeyCompare>
template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
typename TCompactFlatMap<TKey, TValue, N, TKeyCompare>::iterator TCompactFlatMap<TKey, TValue, N, TKeyCompare>::upper_bound(const TOtherKey& k)
{
    return std::ranges::upper_bound(Storage_, k, {}, &value_type::first);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
