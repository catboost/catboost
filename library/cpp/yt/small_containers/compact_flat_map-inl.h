#ifndef COMPACT_FLAT_MAP_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_flat_map.h"
// For the sake of sane code completion.
#include "compact_flat_map.h"
#endif

namespace NYT {

///////////////////////////////////////////////////////////////////////////////

template <class K, class V, unsigned N>
template <class TInputIterator>
TCompactFlatMap<K, V, N>::TCompactFlatMap(TInputIterator begin, TInputIterator end)
{
    insert(begin, end);
}

template <class K, class V, unsigned N>
bool TCompactFlatMap<K, V, N>::operator==(const TCompactFlatMap& rhs) const
{
    return Storage_ == rhs.Storage_;
}

template <class K, class V, unsigned N>
bool TCompactFlatMap<K, V, N>::operator!=(const TCompactFlatMap& rhs) const
{
    return !(*this == rhs);
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::iterator TCompactFlatMap<K, V, N>::begin()
{
    return Storage_.begin();
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::const_iterator TCompactFlatMap<K, V, N>::begin() const
{
    return Storage_.begin();
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::iterator TCompactFlatMap<K, V, N>::end()
{
    return Storage_.end();
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::const_iterator TCompactFlatMap<K, V, N>::end() const
{
    return Storage_.end();
}

template <class K, class V, unsigned N>
void TCompactFlatMap<K, V, N>::reserve(size_type n)
{
    Storage_.reserve(n);
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::size_type TCompactFlatMap<K, V, N>::size() const
{
    return Storage_.size();
}

template <class K, class V, unsigned N>
int TCompactFlatMap<K, V, N>::ssize() const
{
    return static_cast<int>(Storage_.size());
}

template <class K, class V, unsigned N>
bool TCompactFlatMap<K, V, N>::empty() const
{
    return Storage_.empty();
}

template <class K, class V, unsigned N>
void TCompactFlatMap<K, V, N>::clear()
{
    Storage_.clear();
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::iterator TCompactFlatMap<K, V, N>::find(const K& k)
{
    auto [rangeBegin, rangeEnd] = EqualRange(k);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class K, class V, unsigned N>
typename TCompactFlatMap<K, V, N>::const_iterator TCompactFlatMap<K, V, N>::find(const K& k) const
{
    auto [rangeBegin, rangeEnd] = EqualRange(k);
    return rangeBegin == rangeEnd ? end() : rangeBegin;
}

template <class K, class V, unsigned N>
bool TCompactFlatMap<K, V, N>::contains(const K& k) const
{
    return find(k) != end();
}

template <class K, class V, unsigned N>
auto TCompactFlatMap<K, V, N>::insert(const value_type& value) -> std::pair<iterator, bool>
{
    auto [rangeBegin, rangeEnd] = EqualRange(value.first);
    if (rangeBegin != rangeEnd) {
        return {rangeBegin, false};
    } else {
        auto it = Storage_.insert(rangeBegin, value);
        return {it, true};
    }
}

template <class K, class V, unsigned N>
template <class TInputIterator>
void TCompactFlatMap<K, V, N>::insert(TInputIterator begin, TInputIterator end)
{
    for (auto it = begin; it != end; ++it) {
        insert(*it);
    }
}

template <class K, class V, unsigned N>
template <class... TArgs>
auto TCompactFlatMap<K, V, N>::emplace(TArgs&&... args) -> std::pair<iterator, bool>
{
    return insert(value_type(std::forward<TArgs>(args)...));
}

template <class K, class V, unsigned N>
V& TCompactFlatMap<K, V, N>::operator[](const K& k)
{
    auto [it, inserted] = insert({k, V()});
    return it->second;
}

template <class K, class V, unsigned N>
void TCompactFlatMap<K, V, N>::erase(const K& k)
{
    auto [rangeBegin, rangeEnd] = EqualRange(k);
    erase(rangeBegin, rangeEnd);
}

template <class K, class V, unsigned N>
void TCompactFlatMap<K, V, N>::erase(iterator pos)
{
    Storage_.erase(pos);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class K, class V, unsigned N>
void TCompactFlatMap<K, V, N>::erase(iterator b, iterator e)
{
    Storage_.erase(b, e);

    // Try to keep the storage inline. This is why erase doesn't return an iterator.
    Storage_.shrink_to_small();
}

template <class K, class V, unsigned N>
std::pair<typename TCompactFlatMap<K, V, N>::iterator, typename TCompactFlatMap<K, V, N>::iterator>
TCompactFlatMap<K, V, N>::EqualRange(const K& k)
{
    auto result = std::equal_range(Storage_.begin(), Storage_.end(), k, TKeyComparer());
    YT_ASSERT(std::distance(result.first, result.second) <= 1);
    return result;
}

template <class K, class V, unsigned N>
std::pair<typename TCompactFlatMap<K, V, N>::const_iterator, typename TCompactFlatMap<K, V, N>::const_iterator>
TCompactFlatMap<K, V, N>::EqualRange(const K& k) const
{
    auto result = std::equal_range(Storage_.begin(), Storage_.end(), k, TKeyComparer());
    YT_ASSERT(std::distance(result.first, result.second) <= 1);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
