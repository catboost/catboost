#ifndef ORDERED_HASH_MAP_INL_H_
#error "Direct inclusion of this file is not allowed, include ordered_hash_map.h"
// For the sake of sane code completion.
#include "ordered_hash_map.h"
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
std::pair<const TKey, TValue>& TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::TSelectPair::operator()(TItem& item) const
{
    return item;
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
const std::pair<const TKey, TValue>& TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::TSelectPair::operator()(const TItem& item) const
{
    return item;
}

////////////////////////////////////////////////////////////////////////////////

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::TOrderedHashMap(const TOrderedHashMap& other)
{
    *this = other;
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::operator=(const TOrderedHashMap& other) -> TOrderedHashMap&
{
    if (this == &other) {
        return *this;
    }
    clear();
    Table_.reserve(other.size());
    for (const auto& item : other) {
        emplace(item.first, item.second);
    }
    return *this;
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class... TArgs>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::emplace(TArgs&&... args) -> std::pair<iterator, bool>
{
    auto [it, success] = Table_.emplace_unique(std::forward<TArgs>(args)...);
    if (success) {
        List_.PushBack(&*it);
    }
    return {MakeMappedIterator(TListIterator(&*it), TSelectPair{}), success};
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class TOtherKey>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::find(const TOtherKey& key) -> iterator
{
    auto it = Table_.find(key);
    if (it == Table_.end()) {
        return MakeMappedIterator(List_.end(), TSelectPair{});
    }
    return MakeMappedIterator(TListIterator(&*it), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class TOtherKey>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::find(const TOtherKey& key) const -> const_iterator
{
    auto it = Table_.find(key);
    if (it == Table_.end()) {
        return MakeMappedIterator(List_.end(), TSelectPair{});
    }
    return MakeMappedIterator(TListConstIterator(&*it), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class TOtherKey>
bool TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::contains(const TOtherKey& key) const
{
    return Table_.find(key) != Table_.end();
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class TOtherKey>
TValue& TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::operator[](const TOtherKey& key)
{
    typename TTable::insert_ctx ctx = nullptr;
    auto it = Table_.find_i(key, ctx);
    if (it != Table_.end()) {
        return it->second;
    }
    it = Table_.emplace_direct(ctx, std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple());
    List_.PushBack(&*it);
    return it->second;
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
template <class TOtherKey>
size_t TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::erase(const TOtherKey& key)
{
    return Table_.erase_one(key);
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
void TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::erase(iterator it)
{
    erase(it->first);
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::begin() -> iterator
{
    return MakeMappedIterator(List_.begin(), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::end() -> iterator
{
    return MakeMappedIterator(List_.end(), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::begin() const -> const_iterator
{
    return MakeMappedIterator(List_.begin(), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
auto TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::end() const -> const_iterator
{
    return MakeMappedIterator(List_.end(), TSelectPair{});
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
size_t TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::size() const
{
    return Table_.size();
}

template <class TKey, class TValue, class THashFunction, class TEqualFunction>
void TOrderedHashMap<TKey, TValue, THashFunction, TEqualFunction>::clear()
{
    Table_.clear();
    YT_ASSERT(List_.begin() == List_.end());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
