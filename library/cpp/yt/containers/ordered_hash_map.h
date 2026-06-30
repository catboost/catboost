#pragma once

#include <library/cpp/iterator/mapped.h>

#include <util/generic/hash_table.h>
#include <util/generic/intrlist.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A hash map where the iteration order is the insertion order.
template <
    class TKey,
    class TValue,
    class THashFunction = THash<TKey>,
    class TEqualFunction = TEqualTo<TKey>
>
class TOrderedHashMap
{
private:
    struct TItem
        : public std::pair<const TKey, TValue>
        , public TIntrusiveListItem<TItem>
    {
        using std::pair<const TKey, TValue>::pair;
    };

    struct TSelectPair
    {
        std::pair<const TKey, TValue>& operator()(TItem& item) const;
        const std::pair<const TKey, TValue>& operator()(const TItem& item) const;
    };

    using TList = TIntrusiveList<TItem>;
    using TListIterator = TList::iterator;
    using TListConstIterator = TList::const_iterator;

public:
    using iterator = TMappedIterator<TListIterator, TSelectPair>;
    using const_iterator = TMappedIterator<TListConstIterator, TSelectPair>;

public:
    TOrderedHashMap() = default;
    TOrderedHashMap(const TOrderedHashMap& other);
    TOrderedHashMap(TOrderedHashMap&& other) = default;

    TOrderedHashMap& operator=(const TOrderedHashMap& other);
    TOrderedHashMap& operator=(TOrderedHashMap&& other) = default;

    template <typename... TArgs>
    std::pair<iterator, bool> emplace(TArgs&&... args);

    template <class TOtherKey>
    iterator find(const TOtherKey& key);

    template <class TOtherKey>
    const_iterator find(const TOtherKey& key) const;

    template <class TOtherKey>
    bool contains(const TOtherKey& key) const;

    template <class TOtherKey>
    TValue& operator[](const TOtherKey& key);

    template <class TOtherKey>
    size_t erase(const TOtherKey& key);
    void erase(iterator it);

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    size_t size() const;

    void clear();

private:
    using TTable = THashTable<TItem, TKey, THashFunction, TSelect1st, TEqualFunction, std::allocator<TKey>>;

private:
    TList List_;
    TTable Table_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ORDERED_HASH_MAP_INL_H_
#include "ordered_hash_map-inl.h"
#undef ORDERED_HASH_MAP_INL_H_
