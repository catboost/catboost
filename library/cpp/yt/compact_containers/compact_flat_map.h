#pragma once

#include "compact_vector.h"

namespace NYT {

///////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <typename T>
concept CHasIsTransparentFlag = requires {
    typename T::is_transparent;
};

template <typename T, typename U, typename TCompare>
concept CComparisonAllowed = std::same_as<T, U> || CHasIsTransparentFlag<TCompare>;

} // namespace NDetail

///////////////////////////////////////////////////////////////////////////////

//! A flat map implementation over TCompactVector that tries to keep data inline.
/*!
 *  Similarly to SmallSet, this is implemented via binary search over a sorted
 *  vector. Unlike SmallSet, however, this one never falls back to std::map (or
 *  set) for larger sizes. This means that the flat map is only useful
 *    - at small sizes, when there's absolutely no chance of it getting big, or
 *    - when it's filled once and is then only read from.
 *
 *  In return, the flat map provides
 *    - a smaller size overhead and
 *    - a guarantee that if data fits into inline storage, it goes there.
 *
 *  Because of the latter, one should be very careful with iterators: virtually
 *  any call to insert or erase may potentially invalidate all iterators.
 */
template <class TKey, class TValue, size_t N, class TKeyCompare = std::ranges::less>
class TCompactFlatMap
{
public:
    // NB: can't make this pair<const TKey, TValue> as TCompactTValueector requires its type
    // parameter to be copy-assignable.
    using value_type = std::pair<TKey, TValue>;
    using key_type = TKey;
    using mapped_type = TValue;
    using key_compare = TKeyCompare;

private:
    using TStorage = TCompactVector<value_type, N>;

public:
    using iterator = typename TStorage::iterator;
    using const_iterator = typename TStorage::const_iterator;
    using size_type = size_t;

    TCompactFlatMap() = default;

    template <class TInputIterator>
    TCompactFlatMap(TInputIterator begin, TInputIterator end);

    TCompactFlatMap(std::initializer_list<value_type> values);

    bool operator==(const TCompactFlatMap& rhs) const;
    bool operator!=(const TCompactFlatMap& rhs) const;

    iterator begin();
    const_iterator begin() const;
    const_iterator cbegin() const;

    iterator end();
    const_iterator end() const;
    const_iterator cend() const;

    void reserve(size_type n);

    size_type size() const;
    int ssize() const;

    [[nodiscard]] bool empty() const;
    void clear();

    void shrink_to_small();

    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    iterator find(const TOtherKey& k);
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    const_iterator find(const TOtherKey& k) const;

    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    iterator lower_bound(const TOtherKey& k);
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    const_iterator lower_bound(const TOtherKey& k) const;
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    iterator upper_bound(const TOtherKey& k);
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    const_iterator upper_bound(const TOtherKey& k) const;
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    std::pair<iterator, iterator> equal_range(const TOtherKey& k);
    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    std::pair<const_iterator, const_iterator> equal_range(const TOtherKey& k) const;

    template <NDetail::CComparisonAllowed<TKey, TKeyCompare> TOtherKey>
    bool contains(const TOtherKey& k) const;

    std::pair<iterator, bool> insert(const value_type& value);
    std::pair<iterator, bool> insert(value_type&& value);

    template <class TInputIterator>
    void insert(TInputIterator begin, TInputIterator end);

    template <class... TArgs>
    std::pair<iterator, bool> emplace(TArgs&&... args);

    TValue& operator[](const TKey& k);

    void erase(const TKey& k);
    void erase(iterator pos);
    void erase(iterator b, iterator e);

private:
    TStorage Storage_;

    template <class TArg>
    std::pair<iterator, bool> do_insert(TArg&& value);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPACT_FLAT_MAP_INL_H_
#include "compact_flat_map-inl.h"
#undef COMPACT_FLAT_MAP_INL_H_
