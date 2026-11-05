#pragma once

#include "compact_vector.h"

namespace NYT {

///////////////////////////////////////////////////////////////////////////////

//! A flat set implementation over TCompactVector that tries to keep data inline.
/*!
 *  Similarly to TCompactSet, this is implemented via binary search over a sorted
 *  vector. Unlike TCompactSet, however, this one never falls back to std::map (or
 *  set) for larger sizes. This means that the flat set is only useful
 *    - at small sizes, when there's absolutely no chance of it getting big, or
 *    - when it's filled once and is then only read from.
 *
 *  In return, the flat set provides
 *    - a smaller size overhead and
 *    - a guarantee that if data fits into inline storage, it goes there.
 *
 *  Because of the latter, one should be very careful with iterators: virtually
 *  any call to insert or erase may potentially invalidate all iterators.
 */
template <class TValue, size_t N>
class TCompactFlatSet
{
private:
    using TStorage = TCompactVector<TValue, N>;

public:
    using iterator = typename TStorage::iterator;
    using const_iterator = typename TStorage::const_iterator;

    using const_reverse_iterator = typename TStorage::const_reverse_iterator;
    using reverse_iterator = typename TStorage::reverse_iterator;

    using size_type = std::size_t;
    using key_type = TValue;
    using value_type = TValue;

    TCompactFlatSet() = default;

    template <class TInputIterator>
    TCompactFlatSet(TInputIterator begin, TInputIterator end);

    TCompactFlatSet(std::initializer_list<TValue> values);

    bool operator==(const TCompactFlatSet& rhs) const = default;

    iterator begin();
    const_iterator begin() const;
    const_iterator cbegin() const;

    iterator end();
    const_iterator end() const;
    const_iterator cend() const;

    reverse_iterator rbegin();
    const_reverse_iterator rbegin() const;

    reverse_iterator rend();
    const_reverse_iterator rend() const;

    void reserve(size_type size);

    size_type size() const;
    int ssize() const;

    [[nodiscard]] bool empty() const;
    void clear();

    void shrink_to_small();

    iterator find(const TValue& value);
    const_iterator find(const TValue& value) const;

    size_type count(const TValue& value) const;

    iterator lower_bound(const TValue& value);
    const_iterator lower_bound(const TValue& value) const;
    iterator upper_bound(const TValue& value);
    const_iterator upper_bound(const TValue& value) const;
    std::pair<iterator, iterator> equal_range(const TValue& value);
    std::pair<const_iterator, const_iterator> equal_range(const TValue& value) const;

    bool contains(const TValue& value) const;

    std::pair<iterator, bool> insert(const TValue& value);
    std::pair<iterator, bool> insert(TValue&& value);

    template <class TInputIterator>
    void insert(TInputIterator begin, TInputIterator end);

    bool erase(const TValue& value);
    void erase(iterator pos);
    void erase(iterator begin, iterator end);

private:
    TStorage Storage_;

    template <class TArg>
    std::pair<iterator, bool> DoInsert(TArg&& value);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define COMPACT_FLAT_SET_INL_H_
#include "compact_flat_set-inl.h"
#undef COMPACT_FLAT_SET_INL_H_
