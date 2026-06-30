#pragma once

#include "table.h"
#include "concepts/iterator.h"

#include <util/generic/algorithm.h>

namespace NFlatHash {

namespace NPrivate {

struct TSimpleKeyGetter {
    template <class T>
    static constexpr auto& Apply(T& t) noexcept { return t; }

    template <class T>
    static constexpr const auto& Apply(const T& t) noexcept { return t; }
};

}  // namespace NPrivate

template <class Key,
          class Hash,
          class KeyEqual,
          class Container,
          class Probing,
          class SizeFitter,
          class Expander>
class TSet : private TTable<Hash,
                            KeyEqual,
                            Container,
                            NPrivate::TSimpleKeyGetter,
                            Probing,
                            SizeFitter,
                            Expander,
                            std::add_const>
{
private:
    using TBase = TTable<Hash,
                         KeyEqual,
                         Container,
                         NPrivate::TSimpleKeyGetter,
                         Probing,
                         SizeFitter,
                         Expander,
                         std::add_const>;

    static_assert(std::is_same_v<Key, typename Container::value_type>);

public:
    using key_type = Key;
    using typename TBase::value_type;
    using typename TBase::size_type;
    using typename TBase::difference_type;
    using typename TBase::hasher;
    using typename TBase::key_equal;
    using typename TBase::reference;
    using typename TBase::const_reference;
    using typename TBase::iterator;
    using typename TBase::const_iterator;
    using typename TBase::allocator_type;
    using typename TBase::pointer;
    using typename TBase::const_pointer;

private:
    static constexpr size_type INIT_SIZE = 8;

public:
    TSet() : TBase(INIT_SIZE) {}

    template <class... Rest>
    TSet(size_type initSize, Rest&&... rest) : TBase(initSize, std::forward<Rest>(rest)...) {}

    template <class I, class... Rest>
    TSet(I first, I last,
         std::enable_if_t<NConcepts::IteratorV<I>, size_type> initSize = INIT_SIZE,
         Rest&&... rest)
        : TBase(initSize, std::forward<Rest>(rest)...)
    {
        insert(first, last);
    }

    template <class... Rest>
    TSet(std::initializer_list<value_type> il, size_type initSize = INIT_SIZE, Rest&&... rest)
        : TBase(initSize, std::forward<Rest>(rest)...)
    {
        insert(il.begin(), il.end());
    }

    TSet(std::initializer_list<value_type> il, size_type initSize = INIT_SIZE)
        : TBase(initSize)
    {
        insert(il.begin(), il.end());
    }

    TSet(const TSet&) = default;
    TSet(TSet&&) = default;

    TSet& operator=(const TSet&) = default;
    TSet& operator=(TSet&&) = default;

    // Iterators
    using TBase::begin;
    using TBase::cbegin;
    using TBase::end;
    using TBase::cend;

    // Capacity
    using TBase::empty;
    using TBase::size;

    // Modifiers
    using TBase::clear;
    using TBase::insert;
    using TBase::emplace;
    using TBase::emplace_hint;
    using TBase::erase;
    using TBase::swap;

    // Lookup
    using TBase::count;
    using TBase::find;
    using TBase::contains;

    // Bucket interface
    using TBase::bucket_count;
    using TBase::bucket_size;

    // Hash policy
    using TBase::load_factor;
    using TBase::rehash;
    using TBase::reserve;

    // Observers
    using TBase::hash_function;
    using TBase::key_eq;

    friend bool operator==(const TSet& lhs, const TSet& rhs) {
        return lhs.size() == rhs.size() && AllOf(lhs, [&rhs](const auto& v) {
            return rhs.contains(v);
        });
    }

    friend bool operator!=(const TSet& lhs, const TSet& rhs) { return !(lhs == rhs); }
};

}  // namespace NFlatHash
