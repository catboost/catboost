#pragma once

#include "table.h"
#include "concepts/iterator.h"

#include <util/generic/algorithm.h>
#include <util/generic/mapfindptr.h>

namespace NFlatHash {

namespace NPrivate {

struct TMapKeyGetter {
    template <class T>
    static constexpr auto& Apply(T& t) noexcept { return t.first; }

    template <class T>
    static constexpr const auto& Apply(const T& t) noexcept { return t.first; }
};

}  // namespace NPrivate

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          class Container,
          class Probing,
          class SizeFitter,
          class Expander>
class TMap : private TTable<Hash,
                            KeyEqual,
                            Container,
                            NPrivate::TMapKeyGetter,
                            Probing,
                            SizeFitter,
                            Expander>,
             public TMapOps<TMap<Key,
                                 T,
                                 Hash,
                                 KeyEqual,
                                 Container,
                                 Probing,
                                 SizeFitter,
                                 Expander>>
{
private:
    using TBase = TTable<Hash,
                         KeyEqual,
                         Container,
                         NPrivate::TMapKeyGetter,
                         Probing,
                         SizeFitter,
                         Expander>;

    static_assert(std::is_same<std::pair<const Key, T>, typename Container::value_type>::value);

public:
    using key_type = Key;
    using mapped_type = T;
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
    TMap() : TBase(INIT_SIZE) {}

    template <class... Rest>
    TMap(size_type initSize, Rest&&... rest) : TBase(initSize, std::forward<Rest>(rest)...) {}

    template <class I, class... Rest>
    TMap(I first, I last,
         std::enable_if_t<NConcepts::IteratorV<I>, size_type> initSize = INIT_SIZE,
         Rest&&... rest)
        : TBase(initSize, std::forward<Rest>(rest)...)
    {
        insert(first, last);
    }

    template <class... Rest>
    TMap(std::initializer_list<value_type> il, size_type initSize = INIT_SIZE, Rest&&... rest)
        : TBase(initSize, std::forward<Rest>(rest)...)
    {
        insert(il.begin(), il.end());
    }

    TMap(std::initializer_list<value_type> il, size_type initSize = INIT_SIZE)
        : TBase(initSize)
    {
        insert(il.begin(), il.end());
    }

    TMap(const TMap&) = default;
    TMap(TMap&&) = default;

    TMap& operator=(const TMap&) = default;
    TMap& operator=(TMap&&) = default;

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

    template <class V>
    std::pair<iterator, bool> insert_or_assign(const key_type& k, V&& v) {
        return InsertOrAssignImpl(k, std::forward<V>(v));
    }
    template <class V>
    std::pair<iterator, bool> insert_or_assign(key_type&& k, V&& v) {
        return InsertOrAssignImpl(std::move(k), std::forward<V>(v));
    }

    template <class V>
    iterator insert_or_assign(const_iterator, const key_type& k, V&& v) { // TODO(tender-bum)
        return insert_or_assign(k, std::forward<V>(v)).first;
    }
    template <class V>
    iterator insert_or_assign(const_iterator, key_type&& k, V&& v) { // TODO(tender-bum)
        return insert_or_assign(std::move(k), std::forward<V>(v)).first;
    }

    template <class... Args>
    std::pair<iterator, bool> try_emplace(const key_type& key, Args&&... args) {
        return TryEmplaceImpl(key, std::forward<Args>(args)...);
    }
    template <class... Args>
    std::pair<iterator, bool> try_emplace(key_type&& key, Args&&... args) {
        return TryEmplaceImpl(std::move(key), std::forward<Args>(args)...);
    }

    template <class... Args>
    iterator try_emplace(const_iterator, const key_type& key, Args&&... args) { // TODO(tender-bum)
        return try_emplace(key, std::forward<Args>(args)...).first;
    }
    template <class... Args>
    iterator try_emplace(const_iterator, key_type&& key, Args&&... args) { // TODO(tender-bum)
        return try_emplace(std::move(key), std::forward<Args>(args)...).first;
    }

    // Lookup
    using TBase::count;
    using TBase::find;
    using TBase::contains;

    template <class K>
    mapped_type& at(const K& key) {
        auto it = find(key);
        if (it == end()) {
            throw std::out_of_range{ "no such key in map" };
        }
        return it->second;
    }

    template <class K>
    const mapped_type& at(const K& key) const { return const_cast<TMap*>(this)->at(key); }

    template <class K>
    Y_FORCE_INLINE mapped_type& operator[](K&& key) {
        return TBase::TryCreate(key, [&](size_type idx) {
            TBase::Buckets_.InitNode(idx, std::forward<K>(key), mapped_type{});
        }).first->second;
    }

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

    friend bool operator==(const TMap& lhs, const TMap& rhs) {
        return lhs.size() == rhs.size() && AllOf(lhs, [&rhs](const auto& v) {
            auto it = rhs.find(v.first);
            return it != rhs.end() && *it == v;
        });
    }

    friend bool operator!=(const TMap& lhs, const TMap& rhs) { return !(lhs == rhs); }

private:
    template <class K, class... Args>
    std::pair<iterator, bool> TryEmplaceImpl(K&& key, Args&&... args) {
        return TBase::TryCreate(key, [&](size_type idx) {
            TBase::Buckets_.InitNode(
                idx,
                std::piecewise_construct,
                std::forward_as_tuple(std::forward<K>(key)),
                std::forward_as_tuple(std::forward<Args>(args)...));
        });
    }

    template <class K, class V>
    std::pair<iterator, bool> InsertOrAssignImpl(K&& key, V&& v) {
        auto p = try_emplace(std::forward<K>(key), std::forward<V>(v));
        if (!p.second) {
            p.first->second = std::forward<V>(v);
        }
        return p;
    }
};

}  // namespace NFlatHash
