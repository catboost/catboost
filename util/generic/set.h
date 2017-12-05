#pragma once

#include "fwd.h"

#include <util/str_stl.h>
#include <util/memory/alloc.h>

#include <initializer_list>
#include <memory>
#include <set>

template <class K, class L, class A>
class TSet: public std::set<K, L, TReboundAllocator<A, K>> {
    using TBase = std::set<K, L, TReboundAllocator<A, K>>;
    using TSelf = TSet<K, L, A>;
    using TKeyCompare = typename TBase::key_compare;
    using TAllocatorType = typename TBase::allocator_type;

public:
    inline TSet() {
    }

    template <class It>
    inline TSet(It f, It l)
        : TBase(f, l)
    {
    }

    inline TSet(std::initializer_list<K> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline TSet(const TSelf& src)
        : TBase(src)
    {
    }

    inline TSet(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        TBase::operator=(std::forward<TSelf>(src));
        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    template <class TheKey>
    inline bool has(const TheKey& key) const {
        return this->find(key) != this->end();
    }
};

template <class K, class L, class A>
class TMultiSet: public std::multiset<K, L, TReboundAllocator<A, K>> {
    using TBase = std::multiset<K, L, TReboundAllocator<A, K>>;
    using TSelf = TMultiSet<K, L, A>;
    using TKeyCompare = typename TBase::key_compare;
    using TAllocatorType = typename TBase::allocator_type;

public:
    inline TMultiSet() {
    }

    template <class It>
    inline TMultiSet(It f, It l)
        : TBase(f, l)
    {
    }

    inline TMultiSet(std::initializer_list<K> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline TMultiSet(const TSelf& src)
        : TBase(src)
    {
    }

    inline TMultiSet(TSelf&& src) noexcept {
        this->swap(src);
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        // Self-move assignment is undefined behavior in the Standard.
        // This implementation ends up with zero-sized multiset.
        this->clear();
        this->swap(src);

        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
