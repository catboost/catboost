#pragma once

#include "fwd.h"

#include <util/str_stl.h>
#include <util/memory/alloc.h>

#include <initializer_list>
#include <memory>
#include <set>

template <class K, class L, class A>
class yset: public std::set<K, L, TReboundAllocator<A, K>> {
    using TBase = std::set<K, L, TReboundAllocator<A, K>>;
    using TSelf = yset<K, L, A>;
    using TKeyCompare = typename TBase::key_compare;
    using TAllocatorType = typename TBase::allocator_type;

public:
    inline yset() {
    }

    template <class It>
    inline yset(It f, It l)
        : TBase(f, l)
    {
    }

    inline yset(std::initializer_list<K> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline yset(const TSelf& src)
        : TBase(src)
    {
    }

    inline yset(TSelf&& src) noexcept
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
class ymultiset: public std::multiset<K, L, TReboundAllocator<A, K>> {
    using TBase = std::multiset<K, L, TReboundAllocator<A, K>>;
    using TSelf = ymultiset<K, L, A>;
    using TKeyCompare = typename TBase::key_compare;
    using TAllocatorType = typename TBase::allocator_type;

public:
    inline ymultiset() {
    }

    template <class It>
    inline ymultiset(It f, It l)
        : TBase(f, l)
    {
    }

    inline ymultiset(std::initializer_list<K> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline ymultiset(const TSelf& src)
        : TBase(src)
    {
    }

    inline ymultiset(TSelf&& src) noexcept {
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
