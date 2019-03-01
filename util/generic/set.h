#pragma once

#include "fwd.h"

#include <util/str_stl.h>
#include <util/memory/alloc.h>

#include <initializer_list>
#include <memory>
#include <set>

template <class K, class L, class A>
class TSet: public std::set<K, L, TReboundAllocator<A, K>> {
public:
    using TBase = std::set<K, L, TReboundAllocator<A, K>>;
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    template <class TheKey>
    inline bool contains(const TheKey& key) const {
        return this->find(key) != this->end();
    }
};

template <class K, class L, class A>
class TMultiSet: public std::multiset<K, L, TReboundAllocator<A, K>> {
public:
    using TBase = std::multiset<K, L, TReboundAllocator<A, K>>;
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    template <class TheKey>
    inline bool contains(const TheKey& key) const {
        return this->find(key) != this->end();
    }
};
