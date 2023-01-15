#pragma once

#include "fwd.h"
#include "mapfindptr.h"

#include <util/str_stl.h>
#include <util/memory/alloc.h>

#include <utility>
#include <initializer_list>
#include <map>
#include <memory>

template <class K, class V, class Less, class A>
class TMap: public std::map<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>, public TMapOps<TMap<K, V, Less, A>> {
    using TBase = std::map<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline bool contains(const K& key) const {
        return this->find(key) != this->end();
    }
};

template <class K, class V, class Less, class A>
class TMultiMap: public std::multimap<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>> {
    using TBase = std::multimap<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline bool contains(const K& key) const {
        return this->find(key) != this->end();
    }
};
