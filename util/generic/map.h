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
    using TSelf = TMap<K, V, Less, A>;
    using TAllocatorType = typename TBase::allocator_type;
    using TKeyCompare = typename TBase::key_compare;
    using TValueType = typename TBase::value_type;

public:
    inline TMap() {
    }

    template <typename TAllocParam>
    inline explicit TMap(TAllocParam* allocator)
        : TBase(Less(), allocator)
    {
    }

    template <class It>
    inline TMap(It f, It l)
        : TBase(f, l)
    {
    }

    inline TMap(std::initializer_list<TValueType> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline TMap(const TSelf& src)
        : TBase(src)
    {
    }

    inline TMap(TSelf&& src) noexcept
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

    inline bool has(const K& key) const {
        return this->find(key) != this->end();
    }
};

template <class K, class V, class Less, class A>
class TMultiMap: public std::multimap<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>> {
    using TBase = std::multimap<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>;
    using TSelf = TMultiMap<K, V, Less, A>;
    using TAllocatorType = typename TBase::allocator_type;
    using TKeyCompare = typename TBase::key_compare;
    using TValueType = typename TBase::value_type;

public:
    inline TMultiMap() {
    }

    template <typename TAllocParam>
    inline explicit TMultiMap(TAllocParam* allocator)
        : TBase(Less(), allocator)
    {
    }

    inline explicit TMultiMap(const Less& less, const TAllocatorType& alloc = TAllocatorType())
        : TBase(less, alloc)
    {
    }

    template <class It>
    inline TMultiMap(It f, It l)
        : TBase(f, l)
    {
    }

    inline TMultiMap(std::initializer_list<TValueType> il, const TKeyCompare& comp = TKeyCompare(), const TAllocatorType& alloc = TAllocatorType())
        : TBase(il, comp, alloc)
    {
    }

    inline TMultiMap(const TSelf& src)
        : TBase(src)
    {
    }

    inline TMultiMap(TSelf&& src) noexcept {
        this->swap(src);
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        // Self-move assignment is undefined behavior in the Standard.
        // This implementation ends up with zero-sized multimap.
        this->clear();
        this->swap(src);

        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline bool has(const K& key) const {
        return this->find(key) != this->end();
    }
};
