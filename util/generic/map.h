#pragma once

#include "fwd.h"
#include "mapfindptr.h"

#include <util/str_stl.h>
#include <util/memory/alloc.h>

#include <utility>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace NMapPrivate {
    template <class T, class = void>
    struct TIsTransparent: std::false_type {
    };

    template <class T>
    struct TIsTransparent<T, std::void_t<typename T::is_transparent>>: std::true_type {
    };
} // namespace NMapPrivate

template <class K, class V, class Less, class A>
class TMap: public std::map<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>, public TMapOps<TMap<K, V, Less, A>> {
    using TBase = std::map<K, V, Less, TReboundAllocator<A, std::pair<const K, V>>>;

    template <class TKey>
    using TEnableHetero = std::enable_if_t<
        NMapPrivate::TIsTransparent<Less>::value &&
            !std::is_same_v<std::decay_t<TKey>, K> &&
            !std::is_integral_v<std::decay_t<TKey>> &&
            !std::is_enum_v<std::decay_t<TKey>> &&
            std::is_invocable_v<const Less&, const K&, const TKey&> &&
            std::is_invocable_v<const Less&, const TKey&, const K&>,
        int>;

public:
    using TBase::TBase;

    using TBase::at;
    using TBase::erase;
    using TBase::insert_or_assign;
    using TBase::operator[];
    using TBase::try_emplace;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    template <class TKey, TEnableHetero<TKey> = 0>
    V& at(const TKey& key) {
        auto it = this->find(key);
        if (it == this->end()) {
            throw std::out_of_range("TMap::at");
        }
        return it->second;
    }

    template <class TKey, TEnableHetero<TKey> = 0>
    const V& at(const TKey& key) const {
        auto it = this->find(key);
        if (it == this->end()) {
            throw std::out_of_range("TMap::at");
        }
        return it->second;
    }

    template <class TKey, TEnableHetero<TKey> = 0>
    V& operator[](TKey&& key) {
        return this->try_emplace(std::forward<TKey>(key)).first->second;
    }

    template <class TKey, class... TArgs, TEnableHetero<TKey> = 0>
    std::pair<typename TBase::iterator, bool> try_emplace(TKey&& key, TArgs&&... args) {
        auto it = this->find(key);
        if (it != this->end()) {
            return {std::move(it), false};
        }
        return {this->emplace(std::piecewise_construct,
                              std::forward_as_tuple(std::forward<TKey>(key)),
                              std::forward_as_tuple(std::forward<TArgs>(args)...))
                    .first,
                true};
    }

    template <class TKey, class M, TEnableHetero<TKey> = 0>
    std::pair<typename TBase::iterator, bool> insert_or_assign(TKey&& key, M&& obj) {
        auto it = this->find(key);
        if (it != this->end()) {
            it->second = std::forward<M>(obj);
            return {std::move(it), false};
        }
        return {this->emplace(std::piecewise_construct,
                              std::forward_as_tuple(std::forward<TKey>(key)),
                              std::forward_as_tuple(std::forward<M>(obj)))
                    .first,
                true};
    }

    template <
        class TKey,
        TEnableHetero<TKey> = 0,
        typename std::enable_if_t<
            !std::is_convertible_v<TKey, typename TBase::iterator> &&
                !std::is_convertible_v<TKey, typename TBase::const_iterator>,
            int> = 0>
    typename TBase::size_type erase(TKey&& key) {
        auto it = this->find(key);
        if (it == this->end()) {
            return 0;
        }
        this->erase(it);
        return 1;
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
};
