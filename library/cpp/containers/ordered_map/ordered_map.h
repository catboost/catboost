#pragma once

#include <util/generic/vector.h>
#include <util/generic/hash.h>

#include <functional>
#include <utility>
#include <list>


namespace NOrderedMap {

    template<class Key, class Value>
    struct TPairTraits {
        using TValue = Value;
        using Item = std::pair<const Key, Value>;
        using StorageItem = Item;
        using TMutableItem = std::pair<Key, Value>;

        static const Key& GetSortProjector(const Item& x) {
            return x.first;
        }

        static auto& GetValueRef(auto& x) {
            return x.second;
        }
    };


    template<class Key>
    struct TSelfTraits {
        using Item = const Key;
        using StorageItem = Key;
        using TMutableItem = Key;
        using TValue = Item;

        static const Key& GetSortProjector(const Item& x) {
            return x;
        }

        static Item& GetValueRef(Item& x) {
            return x;
        }
    };

    template <
        class TKey,
        class Traits,
        class BaseToUse = std::list<typename Traits::StorageItem> //bases: deque or list
    >
    class TOrderedMapImpl : protected BaseToUse {
        using TBase = BaseToUse;

    public:
        using TTraits = Traits;
        using typename TBase::const_iterator;
        using typename TBase::iterator;
        using typename TBase::reference;
        using typename TBase::value_type;
        using TBase::size;
        using TBase::empty;
        using TBase::front;
        using TBase::back;
        using TBase::begin;
        using TBase::end;
        using TBase::rbegin;
        using TBase::rend;

        operator bool() const {
            return empty();
        }

        TOrderedMapImpl() = default;

        TOrderedMapImpl(const TOrderedMapImpl& other) {
            for (const auto& item : other) {
                emplace_back(item);
            }
        }
        TOrderedMapImpl(TOrderedMapImpl&& other) noexcept : TBase(std::move(other)), Map_(std::move(other.Map_)) {
            other.Map_.clear();
        }
        TOrderedMapImpl& operator=(const TOrderedMapImpl& other) {
            if (this != &other) {
                clear();
                for (const auto& item : other) {
                    emplace_back(item);
                }
            }
            return *this;
        }
        TOrderedMapImpl& operator=(TOrderedMapImpl&& other) noexcept {
            if (this != &other) {
                TBase::operator=(std::move(other));
                Map_ = std::move(other.Map_);
                other.Map_.clear();
            }
            return *this;
        }

        template <class TheKey>
        typename Traits::TValue& operator[](const TheKey& key) {
            return Traits::GetValueRef(try_emplace_back(key, typename Traits::TValue{}));
        }

        template <class TheKey>
        const_iterator find(const TheKey& key) const {
            auto mapIt = Map_.find(&key);
            if (mapIt == Map_.end()) {
                return end();
            } else {
                return mapIt->second;
            }
        }

        template <class TheKey>
        iterator find(const TheKey& key) {
            auto mapIt = Map_.find(&key);
            if (mapIt == Map_.end()) {
                return end();
            } else {
                return mapIt->second;
            }
        }

        template <class TheKey>
        bool contains(const TheKey& key) const {
            return Map_.contains(&key);
        }

        void push_back(const value_type& x) {
            emplace_back(x);
        }

        void push_back(value_type&& x) {
            emplace_back(std::move(x));
        }

        template <class... TArgs>
        std::pair<iterator, bool> insert(TArgs&&... args) {
            auto& value = TBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(&Traits::GetSortProjector(value), --TBase::end());
            if (!ok) {
                TBase::pop_back();
            }
            return {mapIt->second, ok};
        }

        template <class... TArgs>
        reference emplace_back(TArgs&&... args) {
            auto& value = TBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(&Traits::GetSortProjector(value), --TBase::end());
            if (!ok) {
                iterator forDeleteFromList = mapIt->second;
                mapIt->first.Data = &Traits::GetSortProjector(value);
                mapIt->second = --TBase::end();
                TBase::erase(forDeleteFromList);
            }
            return value;
        }

        template <class... TArgs>
        reference emplace(TArgs&&... args) {
            return emplace_back(std::forward<TArgs>(args)...);
        }

        template <class... TArgs>
        reference try_emplace_back(TArgs&&... args) {
            auto& value = TBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(&Traits::GetSortProjector(value), --TBase::end());
            if (!ok) {
                TBase::pop_back();
                return *mapIt->second;
            }
            return value;
        }

        reference at(size_t i) {
            Y_ENSURE(i < size());
            using category = typename std::iterator_traits<iterator>::iterator_category;
            if constexpr (std::is_same_v<category, std::random_access_iterator_tag>) {
                return (TBase::begin() + i)->second;
            } else {
                auto iter = TBase::begin();
                for(size_t j = 0; j < i && iter != TBase::end(); ++j) {
                    ++iter;
                }
                return iter->second;
            }
        }

        template <class TheKey>
        const typename Traits::TValue& at(const TheKey& key) const {
            return Traits::GetValueRef(*Map_.at(&key));
        }

        template <class TheKey>
        typename Traits::TValue& at(const TheKey& key) {
            return Traits::GetValueRef(*Map_.at(&key));
        }

        template <class TheKey>
        auto erase(const TheKey& key) {
            auto mapIt = Map_.find(&key);
            if (mapIt == Map_.end()) {
                return end();
            }
            auto forDelete = mapIt->second;
            Map_.erase(mapIt);
            auto res = TBase::erase(forDelete);
            return res;
        }

        iterator erase(iterator it) {
            if (it == end()) {
                return end();
            }
            Map_.erase(&Traits::GetSortProjector(*it));
            return TBase::erase(it);
        }

        void Sort() {
            TVector<typename Traits::TMutableItem> tmp(Reserve(size()));
            for(auto& x : *this) {
                tmp.emplace_back(std::move(x));
            }
            clear();
            SortBy(tmp, Traits::GetSortProjector);
            for (auto&& x : std::move(tmp)) {
                emplace_back(std::move(x));
            }
        }

        template <class TheKey>
        const typename Traits::TValue* FindPtr(const TheKey& key) const {
            auto mapIt = Map_.find(&key);
            return mapIt == Map_.end()
                ? nullptr
                : &mapIt->second->second;
        }

        template <class TheKey>
        typename Traits::TValue* FindPtr(const TheKey& key) {
            auto mapIt = Map_.find(&key);
            return mapIt == Map_.end()
                ? nullptr
                : &mapIt->second->second;
        }

        void clear() noexcept {
            TBase::clear();
            Map_.clear();
        }

    private:
        struct TPtrBackdoorKey {
            mutable const TKey* Data;
            TPtrBackdoorKey(const TKey* x) : Data(x) {}
            TPtrBackdoorKey(TPtrBackdoorKey&&) = default;
            TPtrBackdoorKey(const TPtrBackdoorKey&) = default;

            const TKey& operator*() const {
                return *Data;
            }
            operator bool() const {
                return !!Data;
            }
        };
        struct THashOperationsForPtr : public THash<TKey>, public TEqualTo<TKey> {
            using THashBase = THash<TKey>;
            using TEqualBase = TEqualTo<TKey>;
            template<class TKeyTransparency>
            inline size_t operator()(const TKeyTransparency& ptr) const noexcept {
                Y_DEBUG_ABORT_UNLESS(ptr);
                return this->THashBase::operator()(*ptr);
            }
            template<class TKeyTransparencyA, class TKeyTransparencyB>
            inline bool operator()(const TKeyTransparencyA& a, const TKeyTransparencyB& b) const noexcept {
                Y_DEBUG_ABORT_UNLESS(a);
                Y_DEBUG_ABORT_UNLESS(b);
                return this->TEqualBase::operator()(*a, *b);
            }
        };
        THashMap<TPtrBackdoorKey, typename TBase::iterator, THashOperationsForPtr, THashOperationsForPtr> Map_{};
    };

    template <class TKey, class TValue>
    class TOrderedMap : public TOrderedMapImpl<TKey, TPairTraits<TKey, TValue>> {
    };

    template <class TKey>
    class TOrderedSet : public TOrderedMapImpl<TKey, TSelfTraits<TKey>> {
    public:
        auto operator==(const TOrderedSet& b) const {
            return std::lexicographical_compare_three_way(this->begin(), this->end(), b.begin(), b.end()) == std::strong_ordering::equal;
        }
    };
}
