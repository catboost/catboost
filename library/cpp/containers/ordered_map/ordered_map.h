#pragma once

#include <util/generic/vector.h>
#include <util/generic/hash.h>

#include <functional>
#include <utility>
#include <list>


namespace NOrderedMap {
    template <class TKey, class TValue, class BaseToUse = std::list<std::pair<const TKey, TValue>>> //bases: deque or list
    class TOrderedMap : protected BaseToUse {
        using TBase = BaseToUse;

    public:
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

        TOrderedMap() = default;

        TOrderedMap(const TOrderedMap& other) {
            for (const auto& item : other) {
                emplace_back(item.first, item.second);
            }
        }
        TOrderedMap(TOrderedMap&& other) noexcept : TBase(std::move(other)), Map_(std::move(other.Map_)) {
            other.Map_.clear();
        }
        TOrderedMap& operator=(const TOrderedMap& other) {
            if (this != &other) {
                clear();
                for (const auto& item : other) {
                    emplace_back(item.first, item.second);
                }
            }
            return *this;
        }
        TOrderedMap& operator=(TOrderedMap&& other) noexcept {
            if (this != &other) {
                TBase::operator=(std::move(other));
                Map_ = std::move(other.Map_);
                other.Map_.clear();
            }
            return *this;
        }

        template <class TheKey>
        TValue& operator[](const TheKey& key) {
            return try_emplace_back(key, TValue{}).second;
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
        reference emplace_back(TArgs&&... args) {
            auto& value = TBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(&value.first, --TBase::end());
            if (!ok) {
                iterator forDeleteFromList = mapIt->second;
                //TODO: it is possible to skip double-lookup if be possible to mutate key by iterator, or steal previous value by insert
                Map_.erase(mapIt);
                TBase::erase(forDeleteFromList);
                bool isOk = Map_.try_emplace(&value.first, --TBase::end()).second;
                Y_ASSERT(isOk);
            }
            return value;
        }

        template <class... TArgs>
        reference try_emplace_back(TArgs&&... args) {
            auto& value = TBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(&value.first, --TBase::end());
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
        const TValue& at(const TheKey& key) const {
            return Map_.at(&key)->second;
        }

        template <class TheKey>
        TValue& at(const TheKey& key) {
            return Map_.at(&key)->second;
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
            Map_.erase(&it->first);
            return TBase::erase(it);
        }

        void Sort() {
            TVector<std::pair<TKey, TValue>> tmp(Reserve(size()));
            for(auto& x : *this) {
                tmp.emplace_back(std::move(x));
            }
            clear();
            std::ranges::sort(tmp, {}, &std::pair<TKey, TValue>::first);
            for (auto&& x : std::move(tmp)) {
                emplace_back(std::move(x));
            }
        }

        template <class TheKey>
        const TValue* FindPtr(const TheKey& key) const {
            auto mapIt = Map_.find(&key);
            return mapIt == Map_.end()
                ? nullptr
                : &mapIt->second->second;
        }

        template <class TheKey>
        TValue* FindPtr(const TheKey& key) {
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
        struct THashOperationsForPtr : public THash<TKey>, public TEqualTo<TKey> {
            using THashBase = THash<TKey>;
            using TEqualBase = TEqualTo<TKey>;
            template<class TKeyTransparency>
            inline size_t operator()(const TKeyTransparency* ptr) const noexcept {
                Y_DEBUG_ABORT_UNLESS(ptr);
                return this->THashBase::operator()(*ptr);
            }
            template<class TKeyTransparencyA, class TKeyTransparencyB>
            inline bool operator()(const TKeyTransparencyA* a, const TKeyTransparencyB* b) const noexcept {
                Y_DEBUG_ABORT_UNLESS(a);
                Y_DEBUG_ABORT_UNLESS(b);
                return this->TEqualBase::operator()(*a, *b);
            }
        };
        THashMap<const TKey*, typename TBase::iterator, THashOperationsForPtr, THashOperationsForPtr> Map_{};
    };
}
