#pragma once

#include <util/generic/vector.h>
#include <util/generic/hash.h>

#include <functional>
#include <utility>


namespace NOrderedMap {
    template <class TKey, class TValue>
    class TOrderedMap : protected TVector<std::pair<const TKey, TValue>> {
        using TVectorBase = TVector<std::pair<const TKey, TValue>>;

    public:
        using typename TVectorBase::const_iterator;
        using typename TVectorBase::iterator;
        using typename TVectorBase::reference;
        using typename TVectorBase::value_type;
        using TVectorBase::size;
        using TVectorBase::empty;
        using TVectorBase::front;
        using TVectorBase::back;
        using TVectorBase::begin;
        using TVectorBase::end;
        using TVectorBase::operator bool;

        TOrderedMap() = default;

        TOrderedMap(const TOrderedMap& other) {
            for (const auto& item : other) {
                emplace_back(item.first, item.second);
            }
        }
        TOrderedMap(TOrderedMap&& other) noexcept : TVectorBase(std::move(other)), Map_(std::move(other.Map_)) {
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
                TVectorBase::operator=(std::move(other));
                Map_ = std::move(other.Map_);
                other.Map_.clear();
            }
            return *this;
        }

        template <class TheKey>
        TValue& operator[](const TheKey& key) {
            auto [mapIt, inserted] = Map_.try_emplace(key, TVectorBase::size());
            if (inserted) {
                TVectorBase::emplace_back(key, TValue{});
            }
            return TVectorBase::at(mapIt->second).second;
        }

        template <class TheKey>
        const TValue& at(const TheKey& key) const {
            return TVectorBase::at(Map_.at(key)).second;
        }

        template <class TheKey>
        TValue& at(const TheKey& key) {
            return TVectorBase::at(Map_.at(key)).second;
        }

        template <class TheKey>
        const_iterator find(const TheKey& key) const {
            auto mapIt = Map_.find(key);
            if (mapIt == Map_.end()) {
                return end();
            } else {
                return begin() + mapIt->second;
            }
        }

        template <class TheKey>
        iterator find(const TheKey& key) {
            auto mapIt = Map_.find(key);
            if (mapIt == Map_.end()) {
                return end();
            } else {
                return begin() + mapIt->second;
            }
        }

        template <class TheKey>
        bool contains(const TheKey& key) const {
            return Map_.contains(key);
        }

        void push_back(const value_type& x) {
            emplace_back(x);
        }

        void push_back(value_type&& x) {
            emplace_back(std::move(x));
        }

        template <class... TArgs>
        reference emplace_back(TArgs&&... args) {
            auto pos = size();
            auto& value = TVectorBase::emplace_back(std::forward<TArgs>(args)...);
            auto [mapIt, ok] = Map_.try_emplace(value.first, pos);
            if (ok) {
                return value;
            }
            TVectorBase::pop_back();
            return TVectorBase::at(mapIt->second);
        }

                template <class TheKey>
        size_t erase(const TheKey& key) {
            auto it = find(key);
            if (it == end()) {
                return 0;
            }
            erase(it);
            return 1;
        }
        void erase(iterator it) {
            if (it == end()) {
                return;
            }
            Map_.clear();
            TVector<std::pair<const TKey, TValue>> tmp;
            for (auto iter = begin(); iter != end(); ++iter) {
                if (iter != it) {
                    Map_[iter->first] = tmp.size();
                    tmp.emplace_back(std::move(*iter));
                }
            }
            TVectorBase::operator=(std::move(tmp));
        }

        void Sort() {
            TVector<std::pair<TKey, TValue>> tmp(begin(), end());
            TVectorBase::clear();
            std::ranges::sort(tmp, {}, &std::pair<TKey, TValue>::first);
            for (auto&& [k, v] : std::move(tmp)) {
                Map_[k] = size();
                TVectorBase::emplace_back(std::move(k), std::move(v));
            }
        }

        template <class TheKey>
        const TValue* FindPtr(const TheKey& key) const {
            auto mapIt = Map_.find(key);
            return mapIt != Map_.end()
                ? &TVectorBase::at(mapIt->second).second
                : nullptr;
        }

        template <class TheKey>
        TValue* FindPtr(const TheKey& key) {
            auto mapIt = Map_.find(key);
            return mapIt != Map_.end()
                ? &TVectorBase::at(mapIt->second).second
                : nullptr;
        }

        void clear() noexcept {
            TVectorBase::clear();
            Map_.clear();
        }

    private:
        THashMap<TKey, size_t> Map_{};
    };
}
