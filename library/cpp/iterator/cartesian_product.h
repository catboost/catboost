#pragma once

#include <util/generic/store_policy.h>

#include <tuple>


namespace NPrivate {

    template <typename... TContainers>
    struct TCartesianMultiplier {
        template <std::size_t... I>
        struct TCartesianMultiplierWithIndex {
        private:
            using THolders = std::tuple<TAutoEmbedOrPtrPolicy<TContainers>...>;
            using TValue = std::tuple<decltype(*std::begin(std::declval<TContainers&>()))...>;
            using TIteratorState = std::tuple<int, decltype(std::begin(std::declval<TContainers&>()))...>;
            using TSentinelState = std::tuple<int, decltype(std::end(std::declval<TContainers&>()))...>;

            struct TIterator;
            struct TSentinelCandidate {
                TSentinelState Iterators_;
                THolders* HoldersPtr_;
            };
            using TSentinel = std::conditional_t<std::is_same_v<TIteratorState, TSentinelState>,
                                                 TIterator, TSentinelCandidate>;

            struct TIterator {
            private:
                //! Return value is true when iteration is not finished
                template <std::size_t position = sizeof...(TContainers)>
                void IncrementIteratorsTuple() {
                    auto& currentIterator = std::get<position>(Iterators_);
                    ++currentIterator;

                    if (currentIterator != std::end(*std::get<position - 1>(*HoldersPtr_).Ptr())) {
                        return;
                    } else {
                        currentIterator = std::begin(*std::get<position - 1>(*HoldersPtr_).Ptr());
                        if constexpr (position != 1) {
                            IncrementIteratorsTuple<position - 1>();
                        } else {
                            std::get<0>(Iterators_) = 1;
                        }
                    }
                }
            public:
                using difference_type = std::ptrdiff_t;
                using value_type = TValue;
                using pointer = TValue*;
                using reference = TValue&;
                using iterator_category = std::input_iterator_tag;

                TValue operator*() {
                    return {*std::get<I + 1>(Iterators_)...};
                }
                TValue operator*() const {
                    return {*std::get<I + 1>(Iterators_)...};
                }
                void operator++() {
                    IncrementIteratorsTuple();
                }
                bool operator!=(const TSentinel& other) const {
                    // not finished iterator VS sentinel (most frequent case)
                    if (std::get<0>(Iterators_) != std::get<0>(other.Iterators_)) {
                        return true;
                    }
                    // do not compare sentinels and finished iterators
                    if (std::get<0>(other.Iterators_)) {
                        return false;
                    }
                    // compare not finished iterators
                    return ((std::get<I + 1>(Iterators_) != std::get<I + 1>(other.Iterators_)) || ...);
                }
                bool operator==(const TSentinel& other) const {
                    return !(*this != other);
                }

                TIteratorState Iterators_;
                THolders* HoldersPtr_;
            };
        public:
            using iterator = TIterator;
            using const_iterator = TIterator;
            using value_type = typename TIterator::value_type;
            using reference = typename TIterator::reference;
            using const_reference = typename TIterator::reference;

            TIterator begin() const {
                bool isEmpty = !((std::begin(*std::get<I>(Holders_).Ptr()) != std::end(*std::get<I>(Holders_).Ptr())) && ...);
                return {TIteratorState{int(isEmpty), std::begin(*std::get<I>(Holders_).Ptr())...}, &Holders_};
            }

            TSentinel end() const {
                return {TSentinelState{1, std::end(*std::get<I>(Holders_).Ptr())...}, &Holders_};
            }

            mutable THolders Holders_;
        };

        template <std::size_t... I>
        static auto CartesianMultiply(TContainers&&... containers, std::index_sequence<I...>) {
            return TCartesianMultiplierWithIndex<I...>{{std::forward<TContainers>(containers)...}};
        }
    };

}

//! Usage: for (auto [ai, bi] : CartesianProduct(a, b)) {...}
//! Equivalent: for (auto& ai : a) { for (auto& bi : b) {...} }
template <typename... TContainers>
auto CartesianProduct(TContainers&&... containers) {
    return NPrivate::TCartesianMultiplier<TContainers...>::CartesianMultiply(
        std::forward<TContainers>(containers)..., std::make_index_sequence<sizeof...(TContainers)>{});
}
