#pragma once

#include <util/generic/store_policy.h>

#include <algorithm>
#include <tuple>


namespace NPrivate {

    template <typename TContainer, typename TIteratorCategory = typename std::iterator_traits<decltype(std::begin(std::declval<TContainer>()))>::iterator_category>
    static constexpr bool HasRandomAccessIterator(int32_t) {
        return std::is_same_v<TIteratorCategory, std::random_access_iterator_tag>;
    }

    template <typename TContainer>
    static constexpr bool HasRandomAccessIterator(uint32_t) {
        return false;
    }

    template <typename... TContainers>
    struct TZipper {
        template <std::size_t... I>
        struct TZipperWithIndex {
        private:
            using THolders = std::tuple<TAutoEmbedOrPtrPolicy<TContainers>...>;
            using TValue = std::tuple<decltype(*std::begin(std::declval<TContainers&>()))...>;
            using TIteratorState = std::tuple<decltype(std::begin(std::declval<TContainers&>()))...>;
            using TSentinelState = std::tuple<decltype(std::end(std::declval<TContainers&>()))...>;

            static constexpr bool TrivialSentinel = std::is_same_v<TIteratorState, TSentinelState>;

            struct TIterator;
            struct TSentinelCandidate {
                TSentinelState Iterators_;
            };
            using TSentinel = std::conditional_t<TrivialSentinel, TIterator, TSentinelCandidate>;

#ifndef _MSC_VER
            // windows compiler crashes here
            static constexpr bool LimitByFirstContainer = TrivialSentinel &&
                ((HasRandomAccessIterator<TContainers>(0)) && ...);
#else
            static constexpr bool LimitByFirstContainer = false;
#endif

            struct TIterator {
                using difference_type = std::ptrdiff_t;
                using value_type = TValue;
                using pointer = TValue*;
                using reference = TValue&;
                using const_reference = const TValue&;
                using iterator_category = std::input_iterator_tag;

                TValue operator*() {
                    return {*std::get<I>(Iterators_)...};
                }
                TValue operator*() const {
                    return {*std::get<I>(Iterators_)...};
                }

                TIterator& operator++() {
                    (++std::get<I>(Iterators_), ...);
                    return *this;
                }

                TIterator operator++(int) {
                    return TIterator{TIteratorState{std::get<I>(Iterators_)++...}};
                }

                bool operator!=(const TSentinel& other) const {
                    if constexpr (LimitByFirstContainer) {
                        return std::get<0>(Iterators_) != std::get<0>(other.Iterators_);
                    } else {
                        // yes, for all correct iterators but end() it is a correct way to compare
                        return ((std::get<I>(Iterators_) != std::get<I>(other.Iterators_)) && ...);
                    }
                }
                bool operator==(const TSentinel& other) const {
                    return !(*this != other);
                }

                TIteratorState Iterators_;
            };
        public:
            using iterator = TIterator;
            using const_iterator = TIterator;
            using value_type = typename TIterator::value_type;
            using reference = typename TIterator::reference;
            using const_reference = typename TIterator::const_reference;

            TIterator begin() const {
                return {TIteratorState{std::begin(*std::get<I>(Holders_).Ptr())...}};
            }

            TSentinel end() const {
                if constexpr (LimitByFirstContainer) {
                    auto endOfFirst = std::begin(*std::get<0>(Holders_).Ptr()) + std::min({
                        std::end(*std::get<I>(Holders_).Ptr()) - std::begin(*std::get<I>(Holders_).Ptr())...});
                    TIterator iter{TSentinelState{std::end(*std::get<I>(Holders_).Ptr())...}};
                    std::get<0>(iter.Iterators_) = endOfFirst;
                    return iter;
                } else {
                    return {TSentinelState{std::end(*std::get<I>(Holders_).Ptr())...}};
                }
            }

            mutable THolders Holders_;
        };

        template <std::size_t... I>
        static auto Zip(TContainers&&... containers, std::index_sequence<I...>) {
            return TZipperWithIndex<I...>{{std::forward<TContainers>(containers)...}};
        }
    };

}


//! Acts as pythonic zip, BUT result length is equal to shortest length of input containers
//! Usage: for (auto [ai, bi, ci] : Zip(a, b, c)) {...}
template <typename... TContainers>
auto Zip(TContainers&&... containers) {
    return ::NPrivate::TZipper<TContainers...>::Zip(
        std::forward<TContainers>(containers)...,
        std::make_index_sequence<sizeof...(TContainers)>{}
    );
}
