#pragma once

#include <util/generic/store_policy.h>

#include <limits>
#include <tuple>


namespace NPrivate {

    template <typename TContainer, typename TSize>
    struct TEnumerator {
    private:
        using TStorage = TAutoEmbedOrPtrPolicy<TContainer>;
        using TValue = std::tuple<const TSize, decltype(*std::begin(std::declval<TContainer&>()))>;
        using TIteratorState = decltype(std::begin(std::declval<TContainer&>()));
        using TSentinelState = decltype(std::end(std::declval<TContainer&>()));

        static constexpr bool TrivialSentinel = std::is_same_v<TIteratorState, TSentinelState>;

        struct TIterator;
        struct TSentinelCandidate {
            TSentinelState Iterator_;
        };
        using TSentinel = std::conditional_t<TrivialSentinel, TIterator, TSentinelCandidate>;

        struct TIterator {
            using difference_type = std::ptrdiff_t;
            using value_type = TValue;
            using pointer = void;
            using reference = value_type;
            using iterator_category = std::input_iterator_tag;

            reference operator*() const {
                return {Index_, *Iterator_};
            }

            TIterator& operator++() {
                ++Index_;
                ++Iterator_;
                return *this;
            }

            TIterator operator++(int) {
                TIterator result = *this;
                ++(*this);
                return result;
            }

            bool operator!=(const TSentinel& other) const {
                return Iterator_ != other.Iterator_;
            }

            bool operator==(const TSentinel& other) const {
                return Iterator_ == other.Iterator_;
            }

            TSize Index_;
            TIteratorState Iterator_;
        };

    public:
        using iterator = TIterator;
        using const_iterator = TIterator;
        using value_type = typename TIterator::value_type;
        using reference = typename TIterator::reference;
        using const_reference = typename TIterator::reference;

        TIterator begin() const {
            return {0, std::begin(*Storage_.Ptr())};
        }

        TSentinel end() const {
            if constexpr (TrivialSentinel) {
                return TIterator{std::numeric_limits<TSize>::max(), std::end(*Storage_.Ptr())};
            } else {
                return TSentinel{std::end(*Storage_.Ptr())};
            }
        }

        mutable TStorage Storage_;
    };

}

//! Usage: for (auto [i, x] : Enumerate(container)) {...}
template <typename TContainerOrRef>
auto Enumerate(TContainerOrRef&& container) {
    return NPrivate::TEnumerator<TContainerOrRef, std::size_t>{std::forward<TContainerOrRef>(container)};
}

//! Usage: for (auto [i, x] : SEnumerate(container)) {...}
// The index is signed for codebases that prefer signed numerics (such as YTsaurus).
template <typename TContainerOrRef>
auto SEnumerate(TContainerOrRef&& container) {
    return NPrivate::TEnumerator<TContainerOrRef, std::ptrdiff_t>{std::forward<TContainerOrRef>(container)};
}
