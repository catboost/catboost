#pragma once

#include <util/generic/store_policy.h>

#include <iterator>
#include <tuple>


namespace NPrivate {

    template <typename TValue_, typename... TContainers>
    struct TConcatenator {
        template <std::size_t... I>
        struct TConcatenatorWithIndex {
        private:
            using THolders = std::tuple<TAutoEmbedOrPtrPolicy<TContainers>...>;
            using TValue = TValue_;
            using TIteratorState = std::tuple<decltype(std::begin(std::declval<TContainers&>()))...>;
            using TSentinelState = std::tuple<decltype(std::end(std::declval<TContainers&>()))...>;

            struct TIterator;
            struct TSentinelCandidate {
                TSentinelState Iterators_;
                std::size_t Position_;
                THolders* HoldersPtr_;
            };
            using TSentinel = std::conditional_t<std::is_same_v<TIteratorState, TSentinelState>,
                                                 TIterator, TSentinelCandidate>;

            struct TIterator {
            private:
                friend struct TConcatenatorWithIndex<I...>;

                // important, that it is a static function, compiler better optimizes such code
                template <std::size_t index = 0, typename TMaybeConstIteratorState>
                static TValue GetCurrentValue(std::size_t position, TMaybeConstIteratorState& iterators) {
                    if constexpr (index >= sizeof...(TContainers)) {
                        // never happened when use of iterator is correct
                        return *std::get<0>(iterators);
                    } else {
                        if (position == index) {
                            return *std::get<index>(iterators);
                        } else {
                            return GetCurrentValue<index + 1>(position, iterators);
                        }
                    }
                }

                template <bool needIncrement, std::size_t index = 0>
                void MaybeIncrementIteratorAndSkipExhaustedContainers() {
                    if constexpr (index >= sizeof...(TContainers)) {
                        return;
                    } else {
                        if (Position_ == index) {
                            if constexpr (needIncrement) {
                                ++std::get<index>(Iterators_);
                            }
                            if (!(std::get<index>(Iterators_) != std::end(*std::get<index>(*HoldersPtr_).Ptr()))) {
                                ++Position_;
                                MaybeIncrementIteratorAndSkipExhaustedContainers<false, index + 1>();
                            }
                        } else {
                            MaybeIncrementIteratorAndSkipExhaustedContainers<needIncrement, index + 1>();
                        }
                    }
                }
            public:
                using difference_type = std::ptrdiff_t;
                using value_type = TValue;
                using pointer = std::remove_reference_t<TValue>*;
                using reference = std::remove_reference_t<TValue>&;
                using iterator_category = std::input_iterator_tag;

                TValue operator*() {
                    return GetCurrentValue(Position_, Iterators_);
                }
                TValue operator*() const {
                    return GetCurrentValue(Position_, Iterators_);
                }
                TIterator& operator++() {
                    MaybeIncrementIteratorAndSkipExhaustedContainers<true>();
                    return *this;
                }
                bool operator!=(const TSentinel& other) const {
                    // give compiler an opportunity to optimize sentinel case (-70% of time)
                    if (other.Position_ == sizeof...(TContainers)) {
                        return Position_ < sizeof...(TContainers);
                    } else {
                        return (Position_ != other.Position_ ||
                                ((std::get<I>(Iterators_) != std::get<I>(other.Iterators_)) || ...));
                    }
                }
                bool operator==(const TSentinel& other) const {
                    return !(*this != other);
                }

                TIteratorState Iterators_;
                std::size_t Position_;
                THolders* HoldersPtr_;
            };
        public:
            using iterator = TIterator;
            using const_iterator = TIterator;
            using value_type = typename TIterator::value_type;
            using reference = typename TIterator::reference;
            using const_reference = typename TIterator::reference;

            TIterator begin() const {
                TIterator iterator{TIteratorState{std::begin(*std::get<I>(Holders_).Ptr())...}, 0, &Holders_};
                iterator.template MaybeIncrementIteratorAndSkipExhaustedContainers<false>();
                return iterator;
            }

            TSentinel end() const {
                return {TSentinelState{std::end(*std::get<I>(Holders_).Ptr())...}, sizeof...(TContainers), &Holders_};
            }

            mutable THolders Holders_;
        };

        template <std::size_t... I>
        static auto Concatenate(TContainers&&... containers, std::index_sequence<I...>) {
            return TConcatenatorWithIndex<I...>{{std::forward<TContainers>(containers)...}};
        }
    };

}


//! Usage: for (auto x : Concatenate(a, b)) {...}
template <typename TFirstContainer, typename... TContainers>
auto Concatenate(TFirstContainer&& container, TContainers&&... containers) {
    return NPrivate::TConcatenator<decltype(*std::begin(container)), TFirstContainer, TContainers...>::Concatenate(
        std::forward<TFirstContainer>(container), std::forward<TContainers>(containers)...,
        std::make_index_sequence<sizeof...(TContainers) + 1>{});
}
