#pragma once

#include "store_policy.h"
#include "typetraits.h"

namespace NPrivate {

    template <typename TDerived, bool HasReverseIterators = true>
    struct TForwardFromBackwardIteratorsHelper {
        auto begin() const {
            return static_cast<const TDerived*>(this)->Container.Ptr()->rbegin();
        }

        auto end() const {
            return static_cast<const TDerived*>(this)->Container.Ptr()->rend();
        }

        auto begin() {
            return static_cast<TDerived*>(this)->Container.Ptr()->rbegin();
        }

        auto end() {
            return static_cast<TDerived*>(this)->Container.Ptr()->rend();
        }
    };

    template <typename TDerived>
    struct TForwardFromBackwardIteratorsHelper<TDerived, false> {
        auto begin() const {
            using std::end;
            return std::make_reverse_iterator(end(*static_cast<const TDerived*>(this)->Container.Ptr()));
        }

        auto end() const {
            using std::begin;
            return std::make_reverse_iterator(begin(*static_cast<const TDerived*>(this)->Container.Ptr()));
        }

        auto begin() {
            using std::end;
            return std::make_reverse_iterator(end(*static_cast<TDerived*>(this)->Container.Ptr()));
        }

        auto end() {
            using std::begin;
            return std::make_reverse_iterator(begin(*static_cast<TDerived*>(this)->Container.Ptr()));
        }
    };

    template <typename TContainerRefOrObject>
    constexpr bool HasReverseIterators(i32 priorityArgument, decltype(std::declval<TContainerRefOrObject>().rbegin())* unused) {
        Y_UNUSED(priorityArgument);
        Y_UNUSED(unused);
        return true;
    }

    template <typename TContainerRefOrObject>
    constexpr bool HasReverseIterators(char priorityArgument, std::nullptr_t* unused) {
        Y_UNUSED(priorityArgument);
        Y_UNUSED(unused);
        return false;
    }

    template <typename TContainerRefOrObject>
    struct TReverseImpl : TForwardFromBackwardIteratorsHelper<TReverseImpl<TContainerRefOrObject>,
                                                              HasReverseIterators<TContainerRefOrObject>((i32)0, nullptr)> {
        using TContainerHolder = TAutoEmbedOrPtrPolicy<TContainerRefOrObject>;
        TContainerHolder Container;

        TReverseImpl(TReverseImpl&&) = default;
        TReverseImpl(const TReverseImpl&) = default;

        TReverseImpl(typename TContainerHolder::TObject& container)
            : Container(container)
        {
        }

        auto rbegin() const {
            using std::begin;
            return begin(*Container.Ptr());
        }

        auto rend() const {
            using std::end;
            return end(*Container.Ptr());
        }

        auto rbegin() {
            using std::begin;
            return begin(*Container.Ptr());
        }

        auto rend() {
            using std::end;
            return end(*Container.Ptr());
        }
    };
}

/**
 * Provides a reverse view into the provided container.
 *
 * Example usage:
 * @code
 * for(auto&& value: Reversed(container)) {
 *     // use value here.
 * }
 * @endcode
 *
 * @param cont                          Container to provide a view into. Must be an lvalue.
 * @returns                             A reverse view into the provided container.
 */
template <typename TContainerRefOrObject>
constexpr ::NPrivate::TReverseImpl<TContainerRefOrObject> Reversed(TContainerRefOrObject&& cont) {
    return ::NPrivate::TReverseImpl<TContainerRefOrObject>(cont);
}
