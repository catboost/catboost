#pragma once

#ifndef VARIANT_TRAITS_H_
#error "Direct inclusion of this file is not allowed, include variant.h"
#endif

namespace NVariant {
    template <class... Ts>
    struct TVisitTraits;

    template <class T, class... Ts>
    struct TVisitTraits<T, Ts...> {
        template <class Result, class Visitor>
        static Result Visit(int tag, const void* storage, Visitor&& visitor) {
            if (tag == 0) {
                return visitor(*reinterpret_cast<const T*>(storage));
            } else {
                return TVisitTraits<Ts...>::template Visit<Result>(tag - 1, storage, std::forward<Visitor>(visitor));
            }
        }

        template <class Result, class Visitor>
        static Result Visit(int tag, void* storage, Visitor&& visitor) {
            if (tag == 0) {
                return visitor(*reinterpret_cast<T*>(storage));
            } else {
                return TVisitTraits<Ts...>::template Visit<Result>(tag - 1, storage, std::forward<Visitor>(visitor));
            }
        }
    };

    template <>
    struct TVisitTraits<> {
        template <class Result, class Visitor>
        static Result Visit(int /*tag*/, const void* /*storage*/, Visitor&& /*visitor*/) {
            // Invalid TVariant tag.
            assert(false);
            return Result();
        }

        template <class Result, class Visitor>
        static Result Visit(int /*tag*/, void* /*storage*/, Visitor&& /*visitor*/) {
            // Invalid TVariant tag.
            assert(false);
            return Result();
        }
    };

    template <class X>
    struct TTagTraits<X> {
        static const int Tag = -1;
    };

    template <class X, class... Ts>
    struct TTagTraits<X, X, Ts...> {
        static const int Tag = 0;
    };

    template <class X, class T, class... Ts>
    struct TTagTraits<X, T, Ts...> {
        static const int Tag = TTagTraits<X, Ts...>::Tag != -1 ? TTagTraits<X, Ts...>::Tag + 1 : -1;
    };

    template <class T, class... Ts>
    struct TTypeTraits<T, Ts...> {
        static const bool NoRefs = !std::is_reference<T>::value && TTypeTraits<Ts...>::NoRefs;
        static const bool NoDuplicates = TTagTraits<T, Ts...>::Tag == -1 && TTypeTraits<Ts...>::NoDuplicates;
    };

    template <>
    struct TTypeTraits<> {
        static const bool NoRefs = true;
        static const bool NoDuplicates = true;
    };

    struct TEmptyVisitorResult;

    template <class Visitor, class... Ts>
    struct TVisitorResult;

    template <class Visitor, class T, class... Ts>
    struct TVisitorResult<Visitor, T, Ts...> {
        using TType = decltype(std::declval<Visitor>()(std::declval<T&>()));

        using TNextType = typename TVisitorResult<Visitor, Ts...>::TType;

        static_assert(std::is_same<TNextType, TType>::value || std::is_same<TNextType, TEmptyVisitorResult>::value, "Don't mess with variant visitors!!!");
    };

    template <class Visitor>
    struct TVisitorResult<Visitor> {
        using TType = TEmptyVisitorResult;
    };

} // namespace NVariant
