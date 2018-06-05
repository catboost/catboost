#pragma once

#include <util/system/yassert.h>

#include <type_traits>

namespace NVariant {
    // TODO(velavokr): T_EMPTY = 0 and T_FIRST = 1 would play better because zero memory it an empty TVariant then.
    // Changing Tag semantics now would break the existing gdb pretty printers though.
    enum ETags {
        T_INVALID = -1,
        T_EMPTY = -2, // 0,
        T_FIRST = 0, // 1,
    };

    template <class... Ts>
    struct TVisitTraits;

    template <class T, class... Ts>
    struct TVisitTraits<T, Ts...> {
        template <class Result, class Visitor>
        static Result Visit(int tag, const void* storage, Visitor&& visitor) {
            if (tag == T_FIRST) {
                return visitor(*reinterpret_cast<const T*>(storage));
            } else {
                return TVisitTraits<Ts...>::template Visit<Result>(tag - 1, storage, std::forward<Visitor>(visitor));
            }
        }

        template <class Result, class Visitor>
        static Result Visit(int tag, void* storage, Visitor&& visitor) {
            if (tag == T_FIRST) {
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
            Y_FAIL("Invalid TVariant tag");
        }

        template <class Result, class Visitor>
        static Result Visit(int /*tag*/, void* /*storage*/, Visitor&& /*visitor*/) {
            Y_FAIL("Invalid TVariant tag");
        }
    };


    template <class X, class... Ts>
    struct TTagTraits;

    template <class X>
    struct TTagTraits<X> {
        static const int Tag = T_INVALID;
    };

    template <class X, class... Ts>
    struct TTagTraits<X, X, Ts...> {
        static const int Tag = T_FIRST;
    };

    template <class X, class T, class... Ts>
    struct TTagTraits<X, T, Ts...> {
        static const int Tag = TTagTraits<X, Ts...>::Tag != T_INVALID ? TTagTraits<X, Ts...>::Tag + 1 : T_INVALID;
    };


    template <class... Ts>
    struct TTypeTraits;

    template <class T, class... Ts>
    struct TTypeTraits<T, Ts...> {
        static const bool NoRefs = !std::is_reference<T>::value && TTypeTraits<Ts...>::NoRefs;
        static const bool NoDuplicates = TTagTraits<T, Ts...>::Tag == T_INVALID && TTypeTraits<Ts...>::NoDuplicates;
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

        static_assert(std::is_same<TNextType, TType>::value || std::is_same<TNextType, TEmptyVisitorResult>::value,
                      "Don't mess with variant visitors!!!");
    };

    template <class Visitor>
    struct TVisitorResult<Visitor> {
        using TType = TEmptyVisitorResult;
    };
}
