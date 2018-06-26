#pragma once

#include "typetraits.h"

#include <util/system/yassert.h>

#include <utility>
#include <type_traits>

template <class... Ts>
class TVariant;

namespace NVariant {
    constexpr size_t T_NPOS = -1;


    template <class T>
    struct TTypeHolder {
        using type = T;
    };


    template <class X, class... Ts>
    constexpr size_t GetIndex() {
        bool bs[] = { std::is_same<X, Ts>::value... };
        int ret = 0;
        for (auto b : bs) {
            if (b) {
                return ret;
            }
            ++ret;
        }
        return T_NPOS;
    }

    template <class X, class... Ts>
    struct TIndexOf : std::integral_constant<size_t, GetIndex<X, Ts...>()> {};


    template <class... Ts>
    struct TTypeTraits {
        struct TNoRefs : TConjunction<TNegation<std::is_reference<Ts>>...> {};
        struct TNoVoids : TConjunction<TNegation<std::is_same<Ts, void>>...> {};
        struct TNoArrays : TConjunction<TNegation<std::is_array<Ts>>...> {};
        struct TNotEmpty : std::integral_constant<bool, (sizeof...(Ts) > 0)> {};
    };


    template <size_t I, class... Ts>
    struct TTypeByIndex;

    template <size_t I, class T, class... Ts>
    struct TTypeByIndex<I, T, Ts...> {
        using type = typename TTypeByIndex<I - 1, Ts...>::type;
    };

    template <class T, class... Ts>
    struct TTypeByIndex<0, T, Ts...> {
        using type = T;
    };


    template <class FRef, class VRef, size_t I = 0>
    using TReturnType = decltype(std::declval<FRef>()(std::declval<VRef>().template Get<I>()));

    template <class FRef, class VRef, size_t... Is>
    constexpr bool CheckReturnTypes(std::index_sequence<Is...>) {
        using R = TReturnType<FRef, VRef>;
        bool tests[] = {
            std::is_same<R, TReturnType<FRef, VRef, Is>>::value...
        };
        for (auto b : tests) {
            if (!b) {
                return false;
            }
        }
        return true;
    }

    template <class ReturnType, class T, class FRef, class VRef>
    ReturnType VisitImplImpl(FRef f, VRef v) {
        return std::forward<FRef>(f)(std::forward<VRef>(v).template Get<T>());
    }

    template <class F, class V, class... Ts>
    decltype(auto) VisitImpl(F&& f, V&& v, TTypeHolder<TVariant<Ts...>>) {
        using FRef = decltype(std::forward<F>(f));
        using VRef = decltype(std::forward<V>(v));
        using ReturnType = TReturnType<FRef, VRef>;
        using LambdaType = ReturnType (*)(FRef, VRef);
        static constexpr LambdaType handlers[] = { VisitImplImpl<ReturnType, Ts, FRef, VRef>... };
        return handlers[v.Index()](std::forward<F>(f), std::forward<V>(v));
    }

    template <class F, class V>
    void VisitWrapForVoid(F&& f, V&& v, std::true_type) {
        // We need to make special wrapper when return type equals void
        auto l = [&](auto&& x) {
            std::forward<F>(f)(std::forward<decltype(x)>(x));
            return 0;
        };
        VisitImpl(l, std::forward<V>(v), TTypeHolder<std::decay_t<V>>{});
    }

    template <class F, class V>
    decltype(auto) VisitWrapForVoid(F&& f, V&& v, std::false_type) {
        return VisitImpl(
            std::forward<F>(f), std::forward<V>(v), TTypeHolder<std::decay_t<V>>{});
    }

}
