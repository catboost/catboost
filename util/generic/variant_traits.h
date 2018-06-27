#pragma once

#include "typetraits.h"

#include <util/system/yassert.h>

#include <utility>
#include <type_traits>

template <class... Ts>
class TVariant;


namespace NVariant {
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

    template <size_t I, class... Ts>
    using TTypeByIndexType = typename TTypeByIndex<I, Ts...>::type;


    template <size_t I, class V>
    struct TAlternative;

    template <size_t I, class... Ts>
    struct TAlternative<I, TVariant<Ts...>> {
        using type = TTypeByIndexType<I, Ts...>;
    };

    template <size_t I, class V>
    using TAlternativeType = typename TAlternative<I, V>::type;


    template <class V>
    struct TSize;

    template <class... Ts>
    struct TSize<TVariant<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)> {};
}


template <class F, class V>
decltype(auto) Visit(F&& f, V&& v);


template <class T, class... Ts>
constexpr bool HoldsAlternative(const TVariant<Ts...>& v);


template <class T, class... Ts>
T& Get(TVariant<Ts...>& v);

template <class T, class... Ts>
const T& Get(const TVariant<Ts...>& v);

template <class T, class... Ts>
T&& Get(TVariant<Ts...>&& v);

template <size_t I, class... Ts>
::NVariant::TAlternativeType<I, TVariant<Ts...>>& Get(TVariant<Ts...>& v);

template <size_t I, class... Ts>
const ::NVariant::TAlternativeType<I, TVariant<Ts...>>& Get(const TVariant<Ts...>& v);

template <size_t I, class... Ts>
::NVariant::TAlternativeType<I, TVariant<Ts...>>&& Get(TVariant<Ts...>&& v);


template <class T, class... Ts>
T* GetIf(TVariant<Ts...>* v);

template <class T, class... Ts>
const T* GetIf(const TVariant<Ts...>* v);

template <size_t I, class... Ts>
::NVariant::TAlternativeType<I, TVariant<Ts...>>* GetIf(TVariant<Ts...>* v);

template <size_t I, class... Ts>
const ::NVariant::TAlternativeType<I, TVariant<Ts...>>* GetIf(const TVariant<Ts...>* v);


namespace NVariant {
    constexpr size_t T_NPOS = -1;


    template <class T>
    struct TTypeHolder {
        using type = T;
    };


    template <class X, class... Ts>
    struct TIndexOf;

    template <class X, class... Ts>
    struct TIndexOf<X, X, Ts...> : std::integral_constant<size_t, 0> {};

    template <class X, class T, class... Ts>
    struct TIndexOf<X, T, Ts...>
        : std::conditional_t<TIndexOf<X, Ts...>::value == T_NPOS
        , std::integral_constant<size_t, T_NPOS>
        , std::integral_constant<size_t, TIndexOf<X, Ts...>::value + 1>> {};

    template <class X>
    struct TIndexOf<X> : std::integral_constant<size_t, T_NPOS> {};


    template <class... Ts>
    struct TTypeTraits {
        struct TNoRefs : TConjunction<TNegation<std::is_reference<Ts>>...> {};
        struct TNoVoids : TConjunction<TNegation<std::is_same<Ts, void>>...> {};
        struct TNoArrays : TConjunction<TNegation<std::is_array<Ts>>...> {};
        struct TNotEmpty : std::integral_constant<bool, (sizeof...(Ts) > 0)> {};
    };


    template <class FRef, class VRef, size_t I = 0>
    using TReturnType = decltype(std::declval<FRef>()(::Get<I>(std::declval<VRef>())));

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
        return std::forward<FRef>(f)(::Get<T>(std::forward<VRef>(v)));
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
