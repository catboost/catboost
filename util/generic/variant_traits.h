#pragma once

#include "typetraits.h"
#include "yexception.h"

#include <utility>
#include <type_traits>

template <class... Ts>
class TVariant;

class TWrongVariantError: public yexception {};

namespace NVariant {
    template <size_t I, class T>
    struct TIndexedType {
        static constexpr size_t value = I;
        using type = T;
    };

    template <class, class... Ts>
    struct TIndexedTypesImpl;

    template <std::size_t... Is, class... Ts>
    struct TIndexedTypesImpl<std::index_sequence<Is...>, Ts...> {
        struct type : TIndexedType<Is, Ts>... {};
    };

    template <class... Ts>
    using TIndexedTypes = typename TIndexedTypesImpl<std::index_sequence_for<Ts...>, Ts...>::type;

    template <size_t I, class T>
    constexpr auto GetIndexedType(TIndexedType<I, T> t) {
        return t;
    }

    template <size_t I, class... Ts>
    struct TTypeByIndex {
        using TTypeImpl = decltype(GetIndexedType<I>(TIndexedTypes<Ts...>{})); // MSVC workaround
        using type = typename TTypeImpl::type;
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

    struct TVariantAccessor {
        template <size_t I, class... Ts>
        static TAlternativeType<I, TVariant<Ts...>>& Get(TVariant<Ts...>& v);

        template <size_t I, class... Ts>
        static const TAlternativeType<I, TVariant<Ts...>>& Get(const TVariant<Ts...>& v);

        template <size_t I, class... Ts>
        static TAlternativeType<I, TVariant<Ts...>>&& Get(TVariant<Ts...>&& v);

        template <size_t I, class... Ts>
        static const TAlternativeType<I, TVariant<Ts...>>&& Get(const TVariant<Ts...>&& v);

        template <class... Ts>
        static constexpr size_t Index(const TVariant<Ts...>& v) noexcept;
    };

    constexpr size_t T_NPOS = -1;

    template <class X, class... Ts>
    constexpr size_t IndexOfImpl() {
        bool bs[] = {std::is_same<X, Ts>::value...};
        for (size_t i = 0; i < sizeof...(Ts); ++i) {
            if (bs[i]) {
                return i;
            }
        }
        return T_NPOS;
    }

    template <class X, class... Ts>
    struct TIndexOf : std::integral_constant<size_t, IndexOfImpl<X, Ts...>()> {};

    template <class X, class V>
    struct TAlternativeIndex;

    template <class X, class... Ts>
    struct TAlternativeIndex<X, TVariant<Ts...>> : TIndexOf<X, Ts...> {};

    template <class... Ts>
    struct TTypeTraits {
        using TNoRefs = TConjunction<TNegation<std::is_reference<Ts>>...>;
        using TNoVoids = TConjunction<TNegation<std::is_same<Ts, void>>...>;
        using TNoArrays = TConjunction<TNegation<std::is_array<Ts>>...>;
        using TNotEmpty = std::integral_constant<bool, (sizeof...(Ts) > 0)>;
    };

    template <class FRef, class VRef, size_t I = 0>
    using TReturnType = decltype(
        std::declval<FRef>()(TVariantAccessor::Get<I>(std::declval<VRef>())));

    template <class FRef, class VRef, size_t... Is>
    constexpr bool CheckReturnTypes(std::index_sequence<Is...>) {
        using R = TReturnType<FRef, VRef>;
        return TConjunction<std::is_same<R, TReturnType<FRef, VRef, Is>>...>::value;
    }

    template <class ReturnType, size_t I, class FRef, class VRef>
    ReturnType VisitImplImpl(FRef f, VRef v) {
        return std::forward<FRef>(f)(TVariantAccessor::Get<I>(std::forward<VRef>(v)));
    }

    template <class ReturnType, class FRef, class VRef>
    ReturnType VisitImplFail(FRef, VRef) {
        throw TWrongVariantError{};
    }

    template <class F, class V, size_t... Is>
    decltype(auto) VisitImpl(F&& f, V&& v, std::index_sequence<Is...>) {
        using FRef = decltype(std::forward<F>(f));
        using VRef = decltype(std::forward<V>(v));
        using ReturnType = TReturnType<FRef, VRef>;
        using LambdaType = ReturnType (*)(FRef, VRef);
        static constexpr LambdaType handlers[] = {
            VisitImplImpl<ReturnType, Is, FRef, VRef>...,
            VisitImplFail<ReturnType, FRef, VRef>};
        return handlers[TVariantAccessor::Index(v)](std::forward<F>(f), std::forward<V>(v));
    }

    template <class F, class V>
    void VisitWrapForVoid(F&& f, V&& v, std::true_type) {
        // We need to make special wrapper when return type equals void
        auto l = [&](auto&& x) {
            std::forward<F>(f)(std::forward<decltype(x)>(x));
            return 0;
        };
        VisitImpl(l, std::forward<V>(v), std::make_index_sequence<TSize<std::decay_t<V>>::value>{});
    }

    template <class F, class V>
    decltype(auto) VisitWrapForVoid(F&& f, V&& v, std::false_type) {
        return VisitImpl(std::forward<F>(f),
                         std::forward<V>(v),
                         std::make_index_sequence<TSize<std::decay_t<V>>::value>{});
    }

    // Can be simplified with c++17: IGNIETFERRO-982
    template <class Ret, class F, class T, class U>
    std::enable_if_t<std::is_same<std::decay_t<T>, std::decay_t<U>>::value,
    Ret> CallIfSame(F f, T&& a, U&& b) {
        return f(std::forward<T>(a), std::forward<U>(b));
    }

    template <class Ret, class F, class T, class U>
    std::enable_if_t<!std::is_same<std::decay_t<T>, std::decay_t<U>>::value,
    Ret> CallIfSame(F, T&&, U&&) { // Will never be called
        Y_FAIL();
    }
}
