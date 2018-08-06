#pragma once

#include "typetraits.h"
#include "typelist.h"

#include <functional>

namespace NPrivate {
    template <class F>
    struct TRemoveClassImpl {
        using TSignature = F;
    };

    template <typename C, typename R, typename... Args>
    struct TRemoveClassImpl<R (C::*)(Args...)> {
        typedef R TSignature(Args...);
    };

    template <typename C, typename R, typename... Args>
    struct TRemoveClassImpl<R (C::*)(Args...) const> {
        typedef R TSignature(Args...);
    };

    template <class T>
    struct TRemoveNoExceptImpl {
        using Type = T;
    };

    template <typename R, typename... Args>
    struct TRemoveNoExceptImpl<R(Args...) noexcept> {
        using Type = R(Args...);
    };

    template <typename R, typename C, typename... Args>
    struct TRemoveNoExceptImpl<R (C::*)(Args...) noexcept> {
        using Type = R (C::*)(Args...);
    };

    template <class T>
    using TRemoveNoExcept = typename TRemoveNoExceptImpl<T>::Type;

    template <class F>
    using TRemoveClass = typename TRemoveClassImpl<TRemoveNoExcept<F>>::TSignature;

    template <class C>
    struct TFuncInfo {
        using TSignature = TRemoveClass<decltype(&C::operator())>;
    };

    template <class R, typename... Args>
    struct TFuncInfo<R(Args...)> {
        using TResult = R;
        typedef R TSignature(Args...);
    };
}

template <class C>
using TFunctionSignature = typename ::NPrivate::TFuncInfo< ::NPrivate::TRemoveClass<std::remove_reference_t<std::remove_pointer_t<C>>>>::TSignature;

template <typename F>
struct TCallableTraits: public TCallableTraits<TFunctionSignature<F>> {
};

template <typename R, typename... Args>
struct TCallableTraits<R(Args...)> {
    using TResult = R;
    using TArgs = TTypeList<Args...>;
    typedef R TSignature(Args...);
};

template <typename C>
using TFunctionResult = typename TCallableTraits<C>::TResult;

template <typename C>
using TFunctionArgs = typename TCallableTraits<C>::TArgs;

template <typename C, size_t N>
struct TFunctionArgImpl {
    using TArgs = TFunctionArgs<C>;
    using TResult = typename TArgs::template TGet<N>;
};

template <typename C, size_t N>
using TFunctionArg = typename TFunctionArgImpl<C, N>::TResult;

// temporary before std::apply appearance

template <typename F, typename Tuple, size_t... I>
auto ApplyImpl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return f(std::get<I>(std::forward<Tuple>(t))...);
}

// change to std::apply after c++ 17
template <typename F, typename Tuple>
auto Apply(F&& f, Tuple&& t) {
    return ApplyImpl(f, t, std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

// change to std::apply after c++ 17
template <typename F>
auto Apply(F&& f, std::tuple<>) {
    return f();
}
