#pragma once

#include "typelist.h"

#include <functional>

namespace NPrivate {
    template <class F>
    struct TRemoveClassImpl {
        using TSignature = F;
    };

#define Y_EMPTY_REF_QUALIFIER
#define Y_FOR_EACH_REF_QUALIFIERS_COMBINATION(XX) \
    XX(Y_EMPTY_REF_QUALIFIER)                     \
    XX(&)                                         \
    XX(&&)                                        \
    XX(const)                                     \
    XX(const&)                                    \
    XX(const&&)

#define Y_DECLARE_REMOVE_CLASS_IMPL(qualifiers)             \
    template <typename C, typename R, typename... Args>     \
    struct TRemoveClassImpl<R (C::*)(Args...) qualifiers> { \
        typedef R TSignature(Args...);                      \
    };

    Y_FOR_EACH_REF_QUALIFIERS_COMBINATION(Y_DECLARE_REMOVE_CLASS_IMPL)
#undef Y_DECLARE_REMOVE_CLASS_IMPL

    template <class T>
    struct TRemoveNoExceptImpl {
        using Type = T;
    };

    template <typename R, typename... Args>
    struct TRemoveNoExceptImpl<R(Args...) noexcept> {
        using Type = R(Args...);
    };

#define Y_DECLARE_REMOVE_NOEXCEPT_IMPL(qualifiers)                      \
    template <typename R, typename C, typename... Args>                 \
    struct TRemoveNoExceptImpl<R (C::*)(Args...) qualifiers noexcept> { \
        using Type = R (C::*)(Args...);                                 \
    };

    Y_FOR_EACH_REF_QUALIFIERS_COMBINATION(Y_DECLARE_REMOVE_NOEXCEPT_IMPL)
#undef Y_DECLARE_REMOVE_NOEXCEPT_IMPL

#undef Y_FOR_EACH_REF_QUALIFIERS_COMBINATION
#undef Y_EMPTY_REF_QUALIFIER

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
        typedef R TSignature(Args...);
    };
}

template <class C>
using TFunctionSignature = typename ::NPrivate::TFuncInfo<::NPrivate::TRemoveClass<std::remove_reference_t<std::remove_pointer_t<C>>>>::TSignature;

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
