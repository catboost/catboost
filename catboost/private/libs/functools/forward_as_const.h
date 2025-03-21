#pragma once

#include <util/generic/yexception.h>
#include <util/system/yassert.h>

#include <type_traits>

namespace NPrivateIntConstFwd {
    template <typename... T>
    struct is_integral_constants;

    template <>
    struct is_integral_constants<> {
        static const bool value = true;
    };

    template <typename T, typename... TOtherTypes>
    struct is_integral_constants<T, TOtherTypes...> {
        static const bool value = false;
    };

    template <typename TInt, TInt constantValue, typename... TOtherTypes>
    struct is_integral_constants<std::integral_constant<TInt, constantValue>, TOtherTypes...> {
        static const bool value = is_integral_constants<TOtherTypes...>::value;
    };
}

/**
 * Holder for integer variable with restricted number of allowed values known at compile time.
 */
template <typename TInt, TInt... allowedOptions>
struct TIntOption;

/**
 * Invokes appropriate instantiation of template callable 'func' depending on values of integer params.
 * It's assumed that callable takes several std::integral_constant<...> objects as input.
 * Usage example:
 *      @code
 *      auto doJobLambda = [&](auto paramOne, auto paramTwo) {
 *          return DoJobFunction<paramOne, paramTwo>(...);
 *      };
 *      auto result = ForwardArgsAsIntegralConst(doJobLambda, true, false);
 *      @endcode
 */
template<class Functor, typename TInt, TInt paramValue, typename... TConstants>
auto ForwardArgsAsIntegralConst(
        Functor&& func, std::integral_constant<TInt, paramValue> param, TConstants... constants
) {
    static_assert(NPrivateIntConstFwd::is_integral_constants<TConstants...>::value);
    return func(param, constants...);
}

template<class Functor, typename TInt, TInt... options, typename... Params>
auto ForwardArgsAsIntegralConst(
        Functor&& func, TIntOption<TInt, options...> param, Params... otherParams
) {
    return param.ForwardSelfAsConst(std::forward<Functor>(func), otherParams...);
}

template<class Functor, typename... Params>
auto ForwardArgsAsIntegralConst(Functor&& func, bool param, Params... otherParams) {
    if (param) {
        return ForwardArgsAsIntegralConst(std::forward<Functor>(func), otherParams..., std::true_type());
    } else {
        return ForwardArgsAsIntegralConst(std::forward<Functor>(func), otherParams..., std::false_type());
    }
}

template <typename TInt>
struct TIntOption<TInt> {
public:
    static bool CheckValue(const TInt&) {
        return false;
    }

protected:
    explicit TIntOption(TInt value) : Value(value) {}
    TIntOption(TInt value, bool validate) : TIntOption(value) {
        Y_ASSERT(!validate);
    }

    const TInt Value;
};

template <typename TInt, TInt option, TInt... otherOptions>
struct TIntOption<TInt, option, otherOptions...> : public TIntOption<TInt, otherOptions...> {
public:
    using TParent = TIntOption<TInt, otherOptions...>;
    using TParent::Value;

    explicit TIntOption(TInt value) : TIntOption(value, true) {}

    static bool CheckValue(const TInt& value) {
        if (value == option) {
            return true;
        }
        return TParent::CheckValue(value);
    }

    template <class Functor, typename... Params>
    auto ForwardSelfAsConst(Functor&& func, Params... otherParams) {
        if (Value == option) {
            return ForwardArgsAsIntegralConst(
                    std::forward<Functor>(func), otherParams..., std::integral_constant<TInt, option>());
        } else {
            if constexpr (!std::is_same<TParent, TIntOption<TInt>>::value) {
                return TParent::ForwardSelfAsConst(
                        std::forward<Functor>(func), otherParams...);
            } else {
                Y_UNREACHABLE();
            }
        }
    }

protected:
    TIntOption(TInt value, bool validate) : TParent(value, false) {
        if (validate && !CheckValue(value)) {
            ythrow yexception() << "Invalid value for initialization of TIntOption.";
        }
    }
};
