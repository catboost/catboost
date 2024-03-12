#pragma once

#include "va_args.h"

#include <util/system/defaults.h>

#include <iterator>
#include <type_traits>
#include <stlfwd>

#if _LIBCPP_STD_VER >= 17
template <bool B>
using TBoolConstant = std::bool_constant<B>;
#else
template <bool B>
struct TBoolConstant: std::integral_constant<bool, B> {};
#endif

#if _LIBCPP_STD_VER >= 17
template <class B>
using TNegation = std::negation<B>;
#else
template <class B>
struct TNegation: ::TBoolConstant<!bool(B::value)> {};
#endif

namespace NPrivate {
    template <class... Bs>
    constexpr bool ConjunctionImpl() {
        bool bs[] = {(bool)Bs::value...};
        for (auto b : bs) {
            if (!b) {
                return false;
            }
        }
        return true;
    }

    template <class... Bs>
    constexpr bool DisjunctionImpl() {
        bool bs[] = {(bool)Bs::value...};
        for (auto b : bs) {
            if (b) {
                return true;
            }
        }
        return false;
    }
}

#if _LIBCPP_STD_VER >= 17 && !defined(_MSC_VER)
// Disable std::conjunction for MSVC by analogy with std::disjunction.
template <class... Bs>
using TConjunction = std::conjunction<Bs...>;
#else
template <class... Bs>
struct TConjunction: ::TBoolConstant<::NPrivate::ConjunctionImpl<Bs...>()> {};
#endif

#if _LIBCPP_STD_VER >= 17 && !defined(_MSC_VER)
// Disable std::disjunction for MSVC.
// It reduces build time (500 -> 20 seconds) and memory consumption (20 GB -> less than 1 GB)
// for some files (notably search/dssm_boosting/dssm_boosting_calcer.cpp).
template <class... Bs>
using TDisjunction = std::disjunction<Bs...>;
#else
template <class... Bs>
struct TDisjunction: ::TBoolConstant<::NPrivate::DisjunctionImpl<Bs...>()> {};
#endif

#if _LIBCPP_STD_VER >= 17
template <class... Bs>
using TVoidT = std::void_t<Bs...>;
#else
template <class...>
using TVoidT = void;
#endif

template <class T>
struct TPodTraits {
    enum {
        IsPod = false
    };
};

template <class T>
class TTypeTraitsBase {
public:
    static constexpr bool IsPod = (TPodTraits<std::remove_cv_t<T>>::IsPod || std::is_scalar<std::remove_all_extents_t<T>>::value ||
                                   TPodTraits<std::remove_cv_t<std::remove_all_extents_t<T>>>::IsPod);
};

namespace NPrivate {
    template <class T>
    struct TIsSmall: std::integral_constant<bool, (sizeof(T) <= sizeof(void*))> {};
}

template <class T>
class TTypeTraits: public TTypeTraitsBase<T> {
    using TBase = TTypeTraitsBase<T>;

    /*
     * can be effectively passed to function as value
     */
    static constexpr bool IsValueType = std::is_scalar<T>::value ||
                                        std::is_array<T>::value ||
                                        std::is_reference<T>::value ||
                                        (TBase::IsPod &&
                                         std::conditional_t<
                                             std::is_function<T>::value,
                                             std::false_type,
                                             ::NPrivate::TIsSmall<T>>::value);

public:
    /*
     * can be used in function templates for effective parameters passing
     */
    using TFuncParam = std::conditional_t<IsValueType, T, const std::remove_reference_t<T>&>;
};

template <>
class TTypeTraits<void>: public TTypeTraitsBase<void> {};

#define Y_DECLARE_PODTYPE(type) \
    template <>                 \
    struct TPodTraits<type> {   \
        enum { IsPod = true };  \
    }

#define Y_HAS_MEMBER_IMPL_2(method, name)                                                 \
    template <class T>                                                                    \
    struct TClassHas##name {                                                              \
        struct TBase {                                                                    \
            void method();                                                                \
        };                                                                                \
        class THelper: public T, public TBase {                                           \
        public:                                                                           \
            template <class T1>                                                           \
            inline THelper(const T1& = T1()) {                                            \
            }                                                                             \
        };                                                                                \
        template <class T1, T1 val>                                                       \
        class TChecker {};                                                                \
        struct TNo {                                                                      \
            char ch;                                                                      \
        };                                                                                \
        struct TYes {                                                                     \
            char arr[2];                                                                  \
        };                                                                                \
        template <class T1>                                                               \
        static TNo CheckMember(T1*, TChecker<void (TBase::*)(), &T1::method>* = nullptr); \
        static TYes CheckMember(...);                                                     \
        static constexpr bool value =                                                     \
            (sizeof(TYes) == sizeof(CheckMember((THelper*)nullptr)));                     \
    };                                                                                    \
    template <class T, bool isClassType>                                                  \
    struct TBaseHas##name: std::false_type {};                                            \
    template <class T>                                                                    \
    struct TBaseHas##name<T, true>                                                        \
        : std::integral_constant<bool, TClassHas##name<T>::value> {};                     \
    template <class T>                                                                    \
    struct THas##name                                                                     \
        : TBaseHas##name<T, std::is_class<T>::value || std::is_union<T>::value> {}

#define Y_HAS_MEMBER_IMPL_1(name) Y_HAS_MEMBER_IMPL_2(name, name)

/* @def Y_HAS_MEMBER
 *
 * This macro should be used to define compile-time introspection helper classes for template
 * metaprogramming.
 *
 * Macro accept one or two parameters, when used with two parameters e.g. `Y_HAS_MEMBER(xyz, ABC)`
 * will define class `THasABC` with static member `value` of type bool. Usage with one parameter
 * e.g. `Y_HAS_MEMBER(xyz)` will produce the same result as `Y_HAS_MEMBER(xyz, xyz)`.
 *
 * @code
 * #include <type_traits>
 *
 * Y_HAS_MEMBER(push_front, PushFront);
 *
 * template <typename T, typename U>
 * std::enable_if_t<THasPushFront<T>::value, void>
 * PushFront(T& container, const U value) {
 *     container.push_front(x);
 * }
 *
 * template <typename T, typename U>
 * std::enable_if_t<!THasPushFront<T>::value, void>
 * PushFront(T& container, const U value) {
 *     container.insert(container.begin(), x);
 * }
 * @endcode
 */
#define Y_HAS_MEMBER(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_HAS_MEMBER_IMPL_2, Y_HAS_MEMBER_IMPL_1)(__VA_ARGS__))

#define Y_HAS_SUBTYPE_IMPL_2(subtype, name) \
    template <class T, class = void>        \
    struct THas##name: std::false_type {};  \
    template <class T>                      \
    struct THas##name<T, ::TVoidT<typename T::subtype>>: std::true_type {}

#define Y_HAS_SUBTYPE_IMPL_1(name) Y_HAS_SUBTYPE_IMPL_2(name, name)

/* @def Y_HAS_SUBTYPE
 *
 * This macro should be used to define compile-time introspection helper classes for template
 * metaprogramming.
 *
 * Macro accept one or two parameters, when used with two parameters e.g. `Y_HAS_SUBTYPE(xyz, ABC)`
 * will define class `THasABC` with static member `value` of type bool. Usage with one parameter
 * e.g. `Y_HAS_SUBTYPE(xyz)` will produce the same result as `Y_HAS_SUBTYPE(xyz, xyz)`.
 *
 * @code
 * Y_HAS_MEMBER(find, FindMethod);
 * Y_HAS_SUBTYPE(const_iterator, ConstIterator);
 * Y_HAS_SUBTYPE(key_type, KeyType);
 *
 * template <typename T>
 * using TIsAssocCont = std::conditional_t<
 *     THasFindMethod<T>::value && THasConstIterator<T>::value && THasKeyType<T>::value,
 *     std::true_type,
 *     std::false_type,
 * >;
 *
 * static_assert(TIsAssocCont<TVector<int>>::value == false, "");
 * static_assert(TIsAssocCont<THashMap<int>>::value == true, "");
 * @endcode
 */
#define Y_HAS_SUBTYPE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_HAS_SUBTYPE_IMPL_2, Y_HAS_SUBTYPE_IMPL_1)(__VA_ARGS__))

template <class T1, class T2>
struct TPodTraits<std::pair<T1, T2>> {
    enum {
        IsPod = TTypeTraits<T1>::IsPod && TTypeTraits<T2>::IsPod
    };
};

template <class T>
struct TIsPointerToConstMemberFunction: std::false_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const>: std::true_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const&>: std::true_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const&&>: std::true_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const>: std::true_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const&>: std::true_type {
};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const&&>: std::true_type {
};

namespace NPrivate {
    template <template <typename...> class TBase, class TDerived>
    struct TIsBaseOfTemplateHelper {
        template <typename... Ts>
        static constexpr std::true_type Check(const TBase<Ts...>*);

        static constexpr std::false_type Check(...);

        using TType = decltype(Check(std::declval<TDerived*>()));
    };
}

template <template <class...> class T, class U>
struct TIsSpecializationOf: std::false_type {};

template <template <class...> class T, class... Ts>
struct TIsSpecializationOf<T, T<Ts...>>: std::true_type {};

template <template <typename...> class TBase, class TDerived>
using TIsTemplateBaseOf = typename ::NPrivate::TIsBaseOfTemplateHelper<TBase, TDerived>::TType;

/*
 * TDependentFalse is a constant dependent on a template parameter.
 * Use it in static_assert in a false branch of if constexpr to produce a compile error.
 * See an example with dependent_false at https://en.cppreference.com/w/cpp/language/if
 *
 * if constexpr (std::is_same<T, someType1>) {
 * } else if constexpr (std::is_same<T, someType2>) {
 * } else {
 *     static_assert(TDependentFalse<T>, "unknown type");
 * }
 */
template <typename... T>
constexpr bool TDependentFalse = false;

// FIXME: neither nvcc10 nor nvcc11 support using auto in this context
#if defined(__NVCC__)
template <size_t Value>
constexpr bool TValueDependentFalse = false;
#else
template <auto... Values>
constexpr bool TValueDependentFalse = false;
#endif

/*
 * shortcut for std::enable_if_t<...> which checks that T is std::tuple or std::pair
 */
template <class T, class R = void>
using TEnableIfTuple = std::enable_if_t<::TDisjunction<::TIsSpecializationOf<std::tuple, std::decay_t<T>>,
                                                       ::TIsSpecializationOf<std::pair, std::decay_t<T>>>::value,
                                        R>;

namespace NPrivate {
    // To allow ADL with custom begin/end
    using std::begin;
    using std::end;

    template <typename T>
    auto IsIterableImpl(int) -> decltype(begin(std::declval<T&>()) != end(std::declval<T&>()),   // begin/end and operator !=
                                         ++std::declval<decltype(begin(std::declval<T&>()))&>(), // operator ++
                                         *begin(std::declval<T&>()),                             // operator*
                                         std::true_type{});

    template <typename T>
    std::false_type IsIterableImpl(...);
}

template <typename T>
using TIsIterable = decltype(NPrivate::IsIterableImpl<T>(0));
