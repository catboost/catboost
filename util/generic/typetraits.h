#pragma once

#include "typelist.h"
#include "va_args.h"

#include <util/system/defaults.h>

#include <type_traits>
#include <stlfwd>

template <class T>
struct TPodTraits {
    enum {
        IsPod = false
    };
};

namespace NTypeTrait {
    enum ETypeFlag {
        BITWISE_COMPARABLE = 0x1,
        BITWISE_COPYABLE = 0x2,
        BITWISE_SERIALIZABLE = 0x4
    };
}

namespace NPrivate {
    template <typename T>
    struct TUserTypeTrait {
        static constexpr ui64 TypeTraitFlags = 0;
    };
}

template <class T>
class TTypeTraitsBase {
    /*
     * some helpers
     */

    template <class C>
    struct TConstTraits {
        using TResult = C;
    };

    template <class C>
    struct TConstTraits<const C> {
        using TResult = C;
    };

    template <class C>
    struct TVolatileTraits {
        using TResult = C;
    };

    template <class C>
    struct TVolatileTraits<volatile C> {
        using TResult = C;
    };

    template <class C, bool isConst, bool isVolatile>
    struct TApplyQualifiers {
        using TResult = C;
    };

    template <class C>
    struct TApplyQualifiers<C, false, true> {
        using TResult = volatile C;
    };

    template <class C>
    struct TApplyQualifiers<C, true, false> {
        using TResult = const C;
    };

    template <class C>
    struct TApplyQualifiers<C, true, true> {
        using TResult = const volatile C;
    };

public:
    /*
     * qualifier traits
     */

    /*
     * type without 'volatile' qualifier
     */
    using TNonVolatile = typename TVolatileTraits<T>::TResult;

    /*
     * type without qualifiers
     */
    using TNonQualified = typename TConstTraits<TNonVolatile>::TResult;

    /*
     * traits too
     */

    enum {
        IsPod = TPodTraits<TNonQualified>::IsPod || std::is_scalar<std::remove_all_extents_t<T>>::value ||
                TPodTraits<std::remove_cv_t<std::remove_all_extents_t<T>>>::IsPod
    };

    enum {
        IsBitwiseComparable = ::NPrivate::TUserTypeTrait<TNonQualified>::TypeTraitFlags & NTypeTrait::BITWISE_COMPARABLE || std::is_pod<TNonQualified>::value
    };

    enum {
        IsBitwiseCopyable = ::NPrivate::TUserTypeTrait<TNonQualified>::TypeTraitFlags & NTypeTrait::BITWISE_COPYABLE || std::is_pod<TNonQualified>::value
    };

    enum {
        IsBitwiseSerializable = ::NPrivate::TUserTypeTrait<TNonQualified>::TypeTraitFlags & NTypeTrait::BITWISE_SERIALIZABLE || std::is_pod<TNonQualified>::value
    };
};

namespace NPrivate {
    template <class T>
    struct TIsSmall : std::integral_constant<bool, (sizeof(T) <= sizeof(void*))> {
    };
}

template <class T>
class TTypeTraits: public TTypeTraitsBase<T> {
    using TBase = TTypeTraitsBase<T>;

public:
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

    /*
     * can be used in function templates for effective parameters passing
     */
    using TFuncParam = std::conditional_t<IsValueType, T, const std::remove_reference_t<T>&>;
};

template <>
class TTypeTraits<void>: public TTypeTraitsBase<void> {
};

template <template <typename> class E, typename T>
struct TIsCorrectExpression {
private:
    template <typename U>
    static auto TryEvaluate(int) -> decltype((void)std::declval<E<U>>(), std::true_type());

    template <typename>
    static auto TryEvaluate(...) -> std::false_type;

public:
    using Type = decltype(TryEvaluate<T>(0));
    enum { Result = Type::value };
};

namespace NPrivate {
    template <typename... Params>
    struct TTryCall {
        template <typename TFunc>
        using Type = decltype((void)std::declval<TFunc>()(std::declval<Params>()...));
    };
}

/**
 * Checks if TFunc function type can be called with Params as parameters in compile time
 * TIsCallableWith::Result stores the result
 * Example:
 * using TFunc = void(*)(int, int);
 * static_assert(TIsCallableWith<TFunc, int, int>::Result, "");     // OK
 * static_assert(TIsCallableWith<TFunc, double, int>::Result, "");  // OK. Conversion performed
 * static_assert(!TIsCallableWith<TFunc, int>::Result, "");         // Wrong number of arguments
 * static_assert(!TIsCallableWith<TFunc, int, TString>::Result, ""); // There is no conversion from TString to int
 */
template <typename TFunc, typename... Params>
struct TIsCallableWith: public TIsCorrectExpression< ::NPrivate::TTryCall<Params...>::template Type, TFunc> {};

#define Y_DECLARE_TYPE_FLAGS(type, flags)                 \
    namespace NPrivate {                                  \
        template <>                                       \
        struct TUserTypeTrait<type> {                     \
            static constexpr ui64 TypeTraitFlags = flags; \
        };                                                \
    }

#define Y_DECLARE_PODTYPE(type) \
    template <>                 \
    struct TPodTraits<type> {   \
        enum { IsPod = true };  \
    }

#define Y_HAS_MEMBER_IMPL_2(method, name)                                                              \
    template <class T>                                                                                 \
    struct TClassHas##name {                                                                           \
        struct TBase {                                                                                 \
            void method();                                                                             \
        };                                                                                             \
        class THelper: public T, public TBase {                                                        \
        public:                                                                                        \
            template <class T1>                                                                        \
            inline THelper(const T1& = T1()) {                                                         \
            }                                                                                          \
        };                                                                                             \
        template <class T1, T1 val>                                                                    \
        class TChecker {};                                                                             \
        struct TNo {                                                                                   \
            char ch;                                                                                   \
        };                                                                                             \
        struct TYes {                                                                                  \
            char arr[2];                                                                               \
        };                                                                                             \
        template <class T1>                                                                            \
        static TNo CheckMember(T1*, TChecker<void (TBase::*)(), &T1::method>* = 0);                    \
        static TYes CheckMember(...);                                                                  \
        enum { Result = (sizeof(TYes) == sizeof(CheckMember((THelper*)0))) };                          \
    };                                                                                                 \
    template <class T, bool isClassType>                                                               \
    struct TBaseHas##name {                                                                            \
        enum { Result = false };                                                                       \
    };                                                                                                 \
    template <class T>                                                                                 \
    struct TBaseHas##name<T, true>: public TClassHas##name<T> {                                        \
    };                                                                                                 \
    template <class T>                                                                                 \
    struct THas##name: public TBaseHas##name<T, std::is_class<T>::value || std::is_union<T>::value> {  \
    }

#define Y_HAS_MEMBER_IMPL_1(name) Y_HAS_MEMBER_IMPL_2(name, name)

/* @def Y_HAS_MEMBER
 *
 * This macro should be used to define compile-time introspection helper classes for template
 * metaprogramming.
 *
 * Macro accept one or two parameters, when used with two parameters e.g. `Y_HAS_MEMBER(xyz, ABC)`
 * will define class `THasABC` with static member `Result` of type bool. Usage with one parameter
 * e.g. `Y_HAS_MEMBER(xyz)` will produce the same result as `Y_HAS_MEMBER(xyz, xyz)`.
 *
 * @code
 * #include <type_traits>
 *
 * Y_HAS_MEMBER(push_front, PushFront);
 *
 * template <typename T, typename U>
 * std::enable_if_t<THasPushFront<T>::Result, void>
 * PushFront(T& container, const U value) {
 *     container.push_front(x);
 * }
 *
 * template <typename T, typename U>
 * std::enable_if_t<!THasPushFront<T>::Result, void>
 * PushFront(T& container, const U value) {
 *     container.insert(container.begin(), x);
 * }
 * @endcode
 */
#define Y_HAS_MEMBER(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_HAS_MEMBER_IMPL_2, Y_HAS_MEMBER_IMPL_1)(__VA_ARGS__))

#define Y_HAS_SUBTYPE_IMPL_2(subtype, name)                             \
    template <class T>                                                  \
    struct THas##name {                                                 \
        struct TNo {                                                    \
            char ch;                                                    \
        };                                                              \
        struct TYes {                                                   \
            char arr[2];                                                \
        };                                                              \
        template <class T1>                                             \
        static TYes CheckSubtype(typename T1::subtype*);                \
        template <class T1>                                             \
        static TNo CheckSubtype(...);                                   \
        enum { Result = (sizeof(TYes) == sizeof(CheckSubtype<T>(0))) }; \
    }

#define Y_HAS_SUBTYPE_IMPL_1(name) Y_HAS_SUBTYPE_IMPL_2(name, name)

/* @def Y_HAS_SUBTYPE
 *
 * This macro should be used to define compile-time introspection helper classes for template
 * metaprogramming.
 *
 * Macro accept one or two parameters, when used with two parameters e.g. `Y_HAS_SUBTYPE(xyz, ABC)`
 * will define class `THasABC` with static member `Result` of type bool. Usage with one parameter
 * e.g. `Y_HAS_SUBTYPE(xyz)` will produce the same result as `Y_HAS_SUBTYPE(xyz, xyz)`.
 *
 * @code
 * Y_HAS_MEMBER(find, FindMethod);
 * Y_HAS_SUBTYPE(const_iterator, ConstIterator);
 * Y_HAS_SUBTYPE(key_type, KeyType);
 *
 * template <typename T>
 * using TIsAssocCont = std::conditional_t<
 *     THasFindMethod<T>::Result && THasConstIterator<T>::Result && THasKeyType<T>::Result,
 *     std::true_type,
 *     std::false_type,
 * >;
 *
 * static_assert(TIsAssocCont<yvector<int>>::value == false, "");
 * static_assert(TIsAssocCont<yhash<int>>::value == true, "");
 * @endcode
 */
#define Y_HAS_SUBTYPE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_HAS_SUBTYPE_IMPL_2, Y_HAS_SUBTYPE_IMPL_1)(__VA_ARGS__))

template <class T>
struct TDecayArrayImpl {
    using U = std::remove_reference_t<T>;
    using UTraits = TTypeTraits<U>;

    using TResult = std::conditional_t<
        std::is_array<U>::value,
        std::remove_extent_t<U>*,
        std::remove_cv_t<U>>;
};

template <class T>
using TDecayArray = typename TDecayArrayImpl<T>::TResult;

template <class T1, class T2>
struct TPodTraits<std::pair<T1, T2>> {
    enum {
        IsPod = TTypeTraits<T1>::IsPod && TTypeTraits<T2>::IsPod
    };
};

template <typename T>
using TFixedWidthSignedInt = typename TFixedWidthSignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::TResult;

template <typename T>
using TFixedWidthUnsignedInt = typename TFixedWidthUnsignedInts::template TSelectBy<TSizeOfPredicate<sizeof(T)>::template TResult>::TResult;

//NOTE: to be replaced with std::as_const in c++17
template <class T>
constexpr std::add_const_t<T>& AsConst(T& t) noexcept {
    return t;
}
template <class T>
void AsConst(T&& t) = delete;

//NOTE: to be replaced with std::negation in c++17
template <class B>
struct TNegation : std::integral_constant<bool, !bool(B::value)> {};

template <class T>
struct TIsPointerToConstMemberFunction : std::false_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const> : std::true_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const &> : std::true_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args...) const &&> : std::true_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const> : std::true_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const &> : std::true_type {};

template <class R, class T, class... Args>
struct TIsPointerToConstMemberFunction<R (T::*)(Args..., ...) const &&> : std::true_type {};
