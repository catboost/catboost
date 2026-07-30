#pragma once

#include <util/generic/typetraits.h>

#include <type_traits>

// See the following references for an inspiration:
//   * http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/type_traits?revision=HEAD&view=markup
//   * http://www.boost.org/doc/libs/1_48_0/libs/type_traits/doc/html/index.html
//   * http://www.boost.org/doc/libs/1_48_0/libs/mpl/doc/index.html

namespace NYT::NMpl {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, bool isPrimitive>
struct TCallTraitsHelper
{ };

template <class T>
struct TCallTraitsHelper<T, true>
{
    using TType = T;
};

template <class T>
struct TCallTraitsHelper<T, false>
{
    using TType = const T&;
};

template <class T>
struct TIsEmpty
    : public T
{
    int Dummy;

    static constexpr bool Value = (sizeof(TIsEmpty) == sizeof(int));
};

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

//! A trait for choosing appropriate argument and return types for functions.
/*!
 *  All types except for primitive ones should be passed to functions
 *  and returned from const getters by const ref.
 */
template <class T>
struct TCallTraits
    : public NDetail::TCallTraitsHelper<T, !std::is_class<T>::value>
{ };

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TIsPod
    : std::integral_constant<bool, ::TTypeTraits<T>::IsPod>
{ };

////////////////////////////////////////////////////////////////////////////////

// Inspired by https://stackoverflow.com/questions/51032671/idiomatic-way-to-write-concept-that-says-that-type-is-a-stdvector
template <class, template <class...> class>
inline constexpr bool IsSpecialization = false;

template <template <class...> class T, class... Args>
inline constexpr bool IsSpecialization<T<Args...>, T> = true;

////////////////////////////////////////////////////////////////////////////////

template <class T>
constexpr bool IsEmptyClass()
{
    return NDetail::TIsEmpty<T>::Value;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NMpl
