///////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2018 NVIDIA Corporation
//  Copyright (c) 2015-2018 Bryce Adelstein Lelbach aka wash
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

/*! \file logical_metafunctions.h
 *  \brief C++17's \c conjunction, \c disjunction, and \c negation metafunctions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <type_traits>

THRUST_NAMESPACE_BEGIN

#if THRUST_CPP_DIALECT >= 2017

/// An \c integral_constant whose value is <code>(... && Ts::value)</code>. 
template <typename... Ts>
using conjunction = std::conjunction<Ts...>;

/// A <code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>.
template <typename... Ts>
constexpr bool conjunction_v = conjunction<Ts...>::value;

/// An \c integral_constant whose value is <code>(... || Ts::value)</code>. 
template <typename... Ts>
using disjunction = std::disjunction<Ts...>;

/// A <code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>.
template <typename... Ts>
constexpr bool disjunction_v = disjunction<Ts...>::value;

/// An \c integral_constant whose value is <code>!Ts::value</code>. 
template <typename T>
using negation = std::negation<T>;

/// A <code>constexpr bool</code> whose value is <code>!Ts::value</code>.
template <typename T>
constexpr bool negation_v = negation<T>::value;

///////////////////////////////////////////////////////////////////////////////

#else // Older than C++17.

/// An \c integral_constant whose value is <code>(... && Ts::value)</code>. 
template <typename... Ts>
struct conjunction;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>(... && Ts::value)</code>.
template <typename... Ts>
constexpr bool conjunction_v = conjunction<Ts...>::value;
#endif

template <>
struct conjunction<> : std::true_type {};

template <typename T>
struct conjunction<T> : T {};

template <typename T0, typename T1>
struct conjunction<T0, T1> : std::conditional<T0::value, T1, T0>::type {};

template<typename T0, typename T1, typename T2, typename... TN>
struct conjunction<T0, T1, T2, TN...>
  : std::conditional<T0::value, conjunction<T1, T2, TN...>, T0>::type {};

///////////////////////////////////////////////////////////////////////////////

/// An \c integral_constant whose value is <code>(... || Ts::value)</code>. 
template <typename... Ts>
struct disjunction;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>(... || Ts::value)</code>.
template <typename... Ts>
constexpr bool disjunction_v = disjunction<Ts...>::value;
#endif

template <>
struct disjunction<> : std::false_type {};

template <typename T>
struct disjunction<T> : T {};

template <typename T0, typename... TN>
struct disjunction<T0, TN...>
  : std::conditional<T0::value != false, T0, disjunction<TN...> >::type {};

///////////////////////////////////////////////////////////////////////////////

/// An \c integral_constant whose value is <code>!T::value</code>. 
template <typename T>
struct negation;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>!T::value</code>.
template <typename T>
constexpr bool negation_v = negation<T>::value;
#endif

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

#endif // THRUST_CPP_DIALECT >= 2017

///////////////////////////////////////////////////////////////////////////////

/// An \c integral_constant whose value is <code>(... && Bs)</code>. 
template <bool... Bs>
struct conjunction_value;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>(... && Bs)</code>.
template <bool... Bs>
constexpr bool conjunction_value_v = conjunction_value<Bs...>::value;
#endif

template <>
struct conjunction_value<> : std::true_type {};

template <bool B>
struct conjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B, bool... Bs>
struct conjunction_value<B, Bs...>
  : std::integral_constant<bool, B && conjunction_value<Bs...>::value> {};

///////////////////////////////////////////////////////////////////////////////

/// An \c integral_constant whose value is <code>(... || Bs)</code>. 
template <bool... Bs>
struct disjunction_value;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>(... || Bs)</code>.
template <bool... Bs>
constexpr bool disjunction_value_v = disjunction_value<Bs...>::value;
#endif

template <>
struct disjunction_value<> : std::false_type {};

template <bool B>
struct disjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B, bool... Bs>
struct disjunction_value<B, Bs...>
  : std::integral_constant<bool, B || disjunction_value<Bs...>::value> {};

///////////////////////////////////////////////////////////////////////////////

/// An \c integral_constant whose value is <code>!B</code>. 
template <bool B>
struct negation_value;

#if THRUST_CPP_DIALECT >= 2014
/// A <code>constexpr bool</code> whose value is <code>!B</code>.
template <bool B>
constexpr bool negation_value_v = negation_value<B>::value;
#endif

template <bool B>
struct negation_value : std::integral_constant<bool, !B> {};

THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

