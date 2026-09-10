#pragma once

#include <type_traits>
#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Holds a nullary functor standing in for a value that must not be computed
//! unless it is actually needed.
/*!
 *  Build one with |YT_LAZY(expr)| and consume it with #Unlazy. Since the
 *  expression runs only when the consumer asks for it, it may safely touch
 *  state that is valid only under the condition guarding its use.
 */
template <class TFunctor>
struct TLazy
{
    TFunctor Functor;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
constexpr bool IsLazy = false;

template <class TFunctor>
constexpr bool IsLazy<TLazy<TFunctor>> = true;

template <class T>
concept CLazy = IsLazy<std::remove_cvref_t<T>>;

////////////////////////////////////////////////////////////////////////////////

//! Evaluates a lazy #value; passes any other #value through untouched.
template <class T>
decltype(auto) Unlazy(T&& value);

namespace NDetail {

template <class T>
struct TUnlazyTraits
{
    using TType = T;
};

template <class TFunctor>
struct TUnlazyTraits<TLazy<TFunctor>>
{
    using TType = decltype(std::declval<const TFunctor&>()());
};

template <class TFunctor>
struct TUnlazyTraits<TLazy<TFunctor>&>
    : public TUnlazyTraits<TLazy<TFunctor>>
{ };

template <class TFunctor>
struct TUnlazyTraits<const TLazy<TFunctor>&>
    : public TUnlazyTraits<TLazy<TFunctor>>
{ };

} // namespace NDetail

//! The type a (possibly lazy) #T stands for; #T itself when it is not lazy.
/*!
 *  Deliberately preserves the reference-ness of a non-lazy #T: it names the very
 *  type a forwarding parameter pack would deduce, so a |TFormatString<TUnlazy<TArgs>...>|
 *  still matches the |TFormatString<TArgs...>| it forwards to.
 */
template <class T>
using TUnlazy = typename NDetail::TUnlazyTraits<T>::TType;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

//! Defers #expr until it is consumed.
/*!
 *  Captures by reference, so the result must not outlive the state #expr touches.
 *  expression.
 */
#define YT_LAZY(...) ::NYT::TLazy{[&] () -> decltype(auto) { return (__VA_ARGS__); }}

#define LAZY_INL_H_
#include "lazy-inl.h"
#undef LAZY_INL_H_
