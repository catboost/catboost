#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Holds a nullary functor standing in for a value that must not be computed
//! unless it is actually needed.
/*!
 *  Build one with |YT_LAZY(expr)| and consume it with #Force. The expression runs only
 *  when the consumer asks for it, so it may touch state that is valid only under the
 *  condition guarding its use. Not memoized: every #Force re-runs it.
 */
template <class TFunctor>
struct TLazy
{
    static_assert(
        std::invocable<const TFunctor&>,
        "TLazy functor must be invocable on a const instance");

    TFunctor Functor;
};

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
constexpr bool IsLazy = false;

template <class TFunctor>
constexpr bool IsLazy<TLazy<TFunctor>> = true;

} // namespace NDetail

template <class T>
concept CLazy = NDetail::IsLazy<std::remove_cvref_t<T>>;

////////////////////////////////////////////////////////////////////////////////

//! Evaluates a lazy #value; passes any other #value through untouched.
template <class T>
decltype(auto) Force(T&& value);

namespace NDetail {

// A forwarding pack deduces a plain value type from an xvalue, never an rvalue reference.
template <class T>
using TCollapseRvalueRef = std::conditional_t<
    std::is_rvalue_reference_v<T>,
    std::remove_reference_t<T>,
    T
>;

template <class T>
struct TForcedTraits
{
    using TType = TCollapseRvalueRef<T>;
};

template <CLazy T>
struct TForcedTraits<T>
{
    using TType = TCollapseRvalueRef<decltype(std::declval<const std::remove_cvref_t<T>&>().Functor())>;
};

} // namespace NDetail

//! The type a (possibly lazy) #T stands for.
/*!
 *  Exactly what a forwarding pack deduces from |Force(...)|, so |TFormatString<TForced<TArgs>...>|
 *  still matches the |TFormatString<TArgs...>| it forwards to. References therefore survive.
 */
template <class T>
using TForced = typename NDetail::TForcedTraits<T>::TType;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

//! Defers the given expression until it is consumed.
/*!
 *  Captures by reference, so the result must not outlive the scope that built it.
 *
 *  The operand keeps its value category, as #Force does with a non-lazy value, so it must
 *  not root a reference in a temporary: |YT_LAZY(GetOptions().Name)| dangles, the
 *  temporary dying with the enclosing |return| -- bind it first.
 */
#define YT_LAZY(...) ::NYT::TLazy{[&] () -> decltype(auto) { return (__VA_ARGS__); }}

#define LAZY_INL_H_
#include "lazy-inl.h"
#undef LAZY_INL_H_
