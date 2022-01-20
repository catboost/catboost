#pragma once

#include <variant>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, class... Ts>
struct TIndexOf;

} // namespace NDetail

template <class T, class V>
struct TVariantIndex;

template <class T, class... Ts>
struct TVariantIndex<T, std::variant<Ts...>>
    : std::integral_constant<size_t, NDetail::TIndexOf<T, Ts...>::Value>
{ };

template <class T, class V>
constexpr size_t VariantIndexV = TVariantIndex<T, V>::value;

////////////////////////////////////////////////////////////////////////////////

class TStringBuilderBase;

template <class... Ts>
void FormatValue(TStringBuilderBase* builder, const std::variant<Ts...>& variant, TStringBuf spec);

template <class... Ts>
TString ToString(const std::variant<Ts...>& variant);

////////////////////////////////////////////////////////////////////////////////

//! A concise way of creating a functor with an overloaded operator().
/*!
 *  Very useful for std::visit-ing variants. For example:
 *
 *      std::visit(TOverloaded{
 *          [] (int i)                { printf("The variant holds an int: %d!", i); },
 *          [] (const std::string& s) { printf("The variant holds a string: '%s'!", s); }
 *      }, variantVariable);
 */
template<class... Ts> struct TOverloaded : Ts... { using Ts::operator()...; };
template<class... Ts> TOverloaded(Ts...) -> TOverloaded<Ts...>;

////////////////////////////////////////////////////////////////////////////////

//! An alternative to std::visit that takes its variant argument first.
/*!
 *  This deprives it of being able to visit a Cartesian product of variants but
 *  in exchange allows to receive multiple visitor functors. All of operator()s
 *  these functors have are used to visit the variant after a single unified
 *  overload resolution. For example:
 *
 *      Visit(variantVariable,
 *          [] (int i)                { printf("The variant holds an int: %d!", i); },
 *          [] (const std::string& s) { printf("The variant holds a string: '%s'!", s); });
 */
template <class T, class... U>
auto Visit(T&& variant, U&&... visitorOverloads)
{
    return std::visit(TOverloaded{std::forward<U>(visitorOverloads)...}, std::forward<T>(variant));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define VARIANT_INL_H_
#include "variant-inl.h"
#undef VARIANT_INL_H_
