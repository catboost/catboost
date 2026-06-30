#pragma once

#include <util/generic/string.h>
#include <util/generic/variant.h>

namespace NYT {

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
