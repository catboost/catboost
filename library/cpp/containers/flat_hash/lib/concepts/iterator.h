#pragma once

#include <iterator>

/* Concepts:
 * Iterator
 */
namespace NFlatHash::NConcepts {

template <class T, class = void>
struct Iterator : std::false_type {};

template <class T>
struct Iterator<T, std::void_t<typename std::iterator_traits<T>::iterator_category>>
    : std::true_type {};

template <class T>
constexpr bool IteratorV = Iterator<T>::value;

}  // namespace NFlatHash::NConcepts
