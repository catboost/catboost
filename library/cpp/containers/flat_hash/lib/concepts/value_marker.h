#pragma once

#include <type_traits>
#include <utility>

/* Concepts:
 * ValueMarker
 */
namespace NFlatHash::NConcepts {

#define DCV(type) std::declval<type>()
#define DCT(object) decltype(object)

template <class T, class = void>
struct ValueMarker : std::false_type {};

template <class T>
struct ValueMarker<T, std::void_t<
    typename T::value_type,
    DCT(DCV(const T).Create()),
    DCT(DCV(const T).Equals(DCV(const typename T::value_type&)))>>
    : std::conjunction<std::is_constructible<typename T::value_type, DCT(DCV(const T).Create())>,
                       std::is_same<DCT(DCV(const T).Equals(DCV(const typename T::value_type&))), bool>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <class T>
constexpr bool ValueMarkerV = ValueMarker<T>::value;

#undef DCV
#undef DCT

}  // namespace NFlatHash::NConcepts
