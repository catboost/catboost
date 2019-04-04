#pragma once

#include <type_traits>

/* Concepts:
 * Container
 * RemovalContainer
 */
namespace NFlatHash::NConcepts {

#define DCV(type) std::declval<type>()
#define DCT(object) decltype(object)

template <class T, class = void>
struct Container : std::false_type {};

template <class T>
struct Container<T, std::void_t<
    typename T::value_type,
    typename T::size_type,
    typename T::difference_type,
    DCT(DCV(T).Node(DCV(typename T::size_type))),
    DCT(DCV(const T).Node(DCV(typename T::size_type))),
    DCT(DCV(const T).Size()),
    DCT(DCV(const T).Taken()),
    DCT(DCV(const T).Empty()),
    DCT(DCV(const T).IsEmpty(DCV(typename T::size_type))),
    DCT(DCV(const T).IsTaken(DCV(typename T::size_type))),
    DCT(DCV(T).Swap(DCV(T&))),
    DCT(DCV(const T).Clone(DCV(typename T::size_type)))>>
    : std::conjunction<std::is_same<DCT(DCV(T).Node(DCV(typename T::size_type))),
                                    typename T::value_type&>,
                       std::is_same<DCT(DCV(const T).Node(DCV(typename T::size_type))),
                                    const typename T::value_type&>,
                       std::is_same<DCT(DCV(const T).Size()), typename T::size_type>,
                       std::is_same<DCT(DCV(const T).Taken()), typename T::size_type>,
                       std::is_same<DCT(DCV(const T).Empty()), typename T::size_type>,
                       std::is_same<DCT(DCV(const T).IsEmpty(DCV(typename T::size_type))), bool>,
                       std::is_same<DCT(DCV(const T).IsTaken(DCV(typename T::size_type))), bool>,
                       std::is_same<DCT(DCV(const T).Clone(DCV(typename T::size_type))), T>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <class T>
constexpr bool ContainerV = Container<T>::value;

template <class T, class = void>
struct RemovalContainer : std::false_type {};

template <class T>
struct RemovalContainer<T, std::void_t<
    DCT(DCV(T).DeleteNode(DCV(typename T::size_type))),
    DCT(DCV(const T).IsDeleted(DCV(typename T::size_type)))>>
    : std::conjunction<Container<T>,
                       std::is_same<DCT(DCV(const T).IsDeleted(DCV(typename T::size_type))),
                                    bool>> {};

template <class T>
constexpr bool RemovalContainerV = RemovalContainer<T>::value;

#undef DCV
#undef DCT

}  // namespace NFlatHash::NConcepts
