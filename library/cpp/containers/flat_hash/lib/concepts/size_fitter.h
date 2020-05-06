#pragma once

#include <type_traits>

/* Concepts:
 * SizeFitter
 */
namespace NFlatHash::NConcepts {

#define DCV(type) std::declval<type>()
#define DCT(object) decltype(object)

template <class T, class = void>
struct SizeFitter : std::false_type {};

template <class T>
struct SizeFitter<T, std::void_t<
    DCT(DCV(const T).EvalIndex(DCV(size_t), DCV(size_t))),
    DCT(DCV(const T).EvalSize(DCV(size_t))),
    DCT(DCV(T).Update(DCV(size_t)))>>
    : std::conjunction<std::is_same<DCT(DCV(const T).EvalIndex(DCV(size_t), DCV(size_t))), size_t>,
                       std::is_same<DCT(DCV(const T).EvalSize(DCV(size_t))), size_t>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <class T>
constexpr bool SizeFitterV = SizeFitter<T>::value;

#undef DCV
#undef DCT

}  // namespace NFlatHash::NConcepts
