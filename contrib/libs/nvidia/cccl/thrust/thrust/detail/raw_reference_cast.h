/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/tuple.h>

#include <cuda/std/type_traits>

// the order of declarations and definitions in this file is totally goofy
// this header defines raw_reference_cast, which has a few overloads towards the bottom of the file
// raw_reference_cast depends on metafunctions such as is_unwrappable and raw_reference
// we need to be sure that these metafunctions are completely defined (including specializations) before they are
// instantiated by raw_reference_cast

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename... Ts>
class tuple_of_iterator_references;

__THRUST_DEFINE_HAS_NESTED_TYPE(is_wrapped_reference, wrapped_reference_hint)

// wrapped reference-like things which aren't strictly wrapped references
// (e.g. tuples of wrapped references) are considered unwrappable
template <typename T>
inline constexpr bool can_unwrap = is_wrapped_reference<T>::value;

// specialize is_unwrappable
// a tuple is_unwrappable if any of its elements is_unwrappable
template <typename... Ts>
inline constexpr bool can_unwrap<tuple<Ts...>> = (can_unwrap<Ts> || ...);

// specialize is_unwrappable
// a tuple_of_iterator_references is_unwrappable if any of its elements is_unwrappable
template <typename... Ts>
inline constexpr bool can_unwrap<tuple_of_iterator_references<Ts...>> = (can_unwrap<Ts> || ...);

namespace raw_reference_detail
{

template <typename T, typename SFINAE = void>
struct raw_reference_impl : ::cuda::std::add_lvalue_reference<T>
{};

template <typename T>
struct raw_reference_impl<T, ::cuda::std::enable_if_t<is_wrapped_reference<::cuda::std::remove_cv_t<T>>::value>>
    : ::cuda::std::add_lvalue_reference<typename pointer_element<typename T::pointer>::type>
{};

template <typename T>
struct raw_reference_impl<T, ::cuda::std::enable_if_t<is_proxy_reference_v<::cuda::std::remove_cv_t<T>>>>
{
  using type = T;
};

} // namespace raw_reference_detail

template <typename T>
struct raw_reference : raw_reference_detail::raw_reference_impl<T>
{};

namespace raw_reference_detail
{

// unlike raw_reference,
// raw_reference_tuple_helper needs to return a value
// when it encounters one, rather than a reference
// upon encountering tuple, recurse
//
// we want the following behavior:
//  1. T                                -> T
//  2. T&                               -> T&
//  3. null_type                        -> null_type
//  4. reference<T>                     -> T&
//  5. tuple_of_iterator_references<T>  -> tuple_of_iterator_references<raw_reference_tuple_helper<T>::type>

// wrapped references are unwrapped using raw_reference, otherwise, return T
template <typename T>
struct raw_reference_tuple_helper
    : eval_if<can_unwrap<::cuda::std::remove_cv_t<T>>, raw_reference<T>, ::cuda::std::type_identity<T>>
{};

// recurse on tuples
template <typename... Ts>
struct raw_reference_tuple_helper<tuple<Ts...>>
{
  using type = tuple<typename raw_reference_tuple_helper<Ts>::type...>;
};

template <typename... Ts>
struct raw_reference_tuple_helper<tuple_of_iterator_references<Ts...>>
{
  using type = tuple_of_iterator_references<typename raw_reference_tuple_helper<Ts>::type...>;
};

} // namespace raw_reference_detail

// a couple of specializations of raw_reference for tuples follow

// if a tuple "tuple_type" is_unwrappable,
//   then the raw_reference of tuple_type is a tuple of its members' raw_references
//   else the raw_reference of tuple_type is tuple_type &
template <typename... Ts>
struct raw_reference<tuple<Ts...>>
{
private:
  using tuple_type = tuple<Ts...>;

public:
  using type = typename eval_if<can_unwrap<tuple_type>,
                                raw_reference_detail::raw_reference_tuple_helper<tuple_type>,
                                ::cuda::std::add_lvalue_reference<tuple_type>>::type;
};

template <typename... Ts>
struct raw_reference<tuple_of_iterator_references<Ts...>>
{
  using type = typename raw_reference_detail::raw_reference_tuple_helper<tuple_of_iterator_references<Ts...>>::type;
};

} // namespace detail

// provide declarations of raw_reference_cast's overloads for raw_reference_caster below
template <typename T>
_CCCL_HOST_DEVICE typename detail::raw_reference<T>::type raw_reference_cast(T& ref)
{
  return *thrust::raw_pointer_cast(&ref);
}

template <typename T>
_CCCL_HOST_DEVICE typename detail::raw_reference<const T>::type raw_reference_cast(const T& ref)
{
  return *thrust::raw_pointer_cast(&ref);
}

template <typename T, ::cuda::std::enable_if_t<detail::is_proxy_reference_v<::cuda::std::remove_cv_t<T>>, int> = 0>
_CCCL_HOST_DEVICE typename detail::raw_reference<T>::type raw_reference_cast(T&& t)
{
  return t;
}

template <typename... Ts>
_CCCL_HOST_DEVICE auto raw_reference_cast(detail::tuple_of_iterator_references<Ts...> t) ->
  typename detail::raw_reference<detail::tuple_of_iterator_references<Ts...>>::type
{
  if constexpr (detail::can_unwrap<detail::tuple_of_iterator_references<Ts...>>)
  {
    using ResultTuple = tuple<typename detail::raw_reference_detail::raw_reference_tuple_helper<Ts>::type...>;
    return ::cuda::std::apply(
      [](auto&&... refs) {
        return ResultTuple{raw_reference_cast(::cuda::std::forward<decltype(refs)>(refs))...};
      },
      static_cast<tuple<Ts...>&>(t));
  }
  else
  {
    return t;
  }
}

THRUST_NAMESPACE_END
