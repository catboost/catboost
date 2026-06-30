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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/detail/iterator_category_to_system.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/iterator_categories.h>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
inline constexpr bool is_host_iterator_category =
  ::cuda::std::is_convertible_v<T, input_host_iterator_tag>
  || ::cuda::std::is_convertible_v<T, output_host_iterator_tag>;

template <typename T>
inline constexpr bool is_device_iterator_category =
  ::cuda::std::is_convertible_v<T, input_device_iterator_tag>
  || ::cuda::std::is_convertible_v<T, output_device_iterator_tag>;

template <typename T>
inline constexpr bool is_iterator_category = is_host_iterator_category<T> || is_device_iterator_category<T>;

// adapted from http://www.boost.org/doc/libs/1_37_0/libs/iterator/doc/iterator_facade.html#iterator-category
//
// in our implementation, R need not be a reference type to result in a category
// derived from forward_XXX_iterator_tag
//
// iterator-category(T,V,R) :=
//   if(T is convertible to input_host_iterator_tag
//      || T is convertible to output_host_iterator_tag
//      || T is convertible to input_device_iterator_tag
//      || T is convertible to output_device_iterator_tag
//   )
//     return T
//
//   else if (T is not convertible to incrementable_traversal_tag)
//     the program is ill-formed
//
//   else return a type X satisfying the following two constraints:
//
//     1. X is convertible to X1, and not to any more-derived
//        type, where X1 is defined by:
//
//        if (T is convertible to forward_traversal_tag)
//        {
//          if (T is convertible to random_access_traversal_tag)
//            X1 = random_access_host_iterator_tag
//          else if (T is convertible to bidirectional_traversal_tag)
//            X1 = bidirectional_host_iterator_tag
//          else
//            X1 = forward_host_iterator_tag
//        }
//        else
//        {
//          if (T is convertible to single_pass_traversal_tag
//              && R is convertible to V)
//            X1 = input_host_iterator_tag
//          else
//            X1 = T
//        }
//
//     2. category-to-traversal(X) is convertible to the most
//        derived traversal tag type to which X is also convertible,
//        and not to any more-derived traversal tag type.

// Thrust's implementation of iterator_facade_default_category is slightly
// different from Boost's equivalent.
// Thrust does not check is_convertible_v<Reference, ValueParam> because Reference
// may not be a complete type at this point, and implementations of is_convertible_v
// typically require that both types be complete.
// Instead, it simply assumes that if is_convertible_v<Traversal, single_pass_traversal_tag>,
// then the category is input_iterator_tag

// this is the function for standard system iterators
template <typename Traversal, typename ValueParam, typename Reference>
using iterator_facade_default_category_std = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<Traversal, forward_traversal_tag>,
  ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, random_access_traversal_tag>,
                   ::cuda::std::random_access_iterator_tag,
                   ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, bidirectional_traversal_tag>,
                                    ::cuda::std::bidirectional_iterator_tag,
                                    ::cuda::std::forward_iterator_tag>>,
  ::cuda::std::_If< // we differ from Boost here
    ::cuda::std::is_convertible_v<Traversal, single_pass_traversal_tag>,
    ::cuda::std::input_iterator_tag,
    Traversal>>;

// this is the function for host system iterators
template <typename Traversal, typename ValueParam, typename Reference>
using iterator_facade_default_category_host = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<Traversal, forward_traversal_tag>,
  ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, random_access_traversal_tag>,
                   random_access_host_iterator_tag,
                   ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, bidirectional_traversal_tag>,
                                    bidirectional_host_iterator_tag,
                                    forward_host_iterator_tag>>,
  ::cuda::std::_If< // we differ from Boost here
    ::cuda::std::is_convertible_v<Traversal, single_pass_traversal_tag>,
    input_host_iterator_tag,
    Traversal>>;

// this is the function for device system iterators
template <typename Traversal, typename ValueParam, typename Reference>
using iterator_facade_default_category_device = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<Traversal, forward_traversal_tag>,
  ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, random_access_traversal_tag>,
                   random_access_device_iterator_tag,
                   ::cuda::std::_If<::cuda::std::is_convertible_v<Traversal, bidirectional_traversal_tag>,
                                    bidirectional_device_iterator_tag,
                                    forward_device_iterator_tag>>,
  ::cuda::std::_If<
    // XXX note we differ from Boost here
    ::cuda::std::is_convertible_v<Traversal, single_pass_traversal_tag>,
    input_device_iterator_tag,
    Traversal>>;

// this is the function for any system iterators
template <typename Traversal, typename ValueParam, typename Reference>
using iterator_facade_default_category_any =
  iterator_category_with_system_and_traversal<iterator_facade_default_category_std<Traversal, ValueParam, Reference>,
                                              any_system_tag,
                                              Traversal>;

template <typename System, typename Traversal, typename ValueParam, typename Reference>
using iterator_facade_default_category = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<System, any_system_tag>,
  iterator_facade_default_category_any<Traversal, ValueParam, Reference>,
  // check for host system
  ::cuda::std::_If<::cuda::std::is_convertible_v<System, host_system_tag>,
                   iterator_facade_default_category_host<Traversal, ValueParam, Reference>,
                   // check for device system
                   ::cuda::std::_If<::cuda::std::is_convertible_v<System, device_system_tag>,
                                    iterator_facade_default_category_device<Traversal, ValueParam, Reference>,
                                    // if we don't recognize the system, get a standard iterator category
                                    // and combine it with System & Traversal
                                    iterator_category_with_system_and_traversal<
                                      iterator_facade_default_category_std<Traversal, ValueParam, Reference>,
                                      System,
                                      Traversal>>>>;

template <typename System, typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_category_impl
{
  using category = iterator_facade_default_category<System, Traversal, ValueParam, Reference>;

  // we must be able to deduce both Traversal & System from category, otherwise, munge them all together
  using type =
    ::cuda::std::_If<::cuda::std::is_same_v<Traversal, typename iterator_category_to_traversal<category>::type>&& ::
                       cuda::std::is_same_v<System, typename iterator_category_to_system<category>::type>,
                     category,
                     iterator_category_with_system_and_traversal<category, System, Traversal>>;
};

template <typename CategoryOrSystem, typename CategoryOrTraversal, typename ValueParam, typename Reference>
struct iterator_facade_category
{
  using type = typename ::cuda::std::_If<
    is_iterator_category<CategoryOrTraversal>,
    ::cuda::std::type_identity<CategoryOrTraversal>,
    iterator_facade_category_impl<CategoryOrSystem, CategoryOrTraversal, ValueParam, Reference>>::type;
}; // end iterator_facade_category

} // namespace detail
THRUST_NAMESPACE_END
