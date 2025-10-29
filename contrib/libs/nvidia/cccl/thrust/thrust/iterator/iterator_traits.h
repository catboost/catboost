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

/*! \file thrust/iterator/iterator_traits.h
 *  \brief Traits and metafunctions for reasoning about the traits of iterators
 */

/*
 * (C) Copyright David Abrahams 2003.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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

#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/iterator_category_to_system.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_categories.h>

#include <cuda/iterator>
#include <cuda/std/__type_traits/void_t.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// the following iterator helpers are not named it_value_t etc, like the C++20 facilities, because they are defined in
// terms of C++17 iterator_traits and not the new C++20 indirectly_readable trait etc. This allows them to detect nested
// value_type, difference_type and reference aliases, which the new C+20 traits do not consider (they only consider
// specializations of iterator_traits). Also, a value_type of void remains supported (needed by some output iterators).

template <typename It>
using it_value_t = typename ::cuda::std::iterator_traits<It>::value_type;

template <typename It>
using it_reference_t = typename ::cuda::std::iterator_traits<It>::reference;

template <typename It>
using it_difference_t = typename ::cuda::std::iterator_traits<It>::difference_type;

template <typename It>
using it_pointer_t = typename ::cuda::std::iterator_traits<It>::pointer;

// use this whenever you need to lazily evaluate a trait. E.g., as an alternative in replace_if_use_default.
template <template <typename...> typename Trait, typename... Args>
struct lazy_trait
{
  using type = Trait<Args...>;
};
} // namespace detail

//! \p iterator_traits is a type trait class that provides a uniform interface for querying the properties of iterators
//! at compile-time. You can specialize cuda::std::iterator_traits for your own iterator types if needed.
//! deprecated [Since 3.0]
template <typename T>
using iterator_traits CCCL_DEPRECATED_BECAUSE("Use cuda::std::iterator_traits instead") =
// FIXME(bgruber): switching to ::cuda::std::iterator_traits<T> breaks some tests, e.g. cub.test.device_merge_sort.lid_1
#if _CCCL_COMPILER(NVRTC)
  ::cuda
#endif // _CCCL_COMPILER(NVRTC)
  ::std::iterator_traits<T>;

_CCCL_SUPPRESS_DEPRECATED_PUSH

// value

//! deprecated [Since 3.0]
template <typename Iterator>
struct CCCL_DEPRECATED_BECAUSE("Use cuda::std::iterator_traits<>::value_type or cuda::std::iter_value_t instead")
  iterator_value
{
  using type = typename iterator_traits<Iterator>::value_type;
};

//! deprecated [Since 3.0]
template <typename Iterator>
using iterator_value_t CCCL_DEPRECATED_BECAUSE("Use cuda::std::iterator_traits<>::value_type or "
                                               "cuda::std::iter_value_t instead") = iterator_value<Iterator>;

// pointer

//! deprecated [Since 3.0]
template <typename Iterator>
struct CCCL_DEPRECATED iterator_pointer
{
  using type = typename iterator_traits<Iterator>::pointer;
};

//! deprecated [Since 3.0]
template <typename Iterator>
using iterator_pointer_t CCCL_DEPRECATED = typename iterator_pointer<Iterator>::type;

// reference

//! deprecated [Since 3.0]
template <typename Iterator>
struct CCCL_DEPRECATED_BECAUSE("Use cuda::std::iterator_traits<>::reference or cuda::std::iter_reference_t instead")
  iterator_reference
{
  using type = typename iterator_traits<Iterator>::reference;
};

//! deprecated [Since 3.0]
template <typename Iterator>
using iterator_reference_t CCCL_DEPRECATED_BECAUSE(
  "Use cuda::std::iterator_traits<>::reference or "
  "cuda::std::iter_reference_t instead") = typename iterator_reference<Iterator>::type;

// difference

//! deprecated [Since 3.0]
template <typename Iterator>
struct CCCL_DEPRECATED_BECAUSE("Use cuda::std::iterator_traits<>::difference_t or cuda::std::iter_difference_t instead")
  iterator_difference
{
  using type = typename iterator_traits<Iterator>::difference_type;
};

//! deprecated [Since 3.0]
template <typename Iterator>
using iterator_difference_t CCCL_DEPRECATED_BECAUSE(
  "Use cuda::std::iterator_traits<>::difference_t or "
  "cuda::std::iter_difference_t instead") = typename iterator_difference<Iterator>::type;

// traversal

template <typename Iterator>
struct iterator_traversal
    : detail::iterator_category_to_traversal<typename iterator_traits<Iterator>::iterator_category>
{};

template <typename Iterator>
using iterator_traversal_t = typename iterator_traversal<Iterator>::type;

// system

namespace detail
{
template <typename Iterator, typename = void>
struct iterator_system_impl
{};

template <typename Iterator>
struct iterator_system_impl<Iterator, ::cuda::std::void_t<typename iterator_traits<Iterator>::iterator_category>>
    : iterator_category_to_system<typename iterator_traits<Iterator>::iterator_category>
{};
} // namespace detail

_CCCL_SUPPRESS_DEPRECATED_POP

template <typename Iterator>
struct iterator_system : detail::iterator_system_impl<Iterator>
{};

// specialize iterator_system for void *, which has no category
template <>
struct iterator_system<void*> : iterator_system<int*>
{};

template <>
struct iterator_system<const void*> : iterator_system<const int*>
{};

template <typename Iterator>
using iterator_system_t = typename iterator_system<Iterator>::type;

// specialize the respective cuda iterators
template <>
struct iterator_system<::cuda::discard_iterator>
{
  using type = any_system_tag;
};
template <>
struct iterator_traversal<::cuda::discard_iterator>
{
  using type = random_access_traversal_tag;
};

template <class T, class Index>
struct iterator_system<::cuda::constant_iterator<T, Index>>
{
  using type = any_system_tag;
};
template <class T, class Index>
struct iterator_traversal<::cuda::constant_iterator<T, Index>>
{
  using type = random_access_traversal_tag;
};

template <class Start>
struct iterator_system<::cuda::counting_iterator<Start>>
{
  using type = any_system_tag;
};
template <class Start>
struct iterator_traversal<::cuda::counting_iterator<Start>>
{
  using type = random_access_traversal_tag;
};

template <class Iter, class Offset>
struct iterator_system<::cuda::permutation_iterator<Iter, Offset>>
{
  using type = detail::minimum_system_t<iterator_system_t<Iter>, iterator_system_t<Offset>>;
};
template <class Iter, class Offset>
struct iterator_traversal<::cuda::permutation_iterator<Iter, Offset>>
{
  using type = random_access_traversal_tag;
};

template <class Iter, class Stride>
struct iterator_system<::cuda::strided_iterator<Iter, Stride>> : iterator_system<Iter>
{};
template <class Iter, class Stride>
struct iterator_traversal<::cuda::strided_iterator<Iter, Stride>> : iterator_traversal<Iter>
{};

template <class Fn, class Index>
struct iterator_system<::cuda::tabulate_output_iterator<Fn, Index>>
{
  using type = any_system_tag;
};
template <class Fn, class Index>
struct iterator_traversal<::cuda::tabulate_output_iterator<Fn, Index>>
{
  using type = random_access_traversal_tag;
};

template <class Fn, class Iter>
struct iterator_system<::cuda::transform_output_iterator<Fn, Iter>> : iterator_system<Iter>
{};
template <class Fn, class Iter>
struct iterator_traversal<::cuda::transform_output_iterator<Fn, Iter>> : iterator_traversal<Iter>
{};

template <class Fn, class Iter>
struct iterator_system<::cuda::transform_iterator<Fn, Iter>> : iterator_system<Iter>
{};
template <class Fn, class Iter>
struct iterator_traversal<::cuda::transform_iterator<Fn, Iter>> : iterator_traversal<Iter>
{};

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/iterator_traversal_tags.h>
