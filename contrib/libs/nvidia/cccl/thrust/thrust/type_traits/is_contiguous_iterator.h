/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

/*! \file
 *  \brief An extensible type trait for determining if an iterator satisfies the
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  requirements (aka is pointer-like).
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
#include <thrust/detail/type_traits/is_thrust_pointer.h>

#include <cuda/std/iterator>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */
/*! \brief Customization point that can be customized to indicate that an
 *  iterator type \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory.
 *
 * \see is_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
struct proclaim_contiguous_iterator : false_type
{};

/*! \brief Declares that the iterator \c Iterator is
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  by specializing \c proclaim_contiguous_iterator.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 */
#define THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)                            \
  THRUST_NAMESPACE_BEGIN                                                         \
  template <>                                                                    \
  struct proclaim_contiguous_iterator<Iterator> : THRUST_NS_QUALIFIER::true_type \
  {};                                                                            \
  THRUST_NAMESPACE_END                                                           \
  /**/

/*! \cond
 */

namespace detail
{
template <typename Iterator>
inline constexpr bool is_libcxx_wrap_iter_v = false;

#if defined(_LIBCPP_VERSION)
template <typename Iterator>
inline constexpr bool is_libcxx_wrap_iter_v<
#  if _LIBCPP_VERSION < 14000
  _VSTD::__wrap_iter<Iterator>
#  else
  std::__wrap_iter<Iterator>
#  endif
  > = true;
#endif

template <typename Iterator>
inline constexpr bool is_libstdcxx_normal_iterator_v = false;

#if defined(__GLIBCXX__)
template <typename Iterator, typename Container>
inline constexpr bool is_libstdcxx_normal_iterator_v<::__gnu_cxx::__normal_iterator<Iterator, Container>> = true;
#endif

#if _CCCL_COMPILER(MSVC)
template <typename Iterator>
inline constexpr bool is_msvc_contiguous_iterator_v = ::cuda::std::is_pointer_v<::std::remove_reference_t<::std::remove_cv_t<Iterator>>>;
#else
template <typename Iterator>
inline constexpr bool is_msvc_contiguous_iterator_v = false;
#endif

template <typename Iterator>
inline constexpr bool is_contiguous_iterator_impl_v =
  ::cuda::std::contiguous_iterator<Iterator> || is_thrust_pointer_v<Iterator> || is_libcxx_wrap_iter_v<Iterator>
  || is_libstdcxx_normal_iterator_v<Iterator> || is_msvc_contiguous_iterator_v<Iterator>
  || proclaim_contiguous_iterator<Iterator>::value;

} // namespace detail

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false_type
 *  otherwise.
 *
 * \see is_contiguous_iterator_v
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
using is_contiguous_iterator =
  ::cuda::std::bool_constant<detail::is_contiguous_iterator_impl_v<::cuda::std::remove_cvref_t<Iterator>>>;

/*! \brief <tt>constexpr bool</tt> that is \c true if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false
 *  otherwise.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
constexpr bool is_contiguous_iterator_v = detail::is_contiguous_iterator_impl_v<::cuda::std::remove_cvref_t<Iterator>>;

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
