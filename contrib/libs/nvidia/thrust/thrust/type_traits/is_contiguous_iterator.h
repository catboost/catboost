/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file is_contiguous_iterator.h
 *  \brief An extensible type trait for determining if an iterator satisifies
 *         the <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *         requirements (e.g. is pointer-like).
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <iterator>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC && _MSC_VER < 1916 // MSVC 2017 version 15.9
  #include <vector>
  #include <string>
  #include <array>

  #if THRUST_CPP_DIALECT >= 2017
    #include <string_view>
  #endif
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename Iterator>
struct is_contiguous_iterator_impl;

} // namespace detail

/// Unary metafunction returns \c true_type if \c Iterator satisfies
/// <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
/// e.g. it points to elements that are contiguous in memory, and \c false_type
/// otherwise.
template <typename Iterator>
#if THRUST_CPP_DIALECT >= 2011
using is_contiguous_iterator =
#else
struct is_contiguous_iterator :
#endif
  detail::is_contiguous_iterator_impl<Iterator>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c Iterator satisfies
/// <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
/// e.g. it points to elements that are contiguous in memory, and \c false
/// otherwise.
template <typename Iterator>
constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<Iterator>::value;
#endif

/// Customization point that can be customized to indicate that an iterator
/// type \c Iterator satisfies
/// <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
/// e.g. it points to elements that are contiguous in memory.
template <typename Iterator>
struct proclaim_contiguous_iterator : false_type {};

/// Declares that the iterator \c Iterator is
/// <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
/// by specializing `thrust::proclaim_contiguous_iterator`.
#define THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)                         \
  THRUST_NAMESPACE_BEGIN                                                      \
  template <>                                                                 \
  struct proclaim_contiguous_iterator<Iterator>                               \
      : THRUST_NS_QUALIFIER::true_type {};                                    \
  THRUST_NAMESPACE_END                                                        \
  /**/

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

template <typename Iterator>
struct is_libcxx_wrap_iter : false_type {};

#if defined(_LIBCPP_VERSION)
template <typename Iterator>
struct is_libcxx_wrap_iter<
  _VSTD::__wrap_iter<Iterator>
> : true_type {};
#endif

template <typename Iterator>
struct is_libstdcxx_normal_iterator : false_type {};

#if defined(__GLIBCXX__)
template <typename Iterator, typename Container>
struct is_libstdcxx_normal_iterator<
  ::__gnu_cxx::__normal_iterator<Iterator, Container>
> : true_type {};
#endif

#if   _MSC_VER >= 1916 // MSVC 2017 version 15.9.
template <typename Iterator>
struct is_msvc_contiguous_iterator
  : is_pointer<::std::_Unwrapped_t<Iterator> > {};
#elif _MSC_VER >= 1700 // MSVC 2012.
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
  ::std::_Vector_const_iterator<Vector>
> : true_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
  ::std::_Vector_iterator<Vector>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
  ::std::_String_const_iterator<String>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
  ::std::_String_iterator<String>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
  ::std::_Array_const_iterator<T, N>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
  ::std::_Array_iterator<T, N>
> : true_type {};

#if THRUST_CPP_DIALECT >= 2017
template <typename Traits>
struct is_msvc_contiguous_iterator<
  ::std::_String_view_iterator<Traits>
> : true_type {};
#endif
#else
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};
#endif


template <typename Iterator>
struct is_contiguous_iterator_impl
  : integral_constant<
      bool
    ,    is_pointer<Iterator>::value
      || is_thrust_pointer<Iterator>::value
      || is_libcxx_wrap_iter<Iterator>::value
      || is_libstdcxx_normal_iterator<Iterator>::value
      || is_msvc_contiguous_iterator<Iterator>::value
      || proclaim_contiguous_iterator<Iterator>::value
    >
{};

} // namespace detail

THRUST_NAMESPACE_END

