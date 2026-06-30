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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// Type traits for contiguous iterators:
template <typename Iterator>
struct contiguous_iterator_traits
{
  static_assert(thrust::is_contiguous_iterator_v<Iterator>,
                "contiguous_iterator_traits requires a contiguous iterator.");

  using raw_pointer =
    typename thrust::detail::pointer_traits<decltype(&*::cuda::std::declval<Iterator>())>::raw_pointer;
};
} // namespace detail

//! Converts a contiguous iterator type to its underlying raw pointer type.
template <typename ContiguousIterator>
using unwrap_contiguous_iterator_t = typename detail::contiguous_iterator_traits<ContiguousIterator>::raw_pointer;

//! Converts a contiguous iterator to its underlying raw pointer.
template <typename ContiguousIterator>
_CCCL_HOST_DEVICE auto unwrap_contiguous_iterator(ContiguousIterator it)
  -> unwrap_contiguous_iterator_t<ContiguousIterator>
{
  static_assert(thrust::is_contiguous_iterator_v<ContiguousIterator>,
                "unwrap_contiguous_iterator called with non-contiguous iterator.");
  return thrust::raw_pointer_cast(&*it);
}

namespace detail
{
// Implementation for non-contiguous iterators -- passthrough.
template <typename Iterator, bool IsContiguous = thrust::is_contiguous_iterator_v<Iterator>>
struct try_unwrap_contiguous_iterator_impl
{
  using type = Iterator;

  static _CCCL_HOST_DEVICE type get(Iterator it)
  {
    return it;
  }
};

// Implementation for contiguous iterators -- unwraps to raw pointer.
template <typename Iterator>
struct try_unwrap_contiguous_iterator_impl<Iterator, true /*is_contiguous*/>
{
  using type = unwrap_contiguous_iterator_t<Iterator>;

  static _CCCL_HOST_DEVICE type get(Iterator it)
  {
    return unwrap_contiguous_iterator(it);
  }
};
} // namespace detail

//! Takes an iterator type and, if it is contiguous, yields the raw pointer type it represents. Otherwise returns the
//! iterator type unmodified.
template <typename Iterator>
using try_unwrap_contiguous_iterator_t = typename detail::try_unwrap_contiguous_iterator_impl<Iterator>::type;

//! Takes an iterator and, if it is contiguous, unwraps it to the raw pointer it represents. Otherwise returns the
//! iterator unmodified.
template <typename Iterator>
_CCCL_HOST_DEVICE auto try_unwrap_contiguous_iterator(Iterator it) -> try_unwrap_contiguous_iterator_t<Iterator>
{
  return detail::try_unwrap_contiguous_iterator_impl<Iterator>::get(it);
}

THRUST_NAMESPACE_END
