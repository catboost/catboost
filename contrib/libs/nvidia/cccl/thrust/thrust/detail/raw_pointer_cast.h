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
#include <thrust/detail/type_traits/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename Pointer>
_CCCL_HOST_DEVICE typename thrust::detail::pointer_traits<Pointer>::raw_pointer raw_pointer_cast(Pointer ptr)
{
  return thrust::detail::pointer_traits<Pointer>::get(ptr);
}

template <typename ToPointer, typename FromPointer>
_CCCL_HOST_DEVICE ToPointer reinterpret_pointer_cast(FromPointer ptr)
{
  using to_element = typename thrust::detail::pointer_element<ToPointer>::type;
  return ToPointer(reinterpret_cast<to_element*>(thrust::raw_pointer_cast(ptr)));
}

template <typename ToPointer, typename FromPointer>
_CCCL_HOST_DEVICE ToPointer static_pointer_cast(FromPointer ptr)
{
  using to_element = typename thrust::detail::pointer_element<ToPointer>::type;
  return ToPointer(static_cast<to_element*>(thrust::raw_pointer_cast(ptr)));
}

THRUST_NAMESPACE_END
