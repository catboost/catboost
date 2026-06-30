/*
 *  Copyright 2018 NVIDIA Corporation
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
#include <thrust/detail/alignment.h>
#include <thrust/detail/execute_with_allocator_fwd.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

namespace mr
{

template <typename T, class MR>
class allocator;

}

namespace detail
{

template <template <typename> class ExecutionPolicyCRTPBase>
struct allocator_aware_execution_policy
{
  template <typename MemoryResource>
  struct execute_with_memory_resource_type
  {
    using type =
      thrust::detail::execute_with_allocator<thrust::mr::allocator<thrust::detail::max_align_t, MemoryResource>,
                                             ExecutionPolicyCRTPBase>;
  };

  template <typename Allocator>
  struct execute_with_allocator_type
  {
    using type = thrust::detail::execute_with_allocator<Allocator, ExecutionPolicyCRTPBase>;
  };

  _CCCL_EXEC_CHECK_DISABLE
  template <typename MemoryResource>
  _CCCL_HOST_DEVICE typename execute_with_memory_resource_type<MemoryResource>::type
  operator()(MemoryResource* mem_res) const
  {
    return typename execute_with_memory_resource_type<MemoryResource>::type(mem_res);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Allocator>
  _CCCL_HOST_DEVICE typename execute_with_allocator_type<Allocator&>::type operator()(Allocator& alloc) const
  {
    return typename execute_with_allocator_type<Allocator&>::type(alloc);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Allocator>
  _CCCL_HOST_DEVICE typename execute_with_allocator_type<Allocator>::type operator()(const Allocator& alloc) const
  {
    return typename execute_with_allocator_type<Allocator>::type(alloc);
  }

  // just the rvalue overload
  // perfect forwarding doesn't help, because a const reference has to be turned
  // into a value by copying for the purpose of storing it in execute_with_allocator
  _CCCL_EXEC_CHECK_DISABLE
  template <typename Allocator,
            typename ::cuda::std::enable_if<!::cuda::std::is_lvalue_reference<Allocator>::value>::type* = nullptr>
  _CCCL_HOST_DEVICE typename execute_with_allocator_type<Allocator>::type operator()(Allocator&& alloc) const
  {
    return typename execute_with_allocator_type<Allocator>::type(::cuda::std::move(alloc));
  }
};

} // end namespace detail

THRUST_NAMESPACE_END
