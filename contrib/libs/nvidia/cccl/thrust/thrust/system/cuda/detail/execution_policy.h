/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/system/cuda/config.h>

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/version.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{

struct tag;

template <class>
struct execution_policy;

template <>
struct execution_policy<tag> : thrust::execution_policy<tag>
{
  using tag_type = tag;
};

struct tag
    : execution_policy<tag>
    , thrust::detail::allocator_aware_execution_policy<cuda_cub::execution_policy>
{};

template <class Derived>
struct execution_policy : thrust::execution_policy<Derived>
{
  using tag_type = tag;
  _CCCL_HOST_DEVICE operator tag() const
  {
    return tag();
  }
};

} // namespace cuda_cub

namespace system
{
namespace cuda
{
namespace detail
{

using thrust::cuda_cub::execution_policy;
using thrust::cuda_cub::tag;

} // namespace detail
} // namespace cuda
} // namespace system

namespace system
{
namespace cuda
{

using thrust::cuda_cub::execution_policy;
using thrust::cuda_cub::tag;

} // namespace cuda
} // namespace system

namespace cuda
{

using thrust::cuda_cub::execution_policy;
using thrust::cuda_cub::tag;

} // namespace cuda

THRUST_NAMESPACE_END
