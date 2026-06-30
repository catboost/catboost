/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace merge
{
template <typename KeyT, typename ValueT>
struct policy_hub
{
  static constexpr bool has_values = !::cuda::std::is_same_v<ValueT, NullType>;

  using tune_type = char[has_values ? sizeof(KeyT) + sizeof(ValueT) : sizeof(KeyT)];

  struct policy500 : ChainedPolicy<500, policy500, policy500>
  {
    using merge_policy =
      agent_policy_t<256,
                     Nominal4BItemsToItems<tune_type>(11),
                     BLOCK_LOAD_WARP_TRANSPOSE,
                     LOAD_LDG,
                     BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy520 : ChainedPolicy<520, policy520, policy500>
  {
    using merge_policy =
      agent_policy_t<512,
                     Nominal4BItemsToItems<tune_type>(13),
                     BLOCK_LOAD_WARP_TRANSPOSE,
                     LOAD_LDG,
                     BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy600 : ChainedPolicy<600, policy600, policy520>
  {
    using merge_policy =
      agent_policy_t<512,
                     Nominal4BItemsToItems<tune_type>(15),
                     BLOCK_LOAD_WARP_TRANSPOSE,
                     LOAD_DEFAULT,
                     BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using max_policy = policy600;
};
} // namespace merge
} // namespace detail

CUB_NAMESPACE_END
