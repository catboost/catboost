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

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{

template <typename PolicyT, typename = void>
struct MergeSortPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION MergeSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct MergeSortPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::MergeSortPolicy::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION MergeSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(MergeSort);
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION MergeSortPolicyWrapper<PolicyT> MakeMergeSortPolicyWrapper(PolicyT policy)
{
  return MergeSortPolicyWrapper<PolicyT>{policy};
}

template <typename KeyIteratorT>
struct policy_hub
{
  using KeyT = it_value_t<KeyIteratorT>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  // NVBug 3384810
#if defined(_NVHPC_CUDA)
  using Policy520 = Policy500;
#else
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<512,
                           Nominal4BItemsToItems<KeyT>(15),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };
#endif

  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(17),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_DEFAULT,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy600;
};

} // namespace detail::merge_sort

CUB_NAMESPACE_END
