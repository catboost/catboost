/******************************************************************************
 * Copyright (c) 2023-24, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * This file contains facilities that help to prevent exceeding the available shared memory per thread block
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_arch.cuh>
#include <cub/util_policy_wrapper_t.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/discard_memory>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{

/**
 * @brief Helper struct to wrap all the information needed to implement virtual shared memory that's passed to a kernel.
 *
 */
struct vsmem_t
{
  void* gmem_ptr;
};

/**
 * @brief Class template that helps to prevent exceeding the available shared memory per thread block.
 *
 * @tparam AgentT The agent for which we check whether per-thread block shared memory is sufficient or whether virtual
 * shared memory is needed.
 */
template <typename AgentT>
class vsmem_helper_impl
{
private:
  // Per-block virtual shared memory may be padded to make sure vsmem is an integer multiple of `line_size`
  static constexpr ::cuda::std::size_t line_size = 128;

  // The amount of shared memory or virtual shared memory required by the algorithm's agent
  static constexpr ::cuda::std::size_t required_smem = sizeof(typename AgentT::TempStorage);

  // Whether we need to allocate global memory-backed virtual shared memory
  static constexpr bool needs_vsmem = required_smem > max_smem_per_block;

  // Padding bytes to an integer multiple of `line_size`. Only applies to virtual shared memory
  static constexpr ::cuda::std::size_t padding_bytes =
    (required_smem % line_size == 0) ? 0 : (line_size - (required_smem % line_size));

public:
  // Type alias to be used for static temporary storage declaration within the algorithm's kernel
  using static_temp_storage_t = ::cuda::std::_If<needs_vsmem, cub::NullType, typename AgentT::TempStorage>;

  // The amount of global memory-backed virtual shared memory needed, padded to an integer multiple of 128 bytes
  static constexpr ::cuda::std::size_t vsmem_per_block = needs_vsmem ? (required_smem + padding_bytes) : 0;

  /**
   * @brief Used from within the device algorithm's kernel to get the temporary storage that can be
   * passed to the agent, specialized for the case when we can use native shared memory as temporary
   * storage.
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE typename AgentT::TempStorage&
  get_temp_storage(typename AgentT::TempStorage& static_temp_storage, vsmem_t&)
  {
    return static_temp_storage;
  }

  /**
   * @brief Used from within the device algorithm's kernel to get the temporary storage that can be
   * passed to the agent, specialized for the case when we can use native shared memory as temporary
   * storage and taking a linear block id.
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE typename AgentT::TempStorage&
  get_temp_storage(typename AgentT::TempStorage& static_temp_storage, vsmem_t&, ::cuda::std::size_t)
  {
    return static_temp_storage;
  }

  /**
   * @brief Used from within the device algorithm's kernel to get the temporary storage that can be
   * passed to the agent, specialized for the case when we have to use global memory-backed
   * virtual shared memory as temporary storage.
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE typename AgentT::TempStorage&
  get_temp_storage(cub::NullType& static_temp_storage, vsmem_t& vsmem)
  {
    return *reinterpret_cast<typename AgentT::TempStorage*>(
      static_cast<char*>(vsmem.gmem_ptr) + (vsmem_per_block * blockIdx.x));
  }

  /**
   * @brief Used from within the device algorithm's kernel to get the temporary storage that can be
   * passed to the agent, specialized for the case when we have to use global memory-backed
   * virtual shared memory as temporary storage and taking a linear block id.
   */
  static _CCCL_DEVICE _CCCL_FORCEINLINE typename AgentT::TempStorage&
  get_temp_storage(cub::NullType& static_temp_storage, vsmem_t& vsmem, ::cuda::std::size_t linear_block_id)
  {
    return *reinterpret_cast<typename AgentT::TempStorage*>(
      static_cast<char*>(vsmem.gmem_ptr) + (vsmem_per_block * linear_block_id));
  }

  /**
   * @brief Hints to discard modified cache lines of the used virtual shared memory.
   * modified cache lines.
   *
   * @note Needs to be followed by `__syncthreads()` if the function returns true and the virtual shared memory is
   * supposed to be reused after this function call.
   */
  template <bool needs_vsmem_ = needs_vsmem, ::cuda::std::enable_if_t<!needs_vsmem_, int> = 0>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool discard_temp_storage(typename AgentT::TempStorage& temp_storage)
  {
    return false;
  }

  /**
   * @brief Hints to discard modified cache lines of the used virtual shared memory.
   * modified cache lines.
   *
   * @note Needs to be followed by `__syncthreads()` if the function returns true and the virtual shared memory is
   * supposed to be reused after this function call.
   */
  template <bool needs_vsmem_ = needs_vsmem, ::cuda::std::enable_if_t<needs_vsmem_, int> = 0>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool discard_temp_storage(typename AgentT::TempStorage& temp_storage)
  {
    // Ensure all threads finished using temporary storage
    __syncthreads();

    const ::cuda::std::size_t linear_tid   = threadIdx.x;
    const ::cuda::std::size_t block_stride = line_size * blockDim.x;

    char* ptr    = reinterpret_cast<char*>(&temp_storage);
    auto ptr_end = ptr + vsmem_per_block;

    // 128 byte-aligned virtual shared memory discard
    for (auto thread_ptr = ptr + (linear_tid * line_size); thread_ptr < ptr_end; thread_ptr += block_stride)
    {
      ::cuda::discard_memory(thread_ptr, line_size);
    }
    return true;
  }
};

template <class DefaultAgentT, class FallbackAgentT>
_CCCL_HOST_DEVICE constexpr bool use_fallback_agent()
{
  return (sizeof(typename DefaultAgentT::TempStorage) > max_smem_per_block)
      && (sizeof(typename FallbackAgentT::TempStorage) <= max_smem_per_block);
}

/**
 * @brief Class template that helps to prevent exceeding the available shared memory per thread block with two measures:
 * (1) If an agent's `TempStorage` declaration exceeds the maximum amount of shared memory per thread block, we check
 * whether using a fallback policy, e.g., with a smaller tile size, would fit into shared memory.
 * (2) If the fallback still doesn't fit into shared memory, we make use of virtual shared memory that is backed by
 * global memory.
 *
 * @tparam DefaultAgentPolicyT The default tuning policy that is used if the default agent's shared memory requirements
 * fall within the bounds of `max_smem_per_block` or when virtual shared memory is needed
 * @tparam DefaultAgentT The default agent, instantiated with the given default tuning policy
 * @tparam FallbackAgentPolicyT A fallback tuning policy that may exhibit lower shared memory requirements, e.g., by
 * using a smaller tile size, than the default. This fallback policy is used if and only if the shared memory
 * requirements of the default agent exceed `max_smem_per_block`, yet the shared memory requirements of the fallback
 * agent falls within the bounds of `max_smem_per_block`.
 * @tparam FallbackAgentT The fallback agent, instantiated with the given fallback tuning policy
 */
template <typename DefaultAgentPolicyT,
          typename DefaultAgentT,
          typename FallbackAgentPolicyT = DefaultAgentPolicyT,
          typename FallbackAgentT       = DefaultAgentT,
          bool UseFallbackPolicy        = use_fallback_agent<DefaultAgentT, FallbackAgentT>()>
struct vsmem_helper_with_fallback_impl : public vsmem_helper_impl<DefaultAgentT>
{
  using agent_t        = DefaultAgentT;
  using agent_policy_t = DefaultAgentPolicyT;
};
template <typename DefaultAgentPolicyT, typename DefaultAgentT, typename FallbackAgentPolicyT, typename FallbackAgentT>
struct vsmem_helper_with_fallback_impl<DefaultAgentPolicyT, DefaultAgentT, FallbackAgentPolicyT, FallbackAgentT, true>
    : public vsmem_helper_impl<FallbackAgentT>
{
  using agent_t        = FallbackAgentT;
  using agent_policy_t = FallbackAgentPolicyT;
};

/**
 * @brief Alias template for the `vsmem_helper_with_fallback_impl` that instantiates the given AgentT template with the
 * respective policy as first template parameter, followed by the parameters captured by the `AgentParamsT` template
 * parameter pack.
 */
template <typename DefaultPolicyT, typename FallbackPolicyT, template <typename...> class AgentT, typename... AgentParamsT>
using vsmem_helper_fallback_policy_t =
  vsmem_helper_with_fallback_impl<DefaultPolicyT,
                                  AgentT<DefaultPolicyT, AgentParamsT...>,
                                  FallbackPolicyT,
                                  AgentT<FallbackPolicyT, AgentParamsT...>>;

/**
 * @brief Alias template for the `vsmem_helper_t` by using a simple fallback policy that uses `DefaultPolicyT` as basis,
 * overwriting `64` threads per block and `1` item per thread.
 */
template <typename DefaultPolicyT, template <typename...> class AgentT, typename... AgentParamsT>
using vsmem_helper_default_fallback_policy_t =
  vsmem_helper_fallback_policy_t<DefaultPolicyT, policy_wrapper_t<DefaultPolicyT, 64, 1>, AgentT, AgentParamsT...>;

} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
