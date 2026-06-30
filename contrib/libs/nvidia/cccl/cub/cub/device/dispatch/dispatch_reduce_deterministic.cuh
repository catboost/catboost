/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file This file device-wide, parallel operations for
 *       computing a reduction across a sequence of data items residing within
 *       device-accessible memory. Current reduction operator supported is ::cuda::std::plus
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

#include <cub/agent/agent_reduce.cuh>
#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/functional>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace rfa
{

template <typename Invocable, typename InputT>
using transformed_input_t = _CUDA_VSTD::decay_t<typename _CUDA_VSTD::__invoke_of<Invocable, InputT>::type>;

template <typename InitT, typename InputIteratorT, typename TransformOpT>
using accum_t =
  _CUDA_VSTD::__accumulator_t<_CUDA_VSTD::plus<>, InitT, transformed_input_t<TransformOpT, it_value_t<InputIteratorT>>>;

template <typename FloatType                                                              = float,
          typename ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<FloatType>>* = nullptr>
struct deterministic_sum_t
{
  using DeterministicAcc = detail::rfa::ReproducibleFloatingAccumulator<FloatType>;

  _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc acc, FloatType f)
  {
    acc += f;
    return acc;
  }

  _CCCL_DEVICE DeterministicAcc operator()(FloatType f, DeterministicAcc acc)
  {
    return this->operator()(acc, f);
  }

  _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc lhs, DeterministicAcc rhs)
  {
    DeterministicAcc rtn = lhs;
    rtn += rhs;
    return rtn;
  }

  _CCCL_DEVICE FloatType operator()(FloatType lhs, FloatType rhs)
  {
    return lhs + rhs;
  }
};

} // namespace rfa

/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction in deterministic fashion
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam InitT
 *   Initial value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename InitT,
          typename TransformOpT = ::cuda::std::identity,
          typename AccumT       = rfa::accum_t<InitT, InputIteratorT, TransformOpT>,
          typename PolicyHub    = detail::rfa::policy_hub<AccumT, OffsetT, ::cuda::std::plus<>>>
struct DispatchReduceDeterministic
{
  using deterministic_add_t = rfa::deterministic_sum_t<AccumT>;
  using reduction_op_t      = deterministic_add_t;

  using deterministic_accum_t = typename deterministic_add_t::DeterministicAcc;
  using input_unwrapped_it_t  = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;

  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Unwrapped Pointer to the input sequence of data items
  input_unwrapped_it_t d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  reduction_op_t reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  TransformOpT transform_op = {};

  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core deterministically
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel)
  {
    // Return if the caller is simply requesting the size of the storage
    // allocation
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }
// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
            (long long) stream,
            ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif
    // Invoke single_reduce_sweep_kernel
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
      .doit(single_tile_kernel, d_in, d_out, num_items, reduction_op, init, transform_op);
    // Check for failure to launch
    auto error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }
    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
    return cudaSuccess;
  }

  //---------------------------------------------------------------------------
  // Normal problem size invocation (two-pass)
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke two-passes to reduce deteerministically
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeterministicDeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeterministicDeviceReduceKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename ReduceKernelT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel, SingleTileKernelT single_tile_kernel)
  {
    const auto tile_size = ActivePolicyT::ReducePolicy::BLOCK_THREADS * ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD;
    // Get device ordinal
    int device_ordinal;
    auto error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    int sm_count;
    error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
    if (cudaSuccess != error)
    {
      return error;
    }

    KernelConfig reduce_config;
    error = CubDebug(reduce_config.Init<typename ActivePolicyT::ReducePolicy>(reduce_kernel));
    if (cudaSuccess != error)
    {
      return error;
    }

    const int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
    const int max_blocks              = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);
    const int resulting_grid_size     = (num_items + tile_size - 1) / tile_size;

    // Get grid size for device_reduce_sweep_kernel
    const int reduce_grid_size = resulting_grid_size > max_blocks ? max_blocks : resulting_grid_size;

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {
      reduce_grid_size * sizeof(deterministic_accum_t) // bytes needed for privatized block
                                                       // reductions
    };

    // Alias the temporary allocations from the single storage blob (or
    // compute the necessary size of the blob)
    error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      return cudaSuccess;
    }

    // Alias the allocation for the privatized per-block reductions
    deterministic_accum_t* d_block_reductions = (deterministic_accum_t*) allocations[0];

    // Log device_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking DeterministicDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
            "per thread, %d SM occupancy\n",
            reduce_grid_size,
            ActivePolicyT::ReducePolicy::BLOCK_THREADS,
            (long long) stream,
            ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD,
            reduce_config.sm_occupancy);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
      reduce_grid_size, ActivePolicyT::ReducePolicy::BLOCK_THREADS, 0, stream)
      .doit(reduce_kernel, d_in, d_block_reductions, num_items, reduction_op, transform_op, reduce_grid_size);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
            (long long) stream,
            ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

    // Invoke DeterministicDeviceReduceSingleTileKernel
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
      .doit(single_tile_kernel,
            d_block_reductions,
            d_out,
            reduce_grid_size, // triple_chevron is not type safe, make sure to use int
            reduction_op,
            init,
            ::cuda::std::identity{});

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }

    return cudaSuccess;
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation Deterministic
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using SingleTilePolicyT = typename ActivePolicyT::SingleTilePolicy;
    using MaxPolicyT        = typename PolicyHub::MaxPolicy;

    // Force kernel code-generation in all compiler passes
    if (num_items <= (SingleTilePolicyT::BLOCK_THREADS * SingleTilePolicyT::ITEMS_PER_THREAD))
    {
      return InvokeSingleTile<ActivePolicyT>(
        detail::reduce::DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          input_unwrapped_it_t,
          OutputIteratorT,
          OffsetT,
          reduction_op_t,
          InitT,
          deterministic_accum_t,
          TransformOpT>);
    }
    else
    {
      return InvokePasses<ActivePolicyT>(
        detail::reduce::DeterministicDeviceReduceKernel<
          MaxPolicyT,
          input_unwrapped_it_t,
          OffsetT,
          reduction_op_t,
          deterministic_accum_t,
          TransformOpT>,
        detail::reduce::DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          deterministic_accum_t*,
          OutputIteratorT,
          int, // Always used with int offsets
          reduction_op_t,
          InitT,
          deterministic_accum_t>);
    }
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide deterministic reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    InitT init                = {},
    cudaStream_t stream       = {},
    TransformOpT transform_op = {})
  {
    static_assert(sizeof(OffsetT) <= 4, "OffsetT must be 4 bytes or less for deterministic reduction");

    cudaError error = cudaSuccess;

    // Get PTX version
    int ptx_version = 0;
    error           = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    input_unwrapped_it_t d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

    // Create dispatch functor
    DispatchReduceDeterministic dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_in_unwrapped,
      d_out,
      num_items,
      deterministic_add_t{},
      init,
      stream,
      ptx_version,
      transform_op};

    // Dispatch to chained policy
    error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
    return error;
  }
};
} // namespace detail
CUB_NAMESPACE_END
