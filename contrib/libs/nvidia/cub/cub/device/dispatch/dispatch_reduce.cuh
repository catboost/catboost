/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * @file cub::DeviceReduce provides device-wide, parallel operations for 
 *       computing a reduction across a sequence of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <stdio.h>
#include <iterator>

#include <cub/agent/agent_reduce.cuh>
#include <cub/config.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Reduce region kernel entry point (multi-block). Computes privatized 
 *        reductions, one per thread block.
 *
 * @tparam ChainedPolicyT 
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items \iterator
 * 
 * @tparam OffsetT
 *   Signed integer type for global offsets
 * 
 * @tparam ReductionOpT
 *   Binary reduction functor type having member 
 *   `auto operator()(const T &a, const U &b)`
 * 
 * @tparam InitT
 *   Initial value type
 * 
 * @tparam AccumT
 *   Accumulator type
 * 
 * @param[in] d_in 
 *   Pointer to the input sequence of data items
 * 
 * @param[out] d_out 
 *   Pointer to the output aggregate
 * 
 * @param[in] num_items 
 *   Total number of input data items
 * 
 * @param[in] even_share 
 *   Even-share descriptor for mapping an equal number of tiles onto each 
 *   thread block
 * 
 * @param[in] reduction_op 
 *   Binary reduction functor
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) 
__global__ void DeviceReduceKernel(InputIteratorT d_in,
                                   AccumT* d_out,
                                   OffsetT num_items,
                                   GridEvenShare<OffsetT> even_share,
                                   ReductionOpT reduction_op)
{
  // Thread block type for reducing input tiles
  using AgentReduceT =
    AgentReduce<typename ChainedPolicyT::ActivePolicy::ReducePolicy,
                InputIteratorT,
                AccumT*,
                OffsetT,
                ReductionOpT,
                AccumT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // Consume input tiles
  AccumT block_aggregate =
    AgentReduceT(temp_storage, d_in, reduction_op).ConsumeTiles(even_share);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy(d_out + blockIdx.x, block_aggregate);
  }
}

/**
 * @brief Reduce a single tile kernel entry point (single-block). Can be used 
 *        to aggregate privatized thread block reductions from a previous 
 *        multi-block reduction pass.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items \iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate \iterator
 * 
 * @tparam OffsetT
 *   Signed integer type for global offsets
 * 
 * @tparam ReductionOpT
 *   Binary reduction functor type having member 
 *   `T operator()(const T &a, const U &b)`
 * 
 * @tparam InitT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type 
 *
 * @param[in] d_in 
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out 
 *   Pointer to the output aggregate
 *
 * @param[in] num_items 
 *   Total number of input data items
 *
 * @param[in] reduction_op 
 *   Binary reduction functor
 *
 * @param[in] init 
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS), 1) 
__global__ void DeviceReduceSingleTileKernel(InputIteratorT d_in,
                                             OutputIteratorT d_out,
                                             OffsetT num_items,
                                             ReductionOpT reduction_op,
                                             InitT init)
{
  // Thread block type for reducing input tiles
  using AgentReduceT =
    AgentReduce<typename ChainedPolicyT::ActivePolicy::SingleTilePolicy,
                InputIteratorT,
                OutputIteratorT,
                OffsetT,
                ReductionOpT,
                AccumT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = init;
    }

    return;
  }

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op)
                             .ConsumeRange(OffsetT(0), num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    *d_out = reduction_op(init, block_aggregate);
  }
}

/// Normalize input iterator to segment offset
template <typename T, typename OffsetT, typename IteratorT>
__device__ __forceinline__ void NormalizeReductionOutput(T & /*val*/,
                                                         OffsetT /*base_offset*/,
                                                         IteratorT /*itr*/)
{}

/// Normalize input iterator to segment offset (specialized for arg-index)
template <typename KeyValuePairT,
          typename OffsetT,
          typename WrappedIteratorT,
          typename OutputValueT>
__device__ __forceinline__ void NormalizeReductionOutput(
  KeyValuePairT &val,
  OffsetT base_offset,
  ArgIndexInputIterator<WrappedIteratorT, OffsetT, OutputValueT> /*itr*/)
{
  val.key -= base_offset;
}

/**
 * Segmented reduction (one block per segment)
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 * 
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items \iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate \iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets 
 *   \iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets 
 *   \iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member 
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @param[in] d_in 
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out 
 *   Pointer to the output aggregate
 *
 * @param[in] d_begin_offsets 
 *   Random-access input iterator to the sequence of beginning offsets of 
 *   length `num_segments`, such that `d_begin_offsets[i]` is the first element 
 *   of the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets 
 *   Random-access input iterator to the sequence of ending offsets of length 
 *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
 *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
 *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
 *   considered empty.
 *
 * @param[in] num_segments 
 *   The number of segments that comprise the sorting data
 *
 * @param[in] reduction_op 
 *   Binary reduction functor
 *
 * @param[in] init 
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) 
__global__ void DeviceSegmentedReduceKernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int /*num_segments*/,
    ReductionOpT reduction_op,
    InitT init)
{
  // Thread block type for reducing input tiles
  using AgentReduceT =
    AgentReduce<typename ChainedPolicyT::ActivePolicy::ReducePolicy,
                InputIteratorT,
                OutputIteratorT,
                OffsetT,
                ReductionOpT,
                AccumT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  OffsetT segment_begin = d_begin_offsets[blockIdx.x];
  OffsetT segment_end   = d_end_offsets[blockIdx.x];

  // Check if empty problem
  if (segment_begin == segment_end)
  {
    if (threadIdx.x == 0)
    {
      d_out[blockIdx.x] = init;
    }
    return;
  }

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op)
                             .ConsumeRange(segment_begin, segment_end);

  // Normalize as needed
  NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

  if (threadIdx.x == 0)
  {
    d_out[blockIdx.x] = reduction_op(init, block_aggregate);
  }
}

/******************************************************************************
 * Policy
 ******************************************************************************/

/**
 * @tparam AccumT
 *   Accumulator data type
 *
 * OffsetT
 *   Signed integer type for global offsets
 *
 * ReductionOpT
 *   Binary reduction functor type having member 
 *   `auto operator()(const T &a, const U &b)`
 */
template <
    typename AccumT,            
    typename OffsetT,          
    typename ReductionOpT>    
struct DeviceReducePolicy
{
  //---------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //---------------------------------------------------------------------------

  /// SM30
  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 2;

    // ReducePolicy (GTX670: 154.0 @ 48M 4B items)
    using ReducePolicy = AgentReducePolicy<threads_per_block,
                                           items_per_thread,
                                           AccumT,
                                           items_per_vec_load,
                                           BLOCK_REDUCE_WARP_REDUCTIONS,
                                           LOAD_DEFAULT>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B
    // items)
    using ReducePolicy = AgentReducePolicy<threads_per_block,
                                           items_per_thread,
                                           AccumT,
                                           items_per_vec_load,
                                           BLOCK_REDUCE_WARP_REDUCTIONS,
                                           LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  /// SM60
  struct Policy600 : ChainedPolicy<600, Policy600, Policy350>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy = AgentReducePolicy<threads_per_block,
                                           items_per_thread,
                                           AccumT,
                                           items_per_vec_load,
                                           BLOCK_REDUCE_WARP_REDUCTIONS,
                                           LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy600;
};



/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for 
 *        device-wide reduction
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items \iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate \iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member 
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT 
 *   Initial value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename InitT = 
    cub::detail::non_void_value_t<
      OutputIteratorT, 
      cub::detail::value_t<InputIteratorT>>,
  typename AccumT = 
    detail::accumulator_t<
      ReductionOpT, 
      InitT, 
      cub::detail::value_t<InputIteratorT>>,
  typename SelectedPolicy = DeviceReducePolicy<AccumT, OffsetT, ReductionOpT>>
struct DispatchReduce : SelectedPolicy
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void *d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t &temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION __forceinline__ DispatchReduce(void *d_temp_storage,
                                                      size_t &temp_storage_bytes,
                                                      InputIteratorT d_in,
                                                      OutputIteratorT d_out,
                                                      OffsetT num_items,
                                                      ReductionOpT reduction_op,
                                                      InitT init,
                                                      cudaStream_t stream,
                                                      int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
  {} 

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchReduce(void* d_temp_storage,
                 size_t &temp_storage_bytes,
                 InputIteratorT d_in,
                 OutputIteratorT d_out,
                 OffsetT num_items,
                 ReductionOpT reduction_op,
                 InitT init,
                 cudaStream_t stream,
                 bool debug_synchronous,
                 int ptx_version)
    : d_temp_storage(d_temp_storage)
    , temp_storage_bytes(temp_storage_bytes)
    , d_in(d_in)
    , d_out(d_out)
    , num_items(num_items)
    , reduction_op(reduction_op)
    , init(init)
    , stream(stream)
    , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of 
   *   cub::DeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == NULL)
      {
        temp_storage_bytes = 1;
        break;
      }

      // Log single_reduce_sweep_kernel configuration
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
              (long long)stream,
              ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
      #endif

      // Invoke single_reduce_sweep_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
        .doit(single_tile_kernel, d_in, d_out, num_items, reduction_op, init);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Normal problem size invocation (two-pass)
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke two-passes to reduce
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeviceReduceKernel
   *
   * @param[in] single_tile_kernel 
   *   Kernel function pointer to parameterization of 
   *   cub::DeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT,
            typename ReduceKernelT,
            typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel,
               SingleTileKernelT single_tile_kernel)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      if (CubDebug(error = cudaGetDevice(&device_ordinal)))
        break;

      // Get SM count
      int sm_count;
      if (CubDebug(
            error = cudaDeviceGetAttribute(&sm_count,
                                           cudaDevAttrMultiProcessorCount,
                                           device_ordinal)))
      {
        break;
      }

      // Init regular kernel configuration
      KernelConfig reduce_config;
      if (CubDebug(
            error = reduce_config.Init<typename ActivePolicyT::ReducePolicy>(
              reduce_kernel)))
      {
        break;
      }

      int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;

      // Even-share work distribution
      int max_blocks = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);
      GridEvenShare<OffsetT> even_share;
      even_share.DispatchInit(num_items, max_blocks, reduce_config.tile_size);

      // Temporary storage allocation requirements
      void *allocations[1]       = {};
      size_t allocation_sizes[1] = {
        max_blocks * sizeof(AccumT) // bytes needed for privatized block
                                    // reductions
      };

      // Alias the temporary allocations from the single storage blob (or
      // compute the necessary size of the blob)
      if (CubDebug(error = AliasTemporaries(d_temp_storage,
                                            temp_storage_bytes,
                                            allocations,
                                            allocation_sizes)))
      {
        break;
      }

      if (d_temp_storage == NULL)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        return cudaSuccess;
      }

      // Alias the allocation for the privatized per-block reductions
      AccumT *d_block_reductions = (AccumT *)allocations[0];

      // Get grid size for device_reduce_sweep_kernel
      int reduce_grid_size = even_share.grid_size;

      // Log device_reduce_sweep_kernel configuration
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
              "per thread, %d SM occupancy\n",
              reduce_grid_size,
              ActivePolicyT::ReducePolicy::BLOCK_THREADS,
              (long long)stream,
              ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD,
              reduce_config.sm_occupancy);
      #endif

      // Invoke DeviceReduceKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        reduce_grid_size,
        ActivePolicyT::ReducePolicy::BLOCK_THREADS,
        0,
        stream)
        .doit(reduce_kernel,
              d_in,
              d_block_reductions,
              num_items,
              even_share,
              reduction_op);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }

      // Log single_reduce_sweep_kernel configuration
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
              (long long)stream,
              ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
      #endif

      // Invoke DeviceReduceSingleTileKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        1,
        ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
        0,
        stream)
        .doit(single_tile_kernel,
              d_block_reductions,
              d_out,
              reduce_grid_size, // triple_chevron is not type safe, make sure to use int
              reduction_op,
              init);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    typedef typename ActivePolicyT::SingleTilePolicy SingleTilePolicyT;
    typedef typename DispatchReduce::MaxPolicy MaxPolicyT;

    // Force kernel code-generation in all compiler passes
    if (num_items <= (SingleTilePolicyT::BLOCK_THREADS *
                      SingleTilePolicyT::ITEMS_PER_THREAD))
    {
      // Small, single tile size
      return InvokeSingleTile<ActivePolicyT>(
        DeviceReduceSingleTileKernel<MaxPolicyT,
                                     InputIteratorT,
                                     OutputIteratorT,
                                     OffsetT,
                                     ReductionOpT,
                                     InitT,
                                     AccumT>);
    }
    else
    {
      // Regular size
      return InvokePasses<ActivePolicyT>(
        DeviceReduceKernel<typename DispatchReduce::MaxPolicy,
                           InputIteratorT,
                           OffsetT,
                           ReductionOpT,
                           AccumT>,
        DeviceReduceSingleTileKernel<MaxPolicyT,
                                     AccumT *,
                                     OutputIteratorT,
                                     int, // Always used with int offsets
                                     ReductionOpT,
                                     InitT,
                                     AccumT>);
    }
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
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
   * @param[in] reduction_op 
   *   Binary reduction functor
   *
   * @param[in] init 
   *   The initial value of the reduction
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within. 
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           InputIteratorT d_in,
           OutputIteratorT d_out,
           OffsetT num_items,
           ReductionOpT reduction_op,
           InitT init,
           cudaStream_t stream)
  {
    typedef typename DispatchReduce::MaxPolicy MaxPolicyT;

    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Create dispatch functor
      DispatchReduce dispatch(d_temp_storage,
                              temp_storage_bytes,
                              d_in,
                              d_out,
                              num_items,
                              reduction_op,
                              init,
                              stream,
                              ptx_version);

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           InputIteratorT d_in,
           OutputIteratorT d_out,
           OffsetT num_items,
           ReductionOpT reduction_op,
           InitT init,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_in,
                    d_out,
                    num_items,
                    reduction_op,
                    init,
                    stream);
  }
};



/******************************************************************************
 * Segmented dispatch
 *****************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for 
 *        device-wide reduction
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items \iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate \iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets 
 *   \iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets 
 *   \iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member 
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT 
 *   value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename InitT = 
    cub::detail::non_void_value_t<
      OutputIteratorT, 
      cub::detail::value_t<InputIteratorT>>,
  typename AccumT = 
    detail::accumulator_t<
      ReductionOpT, 
      InitT, 
      cub::detail::value_t<InputIteratorT>>,
  typename SelectedPolicy = DeviceReducePolicy<AccumT, OffsetT, ReductionOpT>>
struct DispatchSegmentedReduce : SelectedPolicy
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void *d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t &temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// The number of segments that comprise the sorting data
  OffsetT num_segments;

  /// Random-access input iterator to the sequence of beginning offsets of
  /// length `num_segments`, such that `d_begin_offsets[i]` is the first
  /// element of the *i*<sup>th</sup> data segment in `d_keys_*` and
  /// `d_values_*`
  BeginOffsetIteratorT d_begin_offsets;

  /// Random-access input iterator to the sequence of ending offsets of length
  /// `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
  /// the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
  /// If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
  /// considered empty.
  EndOffsetIteratorT d_end_offsets;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchSegmentedReduce(void *d_temp_storage,
                          size_t &temp_storage_bytes,
                          InputIteratorT d_in,
                          OutputIteratorT d_out,
                          OffsetT num_segments,
                          BeginOffsetIteratorT d_begin_offsets,
                          EndOffsetIteratorT d_end_offsets,
                          ReductionOpT reduction_op,
                          InitT init,
                          cudaStream_t stream,
                          int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchSegmentedReduce(void *d_temp_storage,
                          size_t &temp_storage_bytes,
                          InputIteratorT d_in,
                          OutputIteratorT d_out,
                          OffsetT num_segments,
                          BeginOffsetIteratorT d_begin_offsets,
                          EndOffsetIteratorT d_end_offsets,
                          ReductionOpT reduction_op,
                          InitT init,
                          cudaStream_t stream,
                          bool debug_synchronous,
                          int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invocation
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam DeviceSegmentedReduceKernelT
   *   Function type of cub::DeviceSegmentedReduceKernel
   *
   * @param[in] segmented_reduce_kernel
   *   Kernel function pointer to parameterization of 
   *   cub::DeviceSegmentedReduceKernel
   */
  template <typename ActivePolicyT, typename DeviceSegmentedReduceKernelT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  InvokePasses(DeviceSegmentedReduceKernelT segmented_reduce_kernel)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == NULL)
      {
        temp_storage_bytes = 1;
        return cudaSuccess;
      }

      // Init kernel configuration
      KernelConfig segmented_reduce_config;
      if (CubDebug(
            error = segmented_reduce_config
                      .Init<typename ActivePolicyT::SegmentedReducePolicy>(
                        segmented_reduce_kernel)))
      {
        break;
      }

      // Log device_reduce_sweep_kernel configuration
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking SegmentedDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), "
              "%d items per thread, %d SM occupancy\n",
              num_segments,
              ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS,
              (long long)stream,
              ActivePolicyT::SegmentedReducePolicy::ITEMS_PER_THREAD,
              segmented_reduce_config.sm_occupancy);
      #endif

      // Invoke DeviceReduceKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        num_segments,
        ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS,
        0,
        stream)
        .doit(segmented_reduce_kernel,
              d_in,
              d_out,
              d_begin_offsets,
              d_end_offsets,
              num_segments,
              reduction_op,
              init);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }
    } while (0);

    return error;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    typedef typename DispatchSegmentedReduce::MaxPolicy MaxPolicyT;

    // Force kernel code-generation in all compiler passes
    return InvokePasses<ActivePolicyT>(
      DeviceSegmentedReduceKernel<MaxPolicyT,
                                  InputIteratorT,
                                  OutputIteratorT,
                                  BeginOffsetIteratorT,
                                  EndOffsetIteratorT,
                                  OffsetT,
                                  ReductionOpT,
                                  InitT,
                                  AccumT>);
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
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
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of 
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           InputIteratorT d_in,
           OutputIteratorT d_out,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           ReductionOpT reduction_op,
           InitT init,
           cudaStream_t stream)
  {
    typedef typename DispatchSegmentedReduce::MaxPolicy MaxPolicyT;

    if (num_segments <= 0)
    {
      return cudaSuccess;
    }

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedReduce dispatch(d_temp_storage,
                                       temp_storage_bytes,
                                       d_in,
                                       d_out,
                                       num_segments,
                                       d_begin_offsets,
                                       d_end_offsets,
                                       reduction_op,
                                       init,
                                       stream,
                                       ptx_version);

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           InputIteratorT d_in,
           OutputIteratorT d_out,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           ReductionOpT reduction_op,
           InitT init,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_in,
                    d_out,
                    num_segments,
                    d_begin_offsets,
                    d_end_offsets,
                    reduction_op,
                    init,
                    stream);
  }
};


CUB_NAMESPACE_END
