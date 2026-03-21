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
 * @file cub::DeviceReduceByKey provides device-wide, parallel operations for 
 *       reducing segments of values residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <nv/target>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Multi-block reduce-by-key sweep kernel entry point
 *
 * @tparam AgentReduceByKeyPolicyT
 *   Parameterized AgentReduceByKeyPolicyT tuning policy type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param d_unique_out
 *   Pointer to the output sequence of unique keys (one key per run)
 *
 * @param d_values_in
 *   Pointer to the input sequence of corresponding values
 *
 * @param d_aggregates_out
 *   Pointer to the output sequence of value aggregates (one aggregate per run)
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs encountered
 *   (i.e., the length of d_unique_out)
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   KeyT equality operator
 *
 * @param reduction_op
 *   ValueT reduction operator
 *
 * @param num_items
 *   Total number of items to select from
 */
template <typename AgentReduceByKeyPolicyT,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT>
__launch_bounds__(int(AgentReduceByKeyPolicyT::BLOCK_THREADS)) __global__
  void DeviceReduceByKeyKernel(KeysInputIteratorT d_keys_in,
                               UniqueOutputIteratorT d_unique_out,
                               ValuesInputIteratorT d_values_in,
                               AggregatesOutputIteratorT d_aggregates_out,
                               NumRunsOutputIteratorT d_num_runs_out,
                               ScanTileStateT tile_state,
                               int start_tile,
                               EqualityOpT equality_op,
                               ReductionOpT reduction_op,
                               OffsetT num_items)
{
  // Thread block type for reducing tiles of value segments
  using AgentReduceByKeyT = AgentReduceByKey<AgentReduceByKeyPolicyT,
                                             KeysInputIteratorT,
                                             UniqueOutputIteratorT,
                                             ValuesInputIteratorT,
                                             AggregatesOutputIteratorT,
                                             NumRunsOutputIteratorT,
                                             EqualityOpT,
                                             ReductionOpT,
                                             OffsetT,
                                             AccumT>;

  // Shared memory for AgentReduceByKey
  __shared__ typename AgentReduceByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentReduceByKeyT(temp_storage,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    d_aggregates_out,
                    d_num_runs_out,
                    equality_op,
                    reduction_op)
    .ConsumeRange(num_items, tile_state, start_tile);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceReduceByKey
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 */
template <typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT = 
            detail::accumulator_t<
              ReductionOpT, 
              cub::detail::value_t<ValuesInputIteratorT>,
              cub::detail::value_t<ValuesInputIteratorT>>>
struct DispatchReduceByKey
{
  //-------------------------------------------------------------------------
  // Types and constants
  //-------------------------------------------------------------------------

  // The input keys type
  using KeyInputT = cub::detail::value_t<KeysInputIteratorT>;

  // The output keys type
  using KeyOutputT =
    cub::detail::non_void_value_t<UniqueOutputIteratorT, KeyInputT>;

  // The input values type
  using ValueInputT = cub::detail::value_t<ValuesInputIteratorT>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  static constexpr int MAX_INPUT_BYTES = CUB_MAX(sizeof(KeyOutputT),
                                                 sizeof(AccumT));

  static constexpr int COMBINED_INPUT_BYTES = sizeof(KeyOutputT) +
                                              sizeof(AccumT);

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

  //-------------------------------------------------------------------------
  // Tuning policies
  //-------------------------------------------------------------------------

  /// SM35
  struct Policy350
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 6;
    static constexpr int ITEMS_PER_THREAD =
      (MAX_INPUT_BYTES <= 8)
        ? 6
        : CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD,
                  CUB_MAX(1,
                          ((NOMINAL_4B_ITEMS_PER_THREAD * 8) +
                           COMBINED_INPUT_BYTES - 1) /
                            COMBINED_INPUT_BYTES));

    using ReduceByKeyPolicyT = AgentReduceByKeyPolicy<128,
                                                      ITEMS_PER_THREAD,
                                                      BLOCK_LOAD_DIRECT,
                                                      LOAD_LDG,
                                                      BLOCK_SCAN_WARP_SCANS>;
  };

  /******************************************************************************
   * Tuning policies of current PTX compiler pass
   ******************************************************************************/

  using PtxPolicy = Policy350;

  // "Opaque" policies (whose parameterizations aren't reflected in the type
  // signature)
  struct PtxReduceByKeyPolicy : PtxPolicy::ReduceByKeyPolicyT
  {};

  /******************************************************************************
   * Utilities
   ******************************************************************************/

  /**
   * Initialize kernel dispatch configurations with the policies corresponding
   * to the PTX assembly we will use
   */
  template <typename KernelConfig>
  CUB_RUNTIME_FUNCTION __forceinline__ static void
  InitConfigs(int /*ptx_version*/, KernelConfig &reduce_by_key_config)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (
                   // We're on the device, so initialize the kernel dispatch
                   // configurations with the current PTX policy
                   reduce_by_key_config.template Init<PtxReduceByKeyPolicy>();),
                 (
                   // We're on the host, so lookup and initialize the kernel
                   // dispatch configurations with the policies that match the
                   // device's PTX version

                   // (There's only one policy right now)
                   reduce_by_key_config
                     .template Init<typename Policy350::ReduceByKeyPolicyT>();));
  }

  /**
   * Kernel kernel dispatch configuration.
   */
  struct KernelConfig
  {
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__ void Init()
    {
      block_threads    = PolicyT::BLOCK_THREADS;
      items_per_thread = PolicyT::ITEMS_PER_THREAD;
      tile_items       = block_threads * items_per_thread;
    }
  };

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide
   *        reduce-by-key using the specified kernel functions.
   *
   * @tparam ScanInitKernelT
   *   Function type of cub::DeviceScanInitKernel
   *
   * @tparam ReduceByKeyKernelT
   *   Function type of cub::DeviceReduceByKeyKernelT
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of corresponding values
   *
   * @param[out] d_aggregates_out
   *   Pointer to the output sequence of value aggregates
   *   (one aggregate per run)
   *
   * @param[out] d_num_runs_out
   *   Pointer to total number of runs encountered
   *   (i.e., the length of d_unique_out)
   *
   * @param[in] equality_op
   *   KeyT equality operator
   *
   * @param[in] reduction_op
   *   ValueT reduction operator
   *
   * @param[in] num_items
   *   Total number of items to select from
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   *
   * @param[in] ptx_version
   *   PTX version of dispatch kernels
   *
   * @param[in] init_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceScanInitKernel
   *
   * @param[in] reduce_by_key_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceReduceByKeyKernel
   *
   * @param[in] reduce_by_key_config
   *   Dispatch parameters that match the policy that
   *   `reduce_by_key_kernel` was compiled for
   */
  template <typename ScanInitKernelT, typename ReduceByKeyKernelT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream,
           int /*ptx_version*/,
           ScanInitKernelT init_kernel,
           ReduceByKeyKernelT reduce_by_key_kernel,
           KernelConfig reduce_by_key_config)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      if (CubDebug(error = cudaGetDevice(&device_ordinal)))
      {
        break;
      }

      // Number of input tiles
      int tile_size = reduce_by_key_config.block_threads *
                      reduce_by_key_config.items_per_thread;
      int num_tiles =
        static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles,
                                                          allocation_sizes[0])))
      {
        break; // bytes needed for tile status descriptors
      }

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void *allocations[1] = {};
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
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_state;
      if (CubDebug(error = tile_state.Init(num_tiles,
                                           allocations[0],
                                           allocation_sizes[0])))
      {
        break;
      }

      // Log init_kernel configuration
      int init_grid_size =
        CUB_MAX(1, cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));

      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              (long long)stream);
      #endif

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        init_grid_size,
        INIT_KERNEL_THREADS,
        0,
        stream)
        .doit(init_kernel, tile_state, num_tiles, d_num_runs_out);

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

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Get SM occupancy for reduce_by_key_kernel
      int reduce_by_key_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(reduce_by_key_sm_occupancy,
                                          reduce_by_key_kernel,
                                          reduce_by_key_config.block_threads)))
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x,
                                                  cudaDevAttrMaxGridDimX,
                                                  device_ordinal)))
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles;
           start_tile += scan_grid_size)
      {
        // Log reduce_by_key_kernel configuration
        #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking %d reduce_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                reduce_by_key_config.block_threads,
                (long long)stream,
                reduce_by_key_config.items_per_thread,
                reduce_by_key_sm_occupancy);
        #endif

        // Invoke reduce_by_key_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          scan_grid_size,
          reduce_by_key_config.block_threads,
          0,
          stream)
          .doit(reduce_by_key_kernel,
                d_keys_in,
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                tile_state,
                start_tile,
                equality_op,
                reduction_op,
                num_items);

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
      }
    } while (0);

    return error;
  }

  template <typename ScanInitKernelT, typename ReduceByKeyKernelT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous,
           int ptx_version,
           ScanInitKernelT init_kernel,
           ReduceByKeyKernelT reduce_by_key_kernel,
           KernelConfig reduce_by_key_config)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch<ScanInitKernelT, ReduceByKeyKernelT>(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys_in,
                                                         d_unique_out,
                                                         d_values_in,
                                                         d_aggregates_out,
                                                         d_num_runs_out,
                                                         equality_op,
                                                         reduction_op,
                                                         num_items,
                                                         stream,
                                                         ptx_version,
                                                         init_kernel,
                                                         reduce_by_key_kernel,
                                                         reduce_by_key_config);
  }

  /**
   * Internal dispatch routine
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of corresponding values
   *
   * @param[out] d_aggregates_out
   *   Pointer to the output sequence of value aggregates
   *   (one aggregate per run)
   *
   * @param[out] d_num_runs_out
   *   Pointer to total number of runs encountered
   *   (i.e., the length of d_unique_out)
   *
   * @param[in] equality_op
   *   KeyT equality operator
   *
   * @param[in] reduction_op
   *   ValueT reduction operator
   *
   * @param[in] num_items
   *   Total number of items to select from
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Get kernel kernel dispatch configurations
      KernelConfig reduce_by_key_config;
      InitConfigs(ptx_version, reduce_by_key_config);

      // Dispatch
      if (CubDebug(
            error = Dispatch(
              d_temp_storage,
              temp_storage_bytes,
              d_keys_in,
              d_unique_out,
              d_values_in,
              d_aggregates_out,
              d_num_runs_out,
              equality_op,
              reduction_op,
              num_items,
              stream,
              ptx_version,
              DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
              DeviceReduceByKeyKernel<PtxReduceByKeyPolicy,
                                      KeysInputIteratorT,
                                      UniqueOutputIteratorT,
                                      ValuesInputIteratorT,
                                      AggregatesOutputIteratorT,
                                      NumRunsOutputIteratorT,
                                      ScanTileStateT,
                                      EqualityOpT,
                                      ReductionOpT,
                                      OffsetT,
                                      AccumT>,
              reduce_by_key_config)))
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
           KeysInputIteratorT d_keys_in,
           UniqueOutputIteratorT d_unique_out,
           ValuesInputIteratorT d_values_in,
           AggregatesOutputIteratorT d_aggregates_out,
           NumRunsOutputIteratorT d_num_runs_out,
           EqualityOpT equality_op,
           ReductionOpT reduction_op,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    d_aggregates_out,
                    d_num_runs_out,
                    equality_op,
                    reduction_op,
                    num_items,
                    stream);
  }
};

CUB_NAMESPACE_END
