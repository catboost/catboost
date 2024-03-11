/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#pragma clang system_header


#include <iterator>
#include <cstdio>

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <nv/target>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

template <typename AgentThreeWayPartitionPolicyT,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT>
__launch_bounds__(int(AgentThreeWayPartitionPolicyT::BLOCK_THREADS)) __global__
void DeviceThreeWayPartitionKernel(InputIteratorT d_in,
                                   FirstOutputIteratorT d_first_part_out,
                                   SecondOutputIteratorT d_second_part_out,
                                   UnselectedOutputIteratorT d_unselected_out,
                                   NumSelectedIteratorT d_num_selected_out,
                                   ScanTileStateT tile_status_1,
                                   ScanTileStateT tile_status_2,
                                   SelectFirstPartOp select_first_part_op,
                                   SelectSecondPartOp select_second_part_op,
                                   OffsetT num_items,
                                   int num_tiles)
{
  // Thread block type for selecting data from input tiles
  using AgentThreeWayPartitionT =
    AgentThreeWayPartition<AgentThreeWayPartitionPolicyT,
                           InputIteratorT,
                           FirstOutputIteratorT,
                           SecondOutputIteratorT,
                           UnselectedOutputIteratorT,
                           SelectFirstPartOp,
                           SelectSecondPartOp,
                           OffsetT>;

  // Shared memory for AgentThreeWayPartition
  __shared__ typename AgentThreeWayPartitionT::TempStorage temp_storage;

  // Process tiles
  AgentThreeWayPartitionT(temp_storage,
                          d_in,
                          d_first_part_out,
                          d_second_part_out,
                          d_unselected_out,
                          select_first_part_op,
                          select_second_part_op,
                          num_items)
    .ConsumeRange(num_tiles, tile_status_1, tile_status_2, d_num_selected_out);
}

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state_1
 *   Tile status interface
 *
 * @param[in] tile_state_2
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_selected_out)
 */
template <typename ScanTileStateT,
          typename NumSelectedIteratorT>
__global__ void
DeviceThreeWayPartitionInitKernel(ScanTileStateT tile_state_1,
                                  ScanTileStateT tile_state_2,
                                  int num_tiles,
                                  NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state_1.InitializeStatus(num_tiles);
  tile_state_2.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if (blockIdx.x == 0)
  {
    if (threadIdx.x < 2)
    {
      d_num_selected_out[threadIdx.x] = 0;
    }
  }
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

template <typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT>
struct DispatchThreeWayPartitionIf
{
  /*****************************************************************************
   * Types and constants
   ****************************************************************************/

  using InputT = cub::detail::value_t<InputIteratorT>;
  using ScanTileStateT = cub::ScanTileState<OffsetT>;

  constexpr static int INIT_KERNEL_THREADS = 256;


  /*****************************************************************************
   * Tuning policies
   ****************************************************************************/

  /// SM35
  struct Policy350
  {
    constexpr static int ITEMS_PER_THREAD = Nominal4BItemsToItems<InputT>(9);

    using ThreeWayPartitionPolicy =
      cub::AgentThreeWayPartitionPolicy<256,
                                        ITEMS_PER_THREAD,
                                        cub::BLOCK_LOAD_DIRECT,
                                        cub::LOAD_DEFAULT,
                                        cub::BLOCK_SCAN_WARP_SCANS>;
  };

  /*****************************************************************************
   * Tuning policies of current PTX compiler pass
   ****************************************************************************/

  using PtxPolicy = Policy350;

  // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
  struct PtxThreeWayPartitionPolicyT : PtxPolicy::ThreeWayPartitionPolicy {};


  /*****************************************************************************
   * Utilities
   ****************************************************************************/

  /**
   * Initialize kernel dispatch configurations with the policies corresponding
   * to the PTX assembly we will use
   */
  template <typename KernelConfig>
  CUB_RUNTIME_FUNCTION __forceinline__
  static void InitConfigs(
    int             ptx_version,
    KernelConfig    &select_if_config)
  {
    NV_IF_TARGET(
      NV_IS_DEVICE,
      ((void)ptx_version;
       // We're on the device, so initialize the kernel dispatch configurations
       // with the current PTX policy
       select_if_config.template Init<PtxThreeWayPartitionPolicyT>();),
      (// We're on the host, so lookup and initialize the kernel dispatch
       // configurations with the policies that match the device's PTX version
       // (There's only one policy right now)
       (void)ptx_version;
       select_if_config
         .template Init<typename Policy350::ThreeWayPartitionPolicy>();));
  }


  /**
   * Kernel dispatch configuration.
   */
  struct KernelConfig
  {
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    void Init()
    {
      block_threads       = PolicyT::BLOCK_THREADS;
      items_per_thread    = PolicyT::ITEMS_PER_THREAD;
      tile_items          = block_threads * items_per_thread;
    }
  };


  /*****************************************************************************
   * Dispatch entrypoints
   ****************************************************************************/

  template <typename ScanInitKernelPtrT,
            typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           InputIteratorT d_in,
           FirstOutputIteratorT d_first_part_out,
           SecondOutputIteratorT d_second_part_out,
           UnselectedOutputIteratorT d_unselected_out,
           NumSelectedIteratorT d_num_selected_out,
           SelectFirstPartOp select_first_part_op,
           SelectSecondPartOp select_second_part_op,
           OffsetT num_items,
           cudaStream_t stream,
           int /*ptx_version*/,
           ScanInitKernelPtrT three_way_partition_init_kernel,
           SelectIfKernelPtrT three_way_partition_kernel,
           KernelConfig three_way_partition_config)
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
      int tile_size = three_way_partition_config.block_threads *
                      three_way_partition_config.items_per_thread;
      int num_tiles = static_cast<int>(DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[2]; // bytes needed for tile status descriptors

      if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles,
                                                          allocation_sizes[0])))
      {
        break;
      }

      allocation_sizes[1] = allocation_sizes[0];

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void* allocations[2] = {};
      if (CubDebug(error = cub::AliasTemporaries(d_temp_storage,
                                                 temp_storage_bytes,
                                                 allocations,
                                                 allocation_sizes)))
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_status_1;
      ScanTileStateT tile_status_2;

      if (CubDebug(error = tile_status_1.Init(num_tiles,
                                              allocations[0],
                                              allocation_sizes[0])))
      {
        break;
      }

      if (CubDebug(error = tile_status_2.Init(num_tiles,
                                              allocations[1],
                                              allocation_sizes[1])))
      {
        break;
      }

      // Log three_way_partition_init_kernel configuration
      int init_grid_size = CUB_MAX(1, DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));

      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking three_way_partition_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              reinterpret_cast<long long>(stream));
      #endif

      // Invoke three_way_partition_init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        init_grid_size, INIT_KERNEL_THREADS, 0, stream
      ).doit(three_way_partition_init_kernel,
             tile_status_1,
             tile_status_2,
             num_tiles,
             d_num_selected_out);

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

      // Get max x-dimension of grid
      int max_dim_x;
      if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x,
                                                  cudaDevAttrMaxGridDimX,
                                                  device_ordinal)))
      {
        break;
      }

      // Get grid size for scanning tiles
      dim3 scan_grid_size;
      scan_grid_size.z = 1;
      scan_grid_size.y = DivideAndRoundUp(num_tiles, max_dim_x);
      scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

      // Log select_if_kernel configuration
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      {
        // Get SM occupancy for select_if_kernel
        int range_select_sm_occupancy;
        if (CubDebug(error = MaxSmOccupancy(
          range_select_sm_occupancy,            // out
          three_way_partition_kernel,
          three_way_partition_config.block_threads)))
        {
          break;
        }

        _CubLog("Invoking three_way_partition_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                scan_grid_size.x,
                scan_grid_size.y,
                scan_grid_size.z,
                three_way_partition_config.block_threads,
                reinterpret_cast<long long>(stream),
                three_way_partition_config.items_per_thread,
                range_select_sm_occupancy);
      }
      #endif

      // Invoke select_if_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        scan_grid_size, three_way_partition_config.block_threads, 0, stream
      ).doit(three_way_partition_kernel,
             d_in,
             d_first_part_out,
             d_second_part_out,
             d_unselected_out,
             d_num_selected_out,
             tile_status_1,
             tile_status_2,
             select_first_part_op,
             select_second_part_op,
             num_items,
             num_tiles);

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
    while (0);

    return error;
  }

  template <typename ScanInitKernelPtrT,
            typename SelectIfKernelPtrT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           InputIteratorT d_in,
           FirstOutputIteratorT d_first_part_out,
           SecondOutputIteratorT d_second_part_out,
           UnselectedOutputIteratorT d_unselected_out,
           NumSelectedIteratorT d_num_selected_out,
           SelectFirstPartOp select_first_part_op,
           SelectSecondPartOp select_second_part_op,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous,
           int ptx_version,
           ScanInitKernelPtrT three_way_partition_init_kernel,
           SelectIfKernelPtrT three_way_partition_kernel,
           KernelConfig three_way_partition_config)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch<ScanInitKernelPtrT, SelectIfKernelPtrT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      select_first_part_op,
      select_second_part_op,
      num_items,
      stream,
      ptx_version,
      three_way_partition_init_kernel,
      three_way_partition_kernel,
      three_way_partition_config);
  }


  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION __forceinline__
  static cudaError_t Dispatch(
    void*                       d_temp_storage,
    std::size_t&                temp_storage_bytes,
    InputIteratorT              d_in,
    FirstOutputIteratorT        d_first_part_out,
    SecondOutputIteratorT       d_second_part_out,
    UnselectedOutputIteratorT   d_unselected_out,
    NumSelectedIteratorT        d_num_selected_out,
    SelectFirstPartOp           select_first_part_op,
    SelectSecondPartOp          select_second_part_op,
    OffsetT                     num_items,
    cudaStream_t                stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = cub::PtxVersion(ptx_version)))
      {
        break;
      }

      // Get kernel kernel dispatch configurations
      KernelConfig select_if_config;
      InitConfigs(ptx_version, select_if_config);

      // Dispatch
      if (CubDebug(error = Dispatch(
                     d_temp_storage,
                     temp_storage_bytes,
                     d_in,
                     d_first_part_out,
                     d_second_part_out,
                     d_unselected_out,
                     d_num_selected_out,
                     select_first_part_op,
                     select_second_part_op,
                     num_items,
                     stream,
                     ptx_version,
                     DeviceThreeWayPartitionInitKernel<ScanTileStateT,
                                                       NumSelectedIteratorT>,
                     DeviceThreeWayPartitionKernel<PtxThreeWayPartitionPolicyT,
                                                   InputIteratorT,
                                                   FirstOutputIteratorT,
                                                   SecondOutputIteratorT,
                                                   UnselectedOutputIteratorT,
                                                   NumSelectedIteratorT,
                                                   ScanTileStateT,
                                                   SelectFirstPartOp,
                                                   SelectSecondPartOp,
                                                   OffsetT>,
                     select_if_config)))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__
  static cudaError_t Dispatch(
    void*                       d_temp_storage,
    std::size_t&                temp_storage_bytes,
    InputIteratorT              d_in,
    FirstOutputIteratorT        d_first_part_out,
    SecondOutputIteratorT       d_second_part_out,
    UnselectedOutputIteratorT   d_unselected_out,
    NumSelectedIteratorT        d_num_selected_out,
    SelectFirstPartOp           select_first_part_op,
    SelectSecondPartOp          select_second_part_op,
    OffsetT                     num_items,
    cudaStream_t                stream,
    bool                        debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_in,
                    d_first_part_out,
                    d_second_part_out,
                    d_unselected_out,
                    d_num_selected_out,
                    select_first_part_op,
                    select_second_part_op,
                    num_items,
                    stream);
  }
};


CUB_NAMESPACE_END
