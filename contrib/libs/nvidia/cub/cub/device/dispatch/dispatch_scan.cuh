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
 * @file cub::DeviceScan provides device-wide, parallel operations for
 *       computing a prefix scan across a sequence of data items residing
 *       within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>

#include <cub/agent/agent_scan.cuh>
#include <cub/config.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 */
template <typename ScanTileStateT>
__global__ void DeviceScanInitKernel(ScanTileStateT tile_state, int num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of `d_selected_out`)
 */
template <typename ScanTileStateT, typename NumSelectedIteratorT>
__global__ void DeviceCompactInitKernel(ScanTileStateT tile_state,
                                        int num_tiles,
                                        NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if ((blockIdx.x == 0) && (threadIdx.x == 0))
  {
    *d_num_selected_out = 0;
  }
}

/**
 * @brief Scan kernel entry point (multi-block)
 *
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading scan inputs \iterator
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type for writing scan outputs \iterator
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam ScanOpT
 *   Binary scan functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   Initial value to seed the exclusive scan
 *   (cub::NullType for inclusive scans)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @paramInput d_in
 *   data
 *
 * @paramOutput d_out
 *   data
 *
 * @paramTile tile_state
 *   status interface
 *
 * @paramThe start_tile
 *   starting tile for the current grid
 *
 * @paramBinary scan_op
 *   scan functor
 *
 * @paramInitial init_value
 *   value to seed the exclusive scan
 *
 * @paramTotal num_items
 *   number of scan items for the entire problem
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanTileStateT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
  __global__ void DeviceScanKernel(InputIteratorT d_in,
                                   OutputIteratorT d_out,
                                   ScanTileStateT tile_state,
                                   int start_tile,
                                   ScanOpT scan_op,
                                   InitValueT init_value,
                                   OffsetT num_items)
{
  using RealInitValueT = typename InitValueT::value_type;
  typedef typename ChainedPolicyT::ActivePolicy::ScanPolicyT ScanPolicyT;

  // Thread block type for scanning input tiles
  typedef AgentScan<ScanPolicyT,
                    InputIteratorT,
                    OutputIteratorT,
                    ScanOpT,
                    RealInitValueT,
                    OffsetT,
                    AccumT>
    AgentScanT;

  // Shared memory for AgentScan
  __shared__ typename AgentScanT::TempStorage temp_storage;

  RealInitValueT real_init_value = init_value;

  // Process tiles
  AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value)
    .ConsumeRange(num_items, tile_state, start_tile);
}

/******************************************************************************
 * Policy
 ******************************************************************************/

template <typename AccumT> ///< Data type
struct DeviceScanPolicy
{
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr bool LargeValues = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm ScanTransposedLoad =
    LargeValues ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED
                : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm ScanTransposedStore =
    LargeValues ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED
                : BLOCK_STORE_WARP_TRANSPOSE;

  /// SM350
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
    typedef AgentScanPolicy<128,
                            12, ///< Threads per block, items per thread
                            AccumT,
                            BLOCK_LOAD_DIRECT,
                            LOAD_CA,
                            BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                            BLOCK_SCAN_RAKING>
      ScanPolicyT;
  };

  /// SM520
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    // Titan X: 32.47B items/s @ 48M 32-bit T
    typedef AgentScanPolicy<128,
                            12, ///< Threads per block, items per thread
                            AccumT,
                            BLOCK_LOAD_DIRECT,
                            LOAD_CA,
                            ScanTransposedStore,
                            BLOCK_SCAN_WARP_SCANS>
      ScanPolicyT;
  };

  /// SM600
  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    typedef AgentScanPolicy<128,
                            15, ///< Threads per block, items per thread
                            AccumT,
                            ScanTransposedLoad,
                            LOAD_DEFAULT,
                            ScanTransposedStore,
                            BLOCK_SCAN_WARP_SCANS>
      ScanPolicyT;
  };

  /// MaxPolicy
  typedef Policy600 MaxPolicy;
};

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceScan
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading scan inputs \iterator
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type for writing scan outputs \iterator
 *
 * @tparam ScanOpT
 *   Binary scan functor type having member 
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   The init_value element type for ScanOpT (cub::NullType for inclusive scans)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT = 
            detail::accumulator_t<
              ScanOpT, 
              cub::detail::conditional_t<
                std::is_same<InitValueT, NullType>::value,
                cub::detail::value_t<InputIteratorT>,
                typename InitValueT::value_type>,
              cub::detail::value_t<InputIteratorT>>,
          typename SelectedPolicy = DeviceScanPolicy<AccumT>>
struct DispatchScan : SelectedPolicy
{
  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  // The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  /// Device-accessible allocation of temporary storage.  When NULL, the
  /// required allocation size is written to \p temp_storage_bytes and no work
  /// is done.
  void *d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  size_t &temp_storage_bytes;

  /// Iterator to the input sequence of data items
  InputIteratorT d_in;

  /// Iterator to the output sequence of data items
  OutputIteratorT d_out;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Total number of input items (i.e., the length of \p d_in)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  /**
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ DispatchScan(void *d_temp_storage,
                                                    size_t &temp_storage_bytes,
                                                    InputIteratorT d_in,
                                                    OutputIteratorT d_out,
                                                    OffsetT num_items,
                                                    ScanOpT scan_op,
                                                    InitValueT init_value,
                                                    cudaStream_t stream,
                                                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ DispatchScan(void *d_temp_storage,
                                                    size_t &temp_storage_bytes,
                                                    InputIteratorT d_in,
                                                    OutputIteratorT d_out,
                                                    OffsetT num_items,
                                                    ScanOpT scan_op,
                                                    InitValueT init_value,
                                                    cudaStream_t stream,
                                                    bool debug_synchronous,
                                                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  template <typename ActivePolicyT, typename InitKernel, typename ScanKernel>
  CUB_RUNTIME_FUNCTION __host__ __forceinline__ cudaError_t
  Invoke(InitKernel init_kernel, ScanKernel scan_kernel)
  {
    typedef typename ActivePolicyT::ScanPolicyT Policy;
    typedef typename cub::ScanTileState<AccumT> ScanTileStateT;

    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    static_assert(Policy::LOAD_MODIFIER != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");

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
      int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
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

      // Return if empty problem
      if (num_items == 0)
      {
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
      int init_grid_size = cub::DivideAndRoundUp(num_tiles,
                                                 INIT_KERNEL_THREADS);

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
        .doit(init_kernel, tile_state, num_tiles);

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

      // Get SM occupancy for scan_kernel
      int scan_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(scan_sm_occupancy, // out
                                          scan_kernel,
                                          Policy::BLOCK_THREADS)))
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
        // Log scan_kernel configuration
        #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
          _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items "
                  "per thread, %d SM occupancy\n",
                  start_tile,
                  scan_grid_size,
                  Policy::BLOCK_THREADS,
                  (long long)stream,
                  Policy::ITEMS_PER_THREAD,
                  scan_sm_occupancy);
        #endif

        // Invoke scan_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          scan_grid_size,
          Policy::BLOCK_THREADS,
          0,
          stream)
          .doit(scan_kernel,
                d_in,
                d_out,
                tile_state,
                start_tile,
                scan_op,
                init_value,
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

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __host__ __forceinline__ cudaError_t Invoke()
  {
    typedef typename DispatchScan::MaxPolicy MaxPolicyT;
    typedef typename cub::ScanTileState<AccumT> ScanTileStateT;
    // Ensure kernels are instantiated.
    return Invoke<ActivePolicyT>(DeviceScanInitKernel<ScanTileStateT>,
                                 DeviceScanKernel<MaxPolicyT,
                                                  InputIteratorT,
                                                  OutputIteratorT,
                                                  ScanTileStateT,
                                                  ScanOpT,
                                                  InitValueT,
                                                  OffsetT,
                                                  AccumT>);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   *
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           InputIteratorT d_in,
           OutputIteratorT d_out,
           ScanOpT scan_op,
           InitValueT init_value,
           OffsetT num_items,
           cudaStream_t stream)
  {
    typedef typename DispatchScan::MaxPolicy MaxPolicyT;

    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Create dispatch functor
      DispatchScan dispatch(d_temp_storage,
                            temp_storage_bytes,
                            d_in,
                            d_out,
                            num_items,
                            scan_op,
                            init_value,
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
           ScanOpT scan_op,
           InitValueT init_value,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_in,
                    d_out,
                    scan_op,
                    init_value,
                    num_items,
                    stream);
  }
};

CUB_NAMESPACE_END
