/******************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * @file DeviceScan provides device-wide, parallel operations for computing a 
 *       prefix scan across a sequence of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>

#include <cub/agent/agent_scan_by_key.cuh>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
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
 * @brief Scan kernel entry point (multi-block)
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ScanByKeyTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOp
 *   Equality functor type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_keys_in
 *   Input keys data
 *
 * @param d_keys_prev_in
 *   Predecessor items for each tile
 *
 * @param d_values_in
 *   Input values data
 *
 * @param d_values_out
 *   Output values data
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   Binary equality functor
 *
 * @param scan_op
 *   Binary scan functor
 *
 * @param init_value
 *   Initial value to seed the exclusive scan
 *
 * @param num_items
 *   Total number of scan items for the entire problem
 */
template <typename ChainedPolicyT,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename ScanByKeyTileStateT,
          typename EqualityOp,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          typename KeyT = cub::detail::value_t<KeysInputIteratorT>>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT::BLOCK_THREADS)) 
__global__ void DeviceScanByKeyKernel(KeysInputIteratorT d_keys_in,
                                      KeyT *d_keys_prev_in,
                                      ValuesInputIteratorT d_values_in,
                                      ValuesOutputIteratorT d_values_out,
                                      ScanByKeyTileStateT tile_state,
                                      int start_tile,
                                      EqualityOp equality_op,
                                      ScanOpT scan_op,
                                      InitValueT init_value,
                                      OffsetT num_items)
{
  using ScanByKeyPolicyT =
    typename ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT;

  // Thread block type for scanning input tiles
  using AgentScanByKeyT = AgentScanByKey<ScanByKeyPolicyT,
                                         KeysInputIteratorT,
                                         ValuesInputIteratorT,
                                         ValuesOutputIteratorT,
                                         EqualityOp,
                                         ScanOpT,
                                         InitValueT,
                                         OffsetT,
                                         AccumT>;

  // Shared memory for AgentScanByKey
  __shared__ typename AgentScanByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentScanByKeyT(temp_storage,
                  d_keys_in,
                  d_keys_prev_in,
                  d_values_in,
                  d_values_out,
                  equality_op,
                  scan_op,
                  init_value)
    .ConsumeRange(num_items, tile_state, start_tile);
}

template <typename ScanTileStateT, typename KeysInputIteratorT>
__global__ void DeviceScanByKeyInitKernel(
  ScanTileStateT tile_state,
  KeysInputIteratorT d_keys_in,
  cub::detail::value_t<KeysInputIteratorT> *d_keys_prev_in,
  unsigned items_per_tile,
  int num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  const unsigned tid       = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned tile_base = tid * items_per_tile;

  if (tid > 0 && tid < num_tiles)
  {
    d_keys_prev_in[tid] = d_keys_in[tile_base - 1];
  }
}

/******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeysInputIteratorT,
          typename AccumT>
struct DeviceScanByKeyPolicy
{
  using KeyT = cub::detail::value_t<KeysInputIteratorT>;

  static constexpr size_t MaxInputBytes = (cub::max)(sizeof(KeyT),
                                                     sizeof(AccumT));

  static constexpr size_t CombinedInputBytes = sizeof(KeyT) + sizeof(AccumT);

  // SM350
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 6;
    static constexpr int ITEMS_PER_THREAD =
      ((MaxInputBytes <= 8)
         ? 6
         : Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD,
                                         CombinedInputBytes));

    using ScanByKeyPolicyT = AgentScanByKeyPolicy<128,
                                                  ITEMS_PER_THREAD,
                                                  BLOCK_LOAD_WARP_TRANSPOSE,
                                                  LOAD_CA,
                                                  BLOCK_SCAN_WARP_SCANS,
                                                  BLOCK_STORE_WARP_TRANSPOSE>;
  };

  // SM520
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 9;
    static constexpr int ITEMS_PER_THREAD =
      ((MaxInputBytes <= 8)
         ? 9
         : Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD,
                                         CombinedInputBytes));

    using ScanByKeyPolicyT = AgentScanByKeyPolicy<256,
                                                  ITEMS_PER_THREAD,
                                                  BLOCK_LOAD_WARP_TRANSPOSE,
                                                  LOAD_CA,
                                                  BLOCK_SCAN_WARP_SCANS,
                                                  BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy520;
};

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels
 *        for DeviceScan
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam EqualityOp
 *   Equality functor type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 */
template <
  typename KeysInputIteratorT,
  typename ValuesInputIteratorT,
  typename ValuesOutputIteratorT,
  typename EqualityOp,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT = 
    detail::accumulator_t<
      ScanOpT, 
      cub::detail::conditional_t<
        std::is_same<InitValueT, NullType>::value,
        cub::detail::value_t<ValuesInputIteratorT>,
        InitValueT>,
      cub::detail::value_t<ValuesInputIteratorT>>,
  typename SelectedPolicy =
    DeviceScanByKeyPolicy<KeysInputIteratorT, AccumT>>
struct DispatchScanByKey : SelectedPolicy
{
  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  // The input key type
  using KeyT = cub::detail::value_t<KeysInputIteratorT>;

  // The input value type
  using InputT = cub::detail::value_t<ValuesInputIteratorT>;

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void *d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t &temp_storage_bytes;

  /// Iterator to the input sequence of key items
  KeysInputIteratorT d_keys_in;

  /// Iterator to the input sequence of value items
  ValuesInputIteratorT d_values_in;

  /// Iterator to the input sequence of value items
  ValuesOutputIteratorT d_values_out;

  /// Binary equality functor
  EqualityOp equality_op;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Total number of input items (i.e., the length of `d_in`)
  OffsetT num_items;

  /// CUDA stream to launch kernels within.
  cudaStream_t stream;
  int ptx_version;

  /**
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Iterator to the input sequence of key items
   *
   * @param[in] d_values_in
   *   Iterator to the input sequence of value items
   *
   * @param[out] d_values_out
   *   Iterator to the input sequence of value items
   *
   * @param[in] equality_op
   *   Binary equality functor
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
   *   CUDA stream to launch kernels within.
   */
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchScanByKey(void *d_temp_storage,
                    size_t &temp_storage_bytes,
                    KeysInputIteratorT d_keys_in,
                    ValuesInputIteratorT d_values_in,
                    ValuesOutputIteratorT d_values_out,
                    EqualityOp equality_op,
                    ScanOpT scan_op,
                    InitValueT init_value,
                    OffsetT num_items,
                    cudaStream_t stream,
                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , equality_op(equality_op)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchScanByKey(void *d_temp_storage,
                    size_t &temp_storage_bytes,
                    KeysInputIteratorT d_keys_in,
                    ValuesInputIteratorT d_values_in,
                    ValuesOutputIteratorT d_values_out,
                    EqualityOp equality_op,
                    ScanOpT scan_op,
                    InitValueT init_value,
                    OffsetT num_items,
                    cudaStream_t stream,
                    bool debug_synchronous,
                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , equality_op(equality_op)
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
    using Policy = typename ActivePolicyT::ScanByKeyPolicyT;
    using ScanByKeyTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

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
      size_t allocation_sizes[2];
      if (CubDebug(
            error = ScanByKeyTileStateT::AllocationSize(num_tiles,
                                                        allocation_sizes[0])))
      {
        break; // bytes needed for tile status descriptors
      }

      allocation_sizes[1] = sizeof(KeyT) * (num_tiles + 1);

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void *allocations[2] = {};
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

      KeyT *d_keys_prev_in = reinterpret_cast<KeyT *>(allocations[1]);

      // Construct the tile status interface
      ScanByKeyTileStateT tile_state;
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
        .doit(init_kernel,
              tile_state,
              d_keys_in,
              d_keys_prev_in,
              tile_size,
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
                d_keys_in,
                d_keys_prev_in,
                d_values_in,
                d_values_out,
                tile_state,
                start_tile,
                equality_op,
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
    using MaxPolicyT = typename DispatchScanByKey::MaxPolicy;
    using ScanByKeyTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

    // Ensure kernels are instantiated.
    return Invoke<ActivePolicyT>(
      DeviceScanByKeyInitKernel<ScanByKeyTileStateT, KeysInputIteratorT>,
      DeviceScanByKeyKernel<MaxPolicyT,
                            KeysInputIteratorT,
                            ValuesInputIteratorT,
                            ValuesOutputIteratorT,
                            ScanByKeyTileStateT,
                            EqualityOp,
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
   * @param[in] d_keys_in
   *   Iterator to the input sequence of key items
   *
   * @param[in] d_values_in
   *   Iterator to the input sequence of value items
   *
   * @param[out] d_values_out
   *   Iterator to the input sequence of value items
   *
   * @param[in] equality_op
   *   Binary equality functor
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
   *   CUDA stream to launch kernels within.
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           KeysInputIteratorT d_keys_in,
           ValuesInputIteratorT d_values_in,
           ValuesOutputIteratorT d_values_out,
           EqualityOp equality_op,
           ScanOpT scan_op,
           InitValueT init_value,
           OffsetT num_items,
           cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchScanByKey::MaxPolicy;

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
      DispatchScanByKey dispatch(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_values_in,
                                 d_values_out,
                                 equality_op,
                                 scan_op,
                                 init_value,
                                 num_items,
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
           KeysInputIteratorT d_keys_in,
           ValuesInputIteratorT d_values_in,
           ValuesOutputIteratorT d_values_out,
           EqualityOp equality_op,
           ScanOpT scan_op,
           InitValueT init_value,
           OffsetT num_items,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_keys_in,
                    d_values_in,
                    d_values_out,
                    equality_op,
                    scan_op,
                    init_value,
                    num_items,
                    stream);
  }
};

CUB_NAMESPACE_END
