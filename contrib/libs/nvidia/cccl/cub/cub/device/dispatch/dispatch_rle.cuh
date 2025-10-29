/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * @file
 *   cub::DeviceRle provides device-wide, parallel operations for run-length-encoding sequences of
 *   data items residing within device-accessible memory.
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

#include <cub/agent/agent_rle.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_run_length_encode.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm_>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::rle
{

/**
 * Select kernel entry point (multi-block)
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam AgentRlePolicyT
 *   Parameterized AgentRlePolicyT tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetsOutputIteratorT
 *   Random-access output iterator type for writing run-offset values @iterator
 *
 * @tparam LengthsOutputIteratorT
 *   Random-access output iterator type for writing run-length values @iterator
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording the number of runs encountered @iterator
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   T equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_in
 *   Pointer to input sequence of data items
 *
 * @param d_offsets_out
 *   Pointer to output sequence of run-offsets
 *
 * @param d_lengths_out
 *   Pointer to output sequence of run-lengths
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs (i.e., length of `d_offsets_out`)
 *
 * @param tile_status
 *   Tile status interface
 *
 * @param equality_op
 *   Equality operator for input items
 *
 * @param num_items
 *   Total number of input items (i.e., length of `d_in`)
 *
 * @param num_tiles
 *   Total number of tiles for the entire problem
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetsOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::RleSweepPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRleSweepKernel(
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ScanTileStateT tile_status,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles)
{
  using AgentRlePolicyT = typename ChainedPolicyT::ActivePolicy::RleSweepPolicyT;

  // Thread block type for selecting data from input tiles
  using AgentRleT =
    AgentRle<AgentRlePolicyT, InputIteratorT, OffsetsOutputIteratorT, LengthsOutputIteratorT, EqualityOpT, OffsetT>;

  // Shared memory for AgentRle
  __shared__ typename AgentRleT::TempStorage temp_storage;

  // Process tiles
  AgentRleT(temp_storage, d_in, d_offsets_out, d_lengths_out, equality_op, num_items)
    .ConsumeRange(num_tiles, tile_status, d_num_runs_out);
}
} // namespace detail::rle

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRle
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetsOutputIteratorT
 *   Random-access output iterator type for writing run-offset values @iterator
 *
 * @tparam LengthsOutputIteratorT
 *   Random-access output iterator type for writing run-length values @iterator
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording the number of runs encountered @iterator
 *
 * @tparam EqualityOpT
 *   T equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <typename InputIteratorT,
          typename OffsetsOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT,
          typename PolicyHub =
            detail::rle::non_trivial_runs::policy_hub<cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>,
                                                      cub::detail::it_value_t<InputIteratorT>>>
struct DeviceRleDispatch
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  // The lengths output value type
  using LengthT = cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>;

  enum
  {
    INIT_KERNEL_THREADS = 128,
  };

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<LengthT, OffsetT>;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputIteratorT d_in;
  OffsetsOutputIteratorT d_offsets_out;
  LengthsOutputIteratorT d_lengths_out;
  NumRunsOutputIteratorT d_num_runs_out;
  EqualityOpT equality_op;
  OffsetT num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DeviceRleDispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_offsets_out(d_offsets_out)
      , d_lengths_out(d_lengths_out)
      , d_num_runs_out(d_num_runs_out)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide run-length-encode using the
   * specified kernel functions.
   *
   * @tparam DeviceScanInitKernelPtr
   *   Function type of cub::DeviceScanInitKernel
   *
   * @tparam DeviceRleSweepKernelPtr
   *   Function type of cub::DeviceRleSweepKernelPtr
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_offsets_out
   *   Pointer to the output sequence of run-offsets
   *
   * @param d_lengths_out
   *   Pointer to the output sequence of run-lengths
   *
   * @param d_num_runs_out
   *   Pointer to the total number of runs encountered (i.e., length of `d_offsets_out`)
   *
   * @param equality_op
   *   Equality operator for input items
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   *
   * @param ptx_version
   *   PTX version of dispatch kernels
   *
   * @param device_scan_init_kernel
   *   Kernel function pointer to parameterization of cub::DeviceScanInitKernel
   *
   * @param device_rle_sweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRleSweepKernel
   */
  template <typename ActivePolicyT, typename DeviceScanInitKernelPtr, typename DeviceRleSweepKernelPtr>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(DeviceScanInitKernelPtr device_scan_init_kernel, DeviceRleSweepKernelPtr device_rle_sweep_kernel)
  {
    cudaError error = cudaSuccess;

    constexpr int block_threads    = ActivePolicyT::RleSweepPolicyT::BLOCK_THREADS;
    constexpr int items_per_thread = ActivePolicyT::RleSweepPolicyT::ITEMS_PER_THREAD;

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = block_threads * items_per_thread;
      int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break; // bytes needed for tile status descriptors
      }

      // Compute allocation pointers into the single storage blob (or compute the necessary size of
      // the blob)
      void* allocations[1] = {};

      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (error != cudaSuccess)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_status;
      error = CubDebug(tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log device_scan_init_kernel configuration
      int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(num_tiles, int{INIT_KERNEL_THREADS}));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking device_scan_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke device_scan_init_kernel to initialize tile descriptors and queue descriptors
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(device_scan_init_kernel, tile_status, num_tiles, d_num_runs_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Return if empty problem: note, we're initializing d_num_runs_out to 0 in device_scan_init_kernel above
      if (num_items <= 1)
      {
        break;
      }

      // Get SM occupancy for device_rle_sweep_kernel
      int device_rle_kernel_sm_occupancy;
      error = CubDebug(MaxSmOccupancy(device_rle_kernel_sm_occupancy, // out
                                      device_rle_sweep_kernel,
                                      block_threads));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get grid size for scanning tiles
      dim3 scan_grid_size;
      scan_grid_size.z = 1;
      scan_grid_size.y = ::cuda::ceil_div(num_tiles, max_dim_x);
      scan_grid_size.x = _CUDA_VSTD::min(num_tiles, max_dim_x);

// Log device_rle_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking device_rle_sweep_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per "
              "thread, %d SM occupancy\n",
              scan_grid_size.x,
              scan_grid_size.y,
              scan_grid_size.z,
              block_threads,
              (long long) stream,
              items_per_thread,
              device_rle_kernel_sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke device_rle_sweep_kernel
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(scan_grid_size, block_threads, 0, stream)
        .doit(device_rle_sweep_kernel,
              d_in,
              d_offsets_out,
              d_lengths_out,
              d_num_runs_out,
              tile_status,
              equality_op,
              num_items,
              num_tiles);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  template <class ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return Invoke<ActivePolicyT>(
      detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
      detail::rle::DeviceRleSweepKernel<
        typename PolicyHub::MaxPolicy,
        InputIteratorT,
        OffsetsOutputIteratorT,
        LengthsOutputIteratorT,
        NumRunsOutputIteratorT,
        ScanTileStateT,
        EqualityOpT,
        OffsetT>);
  }

  /**
   * Internal dispatch routine
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to input sequence of data items
   *
   * @param d_offsets_out
   *   Pointer to output sequence of run-offsets
   *
   * @param d_lengths_out
   *   Pointer to output sequence of run-lengths
   *
   * @param d_num_runs_out
   *   Pointer to total number of runs (i.e., length of `d_offsets_out`)
   *
   * @param equality_op
   *   Equality operator for input items
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      DeviceRleDispatch dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_offsets_out,
        d_lengths_out,
        d_num_runs_out,
        equality_op,
        num_items,
        stream);

      // Dispatch
      error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
