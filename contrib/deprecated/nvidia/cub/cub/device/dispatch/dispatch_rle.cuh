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
 * \file
 * cub::DeviceRle provides device-wide, parallel operations for run-length-encoding sequences of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <cub/agent/agent_rle.cuh>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <nv/target>

#include <cstdio>
#include <iterator>

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Select kernel entry point (multi-block)
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename            AgentRlePolicyT,        ///< Parameterized AgentRlePolicyT tuning policy type
    typename            InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename            OffsetsOutputIteratorT,     ///< Random-access output iterator type for writing run-offset values \iterator
    typename            LengthsOutputIteratorT,     ///< Random-access output iterator type for writing run-length values \iterator
    typename            NumRunsOutputIteratorT,     ///< Output iterator type for recording the number of runs encountered \iterator
    typename            ScanTileStateT,              ///< Tile status interface type
    typename            EqualityOpT,                 ///< T equality operator type
    typename            OffsetT>                    ///< Signed integer type for global offsets
__launch_bounds__ (int(AgentRlePolicyT::BLOCK_THREADS))
__global__ void DeviceRleSweepKernel(
    InputIteratorT              d_in,               ///< [in] Pointer to input sequence of data items
    OffsetsOutputIteratorT      d_offsets_out,      ///< [out] Pointer to output sequence of run-offsets
    LengthsOutputIteratorT      d_lengths_out,      ///< [out] Pointer to output sequence of run-lengths
    NumRunsOutputIteratorT      d_num_runs_out,     ///< [out] Pointer to total number of runs (i.e., length of \p d_offsets_out)
    ScanTileStateT              tile_status,        ///< [in] Tile status interface
    EqualityOpT                 equality_op,        ///< [in] Equality operator for input items
    OffsetT                     num_items,          ///< [in] Total number of input items (i.e., length of \p d_in)
    int                         num_tiles)          ///< [in] Total number of tiles for the entire problem
{
    // Thread block type for selecting data from input tiles
    typedef AgentRle<
        AgentRlePolicyT,
        InputIteratorT,
        OffsetsOutputIteratorT,
        LengthsOutputIteratorT,
        EqualityOpT,
        OffsetT> AgentRleT;

    // Shared memory for AgentRle
    __shared__ typename AgentRleT::TempStorage temp_storage;

    // Process tiles
    AgentRleT(temp_storage, d_in, d_offsets_out, d_lengths_out, equality_op, num_items).ConsumeRange(
        num_tiles,
        tile_status,
        d_num_runs_out);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRle
 */
template <
    typename            InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename            OffsetsOutputIteratorT,     ///< Random-access output iterator type for writing run-offset values \iterator
    typename            LengthsOutputIteratorT,     ///< Random-access output iterator type for writing run-length values \iterator
    typename            NumRunsOutputIteratorT,     ///< Output iterator type for recording the number of runs encountered \iterator
    typename            EqualityOpT,                ///< T equality operator type
    typename            OffsetT>                    ///< Signed integer type for global offsets
struct DeviceRleDispatch
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    // The input value type
    using T = cub::detail::value_t<InputIteratorT>;

    // The lengths output value type
    using LengthT =
      cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>;

    enum
    {
        INIT_KERNEL_THREADS = 128,
    };

    // Tile status descriptor interface type
    using ScanTileStateT = ReduceByKeyScanTileState<LengthT, OffsetT>;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 15,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef AgentRlePolicy<
                96,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                true,
                BLOCK_SCAN_WARP_SCANS>
            RleSweepPolicy;
    };

    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

    typedef Policy350 PtxPolicy;

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxRleSweepPolicy : PtxPolicy::RleSweepPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             /*ptx_version*/,
        KernelConfig&   device_rle_config)
    {
      NV_IF_TARGET(NV_IS_DEVICE,
      (
          // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
          device_rle_config.template Init<PtxRleSweepPolicy>();
      ), (
          // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version

          // (There's only one policy right now)
          device_rle_config.template Init<typename Policy350::RleSweepPolicy>();
      ));
    }


    /**
     * Kernel kernel dispatch configuration.  Mirrors the constants within AgentRlePolicyT.
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_policy;
        bool                    store_warp_time_slicing;
        BlockScanAlgorithm      scan_algorithm;

        template <typename AgentRlePolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = AgentRlePolicyT::BLOCK_THREADS;
            items_per_thread            = AgentRlePolicyT::ITEMS_PER_THREAD;
            load_policy                 = AgentRlePolicyT::LOAD_ALGORITHM;
            store_warp_time_slicing     = AgentRlePolicyT::STORE_WARP_TIME_SLICING;
            scan_algorithm              = AgentRlePolicyT::SCAN_ALGORITHM;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                load_policy,
                store_warp_time_slicing,
                scan_algorithm);
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide run-length-encode using the
     * specified kernel functions.
     */
    template <
        typename                    DeviceScanInitKernelPtr,        ///< Function type of cub::DeviceScanInitKernel
        typename                    DeviceRleSweepKernelPtr>        ///< Function type of cub::DeviceRleSweepKernelPtr
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                       d_temp_storage,                 ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
        OffsetsOutputIteratorT      d_offsets_out,                  ///< [out] Pointer to the output sequence of run-offsets
        LengthsOutputIteratorT      d_lengths_out,                  ///< [out] Pointer to the output sequence of run-lengths
        NumRunsOutputIteratorT      d_num_runs_out,                 ///< [out] Pointer to the total number of runs encountered (i.e., length of \p d_offsets_out)
        EqualityOpT                 equality_op,                    ///< [in] Equality operator for input items
        OffsetT                     num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        int                         /*ptx_version*/,                ///< [in] PTX version of dispatch kernels
        DeviceScanInitKernelPtr     device_scan_init_kernel,        ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        DeviceRleSweepKernelPtr     device_rle_sweep_kernel,        ///< [in] Kernel function pointer to parameterization of cub::DeviceRleSweepKernel
        KernelConfig                device_rle_config)              ///< [in] Dispatch parameters that match the policy that \p device_rle_sweep_kernel was compiled for
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = device_rle_config.block_threads * device_rle_config.items_per_thread;
            int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[1];
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void* allocations[1] = {};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Construct the tile status interface
            ScanTileStateT tile_status;
            if (CubDebug(error = tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log device_scan_init_kernel configuration
            int init_grid_size = CUB_MAX(1, cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));

            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking device_scan_init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
            #endif

            // Invoke device_scan_init_kernel to initialize tile descriptors and queue descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(device_scan_init_kernel,
                tile_status,
                num_tiles,
                d_num_runs_out);

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

            // Get SM occupancy for device_rle_sweep_kernel
            int device_rle_kernel_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                device_rle_kernel_sm_occupancy,            // out
                device_rle_sweep_kernel,
                device_rle_config.block_threads))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Get grid size for scanning tiles
            dim3 scan_grid_size;
            scan_grid_size.z = 1;
            scan_grid_size.y = cub::DivideAndRoundUp(num_tiles, max_dim_x);
            scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

            // Log device_rle_sweep_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking device_rle_sweep_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                scan_grid_size.x, scan_grid_size.y, scan_grid_size.z, device_rle_config.block_threads, (long long) stream, device_rle_config.items_per_thread, device_rle_kernel_sm_occupancy);
            #endif

            // Invoke device_rle_sweep_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                scan_grid_size, device_rle_config.block_threads, 0, stream
            ).doit(device_rle_sweep_kernel,
                d_in,
                d_offsets_out,
                d_lengths_out,
                d_num_runs_out,
                tile_status,
                equality_op,
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


    template <typename DeviceScanInitKernelPtr, typename DeviceRleSweepKernelPtr>
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    Dispatch(void *d_temp_storage,
             size_t &temp_storage_bytes,
             InputIteratorT d_in,
             OffsetsOutputIteratorT d_offsets_out,
             LengthsOutputIteratorT d_lengths_out,
             NumRunsOutputIteratorT d_num_runs_out,
             EqualityOpT equality_op,
             OffsetT num_items,
             cudaStream_t stream,
             bool debug_synchronous,
             int ptx_version,
             DeviceScanInitKernelPtr device_scan_init_kernel,
             DeviceRleSweepKernelPtr device_rle_sweep_kernel,
             KernelConfig device_rle_config)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch<DeviceScanInitKernelPtr, DeviceRleSweepKernelPtr>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_offsets_out,
        d_lengths_out,
        d_num_runs_out,
        equality_op,
        num_items,
        stream,
        ptx_version,
        device_scan_init_kernel,
        device_rle_sweep_kernel,
        device_rle_config);
    }

    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                       d_temp_storage,                 ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to input sequence of data items
        OffsetsOutputIteratorT      d_offsets_out,                  ///< [out] Pointer to output sequence of run-offsets
        LengthsOutputIteratorT      d_lengths_out,                  ///< [out] Pointer to output sequence of run-lengths
        NumRunsOutputIteratorT      d_num_runs_out,                 ///< [out] Pointer to total number of runs (i.e., length of \p d_offsets_out)
        EqualityOpT                 equality_op,                    ///< [in] Equality operator for input items
        OffsetT                     num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream)                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Get kernel kernel dispatch configurations
            KernelConfig device_rle_config;
            InitConfigs(ptx_version, device_rle_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_offsets_out,
                d_lengths_out,
                d_num_runs_out,
                equality_op,
                num_items,
                stream,
                ptx_version,
                DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
                DeviceRleSweepKernel<PtxRleSweepPolicy, InputIteratorT, OffsetsOutputIteratorT, LengthsOutputIteratorT, NumRunsOutputIteratorT, ScanTileStateT, EqualityOpT, OffsetT>,
                device_rle_config))) break;
        }
        while (0);

        return error;
    }

    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    Dispatch(void *d_temp_storage,
             size_t &temp_storage_bytes,
             InputIteratorT d_in,
             OffsetsOutputIteratorT d_offsets_out,
             LengthsOutputIteratorT d_lengths_out,
             NumRunsOutputIteratorT d_num_runs_out,
             EqualityOpT equality_op,
             OffsetT num_items,
             cudaStream_t stream,
             bool debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_offsets_out,
                      d_lengths_out,
                      d_num_runs_out,
                      equality_op,
                      num_items,
                      stream);
    }
};


CUB_NAMESPACE_END
