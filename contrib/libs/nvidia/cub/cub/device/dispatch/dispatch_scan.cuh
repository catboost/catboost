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
 * cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>

#include "../../agent/agent_scan.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../config.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_math.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename            ScanTileStateT>     ///< Tile status interface type
__global__ void DeviceScanInitKernel(
    ScanTileStateT      tile_state,         ///< [in] Tile status interface
    int                 num_tiles)          ///< [in] Number of tiles
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename                ScanTileStateT,         ///< Tile status interface type
    typename                NumSelectedIteratorT>   ///< Output iterator type for recording the number of items selected
__global__ void DeviceCompactInitKernel(
    ScanTileStateT          tile_state,             ///< [in] Tile status interface
    int                     num_tiles,              ///< [in] Number of tiles
    NumSelectedIteratorT    d_num_selected_out)     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);

    // Initialize d_num_selected_out
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
        *d_num_selected_out = 0;
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename            ChainedPolicyT,     ///< Chained tuning policy
    typename            InputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename            OutputIteratorT,    ///< Random-access output iterator type for writing scan outputs \iterator
    typename            ScanTileStateT,     ///< Tile status interface type
    typename            ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            InitValueT,         ///< Initial value to seed the exclusive scan (cub::NullType for inclusive scans)
    typename            OffsetT>            ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS))
__global__ void DeviceScanKernel(
    InputIteratorT      d_in,               ///< Input data
    OutputIteratorT     d_out,              ///< Output data
    ScanTileStateT      tile_state,         ///< Tile status interface
    int                 start_tile,         ///< The starting tile for the current grid
    ScanOpT             scan_op,            ///< Binary scan functor
    InitValueT          init_value,         ///< Initial value to seed the exclusive scan
    OffsetT             num_items)          ///< Total number of scan items for the entire problem
{
    using RealInitValueT = typename InitValueT::value_type;
    typedef typename ChainedPolicyT::ActivePolicy::ScanPolicyT ScanPolicyT;

    // Thread block type for scanning input tiles
    typedef AgentScan<
        ScanPolicyT,
        InputIteratorT,
        OutputIteratorT,
        ScanOpT,
        RealInitValueT,
        OffsetT> AgentScanT;

    // Shared memory for AgentScan
    __shared__ typename AgentScanT::TempStorage temp_storage;

    RealInitValueT real_init_value = init_value;

    // Process tiles
    AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value).ConsumeRange(
        num_items,
        tile_state,
        start_tile);
}


/******************************************************************************
 * Policy
 ******************************************************************************/

template <
    typename OutputT> ///< Data type
struct DeviceScanPolicy
{
    // For large values, use timesliced loads/stores to fit shared memory.
    static constexpr bool LargeValues = sizeof(OutputT) > 128;
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
        typedef AgentScanPolicy<
                128, 12,                                        ///< Threads per block, items per thread
                OutputT,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
                BLOCK_SCAN_RAKING>
            ScanPolicyT;
    };

    /// SM520
    struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
    {
        // Titan X: 32.47B items/s @ 48M 32-bit T
        typedef AgentScanPolicy<
                128, 12,                                        ///< Threads per block, items per thread
                OutputT,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                ScanTransposedStore,
                BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };

    /// SM600
    struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
    {
        typedef AgentScanPolicy<
                128, 15,                                        ///< Threads per block, items per thread
                OutputT,
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
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename InputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename OutputIteratorT,    ///< Random-access output iterator type for writing scan outputs \iterator
    typename ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename InitValueT,         ///< The init_value element type for ScanOpT (cub::NullType for inclusive scans)
    typename OffsetT,            ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceScanPolicy<
      // Accumulator type.
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 cub::detail::value_t<InputIteratorT>,
                                 typename InitValueT::value_type>>>
struct DispatchScan:
    SelectedPolicy
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT::value_type if provided, otherwise the
    // input iterator's value type.
    using OutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 InputT,
                                 typename InitValueT::value_type>;

    void*           d_temp_storage;         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&         temp_storage_bytes;     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT  d_in;                   ///< [in] Iterator to the input sequence of data items
    OutputIteratorT d_out;                  ///< [out] Iterator to the output sequence of data items
    ScanOpT         scan_op;                ///< [in] Binary scan functor
    InitValueT      init_value;             ///< [in] Initial value to seed the exclusive scan
    OffsetT         num_items;              ///< [in] Total number of input items (i.e., the length of \p d_in)
    cudaStream_t    stream;                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool            debug_synchronous;
    int             ptx_version;

    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchScan(
        void*           d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                   ///< [in] Iterator to the input sequence of data items
        OutputIteratorT d_out,                  ///< [out] Iterator to the output sequence of data items
        OffsetT         num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        ScanOpT         scan_op,                ///< [in] Binary scan functor
        InitValueT      init_value,             ///< [in] Initial value to seed the exclusive scan
        cudaStream_t    stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous,
        int             ptx_version
    ):
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        init_value(init_value),
        num_items(num_items),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}

    template <typename ActivePolicyT, typename InitKernel, typename ScanKernel>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke(InitKernel init_kernel, ScanKernel scan_kernel)
    {
#ifndef CUB_RUNTIME_ENABLED

        (void)init_kernel;
        (void)scan_kernel;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else

        typedef typename ActivePolicyT::ScanPolicyT Policy;
        typedef typename cub::ScanTileState<OutputT> ScanTileStateT;

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
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

            // Return if empty problem
            if (num_items == 0)
                break;

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(init_kernel,
                tile_state,
                num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;


            // Get SM occupancy for scan_kernel
            int scan_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                scan_sm_occupancy,            // out
                scan_kernel,
                Policy::BLOCK_THREADS))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Run grids in epochs (in case number of tiles exceeds max x-dimension
            int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
            for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
            {
                // Log scan_kernel configuration
                if (debug_synchronous) _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    start_tile, scan_grid_size, Policy::BLOCK_THREADS, (long long) stream, Policy::ITEMS_PER_THREAD, scan_sm_occupancy);

                // Invoke scan_kernel
                THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                    scan_grid_size, Policy::BLOCK_THREADS, 0, stream
                ).doit(scan_kernel,
                    d_in,
                    d_out,
                    tile_state,
                    start_tile,
                    scan_op,
                    init_value,
                    num_items);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke()
    {
        typedef typename DispatchScan::MaxPolicy MaxPolicyT;
        typedef typename cub::ScanTileState<OutputT> ScanTileStateT;
        // Ensure kernels are instantiated.
        return Invoke<ActivePolicyT>(
            DeviceScanInitKernel<ScanTileStateT>,
            DeviceScanKernel<MaxPolicyT, InputIteratorT, OutputIteratorT, ScanTileStateT, ScanOpT, InitValueT, OffsetT>
        );
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*           d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                   ///< [in] Iterator to the input sequence of data items
        OutputIteratorT d_out,                  ///< [out] Iterator to the output sequence of data items
        ScanOpT         scan_op,                ///< [in] Binary scan functor
        InitValueT      init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT         num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename DispatchScan::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchScan dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                num_items,
                scan_op,
                init_value,
                stream,
                debug_synchronous,
                ptx_version
            );
            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};



CUB_NAMESPACE_END
