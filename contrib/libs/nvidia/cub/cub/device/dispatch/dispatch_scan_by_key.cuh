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
 * DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>

#include "../../agent/agent_scan_by_key.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../config.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_math.cuh"
#include "dispatch_scan.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN


/******************************************************************************
* Kernel entry points
*****************************************************************************/

/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename ChainedPolicyT,              ///< Chained tuning policy
    typename KeysInputIteratorT,          ///< Random-access input iterator type
    typename ValuesInputIteratorT,        ///< Random-access input iterator type
    typename ValuesOutputIteratorT,       ///< Random-access output iterator type
    typename ScanByKeyTileStateT,         ///< Tile status interface type
    typename EqualityOp,                  ///< Equality functor type
    typename ScanOpT,                     ///< Scan functor type
    typename InitValueT,                  ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>                     ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT::BLOCK_THREADS))
__global__ void DeviceScanByKeyKernel(
    KeysInputIteratorT    d_keys_in,          ///< Input keys data
    ValuesInputIteratorT  d_values_in,        ///< Input values data
    ValuesOutputIteratorT d_values_out,       ///< Output values data
    ScanByKeyTileStateT   tile_state,         ///< Tile status interface
    int                   start_tile,         ///< The starting tile for the current grid
    EqualityOp            equality_op,        ///< Binary equality functor
    ScanOpT               scan_op,            ///< Binary scan functor
    InitValueT            init_value,         ///< Initial value to seed the exclusive scan
    OffsetT               num_items)          ///< Total number of scan items for the entire problem
{
    typedef typename ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT ScanByKeyPolicyT;

    // Thread block type for scanning input tiles
    typedef AgentScanByKey<
        ScanByKeyPolicyT,
        KeysInputIteratorT,
        ValuesInputIteratorT,
        ValuesOutputIteratorT,
        EqualityOp,
        ScanOpT,
        InitValueT,
        OffsetT> AgentScanByKeyT;

    // Shared memory for AgentScanByKey
    __shared__ typename AgentScanByKeyT::TempStorage temp_storage;

    // Process tiles
    AgentScanByKeyT(
        temp_storage,
        d_keys_in,
        d_values_in,
        d_values_out,
        equality_op,
        scan_op,
        init_value
    ).ConsumeRange(
        num_items,
        tile_state,
        start_tile);
}


/******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename InitValueT>
struct DeviceScanByKeyPolicy
{
    using KeyT = cub::detail::value_t<KeysInputIteratorT>;
    using ValueT = cub::detail::conditional_t<
        std::is_same<InitValueT, NullType>::value,
        cub::detail::value_t<ValuesInputIteratorT>,
        InitValueT>;
    static constexpr size_t MaxInputBytes = (sizeof(KeyT) > sizeof(ValueT) ? sizeof(KeyT) : sizeof(ValueT));
    static constexpr size_t CombinedInputBytes = sizeof(KeyT) + sizeof(ValueT);

    // SM350
    struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
    {
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 6,
            ITEMS_PER_THREAD = ((MaxInputBytes <= 8) ? 6 :
                Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD, CombinedInputBytes)),
        };

        typedef AgentScanByKeyPolicy<
                128, ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_STORE_WARP_TRANSPOSE>
            ScanByKeyPolicyT;
    };

    // SM520
    struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
    {
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,

            ITEMS_PER_THREAD = ((MaxInputBytes <= 8) ? 9 :
                Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD, CombinedInputBytes)),
        };

        typedef AgentScanByKeyPolicy<
                256, ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_STORE_WARP_TRANSPOSE>
            ScanByKeyPolicyT;
    };

    /// MaxPolicy
    typedef Policy520 MaxPolicy;
};


/******************************************************************************
 * Dispatch
 ******************************************************************************/


/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename KeysInputIteratorT,          ///< Random-access input iterator type
    typename ValuesInputIteratorT,        ///< Random-access input iterator type
    typename ValuesOutputIteratorT,       ///< Random-access output iterator type
    typename EqualityOp,                  ///< Equality functor type
    typename ScanOpT,                     ///< Scan functor type
    typename InitValueT,                  ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT,                     ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceScanByKeyPolicy<KeysInputIteratorT, ValuesInputIteratorT, InitValueT>
>
struct DispatchScanByKey:
    SelectedPolicy
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The input key type
    using KeyT = cub::detail::value_t<KeysInputIteratorT>;

    // The input value type
    using InputT = cub::detail::value_t<ValuesInputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT if provided, otherwise the
    // input iterator's value type.
    using OutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 InputT,
                                 InitValueT>;

    void*                 d_temp_storage;         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&               temp_storage_bytes;     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    KeysInputIteratorT    d_keys_in;              ///< [in] Iterator to the input sequence of key items
    ValuesInputIteratorT  d_values_in;            ///< [in] Iterator to the input sequence of value items
    ValuesOutputIteratorT d_values_out;           ///< [out] Iterator to the input sequence of value items
    EqualityOp            equality_op;            ///< [in]Binary equality functor
    ScanOpT               scan_op;                ///< [in] Binary scan functor
    InitValueT            init_value;             ///< [in] Initial value to seed the exclusive scan
    OffsetT               num_items;              ///< [in] Total number of input items (i.e., the length of \p d_in)
    cudaStream_t          stream;                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                  debug_synchronous;
    int                   ptx_version;

    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchScanByKey(
        void*                 d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&               temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,              ///< [in] Iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,            ///< [in] Iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,           ///< [out] Iterator to the input sequence of value items
        EqualityOp            equality_op,            ///< [in] Binary equality functor
        ScanOpT               scan_op,                ///< [in] Binary scan functor
        InitValueT            init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT               num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t          stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous,
        int                   ptx_version
    ):
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys_in(d_keys_in),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
        equality_op(equality_op),
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
        typedef typename ActivePolicyT::ScanByKeyPolicyT Policy;
        typedef ReduceByKeyScanTileState<OutputT, OffsetT> ScanByKeyTileStateT;

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
            if (CubDebug(error = ScanByKeyTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

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
            ScanByKeyTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(init_kernel, tile_state, num_tiles);

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
                ).doit(
                    scan_kernel,
                    d_keys_in,
                    d_values_in,
                    d_values_out,
                    tile_state,
                    start_tile,
                    equality_op,
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
        typedef typename DispatchScanByKey::MaxPolicy MaxPolicyT;
        typedef ReduceByKeyScanTileState<OutputT, OffsetT> ScanByKeyTileStateT;
        // Ensure kernels are instantiated.
        return Invoke<ActivePolicyT>(
            DeviceScanInitKernel<ScanByKeyTileStateT>,
            DeviceScanByKeyKernel<
                MaxPolicyT, KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT,
                ScanByKeyTileStateT, EqualityOp, ScanOpT, InitValueT, OffsetT>
        );
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                 d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&               temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,              ///< [in] Iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,            ///< [in] Iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,           ///< [out] Iterator to the input sequence of value items
        EqualityOp            equality_op,            ///< [in] Binary equality functor
        ScanOpT               scan_op,                ///< [in] Binary scan functor
        InitValueT            init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT               num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t          stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous)
    {
        typedef typename DispatchScanByKey::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchScanByKey dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_keys_in,
                d_values_in,
                d_values_out,
                equality_op,
                scan_op,
                init_value,
                num_items,
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
