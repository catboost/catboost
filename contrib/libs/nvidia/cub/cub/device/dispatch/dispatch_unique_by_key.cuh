
/******************************************************************************
 * Copyright (c) NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSelect::UniqueByKey provides device-wide, parallel operations for selecting unique items by key from sequences of data items residing within device-accessible memory.
 */

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Unique by key kernel entry point (multi-block)
 */
template <
    typename AgentUniqueByKeyPolicyT,               ///< Parameterized AgentUniqueByKeyPolicy tuning policy type
    typename KeyInputIteratorT,                     ///< Random-access input iterator type for keys
    typename ValueInputIteratorT,                   ///< Random-access input iterator type for values
    typename KeyOutputIteratorT,                    ///< Random-access output iterator type for keys
    typename ValueOutputIteratorT,                  ///< Random-access output iterator type for values
    typename NumSelectedIteratorT,                  ///< Output iterator type for recording the number of items selected
    typename ScanTileStateT,                        ///< Tile status interface type
    typename EqualityOpT,                           ///< Equality operator type
    typename OffsetT>                               ///< Signed integer type for global offsets
__launch_bounds__ (int(AgentUniqueByKeyPolicyT::UniqueByKeyPolicyT::BLOCK_THREADS))
__global__ void DeviceUniqueByKeySweepKernel(
    KeyInputIteratorT       d_keys_in,              ///< [in] Pointer to the input sequence of keys
    ValueInputIteratorT     d_values_in,            ///< [in] Pointer to the input sequence of values
    KeyOutputIteratorT      d_keys_out,             ///< [out] Pointer to the output sequence of selected data items
    ValueOutputIteratorT    d_values_out,           ///< [out] Pointer to the output sequence of selected data items
    NumSelectedIteratorT    d_num_selected_out,     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_keys_out or \p d_values_out)
    ScanTileStateT          tile_state,             ///< [in] Tile status interface
    EqualityOpT             equality_op,            ///< [in] Equality operator
    OffsetT                 num_items,              ///< [in] Total number of input items (i.e., length of \p d_keys_in or \p d_values_in)
    int                     num_tiles)              ///< [in] Total number of tiles for the entire problem
{
    // Thread block type for selecting data from input tiles
    using AgentUniqueByKeyT = AgentUniqueByKey<
        typename AgentUniqueByKeyPolicyT::UniqueByKeyPolicyT,
        KeyInputIteratorT,
        ValueInputIteratorT,
        KeyOutputIteratorT,
        ValueOutputIteratorT,
        EqualityOpT,
        OffsetT>;

    // Shared memory for AgentUniqueByKey
    __shared__ typename AgentUniqueByKeyT::TempStorage temp_storage;

    // Process tiles
    AgentUniqueByKeyT(temp_storage, d_keys_in, d_values_in, d_keys_out, d_values_out, equality_op, num_items).ConsumeRange(
        num_tiles,
        tile_state,
        d_num_selected_out);
}


/******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeyInputIteratorT>
struct DeviceUniqueByKeyPolicy
{
    using KeyT = typename std::iterator_traits<KeyInputIteratorT>::value_type;

    // SM350
    struct Policy350 : ChainedPolicy<350, Policy350, Policy350> {
        const static int INPUT_SIZE = sizeof(KeyT);
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,
            ITEMS_PER_THREAD = Nominal4BItemsToItems<KeyT>(NOMINAL_4B_ITEMS_PER_THREAD),
        };

        using UniqueByKeyPolicyT = AgentUniqueByKeyPolicy<128,
                          ITEMS_PER_THREAD,
                          cub::BLOCK_LOAD_WARP_TRANSPOSE,
                          cub::LOAD_LDG,
                          cub::BLOCK_SCAN_WARP_SCANS>;
    };

    // SM520
    struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
    {
        const static int INPUT_SIZE = sizeof(KeyT);
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 11,
            ITEMS_PER_THREAD = Nominal4BItemsToItems<KeyT>(NOMINAL_4B_ITEMS_PER_THREAD),
        };

        using UniqueByKeyPolicyT =  AgentUniqueByKeyPolicy<64,
                            ITEMS_PER_THREAD,
                            cub::BLOCK_LOAD_WARP_TRANSPOSE,
                            cub::LOAD_LDG,
                            cub::BLOCK_SCAN_WARP_SCANS>;
    };

    /// MaxPolicy
    using MaxPolicy = Policy520;
};


/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSelect
 */
template <
    typename KeyInputIteratorT,                 ///< Random-access input iterator type for keys
    typename ValueInputIteratorT,               ///< Random-access input iterator type for values
    typename KeyOutputIteratorT,                ///< Random-access output iterator type for keys
    typename ValueOutputIteratorT,              ///< Random-access output iterator type for values
    typename NumSelectedIteratorT,              ///< Output iterator type for recording the number of items selected
    typename EqualityOpT,                       ///< Equality operator type
    typename OffsetT,                           ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceUniqueByKeyPolicy<KeyInputIteratorT>>
struct DispatchUniqueByKey: SelectedPolicy
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    enum
    {
        INIT_KERNEL_THREADS = 128,
    };

    // The input key and value type
    using KeyT = typename std::iterator_traits<KeyInputIteratorT>::value_type;
    using ValueT = typename std::iterator_traits<ValueInputIteratorT>::value_type;

    // Tile status descriptor interface type
    using ScanTileStateT = ScanTileState<OffsetT>;


    void*                   d_temp_storage;             ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&                 temp_storage_bytes;         ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    KeyInputIteratorT       d_keys_in;                  ///< [in] Pointer to the input sequence of keys
    ValueInputIteratorT     d_values_in;                ///< [in] Pointer to the input sequence of values
    KeyOutputIteratorT      d_keys_out;                 ///< [out] Pointer to the output sequence of selected data items
    ValueOutputIteratorT    d_values_out;               ///< [out] Pointer to the output sequence of selected data items
    NumSelectedIteratorT    d_num_selected_out;         ///< [out] Pointer to the total number of items selected (i.e., length of \p d_keys_out or \p d_values_out)
    EqualityOpT             equality_op;                ///< [in] Equality operator
    OffsetT                 num_items;                  ///< [in] Total number of input items (i.e., length of \p d_keys_in or \p d_values_in)
    cudaStream_t            stream;                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                    debug_synchronous;

    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchUniqueByKey(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeyInputIteratorT       d_keys_in,              ///< [in] Pointer to the input sequence of keys
        ValueInputIteratorT     d_values_in,            ///< [in] Pointer to the input sequence of values
        KeyOutputIteratorT      d_keys_out,             ///< [out] Pointer to the output sequence of selected data items
        ValueOutputIteratorT    d_values_out,           ///< [out] Pointer to the output sequence of selected data items
        NumSelectedIteratorT    d_num_selected_out,     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_keys_out or \p d_values_out)
        EqualityOpT             equality_op,            ///< [in] Equality operator
        OffsetT                 num_items,              ///< [in] Total number of input items (i.e., length of \p d_keys_in or \p d_values_in)
        cudaStream_t            stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous
    ):
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys_in(d_keys_in),
        d_values_in(d_values_in),
        d_keys_out(d_keys_out),
        d_values_out(d_values_out),
        d_num_selected_out(d_num_selected_out),
        equality_op(equality_op),
        num_items(num_items),
        stream(stream),
        debug_synchronous(debug_synchronous)
    {}


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

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

        using Policy = typename ActivePolicyT::UniqueByKeyPolicyT;
        using UniqueByKeyAgentT = AgentUniqueByKey<Policy,
                                                   KeyInputIteratorT,
                                                   ValueInputIteratorT,
                                                   KeyOutputIteratorT,
                                                   ValueOutputIteratorT,
                                                   EqualityOpT,
                                                   OffsetT>;

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
            int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

            // Size of virtual shared memory
            int max_shmem = 0;
            if (CubDebug(
                error = cudaDeviceGetAttribute(&max_shmem,
                                               cudaDevAttrMaxSharedMemoryPerBlock,
                                               device_ordinal)))
            {
                break;
            }
            std::size_t vshmem_size = detail::VshmemSize(max_shmem, sizeof(typename UniqueByKeyAgentT::TempStorage), num_tiles);

            // Specify temporary storage allocation requirements
            size_t allocation_sizes[2] = {0, vshmem_size};
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void *allocations[2] = {NULL, NULL};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            num_tiles = CUB_MAX(1, num_tiles);
            int init_grid_size = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(init_kernel, tile_state, num_tiles, d_num_selected_out);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Return if empty problem
            if (num_items == 0) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Get grid size for scanning tiles
            dim3 scan_grid_size;
            scan_grid_size.z = 1;
            scan_grid_size.y = cub::DivideAndRoundUp(num_tiles, max_dim_x);
            scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

            // Log select_if_kernel configuration
            if (debug_synchronous)
            {
              // Get SM occupancy for unique_by_key_kernel
              int scan_sm_occupancy;
              if (CubDebug(error = MaxSmOccupancy(scan_sm_occupancy, // out
                                                  scan_kernel,
                                                  Policy::BLOCK_THREADS)))
              {
                break;
              }

              _CubLog("Invoking unique_by_key_kernel<<<{%d,%d,%d}, %d, 0, "
                      "%lld>>>(), %d items per thread, %d SM occupancy\n",
                      scan_grid_size.x,
                      scan_grid_size.y,
                      scan_grid_size.z,
                      Policy::BLOCK_THREADS,
                      (long long)stream,
                      Policy::ITEMS_PER_THREAD,
                      scan_sm_occupancy);
            }

            // Invoke select_if_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                scan_grid_size, Policy::BLOCK_THREADS, 0, stream
            ).doit(scan_kernel,
                   d_keys_in,
                   d_values_in,
                   d_keys_out,
                   d_values_out,
                   d_num_selected_out,
                   tile_state,
                   equality_op,
                   num_items,
                   num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while(0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke()
    {
        // Ensure kernels are instantiated.
        return Invoke<ActivePolicyT>(
            DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>,
            DeviceUniqueByKeySweepKernel<
                ActivePolicyT,
                KeyInputIteratorT,
                ValueInputIteratorT,
                KeyOutputIteratorT,
                ValueOutputIteratorT,
                NumSelectedIteratorT,
                ScanTileStateT,
                EqualityOpT,
                OffsetT>
        );
    }


    /**
    * Internal dispatch routine
    */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeyInputIteratorT       d_keys_in,              ///< [in] Pointer to the input sequence of keys
        ValueInputIteratorT     d_values_in,            ///< [in] Pointer to the input sequence of values
        KeyOutputIteratorT      d_keys_out,             ///< [out] Pointer to the output sequence of selected data items
        ValueOutputIteratorT    d_values_out,           ///< [out] Pointer to the output sequence of selected data items
        NumSelectedIteratorT    d_num_selected_out,     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_keys_out or \p d_values_out)
        EqualityOpT             equality_op,            ///< [in] Equality operator
        OffsetT                 num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t            stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        using MaxPolicyT = typename DispatchUniqueByKey::MaxPolicy;

        cudaError_t error;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchUniqueByKey dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_keys_in,
                d_values_in,
                d_keys_out,
                d_values_out,
                d_num_selected_out,
                equality_op,
                num_items,
                stream,
                debug_synchronous
            );

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};

CUB_NAMESPACE_END
