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
 * cub::DeviceSelect provides device-wide, parallel operations for selecting items from sequences of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <cub/agent/agent_select_if.cuh>
#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>
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
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
    typename            AgentSelectIfPolicyT,       ///< Parameterized AgentSelectIfPolicyT tuning policy type
    typename            InputIteratorT,             ///< Random-access input iterator type for reading input items
    typename            FlagsInputIteratorT,        ///< Random-access input iterator type for reading selection flags (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename            SelectedOutputIteratorT,    ///< Random-access output iterator type for writing selected items
    typename            NumSelectedIteratorT,       ///< Output iterator type for recording the number of items selected
    typename            ScanTileStateT,             ///< Tile status interface type
    typename            SelectOpT,                  ///< Selection operator type (NullType if selection flags or discontinuity flagging is to be used for selection)
    typename            EqualityOpT,                ///< Equality operator type (NullType if selection functor or selection flags is to be used for selection)
    typename            OffsetT,                    ///< Signed integer type for global offsets
    bool                KEEP_REJECTS>               ///< Whether or not we push rejected items to the back of the output
__launch_bounds__ (int(AgentSelectIfPolicyT::BLOCK_THREADS))
__global__ void DeviceSelectSweepKernel(
    InputIteratorT          d_in,                   ///< [in] Pointer to the input sequence of data items
    FlagsInputIteratorT     d_flags,                ///< [in] Pointer to the input sequence of selection flags (if applicable)
    SelectedOutputIteratorT d_selected_out,         ///< [out] Pointer to the output sequence of selected data items
    NumSelectedIteratorT    d_num_selected_out,     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
    ScanTileStateT          tile_status,            ///< [in] Tile status interface
    SelectOpT               select_op,              ///< [in] Selection operator
    EqualityOpT             equality_op,            ///< [in] Equality operator
    OffsetT                 num_items,              ///< [in] Total number of input items (i.e., length of \p d_in)
    int                     num_tiles)              ///< [in] Total number of tiles for the entire problem
{
    // Thread block type for selecting data from input tiles
    typedef AgentSelectIf<
        AgentSelectIfPolicyT,
        InputIteratorT,
        FlagsInputIteratorT,
        SelectedOutputIteratorT,
        SelectOpT,
        EqualityOpT,
        OffsetT,
        KEEP_REJECTS> AgentSelectIfT;

    // Shared memory for AgentSelectIf
    __shared__ typename AgentSelectIfT::TempStorage temp_storage;

    // Process tiles
    AgentSelectIfT(temp_storage, d_in, d_flags, d_selected_out, select_op, equality_op, num_items).ConsumeRange(
        num_tiles,
        tile_status,
        d_num_selected_out);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSelect
 */
template <
    typename    InputIteratorT,                 ///< Random-access input iterator type for reading input items
    typename    FlagsInputIteratorT,            ///< Random-access input iterator type for reading selection flags (NullType* if a selection functor or discontinuity flagging is to be used for selection)
    typename    SelectedOutputIteratorT,        ///< Random-access output iterator type for writing selected items
    typename    NumSelectedIteratorT,           ///< Output iterator type for recording the number of items selected
    typename    SelectOpT,                      ///< Selection operator type (NullType if selection flags or discontinuity flagging is to be used for selection)
    typename    EqualityOpT,                    ///< Equality operator type (NullType if selection functor or selection flags is to be used for selection)
    typename    OffsetT,                        ///< Signed integer type for global offsets
    bool        KEEP_REJECTS,                   ///< Whether or not we push rejected items to the back of the output
    bool        MayAlias = false>                   
struct DispatchSelectIf
{
    /******************************************************************************
     * Types and constants
     ******************************************************************************/

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    // The flag value type
    using FlagT = cub::detail::value_t<FlagsInputIteratorT>;

    enum
    {
        INIT_KERNEL_THREADS = 128,
    };

    // Tile status descriptor interface type
    typedef ScanTileState<OffsetT> ScanTileStateT;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 10,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(InputT)))),
        };

        typedef AgentSelectIfPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_DIRECT,
                MayAlias ? LOAD_CA : LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS>
            SelectIfPolicyT;
    };

    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

    typedef Policy350 PtxPolicy;

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSelectIfPolicyT : PtxPolicy::SelectIfPolicyT {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &select_if_config)
    {
        NV_IF_TARGET(NV_IS_DEVICE,
        (
            (void)ptx_version;
            // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
            select_if_config.template Init<PtxSelectIfPolicyT>();
        ), (
            // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version

            // (There's only one policy right now)
            (void)ptx_version;
            select_if_config.template Init<typename Policy350::SelectIfPolicyT>();
        ));
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
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide selection using the
     * specified kernel functions.
     */
    template <
        typename                    ScanInitKernelPtrT,             ///< Function type of cub::DeviceScanInitKernel
        typename                    SelectIfKernelPtrT>             ///< Function type of cub::SelectIfKernelPtrT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                       d_temp_storage,                 ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
        FlagsInputIteratorT         d_flags,                        ///< [in] Pointer to the input sequence of selection flags (if applicable)
        SelectedOutputIteratorT     d_selected_out,                 ///< [in] Pointer to the output sequence of selected data items
        NumSelectedIteratorT        d_num_selected_out,             ///< [in] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
        SelectOpT                   select_op,                      ///< [in] Selection operator
        EqualityOpT                 equality_op,                    ///< [in] Equality operator
        OffsetT                     num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        int                         /*ptx_version*/,                ///< [in] PTX version of dispatch kernels
        ScanInitKernelPtrT          scan_init_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        SelectIfKernelPtrT          select_if_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceSelectSweepKernel
        KernelConfig                select_if_config)               ///< [in] Dispatch parameters that match the policy that \p select_if_kernel was compiled for
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = select_if_config.block_threads * select_if_config.items_per_thread;
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

            // Log scan_init_kernel configuration
            int init_grid_size = CUB_MAX(1, cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));

            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking scan_init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
            #endif

            // Invoke scan_init_kernel to initialize tile descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(scan_init_kernel,
                tile_status,
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

            // Return if empty problem
            if (num_items == 0)
                break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Get grid size for scanning tiles
            dim3 scan_grid_size;
            scan_grid_size.z = 1;
            scan_grid_size.y = cub::DivideAndRoundUp(num_tiles, max_dim_x);
            scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

            // Log select_if_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            {
              // Get SM occupancy for select_if_kernel
              int range_select_sm_occupancy;
              if (CubDebug(error = MaxSmOccupancy(range_select_sm_occupancy, // out
                                                  select_if_kernel,
                                                  select_if_config.block_threads)))
              {
                break;
              }

              _CubLog("Invoking select_if_kernel<<<{%d,%d,%d}, %d, 0, "
                      "%lld>>>(), %d items per thread, %d SM occupancy\n",
                      scan_grid_size.x,
                      scan_grid_size.y,
                      scan_grid_size.z,
                      select_if_config.block_threads,
                      (long long)stream,
                      select_if_config.items_per_thread,
                      range_select_sm_occupancy);
            }
            #endif

            // Invoke select_if_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                scan_grid_size, select_if_config.block_threads, 0, stream
            ).doit(select_if_kernel,
                d_in,
                d_flags,
                d_selected_out,
                d_num_selected_out,
                tile_status,
                select_op,
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

    template <typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    Dispatch(void *d_temp_storage,
             size_t &temp_storage_bytes,
             InputIteratorT d_in,
             FlagsInputIteratorT d_flags,
             SelectedOutputIteratorT d_selected_out,
             NumSelectedIteratorT d_num_selected_out,
             SelectOpT select_op,
             EqualityOpT equality_op,
             OffsetT num_items,
             cudaStream_t stream,
             bool debug_synchronous,
             int ptx_version,
             ScanInitKernelPtrT scan_init_kernel,
             SelectIfKernelPtrT select_if_kernel,
             KernelConfig select_if_config)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch<ScanInitKernelPtrT, SelectIfKernelPtrT>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_flags,
        d_selected_out,
        d_num_selected_out,
        select_op,
        equality_op,
        num_items,
        stream,
        ptx_version,
        scan_init_kernel,
        select_if_kernel,
        select_if_config);
    }

    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                       d_temp_storage,                 ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
        FlagsInputIteratorT         d_flags,                        ///< [in] Pointer to the input sequence of selection flags (if applicable)
        SelectedOutputIteratorT     d_selected_out,                 ///< [in] Pointer to the output sequence of selected data items
        NumSelectedIteratorT        d_num_selected_out,             ///< [in] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
        SelectOpT                   select_op,                      ///< [in] Selection operator
        EqualityOpT                 equality_op,                    ///< [in] Equality operator
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
            KernelConfig select_if_config;
            InitConfigs(ptx_version, select_if_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_flags,
                d_selected_out,
                d_num_selected_out,
                select_op,
                equality_op,
                num_items,
                stream,
                ptx_version,
                DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>,
                DeviceSelectSweepKernel<PtxSelectIfPolicyT, InputIteratorT, FlagsInputIteratorT, SelectedOutputIteratorT, NumSelectedIteratorT, ScanTileStateT, SelectOpT, EqualityOpT, OffsetT, KEEP_REJECTS>,
                select_if_config))) break;
        }
        while (0);

        return error;
    }

    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                       d_temp_storage,          
        size_t&                     temp_storage_bytes,       
        InputIteratorT              d_in,                      
        FlagsInputIteratorT         d_flags,                    
        SelectedOutputIteratorT     d_selected_out,              
        NumSelectedIteratorT        d_num_selected_out,           
        SelectOpT                   select_op,                     
        EqualityOpT                 equality_op,                    
        OffsetT                     num_items,            
        cudaStream_t                stream,                
        bool                        debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_flags,
                      d_selected_out,
                      d_num_selected_out,
                      select_op,
                      equality_op,
                      num_items,
                      stream);
    }
};


CUB_NAMESPACE_END
