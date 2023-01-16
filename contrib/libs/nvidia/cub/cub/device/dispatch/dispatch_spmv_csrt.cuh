
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch_scan.cuh"
#include "../../agent/agent_spmv_orig.cuh"
#include "../../util_type.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * SpMV kernel entry points
 *****************************************************************************/

/**
 * Spmv agent entry point
 */
template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
__launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__global__ void DeviceSpmvKernel(
    SpmvParams<ValueT, OffsetT>     spmv_params,                ///< [in] SpMV input parameter bundle
    int                             merge_items_per_block,      ///< [in] Number of merge tiles per block
    KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs)         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
{
    // Spmv agent type specialization
    typedef AgentSpmv<
            SpmvPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentSpmvT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentSpmvT::TempStorage temp_storage;

    AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
        merge_items_per_block, d_tile_carry_pairs);
}


/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
 */
template <
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT>                    ///< Signed integer type for global offsets
struct DispatchSpmv
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // SpmvParams bundle type
    typedef SpmvParams<ValueT, OffsetT> SpmvParamsT;

    // Tuple type for scanning {row id, accumulated value}
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM11
    struct Policy110
    {
        typedef AgentSpmvPolicy<
                128,
                1,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;
    };

    /// SM20
    struct Policy200 
    {
        typedef AgentSpmvPolicy<
                96,
                18,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_RAKING>
            SpmvPolicyT;
    };



    /// SM30
    struct Policy300 
    {
        typedef AgentSpmvPolicy<
                96,
                6,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;
    };


    /// SM35
    struct Policy350
    {
/*
        typedef AgentSpmvPolicy<
                (sizeof(ValueT) > 4) ? 96 : 128,
                (sizeof(ValueT) > 4) ? 4 : 7,
                LOAD_LDG,
                LOAD_CA,
                LOAD_LDG,
                LOAD_LDG,
                LOAD_LDG,
                (sizeof(ValueT) > 4) ? true : false,
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;
*/
        typedef AgentSpmvPolicy<
                128,
                5,
                LOAD_CA,
                LOAD_CA,
                LOAD_LDG,
                LOAD_LDG,
                LOAD_LDG,
                (sizeof(ValueT) > 4) ? true : false,
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;
    };

    /// SM37
    struct Policy370
    {

        typedef AgentSpmvPolicy<
                (sizeof(ValueT) > 4) ? 128 : 128,
                (sizeof(ValueT) > 4) ? 9 : 14,
                LOAD_LDG,
                LOAD_CA,
                LOAD_LDG,
                LOAD_LDG,
                LOAD_LDG,
                false, 
                BLOCK_SCAN_WARP_SCANS>
            SpmvPolicyT;
    };

    /// SM50
    struct Policy500
    {
        typedef AgentSpmvPolicy<
                (sizeof(ValueT) > 4) ? 64 : 128,
                (sizeof(ValueT) > 4) ? 6 : 7,
                LOAD_LDG,
                LOAD_DEFAULT,
                (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
                (sizeof(ValueT) > 4) ? LOAD_LDG : LOAD_DEFAULT,
                LOAD_LDG,
                (sizeof(ValueT) > 4) ? true : false,
                (sizeof(ValueT) > 4) ? BLOCK_SCAN_WARP_SCANS : BLOCK_SCAN_RAKING_MEMOIZE>
            SpmvPolicyT;
    };



    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 500)
    typedef Policy500 PtxPolicy;

#elif (CUB_PTX_ARCH >= 370)
    typedef Policy370 PtxPolicy;

#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#else
    typedef Policy110 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSpmvPolicyT : PtxPolicy::SpmvPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &spmv_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        spmv_config.template Init<PtxSpmvPolicyT>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 500)
        {
            spmv_config.template            Init<typename Policy500::SpmvPolicyT>();
        }
        else if (ptx_version >= 370)
        {
            spmv_config.template            Init<typename Policy370::SpmvPolicyT>();
        }
        else if (ptx_version >= 350)
        {
            spmv_config.template            Init<typename Policy350::SpmvPolicyT>();
        }
        else if (ptx_version >= 300)
        {
            spmv_config.template            Init<typename Policy300::SpmvPolicyT>();
        }
        else if (ptx_version >= 200)
        {
            spmv_config.template            Init<typename Policy200::SpmvPolicyT>();
        }
        else
        {
            spmv_config.template            Init<typename Policy110::SpmvPolicyT>();
        }

    #endif
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


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                SpmvKernelT>                        ///< Function type of cub::AgentSpmvKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        SpmvKernelT             spmv_kernel,                        ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        KernelConfig            spmv_config)                        ///< [in] Dispatch parameters that match the policy that \p spmv_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Total number of spmv work items
            int num_merge_items = spmv_params.num_rows + spmv_params.num_nonzeros;

            // Get SM occupancy for kernels
            int spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                spmv_sm_occupancy,
                spmv_kernel,
                spmv_config.block_threads))) break;
            int spmv_device_occupancy = spmv_sm_occupancy * sm_count;

            // Grid dimensions
            int spmv_grid_size = CUB_MIN(((num_merge_items + spmv_config.block_threads - 1) / spmv_config.block_threads), spmv_device_occupancy);

            // Merge items per block
            int merge_items_per_block = (num_merge_items + spmv_grid_size - 1) / spmv_grid_size;

            // Get the temporary storage allocation requirements
            size_t allocation_sizes[1];
            allocation_sizes[0] = spmv_grid_size * sizeof(KeyValuePairT);       // bytes needed for block carry-out pairs

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void* allocations[1];
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }
            KeyValuePairT* d_tile_carry_pairs = (KeyValuePairT*) allocations[0];  // Agent carry-out pairs

            // Log spmv_kernel configuration
            if (debug_synchronous) _CubLog("Invoking spmv_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                spmv_grid_size, spmv_config.block_threads, (long long) stream, spmv_config.items_per_thread, spmv_sm_occupancy);

            // Invoke spmv_kernel
            spmv_kernel<<<spmv_grid_size, spmv_config.block_threads, 0, stream>>>(
                spmv_params,
                merge_items_per_block,
                d_tile_carry_pairs);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig spmv_config;
            InitConfigs(ptx_version, spmv_config);

            if (CubDebug(error = Dispatch(
                d_temp_storage, 
                temp_storage_bytes, 
                spmv_params, 
                stream, 
                debug_synchronous,
                DeviceSpmvKernel<PtxSpmvPolicyT, ValueT, OffsetT, false, false>,
                spmv_config))) break;

        }
        while (0);

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


