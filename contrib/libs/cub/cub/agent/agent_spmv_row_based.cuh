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
 * cub::AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */

#pragma once

#include <iterator>

#include "../util_type.cuh"
#include "../block/block_reduce.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../thread/thread_search.cuh"
#include "../thread/thread_operators.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/counting_input_iterator.cuh"
#include "../iterator/tex_ref_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSpmv
 */
template <
    int                             _BLOCK_THREADS,                         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,                      ///< Items per thread (per tile of input)
    CacheLoadModifier               _ROW_OFFSETS_SEARCH_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
    CacheLoadModifier               _ROW_OFFSETS_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR row-offsets
    CacheLoadModifier               _COLUMN_INDICES_LOAD_MODIFIER,          ///< Cache load modifier for reading CSR column-indices
    CacheLoadModifier               _VALUES_LOAD_MODIFIER,                  ///< Cache load modifier for reading CSR values
    CacheLoadModifier               _VECTOR_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading vector values
    bool                            _DIRECT_LOAD_NONZEROS,                  ///< Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through shared memory)
    BlockScanAlgorithm              _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentSpmvPolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        DIRECT_LOAD_NONZEROS                                            = _DIRECT_LOAD_NONZEROS,                ///< Whether to load nonzeros directly from global during sequential merging (pre-staged through shared memory)
    };

    static const CacheLoadModifier  ROW_OFFSETS_SEARCH_LOAD_MODIFIER    = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  ROW_OFFSETS_LOAD_MODIFIER           = _ROW_OFFSETS_LOAD_MODIFIER;           ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  COLUMN_INDICES_LOAD_MODIFIER        = _COLUMN_INDICES_LOAD_MODIFIER;        ///< Cache load modifier for reading CSR column-indices
    static const CacheLoadModifier  VALUES_LOAD_MODIFIER                = _VALUES_LOAD_MODIFIER;                ///< Cache load modifier for reading CSR values
    static const CacheLoadModifier  VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading vector values
    static const BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct SpmvParams
{
    ValueT*         d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    OffsetT*        d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    OffsetT*        d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    ValueT*         d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand

    TexRefInputIterator<ValueT, 66778899, OffsetT>  t_vector_x;
};


/**
 * \brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */
template <
    typename    AgentSpmvPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    bool        HAS_ALPHA,                  ///< Whether the input parameter \p alpha is 1
    bool        HAS_BETA,                   ///< Whether the input parameter \p beta is 0
    int         PTX_ARCH = CUB_PTX_ARCH>    ///< PTX compute capability
struct AgentSpmv
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentSpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentSpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    /// 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    /// Input iterator wrapper types (for applying cache modifiers)

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsSearchIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        ColumnIndicesIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        ValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<cub::Sum> ReduceBySegmentOpT;

    // Prefix functor type
    typedef BlockScanRunningPrefixOp<KeyValuePairT, ReduceBySegmentOpT> PrefixOpT;

    // BlockScan specialization
    typedef BlockScan<
            KeyValuePairT,
            BLOCK_THREADS,
            AgentSpmvPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    /// Shared memory type required by this thread block
    struct _TempStorage
    {
        OffsetT tile_nonzero_idx;
        OffsetT tile_nonzero_idx_end;

        // Smem needed for tile scanning
        typename BlockScanT::TempStorage scan;

        // Smem needed for tile of merge items
        ValueT nonzeros[TILE_ITEMS + 1];

    };

    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                   temp_storage;         /// Reference to temp_storage

    SpmvParams<ValueT, OffsetT>&    spmv_params;

    ValueIteratorT                  wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT             wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    ColumnIndicesIteratorT          wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorT            wd_vector_x;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorValueIteratorT            wd_vector_y;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentSpmv(
        TempStorage&                    temp_storage,           ///< Reference to temp_storage
        SpmvParams<ValueT, OffsetT>&    spmv_params)            ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        spmv_params(spmv_params),
        wd_values(spmv_params.d_values),
        wd_row_end_offsets(spmv_params.d_row_end_offsets),
        wd_column_indices(spmv_params.d_column_indices),
        wd_vector_x(spmv_params.d_vector_x),
        wd_vector_y(spmv_params.d_vector_y)
    {}


    __device__ __forceinline__ void InitNan(double& nan_token)
    {
        long long NAN_BITS  = 0xFFF0000000000001;
        nan_token           = reinterpret_cast<ValueT&>(NAN_BITS); // ValueT(0.0) / ValueT(0.0);
    } 


    __device__ __forceinline__ void InitNan(float& nan_token)
    {
        int NAN_BITS        = 0xFF800001;
        nan_token           = reinterpret_cast<ValueT&>(NAN_BITS); // ValueT(0.0) / ValueT(0.0);
    } 


    /**
     *
     */
    template <int NNZ_PER_THREAD>
    __device__ __forceinline__ void ConsumeStrip(
        PrefixOpT&          prefix_op,
        ReduceBySegmentOpT& scan_op,
        ValueT&             row_total,
        ValueT&             row_start,
        OffsetT&            tile_nonzero_idx,
        OffsetT             tile_nonzero_idx_end,
        OffsetT             row_nonzero_idx,
        OffsetT             row_nonzero_idx_end)
    {
        ValueT NAN_TOKEN;
        InitNan(NAN_TOKEN);


        //
        // Gather a strip of nonzeros into shared memory
        //

        #pragma unroll
        for (int ITEM = 0; ITEM < NNZ_PER_THREAD; ++ITEM)
        {

            ValueT nonzero = 0.0;

            OffsetT                 local_nonzero_idx   = (ITEM * BLOCK_THREADS) + threadIdx.x;
            OffsetT                 nonzero_idx         = tile_nonzero_idx + local_nonzero_idx;

            bool in_range = nonzero_idx < tile_nonzero_idx_end;

            OffsetT nonzero_idx2 = (in_range) ?
                nonzero_idx :
                tile_nonzero_idx_end - 1;

            OffsetT column_idx          = wd_column_indices[nonzero_idx2];
            ValueT  value               = wd_values[nonzero_idx2];
            ValueT  vector_value        = wd_vector_x[column_idx];
            nonzero                     = value * vector_value;

            if (!in_range)
                nonzero = 0.0;

            temp_storage.nonzeros[local_nonzero_idx] = nonzero;
        }

        __syncthreads();

        //
        // Swap in NANs at local row start offsets
        //

        OffsetT local_row_nonzero_idx = row_nonzero_idx - tile_nonzero_idx;
        if ((local_row_nonzero_idx >= 0) && (local_row_nonzero_idx < TILE_ITEMS))
        {
            // Thread's row starts in this strip
            row_start = temp_storage.nonzeros[local_row_nonzero_idx];
            temp_storage.nonzeros[local_row_nonzero_idx] = NAN_TOKEN;
        }

        __syncthreads();

        //
        // Segmented scan
        //

        // Read strip of nonzeros into thread-blocked order, setup segment flags
        KeyValuePairT scan_items[NNZ_PER_THREAD];
        for (int ITEM = 0; ITEM < NNZ_PER_THREAD; ++ITEM)
        {
            int     local_nonzero_idx   = (threadIdx.x * NNZ_PER_THREAD) + ITEM;
            ValueT  value               = temp_storage.nonzeros[local_nonzero_idx];
            bool    is_nan              = (value != value);

            scan_items[ITEM].value  = (is_nan) ? 0.0 : value;
            scan_items[ITEM].key    = is_nan;
        }

        KeyValuePairT       tile_aggregate;
        KeyValuePairT       scan_items_out[NNZ_PER_THREAD];

        BlockScanT(temp_storage.scan).ExclusiveScan(scan_items, scan_items_out, scan_op, tile_aggregate, prefix_op);

        // Save the inclusive sum for the last row
        if (threadIdx.x == 0)
        {
            temp_storage.nonzeros[TILE_ITEMS] = prefix_op.running_total.value;
        }

        // Store segment totals
        for (int ITEM = 0; ITEM < NNZ_PER_THREAD; ++ITEM)
        {
            int local_nonzero_idx = (threadIdx.x * NNZ_PER_THREAD) + ITEM;

            if (scan_items[ITEM].key)
                temp_storage.nonzeros[local_nonzero_idx] = scan_items_out[ITEM].value;
        }

        __syncthreads();

        //
        // Update row totals
        //

        OffsetT local_row_nonzero_idx_end = row_nonzero_idx_end - tile_nonzero_idx;
        if ((local_row_nonzero_idx_end >= 0) && (local_row_nonzero_idx_end < TILE_ITEMS))
        {
            // Thread's row ends in this strip
            row_total = temp_storage.nonzeros[local_row_nonzero_idx_end];
        }

        tile_nonzero_idx += NNZ_PER_THREAD * BLOCK_THREADS;
    }



    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        int     tile_idx,
        int     rows_per_tile)
    {
        //
        // Read in tile of row ranges
        //

        // Row range for the thread block
        OffsetT tile_row_idx        = tile_idx * rows_per_tile;
        OffsetT tile_row_idx_end    = CUB_MIN(tile_row_idx + rows_per_tile, spmv_params.num_rows);

        // Thread's row
        OffsetT row_idx             = tile_row_idx + threadIdx.x;
        ValueT  row_total           = 0.0;
        ValueT  row_start           = 0.0;

        // Nonzero range for the thread's row
        OffsetT row_nonzero_idx     = -1;
        OffsetT row_nonzero_idx_end = -1;

        if (row_idx < tile_row_idx_end)
        {
            row_nonzero_idx     = wd_row_end_offsets[row_idx - 1];
            row_nonzero_idx_end = wd_row_end_offsets[row_idx];

            // Share block's starting nonzero offset
            if (threadIdx.x == 0)
                temp_storage.tile_nonzero_idx = row_nonzero_idx;

            // Share block's ending nonzero offset
            if (row_idx == tile_row_idx_end - 1)
                temp_storage.tile_nonzero_idx_end = row_nonzero_idx_end;

            // Zero-length rows don't participate
            if (row_nonzero_idx == row_nonzero_idx_end)
            {
                row_nonzero_idx = -1;
                row_nonzero_idx_end = -1;
            }
        }

        __syncthreads();

        //
        // Process strips of nonzeros
        //

        // Nonzero range for the thread block
        OffsetT tile_nonzero_idx        = temp_storage.tile_nonzero_idx;
        OffsetT tile_nonzero_idx_end    = temp_storage.tile_nonzero_idx_end;

        KeyValuePairT       tile_prefix = {0, 0.0};
        ReduceBySegmentOpT  scan_op;
        PrefixOpT           prefix_op(tile_prefix, scan_op);

        #pragma unroll 1
        while (tile_nonzero_idx < tile_nonzero_idx_end)
        {
            ConsumeStrip<ITEMS_PER_THREAD>(prefix_op, scan_op, row_total, row_start,
                tile_nonzero_idx, tile_nonzero_idx_end, row_nonzero_idx, row_nonzero_idx_end);

            __syncthreads();
        }

        //
        // Output to y
        //

        if (row_idx < tile_row_idx_end)
        {
            if (row_nonzero_idx_end == tile_nonzero_idx_end)
            {
                // Last row grabs the inclusive sum
                row_total = temp_storage.nonzeros[TILE_ITEMS];
            }

            spmv_params.d_vector_y[row_idx] = row_start + row_total;
        }
    }


};




}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

