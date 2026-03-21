/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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
 * agent_radix_sort_onesweep.cuh implements a stateful abstraction of CUDA
 * thread blocks for participating in the device one-sweep radix sort kernel.
 */

#pragma once
#pragma clang system_header


#include "../block/block_radix_rank.cuh"
#include "../block/radix_rank_sort_operations.cuh"
#include "../block/block_store.cuh"
#include "../config.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"

CUB_NAMESPACE_BEGIN

/** \brief cub::RadixSortStoreAlgorithm enumerates different algorithms to write
 * partitioned elements (keys, values) stored in shared memory into global
 * memory. Currently applies only to writing 4B keys in full tiles; in all other cases,
 * RADIX_SORT_STORE_DIRECT is used.
 */
enum RadixSortStoreAlgorithm
{
    /** \brief Elements are statically distributed among block threads, which write them
     * into the appropriate partition in global memory. This results in fewer instructions
     * and more writes in flight at a given moment, but may generate more transactions. */
    RADIX_SORT_STORE_DIRECT,
    /** \brief Elements are distributed among warps in a block distribution. Each warp
     * goes through its elements and tries to write them while minimizing the number of
     * memory transactions. This results in fewer memory transactions, but more
     * instructions and less writes in flight at a given moment. */
    RADIX_SORT_STORE_ALIGNED
};

template <
    int NOMINAL_BLOCK_THREADS_4B,
    int NOMINAL_ITEMS_PER_THREAD_4B,
    typename ComputeT,
    /** \brief Number of private histograms to use in the ranker; 
        ignored if the ranking algorithm is not one of RADIX_RANK_MATCH_EARLY_COUNTS_* */
    int _RANK_NUM_PARTS,
    /** \brief Ranking algorithm used in the onesweep kernel. Only algorithms that
      support warp-strided key arrangement and count callbacks are supported. */
    RadixRankAlgorithm _RANK_ALGORITHM,
    BlockScanAlgorithm _SCAN_ALGORITHM,
    RadixSortStoreAlgorithm _STORE_ALGORITHM,
    int _RADIX_BITS,
    typename ScalingType = RegBoundScaling<
        NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT> >
struct AgentRadixSortOnesweepPolicy : ScalingType
{
    enum
    {
        RANK_NUM_PARTS = _RANK_NUM_PARTS,
        RADIX_BITS = _RADIX_BITS,
    };
    static const RadixRankAlgorithm RANK_ALGORITHM = _RANK_ALGORITHM;
    static const BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
    static const RadixSortStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

template <
    typename AgentRadixSortOnesweepPolicy,
    bool IS_DESCENDING,
    typename KeyT,
    typename ValueT,
    typename OffsetT,
    typename PortionOffsetT>
struct AgentRadixSortOnesweep
{
    // constants
    enum
    {
        ITEMS_PER_THREAD = AgentRadixSortOnesweepPolicy::ITEMS_PER_THREAD,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        BLOCK_THREADS = AgentRadixSortOnesweepPolicy::BLOCK_THREADS,
        RANK_NUM_PARTS = AgentRadixSortOnesweepPolicy::RANK_NUM_PARTS,
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        RADIX_BITS = AgentRadixSortOnesweepPolicy::RADIX_BITS,
        RADIX_DIGITS = 1 << RADIX_BITS,        
        BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        FULL_BINS = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS,
        WARP_THREADS = CUB_PTX_WARP_THREADS,
        BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS,
        WARP_MASK = ~0,
        LOOKBACK_PARTIAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 2),
        LOOKBACK_GLOBAL_MASK = 1 << (PortionOffsetT(sizeof(PortionOffsetT)) * 8 - 1),
        LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK,
        LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK,
    };

    typedef typename Traits<KeyT>::UnsignedBits UnsignedBits;
    typedef PortionOffsetT AtomicOffsetT;
  
    static const RadixRankAlgorithm RANK_ALGORITHM =
                                    AgentRadixSortOnesweepPolicy::RANK_ALGORITHM;
    static const BlockScanAlgorithm SCAN_ALGORITHM =
                                    AgentRadixSortOnesweepPolicy::SCAN_ALGORITHM;
    static const RadixSortStoreAlgorithm STORE_ALGORITHM =
                                    sizeof(UnsignedBits) == sizeof(uint32_t) ?
                                    AgentRadixSortOnesweepPolicy::STORE_ALGORITHM :
                                    RADIX_SORT_STORE_DIRECT;

    typedef RadixSortTwiddle<IS_DESCENDING, KeyT> Twiddle;

    static_assert(RANK_ALGORITHM == RADIX_RANK_MATCH
                  || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ANY
                  || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
        "for onesweep agent, the ranking algorithm must warp-strided key arrangement");

    using BlockRadixRankT = cub::detail::conditional_t<
      RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR,
      BlockRadixRankMatchEarlyCounts<BLOCK_THREADS,
                                     RADIX_BITS,
                                     false,
                                     SCAN_ALGORITHM,
                                     WARP_MATCH_ATOMIC_OR,
                                     RANK_NUM_PARTS>,
      cub::detail::conditional_t<
        RANK_ALGORITHM == RADIX_RANK_MATCH,
        BlockRadixRankMatch<BLOCK_THREADS, RADIX_BITS, false, SCAN_ALGORITHM>,
        BlockRadixRankMatchEarlyCounts<BLOCK_THREADS,
                                       RADIX_BITS,
                                       false,
                                       SCAN_ALGORITHM,
                                       WARP_MATCH_ANY,
                                       RANK_NUM_PARTS>>>;

    // temporary storage
    struct TempStorage_
    {
        union
        {
            UnsignedBits keys_out[TILE_ITEMS];
            ValueT values_out[TILE_ITEMS];
            typename BlockRadixRankT::TempStorage rank_temp_storage;
        };
        union
        {
            OffsetT global_offsets[RADIX_DIGITS];
            PortionOffsetT block_idx;
        };
    };

    using TempStorage = Uninitialized<TempStorage_>;

    // thread variables
    TempStorage_& s;

    // kernel parameters
    AtomicOffsetT* d_lookback;
    AtomicOffsetT* d_ctrs;
    OffsetT* d_bins_out;
    const OffsetT*  d_bins_in;
    UnsignedBits* d_keys_out;
    const UnsignedBits* d_keys_in;
    ValueT* d_values_out;
    const ValueT* d_values_in;
    PortionOffsetT num_items;
    ShiftDigitExtractor<KeyT> digit_extractor;

    // other thread variables
    int warp;
    int lane;
    PortionOffsetT block_idx;
    bool full_block;

    // helper methods
    __device__ __forceinline__ int Digit(UnsignedBits key)
    {
        return digit_extractor.Digit(key);
    }

    __device__ __forceinline__ int ThreadBin(int u)
    {
        return threadIdx.x * BINS_PER_THREAD + u;
    }

    __device__ __forceinline__ void LookbackPartial(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u) 
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                // write the local sum into the bin
                AtomicOffsetT& loc = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value = bins[u] | LOOKBACK_PARTIAL_MASK;
                ThreadStore<STORE_VOLATILE>(&loc, value);
            }
        }
    }

    struct CountsCallback
    {
        typedef AgentRadixSortOnesweep<AgentRadixSortOnesweepPolicy, IS_DESCENDING, KeyT,
                                       ValueT, OffsetT, PortionOffsetT> AgentT;
        AgentT& agent;
        int (&bins)[BINS_PER_THREAD];
        UnsignedBits (&keys)[ITEMS_PER_THREAD];
        static const bool EMPTY = false;
        __device__ __forceinline__ CountsCallback(
            AgentT& agent, int (&bins)[BINS_PER_THREAD], UnsignedBits (&keys)[ITEMS_PER_THREAD])
            : agent(agent), bins(bins), keys(keys) {}
        __device__ __forceinline__ void operator()(int (&other_bins)[BINS_PER_THREAD])
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                bins[u] = other_bins[u];
            }
            agent.LookbackPartial(bins);

            agent.TryShortCircuit(keys, bins);
        }
    };
  
    __device__ __forceinline__ void LookbackGlobal(int (&bins)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                PortionOffsetT inc_sum = bins[u];
                int want_mask = ~0;
                // backtrack as long as necessary
                for (PortionOffsetT block_jdx = block_idx - 1; block_jdx >= 0; --block_jdx)
                {
                    // wait for some value to appear
                    PortionOffsetT value_j = 0;
                    AtomicOffsetT& loc_j = d_lookback[block_jdx * RADIX_DIGITS + bin];
                    do {
                        __threadfence_block(); // prevent hoisting loads from loop
                        value_j = ThreadLoad<LOAD_VOLATILE>(&loc_j);
                    } while (value_j == 0);

                    inc_sum += value_j & LOOKBACK_VALUE_MASK;
                    want_mask = WARP_BALLOT((value_j & LOOKBACK_GLOBAL_MASK) == 0, want_mask);
                    if (value_j & LOOKBACK_GLOBAL_MASK) break;
                }
                AtomicOffsetT& loc_i = d_lookback[block_idx * RADIX_DIGITS + bin];
                PortionOffsetT value_i = inc_sum | LOOKBACK_GLOBAL_MASK;
                ThreadStore<STORE_VOLATILE>(&loc_i, value_i);
                s.global_offsets[bin] += inc_sum - bins[u];
            }
        }
    }

    __device__ __forceinline__
    void LoadKeys(OffsetT tile_offset, UnsignedBits (&keys)[ITEMS_PER_THREAD])
    {
        if (full_block)
        {
            LoadDirectWarpStriped(threadIdx.x, d_keys_in + tile_offset, keys);
        }
        else
        {
            LoadDirectWarpStriped(threadIdx.x, d_keys_in + tile_offset, keys,
                                  num_items - tile_offset, Twiddle::DefaultKey());
        }

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            keys[u] = Twiddle::In(keys[u]);
        }
    }

    __device__ __forceinline__
    void LoadValues(OffsetT tile_offset, ValueT (&values)[ITEMS_PER_THREAD])
    {
        if (full_block)
        {
            LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values);
        }
        else
        {
            int tile_items = num_items - tile_offset;
            LoadDirectWarpStriped(threadIdx.x, d_values_in + tile_offset, values,
                                  tile_items);
        }
    }

    /** Checks whether "short-circuiting" is possible. Short-circuiting happens
     * if all TILE_ITEMS keys fall into the same bin, i.e. have the same digit
     * value (note that it only happens for full tiles). If short-circuiting is
     * performed, the part of the ranking algorithm after the CountsCallback, as
     * well as the rest of the sorting (e.g. scattering keys and values to
     * shared and global memory) are skipped; updates related to decoupled
     * look-back are still performed. Instead, the keys assigned to the current
     * thread block are written cooperatively into a contiguous location in
     * d_keys_out corresponding to their digit. The values (if also sorting
     * values) assigned to the current thread block are similarly copied from
     * d_values_in to d_values_out. */
    __device__ __forceinline__
    void TryShortCircuit(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&bins)[BINS_PER_THREAD])
    {
        // check if any bin can be short-circuited
        bool short_circuit = false;
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            if (FULL_BINS || ThreadBin(u) < RADIX_DIGITS)
            {
                short_circuit = short_circuit || bins[u] == TILE_ITEMS;
            }
        }
        short_circuit = CTA_SYNC_OR(short_circuit);
        if (!short_circuit) return;

        ShortCircuitCopy(keys, bins);
    }

    __device__ __forceinline__
    void ShortCircuitCopy(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&bins)[BINS_PER_THREAD])
    {
        // short-circuit handling; note that global look-back is still required

        // compute offsets
        int common_bin = Digit(keys[0]);
        int offsets[BINS_PER_THREAD];
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            offsets[u] = bin > common_bin ? TILE_ITEMS : 0;
        }

        // global lookback
        LoadBinsToOffsetsGlobal(offsets);
        LookbackGlobal(bins);
        UpdateBinsGlobal(bins, offsets);
        CTA_SYNC();

        // scatter the keys
        OffsetT global_offset = s.global_offsets[common_bin];
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            keys[u] = Twiddle::Out(keys[u]);
        }
        if (full_block)
        {
            StoreDirectWarpStriped(threadIdx.x, d_keys_out + global_offset, keys);
        }
        else
        {
            int tile_items = num_items - block_idx * TILE_ITEMS;
            StoreDirectWarpStriped(threadIdx.x, d_keys_out + global_offset, keys,
                                   tile_items);
        }

        if (!KEYS_ONLY)
        {
            // gather and scatter the values
            ValueT values[ITEMS_PER_THREAD];
            LoadValues(block_idx * TILE_ITEMS, values);
            if (full_block)
            {
                StoreDirectWarpStriped(threadIdx.x, d_values_out + global_offset, values);
            }
            else
            {
                int tile_items = num_items - block_idx * TILE_ITEMS;
                StoreDirectWarpStriped(threadIdx.x, d_values_out + global_offset, values,
                                       tile_items);
            }
        }

        // exit early
        ThreadExit();
    }

    __device__ __forceinline__
    void ScatterKeysShared(UnsignedBits (&keys)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.keys_out[ranks[u]] = keys[u];
        }
    }

    __device__ __forceinline__
    void ScatterValuesShared(ValueT (&values)[ITEMS_PER_THREAD], int (&ranks)[ITEMS_PER_THREAD])
    {
        // write to shared memory
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            s.values_out[ranks[u]] = values[u];
        }
    }

    __device__ __forceinline__ void LoadBinsToOffsetsGlobal(int (&offsets)[BINS_PER_THREAD])
    {
        // global offset - global part
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || bin < RADIX_DIGITS)
            {
                s.global_offsets[bin] = d_bins_in[bin] - offsets[u];
            }
        }        
    }

    __device__ __forceinline__ void UpdateBinsGlobal(int (&bins)[BINS_PER_THREAD],
                                                     int (&offsets)[BINS_PER_THREAD])
    {
        bool last_block = (block_idx + 1) * TILE_ITEMS >= num_items;
        if (d_bins_out != NULL && last_block)
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || bin < RADIX_DIGITS)
                {
                    d_bins_out[bin] = s.global_offsets[bin] + offsets[u] + bins[u];
                }
            }
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterKeysGlobalDirect()
    {
        int tile_items = FULL_TILE ? TILE_ITEMS : num_items - block_idx * TILE_ITEMS;
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            UnsignedBits key = s.keys_out[idx];
            OffsetT global_idx = idx + s.global_offsets[Digit(key)];
            if (FULL_TILE || idx < tile_items)
            {
                d_keys_out[global_idx] = Twiddle::Out(key);
            }
            WARP_SYNC(WARP_MASK);
        }
    }

    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterValuesGlobalDirect(int (&digits)[ITEMS_PER_THREAD])
    {
        int tile_items = FULL_TILE ? TILE_ITEMS : num_items - block_idx * TILE_ITEMS;
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            ValueT value = s.values_out[idx];
            OffsetT global_idx = idx + s.global_offsets[digits[u]];
            if (FULL_TILE || idx < tile_items) d_values_out[global_idx] = value;
            WARP_SYNC(WARP_MASK);
        }
    }

    __device__ __forceinline__ void ScatterKeysGlobalAligned()
    {
        // this only works with full tiles
        const int ITEMS_PER_WARP = TILE_ITEMS / BLOCK_WARPS;
        const int ALIGN = 8;
        const auto CACHE_MODIFIER = STORE_CG;
        
        int warp_start = warp * ITEMS_PER_WARP;
        int warp_end = (warp + 1) * ITEMS_PER_WARP;
        int warp_offset = warp_start;
        while (warp_offset < warp_end - WARP_THREADS)
        {
            int idx = warp_offset + lane;
            UnsignedBits key = s.keys_out[idx];
            UnsignedBits key_out = Twiddle::Out(key);
            OffsetT global_idx = idx + s.global_offsets[Digit(key)];
            int last_lane = WARP_THREADS - 1;
            int num_writes = WARP_THREADS;
            if (lane == last_lane)
            {
                num_writes -= int(global_idx + 1) % ALIGN;
            }
            num_writes = SHFL_IDX_SYNC(num_writes, last_lane, WARP_MASK);
            if (lane < num_writes)
            {
                ThreadStore<CACHE_MODIFIER>(&d_keys_out[global_idx], key_out);
            }
            warp_offset += num_writes;
        }
        {
            int num_writes = warp_end - warp_offset;
            if (lane < num_writes)
            {
                int idx = warp_offset + lane;
                UnsignedBits key = s.keys_out[idx];
                OffsetT global_idx = idx + s.global_offsets[Digit(key)];
                ThreadStore<CACHE_MODIFIER>(&d_keys_out[global_idx], Twiddle::Out(key));
            }
        }
    }

    __device__ __forceinline__ void ScatterKeysGlobal()
    {
        // write block data to global memory
        if (full_block)
        {
            if (STORE_ALGORITHM == RADIX_SORT_STORE_ALIGNED)
            {
                ScatterKeysGlobalAligned();
            }
            else
            {
                ScatterKeysGlobalDirect<true>();
            }
        }
        else
        {
            ScatterKeysGlobalDirect<false>();
        }
    }

    __device__ __forceinline__ void ScatterValuesGlobal(int (&digits)[ITEMS_PER_THREAD])
    {
        // write block data to global memory
        if (full_block)
        {
            ScatterValuesGlobalDirect<true>(digits);
        }
        else
        {
            ScatterValuesGlobalDirect<false>(digits);
        }
    }

    __device__ __forceinline__ void ComputeKeyDigits(int (&digits)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            int idx = threadIdx.x + u * BLOCK_THREADS;
            digits[u] = Digit(s.keys_out[idx]);
        }
    }

    __device__ __forceinline__ void GatherScatterValues(
        int (&ranks)[ITEMS_PER_THREAD], Int2Type<false> keys_only)
    {
        // compute digits corresponding to the keys
        int digits[ITEMS_PER_THREAD];
        ComputeKeyDigits(digits);
        
        // load values
        ValueT values[ITEMS_PER_THREAD];
        LoadValues(block_idx * TILE_ITEMS, values);
        
        // scatter values
        CTA_SYNC();
        ScatterValuesShared(values, ranks);

        CTA_SYNC();
        ScatterValuesGlobal(digits);
    }
        

    __device__ __forceinline__ void GatherScatterValues(
        int (&ranks)[ITEMS_PER_THREAD], Int2Type<true> keys_only) {}

    __device__ __forceinline__ void Process()
    {
        // load keys
        // if warp1 < warp2, all elements of warp1 occur before those of warp2
        // in the source array
        UnsignedBits keys[ITEMS_PER_THREAD];
        LoadKeys(block_idx * TILE_ITEMS, keys);

        // rank keys
        int ranks[ITEMS_PER_THREAD];
        int exclusive_digit_prefix[BINS_PER_THREAD];
        int bins[BINS_PER_THREAD];
        BlockRadixRankT(s.rank_temp_storage).RankKeys(
            keys, ranks, digit_extractor, exclusive_digit_prefix,
            CountsCallback(*this, bins, keys));
        
        // scatter keys in shared memory
        CTA_SYNC();
        ScatterKeysShared(keys, ranks);

        // compute global offsets
        LoadBinsToOffsetsGlobal(exclusive_digit_prefix);
        LookbackGlobal(bins);
        UpdateBinsGlobal(bins, exclusive_digit_prefix);
                
        // scatter keys in global memory
        CTA_SYNC();
        ScatterKeysGlobal();

        // scatter values if necessary
        GatherScatterValues(ranks, Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ //
    AgentRadixSortOnesweep(TempStorage &temp_storage,
                           AtomicOffsetT *d_lookback,
                           AtomicOffsetT *d_ctrs,
                           OffsetT *d_bins_out,
                           const OffsetT *d_bins_in,
                           KeyT *d_keys_out,
                           const KeyT *d_keys_in,
                           ValueT *d_values_out,
                           const ValueT *d_values_in,
                           PortionOffsetT num_items,
                           int current_bit,
                           int num_bits)
        : s(temp_storage.Alias())
        , d_lookback(d_lookback)
        , d_ctrs(d_ctrs)
        , d_bins_out(d_bins_out)
        , d_bins_in(d_bins_in)
        , d_keys_out(reinterpret_cast<UnsignedBits *>(d_keys_out))
        , d_keys_in(reinterpret_cast<const UnsignedBits *>(d_keys_in))
        , d_values_out(d_values_out)
        , d_values_in(d_values_in)
        , num_items(num_items)
        , digit_extractor(current_bit, num_bits)
        , warp(threadIdx.x / WARP_THREADS)
        , lane(LaneId())
    {
        // initialization
        if (threadIdx.x == 0)
        {
            s.block_idx = atomicAdd(d_ctrs, 1);
        }
        CTA_SYNC();
        block_idx = s.block_idx;
        full_block = (block_idx + 1) * TILE_ITEMS <= num_items;
    }
};

CUB_NAMESPACE_END
