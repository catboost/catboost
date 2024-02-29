/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_store.cuh>

#include <nv/target>

#include <thrust/system/cuda/detail/core/util.h>


CUB_NAMESPACE_BEGIN


template <
  int                      WARP_THREADS_ARG,
  int                      ITEMS_PER_THREAD_ARG,
  cub::WarpLoadAlgorithm   LOAD_ALGORITHM_ARG   = cub::WARP_LOAD_DIRECT,
  cub::CacheLoadModifier   LOAD_MODIFIER_ARG    = cub::LOAD_LDG,
  cub::WarpStoreAlgorithm  STORE_ALGORITHM_ARG  = cub::WARP_STORE_DIRECT>
struct AgentSubWarpMergeSortPolicy
{
  static constexpr int WARP_THREADS       = WARP_THREADS_ARG;
  static constexpr int ITEMS_PER_THREAD   = ITEMS_PER_THREAD_ARG;
  static constexpr int ITEMS_PER_TILE     = WARP_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::WarpLoadAlgorithm  LOAD_ALGORITHM   = LOAD_ALGORITHM_ARG;
  static constexpr cub::CacheLoadModifier  LOAD_MODIFIER    = LOAD_MODIFIER_ARG;
  static constexpr cub::WarpStoreAlgorithm STORE_ALGORITHM  = STORE_ALGORITHM_ARG;
};

template <int BLOCK_THREADS_ARG,
          typename SmallPolicy,
          typename MediumPolicy>
struct AgentSmallAndMediumSegmentedSortPolicy
{
  static constexpr int BLOCK_THREADS = BLOCK_THREADS_ARG;
  using SmallPolicyT                 = SmallPolicy;
  using MediumPolicyT                = MediumPolicy;

  constexpr static int SEGMENTS_PER_MEDIUM_BLOCK = BLOCK_THREADS /
                                                   MediumPolicyT::WARP_THREADS;

  constexpr static int SEGMENTS_PER_SMALL_BLOCK = BLOCK_THREADS /
                                                  SmallPolicyT::WARP_THREADS;
};


/**
 * @brief AgentSubWarpSort implements a sub-warp merge sort.
 *
 * This agent can work with any power of two number of threads, not exceeding
 * 32. The number of threads is defined in the `PolicyT::WARP_THREADS`. Virtual
 * warp of `PolicyT::WARP_THREADS` will efficiently load data using
 * `PolicyT::LOAD_ALGORITHM`, sort it using `WarpMergeSort`, and store it back
 * using `PolicyT::STORE_ALGORITHM`.
 *
 * @tparam IS_DESCENDING
 *   Whether or not the sorted-order is high-to-low
 *
 * @tparam PolicyT
 *   Chained tuning policy
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <bool IS_DESCENDING,
          typename PolicyT,
          typename KeyT,
          typename ValueT,
          typename OffsetT>
class AgentSubWarpSort
{
  struct BinaryOpT
  {
    template <typename T>
    __device__ bool operator()(T lhs, T rhs)
    {
      return this->impl(lhs, rhs);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)
    __device__ bool operator()(__half lhs, __half rhs)
    {
      // Need to explicitly cast to float for SM <= 52.
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (return this->impl(lhs, rhs);),
                   (return this->impl(__half2float(lhs), __half2float(rhs));));
    }
#endif

  private:
    template <typename T>
    __device__ bool impl(T lhs, T rhs)
    {
      if (IS_DESCENDING)
      {
        return lhs > rhs;
      }
      else
      {
        return lhs < rhs;
      }
    }
  };

#if defined(__CUDA_FP16_TYPES_EXIST__)
  __device__ static bool equal(__half lhs, __half rhs)
  {
    // Need to explicitly cast to float for SM <= 52.
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (return lhs == rhs;),
                 (return __half2float(lhs) == __half2float(rhs);));
  }
#endif

  template <typename T>
  __device__ static bool equal(T lhs, T rhs)
  {
    return lhs == rhs;
  }

  __device__ static bool get_oob_default(Int2Type<true> /* is bool */) 
  {
    // Traits<KeyT>::MAX_KEY for `bool` is 0xFF which is different from `true` and makes
    // comparison with oob unreliable.
    return !IS_DESCENDING;
  }

  __device__ static KeyT get_oob_default(Int2Type<false> /* is bool */) 
  {
    // For FP64 the difference is:
    // Lowest() -> -1.79769e+308 = 00...00b -> TwiddleIn -> -0 = 10...00b
    // LOWEST   -> -nan          = 11...11b -> TwiddleIn ->  0 = 00...00b

    using UnsignedBitsT = typename Traits<KeyT>::UnsignedBits;
    UnsignedBitsT default_key_bits = IS_DESCENDING ? Traits<KeyT>::LOWEST_KEY
                                                   : Traits<KeyT>::MAX_KEY;
    return reinterpret_cast<KeyT &>(default_key_bits);
  }

public:
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

  using WarpMergeSortT =
    WarpMergeSort<KeyT, PolicyT::ITEMS_PER_THREAD, PolicyT::WARP_THREADS, ValueT>;

  using KeysLoadItT = typename THRUST_NS_QUALIFIER::cuda_cub::core::
    LoadIterator<PolicyT, const KeyT *>::type;
  using ItemsLoadItT = typename THRUST_NS_QUALIFIER::cuda_cub::core::
    LoadIterator<PolicyT, const ValueT *>::type;

  using WarpLoadKeysT  = cub::WarpLoad<KeyT,
                                      PolicyT::ITEMS_PER_THREAD,
                                      PolicyT::LOAD_ALGORITHM,
                                      PolicyT::WARP_THREADS>;
  using WarpLoadItemsT = cub::WarpLoad<ValueT,
                                       PolicyT::ITEMS_PER_THREAD,
                                       PolicyT::LOAD_ALGORITHM,
                                       PolicyT::WARP_THREADS>;

  using WarpStoreKeysT  = cub::WarpStore<KeyT,
                                         PolicyT::ITEMS_PER_THREAD,
                                         PolicyT::STORE_ALGORITHM,
                                         PolicyT::WARP_THREADS>;
  using WarpStoreItemsT = cub::WarpStore<ValueT,
                                         PolicyT::ITEMS_PER_THREAD,
                                         PolicyT::STORE_ALGORITHM,
                                         PolicyT::WARP_THREADS>;

  union _TempStorage
  {
    typename WarpLoadKeysT::TempStorage load_keys;
    typename WarpLoadItemsT::TempStorage load_items;
    typename WarpMergeSortT::TempStorage sort;
    typename WarpStoreKeysT::TempStorage store_keys;
    typename WarpStoreItemsT::TempStorage store_items;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage> {};

  _TempStorage &storage;

  __device__ __forceinline__
  explicit AgentSubWarpSort(TempStorage &temp_storage)
    : storage(temp_storage.Alias())
  {
  }


  __device__ __forceinline__
  void ProcessSegment(int segment_size,
                      KeysLoadItT keys_input,
                      KeyT *keys_output,
                      ItemsLoadItT values_input,
                      ValueT *values_output)
  {
    WarpMergeSortT warp_merge_sort(storage.sort);

    if (segment_size < 3)
    {
      ShortCircuit(warp_merge_sort.get_linear_tid(),
                   segment_size,
                   keys_input,
                   keys_output,
                   values_input,
                   values_output,
                   BinaryOpT{});
    }
    else
    {
      KeyT keys[PolicyT::ITEMS_PER_THREAD];
      ValueT values[PolicyT::ITEMS_PER_THREAD];

      KeyT oob_default = 
        AgentSubWarpSort::get_oob_default(Int2Type<std::is_same<bool, KeyT>::value>{});

      WarpLoadKeysT(storage.load_keys)
        .Load(keys_input, keys, segment_size, oob_default);
      WARP_SYNC(warp_merge_sort.get_member_mask());

      if (!KEYS_ONLY)
      {
        WarpLoadItemsT(storage.load_items)
          .Load(values_input, values, segment_size);

        WARP_SYNC(warp_merge_sort.get_member_mask());
      }

      warp_merge_sort.Sort(keys, values, BinaryOpT{}, segment_size, oob_default);
      WARP_SYNC(warp_merge_sort.get_member_mask());

      WarpStoreKeysT(storage.store_keys).Store(keys_output, keys, segment_size);

      if (!KEYS_ONLY)
      {
        WARP_SYNC(warp_merge_sort.get_member_mask());
        WarpStoreItemsT(storage.store_items)
          .Store(values_output, values, segment_size);
      }
    }
  }

private:
  /**
   * This method implements a shortcut for sorting less than three items.
   * Only the first thread of a virtual warp is used for soring.
   */
  template <typename CompareOpT>
  __device__ __forceinline__ void ShortCircuit(unsigned int linear_tid,
                                               OffsetT segment_size,
                                               KeysLoadItT keys_input,
                                               KeyT *keys_output,
                                               ItemsLoadItT values_input,
                                               ValueT *values_output,
                                               CompareOpT binary_op)
  {
    if (segment_size == 1)
    {
      if (linear_tid == 0)
      {
        if (keys_input.ptr != keys_output)
        {
          keys_output[0] = keys_input[0];
        }

        if (!KEYS_ONLY)
        {
          if (values_input.ptr != values_output)
          {
            values_output[0] = values_input[0];
          }
        }
      }
    }
    else if (segment_size == 2)
    {
      if (linear_tid == 0)
      {
        KeyT lhs = keys_input[0];
        KeyT rhs = keys_input[1];

        if (equal(lhs, rhs) || binary_op(lhs, rhs))
        {
          keys_output[0] = lhs;
          keys_output[1] = rhs;

          if (!KEYS_ONLY)
          {
            if (values_output != values_input.ptr)
            {
              values_output[0] = values_input[0];
              values_output[1] = values_input[1];
            }
          }
        }
        else
        {
          keys_output[0] = rhs;
          keys_output[1] = lhs;

          if (!KEYS_ONLY)
          {
            // values_output might be an alias for values_input, so
            // we have to use registers here

            const ValueT lhs_val = values_input[0];
            const ValueT rhs_val = values_input[1];

            values_output[0] = rhs_val;
            values_output[1] = lhs_val;
          }
        }
      }
    }
  }
};

CUB_NAMESPACE_END
