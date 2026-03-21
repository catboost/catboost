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

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/**
 * @brief Sorts data using odd-even sort method
 *
 * The sorting method is stable. Further details can be found in:
 * A. Nico Habermann. Parallel neighbor sort (or the glory of the induction
 * principle). Technical Report AD-759 248, Carnegie Mellon University, 1972.
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type. If `cub::NullType` is used as `ValueT`, only keys are sorted.
 *
 * @tparam CompareOp
 *   functor type having member `bool operator()(KeyT lhs, KeyT rhs)`
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of items per thread
 *
 * @param[in,out] keys
 *   Keys to sort
 *
 * @param[in,out] items
 *   Values to sort
 *
 * @param[in] compare_op
 *   Comparison function object which returns true if the first argument is
 *   ordered before the second
 */
template <typename KeyT, typename ValueT, typename CompareOp, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StableOddEvenSort(KeyT (&keys)[ITEMS_PER_THREAD], ValueT (&items)[ITEMS_PER_THREAD], CompareOp compare_op)
{
  constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  _CCCL_SORT_MAYBE_UNROLL()
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    _CCCL_SORT_MAYBE_UNROLL()
    for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
    {
      if (compare_op(keys[j + 1], keys[j]))
      {
        using ::cuda::std::swap;
        swap(keys[j], keys[j + 1]);
        if (!KEYS_ONLY)
        {
          swap(items[j], items[j + 1]);
        }
      }
    } // inner loop
  } // outer loop
}

CUB_NAMESPACE_END
