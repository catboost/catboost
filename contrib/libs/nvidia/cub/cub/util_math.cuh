/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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
 * Define helper math functions.
 */

#pragma once

#include <type_traits>

#include "util_namespace.cuh"
#include "util_macro.cuh"

CUB_NAMESPACE_BEGIN

namespace detail
{

template <typename T>
using is_integral_or_enum =
  std::integral_constant<bool,
                         std::is_integral<T>::value || std::is_enum<T>::value>;

__host__ __device__ __forceinline__ constexpr  std::size_t
VshmemSize(std::size_t max_shmem,
           std::size_t shmem_per_block,
           std::size_t num_blocks)
{
  return shmem_per_block > max_shmem ? shmem_per_block * num_blocks : 0;
}

}

/**
 * Divide n by d, round up if any remainder, and return the result.
 *
 * Effectively performs `(n + d - 1) / d`, but is robust against the case where
 * `(n + d - 1)` would overflow.
 */
template <typename NumeratorT, typename DenominatorT>
__host__ __device__ __forceinline__ constexpr NumeratorT
DivideAndRoundUp(NumeratorT n, DenominatorT d)
{
  static_assert(cub::detail::is_integral_or_enum<NumeratorT>::value &&
                cub::detail::is_integral_or_enum<DenominatorT>::value,
                "DivideAndRoundUp is only intended for integral types.");

  // Static cast to undo integral promotion.
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

constexpr __device__ __host__ int
Nominal4BItemsToItemsCombined(int nominal_4b_items_per_thread, int combined_bytes)
{
  return (cub::min)(nominal_4b_items_per_thread,
                    (cub::max)(1,
                               nominal_4b_items_per_thread * 8 /
                               combined_bytes));
}

template <typename T>
constexpr __device__ __host__ int
Nominal4BItemsToItems(int nominal_4b_items_per_thread)
{
  return (cub::min)(nominal_4b_items_per_thread,
                    (cub::max)(1,
                               nominal_4b_items_per_thread * 4 /
                                 static_cast<int>(sizeof(T))));
}

template <typename ItemT>
constexpr __device__ __host__ int
Nominal8BItemsToItems(int nominal_8b_items_per_thread)
{
  return sizeof(ItemT) <= 8u
           ? nominal_8b_items_per_thread
           : (cub::min)(nominal_8b_items_per_thread,
                        (cub::max)(1,
                                   ((nominal_8b_items_per_thread * 8) +
                                    static_cast<int>(sizeof(ItemT)) - 1) /
                                     static_cast<int>(sizeof(ItemT))));
}

/**
 * \brief Computes the midpoint of the integers
 *
 * Extra operation is performed in order to prevent overflow.
 *
 * \return Half the sum of \p begin and \p end
 */
template <typename T>
constexpr __device__ __host__ T MidPoint(T begin, T end)
{
  return begin + (end - begin) / 2;
}

CUB_NAMESPACE_END
