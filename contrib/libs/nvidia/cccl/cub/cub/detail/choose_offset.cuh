/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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

#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

/**
 * choose_offset checks NumItemsT, the type of the num_items parameter, and
 * selects the offset type based on it.
 */
template <typename NumItemsT>
struct choose_offset
{
  // NumItemsT must be an integral type (but not bool).
  static_assert(::cuda::std::is_integral_v<NumItemsT>
                  && !::cuda::std::is_same_v<::cuda::std::remove_cv_t<NumItemsT>, bool>,
                "NumItemsT must be an integral type, but not bool");

  // Unsigned integer type for global offsets.
  using type = ::cuda::std::_If<(sizeof(NumItemsT) <= 4), uint32_t, unsigned long long>;
};

/**
 * choose_offset_t is an alias template that checks NumItemsT, the type of the num_items parameter, and
 * selects the offset type based on it.
 */
template <typename NumItemsT>
using choose_offset_t = typename choose_offset<NumItemsT>::type;

/**
 * promote_small_offset checks NumItemsT, the type of the num_items parameter, and
 * promotes any integral type smaller than 32 bits to a signed 32-bit integer type.
 */
template <typename NumItemsT>
struct promote_small_offset
{
  // NumItemsT must be an integral type (but not bool).
  static_assert(::cuda::std::is_integral_v<NumItemsT>
                  && !::cuda::std::is_same_v<::cuda::std::remove_cv_t<NumItemsT>, bool>,
                "NumItemsT must be an integral type, but not bool");

  // Unsigned integer type for global offsets.
  using type = ::cuda::std::_If<(sizeof(NumItemsT) < 4), int32_t, NumItemsT>;
};

/**
 * promote_small_offset_t is an alias template that checks NumItemsT, the type of the num_items parameter, and
 * promotes any integral type smaller than 32 bits to a signed 32-bit integer type.
 */
template <typename NumItemsT>
using promote_small_offset_t = typename promote_small_offset<NumItemsT>::type;

/**
 * choose_signed_offset checks NumItemsT, the type of the num_items parameter, and
 * selects the offset type to be either int32 or int64, such that the selected offset type covers the range of NumItemsT
 * unless it was uint64, in which case int64 will be used.
 */
template <typename NumItemsT>
struct choose_signed_offset
{
  // NumItemsT must be an integral type (but not bool).
  static_assert(::cuda::std::is_integral_v<NumItemsT>
                  && !::cuda::std::is_same_v<::cuda::std::remove_cv_t<NumItemsT>, bool>,
                "NumItemsT must be an integral type, but not bool");

  // Signed integer type for global offsets.
  // uint32 -> int64, else
  // LEQ 4B -> int32, else
  // int64
  using type = ::cuda::std::_If<(::cuda::std::is_integral_v<NumItemsT> && ::cuda::std::is_unsigned_v<NumItemsT>),
                                ::cuda::std::int64_t,
                                ::cuda::std::_If<(sizeof(NumItemsT) <= 4), ::cuda::std::int32_t, ::cuda::std::int64_t>>;

  /**
   * Checks if the given num_items can be covered by the selected offset type. If not, returns cudaErrorInvalidValue,
   * otherwise returns cudaSuccess.
   */
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t is_exceeding_offset_type(NumItemsT num_items)
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_MSVC(4127) /* conditional expression is constant */
    if (sizeof(NumItemsT) >= 8 && num_items > static_cast<NumItemsT>(::cuda::std::numeric_limits<type>::max()))
    {
      return cudaErrorInvalidValue;
    }
    _CCCL_DIAG_POP
    return cudaSuccess;
  }
};

/**
 * choose_signed_offset_t is an alias template that checks NumItemsT, the type of the num_items parameter, and
 * selects the corresponding signed offset type based on it.
 */
template <typename NumItemsT>
using choose_signed_offset_t = typename choose_signed_offset<NumItemsT>::type;

/**
 * common_iterator_value sets member type to the common_type of
 * value_type for all argument types. used to get OffsetT in
 * DeviceSegmentedReduce.
 */
template <typename... Iter>
struct common_iterator_value
{
  using type = ::cuda::std::common_type_t<::cuda::std::__iter_value_type<Iter>...>;
};
template <typename... Iter>
using common_iterator_value_t = typename common_iterator_value<Iter...>::type;

} // namespace detail

CUB_NAMESPACE_END
