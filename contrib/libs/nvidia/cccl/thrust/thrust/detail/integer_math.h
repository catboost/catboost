/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_deduction.h>

#include <cuda/std/__bit/countl.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename Integer>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool is_power_of_2(Integer x)
{
  return 0 == (x & (x - 1));
}

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename std::enable_if<std::is_signed<T>::value, bool>::type is_negative(T x)
{
  return x < 0;
}

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename std::enable_if<std::is_unsigned<T>::value, bool>::type is_negative(T)
{
  return false;
}

template <typename Integer>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool is_odd(Integer x)
{
  return 1 & x;
}

template <typename Integer>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE Integer log2(Integer x)
{
  Integer num_bits           = 8 * sizeof(Integer);
  Integer num_bits_minus_one = num_bits - 1;

  return num_bits_minus_one - ::cuda::std::countl_zero(::cuda::std::__to_unsigned_like(x));
}

template <typename Integer>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE Integer log2_ri(Integer x)
{
  Integer result = log2(x);

  // This is where we round up to the nearest log.
  if (!is_power_of_2(x))
  {
    ++result;
  }

  return result;
}

// x/y rounding towards +infinity for integers
// Used to determine # of blocks/warps etc.
template <typename Integer0, typename Integer1>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  // FIXME: Should use common_type.
  auto
  divide_ri(Integer0 const x, Integer1 const y) THRUST_DECLTYPE_RETURNS((x + (y - 1)) / y)

  // x/y rounding towards zero for integers.
  // Used to determine # of blocks/warps etc.
  template <typename Integer0, typename Integer1>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto divide_rz(Integer0 const x, Integer1 const y) THRUST_DECLTYPE_RETURNS(x / y)

  // Round x towards infinity to the next multiple of y.
  template <typename Integer0, typename Integer1>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto round_i(Integer0 const x, Integer1 const y)
    THRUST_DECLTYPE_RETURNS(y* divide_ri(x, y))

  // Round x towards 0 to the next multiple of y.
  template <typename Integer0, typename Integer1>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto round_z(Integer0 const x, Integer1 const y)
    THRUST_DECLTYPE_RETURNS(y* divide_rz(x, y))

} // namespace detail

THRUST_NAMESPACE_END
