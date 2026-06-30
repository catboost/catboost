/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * Simple binary operator functor types
 */

/******************************************************************************
 * Simple functor operators
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

#include <cuda/functional> // cuda::maximum, cuda::minimum
#include <cuda/std/cstdint> // uint32_t
#include <cuda/std/functional> // cuda::std::plus
#include <cuda/std/type_traits> // cuda::std::common_type

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/***********************************************************************************************************************
 * SIMD operators
 **********************************************************************************************************************/

namespace detail
{

_CCCL_HOST_DEVICE uint32_t simd_operation_is_not_supported_before_sm90();

template <typename T>
struct SimdMin
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct SimdMin<int16_t>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t operator()(uint32_t a, uint32_t b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (return __vmins2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm90();));
  }
};

template <>
struct SimdMin<uint16_t>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t operator()(uint32_t a, uint32_t b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (return __vminu2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm90();));
  }
};

#  if _CCCL_HAS_NVFP16()

_CCCL_HOST_DEVICE __half2 simd_operation_is_not_supported_before_sm80(__half2);

template <>
struct SimdMin<__half>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmin2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm80(__half2{});));
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

_CCCL_HOST_DEVICE __nv_bfloat162 simd_operation_is_not_supported_before_sm80(__nv_bfloat162);

template <>
struct SimdMin<__nv_bfloat16>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmin2(a, b);),
                 (return simd_operation_is_not_supported_before_sm80(__nv_bfloat162{});));
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMax
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct SimdMax<int16_t>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t operator()(uint32_t a, uint32_t b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (return __vmaxs2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm90();));
  }
};

template <>
struct SimdMax<uint16_t>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t operator()(uint32_t a, uint32_t b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (return __vmaxu2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm90();));
  }
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdMax<__half>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmax2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm80(__half2{});));
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdMax<__nv_bfloat16>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmax2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm80(__nv_bfloat162{});));
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdSum
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

#  if _CCCL_HAS_NVFP16()

_CCCL_HOST_DEVICE __half2 simd_operation_is_not_supported_before_sm53(__half2);

template <>
struct SimdSum<__half>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (return __hadd2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm53(__half2{});));
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

_CCCL_HOST_DEVICE __nv_bfloat162 simd_operation_is_not_supported_before_sm53(__nv_bfloat162);

template <>
struct SimdSum<__nv_bfloat16>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hadd2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm53(__nv_bfloat162{});));
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMul
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdMul<__half>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (return __hmul2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm53(__half2{});));
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdMul<__nv_bfloat16>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmul2(a, b);), //
                 (return simd_operation_is_not_supported_before_sm53(__nv_bfloat162{});));
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename ReductionOp>
inline constexpr bool is_simd_operator_v = false;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdSum<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMul<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMin<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMax<T>> = true;

//----------------------------------------------------------------------------------------------------------------------
// Predefined CUDA operators to SIMD

template <typename ReduceOp, typename T>
struct CudaOperatorToSimd
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

template <typename T>
struct CudaOperatorToSimd<::cuda::minimum<>, T>
{
  using type = SimdMin<T>;
};

template <typename T>
struct CudaOperatorToSimd<::cuda::minimum<T>, T>
{
  using type = SimdMin<T>;
};

template <typename T>
struct CudaOperatorToSimd<::cuda::maximum<>, T>
{
  using type = SimdMax<T>;
};

template <typename T>
struct CudaOperatorToSimd<::cuda::maximum<T>, T>
{
  using type = SimdMax<T>;
};

template <typename T>
struct CudaOperatorToSimd<_CUDA_VSTD::plus<>, T>
{
  using type = SimdSum<T>;
};

template <typename T>
struct CudaOperatorToSimd<_CUDA_VSTD::plus<T>, T>
{
  using type = SimdSum<T>;
};

template <typename T>
struct CudaOperatorToSimd<_CUDA_VSTD::multiplies<>, T>
{
  using type = SimdMul<T>;
};

template <typename T>
struct CudaOperatorToSimd<_CUDA_VSTD::multiplies<T>, T>
{
  using type = SimdMul<T>;
};

template <typename ReduceOp, typename T>
using cub_operator_to_simd_operator_t = typename CudaOperatorToSimd<ReduceOp, T>::type;

//----------------------------------------------------------------------------------------------------------------------
// SIMD type

template <typename T>
struct SimdType
{
  static_assert(_CUDA_VSTD::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct SimdType<int16_t>
{
  using type = uint32_t;
};

template <>
struct SimdType<uint16_t>
{
  using type = uint32_t;
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdType<__half>
{
  using type = __half2;
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdType<__nv_bfloat16>
{
  using type = __nv_bfloat162;
};

#  endif // _CCCL_HAS_NVBF16()

template <typename T>
using simd_type_t = typename SimdType<T>::type;

} // namespace detail

#endif // !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
