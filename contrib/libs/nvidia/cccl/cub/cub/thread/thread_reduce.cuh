/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

//! @file
//! Thread reduction over statically-sized array-like types
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/array_utils.cuh> // to_array()
#include <cub/detail/type_traits.cuh> // are_same()
#include <cub/detail/unsafe_bitcast.cuh>
#include <cub/thread/thread_load.cuh> // UnrolledCopy
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_simd.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/functional> // cuda::maximum
#include <cuda/std/array> // array
#include <cuda/std/cassert> // assert
#include <cuda/std/cstdint> // uint16_t
#include <cuda/std/functional> // cuda::std::plus
#include <cuda/std/iterator> // cuda::std::iter_value_t

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``ThreadReduce`` function computes a reduction of items assigned to a single CUDA thread.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! - A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`__ (or *fold*)
//!   uses a binary combining operator to compute a single aggregate from a list of input elements.
//! - Supports array-like types that are statically-sized and can be indexed with the ``[] operator``:
//!   raw arrays, ``std::array``, ``std::span``,  ``std::mdspan`` (C++23)
//!
//! Main Function and Overloading
//! +++++++++++++++++++++++++++++
//!
//! Reduction over statically-sized array-like types, seeded with the specified prefix
//!
//! .. code-block:: c++
//!
//!    template <typename Input,
//!              typename ReductionOp,
//!              typename ValueT = ..., // type of a single input element
//!              typename AccumT = ...> // accumulator type
//!    [[nodiscard]] __device__ __forceinline__ AccumT
//!    ThreadReduce(const Input& input, ReductionOp reduction_op)
//!
//! .. code-block:: c++
//!
//!    template <typename Input,
//!              typename ReductionOp,
//!              typename PrefixT,
//!              typename ValueT = ..., // type of a single input element
//!              typename AccumT = ...> // accumulator type
//!    [[nodiscard]] __device__ __forceinline__ AccumT
//!    ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
//!
//! Performance Considerations
//! ++++++++++++++++++++++++++
//!
//! The function provides the following optimizations:
//!
//! - *Vectorization/SIMD* for:
//!
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM90+ for ``int16_t/uint16_t``
//!     data types (Hopper DPX instructions)
//!   - Sum (``cuda::std::plus<>``) and Multiplication (``cuda::std::multiplies<>``) on SM80+ for ``__nv_bfloat16``
//!     data type
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM80+ for ``__half/__nv_bfloat16``
//!     data types
//!   - Sum (``cuda::std::plus<>``) and Multiplication (``cuda::std::multiplies<>``) on SM70+ for ``__half`` data type
//!
//! - *Instruction-Level Parallelism (ILP)* by exploiting a *ternary tree reduction* for:
//!
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM90+ for ``int32_t/uint32_t`` data types
//!     (Hopper DPX instructions)
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM80+ for integer data types (Hopper DPX
//!     instructions), ``__half2``, ``__nv_bfloat162``, ``__half`` (after vectorization), and ``__nv_bfloat16``
//!     (after vectorization) data types
//!   - Sum (``cuda::std::plus<>``), Bitwise AND (``cuda::std::bit_and<>``), OR (``cuda::std::bit_or<>``), XOR
//!     (``cuda::std::bit_xor<>``) on SM50+ for integer data types
//!
//! - *Instruction-Level Parallelism (ILP)* by exploiting a *binary tree reduction* for
//!
//!   - All other cases that maps to predefined operators
//!
//! Simple Example
//! ++++++++++++++++++++++++++
//!
//! The code snippet below illustrates a simple sum reductions over 4 integer values.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        int array[4] = {1, 2, 3, 4};
//!        int sum      = cub::ThreadReduce(array, _CUDA_VSTD::plus<>{}); // sum = 10
//!
//! @endrst
//!
//! @brief Reduction over statically-sized array-like types.
//!
//! @tparam Input
//!   <b>[inferred]</b> The data type to be reduced having member
//!   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
//!
//! @tparam ReductionOp
//!   <b>[inferred]</b> Binary reduction operator type having member
//!   <tt>T operator()(const T &a, const T &b)</tt>
//!
//! @param[in] input
//!   Array=like input
//!
//! @param[in] reduction_op
//!   Binary reduction operator
//!
//! @return Accumulation of type (simplified) ``decltype(reduction_op(a, b))`` see
//! <a
//! href="https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2322r6.html#return-the-result-of-the-initial-invocation">P2322</a>
//!

template <typename Input,
          typename ReductionOp,
          typename ValueT = _CUDA_VSTD::iter_value_t<Input>,
          typename AccumT = _CUDA_VSTD::__accumulator_t<ReductionOp, ValueT>>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op);
// forward declaration

/***********************************************************************************************************************
 * Internal Reduction Implementations
 **********************************************************************************************************************/
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace detail
{

/***********************************************************************************************************************
 * Enable SIMD/Tree reduction heuristics (Trait)
 **********************************************************************************************************************/

/// DPX instructions compute min, max, and sum for up to three 16 and 32-bit signed or unsigned integer parameters
/// see DPX documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
/// NOTE: The compiler is able to automatically vectorize all cases with 3 operands
///       However, all other cases with per-halfword comparison need to be explicitly vectorized
///
/// DPX reduction is enabled if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 2 bytes
/// - DPX instructions provide Min, Max SIMD operations
/// If the number of instructions is the same, we favor the compiler
///
/// length | Standard |  DPX
///  2     |    1     |  NA
///  3     |    1     |  NA
///  4     |    2     |  3
///  5     |    2     |  3
///  6     |    3     |  3
///  7     |    3     |  3
///  8     |    4     |  4
///  9     |    4     |  4
/// 10     |    5     |  4 // ***
/// 11     |    5     |  4 // ***
/// 12     |    6     |  5 // ***
/// 13     |    6     |  5 // ***
/// 14     |    7     |  5 // ***
/// 15     |    7     |  5 // ***
/// 16     |    8     |  6 // ***

// TODO: add Blackwell support

//----------------------------------------------------------------------------------------------------------------------
// SM90 SIMD

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm90_simd_reduction_v =
  is_one_of_v<T, int16_t, uint16_t> && is_cuda_minimum_maximum_v<ReductionOp, T> && Length >= 10;

//----------------------------------------------------------------------------------------------------------------------
// SM80 SIMD

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v = false;

#  if _CCCL_HAS_NVFP16()

template <typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v<__half, ReductionOp, Length> =
  (is_cuda_minimum_maximum_v<ReductionOp, __half> || is_cuda_std_plus_mul_v<ReductionOp, __half>) && Length >= 4;

#  endif // defined(_CCCL_HAS_NVFP16)

#  if _CCCL_HAS_NVBF16()

template <typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v<__nv_bfloat16, ReductionOp, Length> =
  (is_cuda_minimum_maximum_v<ReductionOp, __nv_bfloat16> || is_cuda_std_plus_mul_v<ReductionOp, __nv_bfloat16>)
  && Length >= 4;

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------
// SM70 SIMD

#  if _CCCL_HAS_NVFP16()

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm70_simd_reduction_v =
  _CUDA_VSTD::is_same_v<T, __half> && is_cuda_std_plus_mul_v<ReductionOp, T> && Length >= 4;

#  else // _CCCL_HAS_NVFP16() ^^^^ / !_CCCL_HAS_NVFP16() vvvv

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm70_simd_reduction_v = false;

#  endif // !_CCCL_HAS_NVFP16() ^^^^

/***********************************************************************************************************************
 * Enable Ternary Reduction (Trait)
 **********************************************************************************************************************/

template <typename T, typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v =
  is_one_of_v<T, int32_t, uint32_t> && is_cuda_minimum_maximum_v<ReductionOp, T>;

#  if _CCCL_HAS_NVFP16()

template <typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v<__half2, ReductionOp> =
  is_cuda_minimum_maximum_v<ReductionOp, __half2> || is_one_of_v<ReductionOp, SimdMin<__half>, SimdMax<__half>>;

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v<__nv_bfloat162, ReductionOp> =
  is_cuda_minimum_maximum_v<ReductionOp, __nv_bfloat162>
  || is_one_of_v<ReductionOp, SimdMin<__nv_bfloat16>, SimdMax<__nv_bfloat16>>;

#  endif // _CCCL_HAS_NVBF16()

template <typename T, typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm50_v =
  _CUDA_VSTD::is_integral_v<T> && sizeof(T) <= 4
  && (is_cuda_std_plus_v<ReductionOp, T> || is_cuda_std_bitwise_v<ReductionOp, T>);

/***********************************************************************************************************************
 * Internal Reduction Algorithms: Sequential, Binary, Ternary
 **********************************************************************************************************************/

template <typename AccumT, typename Input, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduceSequential(const Input& input, ReductionOp reduction_op)
{
  auto retval = static_cast<AccumT>(input[0]);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 1; i < static_size_v<Input>; ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

template <typename AccumT, typename Input, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduceBinaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = static_size_v<Input>;
  auto array            = cub::detail::to_array<AccumT>(input);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 1; i < length; i *= 2)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j + i < length; j += i * 2)
    {
      array[j] = reduction_op(array[j], array[j + i]);
    }
  }
  return array[0];
}

template <typename AccumT, typename Input, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduceTernaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = static_size_v<Input>;
  auto array            = cub::detail::to_array<AccumT>(input);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 1; i < length; i *= 3)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j + i < length; j += i * 3)
    {
      auto value = reduction_op(array[j], array[j + i]);
      array[j]   = (j + i * 2 < length) ? reduction_op(value, array[j + i * 2]) : value;
    }
  }
  return array[0];
}

/***********************************************************************************************************************
 * SIMD Reduction
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE auto ThreadReduceSimd(const Input& input, ReductionOp)
{
  using cub::detail::unsafe_bitcast;
  using T                       = _CUDA_VSTD::iter_value_t<Input>;
  using SimdReduceOp            = cub_operator_to_simd_operator_t<ReductionOp, T>;
  using SimdType                = simd_type_t<T>;
  constexpr auto length         = static_size_v<Input>;
  constexpr auto simd_ratio     = sizeof(SimdType) / sizeof(T);
  constexpr auto length_rounded = ::cuda::round_down(length, simd_ratio);
  using UnpackedType            = _CUDA_VSTD::array<T, simd_ratio>;
  using SimdArray               = _CUDA_VSTD::array<SimdType, length / simd_ratio>;
  static_assert(simd_ratio == 2, "Only SIMD size == 2 is supported");
  T local_array[length_rounded];
  UnrolledCopy<length_rounded>(input, local_array);
  auto simd_input      = unsafe_bitcast<SimdArray>(local_array);
  auto simd_reduction  = cub::ThreadReduce(simd_input, SimdReduceOp{});
  auto unpacked_values = unsafe_bitcast<UnpackedType>(simd_reduction);
  // Create a reversed copy of the SIMD reduction result and apply the SIMD operator.
  // This avoids redundant instructions for converting to and from 32-bit registers
  T unpacked_values_rev[] = {unpacked_values[1], unpacked_values[0]};
  auto simd_reduction_rev = unsafe_bitcast<SimdType>(unpacked_values_rev);
  SimdType result         = SimdReduceOp{}(simd_reduction, simd_reduction_rev);
  // repeat the same optimization for the last element
  if constexpr (length % simd_ratio == 1)
  {
    T tail[]       = {input[length - 1], T{}};
    auto tail_simd = unsafe_bitcast<SimdType>(tail);
    result         = SimdReduceOp{}(result, tail_simd);
  }
  return unsafe_bitcast<UnpackedType>(result)[0];
}

template <typename ReductionOp, typename T>
inline constexpr bool enable_min_max_promotion_v =
  is_cuda_minimum_maximum_v<ReductionOp, T> && _CUDA_VSTD::is_integral_v<T> && sizeof(T) <= 2;

} // namespace detail

/***********************************************************************************************************************
 * Reduction Interface/Dispatch (public)
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, typename ValueT, typename AccumT>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op)
{
  using namespace cub::detail;
  static_assert(is_fixed_size_random_access_range_v<Input>,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");

  static constexpr auto length = static_size_v<Input>;
  if constexpr (length == 1)
  {
    return static_cast<AccumT>(input[0]);
  }

  using PromT = ::cuda::std::_If<enable_min_max_promotion_v<ReductionOp, ValueT>, int, AccumT>;
  // TODO: should be part of the tuning policy
  if constexpr ((!is_simd_enabled_cuda_operator<ReductionOp, ValueT> && !is_simd_operator_v<ReductionOp>)
                || sizeof(ValueT) >= 8)
  {
    return ThreadReduceSequential<AccumT>(input, reduction_op);
  }

  if constexpr (::cuda::std::is_same_v<ValueT, AccumT> && enable_sm90_simd_reduction_v<ValueT, ReductionOp, length>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return ThreadReduceSimd(input, reduction_op);))
  }

  if constexpr (::cuda::std::is_same_v<ValueT, AccumT> && enable_sm80_simd_reduction_v<ValueT, ReductionOp, length>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return ThreadReduceSimd(input, reduction_op);))
  }

  if constexpr (::cuda::std::is_same_v<ValueT, AccumT> && enable_sm70_simd_reduction_v<ValueT, ReductionOp, length>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_70, (return ThreadReduceSimd(input, reduction_op);))
  }

  if constexpr (length >= 6)
  {
    // apply SM90 min/max ternary reduction only if the input is natively int32/uint32
    if constexpr (enable_ternary_reduction_sm90_v<ValueT, ReductionOp>)
    {
      // with the current tuning policies, SM90/int32/+ uses too many registers (TODO: fix tuning policy)
      if constexpr ((is_one_of_v<ReductionOp, ::cuda::std::plus<>, ::cuda::std::plus<PromT>>
                     && is_one_of_v<PromT, int32_t, uint32_t>)
                    // the compiler generates bad code for int8/uint8 and min/max for SM90
                    || (is_cuda_minimum_maximum_v<ReductionOp, ValueT> && is_one_of_v<PromT, int8_t, uint8_t>) )
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, (return ThreadReduceSequential<PromT>(input, reduction_op);));
      }
      NV_IF_TARGET(NV_PROVIDES_SM_90, (return ThreadReduceTernaryTree<PromT>(input, reduction_op);));
    }

    if constexpr (enable_ternary_reduction_sm50_v<ValueT, ReductionOp>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_50, (return ThreadReduceSequential<PromT>(input, reduction_op);));
    }
  }

  return ThreadReduceBinaryTree<PromT>(input, reduction_op);
}

//! @brief Reduction over statically-sized array-like types, seeded with the specified @p prefix.
//!
//! @tparam Input
//!   <b>[inferred]</b> The data type to be reduced having member
//!   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
//!
//! @tparam ReductionOp
//!   <b>[inferred]</b> Binary reduction operator type having member
//!   <tt>T operator()(const T &a, const T &b)</tt>
//!
//! @tparam PrefixT
//!   <b>[inferred]</b> The prefix type
//!
//! @param[in] input
//!   Input array
//!
//! @param[in] reduction_op
//!   Binary reduction operator
//!
//! @param[in] prefix
//!   Prefix to seed reduction with
//!
//! @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
//!
template <typename Input,
          typename ReductionOp,
          typename PrefixT,
          typename ValueT = _CUDA_VSTD::iter_value_t<Input>,
          typename AccumT = _CUDA_VSTD::__accumulator_t<ReductionOp, ValueT, PrefixT>>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
{
  using namespace cub::detail;
  static_assert(is_fixed_size_random_access_range_v<Input>,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr int length = static_size_v<Input>;
  // copy to a temporary array of type AccumT
  AccumT array[length + 1];
  array[0] = prefix;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < length; ++i)
  {
    array[i + 1] = input[i];
  }
  return cub::ThreadReduce<decltype(array), ReductionOp, AccumT, AccumT>(array, reduction_op);
}

#endif // !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
