// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_merge.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/functional>

CUB_NAMESPACE_BEGIN

//! DeviceMerge provides device-wide, parallel operations for merging two sorted sequences of values (called keys) or
//! key-value pairs in device-accessible memory. The sorting order is determined by a comparison functor (default:
//! less-than), which has to establish a [strict weak ordering].
//!
//! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
struct DeviceMerge
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of values (called keys) into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! The code snippet below illustrates the merging of two device vectors of `int` keys.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-keys
  //!     :end-before: example-end merge-keys
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1 **[deduced]** Random access iterator to the first sorted input sequence. Must have the same
  //! value type as KeyIteratorIn2.
  //! @tparam KeyIteratorIn2 **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //! same value type as KeyIteratorIn1.
  //! @tparam KeyIteratorOut **[deduced]** Random access iterator to the output sequence.
  //! @tparam CompareOp **[deduced]** Binary predicate to compare the input iterator's value types. Must have a
  //! signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @param[in] d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required
  //! allocation size is written to `temp_storage_bytes` and no work is done.
  //! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation.
  //! @param[in] keys_in1 Iterator to the beginning of the first sorted input sequence.
  //! @param[in] num_keys1 Number of keys in the first input sequence.
  //! @param[in] keys_in2 Iterator to the beginning of the second sorted input sequence.
  //! @param[in] num_keys2 Number of keys in the second input sequence.
  //! @param[out] keys_out Iterator to the beginning of the output sequence.
  //! @param[in] compare_op Comparison function object, returning true if the first argument is ordered before the
  //! second. Must establish a [strict weak ordering].
  //! @param[in] stream **[optional]** CUDA stream to launch kernels into. Default is stream<sub>0</sub>.
  //!
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename CompareOp = ::cuda::std::less<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MergeKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 keys_in1,
    ::cuda::std::int64_t num_keys1,
    KeyIteratorIn2 keys_in2,
    ::cuda::std::int64_t num_keys2,
    KeyIteratorOut keys_out,
    CompareOp compare_op = {},
    cudaStream_t stream  = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceMerge::MergeKeys");

    using offset_t = ::cuda::std::int64_t;

    return detail::merge::
      dispatch_t<KeyIteratorIn1, NullType*, KeyIteratorIn2, NullType*, KeyIteratorOut, NullType*, offset_t, CompareOp>::
        dispatch(
          d_temp_storage,
          temp_storage_bytes,
          keys_in1,
          nullptr,
          num_keys1,
          keys_in2,
          nullptr,
          num_keys2,
          keys_out,
          nullptr,
          compare_op,
          stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of key-value pairs into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! The code snippet below illustrates the merging of two device vectors of `int` keys.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-pairs
  //!     :end-before: example-end merge-pairs
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1 **[deduced]** Random access iterator to the keys of the first sorted input sequence. Must
  //! have the same value type as KeyIteratorIn2.
  //! @tparam ValueIteratorIn1 **[deduced]** Random access iterator to the values of the first sorted input sequence.
  //! Must have the same value type as ValueIteratorIn2.
  //! @tparam KeyIteratorIn2 **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //! same value type as KeyIteratorIn1.
  //! @tparam ValueIteratorIn2 **[deduced]** Random access iterator to the values of the second sorted input sequence.
  //! Must have the same value type as ValueIteratorIn1.
  //! @tparam KeyIteratorOut **[deduced]** Random access iterator to the keys of the output sequence.
  //! @tparam ValueIteratorOut **[deduced]** Random access iterator to the values of the output sequence.
  //! @tparam CompareOp **[deduced]** Binary predicate to compare the key input iterator's value types. Must have a
  //! signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @param[in] d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required
  //! allocation size is written to `temp_storage_bytes` and no work is done.
  //! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation.
  //! @param[in] keys_in1 Iterator to the beginning of the keys of the first sorted input sequence.
  //! @param[in] values_in1 Iterator to the beginning of the values of the first sorted input sequence.
  //! @param[in] num_pairs1 Number of key-value pairs in the first input sequence.
  //! @param[in] keys_in2 Iterator to the beginning of the keys of the second sorted input sequence.
  //! @param[in] values_in2 Iterator to the beginning of the values of the second sorted input sequence.
  //! @param[in] num_pairs2 Number of key-value pairs in the second input sequence.
  //! @param[out] keys_out Iterator to the beginning of the keys of the output sequence.
  //! @param[out] values_out Iterator to the beginning of the values of the output sequence.
  //! @param[in] compare_op Comparison function object, returning true if the first argument is ordered before the
  //! second. Must establish a [strict weak ordering].
  //! @param[in] stream **[optional]** CUDA stream to launch kernels into. Default is stream<sub>0</sub>.
  //!
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename CompareOp = ::cuda::std::less<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MergePairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 keys_in1,
    ValueIteratorIn1 values_in1,
    ::cuda::std::int64_t num_pairs1,
    KeyIteratorIn2 keys_in2,
    ValueIteratorIn2 values_in2,
    ::cuda::std::int64_t num_pairs2,
    KeyIteratorOut keys_out,
    ValueIteratorOut values_out,
    CompareOp compare_op = {},
    cudaStream_t stream  = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceMerge::MergePairs");

    using offset_t = ::cuda::std::int64_t;

    return detail::merge::dispatch_t<
      KeyIteratorIn1,
      ValueIteratorIn1,
      KeyIteratorIn2,
      ValueIteratorIn2,
      KeyIteratorOut,
      ValueIteratorOut,
      offset_t,
      CompareOp>::dispatch(d_temp_storage,
                           temp_storage_bytes,
                           keys_in1,
                           values_in1,
                           num_pairs1,
                           keys_in2,
                           values_in2,
                           num_pairs2,
                           keys_out,
                           values_out,
                           compare_op,
                           stream);
  }
};

CUB_NAMESPACE_END
