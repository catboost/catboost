/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/device/dispatch/dispatch_for.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/distance.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace for_each
{

/**
 * `op_wrapper_t` turns bulk into a for-each operation by wrapping the user-provided unary operator.
 */
template <class OffsetT, class OpT, class RandomAccessIteratorT>
struct op_wrapper_t
{
  RandomAccessIteratorT input;
  OpT op;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(OffsetT i)
  {
    // Dereferencing `thrust::device_vector<T>` iterators returns a `thrust::device_reference<T>`
    // instead of `T`. Since user-provided operator expects `T` as an argument, we need to unwrap.
    (void) op(THRUST_NS_QUALIFIER::raw_reference_cast(*(input + i)));
  }
};

/**
 * `op_wrapper_vectorized_t` turns bulk into a for-each-copy operation.
 * `op_wrapper_vectorized_t` is similar to `op_wrapper_t` but does not provide any guarantees about
 * address of the input parameter. `OpT` might be given a copy of the value or an actual reference
 * to the input iterator value (depending on the alignment of input iterator)
 */
template <class OffsetT, class OpT, class T>
struct op_wrapper_vectorized_t
{
  const T* input; // Raw pointer to the input data
  OpT op; // User-provided operator
  OffsetT partially_filled_vector_id; // Index of the vector that doesn't have all elements
  OffsetT num_items; // Total number of non-vectorized items

  // TODO Can be extracted into tuning
  constexpr static int vec_size = 4;

  // Type of the vector that is used to load the input data
  using vector_t = typename CubVector<T, vec_size>::Type;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(OffsetT i)
  {
    // Surrounding `Bulk` call doesn't invoke this operator on invalid indices, so we don't need to
    // check for out-of-bounds access here.
    if (i != partially_filled_vector_id)
    { // Case of fully filled vector
      const vector_t vec = *reinterpret_cast<const vector_t*>(input + vec_size * i);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < vec_size; j++)
      {
        (void) op(*(reinterpret_cast<const T*>(&vec) + j));
      }
    }
    else
    { // Case of partially filled vector
      for (OffsetT j = i * vec_size; j < num_items; j++)
      {
        (void) op(input[j]);
      }
    }
  }
};

} // namespace for_each
} // namespace detail

struct DeviceFor
{
private:
  /**
   * Checks if the pointer is aligned to the given vector type
   */
  template <class VectorT, class T>
  CUB_RUNTIME_FUNCTION static bool is_aligned(const T* ptr)
  {
    return (reinterpret_cast<size_t>(ptr) & (sizeof(VectorT) - 1)) == 0;
  }

  template <bool UseVectorization, class RandomAccessOrContiguousIteratorT, class OffsetT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  for_each_n(RandomAccessOrContiguousIteratorT first, OffsetT num_items, OpT op, cudaStream_t stream)
  {
    if constexpr (UseVectorization)
    {
      auto* unwrapped_first = THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(first);
      using wrapped_op_t =
        detail::for_each::op_wrapper_vectorized_t<OffsetT, OpT, detail::it_value_t<RandomAccessOrContiguousIteratorT>>;

      if (is_aligned<typename wrapped_op_t::vector_t>(unwrapped_first))
      { // Vectorize loads
        const OffsetT num_vec_items = ::cuda::ceil_div(num_items, wrapped_op_t::vec_size);

        return detail::for_each::dispatch_t<OffsetT, wrapped_op_t>::dispatch(
          num_vec_items,
          wrapped_op_t{
            unwrapped_first, op, num_items % wrapped_op_t::vec_size ? num_vec_items - 1 : num_vec_items, num_items},
          stream);
      }

      // Fallback to non-vectorized version
      return for_each_n<false>(first, num_items, op, stream);
    }
    else
    {
      using wrapped_op_t = detail::for_each::op_wrapper_t<OffsetT, OpT, RandomAccessOrContiguousIteratorT>;
      return detail::for_each::dispatch_t<OffsetT, wrapped_op_t>::dispatch(num_items, wrapped_op_t{first, op}, stream);
    }
  }

public:
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each index in the provided shape
  //! The algorithm is similar to
  //! `bulk <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html#design-sender-adaptor-bulk>`_
  //! from P2300.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use Bulk to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-t
  //!     :end-before: example-end bulk-square-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-temp-storage
  //!     :end-before: example-end bulk-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam ShapeT
  //!   is an integral type
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] shape
  //!   Shape of the index space to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to each index in the index space
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class ShapeT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Bulk(void* d_temp_storage, size_t& temp_storage_bytes, ShapeT shape, OpT op, cudaStream_t stream = {})
  {
    static_assert(_CUDA_VSTD::is_integral_v<ShapeT>, "ShapeT must be an integral type");

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Bulk(shape, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, first + num_items)``
  //!
  //! - The return value of ``op``, if any, is ignored.
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachN` to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-ref-t
  //!     :end-before: example-end bulk-square-ref-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-n-temp-storage
  //!     :end-before: example-end for-each-n-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam NumItemsT
  //!   is an integral type representing the number of elements to iterate over
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] num_items
  //!   Number of elements to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachN(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT first,
    NumItemsT num_items,
    OpT op,
    cudaStream_t stream = {})
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return ForEachN(first, num_items, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, last)``
  //!
  //! - The return value of ``op``, if any, is ignored.
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEach` to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-ref-t
  //!     :end-before: example-end bulk-square-ref-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-temp-storage
  //!     :end-before: example-end for-each-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] last
  //!   The end of the sequence
  //!
  //! @param[in] op
  //!   Function object to apply to each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEach(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT first,
    RandomAccessIteratorT last,
    OpT op,
    cudaStream_t stream = {})
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return ForEach(first, last, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, first + num_items)``.
  //! Unlike the ``ForEachN`` algorithm, ``ForEachCopyN`` is allowed to invoke ``op`` on copies of the elements.
  //! This relaxation allows ``ForEachCopyN`` to vectorize loads.
  //!
  //! - Allowed to invoke ``op`` on copies of the elements
  //! - The return value of ``op``, if any, is ignored.
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachCopyN` to count odd elements in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-odd-count-t
  //!     :end-before: example-end bulk-odd-count-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-copy-n-temp-storage
  //!     :end-before: example-end for-each-copy-n-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam NumItemsT
  //!   is an integral type representing the number of elements to iterate over
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] num_items
  //!   Number of elements to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to a copy of each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachCopyN(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT first,
    NumItemsT num_items,
    OpT op,
    cudaStream_t stream = {})
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return ForEachCopyN(first, num_items, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, last)``.
  //! Unlike the ``ForEach`` algorithm, ``ForEachCopy`` is allowed to invoke ``op`` on copies of the elements.
  //! This relaxation allows ``ForEachCopy`` to vectorize loads.
  //!
  //! - Allowed to invoke ``op`` on copies of the elements
  //! - The return value of ``op``, if any, is ignored.
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachCopy` to count odd elements in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-odd-count-t
  //!     :end-before: example-end bulk-odd-count-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-copy-temp-storage
  //!     :end-before: example-end for-each-copy-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] last
  //!   The end of the sequence
  //!
  //! @param[in] op
  //!   Function object to apply to a copy of each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachCopy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT first,
    RandomAccessIteratorT last,
    OpT op,
    cudaStream_t stream = {})
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return ForEachCopy(first, last, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each index in the provided shape
  //! The algorithm is similar to
  //! `bulk <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html#design-sender-adaptor-bulk>`_
  //! from P2300.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use Bulk to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-t
  //!     :end-before: example-end bulk-square-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-wo-temp-storage
  //!     :end-before: example-end bulk-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam ShapeT
  //!   is an integral type
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] shape
  //!   Shape of the index space to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to each index in the index space
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class ShapeT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t Bulk(ShapeT shape, OpT op, cudaStream_t stream = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::Bulk");
    static_assert(_CUDA_VSTD::is_integral_v<ShapeT>, "ShapeT must be an integral type");
    using offset_t = ShapeT;
    return detail::for_each::dispatch_t<offset_t, OpT>::dispatch(static_cast<offset_t>(shape), op, stream);
  }

private:
  // Internal version without NVTX raNGE
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachNNoNVTX(RandomAccessIteratorT first, NumItemsT num_items, OpT op, cudaStream_t stream = {})
  {
    using offset_t = NumItemsT;
    // Disable auto-vectorization for now:
    // constexpr bool use_vectorization =
    //   detail::for_each::can_regain_copy_freedom<detail::it_value_t<RandomAccessIteratorT>, OpT>::value
    //   && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorT>;
    constexpr bool use_vectorization = false;
    return for_each_n<use_vectorization, RandomAccessIteratorT, offset_t, OpT>(first, num_items, op, stream);
  }

public:
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, first + num_items)``
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachN` to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-ref-t
  //!     :end-before: example-end bulk-square-ref-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-n-wo-temp-storage
  //!     :end-before: example-end for-each-n-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam NumItemsT
  //!   is an integral type representing the number of elements to iterate over
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] num_items
  //!   Number of elements to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachN(RandomAccessIteratorT first, NumItemsT num_items, OpT op, cudaStream_t stream = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEachN");
    return ForEachNNoNVTX(first, num_items, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, last)``
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEach` to square each element in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-ref-t
  //!     :end-before: example-end bulk-square-ref-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-wo-temp-storage
  //!     :end-before: example-end for-each-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] last
  //!   The end of the sequence
  //!
  //! @param[in] op
  //!   Function object to apply to each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEach(RandomAccessIteratorT first, RandomAccessIteratorT last, OpT op, cudaStream_t stream = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEach");

    using offset_t       = detail::it_difference_t<RandomAccessIteratorT>;
    const auto num_items = static_cast<offset_t>(_CUDA_VSTD::distance(first, last));
    return ForEachNNoNVTX(first, num_items, op, stream);
  }

private:
  // Internal version without NVTX range
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachCopyNNoNVTX(RandomAccessIteratorT first, NumItemsT num_items, OpT op, cudaStream_t stream = {})
  {
    using offset_t                   = NumItemsT;
    constexpr bool use_vectorization = THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorT>;
    return for_each_n<use_vectorization, RandomAccessIteratorT, offset_t, OpT>(first, num_items, op, stream);
  }

public:
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, first + num_items)``.
  //! Unlike the ``ForEachN`` algorithm, ``ForEachCopyN`` is allowed to invoke ``op`` on copies of the elements.
  //! This relaxation allows ``ForEachCopyN`` to vectorize loads.
  //!
  //! - Allowed to invoke ``op`` on copies of the elements
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachCopyN` to count odd elements in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-odd-count-t
  //!     :end-before: example-end bulk-odd-count-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-copy-n-wo-temp-storage
  //!     :end-before: example-end for-each-copy-n-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam NumItemsT
  //!   is an integral type representing the number of elements to iterate over
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] num_items
  //!   Number of elements to iterate over
  //!
  //! @param[in] op
  //!   Function object to apply to a copy of each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class NumItemsT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachCopyN(RandomAccessIteratorT first, NumItemsT num_items, OpT op, cudaStream_t stream = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEachCopyN");
    return ForEachCopyNNoNVTX(first, num_items, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Applies the function object ``op`` to each element in the range ``[first, last)``.
  //! Unlike the ``ForEach`` algorithm, ``ForEachCopy`` is allowed to invoke ``op`` on copies of the elements.
  //! This relaxation allows ``ForEachCopy`` to vectorize loads.
  //!
  //! - Allowed to invoke ``op`` on copies of the elements
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use `ForEachCopy` to count odd elements in a device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-odd-count-t
  //!     :end-before: example-end bulk-odd-count-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-copy-wo-temp-storage
  //!     :end-before: example-end for-each-copy-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   is a model of Random Access Iterator whose value type is convertible to `op`'s argument type.
  //!
  //! @tparam OpT
  //!   is a model of [Unary Function](https://en.cppreference.com/w/cpp/utility/functional/unary_function)
  //!
  //! @param[in] first
  //!   The beginning of the sequence
  //!
  //! @param[in] last
  //!   The end of the sequence
  //!
  //! @param[in] op
  //!   Function object to apply to a copy of each element in the range
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `0`.
  template <class RandomAccessIteratorT, class OpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachCopy(RandomAccessIteratorT first, RandomAccessIteratorT last, OpT op, cudaStream_t stream = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEachCopy");
    using offset_t       = detail::it_difference_t<RandomAccessIteratorT>;
    const auto num_items = static_cast<offset_t>(_CUDA_VSTD::distance(first, last));
    return ForEachCopyNNoNVTX(first, num_items, op, stream);
  }

  /*********************************************************************************************************************
   * ForEachInExtents
   ********************************************************************************************************************/

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Iterate through a multi-dimensional extents into
  //!
  //! - a single linear index that represents the current iteration
  //! - indices of each extent dimension
  //!
  //! Then apply a function object to the results.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! **Note**: ``DeviceFor::ForEachInExtents`` supports integral index type up to 64-bits.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use ``ForEachInExtents`` to tabulate a 3D array with its
  //! coordinates.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_each_in_extents_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-in-extents-op
  //!     :end-before: example-end for-each-in-extents-op
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_each_in_extents_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-in-extents-example
  //!     :end-before: example-end for-each-in-extents-example
  //!
  //! @endrst
  //!
  //! @tparam IndexType
  //!   is an integral type that represents the extent index space (automatically deduced)
  //!
  //! @tparam Extents
  //!   are the extent sizes for each rank index (automatically deduced)
  //!
  //! @tparam OpType
  //!   is a function object with arity equal to the number of extents + 1 for the linear index (iteration)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] extents
  //!   Extents object that represents a multi-dimensional index space
  //!
  //! @param[in] op
  //!   Function object to apply to each linear index (iteration) and multi-dimensional coordinates
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `NULL`
  //!
  //! @return cudaError_t
  //!   error status
  template <typename IndexType, size_t... Extents, typename OpType>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachInExtents(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const _CUDA_VSTD::extents<IndexType, Extents...>& extents,
    OpType op,
    cudaStream_t stream = {})
  {
    // TODO: check dimensions overflows
    // TODO: check tha arity of OpType is equal to sizeof...(ExtentsType)
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }
    return ForEachInExtents(extents, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Iterate through a multi-dimensional extents producing
  //!
  //! - a single linear index that represents the current iteration
  //! - list of indices containing the coordinates for each extent dimension
  //!
  //! Then apply a function object to each tuple of linear index and multidimensional coordinate list.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! **Note**: ``DeviceFor::ForEachInExtents`` supports integral index type up to 64-bits.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use ``ForEachInExtents`` to tabulate a 3D array with its
  //! coordinates.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_each_in_extents_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-in-extents-op
  //!     :end-before: example-end for-each-in-extents-op
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_each_in_extents_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin for-each-in-extents-example
  //!     :end-before: example-end for-each-in-extents-example
  //!
  //! @endrst
  //!
  //! @tparam IndexType
  //!   is an integral type that represents the extent index space (automatically deduced)
  //!
  //! @tparam Extents
  //!   are the extent sizes for each rank index (automatically deduced)
  //!
  //! @tparam OpType
  //!   is a function object with arity equal to the number of extents + 1 for the linear index (iteration)
  //!
  //! @param[in] extents
  //!   Extents object that represents a multi-dimensional index space
  //!
  //! @param[in] op
  //!   Function object to apply to each linear index (iteration) and multi-dimensional coordinates
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `NULL`
  //!
  //! @return cudaError_t
  //!   error status
  template <typename IndexType, size_t... Extents, typename OpType>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachInExtents(const _CUDA_VSTD::extents<IndexType, Extents...>& extents, OpType op, cudaStream_t stream = {})
  {
    using namespace cub::detail;
    using extents_type      = _CUDA_VSTD::extents<IndexType, Extents...>;
    using extent_index_type = typename extents_type::index_type;
    using fast_mod_array_t  = _CUDA_VSTD::array<fast_div_mod<extent_index_type>, extents_type::rank()>;
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEachInExtents");
    static constexpr auto seq            = _CUDA_VSTD::make_index_sequence<extents_type::rank()>{};
    fast_mod_array_t sub_sizes_div_array = cub::detail::sub_sizes_fast_div_mod(extents, seq);
    fast_mod_array_t extents_div_array   = cub::detail::extents_fast_div_mod(extents, seq);
    for_each::op_wrapper_extents_t<OpType, extents_type, fast_mod_array_t> op_wrapper{
      op, extents, sub_sizes_div_array, extents_div_array};
    return Bulk(static_cast<implicit_prom_t<extent_index_type>>(cub::detail::size(extents)), op_wrapper, stream);
  }
};

CUB_NAMESPACE_END
