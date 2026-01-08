/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <cub/device/dispatch/dispatch_merge_sort.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_namespace.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceMergeSort provides device-wide, parallel operations for
 *        computing a merge sort across a sequence of data items residing within
 *        device-accessible memory.
 *
 * @ingroup SingleModule
 *
 * @par Overview
 * - DeviceMergeSort arranges items into ascending order using a comparison
 *   functor with less-than semantics. Merge sort can handle arbitrary types (as
 *   long as a value of these types is a model of [LessThan Comparable]) and
 *   comparison functors, but is slower than DeviceRadixSort when sorting
 *   arithmetic types into ascending/descending order.
 * - Another difference from RadixSort is the fact that DeviceMergeSort can
 *   handle arbitrary random-access iterators, as shown below.
 *
 * @par A Simple Example
 * @par
 * The code snippet below illustrates a thrust reverse iterator usage.
 * @par
 * @code
 * #include <cub/cub.cuh>  // or equivalently <cub/device/device_merge_sort.cuh>
 *
 * struct CustomLess
 * {
 *   template <typename DataType>
 *   __device__ bool operator()(const DataType &lhs, const DataType &rhs)
 *   {
 *     return lhs < rhs;
 *   }
 * };
 *
 * // Declare, allocate, and initialize device-accessible pointers
 * // for sorting data
 * thrust::device_vector<KeyType> d_keys(num_items);
 * thrust::device_vector<DataType> d_values(num_items);
 * // ...
 *
 * // Initialize iterator
 * using KeyIterator = typename thrust::device_vector<KeyType>::iterator;
 * thrust::reverse_iterator<KeyIterator> reverse_iter(d_keys.end());
 *
 * // Determine temporary device storage requirements
 * std::size_t temp_storage_bytes = 0;
 * cub::DeviceMergeSort::SortPairs(
 *   nullptr,
 *   temp_storage_bytes,
 *   reverse_iter,
 *   thrust::raw_pointer_cast(d_values.data()),
 *   num_items,
 *   CustomLess());
 *
 * // Allocate temporary storage
 * cudaMalloc(&d_temp_storage, temp_storage_bytes);
 *
 * // Run sorting operation
 * cub::DeviceMergeSort::SortPairs(
 *   d_temp_storage,
 *   temp_storage_bytes,
 *   reverse_iter,
 *   thrust::raw_pointer_cast(d_values.data()),
 *   num_items,
 *   CustomLess());
 * @endcode
 *
 * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
 */
struct DeviceMergeSort
{

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * SortPairs is not guaranteed to be stable. That is, suppose that i and j are
   * equivalent: neither one is less than the other. It is not guaranteed
   * that the relative order of these two elements will be preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortPairs(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortPairs(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 2, 1, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam ValueIteratorT
   *   is a model of [Random Access Iterator], and `ValueIteratorT` is mutable.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[in,out] d_items
   *   Pointer to the input sequence of unsorted input values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            std::size_t &temp_storage_bytes,
            KeyIteratorT d_keys,
            ValueIteratorT d_items,
            OffsetT num_items,
            CompareOpT compare_op,
            cudaStream_t stream = 0)
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyIteratorT,
                                                 ValueIteratorT,
                                                 KeyIteratorT,
                                                 ValueIteratorT,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys,
                                        d_items,
                                        d_keys,
                                        d_items,
                                        num_items,
                                        compare_op,
                                        stream);
  }

  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            std::size_t &temp_storage_bytes,
            KeyIteratorT d_keys,
            ValueIteratorT d_items,
            OffsetT num_items,
            CompareOpT compare_op,
            cudaStream_t stream,
            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairs<KeyIteratorT, ValueIteratorT, OffsetT, CompareOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_items,
      num_items,
      compare_op,
      stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * - SortPairsCopy is not guaranteed to be stable. That is, suppose
   *   that `i` and `j` are equivalent: neither one is less than the
   *   other. It is not guaranteed that the relative order of these
   *   two elements will be preserved by sort.
   * - Input arrays `d_input_keys` and `d_input_items` are not modified.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of
   * `int` keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortPairsCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortPairsCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 2, 1, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyInputIteratorT
   *   is a model of [Random Access Iterator]. Its `value_type` is a model of
   *   [LessThan Comparable]. This `value_type`'s ordering relation is a
   *   *strict weak ordering* as defined in the [LessThan Comparable]
   *   requirements.
   *
   * @tparam ValueInputIteratorT
   *   is a model of [Random Access Iterator].
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam ValueIteratorT
   *   is a model of [Random Access Iterator], and `ValueIteratorT` is mutable.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_input_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[in] d_input_items
   *   Pointer to the input sequence of unsorted input values
   *
   * @param[out] d_output_keys
   *   Pointer to the output sequence of sorted input keys
   *
   * @param[out] d_output_items
   *   Pointer to the output sequence of sorted input values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns `true` if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsCopy(void *d_temp_storage,
                std::size_t &temp_storage_bytes,
                KeyInputIteratorT d_input_keys,
                ValueInputIteratorT d_input_items,
                KeyIteratorT d_output_keys,
                ValueIteratorT d_output_items,
                OffsetT num_items,
                CompareOpT compare_op,
                cudaStream_t stream = 0)
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyInputIteratorT,
                                                 ValueInputIteratorT,
                                                 KeyIteratorT,
                                                 ValueIteratorT,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_input_keys,
                                        d_input_items,
                                        d_output_keys,
                                        d_output_items,
                                        num_items,
                                        compare_op,
                                        stream);
  }

  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsCopy(void *d_temp_storage,
                std::size_t &temp_storage_bytes,
                KeyInputIteratorT d_input_keys,
                ValueInputIteratorT d_input_items,
                KeyIteratorT d_output_keys,
                ValueIteratorT d_output_items,
                OffsetT num_items,
                CompareOpT compare_op,
                cudaStream_t stream,
                bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairsCopy<KeyInputIteratorT,
                         ValueInputIteratorT,
                         KeyIteratorT,
                         ValueIteratorT,
                         OffsetT,
                         CompareOpT>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_input_keys,
                                     d_input_items,
                                     d_output_keys,
                                     d_output_items,
                                     num_items,
                                     compare_op,
                                     stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * SortKeys is not guaranteed to be stable. That is, suppose that `i` and `j`
   * are equivalent: neither one is less than the other. It is not guaranteed
   * that the relative order of these two elements will be preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys.
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           KeyIteratorT d_keys,
           OffsetT num_items,
           CompareOpT compare_op,
           cudaStream_t stream = 0)
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyIteratorT,
                                                 NullType *,
                                                 KeyIteratorT,
                                                 NullType *,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys,
                                        static_cast<NullType *>(nullptr),
                                        d_keys,
                                        static_cast<NullType *>(nullptr),
                                        num_items,
                                        compare_op,
                                        stream);
  }

  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           KeyIteratorT d_keys,
           OffsetT num_items,
           CompareOpT compare_op,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeys<KeyIteratorT, OffsetT, CompareOpT>(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys,
                                                       num_items,
                                                       compare_op,
                                                       stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * - SortKeysCopy is not guaranteed to be stable. That is, suppose that `i`
   *   and `j` are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Input array d_input_keys is not modified.
   * - Note that the behavior is undefined if the input and output ranges
   *   overlap in any way.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of
   * `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::SortKeysCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::SortKeysCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyInputIteratorT
   *   is a model of [Random Access Iterator]. Its `value_type` is a model of
   *   [LessThan Comparable]. This `value_type`'s ordering relation is a
   *   *strict weak ordering* as defined in the [LessThan Comparable]
   *   requirements.
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_input_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[out] d_output_keys
   *   Pointer to the output sequence of sorted input keys
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyInputIteratorT,
            typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysCopy(void *d_temp_storage,
               std::size_t &temp_storage_bytes,
               KeyInputIteratorT d_input_keys,
               KeyIteratorT d_output_keys,
               OffsetT num_items,
               CompareOpT compare_op,
               cudaStream_t stream = 0)
  {
    using DispatchMergeSortT = DispatchMergeSort<KeyInputIteratorT,
                                                 NullType *,
                                                 KeyIteratorT,
                                                 NullType *,
                                                 OffsetT,
                                                 CompareOpT>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        d_input_keys,
                                        static_cast<NullType *>(nullptr),
                                        d_output_keys,
                                        static_cast<NullType *>(nullptr),
                                        num_items,
                                        compare_op,
                                        stream);
  }

  template <typename KeyInputIteratorT,
            typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysCopy(void *d_temp_storage,
               std::size_t &temp_storage_bytes,
               KeyInputIteratorT d_input_keys,
               KeyIteratorT d_output_keys,
               OffsetT num_items,
               CompareOpT compare_op,
               cudaStream_t stream,
               bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeysCopy<KeyInputIteratorT, KeyIteratorT, OffsetT, CompareOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_input_keys,
      d_output_keys,
      num_items,
      compare_op,
      stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * StableSortPairs is stable: it preserves the relative ordering of equivalent
   * elements. That is, if x and y are elements such that x precedes y,
   * and if the two elements are equivalent (neither x < y nor y < x) then
   * a postcondition of stable_sort is that x still precedes y.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 6, 5, 3, 0, 9]
   * int  *d_values;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::StableSortPairs(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::StableSortPairs(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, d_values, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 6, 8, 9]
   * // d_values    <-- [5, 4, 3, 1, 2, 0, 6]
   * @endcode
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam ValueIteratorT
   *   is a model of [Random Access Iterator], and `ValueIteratorT` is mutable.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[in,out] d_items
   *   Pointer to the input sequence of unsorted input values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  KeyIteratorT d_keys,
                  ValueIteratorT d_items,
                  OffsetT num_items,
                  CompareOpT compare_op,
                  cudaStream_t stream = 0)
  {
    return SortPairs<KeyIteratorT, ValueIteratorT, OffsetT, CompareOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_items,
      num_items,
      compare_op,
      stream);
  }

  template <typename KeyIteratorT,
            typename ValueIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  KeyIteratorT d_keys,
                  ValueIteratorT d_items,
                  OffsetT num_items,
                  CompareOpT compare_op,
                  cudaStream_t stream,
                  bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortPairs<KeyIteratorT, ValueIteratorT, OffsetT, CompareOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_items,
      num_items,
      compare_op,
      stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * StableSortKeys is stable: it preserves the relative ordering of equivalent
   * elements. That is, if `x` and `y` are elements such that `x` precedes `y`,
   * and if the two elements are equivalent (neither `x < y` nor `y < x`) then
   * a postcondition of stable_sort is that `x` still precedes `y`.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys.
   * \par
   * \code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;       // e.g., 7
   * int  *d_keys;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::StableSortKeys(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::StableSortKeys(
   *   d_temp_storage, temp_storage_bytes,
   *   d_keys, num_items, custom_op);
   *
   * // d_keys      <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 KeyIteratorT d_keys,
                 OffsetT num_items,
                 CompareOpT compare_op,
                 cudaStream_t stream = 0)
  {
    return SortKeys<KeyIteratorT, OffsetT, CompareOpT>(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys,
                                                       num_items,
                                                       compare_op,
                                                       stream);
  }

  template <typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 KeyIteratorT d_keys,
                 OffsetT num_items,
                 CompareOpT compare_op,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortKeys<KeyIteratorT, OffsetT, CompareOpT>(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys,
                                                             num_items,
                                                             compare_op,
                                                             stream);
  }

  /**
   * @brief Sorts items using a merge sorting method.
   *
   * @par
   * - StableSortKeysCopy is stable: it preserves the relative ordering of equivalent
   *   elements. That is, if `x` and `y` are elements such that `x` precedes `y`,
   *   and if the two elements are equivalent (neither `x < y` nor `y < x`) then
   *   a postcondition of stable_sort is that `x` still precedes `y`.
   * - Input array d_input_keys is not modified
   * - Note that the behavior is undefined if the input and output ranges overlap
   *   in any way.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys.
   * \par
   * \code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_merge_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;       // e.g., 7
   * int  *d_input_keys;   // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_output_keys;  // must hold at least num_items elements
   * ...
   *
   * // Initialize comparator
   * CustomOpT custom_op;
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage = nullptr;
   * std::size_t temp_storage_bytes = 0;
   * cub::DeviceMergeSort::StableSortKeysCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_input_keys, d_output_keys, num_items, custom_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceMergeSort::StableSortKeysCopy(
   *   d_temp_storage, temp_storage_bytes,
   *   d_input_keys, d_output_keys, num_items, custom_op);
   *
   * // d_output_keys   <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyInputIteratorT
   *   is a model of [Random Access Iterator]. Its `value_type` is a model of
   *   [LessThan Comparable]. This `value_type`'s ordering relation is a
   *   *strict weak ordering* as defined in the [LessThan Comparable]
   *   requirements.
   *
   * @tparam KeyIteratorT
   *   is a model of [Random Access Iterator]. `KeyIteratorT` is mutable, and
   *   its `value_type` is a model of [LessThan Comparable]. This `value_type`'s
   *   ordering relation is a *strict weak ordering* as defined in
   *   the [LessThan Comparable] requirements.
   *
   * @tparam OffsetT
   *   is an integer type for global offsets.
   *
   * @tparam CompareOpT
   *   is a type of callable object with the signature
   *   `bool operator()(KeyT lhs, KeyT rhs)` that models
   *   the [Strict Weak Ordering] concept.
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_input_keys
   *   Pointer to the input sequence of unsorted input keys
   *
   * @param[out] d_output_keys
   *   Pointer to the output sequence of sorted input keys
   *
   * @param[in] num_items
   *   Number of elements in d_input_keys to sort
   *
   * @param[in] compare_op
   *   Comparison function object which returns true if the first argument is
   *   ordered before the second
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   *
   * [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
   * [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
   * [LessThan Comparable]: https://en.cppreference.com/w/cpp/named_req/LessThanComparable
   */
  template <typename KeyInputIteratorT,
            typename KeyIteratorT,
            typename OffsetT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysCopy(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     KeyInputIteratorT d_input_keys,
                     KeyIteratorT d_output_keys,
                     OffsetT num_items,
                     CompareOpT compare_op,
                     cudaStream_t stream = 0)
  {
    return SortKeysCopy<KeyInputIteratorT, KeyIteratorT, OffsetT, CompareOpT>(d_temp_storage,
                                                                              temp_storage_bytes,
                                                                              d_input_keys,
                                                                              d_output_keys,
                                                                              num_items,
                                                                              compare_op,
                                                                              stream);
  }
};

CUB_NAMESPACE_END
