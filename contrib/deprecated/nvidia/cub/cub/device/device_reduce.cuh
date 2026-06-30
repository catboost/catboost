/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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
 * @file cub::DeviceReduce provides device-wide, parallel operations for 
 *       computing a reduction across a sequence of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>
#include <limits>

#include <cub/config.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceReduce provides device-wide, parallel operations for computing 
 *        a reduction across a sequence of data items residing within 
 *        device-accessible memory. ![](reduce_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)">*reduction*</a> 
 * (or *fold*) uses a binary combining operator to compute a single aggregate 
 * from a sequence of input elements.
 *
 * @par Usage Considerations
 * @cdp_class{DeviceReduce}
 *
 * @par Performance
 * @linear_performance{reduction, reduce-by-key, and run-length encode}
 *
 * @par
 * The following chart illustrates DeviceReduce::Sum
 * performance across different CUDA architectures for \p int32 keys.
 *
 * @image html reduce_int32.png
 *
 * @par
 * The following chart illustrates DeviceReduce::ReduceByKey (summation)
 * performance across different CUDA architectures for `fp32` values. Segments 
 * are identified by `int32` keys, and have lengths uniformly sampled 
 * from `[1, 1000]`.
 *
 * @image html reduce_by_key_fp32_len_500.png
 *
 * @par
 * @plots_below
 *
 */
struct DeviceReduce
{
  /**
   * @brief Computes a device-wide reduction using the specified binary 
   *        `reduction_op` functor and initial value `init`.
   *
   * @par
   * - Does not support binary reduction operators that are non-commutative.
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates a user-defined min-reduction of a 
   * device vector of `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // CustomMin functor
   * struct CustomMin
   * {
   *     template <typename T>
   *     __device__ __forceinline__
   *     T operator()(const T &a, const T &b) const {
   *         return (b < a) ? b : a;
   *     }
   * };
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input and output
   * int          num_items;  // e.g., 7
   * int          *d_in;      // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int          *d_out;     // e.g., [-]
   * CustomMin    min_op;
   * int          init;       // e.g., INT_MAX
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::Reduce(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, num_items, min_op, init);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run reduction
   * cub::DeviceReduce::Reduce(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, num_items, min_op, init);
   *
   * // d_out <-- [0]
   * @endcode
   *
   * @tparam InputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT      
   *   **[inferred]** Output iterator type for recording the reduced 
   *   aggregate \iterator
   *
   * @tparam ReductionOpT         
   *   **[inferred]** Binary reduction functor type having member 
   *   `T operator()(const T &a, const T &b)`
   *
   * @tparam T                    
   *   **[inferred]** Data element type that is convertible to the `value` type 
   *   of `InputIteratorT`
   *
   * @tparam NumItemsT **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in[in] 
   *   Pointer to the input sequence of data items
   *
   * @param d_out[out] 
   *   Pointer to the output aggregate
   *
   * @param num_items[in] 
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param reduction_op[in] 
   *   Binary reduction functor
   *
   * @param[in] init  
   *   Initial value of the reduction
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename T,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 NumItemsT num_items,
                                                 ReductionOpT reduction_op,
                                                 T init,
                                                 cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    return DispatchReduce<InputIteratorT,
                          OutputIteratorT,
                          OffsetT,
                          ReductionOpT,
                          T>::Dispatch(d_temp_storage,
                                       temp_storage_bytes,
                                       d_in,
                                       d_out,
                                       static_cast<OffsetT>(num_items),
                                       reduction_op,
                                       init,
                                       stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename T>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 int num_items,
                                                 ReductionOpT reduction_op,
                                                 T init,
                                                 cudaStream_t stream,
                                                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Reduce<InputIteratorT, OutputIteratorT, ReductionOpT, T>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      reduction_op,
      init,
      stream);
  }

  /**
   * @brief Computes a device-wide sum using the addition (`+`) operator.
   *
   * @par
   * - Uses `0` as the initial value of the reduction.
   * - Does not support \p + operators that are non-commutative..
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated sum-reduction performance across 
   * different CUDA architectures for `int32` and `int64` items, respectively.
   *
   * @image html reduce_int32.png
   * @image html reduce_int64.png
   *
   * @par Snippet
   * The code snippet below illustrates the sum-reduction of a device vector 
   * of `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int  num_items;      // e.g., 7
   * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_out;         // e.g., [-]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::Sum(
   *   d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sum-reduction
   * cub::DeviceReduce::Sum(
   *   d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
   *
   * // d_out <-- [38]
   * @endcode
   *
   * @tparam InputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT    
   *   **[inferred]** Output iterator type for recording the reduced 
   *   aggregate \iterator
   *
   * @tparam NumItemsT **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output aggregate
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT, 
            typename OutputIteratorT, 
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Sum(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              NumItemsT num_items,
                                              cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // The output value type
    using OutputT =
      cub::detail::non_void_value_t<OutputIteratorT,
                                    cub::detail::value_t<InputIteratorT>>;

    using InitT = OutputT; 

    return DispatchReduce<InputIteratorT, 
                          OutputIteratorT,  
                          OffsetT, 
                          cub::Sum, 
                          InitT>::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_in,
                                           d_out,
                                           static_cast<OffsetT>(num_items),
                                           cub::Sum(),
                                           InitT{}, // zero-initialize
                                           stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t Sum(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              int num_items,
                                              cudaStream_t stream,
                                              bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Sum<InputIteratorT, OutputIteratorT>(d_temp_storage,
                                                temp_storage_bytes,
                                                d_in,
                                                d_out,
                                                num_items,
                                                stream);
  }

  /**
   * @brief Computes a device-wide minimum using the less-than ('<') operator.
   *
   * @par
   * - Uses `std::numeric_limits<T>::max()` as the initial value of the reduction.
   * - Does not support `<` operators that are non-commutative.
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the min-reduction of a device vector of 
   * `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int  num_items;      // e.g., 7
   * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_out;         // e.g., [-]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::Min(
   *   d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run min-reduction
   * cub::DeviceReduce::Min(
   *   d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
   *
   * // d_out <-- [0]
   * @endcode
   *
   * @tparam InputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT    
   *   **[inferred]** Output iterator type for recording the reduced 
   *   aggregate \iterator
   *
   * @tparam NumItemsT **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output aggregate
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Min(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              NumItemsT num_items,
                                              cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    using InitT = InputT;

    return DispatchReduce<InputIteratorT,   
                          OutputIteratorT,  
                          OffsetT, 
                          cub::Min,
                          InitT>::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_in,
                                           d_out,
                                           static_cast<OffsetT>(num_items),
                                           cub::Min(),
                                           // replace with 
                                           // std::numeric_limits<T>::max() when
                                           // C++11 support is more prevalent
                                           Traits<InitT>::Max(), 
                                           stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t Min(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              int num_items,
                                              cudaStream_t stream,
                                              bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Min<InputIteratorT, OutputIteratorT>(d_temp_storage,
                                                temp_storage_bytes,
                                                d_in,
                                                d_out,
                                                num_items,
                                                stream);
  }

  /**
   * @brief Finds the first device-wide minimum using the less-than ('<') 
   *        operator, also returning the index of that item.
   *
   * @par
   * - The output value type of `d_out` is cub::KeyValuePair `<int, T>` 
   *   (assuming the value type of `d_in` is `T`)
   *   - The minimum is written to `d_out.value` and its offset in the input 
   *     array is written to `d_out.key`.
   *   - The `{1, std::numeric_limits<T>::max()}` tuple is produced for 
   *     zero-length inputs
   * - Does not support `<` operators that are non-commutative.
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the argmin-reduction of a device vector 
   * of `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int                      num_items;      // e.g., 7
   * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
   * KeyValuePair<int, int>   *d_out;         // e.g., [{-,-}]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::ArgMin(
   *   d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run argmin-reduction
   * cub::DeviceReduce::ArgMin(
   *   d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
   *
   * // d_out <-- [{5, 0}]
   *
   * @endcode
   *
   * @tparam InputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input items 
   *   (of some type `T`) \iterator
   *
   * @tparam OutputIteratorT    
   *   **[inferred]** Output iterator type for recording the reduced aggregate 
   *   (having value type `cub::KeyValuePair<int, T>`) \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to \p temp_storage_bytes and no work is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output aggregate
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT, 
            typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 int num_items,
                                                 cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT =
      cub::detail::non_void_value_t<OutputIteratorT,
                                    KeyValuePair<OffsetT, InputValueT>>;

    using InitT = OutputTupleT;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT =
      ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value

    // replace with std::numeric_limits<T>::max() when C++11 support is
    // more prevalent
    InitT initial_value(1, Traits<InputValueT>::Max()); 

    return DispatchReduce<ArgIndexInputIteratorT,
                          OutputIteratorT,
                          OffsetT,
                          cub::ArgMin,
                          InitT>::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_indexed_in,
                                           d_out,
                                           num_items,
                                           cub::ArgMin(),
                                           initial_value,
                                           stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 int num_items,
                                                 cudaStream_t stream,
                                                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMin<InputIteratorT, OutputIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_items,
                                                   stream);
  }

  /**
   * @brief Computes a device-wide maximum using the greater-than ('>') operator.
   *
   * @par
   * - Uses `std::numeric_limits<T>::lowest()` as the initial value of the 
   *   reduction.
   * - Does not support `>` operators that are non-commutative.
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the max-reduction of a device vector of 
   * `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int  num_items;      // e.g., 7
   * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_out;         // e.g., [-]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::Max(
   *   d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run max-reduction
   * cub::DeviceReduce::Max(
   *   d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
   *
   * // d_out <-- [9]
   * @endcode
   *
   * @tparam InputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT    
   *   **[inferred]** Output iterator type for recording the reduced 
   *   aggregate \iterator
   *
   * @tparam NumItemsT **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output aggregate
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within. 
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT, 
            typename OutputIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Max(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              NumItemsT num_items,
                                              cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    using InitT = InputT;

    return DispatchReduce<InputIteratorT,   
                          OutputIteratorT,  
                          OffsetT, 
                          cub::Max,
                          InitT>::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_in,
                                           d_out,
                                           static_cast<OffsetT>(num_items),
                                           cub::Max(),
                                           // replace with 
                                           // std::numeric_limits<T>::lowest()
                                           // when C++11 support is more 
                                           // prevalent
                                           Traits<InitT>::Lowest(), 
                                           stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t Max(void *d_temp_storage,
                                              size_t &temp_storage_bytes,
                                              InputIteratorT d_in,
                                              OutputIteratorT d_out,
                                              int num_items,
                                              cudaStream_t stream,
                                              bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Max<InputIteratorT, OutputIteratorT>(d_temp_storage,
                                                temp_storage_bytes,
                                                d_in,
                                                d_out,
                                                num_items,
                                                stream);
  }

  /**
   * @brief Finds the first device-wide maximum using the greater-than ('>') 
   *        operator, also returning the index of that item
   *
   * @par
   * - The output value type of `d_out` is cub::KeyValuePair `<int, T>` 
   *   (assuming the value type of `d_in` is `T`)
   *   - The maximum is written to `d_out.value` and its offset in the input 
   *     array is written to `d_out.key`.
   *   - The `{1, std::numeric_limits<T>::lowest()}` tuple is produced for 
   *     zero-length inputs
   * - Does not support `>` operators that are non-commutative.
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - The range `[d_in, d_in + num_items)` shall not overlap `d_out`.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the argmax-reduction of a device vector 
   * of `int` data elements.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_reduce.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int                      num_items;      // e.g., 7
   * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
   * KeyValuePair<int, int>   *d_out;         // e.g., [{-,-}]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::ArgMax(
   *   d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run argmax-reduction
   * cub::DeviceReduce::ArgMax(
   *   d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
   *
   * // d_out <-- [{6, 9}]
   *
   * @endcode
   *
   * @tparam InputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input items 
   *   (of some type \p T) \iterator
   *
   * @tparam OutputIteratorT    
   *   **[inferred]** Output iterator type for recording the reduced aggregate 
   *   (having value type `cub::KeyValuePair<int, T>`) \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output aggregate
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT, 
            typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 int num_items,
                                                 cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT =
      cub::detail::non_void_value_t<OutputIteratorT,
                                    KeyValuePair<OffsetT, InputValueT>>;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    using InitT = OutputTupleT;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT =
      ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value

    // replace with std::numeric_limits<T>::lowest() when C++11 support is
    // more prevalent
    InitT initial_value(1, Traits<InputValueT>::Lowest()); 

    return DispatchReduce<ArgIndexInputIteratorT,
                          OutputIteratorT,
                          OffsetT,
                          cub::ArgMax,
                          InitT>::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_indexed_in,
                                           d_out,
                                           num_items,
                                           cub::ArgMax(),
                                           initial_value,
                                           stream);
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(void *d_temp_storage,
                                                 size_t &temp_storage_bytes,
                                                 InputIteratorT d_in,
                                                 OutputIteratorT d_out,
                                                 int num_items,
                                                 cudaStream_t stream,
                                                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMax<InputIteratorT, OutputIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_items,
                                                   stream);
  }

  /**
   * @brief Reduces segments of values, where segments are demarcated by 
   *        corresponding runs of identical keys.
   *
   * @par
   * This operation computes segmented reductions within `d_values_in` using
   * the specified binary `reduction_op` functor. The segments are identified 
   * by "runs" of corresponding keys in `d_keys_in`, where runs are maximal 
   * ranges of consecutive, identical keys. For the *i*<sup>th</sup> run 
   * encountered, the first key of the run and the corresponding value 
   * aggregate of that run are written to `d_unique_out[i] and 
   * `d_aggregates_out[i]`, respectively. The total number of runs encountered 
   * is written to `d_num_runs_out`.
   *
   * @par
   * - The `==` equality operator is used to determine whether keys are 
   *   equivalent
   * - Provides "run-to-run" determinism for pseudo-associative reduction
   *   (e.g., addition of floating point types) on the same GPU device.
   *   However, results for pseudo-associative reduction may be inconsistent
   *   from one device to a another device of a different compute-capability
   *   because CUB can employ different tile-sizing for different architectures.
   * - Let `out` be any of 
   *   `[d_unique_out, d_unique_out + *d_num_runs_out)`
   *   `[d_aggregates_out, d_aggregates_out + *d_num_runs_out)`
   *   `d_num_runs_out`. The ranges represented by `out` shall not overlap 
   *   `[d_keys_in, d_keys_in + num_items)`,
   *   `[d_values_in, d_values_in + num_items)` nor `out` in any way.
   * - @devicestorage
   *
   * @par Performance
   * The following chart illustrates reduction-by-key (sum) performance across
   * different CUDA architectures for `fp32` and `fp64` values, respectively.  
   * Segments are identified by `int32` keys, and have lengths uniformly 
   * sampled from `[1, 1000]`.
   *
   * @image html reduce_by_key_fp32_len_500.png
   * @image html reduce_by_key_fp64_len_500.png
   *
   * @par
   * The following charts are similar, but with segment lengths uniformly 
   * sampled from [1,10]:
   *
   * @image html reduce_by_key_fp32_len_5.png
   * @image html reduce_by_key_fp64_len_5.png
   *
   * @par Snippet
   * The code snippet below illustrates the segmented reduction of `int` values 
   * grouped by runs of associated `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_reduce.cuh>
   *
   * // CustomMin functor
   * struct CustomMin
   * {
   *     template <typename T>
   *     CUB_RUNTIME_FUNCTION __forceinline__
   *     T operator()(const T &a, const T &b) const {
   *         return (b < a) ? b : a;
   *     }
   * };
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int          num_items;          // e.g., 8
   * int          *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
   * int          *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
   * int          *d_unique_out;      // e.g., [-, -, -, -, -, -, -, -]
   * int          *d_aggregates_out;  // e.g., [-, -, -, -, -, -, -, -]
   * int          *d_num_runs_out;    // e.g., [-]
   * CustomMin    reduction_op;
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceReduce::ReduceByKey(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_keys_in, d_unique_out, d_values_in, 
   *   d_aggregates_out, d_num_runs_out, reduction_op, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run reduce-by-key
   * cub::DeviceReduce::ReduceByKey(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_keys_in, d_unique_out, d_values_in, 
   *   d_aggregates_out, d_num_runs_out, reduction_op, num_items);
   *
   * // d_unique_out      <-- [0, 2, 9, 5, 8]
   * // d_aggregates_out  <-- [0, 1, 6, 2, 4]
   * // d_num_runs_out    <-- [5]
   * @endcode
   *
   * @tparam KeysInputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   keys \iterator
   *
   * @tparam UniqueOutputIteratorT    
   *   **[inferred]** Random-access output iterator type for writing unique 
   *   output keys \iterator
   *
   * @tparam ValuesInputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input 
   *   values \iterator
   *
   * @tparam AggregatesOutputIterator 
   *   **[inferred]** Random-access output iterator type for writing output 
   *   value aggregates \iterator
   *
   * @tparam NumRunsOutputIteratorT   
   *   **[inferred]** Output iterator type for recording the number of runs 
   *   encountered \iterator
   *
   * @tparam ReductionOpT              
   *   **[inferred]*8 Binary reduction functor type having member 
   *   `T operator()(const T &a, const T &b)`
   *
   * @tparam NumItemsT **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in  
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out  
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[in] d_values_in  
   *   Pointer to the input sequence of corresponding values
   *
   * @param[out] d_aggregates_out  
   *   Pointer to the output sequence of value aggregates 
   *   (one aggregate per run)
   *
   * @param[out] d_num_runs_out  
   *   Pointer to total number of runs encountered 
   *   (i.e., the length of `d_unique_out`)
   *
   * @param[in] reduction_op  
   *   Binary reduction functor
   *
   * @param[in] num_items  
   *   Total number of associated key+value pairs 
   *   (i.e., the length of `d_in_keys` and `d_in_values`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename ReductionOpT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  ReduceByKey(void *d_temp_storage,
              size_t &temp_storage_bytes,
              KeysInputIteratorT d_keys_in,
              UniqueOutputIteratorT d_unique_out,
              ValuesInputIteratorT d_values_in,
              AggregatesOutputIteratorT d_aggregates_out,
              NumRunsOutputIteratorT d_num_runs_out,
              ReductionOpT reduction_op,
              NumItemsT num_items,
              cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // FlagT iterator type (not used)

    // Selection op (not used)

    // Default == operator
    typedef Equality EqualityOp;

    return DispatchReduceByKey<KeysInputIteratorT,
                               UniqueOutputIteratorT,
                               ValuesInputIteratorT,
                               AggregatesOutputIteratorT,
                               NumRunsOutputIteratorT,
                               EqualityOp,
                               ReductionOpT,
                               OffsetT>::Dispatch(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_unique_out,
                                                  d_values_in,
                                                  d_aggregates_out,
                                                  d_num_runs_out,
                                                  EqualityOp(),
                                                  reduction_op,
                                                  static_cast<OffsetT>(num_items),
                                                  stream);
  }

  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename ReductionOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  ReduceByKey(void *d_temp_storage,
              size_t &temp_storage_bytes,
              KeysInputIteratorT d_keys_in,
              UniqueOutputIteratorT d_unique_out,
              ValuesInputIteratorT d_values_in,
              AggregatesOutputIteratorT d_aggregates_out,
              NumRunsOutputIteratorT d_num_runs_out,
              ReductionOpT reduction_op,
              int num_items,
              cudaStream_t stream,
              bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ReduceByKey<KeysInputIteratorT,
                       UniqueOutputIteratorT,
                       ValuesInputIteratorT,
                       AggregatesOutputIteratorT,
                       NumRunsOutputIteratorT,
                       ReductionOpT>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys_in,
                                     d_unique_out,
                                     d_values_in,
                                     d_aggregates_out,
                                     d_num_runs_out,
                                     reduction_op,
                                     num_items,
                                     stream);
  }
};

/**
 * @example example_device_reduce.cu
 */

CUB_NAMESPACE_END
