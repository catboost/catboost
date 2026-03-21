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

//! @file
//! cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data
//! items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/std/__functional/invoke.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceScan provides device-wide, parallel operations for computing a
//! prefix scan across a sequence of data items residing within
//! device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Given a sequence of input elements and a binary reduction operator, a
//! `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`_ produces an output
//! sequence where each element is computed to be the reduction of the elements
//! occurring earlier in the input sequence. *Prefix sum* connotes a prefix scan
//! with the addition operator. The term *inclusive* indicates that the
//! *i*\ :sup:`th` output reduction incorporates the *i*\ :sup:`th` input.
//! The term *exclusive* indicates the *i*\ :sup:`th` input is not
//! incorporated into the *i*\ :sup:`th` output reduction. When the input and
//! output sequences are the same, the scan is performed in-place.
//!
//! As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our
//! *"decoupled look-back"* algorithm for performing global prefix scan with
//! only a single pass through the input data, as described in our 2016 technical
//! report [1]_. The central idea is to leverage a small, constant factor of
//! redundant work in order to overlap the latencies of global prefix
//! propagation with local computation. As such, our algorithm requires only
//! ``~2*n*`` data movement (``n`` inputs are read, ``n`` outputs are written), and
//! typically proceeds at "memcpy" speeds. Our algorithm supports inplace operations.
//!
//! .. [1] Duane Merrill and Michael Garland. `Single-pass Parallel Prefix Scan with Decoupled Look-back
//!    <https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back>`_,
//!    *NVIDIA Technical Report NVR-2016-002*, 2016.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceScan}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{prefix scan}
//!
//! @endrst
struct DeviceScan
{
  //! @name Exclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide exclusive prefix sum.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using InitT   = cub::detail::it_value_t<InputIteratorT>;

    // Initial value
    InitT init_value{};

    return DispatchScan<InputIteratorT, OutputIteratorT, ::cuda::std::plus<>, detail::InputValue<InitT>, OffsetT>::
      Dispatch(d_temp_storage,
               temp_storage_bytes,
               d_in,
               d_out,
               ::cuda::std::plus<>{},
               detail::InputValue<InitT>(init_value),
               num_items,
               stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum in-place.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_data``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading scan inputs and wrigin scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage, size_t& temp_storage_bytes, IteratorT d_data, NumItemsT num_items, cudaStream_t stream = 0)
  {
    return ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_data``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an
  //! ``int`` device vector:
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is provided as a future value.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified binary ``scan_op`` functor.
  //! The ``init_value`` value is provided as a future value.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Pointer to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @}  end member group
  //! @name Inclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide inclusive prefix sum.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ::cuda::std::plus<>, NullType, OffsetT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, ::cuda::std::plus<>{}, NullType{}, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum in-place.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage, size_t& temp_storage_bytes, IteratorT d_data, NumItemsT num_items, cudaStream_t stream = 0)
  {
    return InclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_out <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, NullType, OffsetT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, NullType(), num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and ``*d_in``
  //! is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive max-scan of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-inclusive-scan
  //!     :end-before: example-end device-inclusive-scan
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to the size in bytes of the `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the inclusive scan (`scan_op(init_value, d_in[0])`
  //!   is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within.
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanInit");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using AccumT  = ::cuda::std::__accumulator_t<ScanOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>;

    return DispatchScan<
      InputIteratorT,
      OutputIteratorT,
      ScanOpT,
      detail::InputValue<InitValueT>,
      OffsetT,
      AccumT,
      ForceInclusive::Yes>::Dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     scan_op,
                                     detail::InputValue<InitValueT>(init_value),
                                     num_items,
                                     stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_data <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return InclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum-by-key with key equality
  //! defined by ``equality_op``. The value of ``0`` is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_values_out <-- [0, 8, 0, 7, 12, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSumByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using InitT   = cub::detail::it_value_t<ValuesInputIteratorT>;

    // Initial value
    InitT init_value{};

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ::cuda::std::plus<>,
      InitT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         ::cuda::std::plus<>{},
                         init_value,
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan-by-key using the
  //! specified binary ``scan_op`` functor. The key equality is defined by
  //! ``equality_op``.  The ``init_value`` value is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan-by-key of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // d_values_out <-- [2147483647, 8, 2147483647, 7, 5, 2147483647, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the `init_value` value used in Binary scan
  //!   functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage. When `nullptr`, the
  //!    required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary scan functor
  //!
  //!  @param[in] init_value
  //!    Initial value to seed the exclusive scan (and is assigned to the
  //!    beginning of each segment in `d_values_out`)
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and
  //!    `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScanByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ScanOpT,
      InitValueT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         scan_op,
                         init_value,
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum-by-key with key equality defined by ``equality_op``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 7, 12, 15, 0, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSumByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ::cuda::std::plus<>,
      NullType,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         ::cuda::std::plus<>{},
                         NullType{},
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan-by-key using the
  //! specified binary ``scan_op`` functor. The key equality is defined by ``equality_op``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // d_out <-- [8, 6, 7, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary scan functor
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ScanOpT,
      NullType,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         scan_op,
                         NullType(),
                         num_items,
                         stream);
  }

  //! @}  end member group
};

CUB_NAMESPACE_END
