/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

//! @file
//! @rst
//! The ``cub::WarpScan`` class provides :ref:`collective <collective-primitives>` methods for
//! computing a parallel prefix scan of items partitioned across a CUDA thread warp.
//! @endrst

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/specializations/warp_scan_shfl.cuh>
#include <cub/warp/specializations/warp_scan_smem.cuh>

#include <cuda/ptx>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//! @rst
//! The WarpScan class provides :ref:`collective <collective-primitives>` methods for computing a
//! parallel prefix scan of items partitioned across a CUDA thread warp.
//!
//! .. image:: ../../img/warp_scan_logo.png
//!     :align: center
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! * Given a list of input elements and a binary reduction operator, a
//!   `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`__ produces an output list where each
//!   element is computed to be the reduction of the elements occurring earlier in the input list.
//!   *Prefix sum* connotes a prefix scan with the addition operator. The term *inclusive*
//!   indicates that the *i*\ :sup:`th` output reduction incorporates the *i*\ :sup:`th` input.
//!   The term *exclusive* indicates the *i*\ :sup:`th` input is not incorporated into
//!   the *i*\ :sup:`th` output reduction.
//! * Supports non-commutative scan operators
//! * Supports "logical" warps smaller than the physical warp size
//!   (e.g., a logical warp of 8 threads)
//! * The number of entrant threads must be an multiple of ``LOGICAL_WARP_THREADS``
//!
//! Performance Considerations
//! ++++++++++++++++++++++++++
//!
//! * Uses special instructions when applicable (e.g., warp ``SHFL``)
//! * Uses synchronization-free communication between warp lanes when applicable
//! * Incurs zero bank conflicts for most types
//! * Computation is slightly more efficient (i.e., having lower instruction overhead) for:
//!
//!   * Summation (**vs.** generic scan)
//!   * The architecture's warp size is a whole multiple of ``LOGICAL_WARP_THREADS``
//!
//! Simple Examples
//! ++++++++++++++++++++++++++
//!
//! @warpcollective{WarpScan}
//!
//! The code snippet below illustrates four concurrent warp prefix sums within a block of
//! 128 threads (one per each of the 32-thread warps).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize WarpScan for type int
//!        using WarpScan = cub::WarpScan<int>;
//!
//!        // Allocate WarpScan shared memory for 4 warps
//!        __shared__ typename WarpScan::TempStorage temp_storage[4];
//!
//!        // Obtain one input item per thread
//!        int thread_data = ...
//!
//!        // Compute warp-wide prefix sums
//!        int warp_id = threadIdx.x / 32;
//!        WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data, thread_data);
//!
//! Suppose the set of input ``thread_data`` across the block of threads is
//! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` in each of the four warps of
//! threads will be ``0, 1, 2, 3, ..., 31}``.
//!
//! The code snippet below illustrates a single warp prefix sum within a block of
//! 128 threads.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize WarpScan for type int
//!        using WarpScan = cub::WarpScan<int>;
//!
//!        // Allocate WarpScan shared memory for one warp
//!        __shared__ typename WarpScan::TempStorage temp_storage;
//!        ...
//!
//!        // Only the first warp performs a prefix sum
//!        if (threadIdx.x < 32)
//!        {
//!            // Obtain one input item per thread
//!            int thread_data = ...
//!
//!            // Compute warp-wide prefix sums
//!            WarpScan(temp_storage).ExclusiveSum(thread_data, thread_data);
//!
//! Suppose the set of input ``thread_data`` across the warp of threads is
//! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` will be
//! ``{0, 1, 2, 3, ..., 31}``.
//! @endrst
//!
//! @tparam T
//!   The scan input/output element type
//!
//! @tparam LOGICAL_WARP_THREADS
//!   **[optional]** The number of threads per "logical" warp (may be less than the number of
//!   hardware warp threads). Default is the warp size associated with the CUDA Compute Capability
//!   targeted by the compiler (e.g., 32 threads for SM20).
//!
template <typename T, int LOGICAL_WARP_THREADS = detail::warp_threads>
class WarpScan
{
private:
  /******************************************************************************
   * Constants and type definitions
   ******************************************************************************/

  enum
  {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (LOGICAL_WARP_THREADS == detail::warp_threads),

    /// Whether the logical warp size is a power-of-two
    IS_POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),

    /// Whether the data type is an integer (which has fully-associative addition)
    IS_INTEGER = cuda::std::is_integral_v<T>
  };

  /// Internal specialization.
  /// Use SHFL-based scan if LOGICAL_WARP_THREADS is a power-of-two
  using InternalWarpScan = ::cuda::std::
    _If<IS_POW_OF_TWO, detail::WarpScanShfl<T, LOGICAL_WARP_THREADS>, detail::WarpScanSmem<T, LOGICAL_WARP_THREADS>>;

  /// Shared memory storage layout type for WarpScan
  using _TempStorage = typename InternalWarpScan::TempStorage;

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  /// Shared storage reference
  _TempStorage& temp_storage;
  unsigned int lane_id;

  /******************************************************************************
   * Public types
   ******************************************************************************/

public:
  /// @smemstorage{WarpScan}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using the specified memory allocation as temporary storage.
  //!        Logical warp and lane identifiers are constructed from `threadIdx.x`.
  //!
  //! @param[in] temp_storage
  //!   Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpScan(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : ::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS)
  {}

  //! @}  end member group
  //! @name Inclusive prefix sums
  //! @{

  //! @rst
  //! Computes an inclusive prefix sum across the calling warp.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix sums within a
  //! block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute inclusive warp-wide prefix sums
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).InclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` in each of the four warps
  //! of threads will be ``1, 2, 3, ..., 32}``.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with `input`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T input, T& inclusive_output)
  {
    InclusiveScan(input, inclusive_output, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes an inclusive prefix sum across the calling warp.
  //! Also provides every thread with the warp-wide ``warp_aggregate`` of all inputs.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix sums within a
  //! block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute inclusive warp-wide prefix sums
  //!        int warp_aggregate;
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).InclusiveSum(thread_data,
  //!                                                     thread_data,
  //!                                                     warp_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` in each of the four warps
  //! of threads will be ``1, 2, 3, ..., 32}``. Furthermore, ``warp_aggregate`` for all threads
  //! in all warps will be ``32``.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T input, T& inclusive_output, T& warp_aggregate)
  {
    InclusiveScan(input, inclusive_output, ::cuda::std::plus<>{}, warp_aggregate);
  }

  //! @}  end member group
  //! @name Exclusive prefix sums
  //! @{

  //! @rst
  //! Computes an exclusive prefix sum across the calling warp. The value of 0 is applied as the
  //! initial value, and is assigned to ``exclusive_output`` in *lane*\ :sub:`0`.
  //!
  //! * @identityzero
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix sums within a
  //! block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix sums
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` in each of the four warps
  //! of threads will be ``0, 1, 2, ..., 31}``.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item. May be aliased with `input`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T input, T& exclusive_output)
  {
    T initial_value{};
    ExclusiveScan(input, exclusive_output, initial_value, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes an exclusive prefix sum across the calling warp. The value of 0 is applied as the
  //! initial value, and is assigned to ``exclusive_output`` in *lane*\ :sub:`0`.
  //! Also provides every thread with the warp-wide ``warp_aggregate`` of all inputs.
  //!
  //! * @identityzero
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix sums within a
  //! block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix sums
  //!        int warp_aggregate;
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data,
  //!                                                     thread_data,
  //!                                                     warp_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{1, 1, 1, 1, ...}``. The corresponding output ``thread_data`` in each of the four warps
  //! of threads will be ``0, 1, 2, ..., 31}``. Furthermore, ``warp_aggregate`` for all threads
  //! in all warps will be ``32``.
  //! @endrst
  //!
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T input, T& exclusive_output, T& warp_aggregate)
  {
    T initial_value{};
    ExclusiveScan(input, exclusive_output, initial_value, ::cuda::std::plus<>{}, warp_aggregate);
  }

  //! @}  end member group
  //! @name Inclusive prefix scans
  //! @{

  //! @rst
  //! Computes an inclusive prefix scan using the specified binary scan functor across the
  //! calling warp.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute inclusive warp-wide prefix max scans
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).InclusiveScan(thread_data, thread_data, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``0, 0, 2, 2, ..., 30, 30``, the output for the second warp would be
  //! ``32, 32, 34, 34, ..., 62, 62``, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
  {
    InternalWarpScan(temp_storage).InclusiveScan(input, inclusive_output, scan_op);
  }

  //! @rst
  //! Computes an inclusive prefix scan using the specified binary scan functor across the
  //! calling warp.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix sum scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-warp-scan-init-value
  //!     :end-before: example-end inclusive-warp-scan-init-value
  //!
  //! Suppose the set of input ``thread_data`` in the first warp is
  //! ``{0, 1, 2, 3, ..., 31}``, in the second warp is ``{1, 2, 3, 4, ..., 32}`` etc.
  //! The corresponding output ``thread_data`` for a max operation in the first
  //! warp would be ``{3, 3, 3, 3, ..., 31}``, the output for the second warp would be
  //! ``{3, 3, 3, 4, ..., 32}``, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the inclusive scan (uniform across warp)
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, T initial_value, ScanOp scan_op)
  {
    InternalWarpScan internal(temp_storage);

    T exclusive_output;
    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(
      input, inclusive_output, exclusive_output, scan_op, initial_value, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @rst
  //! Computes an inclusive prefix scan using the specified binary scan functor across the
  //! calling warp. Also provides every thread with the warp-wide ``warp_aggregate`` of
  //! all inputs.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute inclusive warp-wide prefix max scans
  //!        int warp_aggregate;
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).InclusiveScan(
  //!            thread_data, thread_data, cuda::maximum<>{}, warp_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``0, 0, 2, 2, ..., 30, 30``, the output for the second warp would be
  //! ``32, 32, 34, 34, ..., 62, 62``, etc.  Furthermore, ``warp_aggregate`` would be assigned
  //! ``30`` for threads in the first warp, ``62`` for threads in the second warp, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with ``input``
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& warp_aggregate)
  {
    InternalWarpScan(temp_storage).InclusiveScan(input, inclusive_output, scan_op, warp_aggregate);
  }

  //! @rst
  //! Computes an inclusive prefix scan using the specified binary scan functor across the
  //! calling warp. Also provides every thread with the warp-wide ``warp_aggregate`` of
  //! all inputs.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide inclusive prefix max scans
  //! within a block of 128 threads (one scan per warp).
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-warp-scan-init-value-aggregate
  //!     :end-before: example-end inclusive-warp-scan-init-value-aggregate
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{1, 1, 1, 1, ..., 1}``. For initial value equal to 3, the corresponding output
  //! ``thread_data`` for a sum operation in the first warp would be
  //! ``{4, 5, 6, 7, ..., 35}``, the output for the second warp would be
  //! ``{4, 5, 6, 7, ..., 35}``, etc.  Furthermore,  ``warp_aggregate`` would be assigned
  //! ``32`` for threads in each warp.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's output item. May be aliased with ``input``
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the inclusive scan (uniform across warp). It is not taken
  //!   into account for warp_aggregate.
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& inclusive_output, T initial_value, ScanOp scan_op, T& warp_aggregate)
  {
    InternalWarpScan internal(temp_storage);

    // Perform the inclusive scan operation
    internal.InclusiveScan(input, inclusive_output, scan_op);

    // Update the inclusive_output and warp_aggregate using the Update function
    T exclusive_output;
    internal.Update(
      input,
      inclusive_output,
      exclusive_output,
      warp_aggregate,
      scan_op,
      initial_value,
      detail::bool_constant_v<IS_INTEGER>);
  }

  //! @}  end member group
  //! @name Exclusive prefix scans
  //! @{

  //! @rst
  //! Computes an exclusive prefix scan using the specified binary scan functor across the
  //! calling warp. Because no initial value is supplied, the ``output`` computed for
  //! *lane*\ :sub:`0` is undefined.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix max scans
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data, thread_data, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``?, 0, 0, 2, ..., 28, 30``, the output for the second warp would be
  //! ``?, 32, 32, 34, ..., 60, 62``, etc.
  //! (The output ``thread_data`` in warp *lane*\ :sub:`0` is undefined.)
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op)
  {
    InternalWarpScan internal(temp_storage);

    T inclusive_output;
    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(input, inclusive_output, exclusive_output, scan_op, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @rst
  //! Computes an exclusive prefix scan using the specified binary scan functor across the
  //! calling warp.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix max scans
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data,
  //!                                                      thread_data,
  //!                                                      INT_MIN,
  //!                                                      cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``INT_MIN, 0, 0, 2, ..., 28, 30``, the output for the second warp would be
  //! ``30, 32, 32, 34, ..., 60, 62``, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the exclusive scan
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op)
  {
    InternalWarpScan internal(temp_storage);

    T inclusive_output;
    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(
      input, inclusive_output, exclusive_output, scan_op, initial_value, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @rst
  //! Computes an exclusive prefix scan using the specified binary scan functor across the
  //! calling warp. Because no initial value is supplied, the ``output`` computed for
  //! *lane*\ :sub:`0` is undefined. Also provides every thread with the warp-wide
  //! ``warp_aggregate`` of all inputs.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix max scans
  //!        int warp_aggregate;
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data,
  //!                                                      thread_data,
  //!                                                      cuda::maximum<>{},
  //!                                                      warp_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``?, 0, 0, 2, ..., 28, 30``, the output for the second warp would be
  //! ``?, 32, 32, 34, ..., 60, 62``, etc. (The output ``thread_data`` in warp *lane*\ :sub:`0`
  //! is undefined). Furthermore, ``warp_aggregate`` would be assigned ``30`` for threads in the
  //! first warp, \p 62 for threads in the second warp, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item. May be aliased with `input`
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op, T& warp_aggregate)
  {
    InternalWarpScan internal(temp_storage);

    T inclusive_output;
    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(
      input, inclusive_output, exclusive_output, warp_aggregate, scan_op, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @rst
  //! Computes an exclusive prefix scan using the specified binary scan functor across the
  //! calling warp. Also provides every thread with the warp-wide ``warp_aggregate`` of
  //! all inputs.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix max scans
  //!        int warp_aggregate;
  //!        int warp_id = threadIdx.x / 32;
  //!        WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data,
  //!                                                      thread_data,
  //!                                                      INT_MIN,
  //!                                                      cuda::maximum<>{},
  //!                                                      warp_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``thread_data`` in the first
  //! warp would be ``INT_MIN, 0, 0, 2, ..., 28, 30``, the output for the second warp would be
  //! ``30, 32, 32, 34, ..., 60, 62``, etc. Furthermore, ``warp_aggregate`` would be assigned
  //! ``30`` for threads in the first warp, ``62`` for threads in the second warp, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's output item.  May be aliased with `input`
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the exclusive scan
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  //!
  //! @param[out] warp_aggregate
  //!   Warp-wide aggregate reduction of input items
  //!
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& exclusive_output, T initial_value, ScanOp scan_op, T& warp_aggregate)
  {
    InternalWarpScan internal(temp_storage);

    T inclusive_output;
    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(
      input,
      inclusive_output,
      exclusive_output,
      warp_aggregate,
      scan_op,
      initial_value,
      detail::bool_constant_v<IS_INTEGER>);
  }

  //! @}  end member group
  //! @name Combination (inclusive & exclusive) prefix scans
  //! @{

  //! @rst
  //! Computes both inclusive and exclusive prefix scans using the specified binary scan functor
  //! across the calling warp. Because no initial value is supplied, the ``exclusive_output``
  //! computed for *lane*\ :sub:`0` is undefined.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans
  //! within a block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute exclusive warp-wide prefix max scans
  //!        int inclusive_partial, exclusive_partial;
  //!        WarpScan(temp_storage[warp_id]).Scan(thread_data,
  //!                                             inclusive_partial,
  //!                                             exclusive_partial,
  //!                                             cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``inclusive_partial`` in the
  //! first warp would be ``0, 0, 2, 2, ..., 30, 30``, the output for the second warp would be
  //! ``32, 32, 34, 34, ..., 62, 62``, etc. The corresponding output ``exclusive_partial`` in the
  //! first warp would be ``?, 0, 0, 2, ..., 28, 30``, the output for the second warp would be
  //! ``?, 32, 32, 34, ..., 60, 62``, etc.
  //! (The output ``thread_data`` in warp *lane*\ :sub:`0` is undefined.)
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's inclusive-scan output item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's exclusive-scan output item
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scan(T input, T& inclusive_output, T& exclusive_output, ScanOp scan_op)
  {
    InternalWarpScan internal(temp_storage);

    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(input, inclusive_output, exclusive_output, scan_op, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @rst
  //! Computes both inclusive and exclusive prefix scans using the specified binary scan functor
  //! across the calling warp.
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp-wide prefix max scans within a
  //! block of 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Compute inclusive warp-wide prefix max scans
  //!        int warp_id = threadIdx.x / 32;
  //!        int inclusive_partial, exclusive_partial;
  //!        WarpScan(temp_storage[warp_id]).Scan(thread_data,
  //!                                             inclusive_partial,
  //!                                             exclusive_partial,
  //!                                             INT_MIN,
  //!                                             cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, -1, 2, -3, ..., 126, -127}``. The corresponding output ``inclusive_partial`` in the
  //! first warp would be ``0, 0, 2, 2, ..., 30, 30``, the output for the second warp would be
  //! ``32, 32, 34, 34, ..., 62, 62``, etc. The corresponding output ``exclusive_partial`` in the
  //! first warp would be ``INT_MIN, 0, 0, 2, ..., 28, 30``, the output for the second warp would
  //! be ``30, 32, 32, 34, ..., 60, 62``, etc.
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] inclusive_output
  //!   Calling thread's inclusive-scan output item
  //!
  //! @param[out] exclusive_output
  //!   Calling thread's exclusive-scan output item
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the exclusive scan
  //!
  //! @param[in] scan_op
  //!   Binary scan operator
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Scan(T input, T& inclusive_output, T& exclusive_output, T initial_value, ScanOp scan_op)
  {
    InternalWarpScan internal(temp_storage);

    internal.InclusiveScan(input, inclusive_output, scan_op);

    internal.Update(
      input, inclusive_output, exclusive_output, scan_op, initial_value, detail::bool_constant_v<IS_INTEGER>);
  }

  //! @}  end member group
  //! @name Data exchange
  //! @{

  //! @rst
  //! Broadcast the value ``input`` from *lane*\ :sub:`src_lane` to all lanes in the warp
  //!
  //! * @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the warp-wide broadcasts of values from *lane*\ :sub:`0`
  //! in each of four warps to all other threads in those warps.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpScan for type int
  //!        using WarpScan = cub::WarpScan<int>;
  //!
  //!        // Allocate WarpScan shared memory for 4 warps
  //!        __shared__ typename WarpScan::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Broadcast from lane0 in each warp to all other threads in the warp
  //!        int warp_id = threadIdx.x / 32;
  //!        thread_data = WarpScan(temp_storage[warp_id]).Broadcast(thread_data, 0);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, 1, 2, 3, ..., 127}``. The corresponding output ``thread_data`` will be
  //! ``{0, 0, ..., 0}`` in warp\ :sub:`0`,
  //! ``{32, 32, ..., 32}`` in warp\ :sub:`1`,
  //! ``{64, 64, ..., 64}`` in warp\ :sub:`2`, etc.
  //! @endrst
  //!
  //! @param[in] input
  //!   The value to broadcast
  //!
  //! @param[in] src_lane
  //!   Which warp lane is to do the broadcasting
  _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T input, unsigned int src_lane)
  {
    return InternalWarpScan(temp_storage).Broadcast(input, src_lane);
  }

  //@}  end member group
};

CUB_NAMESPACE_END
