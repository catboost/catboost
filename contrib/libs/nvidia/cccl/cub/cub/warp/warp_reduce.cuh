/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *       following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written permission.
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
//! @rst
//! The ``cub::WarpReduce`` class provides :ref:`collective <collective-primitives>` methods for
//! computing a parallel reduction of items partitioned across a CUDA thread warp.
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

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/specializations/warp_reduce_shfl.cuh>
#include <cub/warp/specializations/warp_reduce_smem.cuh>

#include <cuda/functional>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``WarpReduce`` class provides :ref:`collective <collective-primitives>` methods for computing a parallel
//! reduction of items partitioned across a CUDA thread warp.
//!
//! .. image:: ../../img/warp_reduce_logo.png
//!     :align: center
//!
//! Overview
//! ++++++++
//!
//! - A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`__ (or *fold*) uses a binary combining
//!   operator to compute a single aggregate from a list of input elements.
//! - Supports "logical" warps smaller than the physical warp size (e.g., logical warps of 8 threads)
//! - The number of entrant threads must be an multiple of ``LogicalWarpThreads``
//!
//! Performance Considerations
//! ++++++++++++++++++++++++++
//!
//! - Uses special instructions when applicable (e.g., warp ``SHFL`` instructions)
//! - Uses synchronization-free communication between warp lanes when applicable
//! - Incurs zero bank conflicts for most types
//! - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
//!
//!   - Summation (**vs.** generic reduction)
//!   - The architecture's warp size is a whole multiple of ``LogicalWarpThreads``
//!
//! Simple Examples
//! +++++++++++++++
//!
//! @warpcollective{WarpReduce}
//!
//! The code snippet below illustrates four concurrent warp sum reductions within a block of 128 threads (one per each
//! of the 32-thread warps).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize WarpReduce for type int
//!        using WarpReduce = cub::WarpReduce<int>;
//!        // Allocate WarpReduce shared memory for 4 warps
//!        __shared__ typename WarpReduce::TempStorage temp_storage[4];
//!        // Obtain one input item per thread
//!        int thread_data = ...
//!        // Return the warp-wide sums to each lane0 (threads 0, 32, 64, and 96)
//!        int warp_id   = threadIdx.x / 32;
//!        int aggregate = WarpReduce(temp_storage[warp_id]).Sum(thread_data);
//!
//! Suppose the set of input ``thread_data`` across the block of threads is ``{0, 1, 2, 3, ..., 127}``.
//! The corresponding output ``aggregate`` in threads 0, 32, 64, and 96 will be
//! ``496``, ``1520``, ``2544``, and ``3568``, respectively (and is undefined in other threads).
//!
//! The code snippet below illustrates a single warp sum reduction within a block of 128 threads.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize WarpReduce for type int
//!        using WarpReduce = cub::WarpReduce<int>;
//!        // Allocate WarpReduce shared memory for one warp
//!        __shared__ typename WarpReduce::TempStorage temp_storage;
//!        ...
//!        // Only the first warp performs a reduction
//!        if (threadIdx.x < 32)
//!        {
//!            // Obtain one input item per thread
//!            int thread_data = ...
//!            // Return the warp-wide sum to lane0
//!            int aggregate = WarpReduce(temp_storage).Sum(thread_data);
//!
//! Suppose the set of input ``thread_data`` across the warp of threads is ``{0, 1, 2, 3, ..., 31}``.
//! The corresponding output ``aggregate`` in thread0 will be ``496`` (and is undefined in other threads).
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type
//!
//! @tparam LogicalWarpThreads
//!   <b>[optional]</b> The number of threads per "logical" warp (may be less than the number of
//!   hardware warp threads).  Default is the warp size of the targeted CUDA compute-capability
//!   (e.g., 32 threads for SM20).
//!
template <typename T, int LogicalWarpThreads = detail::warp_threads>
class WarpReduce
{
  static_assert(LogicalWarpThreads >= 1 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");

  static constexpr bool is_full_warp    = (LogicalWarpThreads == detail::warp_threads);
  static constexpr bool is_power_of_two = _CUDA_VSTD::has_single_bit(uint32_t{LogicalWarpThreads});

public:
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

  /// Internal specialization.
  /// Use SHFL-based reduction if LogicalWarpThreads is a power-of-two
  using InternalWarpReduce = _CUDA_VSTD::
    _If<is_power_of_two, detail::WarpReduceShfl<T, LogicalWarpThreads>, detail::WarpReduceSmem<T, LogicalWarpThreads>>;

#endif // _CCCL_DOXYGEN_INVOKED

private:
  /// Shared memory storage layout type for WarpReduce
  using _TempStorage = typename InternalWarpReduce::TempStorage;

  /// Shared storage reference
  _TempStorage& temp_storage;

public:
  /// \smemstorage{WarpReduce}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Logical warp and lane identifiers are constructed from ``threadIdx.x``.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduce(TempStorage& temp_storage)
      : temp_storage{temp_storage.Alias()}
  {}

  //! @}  end member group
  //! @name Summation reductions
  //! @{

  //! @rst
  //! Computes a warp-wide sum in the calling warp.
  //! The output is valid in warp *lane*\ :sub:`0`.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp sum reductions within a block of 128 threads
  //! (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!        // Allocate WarpReduce shared memory for 4 warps
  //!        __shared__ typename WarpReduce::TempStorage temp_storage[4];
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!        // Return the warp-wide sums to each lane0
  //!        int warp_id = threadIdx.x / 32;
  //!        int aggregate = WarpReduce(temp_storage[warp_id]).Sum(thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``{0, 1, 2, 3, ..., 127}``.
  //! The corresponding output ``aggregate`` in threads 0, 32, 64, and 96 will ``496``, ``1520``, ``2544``, and
  //! ``3568``, respectively (and is undefined in other threads).
  //! @endrst
  //!
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return InternalWarpReduce{temp_storage}.template Reduce<true>(input, LogicalWarpThreads, _CUDA_VSTD::plus<>{});
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(const InputType& input)
  {
    auto thread_reduction = cub::ThreadReduce(input, _CUDA_VSTD::plus<>{});
    return InternalWarpReduce{temp_storage}.template Reduce<true>(
      thread_reduction, LogicalWarpThreads, _CUDA_VSTD::plus<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(T input)
  {
    return InternalWarpReduce{temp_storage}.template Reduce<true>(input, LogicalWarpThreads, ::cuda::maximum<>{});
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(const InputType& input)
  {
    auto thread_reduction = cub::ThreadReduce(input, ::cuda::maximum<>{});
    return InternalWarpReduce{temp_storage}.template Reduce<true>(
      thread_reduction, LogicalWarpThreads, ::cuda::maximum<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(T input)
  {
    return InternalWarpReduce{temp_storage}.template Reduce<true>(input, LogicalWarpThreads, ::cuda::minimum<>{});
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(const InputType& input)
  {
    auto thread_reduction = cub::ThreadReduce(input, ::cuda::minimum<>{});
    return InternalWarpReduce{temp_storage}.template Reduce<true>(
      thread_reduction, LogicalWarpThreads, ::cuda::minimum<>{});
  }

  //! @rst
  //! Computes a partially-full warp-wide sum in the calling warp.
  //! The output is valid in warp *lane*\ :sub:`0`.
  //!
  //! All threads across the calling warp must agree on the same value for ``valid_items``.
  //! Otherwise the result is undefined.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a sum reduction within a single, partially-full
  //! block of 32 threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item per thread if in range
  //!        int thread_data;
  //!        if (threadIdx.x < valid_items)
  //!            thread_data = d_data[threadIdx.x];
  //!
  //!        // Return the warp-wide sums to each lane0
  //!        int aggregate = WarpReduce(temp_storage).Sum(
  //!            thread_data, valid_items);
  //!
  //! Suppose the input ``d_data`` is ``{0, 1, 2, 3, 4, ...`` and ``valid_items`` is ``4``.
  //! The corresponding output ``aggregate`` in *lane*\ :sub:`0` is ``6``
  //! (and is undefined in other threads).
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] valid_items
  //!   Total number of valid items in the calling thread's logical warp
  //!   (may be less than ``LogicalWarpThreads``)
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int valid_items)
  {
    // Determine if we don't need bounds checking
    return InternalWarpReduce{temp_storage}.template Reduce<false>(input, valid_items, _CUDA_VSTD::plus<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(T input, int valid_items)
  {
    // Determine if we don't need bounds checking
    return InternalWarpReduce{temp_storage}.template Reduce<false>(input, valid_items, ::cuda::maximum<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(T input, int valid_items)
  {
    // Determine if we don't need bounds checking
    return InternalWarpReduce{temp_storage}.template Reduce<false>(input, valid_items, ::cuda::minimum<>{});
  }

  //! @rst
  //! Computes a segmented sum in the calling warp where segments are defined by head-flags.
  //! The sum of each segment is returned to the first lane in that segment
  //! (which always includes *lane*\ :sub:`0`).
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a head-segmented warp sum
  //! reduction within a block of 32 threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item and flag per thread
  //!        int thread_data = ...
  //!        int head_flag = ...
  //!
  //!        // Return the warp-wide sums to each lane0
  //!        int aggregate = WarpReduce(temp_storage).HeadSegmentedSum(
  //!            thread_data, head_flag);
  //!
  //! Suppose the set of input ``thread_data`` and ``head_flag`` across the block of threads
  //! is ``{0, 1, 2, 3, ..., 31`` and is ``{1, 0, 0, 0, 1, 0, 0, 0, ..., 1, 0, 0, 0``,
  //! respectively. The corresponding output ``aggregate`` in threads 0, 4, 8, etc. will be
  //! ``6``, ``22``, ``38``, etc. (and is undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] head_flag
  //!   Head flag denoting whether or not `input` is the start of a new segment
  template <typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T HeadSegmentedSum(T input, FlagT head_flag)
  {
    return HeadSegmentedReduce(input, head_flag, _CUDA_VSTD::plus<>{});
  }

  //! @rst
  //! Computes a segmented sum in the calling warp where segments are defined by tail-flags.
  //! The sum of each segment is returned to the first lane in that segment
  //! (which always includes *lane*\ :sub:`0`).
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a tail-segmented warp sum reduction within a block of 32
  //! threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item and flag per thread
  //!        int thread_data = ...
  //!        int tail_flag = ...
  //!
  //!        // Return the warp-wide sums to each lane0
  //!        int aggregate = WarpReduce(temp_storage).TailSegmentedSum(
  //!            thread_data, tail_flag);
  //!
  //! Suppose the set of input ``thread_data`` and ``tail_flag`` across the block of threads
  //! is ``{0, 1, 2, 3, ..., 31}`` and is ``{0, 0, 0, 1, 0, 0, 0, 1, ..., 0, 0, 0, 1}``,
  //! respectively. The corresponding output ``aggregate`` in threads 0, 4, 8, etc. will be
  //! ``6``, ``22``, ``38``, etc. (and is undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] tail_flag
  //!   Head flag denoting whether or not `input` is the start of a new segment
  template <typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T TailSegmentedSum(T input, FlagT tail_flag)
  {
    return TailSegmentedReduce(input, tail_flag, _CUDA_VSTD::plus<>{});
  }

  //! @}  end member group
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a warp-wide reduction in the calling warp using the specified binary reduction
  //! functor. The output is valid in warp *lane*\ :sub:`0`.
  //!
  //! Supports non-commutative reduction operators
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates four concurrent warp max reductions within a block of
  //! 128 threads (one per each of the 32-thread warps).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for 4 warps
  //!        __shared__ typename WarpReduce::TempStorage temp_storage[4];
  //!
  //!        // Obtain one input item per thread
  //!        int thread_data = ...
  //!
  //!        // Return the warp-wide reductions to each lane0
  //!        int warp_id = threadIdx.x / 32;
  //!        int aggregate = WarpReduce(temp_storage[warp_id]).Reduce(
  //!            thread_data, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{0, 1, 2, 3, ..., 127}``. The corresponding output ``aggregate`` in threads 0, 32, 64, and
  //! 96 will be ``31``, ``63``, ``95``, and ``127``, respectively
  //! (and is undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return InternalWarpReduce{temp_storage}.template Reduce<true>(input, LogicalWarpThreads, reduction_op);
  }

  _CCCL_TEMPLATE(typename InputType, typename ReductionOp)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(const InputType& input, ReductionOp reduction_op)
  {
    auto thread_reduction = cub::ThreadReduce(input, reduction_op);
    return WarpReduce<T, LogicalWarpThreads>::Reduce(thread_reduction, LogicalWarpThreads, reduction_op);
  }
  //! @rst
  //! Computes a partially-full warp-wide reduction in the calling warp using the specified binary
  //! reduction functor. The output is valid in warp *lane*\ :sub:`0`.
  //!
  //! All threads across the calling warp must agree on the same value for ``valid_items``.
  //! Otherwise the result is undefined.
  //!
  //! Supports non-commutative reduction operators
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a max reduction within a single, partially-full
  //! block of 32 threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item per thread if in range
  //!        int thread_data;
  //!        if (threadIdx.x < valid_items)
  //!            thread_data = d_data[threadIdx.x];
  //!
  //!        // Return the warp-wide reductions to each lane0
  //!        int aggregate = WarpReduce(temp_storage).Reduce(
  //!            thread_data, cuda::maximum<>{}, valid_items);
  //!
  //! Suppose the input ``d_data`` is ``{0, 1, 2, 3, 4, ... }`` and ``valid_items``
  //! is ``4``. The corresponding output ``aggregate`` in thread0 is ``3`` (and is
  //! undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @param[in] valid_items
  //!   Total number of valid items in the calling thread's logical warp
  //!   (may be less than ``LogicalWarpThreads``)
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int valid_items)
  {
    return InternalWarpReduce{temp_storage}.template Reduce<false>(input, valid_items, reduction_op);
  }

  //! @rst
  //! Computes a segmented reduction in the calling warp where segments are defined by head-flags.
  //! The reduction of each segment is returned to the first lane in that segment
  //! (which always includes *lane*\ :sub:`0`).
  //!
  //! Supports non-commutative reduction operators
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a head-segmented warp max
  //! reduction within a block of 32 threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item and flag per thread
  //!        int thread_data = ...
  //!        int head_flag = ...
  //!
  //!        // Return the warp-wide reductions to each lane0
  //!        int aggregate = WarpReduce(temp_storage).HeadSegmentedReduce(
  //!            thread_data, head_flag, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` and ``head_flag`` across the block of threads
  //! is ``{0, 1, 2, 3, ..., 31}`` and is ``{1, 0, 0, 0, 1, 0, 0, 0, ..., 1, 0, 0, 0}``,
  //! respectively. The corresponding output ``aggregate`` in threads 0, 4, 8, etc. will be
  //! ``3``, ``7``, ``11``, etc. (and is undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] head_flag
  //!   Head flag denoting whether or not `input` is the start of a new segment
  //!
  //! @param[in] reduction_op
  //!   Reduction operator
  template <typename ReductionOp, typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T HeadSegmentedReduce(T input, FlagT head_flag, ReductionOp reduction_op)
  {
    return InternalWarpReduce{temp_storage}.template SegmentedReduce<true>(input, head_flag, reduction_op);
  }

  //! @rst
  //! Computes a segmented reduction in the calling warp where segments are defined by tail-flags.
  //! The reduction of each segment is returned to the first lane in that segment
  //! (which always includes *lane*\ :sub:`0`).
  //!
  //! Supports non-commutative reduction operators
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a tail-segmented warp max
  //! reduction within a block of 32 threads (one warp).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize WarpReduce for type int
  //!        using WarpReduce = cub::WarpReduce<int>;
  //!
  //!        // Allocate WarpReduce shared memory for one warp
  //!        __shared__ typename WarpReduce::TempStorage temp_storage;
  //!
  //!        // Obtain one input item and flag per thread
  //!        int thread_data = ...
  //!        int tail_flag = ...
  //!
  //!        // Return the warp-wide reductions to each lane0
  //!        int aggregate = WarpReduce(temp_storage).TailSegmentedReduce(
  //!            thread_data, tail_flag, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` and ``tail_flag`` across the block of threads
  //! is ``{0, 1, 2, 3, ..., 31}`` and is ``{0, 0, 0, 1, 0, 0, 0, 1, ..., 0, 0, 0, 1}``,
  //! respectively. The corresponding output ``aggregate`` in threads 0, 4, 8, etc. will be
  //! ``3``, ``7``, ``11``, etc. (and is undefined in other threads).
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] tail_flag
  //!   Tail flag denoting whether or not \p input is the end of the current segment
  //!
  //! @param[in] reduction_op
  //!   Reduction operator
  template <typename ReductionOp, typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T TailSegmentedReduce(T input, FlagT tail_flag, ReductionOp reduction_op)
  {
    return InternalWarpReduce{temp_storage}.template SegmentedReduce<false>(input, tail_flag, reduction_op);
  }

  //! @}  end member group
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <typename T>
class WarpReduce<T, 1>
{
private:
  using _TempStorage = cub::NullType;

public:
  struct InternalWarpReduce
  {
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    _CCCL_DEVICE _CCCL_FORCEINLINE InternalWarpReduce(TempStorage& /*temp_storage */) {}

    template <bool ALL_LANES_VALID, typename ReductionOp>
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T
    Reduce(T input, int /* valid_items */, ReductionOp /* reduction_op */)
    {
      return input;
    }

    template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T
    SegmentedReduce(T input, FlagT /* flag */, ReductionOp /* reduction_op */)
    {
      return input;
    }
  };

  using TempStorage = typename InternalWarpReduce::TempStorage;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduce(TempStorage& /*temp_storage */) {}

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return input;
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(const InputType& input)
  {
    return cub::ThreadReduce(input, _CUDA_VSTD::plus<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int /* valid_items */)
  {
    return input;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(T input)
  {
    return input;
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(const InputType& input)
  {
    return cub::ThreadReduce(input, ::cuda::maximum<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Max(T input, int /* valid_items */)
  {
    return input;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(T input)
  {
    return input;
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(const InputType& input)
  {
    return cub::ThreadReduce(input, ::cuda::minimum<>{});
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Min(T input, int /* valid_items */)
  {
    return input;
  }

  template <typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T HeadSegmentedSum(T input, FlagT /* head_flag */)
  {
    return input;
  }

  template <typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T TailSegmentedSum(T input, FlagT /* tail_flag */)
  {
    return input;
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp /* reduction_op */)
  {
    return input;
  }

  _CCCL_TEMPLATE(typename InputType, typename ReductionOp)
  _CCCL_REQUIRES(_CCCL_TRAIT(detail::is_fixed_size_random_access_range, InputType))
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(const InputType& input, ReductionOp reduction_op)
  {
    return cub::ThreadReduce(input, reduction_op);
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp /* reduction_op */, int /* valid_items */)
  {
    return input;
  }

  template <typename ReductionOp, typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T
  HeadSegmentedReduce(T input, FlagT /* head_flag */, ReductionOp /* reduction_op */)
  {
    return input;
  }

  template <typename ReductionOp, typename FlagT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T
  TailSegmentedReduce(T input, FlagT /* tail_flag */, ReductionOp /* reduction_op */)
  {
    return input;
  }
};

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
