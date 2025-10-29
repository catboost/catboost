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

#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/tabulate_output_iterator.h>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{

template <typename GlobalAccumT, typename PromoteToGlobalOpT, typename GlobalReductionOpT, typename FinalResultOutIteratorT>
struct accumulating_transform_output_op
{
  bool first_partition;
  bool last_partition;

  // We use a double-buffer to make assignment idempotent (i.e., allow potential repeated assignment)
  GlobalAccumT* d_previous_aggregate;
  GlobalAccumT* d_aggregate_out;

  // Output iterator to which the final result of type `GlobalAccumT` across all partitions will be assigned
  FinalResultOutIteratorT d_out;

  // Unary promotion operator type that is used to transform a per-partition result to a global result
  PromoteToGlobalOpT promote_op;

  // Reduction operation
  GlobalReductionOpT reduce_op;

  template <typename IndexT, typename AccumT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, AccumT per_partition_aggregate)
  {
    // Add this partitions aggregate to the global aggregate
    if (first_partition)
    {
      *d_aggregate_out = promote_op(per_partition_aggregate);
    }
    else
    {
      *d_aggregate_out = reduce_op(*d_previous_aggregate, promote_op(per_partition_aggregate));
    }

    // If this is the last partition, we write the global aggregate to the user-provided iterator
    if (last_partition)
    {
      *d_out = *d_aggregate_out;
    }
  }

  /**
   * This is a helper function that's invoked after a partition has been fully processed
   */
  template <typename GlobalOffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size, bool next_partition_is_the_last)
  {
    promote_op.advance(partition_size);
    using ::cuda::std::swap;
    swap(d_previous_aggregate, d_aggregate_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
  }
};

/**
 * Unary "promotion" operator type that is used to transform a per-partition result to a global result
 */
template <typename GlobalOffsetT>
struct local_to_global_op
{
  // The current partition's offset to be factored into this partition's index
  GlobalOffsetT current_partition_offset;

  /**
   * This helper function is invoked after a partition has been fully processed, in preparation for the next partition.
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size)
  {
    current_partition_offset += partition_size;
  }

  /**
   * Unary operator called to transform the per-partition aggregate of a partition to a global aggregate type (i.e., one
   * that is used to reduce across partitions).
   */
  template <typename PerPartitionOffsetT, typename AccumT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<GlobalOffsetT, AccumT>
  operator()(KeyValuePair<PerPartitionOffsetT, AccumT> partition_aggregate)
  {
    return KeyValuePair<GlobalOffsetT, AccumT>{
      current_partition_offset + static_cast<GlobalOffsetT>(partition_aggregate.key), partition_aggregate.value};
  }
};

/******************************************************************************
 * Single-problem streaming reduction dispatch
 *****************************************************************************/

// Utility class for dispatching a streaming device-wide argument extremum computation, like `ArgMin` and `ArgMax`.
// Streaming, here, refers to the approach used for large number of items that are processed in multiple partitions.
//
// @tparam InputIteratorT
//   Random-access input iterator type for reading input items @iterator
//
// @tparam OutputIteratorT
//   Output iterator type for writing the result of the (index, extremum)-key-value-pair
//
// @tparam PerPartitionOffsetT
//   Offset type used as the index to access items within one partition, i.e., the offset type used within the kernel
// template specialization
//
// @tparam GlobalOffsetT
//   Offset type used as the index to access items within the total input range, i.e., in the range [d_in, d_in +
// num_items)
//
// @tparam ReductionOpT
//   Binary reduction functor type having a member function that returns the selected extremum of two input items.
//   The streaming reduction requires two overloads, one used for selecting the extremum within one partition and one
//   for selecting the extremum across partitions.
//
// @tparam InitT
//   Initial value type
//
// @tparam PolicyChainT
//   The policy chain passed to the DispatchReduce template specialization
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename PerPartitionOffsetT,
          typename GlobalOffsetT,
          typename ReductionOpT,
          typename InitT,
          typename PolicyChainT =
            detail::reduce::policy_hub<KeyValuePair<PerPartitionOffsetT, InitT>, PerPartitionOffsetT, ReductionOpT>>
struct dispatch_streaming_arg_reduce_t
{
  // Internal dispatch routine for computing a device-wide argument extremum, like `ArgMin` and `ArgMax`
  //
  // @param[in] d_temp_storage
  //   Device-accessible allocation of temporary storage. When `nullptr`, the
  //   required allocation size is written to `temp_storage_bytes` and no work
  //   is done.
  //
  // @param[in,out] temp_storage_bytes
  //   Reference to size in bytes of `d_temp_storage` allocation
  //
  // @param[in] d_in
  //   Pointer to the input sequence of data items
  //
  // @param[out] d_result_out
  //   Iterator to which the  (index, extremum)-key-value-pair is written
  //
  // @param[in] num_items
  //   Total number of input items (i.e., length of `d_in`)
  //
  // @param[in] reduce_op
  //   Reduction operator that returns the (index, extremum)-key-value-pair corresponding to the extremum of the two
  // input items.
  //
  // @param[in] init
  //   The extremum value to be returned for empty problems
  //
  // @param[in] stream
  //   CUDA stream to launch kernels within.
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_result_out,
    GlobalOffsetT num_items,
    ReductionOpT reduce_op,
    InitT init,
    cudaStream_t stream)
  {
    // Constant iterator to provide the offset of the current partition for the user-provided input iterator
    using constant_offset_it_t = THRUST_NS_QUALIFIER::constant_iterator<GlobalOffsetT>;

    // Wrapped input iterator to produce index-value tuples, i.e., <PerPartitionOffsetT, InputT>-tuples
    // We make sure to offset the user-provided input iterator by the current partition's offset
    using arg_index_input_iterator_t = ArgIndexInputIterator<InputIteratorT, PerPartitionOffsetT, InitT>;

    // The type used for the aggregate that the user wants to find the extremum for
    using output_aggregate_t = InitT;

    // The output tuple type (i.e., extremum plus index tuples)
    using per_partition_accum_t = KeyValuePair<PerPartitionOffsetT, output_aggregate_t>;
    using global_accum_t        = KeyValuePair<GlobalOffsetT, output_aggregate_t>;

    // Unary promotion operator type that is used to transform a per-partition result to a global result
    // operator()(per_partition_accum_t) -> global_accum_t
    using local_to_global_op_t = local_to_global_op<GlobalOffsetT>;

    // Reduction operator type that enables accumulating per-partition results to a global reduction result
    using accumulating_transform_output_op_t =
      accumulating_transform_output_op<global_accum_t, local_to_global_op_t, ReductionOpT, OutputIteratorT>;

    // The output iterator that implements the logic to accumulate per-partition result to a global aggregate and,
    // eventually, write to the user-provided output iterators
    using accumulating_transform_out_it_t =
      THRUST_NS_QUALIFIER::tabulate_output_iterator<accumulating_transform_output_op_t>;

    // Empty problem initialization type
    using empty_problem_init_t = empty_problem_init_t<per_partition_accum_t>;

    // Per-partition DispatchReduce template specialization
    using dispatch_reduce_t =
      DispatchReduce<arg_index_input_iterator_t,
                     accumulating_transform_out_it_t,
                     PerPartitionOffsetT,
                     ReductionOpT,
                     empty_problem_init_t,
                     per_partition_accum_t,
                     ::cuda::std::identity,
                     PolicyChainT>;

    // The current partition's input iterator is an ArgIndex iterator that generates indices relative to the beginning
    // of the current partition, i.e., [0, partition_size) along with an OffsetIterator that offsets the user-provided
    // input iterator by the current partition's offset
    arg_index_input_iterator_t d_indexed_offset_in(d_in);

    // Transforms the per-partition result to a global result by adding the current partition's offset to the arg result
    // of a partition
    local_to_global_op_t local_to_global_op{GlobalOffsetT{0}};

    // Upper bound at which we want to cut the input into multiple partitions. Align to 4096 bytes for performance
    // reasons
    static constexpr PerPartitionOffsetT max_offset_size = ::cuda::std::numeric_limits<PerPartitionOffsetT>::max();
    static constexpr PerPartitionOffsetT max_partition_size =
      max_offset_size - (max_offset_size % PerPartitionOffsetT{4096});

    // Whether the given number of items fits into a single partition
    const bool is_single_partition =
      static_cast<GlobalOffsetT>(max_partition_size) >= static_cast<GlobalOffsetT>(num_items);

    // The largest partition size ever encountered
    const auto largest_partition_size =
      is_single_partition ? static_cast<PerPartitionOffsetT>(num_items) : max_partition_size;

    accumulating_transform_output_op_t accumulating_out_op{
      true, is_single_partition, nullptr, nullptr, d_result_out, local_to_global_op, reduce_op};

    empty_problem_init_t initial_value{{PerPartitionOffsetT{1}, init}};

    void* allocations[2]       = {nullptr, nullptr};
    size_t allocation_sizes[2] = {0, 2 * sizeof(global_accum_t)};

    // Query temporary storage requirements for per-partition reduction
    dispatch_reduce_t::Dispatch(
      nullptr,
      allocation_sizes[0],
      d_indexed_offset_in,
      THRUST_NS_QUALIFIER::make_tabulate_output_iterator(accumulating_out_op),
      static_cast<PerPartitionOffsetT>(largest_partition_size),
      reduce_op,
      initial_value,
      stream);

    // Alias the temporary allocations from the single storage blob (or compute the necessary size
    // of the blob)
    cudaError_t error = detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (error != cudaSuccess)
    {
      return error;
    }

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    // Pointer to the double-buffer of global accumulators, which aggregate cross-partition results
    global_accum_t* const d_global_aggregates = static_cast<global_accum_t*>(allocations[1]);

    accumulating_out_op = accumulating_transform_output_op_t{
      true,
      is_single_partition,
      d_global_aggregates,
      (d_global_aggregates + 1),
      d_result_out,
      local_to_global_op,
      reduce_op};

    for (GlobalOffsetT current_partition_offset = 0; current_partition_offset < static_cast<GlobalOffsetT>(num_items);
         current_partition_offset += static_cast<GlobalOffsetT>(max_partition_size))
    {
      const GlobalOffsetT remaining_items = (num_items - current_partition_offset);
      const GlobalOffsetT current_num_items =
        (remaining_items < max_partition_size) ? remaining_items : max_partition_size;

      d_indexed_offset_in = arg_index_input_iterator_t(d_in + current_partition_offset);

      error = dispatch_reduce_t::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_indexed_offset_in,
        THRUST_NS_QUALIFIER::make_tabulate_output_iterator(accumulating_out_op),
        static_cast<PerPartitionOffsetT>(current_num_items),
        reduce_op,
        initial_value,
        stream);

      if (error != cudaSuccess)
      {
        return error;
      }

      // Whether the next partition will be the last partition
      const bool next_partition_is_last =
        (remaining_items - current_num_items) <= static_cast<GlobalOffsetT>(max_partition_size);
      accumulating_out_op.advance(current_num_items, next_partition_is_last);
    }

    return cudaSuccess;
  }
};

} // namespace detail::reduce
CUB_NAMESPACE_END

#endif // !_CCCL_DOXYGEN_INVOKED
