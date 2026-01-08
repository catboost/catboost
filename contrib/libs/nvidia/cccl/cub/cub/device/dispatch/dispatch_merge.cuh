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

#include <cub/agent/agent_merge.cuh>
#include <cub/device/dispatch/tuning/tuning_merge.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN
namespace detail::merge
{
inline constexpr int fallback_BLOCK_THREADS    = 64;
inline constexpr int fallback_ITEMS_PER_THREAD = 1;

template <typename DefaultPolicy, class... Args>
class choose_merge_agent
{
  using default_agent_t = agent_t<DefaultPolicy, Args...>;
  using fallback_agent_t =
    agent_t<policy_wrapper_t<DefaultPolicy, fallback_BLOCK_THREADS, fallback_ITEMS_PER_THREAD>, Args...>;

  // Use fallback if merge agent exceeds maximum shared memory, but the fallback agent still fits
  static constexpr bool use_fallback = sizeof(typename default_agent_t::TempStorage) > max_smem_per_block
                                    && sizeof(typename fallback_agent_t::TempStorage) <= max_smem_per_block;

public:
  using type = ::cuda::std::conditional_t<use_fallback, fallback_agent_t, default_agent_t>;
};

// Computes the merge path intersections at equally wide intervals. The approach is outlined in the paper:
// Odeh et al, "Merge Path - Parallel Merging Made Simple" * doi : 10.1109 / IPDPSW .2012.202
// The algorithm is the same as AgentPartition for merge sort, but that agent handles a lot more.
template <typename MaxPolicy,
          typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp>
CUB_DETAIL_KERNEL_ATTRIBUTES void device_partition_merge_path_kernel(
  KeyIt1 keys1,
  Offset keys1_count,
  KeyIt2 keys2,
  Offset keys2_count,
  Offset num_partitions,
  Offset* merge_partitions,
  CompareOp compare_op)
{
  // items_per_tile must be the same of the merge kernel later, so we have to consider whether a fallback agent will be
  // selected for the merge agent that changes the tile size
  constexpr int items_per_tile =
    choose_merge_agent<typename MaxPolicy::ActivePolicy::merge_policy,
                       KeyIt1,
                       ValueIt1,
                       KeyIt2,
                       ValueIt2,
                       KeyIt3,
                       ValueIt3,
                       Offset,
                       CompareOp>::type::policy::ITEMS_PER_TILE;
  const Offset partition_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (partition_idx < num_partitions)
  {
    const Offset partition_at       = (::cuda::std::min) (partition_idx * items_per_tile, keys1_count + keys2_count);
    merge_partitions[partition_idx] = cub::MergePath(keys1, keys2, keys1_count, keys2_count, partition_at, compare_op);
  }
}

template <typename MaxPolicy,
          typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp>
__launch_bounds__(
  choose_merge_agent<typename MaxPolicy::ActivePolicy::merge_policy,
                     KeyIt1,
                     ValueIt1,
                     KeyIt2,
                     ValueIt2,
                     KeyIt3,
                     ValueIt3,
                     Offset,
                     CompareOp>::type::policy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void device_merge_kernel(
    KeyIt1 keys1,
    ValueIt1 items1,
    Offset num_keys1,
    KeyIt2 keys2,
    ValueIt2 items2,
    Offset num_keys2,
    KeyIt3 keys_result,
    ValueIt3 items_result,
    CompareOp compare_op,
    Offset* merge_partitions,
    vsmem_t global_temp_storage)
{
  // the merge agent loads keys into a local array of KeyIt1::value_type, on which the comparisons are performed
  using key_t = it_value_t<KeyIt1>;
  static_assert(::cuda::std::__invocable<CompareOp, key_t, key_t>::value,
                "Comparison operator cannot compare two keys");
  static_assert(::cuda::std::is_convertible_v<typename ::cuda::std::__invoke_of<CompareOp, key_t, key_t>::type, bool>,
                "Comparison operator must be convertible to bool");

  using MergeAgent = typename choose_merge_agent<
    typename MaxPolicy::ActivePolicy::merge_policy,
    KeyIt1,
    ValueIt1,
    KeyIt2,
    ValueIt2,
    KeyIt3,
    ValueIt3,
    Offset,
    CompareOp>::type;
  using MergePolicy = typename MergeAgent::policy;

  using THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator;
  using vsmem_helper_t = vsmem_helper_impl<MergeAgent>;
  __shared__ typename vsmem_helper_t::static_temp_storage_t shared_temp_storage;
  auto& temp_storage = vsmem_helper_t::get_temp_storage(shared_temp_storage, global_temp_storage);
  MergeAgent{
    temp_storage.Alias(),
    make_load_iterator(MergePolicy{}, keys1),
    make_load_iterator(MergePolicy{}, items1),
    num_keys1,
    make_load_iterator(MergePolicy{}, keys2),
    make_load_iterator(MergePolicy{}, items2),
    num_keys2,
    keys_result,
    items_result,
    compare_op,
    merge_partitions}();
  vsmem_helper_t::discard_temp_storage(temp_storage);
}

template <typename KeyIt1,
          typename ValueIt1,
          typename KeyIt2,
          typename ValueIt2,
          typename KeyIt3,
          typename ValueIt3,
          typename Offset,
          typename CompareOp,
          typename PolicyHub = detail::merge::policy_hub<it_value_t<KeyIt1>, it_value_t<ValueIt1>>>
struct dispatch_t
{
  void* d_temp_storage;
  size_t& temp_storage_bytes;
  KeyIt1 d_keys1;
  ValueIt1 d_values1;
  Offset num_items1;
  KeyIt2 d_keys2;
  ValueIt2 d_values2;
  Offset num_items2;
  KeyIt3 d_keys_out;
  ValueIt3 d_values_out;
  CompareOp compare_op;
  cudaStream_t stream;

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using max_policy_t   = typename PolicyHub::max_policy;
    using merge_policy_t = typename ActivePolicy::merge_policy;
    using agent_t =
      typename choose_merge_agent<merge_policy_t, KeyIt1, ValueIt1, KeyIt2, ValueIt2, KeyIt3, ValueIt3, Offset, CompareOp>::
        type;

    const auto num_tiles = ::cuda::ceil_div(num_items1 + num_items2, agent_t::policy::ITEMS_PER_TILE);
    void* allocations[2] = {nullptr, nullptr};
    {
      const size_t merge_partitions_size      = (1 + num_tiles) * sizeof(Offset);
      const size_t virtual_shared_memory_size = num_tiles * vsmem_helper_impl<agent_t>::vsmem_per_block;
      const size_t allocation_sizes[2]        = {merge_partitions_size, virtual_shared_memory_size};
      const auto error =
        CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    // Return if only temporary storage was requested or there is no work to be done
    if (d_temp_storage == nullptr || num_tiles == 0)
    {
      return cudaSuccess;
    }

    auto merge_partitions = static_cast<Offset*>(allocations[0]);

    // partition the merge path
    {
      const Offset num_partitions               = num_tiles + 1;
      constexpr int threads_per_partition_block = 256; // TODO(bgruber): no policy?
      const int partition_grid_size = static_cast<int>(::cuda::ceil_div(num_partitions, threads_per_partition_block));

      auto error = CubDebug(
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
          partition_grid_size, threads_per_partition_block, 0, stream)
          .doit(device_partition_merge_path_kernel<
                  max_policy_t,
                  KeyIt1,
                  ValueIt1,
                  KeyIt2,
                  ValueIt2,
                  KeyIt3,
                  ValueIt3,
                  Offset,
                  CompareOp>,
                d_keys1,
                num_items1,
                d_keys2,
                num_items2,
                num_partitions,
                merge_partitions,
                compare_op));
      if (cudaSuccess != error)
      {
        return error;
      }
      error = CubDebug(DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    // merge
    if (num_tiles > 0)
    {
      auto vshmem_ptr = vsmem_t{allocations[1]};
      auto error      = CubDebug(
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
          static_cast<int>(num_tiles), static_cast<int>(agent_t::policy::BLOCK_THREADS), 0, stream)
          .doit(
            device_merge_kernel<max_policy_t, KeyIt1, ValueIt1, KeyIt2, ValueIt2, KeyIt3, ValueIt3, Offset, CompareOp>,
            d_keys1,
            d_values1,
            num_items1,
            d_keys2,
            d_values2,
            num_items2,
            d_keys_out,
            d_values_out,
            compare_op,
            merge_partitions,
            vshmem_ptr));
      if (cudaSuccess != error)
      {
        return error;
      }
      error = CubDebug(DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  template <typename... Args>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(Args&&... args)
  {
    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }
    dispatch_t dispatch{::cuda::std::forward<Args>(args)...};
    error = CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
    if (cudaSuccess != error)
    {
      return error;
    }

    return cudaSuccess;
  }
};
} // namespace detail::merge
CUB_NAMESPACE_END
