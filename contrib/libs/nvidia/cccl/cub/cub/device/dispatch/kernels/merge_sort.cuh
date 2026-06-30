// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/util_policy_wrapper_t.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/make_load_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{

/**
 * @brief Helper class template that provides two agent template instantiations: one instantiated with the default
 * policy and one with the fallback policy. This helps to avoid having to enlist all the agent's template parameters
 * twice: once for the default agent and once for the fallback agent
 */
template <typename DefaultPolicyT, typename FallbackPolicyT, template <typename...> class AgentT, typename... AgentParamsT>
struct dual_policy_agent_helper_t
{
  using default_agent_t  = AgentT<DefaultPolicyT, AgentParamsT...>;
  using fallback_agent_t = AgentT<FallbackPolicyT, AgentParamsT...>;

  static constexpr auto default_size  = sizeof(typename default_agent_t::TempStorage);
  static constexpr auto fallback_size = sizeof(typename fallback_agent_t::TempStorage);
};

/**
 * @brief Helper class template for merge sort-specific virtual shared memory handling. The merge sort algorithm in its
 * current implementation relies on the fact that both the sorting as well as the merging kernels use the same tile
 * size. This circumstance needs to be respected when determining whether the fallback policy for large user types is
 * applicable: we must either use the fallback for both or for none of the two agents.
 */
template <typename DefaultPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
class merge_sort_vsmem_helper_t
{
private:
  // Default fallback policy with a smaller tile size
  using fallback_policy_t = cub::detail::policy_wrapper_t<DefaultPolicyT, 64, 1>;

  // Helper for the `AgentBlockSort` template with one member type alias for the agent template instantiated with the
  // default policy and one instantiated with the fallback policy
  using block_sort_helper_t = dual_policy_agent_helper_t<
    DefaultPolicyT,
    fallback_policy_t,
    merge_sort::AgentBlockSort,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;
  using default_block_sort_agent_t  = typename block_sort_helper_t::default_agent_t;
  using fallback_block_sort_agent_t = typename block_sort_helper_t::fallback_agent_t;

  // Helper for the `AgentMerge` template with one member type alias for the agent template instantiated with the
  // default policy and one instantiated with the fallback policy
  using merge_helper_t = dual_policy_agent_helper_t<
    DefaultPolicyT,
    fallback_policy_t,
    merge_sort::AgentMerge,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;
  using default_merge_agent_t  = typename merge_helper_t::default_agent_t;
  using fallback_merge_agent_t = typename merge_helper_t::fallback_agent_t;

  // Use fallback if either (a) the default block sort or (b) the block merge agent exceed the maximum shared memory
  // available per block and both (1) the fallback block sort and (2) the fallback merge agent would not exceed the
  // available shared memory
  static constexpr auto max_default_size =
    (::cuda::std::max) (block_sort_helper_t::default_size, merge_helper_t::default_size);
  static constexpr auto max_fallback_size =
    (::cuda::std::max) (block_sort_helper_t::fallback_size, merge_helper_t::fallback_size);
  static constexpr bool uses_fallback_policy =
    (max_default_size > max_smem_per_block) && (max_fallback_size <= max_smem_per_block);

public:
  using policy_t = ::cuda::std::_If<uses_fallback_policy, fallback_policy_t, DefaultPolicyT>;
  using block_sort_agent_t =
    ::cuda::std::_If<uses_fallback_policy, fallback_block_sort_agent_t, default_block_sort_agent_t>;
  using merge_agent_t = ::cuda::std::_If<uses_fallback_policy, fallback_merge_agent_t, default_merge_agent_t>;
};

// TODO: this class should be templated on `typename... Ts` to avoid repetition,
// but due to an issue with NVCC 12.0 we currently template each member function
// individually instead.
struct VSMemHelper
{
  template <typename ActivePolicyT, typename... Ts>
  using MergeSortVSMemHelperT = merge_sort_vsmem_helper_t<ActivePolicyT, Ts...>;

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr int BlockThreads(ActivePolicyT /*policy*/)
  {
    return MergeSortVSMemHelperT<ActivePolicyT, Ts...>::policy_t::BLOCK_THREADS;
  }

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr int ItemsPerTile(ActivePolicyT /*policy*/)
  {
    return MergeSortVSMemHelperT<ActivePolicyT, Ts...>::policy_t::ITEMS_PER_TILE;
  }

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t BlockSortVSMemPerBlock(ActivePolicyT /*policy*/)
  {
    return detail::vsmem_helper_impl<
      typename MergeSortVSMemHelperT<ActivePolicyT, Ts...>::block_sort_agent_t>::vsmem_per_block;
  }

  template <typename ActivePolicyT, typename... Ts>
  _CCCL_HOST_DEVICE static constexpr ::cuda::std::size_t MergeVSMemPerBlock(ActivePolicyT /*policy*/)
  {
    return detail::vsmem_helper_impl<
      typename MergeSortVSMemHelperT<ActivePolicyT, Ts...>::merge_agent_t>::vsmem_per_block;
  }

  template <typename AgentT>
  using VSmemHelperT = vsmem_helper_impl<AgentT>;
};

template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT,
          typename VSMemHelperT = VSMemHelper>
__launch_bounds__(
  VSMemHelperT::template MergeSortVSMemHelperT<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy_t::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortBlockSortKernel(
    bool ping,
    KeyInputIteratorT keys_in,
    ValueInputIteratorT items_in,
    KeyIteratorT keys_out,
    ValueIteratorT items_out,
    OffsetT keys_count,
    KeyT* tmp_keys_out,
    ValueT* tmp_items_out,
    CompareOpT compare_op,
    vsmem_t vsmem)
{
  using MergeSortHelperT = typename VSMemHelperT::template MergeSortVSMemHelperT<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  using ActivePolicyT = typename MergeSortHelperT::policy_t;

  using AgentBlockSortT = typename MergeSortHelperT::block_sort_agent_t;

  using VSmemHelperT = typename VSMemHelperT::template VSmemHelperT<AgentBlockSortT>;

  // Static shared memory allocation
  __shared__ typename VSmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentBlockSortT::TempStorage& temp_storage = VSmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  AgentBlockSortT agent(
    ping,
    temp_storage,
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), keys_in),
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), items_in),
    keys_count,
    keys_out,
    items_out,
    tmp_keys_out,
    tmp_items_out,
    compare_op);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  VSmemHelperT::discard_temp_storage(temp_storage);
}

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT, typename KeyT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortPartitionKernel(
  bool ping,
  KeyIteratorT keys_ping,
  KeyT* keys_pong,
  OffsetT keys_count,
  OffsetT num_partitions,
  OffsetT* merge_partitions,
  CompareOpT compare_op,
  OffsetT target_merged_tiles_number,
  int items_per_tile)
{
  OffsetT partition_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (partition_idx < num_partitions)
  {
    AgentPartition<KeyIteratorT, OffsetT, CompareOpT, KeyT>{
      ping,
      keys_ping,
      keys_pong,
      keys_count,
      partition_idx,
      merge_partitions,
      compare_op,
      target_merged_tiles_number,
      items_per_tile,
      num_partitions}
      .Process();
  }
}

template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT,
          typename VSMemHelperT = VSMemHelper>
__launch_bounds__(
  VSMemHelperT::template MergeSortVSMemHelperT<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>::policy_t::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceMergeSortMergeKernel(
    bool ping,
    KeyIteratorT keys_ping,
    ValueIteratorT items_ping,
    OffsetT keys_count,
    KeyT* keys_pong,
    ValueT* items_pong,
    CompareOpT compare_op,
    OffsetT* merge_partitions,
    OffsetT target_merged_tiles_number,
    vsmem_t vsmem)
{
  using MergeSortHelperT = typename VSMemHelperT::template MergeSortVSMemHelperT<
    typename ChainedPolicyT::ActivePolicy::MergeSortPolicy,
    KeyInputIteratorT,
    ValueInputIteratorT,
    KeyIteratorT,
    ValueIteratorT,
    OffsetT,
    CompareOpT,
    KeyT,
    ValueT>;

  using ActivePolicyT = typename MergeSortHelperT::policy_t;

  using AgentMergeT = typename MergeSortHelperT::merge_agent_t;

  using VSmemHelperT = typename VSMemHelperT::template VSmemHelperT<AgentMergeT>;

  // Static shared memory allocation
  __shared__ typename VSmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentMergeT::TempStorage& temp_storage = VSmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  AgentMergeT agent(
    ping,
    temp_storage,
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), keys_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), items_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), keys_pong),
    THRUST_NS_QUALIFIER::cuda_cub::core::detail::make_load_iterator(ActivePolicyT(), items_pong),
    keys_count,
    keys_pong,
    items_pong,
    keys_ping,
    items_ping,
    compare_op,
    merge_partitions,
    target_merged_tiles_number);

  agent.Process();

  // If applicable, hints to discard modified cache lines for vsmem
  VSmemHelperT::discard_temp_storage(temp_storage);
}

} // namespace detail::merge_sort

CUB_NAMESPACE_END
