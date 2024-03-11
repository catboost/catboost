/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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


#include <cub/agent/agent_merge_sort.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/detail/integer_math.h>

CUB_NAMESPACE_BEGIN


template <bool UseVShmem,
          typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
void __global__ __launch_bounds__(ChainedPolicyT::ActivePolicy::MergeSortPolicy::BLOCK_THREADS)
DeviceMergeSortBlockSortKernel(bool ping,
                               KeyInputIteratorT keys_in,
                               ValueInputIteratorT items_in,
                               KeyIteratorT keys_out,
                               ValueIteratorT items_out,
                               OffsetT keys_count,
                               KeyT *tmp_keys_out,
                               ValueT *tmp_items_out,
                               CompareOpT compare_op,
                               char *vshmem)
{
  extern __shared__ char shmem[];
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy::MergeSortPolicy;

  using AgentBlockSortT = AgentBlockSort<ActivePolicyT,
                                         KeyInputIteratorT,
                                         ValueInputIteratorT,
                                         KeyIteratorT,
                                         ValueIteratorT,
                                         OffsetT,
                                         CompareOpT,
                                         KeyT,
                                         ValueT>;

  const OffsetT vshmem_offset = blockIdx.x *
                                AgentBlockSortT::SHARED_MEMORY_SIZE;

  typename AgentBlockSortT::TempStorage &storage =
    *reinterpret_cast<typename AgentBlockSortT::TempStorage *>(
      UseVShmem ? vshmem + vshmem_offset : shmem);

  AgentBlockSortT agent(ping,
                        storage,
                        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_in),
                        THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_in),
                        keys_count,
                        keys_out,
                        items_out,
                        tmp_keys_out,
                        tmp_items_out,
                        compare_op);

  agent.Process();
}

template <typename KeyIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT>
__global__ void DeviceMergeSortPartitionKernel(bool ping,
                                               KeyIteratorT keys_ping,
                                               KeyT *keys_pong,
                                               OffsetT keys_count,
                                               OffsetT num_partitions,
                                               OffsetT *merge_partitions,
                                               CompareOpT compare_op,
                                               OffsetT target_merged_tiles_number,
                                               int items_per_tile)
{
  OffsetT partition_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (partition_idx < num_partitions)
  {
    AgentPartition<KeyIteratorT, OffsetT, CompareOpT, KeyT> agent(
      ping,
      keys_ping,
      keys_pong,
      keys_count,
      partition_idx,
      merge_partitions,
      compare_op,
      target_merged_tiles_number,
      items_per_tile);

    agent.Process();
  }
}

template <bool UseVShmem,
          typename ChainedPolicyT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
void __global__ __launch_bounds__(ChainedPolicyT::ActivePolicy::MergeSortPolicy::BLOCK_THREADS)
DeviceMergeSortMergeKernel(bool ping,
                           KeyIteratorT keys_ping,
                           ValueIteratorT items_ping,
                           OffsetT keys_count,
                           KeyT *keys_pong,
                           ValueT *items_pong,
                           CompareOpT compare_op,
                           OffsetT *merge_partitions,
                           OffsetT target_merged_tiles_number,
                           char *vshmem)
{
  extern __shared__ char shmem[];

  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy::MergeSortPolicy;
  using AgentMergeT = AgentMerge<ActivePolicyT,
                                 KeyIteratorT,
                                 ValueIteratorT,
                                 OffsetT,
                                 CompareOpT,
                                 KeyT,
                                 ValueT>;

  const OffsetT vshmem_offset = blockIdx.x * AgentMergeT::SHARED_MEMORY_SIZE;

  typename AgentMergeT::TempStorage &storage =
    *reinterpret_cast<typename AgentMergeT::TempStorage *>(
      UseVShmem ? vshmem + vshmem_offset : shmem);

  AgentMergeT agent(
    ping,
    storage,
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_ping),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), keys_pong),
    THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(ActivePolicyT(), items_pong),
    keys_count,
    keys_pong,
    items_pong,
    keys_ping,
    items_ping,
    compare_op,
    merge_partitions,
    target_merged_tiles_number);

  agent.Process();
}

/*******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeyIteratorT>
struct DeviceMergeSortPolicy
{
  using KeyT = cub::detail::value_t<KeyIteratorT>;

  //----------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //----------------------------------------------------------------------------

  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_LDG,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

// NVBug 3384810
#if defined(_NVHPC_CUDA)
  using Policy520 = Policy350;
#else
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<512,
                           Nominal4BItemsToItems<KeyT>(15),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_LDG,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };
#endif

  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(17),
                           cub::BLOCK_LOAD_WARP_TRANSPOSE,
                           cub::LOAD_DEFAULT,
                           cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };


  /// MaxPolicy
  using MaxPolicy = Policy600;
};

template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename ChainedPolicyT,
          typename ActivePolicyT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT,
          bool AgentFitsIntoDefaultShmemSize>
struct BlockSortLauncher
{
  int num_tiles;
  std::size_t block_sort_shmem_size;
  bool ping;

  KeyInputIteratorT d_input_keys;
  ValueInputIteratorT d_input_items;
  KeyIteratorT d_output_keys;
  ValueIteratorT d_output_items;
  OffsetT num_items;
  CompareOpT compare_op;
  cudaStream_t stream;

  KeyT *keys_buffer;
  ValueT *items_buffer;
  char* vshmem_ptr;

  CUB_RUNTIME_FUNCTION __forceinline__
  BlockSortLauncher(int num_tiles,
                    std::size_t block_sort_shmem_size,
                    bool ping,
                    KeyInputIteratorT d_input_keys,
                    ValueInputIteratorT d_input_items,
                    KeyIteratorT d_output_keys,
                    ValueIteratorT d_output_items,
                    OffsetT num_items,
                    CompareOpT compare_op,
                    cudaStream_t stream,
                    KeyT *keys_buffer,
                    ValueT *items_buffer,
                    char *vshmem_ptr)
      : num_tiles(num_tiles)
      , block_sort_shmem_size(block_sort_shmem_size)
      , ping(ping)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , keys_buffer(keys_buffer)
      , items_buffer(items_buffer)
      , vshmem_ptr(vshmem_ptr)
  {}

  CUB_RUNTIME_FUNCTION __forceinline__
  void launch() const
  {
    if (vshmem_ptr)
    {
      launch_impl<true>();
    }
    else
    {
      launch_impl<false>();
    }
  }

  template <bool UseVShmem>
  CUB_RUNTIME_FUNCTION __forceinline__ void launch_impl() const
  {
    constexpr bool use_vshmem = (AgentFitsIntoDefaultShmemSize == false) &&
                                UseVShmem;

    THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
      num_tiles,
      ActivePolicyT::MergeSortPolicy::BLOCK_THREADS,
      use_vshmem ? 0 : block_sort_shmem_size,
      stream)
      .doit(DeviceMergeSortBlockSortKernel<use_vshmem,
                                           ChainedPolicyT,
                                           KeyInputIteratorT,
                                           ValueInputIteratorT,
                                           KeyIteratorT,
                                           ValueIteratorT,
                                           OffsetT,
                                           CompareOpT,
                                           KeyT,
                                           ValueT>,
            ping,
            d_input_keys,
            d_input_items,
            d_output_keys,
            d_output_items,
            num_items,
            keys_buffer,
            items_buffer,
            compare_op,
            vshmem_ptr);
  }
};

template <typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename ChainedPolicyT,
          typename ActivePolicyT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT,
          bool AgentFitsIntoDefaultShmemSize>
struct MergeLauncher
{
  int num_tiles;
  std::size_t merge_shmem_size;

  KeyIteratorT d_keys;
  ValueIteratorT d_items;
  OffsetT num_items;
  CompareOpT compare_op;
  OffsetT *merge_partitions;
  cudaStream_t stream;

  KeyT *keys_buffer;
  ValueT *items_buffer;
  char *vshmem_ptr;

  CUB_RUNTIME_FUNCTION __forceinline__ MergeLauncher(int num_tiles,
                                                     std::size_t merge_shmem_size,
                                                     KeyIteratorT d_keys,
                                                     ValueIteratorT d_items,
                                                     OffsetT num_items,
                                                     CompareOpT compare_op,
                                                     OffsetT *merge_partitions,
                                                     cudaStream_t stream,
                                                     KeyT *keys_buffer,
                                                     ValueT *items_buffer,
                                                     char *vshmem_ptr)
      : num_tiles(num_tiles)
      , merge_shmem_size(merge_shmem_size)
      , d_keys(d_keys)
      , d_items(d_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , merge_partitions(merge_partitions)
      , stream(stream)
      , keys_buffer(keys_buffer)
      , items_buffer(items_buffer)
      , vshmem_ptr(vshmem_ptr)
  {}

  CUB_RUNTIME_FUNCTION __forceinline__ void
  launch(bool ping, OffsetT target_merged_tiles_number) const
  {
    if (vshmem_ptr)
    {
      launch_impl<true>(ping, target_merged_tiles_number);
    }
    else
    {
      launch_impl<false>(ping, target_merged_tiles_number);
    }
  }

  template <bool UseVShmem>
  CUB_RUNTIME_FUNCTION __forceinline__ void
  launch_impl(bool ping, OffsetT target_merged_tiles_number) const
  {
    constexpr bool use_vshmem = (AgentFitsIntoDefaultShmemSize == false) &&
                                UseVShmem;

    THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
      num_tiles,
      ActivePolicyT::MergeSortPolicy::BLOCK_THREADS,
      use_vshmem ? 0 : merge_shmem_size,
      stream)
      .doit(DeviceMergeSortMergeKernel<use_vshmem,
                                       ChainedPolicyT,
                                       KeyIteratorT,
                                       ValueIteratorT,
                                       OffsetT,
                                       CompareOpT,
                                       KeyT,
                                       ValueT>,
            ping,
            d_keys,
            d_items,
            num_items,
            keys_buffer,
            items_buffer,
            compare_op,
            merge_partitions,
            target_merged_tiles_number,
            vshmem_ptr);
  }
};

template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename SelectedPolicy = DeviceMergeSortPolicy<KeyIteratorT>>
struct DispatchMergeSort : SelectedPolicy
{
  using KeyT   = cub::detail::value_t<KeyIteratorT>;
  using ValueT = cub::detail::value_t<ValueIteratorT>;

  /// Whether or not there are values to be trucked along with keys
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

  // Problem state

  /// Device-accessible allocation of temporary storage. When NULL, the required
  /// allocation size is written to \p temp_storage_bytes and no work is done.
  void *d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  std::size_t &temp_storage_bytes;

  /// Pointer to the input sequence of unsorted input keys
  KeyInputIteratorT d_input_keys;

  /// Pointer to the input sequence of unsorted input values
  ValueInputIteratorT d_input_items;

  /// Pointer to the output sequence of sorted input keys
  KeyIteratorT d_output_keys;

  /// Pointer to the output sequence of sorted input values
  ValueIteratorT d_output_items;

  /// Number of items to sort
  OffsetT num_items;

  /// Comparison function object which returns true if the first argument is
  /// ordered before the second
  CompareOpT compare_op;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // Constructor
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchMergeSort(void *d_temp_storage,
                    std::size_t &temp_storage_bytes,
                    KeyInputIteratorT d_input_keys,
                    ValueInputIteratorT d_input_items,
                    KeyIteratorT d_output_keys,
                    ValueIteratorT d_output_items,
                    OffsetT num_items,
                    CompareOpT compare_op,
                    cudaStream_t stream,
                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchMergeSort(void *d_temp_storage,
                    std::size_t &temp_storage_bytes,
                    KeyInputIteratorT d_input_keys,
                    ValueInputIteratorT d_input_items,
                    KeyIteratorT d_output_keys,
                    ValueIteratorT d_output_items,
                    OffsetT num_items,
                    CompareOpT compare_op,
                    cudaStream_t stream,
                    bool debug_synchronous,
                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_input_keys(d_input_keys)
      , d_input_items(d_input_items)
      , d_output_keys(d_output_keys)
      , d_output_items(d_output_items)
      , num_items(num_items)
      , compare_op(compare_op)
      , stream(stream)
      , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  // Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using MergePolicyT = typename ActivePolicyT::MergeSortPolicy;
    using MaxPolicyT = typename DispatchMergeSort::MaxPolicy;

    using BlockSortAgentT = AgentBlockSort<MergePolicyT,
                                           KeyInputIteratorT,
                                           ValueInputIteratorT,
                                           KeyIteratorT,
                                           ValueIteratorT,
                                           OffsetT,
                                           CompareOpT,
                                           KeyT,
                                           ValueT>;

    using MergeAgentT = AgentMerge<MergePolicyT,
                                   KeyIteratorT,
                                   ValueIteratorT,
                                   OffsetT,
                                   CompareOpT,
                                   KeyT,
                                   ValueT>;

    cudaError error = cudaSuccess;

    if (num_items == 0)
      return error;

    do
    {
      // Get device ordinal
      int device_ordinal = 0;
      if (CubDebug(error = cudaGetDevice(&device_ordinal)))
      {
        break;
      }

      // Get shared memory size
      const auto tile_size = MergePolicyT::ITEMS_PER_TILE;
      const auto num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

      /**
       * Merge sort supports large types, which can lead to excessive shared
       * memory size requirements. In these cases, merge sort allocates virtual
       * shared memory that resides in global memory:
       * ```
       * extern __shared__ char shmem[];
       * typename AgentT::TempStorage &storage =
       *   *reinterpret_cast<typename AgentT::TempStorage *>(
       *     UseVShmem ? vshmem + vshmem_offset : shmem);
       * ```
       * Having `UseVShmem` as a runtime variable leads to the generation of
       * generic loads and stores, which causes a slowdown. Therefore,
       * `UseVShmem` has to be known at compilation time.
       * In the generic case, available shared memory size is queried at runtime
       * to check if kernels requirements are satisfied. Since the query result
       * is not known at compile-time, merge sort kernels are specialized for
       * both cases.
       * To address increased compilation time, the dispatch layer checks
       * whether kernels requirements fit into default shared memory
       * size (48KB). In this case, there's no need for virtual shared
       * memory specialization.
       */
      constexpr std::size_t default_shared_memory_size = 48 * 1024;
      constexpr auto block_sort_shmem_size =
        static_cast<std::size_t>(BlockSortAgentT::SHARED_MEMORY_SIZE);
      constexpr bool block_sort_fits_into_default_shmem =
        block_sort_shmem_size < default_shared_memory_size;

      constexpr auto merge_shmem_size =
        static_cast<std::size_t>(MergeAgentT::SHARED_MEMORY_SIZE);
      constexpr bool merge_fits_into_default_shmem = merge_shmem_size <
                                                     default_shared_memory_size;
      constexpr bool runtime_shmem_size_check_is_required =
        !(merge_fits_into_default_shmem && block_sort_fits_into_default_shmem);

      const auto merge_partitions_size =
        static_cast<std::size_t>(1 + num_tiles) * sizeof(OffsetT);

      const auto temporary_keys_storage_size =
        static_cast<std::size_t>(num_items * sizeof(KeyT));

      const auto temporary_values_storage_size =
        static_cast<std::size_t>(num_items * sizeof(ValueT)) * !KEYS_ONLY;

      std::size_t virtual_shared_memory_size = 0;
      bool block_sort_requires_vshmem = false;
      bool merge_requires_vshmem = false;

      if (runtime_shmem_size_check_is_required)
      {
        int max_shmem = 0;
        if (CubDebug(
              error = cudaDeviceGetAttribute(&max_shmem,
                                             cudaDevAttrMaxSharedMemoryPerBlock,
                                             device_ordinal)))
        {
          break;
        }

        block_sort_requires_vshmem = block_sort_shmem_size >
                                     static_cast<std::size_t>(max_shmem);
        merge_requires_vshmem = merge_shmem_size >
                                static_cast<std::size_t>(max_shmem);

        virtual_shared_memory_size =
          detail::VshmemSize(static_cast<std::size_t>(max_shmem),
                     (cub::max)(block_sort_shmem_size, merge_shmem_size),
                     static_cast<std::size_t>(num_tiles));
      }


      void *allocations[4] = {nullptr, nullptr, nullptr, nullptr};
      std::size_t allocation_sizes[4] = {merge_partitions_size,
                                         temporary_keys_storage_size,
                                         temporary_values_storage_size,
                                         virtual_shared_memory_size};

      if (CubDebug(error = AliasTemporaries(d_temp_storage,
                                            temp_storage_bytes,
                                            allocations,
                                            allocation_sizes)))
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      const int num_passes =
        static_cast<int>(THRUST_NS_QUALIFIER::detail::log2_ri(num_tiles));

      /*
       * The algorithm consists of stages. At each stage, there are input and
       * output arrays. There are two pairs of arrays allocated (keys and items).
       * One pair is from function arguments and another from temporary storage.
       * Ping is a helper variable that controls which of these two pairs of
       * arrays is an input and which is an output for a current stage. If the
       * ping is true - the current stage stores its result in the temporary
       * storage. The temporary storage acts as input data otherwise.
       *
       * Block sort is executed before the main loop. It stores its result in
       * the pair of arrays that will be an input of the next stage. The initial
       * value of the ping variable is selected so that the result of the final
       * stage is stored in the input arrays.
       */
      bool ping = num_passes % 2 == 0;

      auto merge_partitions = reinterpret_cast<OffsetT *>(allocations[0]);
      auto keys_buffer      = reinterpret_cast<KeyT *>(allocations[1]);
      auto items_buffer     = reinterpret_cast<ValueT *>(allocations[2]);

      char *vshmem_ptr = virtual_shared_memory_size > 0
                       ? reinterpret_cast<char *>(allocations[3])
                       : nullptr;

      // Invoke DeviceReduceKernel
      BlockSortLauncher<KeyInputIteratorT,
                        ValueInputIteratorT,
                        KeyIteratorT,
                        ValueIteratorT,
                        OffsetT,
                        MaxPolicyT,
                        ActivePolicyT,
                        CompareOpT,
                        KeyT,
                        ValueT,
                        block_sort_fits_into_default_shmem>
        block_sort_launcher(static_cast<int>(num_tiles),
                            block_sort_shmem_size,
                            ping,
                            d_input_keys,
                            d_input_items,
                            d_output_keys,
                            d_output_items,
                            num_items,
                            compare_op,
                            stream,
                            keys_buffer,
                            items_buffer,
                            block_sort_requires_vshmem ? vshmem_ptr : nullptr);

      block_sort_launcher.launch();

      error = detail::DebugSyncStream(stream);
      if (CubDebug(error))
      {
        break;
      }

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      const OffsetT num_partitions = num_tiles + 1;
      const int threads_per_partition_block = 256;

      const int partition_grid_size = static_cast<int>(
        cub::DivideAndRoundUp(num_partitions, threads_per_partition_block));

      MergeLauncher<KeyIteratorT,
                    ValueIteratorT,
                    OffsetT,
                    MaxPolicyT,
                    ActivePolicyT,
                    CompareOpT,
                    KeyT,
                    ValueT,
                    merge_fits_into_default_shmem>
        merge_launcher(static_cast<int>(num_tiles),
                       merge_shmem_size,
                       d_output_keys,
                       d_output_items,
                       num_items,
                       compare_op,
                       merge_partitions,
                       stream,
                       keys_buffer,
                       items_buffer,
                       merge_requires_vshmem ? vshmem_ptr : nullptr);

      for (int pass = 0; pass < num_passes; ++pass, ping = !ping)
      {
        OffsetT target_merged_tiles_number = OffsetT(2) << pass;

        // Partition
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          partition_grid_size,
          threads_per_partition_block,
          0,
          stream)
          .doit(DeviceMergeSortPartitionKernel<KeyIteratorT,
                                               OffsetT,
                                               CompareOpT,
                                               KeyT>,
                ping,
                d_output_keys,
                keys_buffer,
                num_items,
                num_partitions,
                merge_partitions,
                compare_op,
                target_merged_tiles_number,
                tile_size);

        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }

        // Merge
        merge_launcher.launch(ping, target_merged_tiles_number);

        error = detail::DebugSyncStream(stream);
        if (CubDebug(error))
        {
          break;
        }

        // Check for failure to launch
        if (CubDebug(error = cudaPeekAtLastError()))
        {
          break;
        }
      }
    }
    while (0);

    return error;
  }

  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           KeyInputIteratorT d_input_keys,
           ValueInputIteratorT d_input_items,
           KeyIteratorT d_output_keys,
           ValueIteratorT d_output_items,
           OffsetT num_items,
           CompareOpT compare_op,
           cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchMergeSort::MaxPolicy;

    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Create dispatch functor
      DispatchMergeSort dispatch(d_temp_storage,
                                 temp_storage_bytes,
                                 d_input_keys,
                                 d_input_items,
                                 d_output_keys,
                                 d_output_items,
                                 num_items,
                                 compare_op,
                                 stream,
                                 ptx_version);

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
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

    return Dispatch(d_temp_storage,
                    temp_storage_bytes,
                    d_input_keys,
                    d_input_items,
                    d_output_keys,
                    d_output_items,
                    num_items,
                    compare_op,
                    stream);
  }
};


CUB_NAMESPACE_END
