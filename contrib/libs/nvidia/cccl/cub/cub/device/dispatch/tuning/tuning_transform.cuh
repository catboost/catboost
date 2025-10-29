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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/cmath>
#include <cuda/numeric>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/bit>

CUB_NAMESPACE_BEGIN

namespace detail::transform
{
enum class Algorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch,
  vectorized,
  memcpy_async,
  ublkcp
};

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  static constexpr int min_items_per_thread      = 1;
  static constexpr int max_items_per_thread      = 32;

  // TODO: remove with C++20
  // The value of the below does not matter.
  static constexpr int not_a_vectorized_policy = 0;
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentPrefetchPolicy,
  (always_true),
  (block_threads, BlockThreads, int),
  (items_per_thread_no_input, ItemsPerThreadNoInput, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (not_a_vectorized_policy, NotAVectorizedPolicy, int) ) // TODO: remove with C++20

template <int BlockThreads, int ItemsPerThread, int LoadStoreWordSize>
struct vectorized_policy_t : prefetch_policy_t<BlockThreads>
{
  static constexpr int items_per_thread_vectorized = ItemsPerThread;
  static constexpr int load_store_word_size        = LoadStoreWordSize;

  using not_a_vectorized_policy = void; // TODO: remove with C++20, shadows the variable in prefetch_policy_t
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentVectorizedPolicy,
  (always_true), // TODO: restore with C++20: (TransformAgentPrefetchPolicy),
  (block_threads, BlockThreads, int),
  (items_per_thread_no_input, ItemsPerThreadNoInput, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (items_per_thread_vectorized, ItemsPerThreadVectorized, int),
  (load_store_word_size, LoadStoreWordSize, int) )

template <int BlockThreads, int BulkCopyAlignment>
struct async_copy_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int min_items_per_thread = 1;
  static constexpr int max_items_per_thread = 32;

  static constexpr int bulk_copy_alignment = BulkCopyAlignment;
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentAsyncPolicy,
  (always_true),
  (block_threads, BlockThreads, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (bulk_copy_alignment, BulkCopyAlignment, int) )

_CCCL_TEMPLATE(typename PolicyT)
_CCCL_REQUIRES((!TransformAgentPrefetchPolicy<PolicyT> && !TransformAgentAsyncPolicy<PolicyT>
                && !TransformAgentVectorizedPolicy<PolicyT>) )
__host__ __device__ constexpr PolicyT MakePolicyWrapper(PolicyT policy)
{
  return policy;
}

template <typename PolicyT, typename = void>
struct TransformPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct TransformPolicyWrapper<
  StaticPolicyT,
  ::cuda::std::
    void_t<decltype(StaticPolicyT::algorithm), decltype(StaticPolicyT::min_bif), typename StaticPolicyT::algo_policy>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr Algorithm Algorithm()
  {
    return StaticPolicyT::algorithm;
  }

  _CCCL_HOST_DEVICE static constexpr int MinBif()
  {
    return StaticPolicyT::min_bif;
  }

  _CCCL_HOST_DEVICE static constexpr auto AlgorithmPolicy()
  {
    return MakePolicyWrapper(typename StaticPolicyT::algo_policy());
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"min_bif">()     = value<StaticPolicyT::min_bif>(),
                  key<"algorithm">()   = value<static_cast<int>(StaticPolicyT::algorithm)>(),
                  key<"algo_policy">() = AlgorithmPolicy().EncodedPolicy()>();
  }
#endif // CUB_ENABLE_POLICY_PTX_JSON
};

template <typename PolicyT>
_CCCL_HOST_DEVICE TransformPolicyWrapper<PolicyT> MakeTransformPolicyWrapper(PolicyT base)
{
  return TransformPolicyWrapper<PolicyT>(base);
}

template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(it_value_t<Its>)} + ... + 0);
}

constexpr int ldgsts_size_and_align = 16;

template <typename ItValueSizesAlignments>
_CCCL_HOST_DEVICE constexpr auto memcpy_async_dyn_smem_for_tile_size(
  ItValueSizesAlignments it_value_sizes_alignments, int tile_size, int copy_alignment = ldgsts_size_and_align) -> int
{
  int smem_size = 0;
  for (auto&& [vt_size, vt_alignment] : it_value_sizes_alignments)
  {
    smem_size =
      static_cast<int>(::cuda::round_up(smem_size, ::cuda::std::max(static_cast<int>(vt_alignment), copy_alignment)));
    // max head/tail padding is copy_alignment - sizeof(T) each
    const int max_bytes_to_copy =
      static_cast<int>(vt_size) * tile_size + ::cuda::std::max(copy_alignment - static_cast<int>(vt_size), 0) * 2;
    smem_size += max_bytes_to_copy;
  };
  return smem_size;
}

constexpr int bulk_copy_size_multiple = 16;

_CCCL_HOST_DEVICE constexpr auto bulk_copy_alignment(int sm_arch) -> int
{
  return sm_arch < 1000 ? 128 : 16;
}

template <typename ItValueSizesAlignments>
_CCCL_HOST_DEVICE constexpr auto
bulk_copy_dyn_smem_for_tile_size(ItValueSizesAlignments it_value_sizes_alignments, int tile_size, int bulk_copy_align)
  -> int
{
  // we rely on the tile_size being a multiple of alignments, so shifting offsets/pointers by it retains alignments
  _CCCL_ASSERT(tile_size % bulk_copy_align == 0, "");
  _CCCL_ASSERT(tile_size % bulk_copy_size_multiple == 0, "");

  int tile_padding = bulk_copy_align;
  for (auto&& [_, vt_alignment] : it_value_sizes_alignments)
  {
    tile_padding = ::cuda::std::max(tile_padding, static_cast<int>(vt_alignment));
  }

  int smem_size = tile_padding; // for the barrier and padding
  for (auto&& [vt_size, _] : it_value_sizes_alignments)
  {
    smem_size += tile_padding + static_cast<int>(vt_size) * tile_size;
  }
  return smem_size;
}

_CCCL_HOST_DEVICE constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  if (sm_arch >= 1000)
  {
    return 64 * 1024; // B200
  }
  if (sm_arch >= 900)
  {
    return 48 * 1024; // 32 for H100, 48 for H200
  }
  if (sm_arch >= 800)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024; // V100 and below
}

template <typename T, typename... Ts>
_CCCL_HOST_DEVICE constexpr bool all_equal([[maybe_unused]] T head, Ts... tail)
{
  return ((head == tail) && ...);
}

_CCCL_HOST_DEVICE constexpr bool all_equal()
{
  return true;
}

template <typename T, typename... Ts>
_CCCL_HOST_DEVICE constexpr auto first_item(T head, Ts...) -> T
{
  return head;
}

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE static constexpr auto make_sizes_alignments()
{
  return ::cuda::std::array<::cuda::std::pair<::cuda::std::size_t, ::cuda::std::size_t>,
                            sizeof...(RandomAccessIteratorsIn)>{
    {{sizeof(it_value_t<RandomAccessIteratorsIn>), alignof(it_value_t<RandomAccessIteratorsIn>)}...}};
}

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn, typename RandomAccessIteratorOut>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut>
{
  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_inputs_contiguous =
    (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorsIn> && ...);
  static constexpr bool all_input_values_trivially_reloc =
    (THRUST_NS_QUALIFIER::is_trivially_relocatable_v<it_value_t<RandomAccessIteratorsIn>> && ...);
  static constexpr bool can_memcpy_inputs = all_inputs_contiguous && all_input_values_trivially_reloc;

  // for vectorized policy:
  static constexpr bool all_input_values_same_size = all_equal(sizeof(it_value_t<RandomAccessIteratorsIn>)...);
  static constexpr int load_store_word_size        = 8; // TODO(bgruber): make this 16, and 32 on Blackwell+
  static constexpr int value_type_size             = first_item(int{sizeof(it_value_t<RandomAccessIteratorsIn>)}..., 1);
  static constexpr bool value_type_divides_load_store_size =
    load_store_word_size % value_type_size == 0; // implicitly checks that value_type_size <= load_store_word_size
  static constexpr int target_bytes_per_thread = 32; // guestimate by gevtushenko
  static constexpr int items_per_thread_vec =
    ::cuda::round_up(target_bytes_per_thread, load_store_word_size) / value_type_size;
  using default_vectorized_policy_t = vectorized_policy_t<256, items_per_thread_vec, load_store_word_size>;

  // TODO(bgruber): consider a separate kernel for just filling

  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif       = arch_to_min_bytes_in_flight(300);
    static constexpr bool use_fallback = RequiresStableAddress || !can_memcpy_inputs || no_input_streams
                                      || !all_input_values_same_size || !value_type_divides_load_store_size;
    // TODO(bgruber): we don't need algo, because we can just detect the type of algo_policy
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::vectorized;
    using algo_policy = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, default_vectorized_policy_t>;
  };

  struct policy800 : ChainedPolicy<800, policy800, policy300>
  {
  private:
    static constexpr int block_threads = 256;
    using async_policy                 = async_copy_policy_t<block_threads, ldgsts_size_and_align>;
    // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
    // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
    // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
    static constexpr bool exhaust_smem =
      memcpy_async_dyn_smem_for_tile_size(
        make_sizes_alignments<RandomAccessIteratorsIn...>(),
        block_threads* async_policy::min_items_per_thread,
        ldgsts_size_and_align)
      > int{max_smem_per_block};
    static constexpr bool use_fallback =
      RequiresStableAddress || !can_memcpy_inputs || no_input_streams || exhaust_smem;

  public:
    static constexpr int min_bif    = arch_to_min_bytes_in_flight(800);
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::memcpy_async;
    using algo_policy               = ::cuda::std::_If<use_fallback, prefetch_policy_t<block_threads>, async_policy>;
  };

  template <int AsyncBlockSize, int PtxVersion>
  struct bulk_copy_policy_base
  {
  private:
    static constexpr int alignment = bulk_copy_alignment(PtxVersion);
    using async_policy             = async_copy_policy_t<AsyncBlockSize, alignment>;
    // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
    // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
    // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
    static constexpr bool exhaust_smem =
      bulk_copy_dyn_smem_for_tile_size(
        make_sizes_alignments<RandomAccessIteratorsIn...>(),
        AsyncBlockSize* async_policy::min_items_per_thread,
        alignment)
      > int{max_smem_per_block};

    // if each tile size is a multiple of the bulk copy and maximum value type alignments, the alignment is retained if
    // the base pointer is sufficiently aligned (the correct check would be if it's a multiple of all value types
    // following the current tile). we would otherwise need to realign every SMEM tile individually, which is costly and
    // complex, so let's fall back in this case.
    static constexpr int max_alignment =
      ::cuda::std::max({alignment, int{alignof(it_value_t<RandomAccessIteratorsIn>)}...});
    static constexpr int tile_sizes_retain_alignment =
      (((int{sizeof(it_value_t<RandomAccessIteratorsIn>)} * AsyncBlockSize) % max_alignment == 0) && ...);

    static constexpr bool use_fallback =
      RequiresStableAddress || !can_memcpy_inputs || no_input_streams || exhaust_smem || !tile_sizes_retain_alignment;

  public:
    static constexpr int min_bif    = arch_to_min_bytes_in_flight(PtxVersion);
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::ublkcp;
    using algo_policy               = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, async_policy>;
  };

  struct policy900
      : bulk_copy_policy_base<256, 900>
      , ChainedPolicy<900, policy900, policy800>
  {};

  struct policy1000
      : bulk_copy_policy_base<128, 1000>
      , ChainedPolicy<1000, policy1000, policy900>
  {};

  using max_policy = policy1000;
};

} // namespace detail::transform

CUB_NAMESPACE_END
