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

#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/util_type.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__barrier/aligned_size.h> // cannot include <cuda/barrier> directly on CUDA_ARCH < 700
#include <cuda/cmath>
#include <cuda/memory>
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/expected>

CUB_NAMESPACE_BEGIN

namespace detail::transform
{

template <typename T>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE const char* round_down_ptr(const T* ptr, unsigned alignment)
{
  _CCCL_ASSERT(::cuda::std::has_single_bit(alignment), "");
  return reinterpret_cast<const char*>(
    reinterpret_cast<::cuda::std::uintptr_t>(ptr) & ~::cuda::std::uintptr_t{alignment - 1});
}

// Prefetches (at least on Hopper) a 128 byte cache line. Prefetching out-of-bounds addresses has no side effects
// TODO(bgruber): there is also the cp.async.bulk.prefetch instruction available on Hopper. May improve perf a tiny bit
// as we need to create less instructions to prefetch the same amount of data.
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch(const T* addr)
{
  // TODO(bgruber): prefetch to L1 may be even better
  asm volatile("prefetch.global.L2 [%0];" : : "l"(__cvta_generic_to_global(addr)) : "memory");
}

template <int BlockDim, typename It>
_CCCL_DEVICE _CCCL_FORCEINLINE void prefetch_tile(It begin, int items)
{
  if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>)
  {
    constexpr int prefetch_byte_stride = 128; // TODO(bgruber): should correspond to cache line size. Does this need to
                                              // be architecture dependent?
    const int items_bytes = items * sizeof(it_value_t<It>);

    // prefetch does not stall and unrolling just generates a lot of unnecessary computations and predicate handling
    _CCCL_PRAGMA_NOUNROLL()
    for (int offset = threadIdx.x * prefetch_byte_stride; offset < items_bytes;
         offset += BlockDim * prefetch_byte_stride)
    {
      prefetch(reinterpret_cast<const char*>(::cuda::std::to_address(begin)) + offset);
    }
  }
}

// This kernel guarantees that objects passed as arguments to the user-provided transformation function f reside in
// global memory. No intermediate copies are taken. If the parameter type of f is a reference, taking the address of the
// parameter yields a global memory address.
template <typename PrefetchPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteratorIn>
_CCCL_DEVICE void transform_kernel_prefetch(
  Offset num_items, int num_elem_per_thread, F f, RandomAccessIteratorOut out, RandomAccessIteratorIn... ins)
{
  constexpr int block_threads = PrefetchPolicy::block_threads;
  const int tile_size         = block_threads * num_elem_per_thread;
  const Offset offset         = static_cast<Offset>(blockIdx.x) * tile_size;
  const int valid_items       = static_cast<int>((::cuda::std::min) (num_items - offset, Offset{tile_size}));

  // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
  {
    (..., (ins += offset));
    out += offset;
  }

  (..., prefetch_tile<block_threads>(ins, valid_items));

  auto process_tile = [&](auto full_tile, auto... ins2 /* nvcc fails to compile when just using the captured ins */) {
    // ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost
    // bgruber: but A6000 and H100 show small gains without pragma
    // _CCCL_PRAGMA_NOUNROLL()
    for (int j = 0; j < num_elem_per_thread; ++j)
    {
      const int idx = j * block_threads + threadIdx.x;
      if (full_tile || idx < valid_items)
      {
        // we have to unwrap Thrust's proxy references here for backward compatibility (try zip_iterator.cu test)
        out[idx] = f(THRUST_NS_QUALIFIER::raw_reference_cast(ins2[idx])...);
      }
    }
  };
  if (tile_size == valid_items)
  {
    process_tile(::cuda::std::true_type{}, ins...);
  }
  else
  {
    process_tile(::cuda::std::false_type{}, ins...);
  }
}

#if _CCCL_CTK_BELOW(13, 0)
struct alignas(32) aligned32_t
{
  longlong4 data;
};
#endif // _CCCL_CTK_BELOW(13, 0)

template <int Bytes>
_CCCL_HOST_DEVICE _CCCL_CONSTEVAL auto load_store_type()
{
  static_assert(::cuda::is_power_of_two(Bytes));
  if constexpr (Bytes == 1)
  {
    return ::cuda::std::int8_t{};
  }
  else if constexpr (Bytes == 2)
  {
    return ::cuda::std::int16_t{};
  }
  else if constexpr (Bytes == 4)
  {
    return ::cuda::std::int32_t{};
  }
  else if constexpr (Bytes == 8)
  {
    return ::cuda::std::int64_t{};
  }
  else if constexpr (Bytes == 16)
  {
    static_assert(alignof(int4) == 16);
    return int4{};
  }
  else if constexpr (Bytes == 32)
  {
#if _CCCL_CTK_BELOW(13, 0)
    static_assert(alignof(aligned32_t) == 32);
    return aligned32_t{};
#else // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^ / vvv _CCCL_CTK_AT_LEAST(13, 0) vvv
    return longlong4_32a{};
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  }
  else
  {
    return ::cuda::std::array<int, Bytes / sizeof(int)>{};
  }
}

template <typename T>
inline constexpr size_t size_of = sizeof(T);

template <>
inline constexpr size_t size_of<void> = 0;

template <typename VectorizedPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InputT>
_CCCL_DEVICE void transform_kernel_vectorized(
  Offset num_items,
  int num_elem_per_thread_prefetch,
  bool can_vectorize,
  F f,
  RandomAccessIteratorOut out,
  const InputT*... ins)
{
  constexpr int block_dim        = VectorizedPolicy::block_threads;
  constexpr int items_per_thread = VectorizedPolicy::items_per_thread_vectorized;
  _CCCL_ASSERT(!can_vectorize || (items_per_thread == num_elem_per_thread_prefetch), "");
  constexpr int tile_size = block_dim * items_per_thread;
  const Offset offset     = static_cast<Offset>(blockIdx.x) * tile_size;
  const int valid_items   = static_cast<int>((::cuda::std::min) (num_items - offset, Offset{tile_size}));

  // if we cannot vectorize or don't have a full tile, fall back to prefetch kernel
  if (!can_vectorize || valid_items != tile_size)
  {
    transform_kernel_prefetch<VectorizedPolicy>(
      num_items, num_elem_per_thread_prefetch, ::cuda::std::move(f), ::cuda::std::move(out), ins...);
    return;
  }

  // move index and iterator domain to the block/thread index, to reduce arithmetic in the loops below
  {
    (..., (ins += offset));
    out += offset;
  }

  constexpr int load_store_size  = VectorizedPolicy::load_store_word_size;
  using load_store_t             = decltype(load_store_type<load_store_size>());
  using result_t                 = ::cuda::std::invoke_result_t<F, const InputT&...>;
  using output_t                 = it_value_t<RandomAccessIteratorOut>;
  constexpr int input_type_size  = int{first_item(sizeof(InputT)...)};
  constexpr int load_store_count = (items_per_thread * input_type_size) / load_store_size;
  static_assert((items_per_thread * input_type_size) % load_store_size == 0);
  static_assert(load_store_size % input_type_size == 0);

  constexpr bool can_vectorize_store =
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorOut>
    && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<output_t> && size_of<output_t> == input_type_size;

  // if we can vectorize, we convert f's return type to the output type right away, so we can reinterpret later
  using THRUST_NS_QUALIFIER::cuda_cub::core::detail::uninitialized_array;
  uninitialized_array<::cuda::std::conditional_t<can_vectorize_store, output_t, result_t>, items_per_thread> output;

  auto provide_array = [&](auto... inputs) {
    // load inputs
    // TODO(bgruber): we could support fancy iterators for loading here as well (and only vectorize some inputs)
    [[maybe_unused]] auto load_tile_vectorized = [&](auto* in, auto& input) {
      auto in_vec    = reinterpret_cast<const load_store_t*>(in);
      auto input_vec = reinterpret_cast<load_store_t*>(input.data());
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < load_store_count; ++i)
      {
        input_vec[i] = in_vec[i * VectorizedPolicy::block_threads + threadIdx.x];
      }
    };
    (load_tile_vectorized(ins, inputs), ...);

    // process
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      output[i] = f(inputs[i]...);
    }
  };
  provide_array(uninitialized_array<InputT, items_per_thread>{}...);

  // write output
  if constexpr (can_vectorize_store)
  {
    // vector path
    auto output_vec = reinterpret_cast<const load_store_t*>(output.data());
    auto out_vec    = reinterpret_cast<load_store_t*>(out) + threadIdx.x;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < load_store_count; ++i)
    {
      out_vec[i * VectorizedPolicy::block_threads] = output_vec[i];
    }
  }
  else
  {
    // serial path
    constexpr int elems = load_store_size / input_type_size;
    out += threadIdx.x * elems;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < load_store_count; ++i)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < elems; ++j)
      {
        out[i * elems * VectorizedPolicy::block_threads + j] = output[i * elems + j];
      }
    }
  }
}

// Implementation notes on memcpy_async and UBLKCP kernels regarding copy alignment and padding
//
// For performance considerations of memcpy_async:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidance-for-memcpy-async
//
// We basically have to align the base pointer to 16 bytes, and copy a multiple of 16 bytes. To achieve this, when we
// copy a tile of data from an input buffer, we round down the pointer to the start of the tile to the next lower
// address that is a multiple of 16 bytes. This introduces head padding. We also round up the total number of bytes to
// copy (including head padding) to a multiple of 16 bytes, which introduces tail padding. For the bulk copy kernel, we
// should align to 128 bytes instead of 16 on Hopper.
//
// However, padding memory copies like that may access the input buffer out-of-bounds. Here are some thoughts:
// * According to the CUDA programming guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses), "any address of a variable
// residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is
// always aligned to at least 256 bytes."
// * Memory protection is usually done on memory page level, which is even larger than 256 bytes for CUDA and 4KiB on
// Intel x86 and 4KiB+ ARM. Front and tail padding thus never leaves the memory page of the input buffer.
// * This should count for device memory, but also for device accessible memory living on the host.
// * The base pointer alignment and size rounding also never leaves the size of a cache line.
//
// Copying larger data blocks with head and tail padding should thus be legal. Nevertheless, an out-of-bounds read is
// still technically undefined behavior in C++. Also, compute-sanitizer flags at least such reads after the end of a
// buffer. Therefore, we lean on the safer side and protect against out of bounds reads at the beginning and end.

// A note on size and alignment: The size of a type is at least as large as its alignment. We rely on this fact in some
// conditions.
// This is guaranteed by the C++ standard, and follows from the definition of arrays: the difference between neighboring
// array element addresses is sizeof element type and each array element needs to fulfill the alignment requirement of
// the element type.

// Pointer with metadata to describe readonly input memory for memcpy_async and UBLKCP kernels.
// LDGSTS is most efficient when the data is 16-byte aligned and the size a multiple of 16 bytes
// UBLKCP is most efficient when the data is 128/16-byte aligned (Hopper/Blackwell) and the size a multiple of 16 bytes
template <typename T> // Cannot add alignment to signature, because we need a uniform kernel template instantiation
struct aligned_base_ptr
{
  using value_type = T;

  const char* ptr; // aligned pointer before the original pointer (16-byte or 128-byte). May not be aligned to
                   // alignof(T). E.g.: array of int3 starting at address 4, ptr == 0
  int head_padding; // byte offset between ptr and the original pointer. Value inside [0;15] or [0;127].

  _CCCL_HOST_DEVICE const T* ptr_to_elements() const
  {
    return reinterpret_cast<const T*>(ptr + head_padding);
  }

  _CCCL_HOST_DEVICE friend bool operator==(const aligned_base_ptr& a, const aligned_base_ptr& b)
  {
    return a.ptr == b.ptr && a.head_padding == b.head_padding;
  }
};

template <typename T>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr(const T* ptr, int alignment) -> aligned_base_ptr<T>
{
  const char* base_ptr = round_down_ptr(ptr, alignment);
  return aligned_base_ptr<T>{base_ptr, static_cast<int>(reinterpret_cast<const char*>(ptr) - base_ptr)};
}

template <int BlockThreads>
_CCCL_DEVICE void memcpy_async_aligned(void* dst, const void* src, unsigned int bytes_to_copy)
{
  _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(src) % ldgsts_size_and_align == 0, "");
  _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(dst) % ldgsts_size_and_align == 0, "");
  _CCCL_ASSERT(bytes_to_copy % ldgsts_size_and_align == 0, "");

  // allowing unrolling generates a LOT more instructions and is usually slower (confirmed by benchmark)
  _CCCL_PRAGMA_NOUNROLL()
  for (unsigned int offset = threadIdx.x * ldgsts_size_and_align; offset < bytes_to_copy;
       offset += BlockThreads * ldgsts_size_and_align)
  {
    asm volatile(
      "cp.async.cg.shared.global [%0], [%1], %2, %3;"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(static_cast<char*>(dst) + offset))),
        "l"(static_cast<const char*>(src) + offset),
        "n"(ldgsts_size_and_align),
        "n"(ldgsts_size_and_align)
      : "memory");
    // same as: __pipeline_memcpy_async(static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset,
    //                                  ldgsts_size_and_align);
  }

  asm volatile("cp.async.commit_group;"); // same as: __pipeline_commit();
}

template <int BlockThreads>
_CCCL_DEVICE void memcpy_async_maybe_unaligned(void* dst, const void* src, unsigned int bytes_to_copy, int head_padding)
{
  // early exiting if (head_padding == 0 && bytes_to_copy % ldgsts_size_and_align == 0) does not yield a benefit

  const char* src_ptr = static_cast<const char*>(src);
  char* dst_ptr       = static_cast<char*>(dst);

  // handle tiny copies to simplify head/tail bytes computations below
  if (bytes_to_copy < ldgsts_size_and_align)
  {
    if (threadIdx.x < bytes_to_copy)
    {
      dst_ptr[threadIdx.x] = src_ptr[threadIdx.x];
    }
    return;
  }

  const unsigned int head_bytes = (ldgsts_size_and_align - head_padding) % ldgsts_size_and_align;
  const unsigned int tail_bytes = (bytes_to_copy - head_bytes) % ldgsts_size_and_align;

  // pipeline the async copies before loading the head and tail elements
  _CCCL_ASSERT(bytes_to_copy >= (head_bytes + tail_bytes), "");
  const unsigned int aligned_bytes_to_copy = bytes_to_copy - head_bytes - tail_bytes;
  if (aligned_bytes_to_copy > 0)
  {
    _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(dst_ptr + head_bytes) % ldgsts_size_and_align == 0, "");
    _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(src_ptr + head_bytes) % ldgsts_size_and_align == 0, "");
    _CCCL_ASSERT(aligned_bytes_to_copy % ldgsts_size_and_align == 0, "");
    memcpy_async_aligned<BlockThreads>(dst_ptr + head_bytes, src_ptr + head_bytes, aligned_bytes_to_copy);
  }

  // TODO(bgruber): ahendriksen suggested to copy elements instead of bytes, but it generates about 20 instructions more
  // ahendriksen: we perform both loads first and then both writes. this reduces the total latency
  char head_byte, tail_byte;
  if (threadIdx.x < head_bytes)
  {
    head_byte = src_ptr[threadIdx.x];
  }
  if (threadIdx.x < tail_bytes)
  {
    tail_byte = src_ptr[bytes_to_copy - tail_bytes + threadIdx.x];
  }
  if (threadIdx.x < head_bytes)
  {
    dst_ptr[threadIdx.x] = head_byte;
  }
  if (threadIdx.x < tail_bytes)
  {
    dst_ptr[bytes_to_copy - tail_bytes + threadIdx.x] = tail_byte;
  }
}

// Turning this function into a lambda will make nvcc generate it once for each iterator instead of for each distinct
// value type (which may be less).
template <int BlockThreads, typename AlignedPtr, typename Offset>
_CCCL_DEVICE auto
copy_and_return_smem_dst(AlignedPtr aligned_ptr, int& smem_offset, Offset offset, char* smem, int valid_items)
{
  using T = typename decltype(aligned_ptr)::value_type;
  // because SMEM base pointer and bytes_to_copy are always multiples of ldgsts_size_and_align, we only need to align
  // the SMEM start for types with larger alignment
  _CCCL_ASSERT(smem_offset % ldgsts_size_and_align == 0, "");
  if constexpr (alignof(T) > ldgsts_size_and_align)
  {
    smem_offset = ::cuda::round_up(smem_offset, int{alignof(T)});
  }
  const char* const src = aligned_ptr.ptr + offset * unsigned{sizeof(T)}; // compute expression in U32 if Offset==I32
  char* const dst       = smem + smem_offset;
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % ldgsts_size_and_align == 0, "");
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % ldgsts_size_and_align == 0, "");

  int bytes_to_copy;
  if constexpr (alignof(T) < ldgsts_size_and_align)
  {
    bytes_to_copy = ::cuda::round_up(aligned_ptr.head_padding + int{sizeof(T)} * valid_items, ldgsts_size_and_align);
  }
  else
  {
    _CCCL_ASSERT(aligned_ptr.head_padding == 0, "");
    bytes_to_copy = int{sizeof(T)} * valid_items;
  }

  smem_offset += bytes_to_copy; // leaves aligned address for follow-up copy
  memcpy_async_aligned<BlockThreads>(dst, src, bytes_to_copy);
  const char* const dst_start_of_data = dst + (alignof(T) < ldgsts_size_and_align ? aligned_ptr.head_padding : 0);
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst_start_of_data) % alignof(T) == 0, "");
  return reinterpret_cast<const T*>(dst_start_of_data);
}

template <int BlockThreads, typename AlignedPtr, typename Offset>
_CCCL_DEVICE auto copy_and_return_smem_dst_fallback(
  AlignedPtr aligned_ptr, int& smem_offset, Offset offset, char* smem, int valid_items, int tile_size)
{
  // TODO(bgruber): drop handling of head bytes and just read OOB, since gmem buffers are always sufficiently aligned

  using T = typename decltype(aligned_ptr)::value_type;
  // because SMEM base pointer and tile_size are always multiples of 16-byte, we only need to align the SMEM start
  // for types with larger alignment
  _CCCL_ASSERT(tile_size % ldgsts_size_and_align == 0, "");
  _CCCL_ASSERT(smem_offset % ldgsts_size_and_align == 0, "");
  if constexpr (alignof(T) > ldgsts_size_and_align)
  {
    smem_offset = ::cuda::round_up(smem_offset, int{alignof(T)});
  }
  _CCCL_ASSERT(alignof(T) < ldgsts_size_and_align || aligned_ptr.head_padding == 0, "");
  const int head_padding = alignof(T) < ldgsts_size_and_align ? aligned_ptr.head_padding : 0;

  const char* src = aligned_ptr.ptr + offset * unsigned{sizeof(T)} + head_padding; // compute expression in U32 if
                                                                                   // Offset==I32
  char* dst = smem + smem_offset + head_padding;
  _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(src) % alignof(T) == 0, "");
  _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(dst) % alignof(T) == 0, "");
  const int bytes_to_copy = int{sizeof(T)} * valid_items;
  memcpy_async_maybe_unaligned<BlockThreads>(dst, src, bytes_to_copy, head_padding);

  // add ldgsts_size_and_align to account for this tile's head padding
  smem_offset += ldgsts_size_and_align + int{sizeof(T)} * tile_size;

  return reinterpret_cast<T*>(dst);
}

template <typename LdgstsPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_ldgsts(
  Offset num_items, int num_elem_per_thread, F f, RandomAccessIteratorOut out, aligned_base_ptr<InTs>... aligned_ptrs)
{
  // SMEM is 16-byte aligned by default
  extern __shared__ char smem[];
  static_assert(ldgsts_size_and_align <= 16);
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(smem) % ldgsts_size_and_align == 0, "");

  constexpr int block_threads = LdgstsPolicy::block_threads;
  const int tile_size         = block_threads * num_elem_per_thread;
  const Offset offset         = static_cast<Offset>(blockIdx.x) * tile_size;
  const int valid_items       = static_cast<int>(::cuda::std::min(num_items - offset, Offset{tile_size}));

  [[maybe_unused]] int smem_offset = 0;
  // TODO(bgruber): drop checking first block, since gmem buffers are always sufficiently aligned. But this would not
  // work for inputs in host stack memory ...
  const bool inner_blocks = 0 < blockIdx.x && blockIdx.x + 2 < gridDim.x;
  // TODO(bgruber): if we used SMEM offsets instead of pointers, we need less registers (but no perf increase)
  [[maybe_unused]] const auto smem_ptrs = ::cuda::std::tuple<const InTs*...>{
    (inner_blocks ? copy_and_return_smem_dst<block_threads>(aligned_ptrs, smem_offset, offset, smem, valid_items)
                  : copy_and_return_smem_dst_fallback<block_threads>(
                      aligned_ptrs, smem_offset, offset, smem, valid_items, tile_size))...};

  asm volatile("cp.async.wait_group %0;" : : "n"(0)); // same as: __pipeline_wait_prior(0);
  __syncthreads();

  // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
  out += offset;

  // TODO(bgruber): fbusato suggests to move the valid_items and smem_base_ptrs by threadIdx.x before the loop below

  auto process_tile = [&](auto full_tile) {
    // Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark)
    _CCCL_PRAGMA_NOUNROLL()
    for (int j = 0; j < num_elem_per_thread; ++j)
    {
      const int idx = j * block_threads + threadIdx.x;
      if (full_tile || idx < valid_items)
      {
        out[idx] = ::cuda::std::apply(
          [&](const auto* __restrict__... smem_base_ptrs) {
            return f(smem_base_ptrs[idx]...);
          },
          smem_ptrs);
      }
    }
  };

  // explicitly calling the lambda on literal true/false lets the compiler emit the lambda twice
  if (tile_size == valid_items)
  {
    process_tile(::cuda::std::true_type{});
  }
  else
  {
    process_tile(::cuda::std::false_type{});
  }
}

_CCCL_DEVICE _CCCL_FORCEINLINE static bool elect_one()
{
  return ::cuda::ptx::elect_sync(~0) && threadIdx.x < 32;
}

template <int BulkCopyAlignment>
_CCCL_DEVICE void bulk_copy_maybe_unaligned(
  void* dst,
  const void* src,
  unsigned int bytes_to_copy,
  int head_padding,
  uint64_t& bar,
  /* inout */ ::cuda::std::uint32_t& total_copied,
  bool elected)
{
  const char* src_ptr = static_cast<const char*>(src);
  char* dst_ptr       = static_cast<char*>(dst);

  // handle tiny copies to simplify head/tail bytes computations below
  if (bytes_to_copy < BulkCopyAlignment)
  {
    if (threadIdx.x < bytes_to_copy)
    {
      dst_ptr[threadIdx.x] = src_ptr[threadIdx.x];
    }
    return;
  }

  const unsigned int head_bytes = (BulkCopyAlignment - head_padding) % BulkCopyAlignment;
  const unsigned int tail_bytes = (bytes_to_copy - head_bytes) % bulk_copy_size_multiple;

  // launch the bulk copy only from the elected thread
  if (elected)
  {
    _CCCL_ASSERT(bytes_to_copy >= (head_bytes + tail_bytes), "");
    const unsigned int aligned_bytes_to_copy = bytes_to_copy - head_bytes - tail_bytes;
    if (aligned_bytes_to_copy > 0)
    {
      _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(dst_ptr + head_bytes) % BulkCopyAlignment == 0, "");
      _CCCL_ASSERT(::cuda::std::bit_cast<uintptr_t>(src_ptr + head_bytes) % BulkCopyAlignment == 0, "");
      _CCCL_ASSERT(aligned_bytes_to_copy % bulk_copy_size_multiple == 0, "");

      ::cuda::ptx::cp_async_bulk(
        ::cuda::ptx::space_cluster,
        ::cuda::ptx::space_global,
        dst_ptr + head_bytes,
        src_ptr + head_bytes,
        aligned_bytes_to_copy,
        &bar);
      total_copied += aligned_bytes_to_copy;
    }
  }

  // ahendriksen: we perform both loads first and then both writes. this reduces the total latency
  char head_byte, tail_byte;
  if (threadIdx.x < head_bytes)
  {
    head_byte = src_ptr[threadIdx.x];
  }
  if (threadIdx.x < tail_bytes)
  {
    tail_byte = src_ptr[bytes_to_copy - tail_bytes + threadIdx.x];
  }
  if (threadIdx.x < head_bytes)
  {
    dst_ptr[threadIdx.x] = head_byte;
  }
  if (threadIdx.x < tail_bytes)
  {
    dst_ptr[bytes_to_copy - tail_bytes + threadIdx.x] = tail_byte;
  }
}

template <typename BulkCopyPolicy, typename Offset, typename F, typename RandomAccessIteratorOut, typename... InTs>
_CCCL_DEVICE void transform_kernel_ublkcp(
  Offset num_items, int num_elem_per_thread, F f, RandomAccessIteratorOut out, aligned_base_ptr<InTs>... aligned_ptrs)
{
  constexpr int block_threads       = BulkCopyPolicy::block_threads;
  constexpr int bulk_copy_alignment = BulkCopyPolicy::bulk_copy_alignment;

  // add padding after a tile in shared memory to make space for the next tile's head padding, and retain alignment
  constexpr int max_alignment = ::cuda::std::max({int{alignof(InTs)}...});
  constexpr int tile_padding  = ::cuda::std::max(bulk_copy_alignment, max_alignment);

  // We could use an attribute to align the shared memory. This is unfortunately not respected by nvcc in all cases and
  // fails for example when compiling with -G or -rdc=true. See also NVBug 5093902, NVBug 5329745, and discussion in PR
  // #5122.
  //
  // Also, because extern (__shared__) variables are injected from inside a function (template) scope into the enclosing
  // namespace scope they must have the same type and (alignment) attributes for the same name. So we cannot specify a
  // different alignment based on the template parameters (i.e. the iterator value types). We could use multiple extern
  // __shared__ variables with different alignment and select the right one at compile time though. A variable template
  // won't work, see NVBug 5420296.
  //
  // However, due to CUDA runtime implementation details, the alignment of dynamic shared memory affects the required
  // *static* shared memory of *every* other kernel that is generated into the same TU (translation unit) as the
  // transform kernel here, which causes all kinds of headaches for downstream users. So lets avoid alignment
  // attributes. More internal information: https://github.com/NVIDIA/cccl_private/wiki/Dynamic-shared-memory-alignment.
  //
  // Because we put the barrier into the dynamic shared memory, we don't have any static shared memory, and the dynamic
  // shared memory starts right at the beginning of the shared memory window, which usually has a high alignment (>
  // 1KiB, depends on the GPU architecture). So we could just rely on this fact and don't specify an attribute to not
  // mess with other kernels. This is also what cutlass does. But it is not guaranteed by CUDA. So let's align manually
  // when we actually need it for correctness. That's when any value type's alignment is larger than 16. Hopper does
  // benefit from 128 byte alignment, which is granted by the alignment of the shared memory window. In case this ever
  // changes, it does not break the correctness and may entail up to 5% perf regression for TMA (the worst I have
  // measured for a 16 byte aligned SMEM destination). So no manual alignment is needed to respect bulk_copy_alignment.

  extern __shared__ char smem_with_barrier_base[]; // aligned to 16 bytes by default
  char* smem_with_barrier = smem_with_barrier_base;
  if constexpr (max_alignment > 16)
  {
    // manual alignment is necessary for correctness
    uint32_t smem32 = __cvta_generic_to_shared(smem_with_barrier);
    smem32          = ::cuda::round_up(smem32, tile_padding);
    asm("" : "+r"(smem32)); // avoid NVVM pulling the alignment code into the kernel, gains up to 8.7% some runs on H200
    smem_with_barrier = static_cast<char*>(__cvta_shared_to_generic(smem32));
  }

  uint64_t& bar = *reinterpret_cast<uint64_t*>(smem_with_barrier);
  static_assert(tile_padding >= sizeof(uint64_t));
  char* smem_base = smem_with_barrier + tile_padding;
  _CCCL_ASSERT(::cuda::is_aligned(smem_base, tile_padding), "");

  namespace ptx = ::cuda::ptx;

  const int tile_size   = block_threads * num_elem_per_thread;
  const Offset offset   = static_cast<Offset>(blockIdx.x) * tile_size;
  const int valid_items = (::cuda::std::min) (num_items - offset, Offset{tile_size});

  const bool inner_blocks = 0 < blockIdx.x && blockIdx.x + 2 < gridDim.x;
  if (inner_blocks)
  {
    // use one thread to setup the entire bulk copy
    if (elect_one())
    {
      ptx::mbarrier_init(&bar, 1);
      // an update to the CUDA memory model blesses skipping the following fence
      // ptx::fence_proxy_async(ptx::space_shared);

      char* smem                         = smem_base;
      ::cuda::std::uint32_t total_copied = 0;

      // turning this lambda into a function does not change SASS
      auto bulk_copy_tile = [&](auto aligned_ptr) {
        using T         = typename decltype(aligned_ptr)::value_type;
        const char* src = aligned_ptr.ptr + offset * unsigned{sizeof(T)}; // compute expression in U32 if Offset==I32
        char* dst       = smem;
        _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % bulk_copy_alignment == 0, "");
        _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % bulk_copy_alignment == 0, "");

        // TODO(bgruber): we could precompute bytes_to_copy on the host
        int bytes_to_copy;
        if constexpr (alignof(T) < bulk_copy_size_multiple)
        {
          bytes_to_copy =
            ::cuda::round_up(aligned_ptr.head_padding + int{sizeof(T)} * tile_size, bulk_copy_size_multiple);
        }
        else
        {
          _CCCL_ASSERT(aligned_ptr.head_padding == 0, "");
          bytes_to_copy = int{sizeof(T)} * tile_size;
        }

        ::cuda::ptx::cp_async_bulk(::cuda::ptx::space_cluster, ::cuda::ptx::space_global, dst, src, bytes_to_copy, &bar);
        total_copied += bytes_to_copy;

        smem += tile_padding + int{sizeof(T)} * tile_size;
        _CCCL_ASSERT(bytes_to_copy <= int{sizeof(T)} * tile_size + bulk_copy_alignment, "");
      };

      // Order of evaluation is left-to-right
      (..., bulk_copy_tile(aligned_ptrs));

      // TODO(ahendriksen): this could only have ptx::sem_relaxed, but this is not available yet
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
    }
  }
  else
  {
    const bool elected = elect_one();
    if (elected)
    {
      ptx::mbarrier_init(&bar, 1);
      // an update to the CUDA memory model blesses skipping the following fence
      // ptx::fence_proxy_async(ptx::space_shared);
    }

    // use all threads to copy the head and tail bytes, use the elected thread to start the bulk copy
    char* smem                         = smem_base;
    ::cuda::std::uint32_t total_copied = 0;

    // turning this lambda into a function does not change SASS
    auto bulk_copy_tile_fallback = [&](auto aligned_ptr) {
      using T = typename decltype(aligned_ptr)::value_type;

      _CCCL_ASSERT(alignof(T) < bulk_copy_alignment || aligned_ptr.head_padding == 0, "");
      const int head_padding = alignof(T) < bulk_copy_alignment ? aligned_ptr.head_padding : 0;

      const char* src = aligned_ptr.ptr + offset * unsigned{sizeof(T)} + head_padding; // compute expression in U32 if
                                                                                       // Offset==I32
      char* dst = smem + head_padding;
      _CCCL_ASSERT(reinterpret_cast<uintptr_t>(src) % alignof(T) == 0, "");
      _CCCL_ASSERT(reinterpret_cast<uintptr_t>(dst) % alignof(T) == 0, "");
      const int bytes_to_copy = int{sizeof(T)} * valid_items;
      bulk_copy_maybe_unaligned<bulk_copy_alignment>(
        dst, src, bytes_to_copy, aligned_ptr.head_padding, bar, total_copied, elected);

      // add padding to account for this tile's head padding
      smem += tile_padding + int{sizeof(T)} * tile_size;
    };

    // Order of evaluation is left-to-right
    (..., bulk_copy_tile_fallback(aligned_ptrs));

    if (elected)
    {
      // TODO(ahendriksen): this could only have ptx::sem_relaxed, but this is not available yet
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, total_copied);
    }
  }

  // all threads wait for bulk copy
  __syncthreads(); // TODO: ahendriksen said this is not needed, but compute-sanitizer disagrees
  while (!ptx::mbarrier_try_wait_parity(&bar, 0))
    ;

  // move the whole index and iterator to the block/thread index, to reduce arithmetic in the loops below
  out += offset;

  auto process_tile = [&](auto full_tile) {
    // Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark)
    _CCCL_PRAGMA_NOUNROLL()
    for (int j = 0; j < num_elem_per_thread; ++j)
    {
      // TODO(bgruber): fbusato suggests to hoist threadIdx.x out of the loop below
      const int idx = j * block_threads + threadIdx.x;
      if (full_tile || idx < valid_items)
      {
        char* smem         = smem_base;
        auto fetch_operand = [&](auto aligned_ptr) {
          using T                = typename decltype(aligned_ptr)::value_type;
          const int head_padding = alignof(T) < bulk_copy_alignment ? aligned_ptr.head_padding : 0;
          const char* src        = smem + head_padding;
          smem += tile_padding + int{sizeof(T)} * tile_size;
          return reinterpret_cast<const T*>(src)[idx];
        };

        // need to expand into a tuple for guaranteed order of evaluation
        out[idx] = ::cuda::std::apply(
          [&](auto... values) {
            return f(values...);
          },
          ::cuda::std::tuple<InTs...>{fetch_operand(aligned_ptrs)...});
      }
    }
  };
  // explicitly calling the lambda on literal true/false lets the compiler emit the lambda twice
  if (tile_size == valid_items)
  {
    process_tile(::cuda::std::true_type{});
  }
  else
  {
    process_tile(::cuda::std::false_type{});
  }
}

template <typename It>
union kernel_arg
{
  aligned_base_ptr<it_value_t<It>> aligned_ptr; // first member is trivial
  static_assert(::cuda::std::is_trivial_v<decltype(aligned_ptr)>, "");
  It iterator; // may not be trivially [default|copy]-constructible

  // Sometimes It is not trivially [default|copy]-constructible (e.g.
  // thrust::normal_iterator<thrust::device_pointer<T>>), so because of
  // https://eel.is/c++draft/class.union#general-note-3, kernel_args's special members are deleted. We work around it by
  // explicitly defining them.
  _CCCL_HOST_DEVICE kernel_arg() noexcept {}
  _CCCL_HOST_DEVICE ~kernel_arg() noexcept {}

  _CCCL_HOST_DEVICE kernel_arg(const kernel_arg& other)
  {
    // since we use kernel_arg only to pass data to the device, the contained data is semantically trivially copyable,
    // even if the type system is telling us otherwise.
    ::cuda::std::memcpy(reinterpret_cast<char*>(this), reinterpret_cast<const char*>(&other), sizeof(kernel_arg));
  }
};

template <typename It>
_CCCL_HOST_DEVICE auto make_iterator_kernel_arg(It it) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  // since we switch the active member of the union, we must use placement new or construct_at. This also uses the copy
  // constructor of It, which works in more cases than assignment (e.g. thrust::transform_iterator with
  // non-copy-assignable functor, e.g. in merge sort tests)
  ::cuda::std::__construct_at(&arg.iterator, it);
  return arg;
}

template <typename It>
_CCCL_HOST_DEVICE auto make_aligned_base_ptr_kernel_arg(It ptr, int alignment) -> kernel_arg<It>
{
  kernel_arg<It> arg;
  arg.aligned_ptr = make_aligned_base_ptr(ptr, alignment);
  return arg;
}

// There is only one kernel for all algorithms, that dispatches based on the selected policy. It must be instantiated
// with the same arguments for each algorithm. Only the device compiler will then select the implementation. This
// saves some compile-time and binary size.
template <typename MaxPolicy,
          typename Offset,
          typename F,
          typename RandomAccessIteratorOut,
          typename... RandomAccessIteartorsIn>
__launch_bounds__(MaxPolicy::ActivePolicy::algo_policy::block_threads)
  CUB_DETAIL_KERNEL_ATTRIBUTES void transform_kernel(
    Offset num_items,
    int num_elem_per_thread,
    [[maybe_unused]] bool can_vectorize,
    F f,
    RandomAccessIteratorOut out,
    kernel_arg<RandomAccessIteartorsIn>... ins)
{
  _CCCL_ASSERT(blockDim.y == 1 && blockDim.z == 1, "transform_kernel only supports 1D blocks");

  if constexpr (MaxPolicy::ActivePolicy::algorithm == Algorithm::prefetch)
  {
    transform_kernel_prefetch<typename MaxPolicy::ActivePolicy::algo_policy>(
      num_items, num_elem_per_thread, ::cuda::std::move(f), ::cuda::std::move(out), ::cuda::std::move(ins.iterator)...);
  }
  else if constexpr (MaxPolicy::ActivePolicy::algorithm == Algorithm::vectorized)
  {
    transform_kernel_vectorized<typename MaxPolicy::ActivePolicy::algo_policy>(
      num_items,
      num_elem_per_thread,
      can_vectorize,
      ::cuda::std::move(f),
      ::cuda::std::move(out),
      ::cuda::std::move(ins.iterator)...);
  }
  else if constexpr (MaxPolicy::ActivePolicy::algorithm == Algorithm::memcpy_async)
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (transform_kernel_ldgsts<typename MaxPolicy::ActivePolicy::algo_policy>(
         num_items,
         num_elem_per_thread,
         ::cuda::std::move(f),
         ::cuda::std::move(out),
         ::cuda::std::move(ins.aligned_ptr)...);));
  }
  else if constexpr (MaxPolicy::ActivePolicy::algorithm == Algorithm::ublkcp)
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (transform_kernel_ublkcp<typename MaxPolicy::ActivePolicy::algo_policy>(
         num_items,
         num_elem_per_thread,
         ::cuda::std::move(f),
         ::cuda::std::move(out),
         ::cuda::std::move(ins.aligned_ptr)...);));
  }
  else
  {
    static_assert(!sizeof(Offset), "Algorithm not implemented");
  }
}

} // namespace detail::transform

CUB_NAMESPACE_END
