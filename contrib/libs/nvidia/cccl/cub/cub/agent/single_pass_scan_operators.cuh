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

/**
 * \file
 * Callback operator types for supplying BlockScan prefixes
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/strong_load.cuh>
#include <cub/detail/strong_store.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/type_traits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Prefix functor type for maintaining a running prefix while scanning a
 * region independent of other thread blocks
 ******************************************************************************/

/**
 * Stateful callback operator type for supplying BlockScan prefixes.
 * Maintains a running prefix that can be applied to consecutive
 * BlockScan operations.
 *
 * @tparam T
 *   BlockScan value type
 *
 * @tparam ScanOpT
 *   Wrapped scan operator type
 */
template <typename T, typename ScanOpT>
struct BlockScanRunningPrefixOp
{
  /// Wrapped scan operator
  ScanOpT op;

  /// Running block-wide prefix
  T running_total;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScanRunningPrefixOp(ScanOpT op)
      : op(op)
  {}

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScanRunningPrefixOp(T starting_prefix, ScanOpT op)
      : op(op)
      , running_total(starting_prefix)
  {}

  /**
   * Prefix callback operator.  Returns the block-wide running_total in thread-0.
   *
   * @param block_aggregate
   *   The aggregate sum of the BlockScan inputs
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(const T& block_aggregate)
  {
    T retval      = running_total;
    running_total = op(running_total, block_aggregate);
    return retval;
  }
};

/******************************************************************************
 * Generic tile status interface types for block-cooperative scans
 ******************************************************************************/

/**
 * Enumerations of tile status
 */
enum ScanTileStatus
{
  SCAN_TILE_OOB, // Out-of-bounds (e.g., padding)
  SCAN_TILE_INVALID = 99, // Not yet processed
  SCAN_TILE_PARTIAL, // Tile aggregate is available
  SCAN_TILE_INCLUSIVE, // Inclusive tile prefix is available
};

/**
 * Enum class used for specifying the memory order that shall be enforced while reading and writing the tile status.
 */
enum class MemoryOrder
{
  // Uses relaxed loads when reading a tile's status and relaxed stores when updating a tile's status
  relaxed,
  // Uses load acquire when reading a tile's status and store release when updating a tile's status
  acquire_release
};

namespace detail
{
template <int Delay, unsigned int GridThreshold = 500>
_CCCL_DEVICE _CCCL_FORCEINLINE void delay()
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (if (Delay > 0) {
                 if (gridDim.x < GridThreshold)
                 {
                   __threadfence_block();
                 }
                 else
                 {
                   __nanosleep(Delay);
                 }
               }));
}

template <unsigned int GridThreshold = 500>
_CCCL_DEVICE _CCCL_FORCEINLINE void delay(int ns)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (if (ns > 0) {
                 if (gridDim.x < GridThreshold)
                 {
                   __threadfence_block();
                 }
                 else
                 {
                   __nanosleep(ns);
                 }
               }));
}

template <int Delay>
_CCCL_DEVICE _CCCL_FORCEINLINE void always_delay()
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(Delay);));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void always_delay([[maybe_unused]] int ns)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(ns);));
}

template <unsigned int Delay = 350, unsigned int GridThreshold = 500>
_CCCL_DEVICE _CCCL_FORCEINLINE void delay_or_prevent_hoisting()
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (delay<Delay, GridThreshold>();), (__threadfence_block();));
}

template <unsigned int GridThreshold = 500>
_CCCL_DEVICE _CCCL_FORCEINLINE void delay_or_prevent_hoisting([[maybe_unused]] int ns)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (delay<GridThreshold>(ns);), (__threadfence_block();));
}

template <unsigned int Delay = 350>
_CCCL_DEVICE _CCCL_FORCEINLINE void always_delay_or_prevent_hoisting()
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (always_delay(Delay);), (__threadfence_block();));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void always_delay_or_prevent_hoisting([[maybe_unused]] int ns)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70, (always_delay(ns);), (__threadfence_block();));
}

template <unsigned int L2WriteLatency>
struct no_delay_constructor_t
{
  struct delay_t
  {
    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      NV_IF_TARGET(NV_PROVIDES_SM_70, (), (__threadfence_block();));
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE no_delay_constructor_t(unsigned int /* seed */)
  {
    delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {};
  }
};

template <unsigned int Delay, unsigned int L2WriteLatency, unsigned int GridThreshold = 500>
struct reduce_by_key_delay_constructor_t
{
  struct delay_t
  {
    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      NV_DISPATCH_TARGET(
        NV_IS_EXACTLY_SM_80,
        (delay<Delay, GridThreshold>();),
        NV_PROVIDES_SM_70,
        (delay<0, GridThreshold>();),
        NV_IS_DEVICE,
        (__threadfence_block();));
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE reduce_by_key_delay_constructor_t(unsigned int /* seed */)
  {
    delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {};
  }
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct fixed_delay_constructor_t
{
  struct delay_t
  {
    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      delay_or_prevent_hoisting<Delay>();
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE fixed_delay_constructor_t(unsigned int /* seed */)
  {
    delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backoff_constructor_t
{
  struct delay_t
  {
    int delay;

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      always_delay_or_prevent_hoisting(delay);
      delay <<= 1;
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backoff_constructor_t(unsigned int /* seed */)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {InitialDelay};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backoff_jitter_constructor_t
{
  struct delay_t
  {
    static constexpr unsigned int a = 16807;
    static constexpr unsigned int c = 0;
    static constexpr unsigned int m = 1u << 31;

    unsigned int max_delay;
    unsigned int& seed;

    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int next(unsigned int min, unsigned int max)
    {
      return (seed = (a * seed + c) % m) % (max + 1 - min) + min;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      always_delay_or_prevent_hoisting(next(0, max_delay));
      max_delay <<= 1;
    }
  };

  unsigned int seed;

  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backoff_jitter_constructor_t(unsigned int seed)
      : seed(seed)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {InitialDelay, seed};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backoff_jitter_window_constructor_t
{
  struct delay_t
  {
    static constexpr unsigned int a = 16807;
    static constexpr unsigned int c = 0;
    static constexpr unsigned int m = 1u << 31;

    unsigned int max_delay;
    unsigned int& seed;

    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int next(unsigned int min, unsigned int max)
    {
      return (seed = (a * seed + c) % m) % (max + 1 - min) + min;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      unsigned int next_max_delay = max_delay << 1;
      always_delay_or_prevent_hoisting(next(max_delay, next_max_delay));
      max_delay = next_max_delay;
    }
  };

  unsigned int seed;
  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backoff_jitter_window_constructor_t(unsigned int seed)
      : seed(seed)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    return {InitialDelay, seed};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backon_jitter_window_constructor_t
{
  struct delay_t
  {
    static constexpr unsigned int a = 16807;
    static constexpr unsigned int c = 0;
    static constexpr unsigned int m = 1u << 31;

    unsigned int max_delay;
    unsigned int& seed;

    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int next(unsigned int min, unsigned int max)
    {
      return (seed = (a * seed + c) % m) % (max + 1 - min) + min;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      int prev_delay = max_delay >> 1;
      always_delay_or_prevent_hoisting(next(prev_delay, max_delay));
      max_delay = prev_delay;
    }
  };

  unsigned int seed;
  unsigned int max_delay = InitialDelay;

  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backon_jitter_window_constructor_t(unsigned int seed)
      : seed(seed)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    max_delay >>= 1;
    return {max_delay, seed};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backon_jitter_constructor_t
{
  struct delay_t
  {
    static constexpr unsigned int a = 16807;
    static constexpr unsigned int c = 0;
    static constexpr unsigned int m = 1u << 31;

    unsigned int max_delay;
    unsigned int& seed;

    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int next(unsigned int min, unsigned int max)
    {
      return (seed = (a * seed + c) % m) % (max + 1 - min) + min;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      always_delay_or_prevent_hoisting(next(0, max_delay));
      max_delay >>= 1;
    }
  };

  unsigned int seed;
  unsigned int max_delay = InitialDelay;

  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backon_jitter_constructor_t(unsigned int seed)
      : seed(seed)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    max_delay >>= 1;
    return {max_delay, seed};
  }
};

template <unsigned int InitialDelay, unsigned int L2WriteLatency>
struct exponential_backon_constructor_t
{
  struct delay_t
  {
    unsigned int delay;

    _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
    {
      always_delay_or_prevent_hoisting(delay);
      delay >>= 1;
    }
  };

  unsigned int max_delay = InitialDelay;

  _CCCL_DEVICE _CCCL_FORCEINLINE exponential_backon_constructor_t(unsigned int /* seed */)
  {
    always_delay<L2WriteLatency>();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE delay_t operator()()
  {
    max_delay >>= 1;
    return {max_delay};
  }
};

using default_no_delay_constructor_t = no_delay_constructor_t<450>;
using default_no_delay_t             = default_no_delay_constructor_t::delay_t;

template <class T>
using default_delay_constructor_t =
  ::cuda::std::_If<is_primitive<T>::value, fixed_delay_constructor_t<350, 450>, default_no_delay_constructor_t>;

template <class T>
using default_delay_t = typename default_delay_constructor_t<T>::delay_t;

template <class KeyT, class ValueT>
using default_reduce_by_key_delay_constructor_t =
  ::cuda::std::_If<is_primitive<ValueT>::value && (sizeof(ValueT) + sizeof(KeyT) < 16),
                   reduce_by_key_delay_constructor_t<350, 450>,
                   default_delay_constructor_t<KeyValuePair<KeyT, ValueT>>>;

/**
 * @brief Alias template for a ScanTileState specialized for a given value type, `T`, and memory order `Order`.
 *
 * @tparam T The ScanTileState's value type
 * @tparam Order The memory order to be implemented by the ScanTileState
 */
template <typename ScanTileStateT, MemoryOrder Order>
struct tile_state_with_memory_order
{
  ScanTileStateT& tile_state;
  using T          = typename ScanTileStateT::StatusValueT;
  using StatusWord = typename ScanTileStateT::StatusWord;

  /**
   * Update the specified tile's inclusive value and corresponding status
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetInclusive(int tile_idx, T tile_inclusive)
  {
    tile_state.template SetInclusive<Order>(tile_idx, tile_inclusive);
  }

  /**
   * Update the specified tile's partial value and corresponding status
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetPartial(int tile_idx, T tile_partial)
  {
    tile_state.template SetPartial<Order>(tile_idx, tile_partial);
  }

  /**
   * Wait for the corresponding tile to become non-invalid
   */
  template <class DelayT = detail::default_no_delay_t>
  _CCCL_DEVICE _CCCL_FORCEINLINE void WaitForValid(int tile_idx, StatusWord& status, T& value, DelayT delay = {})
  {
    tile_state.template WaitForValid<DelayT, Order>(tile_idx, status, value, delay);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE T LoadValid(int tile_idx)
  {
    return tile_state.template LoadValid<Order>(tile_idx);
  }
};

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int num_tiles_to_num_tile_states(int num_tiles)
{
  return warp_threads + num_tiles;
}

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE size_t
tile_state_allocation_size(int bytes_per_description, int bytes_per_payload, int num_tiles)
{
  int num_tile_states = num_tiles_to_num_tile_states(num_tiles);
  size_t allocation_sizes[]{
    // bytes needed for tile status descriptors
    static_cast<size_t>(num_tile_states * bytes_per_description),
    // bytes needed for partials
    static_cast<size_t>(num_tile_states * bytes_per_payload),
    // bytes needed for inclusives
    static_cast<size_t>(num_tile_states * bytes_per_payload)};
  // Set the necessary size of the blob
  size_t temp_storage_bytes = 0;
  void* allocations[3]      = {};
  AliasTemporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);

  return temp_storage_bytes;
};

_CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t tile_state_init(
  int bytes_per_description,
  int bytes_per_payload,
  int num_tiles,
  void* d_temp_storage,
  size_t temp_storage_bytes,
  void* (&allocations)[3])
{
  int num_tile_states = num_tiles_to_num_tile_states(num_tiles);
  size_t allocation_sizes[]{
    // bytes needed for tile status descriptors
    static_cast<size_t>(num_tile_states * bytes_per_description),
    // bytes needed for partials
    static_cast<size_t>(num_tile_states * bytes_per_payload),
    // bytes needed for inclusives
    static_cast<size_t>(num_tile_states * bytes_per_payload)};

  // Set the necessary size of the blob
  return AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
}

} // namespace detail

/**
 * Tile status interface.
 */
template <typename T, bool SINGLE_WORD = detail::is_primitive<T>::value>
struct ScanTileState;

/**
 * Tile status interface specialized for scan status and value types
 * that can be combined into one machine word that can be
 * read/written coherently in a single access.
 */
template <typename T>
struct ScanTileState<T, true>
{
  using StatusValueT = T;

  // Status word type
  using StatusWord = ::cuda::std::_If<
    sizeof(T) == 8,
    unsigned long long,
    ::cuda::std::_If<sizeof(T) == 4, unsigned int, ::cuda::std::_If<sizeof(T) == 2, unsigned short, unsigned char>>>;

  // Unit word type
  using TxnWord = ::cuda::std::_If<sizeof(T) == 8, ulonglong2, ::cuda::std::_If<sizeof(T) == 4, uint2, unsigned int>>;

  // Device word type
  struct TileDescriptor
  {
    StatusWord status;
    T value;
  };

  // Constants
  enum
  {
    TILE_STATUS_PADDING = detail::warp_threads,
  };

  // Device storage
  TxnWord* d_tile_descriptors;

  static constexpr size_t description_bytes_per_tile = sizeof(TxnWord);
  static constexpr size_t payload_bytes_per_tile     = 0;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanTileState()
      : d_tile_descriptors(nullptr)
  {}

  /**
   * @brief Initializer
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to \p temp_storage_bytes and no work is
   * done.
   *
   * @param[in] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t
  Init(int /*num_tiles*/, void* d_temp_storage, size_t /*temp_storage_bytes*/)
  {
    d_tile_descriptors = reinterpret_cast<TxnWord*>(d_temp_storage);
    return cudaSuccess;
  }

  /**
   * @brief Compute device memory needed for tile status
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[out] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr cudaError_t
  AllocationSize(int num_tiles, size_t& temp_storage_bytes)
  {
    temp_storage_bytes =
      detail::tile_state_allocation_size(description_bytes_per_tile, payload_bytes_per_tile, num_tiles);
    return cudaSuccess;
  }

  /**
   * Initialize (from device)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeStatus(int num_tiles)
  {
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    TxnWord val                = TxnWord();
    TileDescriptor* descriptor = reinterpret_cast<TileDescriptor*>(&val);

    if (tile_idx < num_tiles)
    {
      // Not-yet-set
      descriptor->status                                 = StatusWord(SCAN_TILE_INVALID);
      d_tile_descriptors[TILE_STATUS_PADDING + tile_idx] = val;
    }

    if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
    {
      // Padding
      descriptor->status              = StatusWord(SCAN_TILE_OOB);
      d_tile_descriptors[threadIdx.x] = val;
    }
  }

private:
  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::relaxed), void>
  StoreStatus(TxnWord* ptr, TxnWord alias)
  {
    detail::store_relaxed(ptr, alias);
  }

  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::acquire_release), void>
  StoreStatus(TxnWord* ptr, TxnWord alias)
  {
    detail::store_release(ptr, alias);
  }

  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::relaxed), TxnWord>
  LoadStatus(TxnWord* ptr)
  {
    return detail::load_relaxed(ptr);
  }

  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::acquire_release), TxnWord>
  LoadStatus(TxnWord* ptr)
  {
    // For pre-volta we hoist the memory barrier to outside the loop, i.e., after reading a valid state
    NV_IF_TARGET(NV_PROVIDES_SM_70, (return detail::load_acquire(ptr);), (return detail::load_relaxed(ptr);));
  }

  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::relaxed), void>
  ThreadfenceForLoadAcqPreVolta()
  {}

  template <MemoryOrder Order>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(Order == MemoryOrder::acquire_release), void>
  ThreadfenceForLoadAcqPreVolta()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_70, (), (__threadfence();));
  }

public:
  template <MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetInclusive(int tile_idx, T tile_inclusive)
  {
    TileDescriptor tile_descriptor;
    tile_descriptor.status = SCAN_TILE_INCLUSIVE;
    tile_descriptor.value  = tile_inclusive;

    TxnWord alias;
    *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;

    StoreStatus<Order>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
  }

  template <MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetPartial(int tile_idx, T tile_partial)
  {
    TileDescriptor tile_descriptor;
    tile_descriptor.status = SCAN_TILE_PARTIAL;
    tile_descriptor.value  = tile_partial;

    TxnWord alias;
    *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;

    StoreStatus<Order>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
  }

  /**
   * Wait for the corresponding tile to become non-invalid
   */
  template <class DelayT = detail::default_delay_t<T>, MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  WaitForValid(int tile_idx, StatusWord& status, T& value, DelayT delay_or_prevent_hoisting = {})
  {
    TileDescriptor tile_descriptor;

    {
      TxnWord alias   = LoadStatus<Order>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
      tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
    }

    while (__any_sync(0xffffffff, (tile_descriptor.status == SCAN_TILE_INVALID)))
    {
      delay_or_prevent_hoisting();
      TxnWord alias   = LoadStatus<Order>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
      tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
    }

    // For pre-Volta and load acquire we emit relaxed loads in LoadStatus and hoist the threadfence here
    ThreadfenceForLoadAcqPreVolta<Order>();

    status = tile_descriptor.status;
    value  = tile_descriptor.value;
  }

  /**
   * Loads and returns the tile's value. The returned value is undefined if either (a) the tile's status is invalid or
   * (b) there is no memory fence between reading a non-invalid status and the call to LoadValid.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE T LoadValid(int tile_idx)
  {
    TxnWord alias                  = d_tile_descriptors[TILE_STATUS_PADDING + tile_idx];
    TileDescriptor tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
    return tile_descriptor.value;
  }
};

/**
 * Tile status interface specialized for scan status and value types that
 * cannot be combined into one machine word.
 */
template <typename T>
struct ScanTileState<T, false>
{
  using StatusValueT = T;

  // Status word type
  using StatusWord = unsigned int;

  // Constants
  enum
  {
    TILE_STATUS_PADDING = detail::warp_threads,
  };

  // Device storage
  StatusWord* d_tile_status;
  T* d_tile_partial;
  T* d_tile_inclusive;

  static constexpr size_t description_bytes_per_tile = sizeof(StatusWord);
  static constexpr size_t payload_bytes_per_tile     = sizeof(Uninitialized<T>);

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanTileState()
      : d_tile_status(nullptr)
      , d_tile_partial(nullptr)
      , d_tile_inclusive(nullptr)
  {}

  /**
   * @brief Initializer
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to \p temp_storage_bytes and no work is
   *   done.
   *
   * @param[in] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  /// Initializer
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t Init(int num_tiles, void* d_temp_storage, size_t temp_storage_bytes)
  {
    cudaError_t error = cudaSuccess;
    do
    {
      void* allocations[3] = {};
      error                = detail::tile_state_init(
        description_bytes_per_tile, payload_bytes_per_tile, num_tiles, d_temp_storage, temp_storage_bytes, allocations);
      if (cudaSuccess != error)
      {
        break;
      }
      // Alias the offsets
      d_tile_status    = reinterpret_cast<StatusWord*>(allocations[0]);
      d_tile_partial   = reinterpret_cast<T*>(allocations[1]);
      d_tile_inclusive = reinterpret_cast<T*>(allocations[2]);
    } while (0);

    return error;
  }

  /**
   * @brief Compute device memory needed for tile status
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[out] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr cudaError_t
  AllocationSize(int num_tiles, size_t& temp_storage_bytes)
  {
    temp_storage_bytes =
      detail::tile_state_allocation_size(description_bytes_per_tile, payload_bytes_per_tile, num_tiles);
    return cudaSuccess;
  }
  /**
   * Initialize (from device)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeStatus(int num_tiles)
  {
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_tiles)
    {
      // Not-yet-set
      d_tile_status[TILE_STATUS_PADDING + tile_idx] = StatusWord(SCAN_TILE_INVALID);
    }

    if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
    {
      // Padding
      d_tile_status[threadIdx.x] = StatusWord(SCAN_TILE_OOB);
    }
  }

  /**
   * Update the specified tile's inclusive value and corresponding status
   */
  template <MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetInclusive(int tile_idx, T tile_inclusive)
  {
    // Update tile inclusive value
    ThreadStore<STORE_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx, tile_inclusive);
    detail::store_release(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_INCLUSIVE));
  }

  /**
   * Update the specified tile's partial value and corresponding status
   */
  template <MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetPartial(int tile_idx, T tile_partial)
  {
    // Update tile partial value
    ThreadStore<STORE_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx, tile_partial);
    detail::store_release(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_PARTIAL));
  }

  /**
   * Wait for the corresponding tile to become non-invalid
   */
  template <class DelayT = detail::default_no_delay_t, MemoryOrder Order = MemoryOrder::relaxed>
  _CCCL_DEVICE _CCCL_FORCEINLINE void WaitForValid(int tile_idx, StatusWord& status, T& value, DelayT delay = {})
  {
    do
    {
      delay();
      status = detail::load_relaxed(d_tile_status + TILE_STATUS_PADDING + tile_idx);
      __threadfence();
    } while (__any_sync(0xffffffff, (status == SCAN_TILE_INVALID)));

    if (status == StatusWord(SCAN_TILE_PARTIAL))
    {
      value = ThreadLoad<LOAD_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx);
    }
    else if (status == StatusWord(SCAN_TILE_INCLUSIVE))
    {
      value = ThreadLoad<LOAD_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx);
    }
  }

  /**
   * Loads and returns the tile's value. The returned value is undefined if either (a) the tile's status is invalid or
   * (b) there is no memory fence between reading a non-invalid status and the call to LoadValid.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE T LoadValid(int tile_idx)
  {
    return d_tile_inclusive[TILE_STATUS_PADDING + tile_idx];
  }
};

/******************************************************************************
 * ReduceByKey tile status interface types for block-cooperative scans
 ******************************************************************************/

/**
 * Tile status interface for reduction by key.
 *
 */
template <typename ValueT,
          typename KeyT,
          bool SINGLE_WORD = detail::is_primitive<ValueT>::value && (sizeof(ValueT) + sizeof(KeyT) < 16)>
struct ReduceByKeyScanTileState;

/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * cannot be combined into one machine word.
 */
template <typename ValueT, typename KeyT>
struct ReduceByKeyScanTileState<ValueT, KeyT, false> : ScanTileState<KeyValuePair<KeyT, ValueT>>
{
  using SuperClass = ScanTileState<KeyValuePair<KeyT, ValueT>>;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyScanTileState()
      : SuperClass()
  {}
};

/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * can be combined into one machine word that can be read/written coherently in a single access.
 */
template <typename ValueT, typename KeyT>
struct ReduceByKeyScanTileState<ValueT, KeyT, true>
{
  using KeyValuePairT = KeyValuePair<KeyT, ValueT>;

  // Constants
  enum
  {
    PAIR_SIZE        = static_cast<int>(sizeof(ValueT) + sizeof(KeyT)),
    TXN_WORD_SIZE    = 1 << Log2<PAIR_SIZE + 1>::VALUE,
    STATUS_WORD_SIZE = TXN_WORD_SIZE - PAIR_SIZE,

    TILE_STATUS_PADDING = detail::warp_threads,
  };

  // Status word type
  using StatusWord = ::cuda::std::_If<
    STATUS_WORD_SIZE == 8,
    unsigned long long,
    ::cuda::std::
      _If<STATUS_WORD_SIZE == 4, unsigned int, ::cuda::std::_If<STATUS_WORD_SIZE == 2, unsigned short, unsigned char>>>;

  // Status word type
  using TxnWord = ::cuda::std::
    _If<TXN_WORD_SIZE == 16, ulonglong2, ::cuda::std::_If<TXN_WORD_SIZE == 8, unsigned long long, unsigned int>>;

  // Device word type (for when sizeof(ValueT) == sizeof(KeyT))
  struct TileDescriptorBigStatus
  {
    KeyT key;
    ValueT value;
    StatusWord status;
  };

  // Device word type (for when sizeof(ValueT) != sizeof(KeyT))
  struct TileDescriptorLittleStatus
  {
    ValueT value;
    StatusWord status;
    KeyT key;
  };

  // Device word type
  using TileDescriptor =
    ::cuda::std::_If<sizeof(ValueT) == sizeof(KeyT), TileDescriptorBigStatus, TileDescriptorLittleStatus>;

  // Device storage
  TxnWord* d_tile_descriptors;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyScanTileState()
      : d_tile_descriptors(nullptr)
  {}

  /**
   * @brief Initializer
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.  When nullptr, the required allocation size
   *   is written to \p temp_storage_bytes and no work is done.
   *
   * @param[in] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t
  Init(int /*num_tiles*/, void* d_temp_storage, size_t /*temp_storage_bytes*/)
  {
    d_tile_descriptors = reinterpret_cast<TxnWord*>(d_temp_storage);
    return cudaSuccess;
  }

  /**
   * @brief Compute device memory needed for tile status
   *
   * @param[in] num_tiles
   *   Number of tiles
   *
   * @param[out] temp_storage_bytes
   *   Size in bytes of \t d_temp_storage allocation
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static cudaError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes)
  {
    // bytes needed for tile status descriptors
    temp_storage_bytes = (num_tiles + TILE_STATUS_PADDING) * sizeof(TxnWord);
    return cudaSuccess;
  }

  /**
   * Initialize (from device)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeStatus(int num_tiles)
  {
    int tile_idx               = (blockIdx.x * blockDim.x) + threadIdx.x;
    TxnWord val                = TxnWord();
    TileDescriptor* descriptor = reinterpret_cast<TileDescriptor*>(&val);

    if (tile_idx < num_tiles)
    {
      // Not-yet-set
      descriptor->status                                 = StatusWord(SCAN_TILE_INVALID);
      d_tile_descriptors[TILE_STATUS_PADDING + tile_idx] = val;
    }

    if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
    {
      // Padding
      descriptor->status              = StatusWord(SCAN_TILE_OOB);
      d_tile_descriptors[threadIdx.x] = val;
    }
  }

  /**
   * Update the specified tile's inclusive value and corresponding status
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void SetInclusive(int tile_idx, KeyValuePairT tile_inclusive)
  {
    TileDescriptor tile_descriptor;
    tile_descriptor.status = SCAN_TILE_INCLUSIVE;
    tile_descriptor.value  = tile_inclusive.value;
    tile_descriptor.key    = tile_inclusive.key;

    TxnWord alias;
    *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;

    detail::store_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void SetPartial(int tile_idx, KeyValuePairT tile_partial)
  {
    TileDescriptor tile_descriptor;
    tile_descriptor.status = SCAN_TILE_PARTIAL;
    tile_descriptor.value  = tile_partial.value;
    tile_descriptor.key    = tile_partial.key;

    TxnWord alias;
    *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;

    detail::store_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
  }

  /**
   * Wait for the corresponding tile to become non-invalid
   */
  template <class DelayT = detail::fixed_delay_constructor_t<350, 450>::delay_t>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  WaitForValid(int tile_idx, StatusWord& status, KeyValuePairT& value, DelayT delay_or_prevent_hoisting = {})
  {
    //        TxnWord         alias           = ThreadLoad<LOAD_CG>(d_tile_descriptors + TILE_STATUS_PADDING +
    //        tile_idx); TileDescriptor  tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
    //
    //        while (tile_descriptor.status == SCAN_TILE_INVALID)
    //        {
    //            __threadfence_block(); // prevent hoisting loads from loop
    //
    //            alias           = ThreadLoad<LOAD_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
    //            tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);
    //        }
    //
    //        status      = tile_descriptor.status;
    //        value.value = tile_descriptor.value;
    //        value.key   = tile_descriptor.key;

    TileDescriptor tile_descriptor;

    do
    {
      delay_or_prevent_hoisting();
      TxnWord alias   = detail::load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
      tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);

    } while (__any_sync(0xffffffff, (tile_descriptor.status == SCAN_TILE_INVALID)));

    status      = tile_descriptor.status;
    value.value = tile_descriptor.value;
    value.key   = tile_descriptor.key;
  }
};

/******************************************************************************
 * Prefix call-back operator for coupling local block scan within a
 * block-cooperative scan
 ******************************************************************************/

/**
 * Stateful block-scan prefix functor.  Provides the the running prefix for
 * the current tile by using the call-back warp to wait on on
 * aggregates/prefixes from predecessor tiles to become available.
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <typename T,
          typename ScanOpT,
          typename ScanTileStateT,
          typename DelayConstructorT = detail::default_delay_constructor_t<T>>
struct TilePrefixCallbackOp
{
  // Parameterized warp reduce
  using WarpReduceT = WarpReduce<T, (1 << (5))>;

  // Temporary storage type
  struct _TempStorage
  {
    typename WarpReduceT::TempStorage warp_reduce;
    T exclusive_prefix;
    T inclusive_prefix;
    T block_aggregate;
  };

  // Alias wrapper allowing temporary storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Type of status word
  using StatusWord = typename ScanTileStateT::StatusWord;

  // Fields
  _TempStorage& temp_storage; ///< Reference to a warp-reduction instance
  ScanTileStateT& tile_status; ///< Interface to tile status
  ScanOpT scan_op; ///< Binary scan operator
  int tile_idx; ///< The current tile index
  T exclusive_prefix; ///< Exclusive prefix for the tile
  T inclusive_prefix; ///< Inclusive prefix for the tile

  // Constructs prefix functor for a given tile index.
  // Precondition: thread blocks processing all of the predecessor tiles were scheduled.
  _CCCL_DEVICE _CCCL_FORCEINLINE
  TilePrefixCallbackOp(ScanTileStateT& tile_status, TempStorage& temp_storage, ScanOpT scan_op, int tile_idx)
      : temp_storage(temp_storage.Alias())
      , tile_status(tile_status)
      , scan_op(scan_op)
      , tile_idx(tile_idx)
  {}

  // Computes the tile index and constructs prefix functor with it.
  // Precondition: thread block per tile assignment.
  _CCCL_DEVICE _CCCL_FORCEINLINE
  TilePrefixCallbackOp(ScanTileStateT& tile_status, TempStorage& temp_storage, ScanOpT scan_op)
      : TilePrefixCallbackOp(tile_status, temp_storage, scan_op, blockIdx.x)
  {}

  /**
   * @brief Block until all predecessors within the warp-wide window have non-invalid status
   *
   * @param predecessor_idx
   *   Preceding tile index to inspect
   *
   * @param[out] predecessor_status
   *   Preceding tile status
   *
   * @param[out] window_aggregate
   *   Relevant partial reduction from this window of preceding tiles
   */
  template <class DelayT = detail::default_delay_t<T>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ProcessWindow(int predecessor_idx, StatusWord& predecessor_status, T& window_aggregate, DelayT delay = {})
  {
    T value;
    tile_status.WaitForValid(predecessor_idx, predecessor_status, value, delay);

    // Perform a segmented reduction to get the prefix for the current window.
    // Use the swizzled scan operator because we are now scanning *down* towards thread0.

    int tail_flag = (predecessor_status == StatusWord(SCAN_TILE_INCLUSIVE));
    window_aggregate =
      WarpReduceT(temp_storage.warp_reduce).TailSegmentedReduce(value, tail_flag, SwizzleScanOp<ScanOpT>(scan_op));
  }

  // BlockScan prefix callback functor (called by the first warp)
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T block_aggregate)
  {
    // Update our status with our tile-aggregate
    if (threadIdx.x == 0)
    {
      detail::uninitialized_copy_single(&temp_storage.block_aggregate, block_aggregate);

      tile_status.SetPartial(tile_idx, block_aggregate);
    }

    int predecessor_idx = tile_idx - threadIdx.x - 1;
    StatusWord predecessor_status;
    T window_aggregate;

    // Wait for the warp-wide window of predecessor tiles to become valid
    DelayConstructorT construct_delay(tile_idx);
    ProcessWindow(predecessor_idx, predecessor_status, window_aggregate, construct_delay());

    // The exclusive tile prefix starts out as the current window aggregate
    exclusive_prefix = window_aggregate;

    // Keep sliding the window back until we come across a tile whose inclusive prefix is known
    while (__all_sync(0xffffffff, (predecessor_status != StatusWord(SCAN_TILE_INCLUSIVE))))
    {
      predecessor_idx -= detail::warp_threads;

      // Update exclusive tile prefix with the window prefix
      ProcessWindow(predecessor_idx, predecessor_status, window_aggregate, construct_delay());
      exclusive_prefix = scan_op(window_aggregate, exclusive_prefix);
    }

    // Compute the inclusive tile prefix and update the status for this tile
    if (threadIdx.x == 0)
    {
      inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
      tile_status.SetInclusive(tile_idx, inclusive_prefix);

      detail::uninitialized_copy_single(&temp_storage.exclusive_prefix, exclusive_prefix);

      detail::uninitialized_copy_single(&temp_storage.inclusive_prefix, inclusive_prefix);
    }

    // Return exclusive_prefix
    return exclusive_prefix;
  }

  // Get the exclusive prefix stored in temporary storage
  _CCCL_DEVICE _CCCL_FORCEINLINE T GetExclusivePrefix()
  {
    return temp_storage.exclusive_prefix;
  }

  // Get the inclusive prefix stored in temporary storage
  _CCCL_DEVICE _CCCL_FORCEINLINE T GetInclusivePrefix()
  {
    return temp_storage.inclusive_prefix;
  }

  // Get the block aggregate stored in temporary storage
  _CCCL_DEVICE _CCCL_FORCEINLINE T GetBlockAggregate()
  {
    return temp_storage.block_aggregate;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE int GetTileIdx() const
  {
    return tile_idx;
  }
};

CUB_NAMESPACE_END
