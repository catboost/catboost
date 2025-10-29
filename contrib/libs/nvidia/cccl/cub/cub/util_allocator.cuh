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

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
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

#include <cub/util_debug.cuh>
#include <cub/util_namespace.cuh>

#include <map>
#include <mutex>
#include <set>

#include <math.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************/

/**
 * @brief A simple caching allocator for device memory allocations.
 *
 * @par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * @par
 * - Allocations from the allocator are associated with an @p active_stream. Once freed,
 *   the allocation becomes available immediately for reuse within the @p active_stream
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other streams when all prior work submitted to @p active_stream has completed.
 * - Allocations are categorized and cached by bin size. A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   @p bin_growth provided during construction. Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below ( @p bin_growth ^ @p min_bin ) are rounded up to
 *   ( @p bin_growth ^ @p min_bin ).
 * - Allocations above ( @p bin_growth ^ @p max_bin ) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - If the total storage of cached allocations on a given device will exceed
 *   @p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * @par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - @p bin_growth          = 8
 * - @p min_bin             = 3
 * - @p max_bin             = 7
 * - @p max_cached_bytes    = 6MB - 1B
 *
 * @par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct CachingDeviceAllocator
{
  //---------------------------------------------------------------------
  // Constants
  //---------------------------------------------------------------------

  /// Out-of-bounds bin
  static constexpr unsigned int INVALID_BIN = (unsigned int) -1;

  /// Invalid size
  static constexpr size_t INVALID_SIZE = (size_t) -1;

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

  /// Invalid device ordinal
  static constexpr int INVALID_DEVICE_ORDINAL = -1;

  //---------------------------------------------------------------------
  // Type definitions and helper types
  //---------------------------------------------------------------------

  /**
   * Descriptor for device memory allocations
   */
  struct BlockDescriptor
  {
    // Device pointer
    void* d_ptr;

    // Size of allocation in bytes
    size_t bytes;

    // Bin enumeration
    unsigned int bin;

    // device ordinal
    int device;

    // Associated associated_stream
    cudaStream_t associated_stream;

    // Signal when associated stream has run to the point at which this block was freed
    cudaEvent_t ready_event;

    // Constructor (suitable for searching maps for a specific block, given its pointer and
    // device)
    BlockDescriptor(void* d_ptr, int device)
        : d_ptr(d_ptr)
        , bytes(0)
        , bin(INVALID_BIN)
        , device(device)
        , associated_stream(0)
        , ready_event(0)
    {}

    // Constructor (suitable for searching maps for a range of suitable blocks, given a device)
    BlockDescriptor(int device)
        : d_ptr(nullptr)
        , bytes(0)
        , bin(INVALID_BIN)
        , device(device)
        , associated_stream(0)
        , ready_event(0)
    {}

    // Comparison functor for comparing device pointers
    static bool PtrCompare(const BlockDescriptor& a, const BlockDescriptor& b)
    {
      if (a.device == b.device)
      {
        return (a.d_ptr < b.d_ptr);
      }
      else
      {
        return (a.device < b.device);
      }
    }

    // Comparison functor for comparing allocation sizes
    static bool SizeCompare(const BlockDescriptor& a, const BlockDescriptor& b)
    {
      if (a.device == b.device)
      {
        return (a.bytes < b.bytes);
      }
      else
      {
        return (a.device < b.device);
      }
    }
  };

  /// BlockDescriptor comparator function interface
  using Compare = bool (*)(const BlockDescriptor&, const BlockDescriptor&);

  class TotalBytes
  {
  public:
    size_t free;
    size_t live;
    TotalBytes()
    {
      free = live = 0;
    }
  };

  /// Set type for cached blocks (ordered by size)
  using CachedBlocks = std::multiset<BlockDescriptor, Compare>;

  /// Set type for live blocks (ordered by ptr)
  using BusyBlocks = std::multiset<BlockDescriptor, Compare>;

  /// Map type of device ordinals to the number of cached bytes cached by each device
  using GpuCachedBytes = std::map<int, TotalBytes>;

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  /**
   * Integer pow function for unsigned base and exponent
   */
  static unsigned int IntPow(unsigned int base, unsigned int exp)
  {
    unsigned int retval = 1;
    while (exp > 0)
    {
      if (exp & 1)
      {
        retval = retval * base; // multiply the result by the current base
      }
      base = base * base; // square the base
      exp  = exp >> 1; // divide the exponent in half
    }
    return retval;
  }

  /**
   * Round up to the nearest power-of
   */
  void NearestPowerOf(unsigned int& power, size_t& rounded_bytes, unsigned int base, size_t value)
  {
    power         = 0;
    rounded_bytes = 1;

    if (value * base < value)
    {
      // Overflow
      power         = sizeof(size_t) * 8;
      rounded_bytes = size_t(0) - 1;
      return;
    }

    while (rounded_bytes < value)
    {
      rounded_bytes *= base;
      power++;
    }
  }

  //---------------------------------------------------------------------
  // Fields
  //---------------------------------------------------------------------

  /// Mutex for thread-safety
  std::mutex mutex;

  /// Geometric growth factor for bin-sizes
  unsigned int bin_growth;

  /// Minimum bin enumeration
  unsigned int min_bin;

  /// Maximum bin enumeration
  unsigned int max_bin;

  /// Minimum bin size
  size_t min_bin_bytes;

  /// Maximum bin size
  size_t max_bin_bytes;

  /// Maximum aggregate cached bytes per device
  size_t max_cached_bytes;

  /// Whether or not to skip a call to FreeAllCached() when destructor is called.
  /// (The CUDA runtime may have already shut down for statically declared allocators)
  const bool skip_cleanup;

  /// Whether or not to print (de)allocation events to stdout
  bool debug;

  /// Map of device ordinal to aggregate cached bytes on that device
  GpuCachedBytes cached_bytes;

  /// Set of cached device allocations available for reuse
  CachedBlocks cached_blocks;

  /// Set of live device allocations currently in use
  BusyBlocks live_blocks;

#endif // _CCCL_DOXYGEN_INVOKED

  //---------------------------------------------------------------------
  // Methods
  //---------------------------------------------------------------------

  /**
   * @brief Constructor.
   *
   * @param bin_growth
   *   Geometric growth factor for bin-sizes
   *
   * @param min_bin
   *   Minimum bin (default is bin_growth ^ 1)
   *
   * @param max_bin
   *   Maximum bin (default is no max bin)
   *
   * @param max_cached_bytes
   *   Maximum aggregate cached bytes per device (default is no limit)
   *
   * @param skip_cleanup
   *   Whether or not to skip a call to @p FreeAllCached() when the destructor is called (default
   *   is to deallocate)
   *
   * @param debug
   *   Whether or not to print (de)allocation events to stdout (default is no stderr output)
   */
  CachingDeviceAllocator(
    unsigned int bin_growth,
    unsigned int min_bin    = 1,
    unsigned int max_bin    = INVALID_BIN,
    size_t max_cached_bytes = INVALID_SIZE,
    bool skip_cleanup       = false)
      : bin_growth(bin_growth)
      , min_bin(min_bin)
      , max_bin(max_bin)
      , min_bin_bytes(IntPow(bin_growth, min_bin))
      , max_bin_bytes(IntPow(bin_growth, max_bin))
      , max_cached_bytes(max_cached_bytes)
      , skip_cleanup(skip_cleanup)
      , debug(false)
      , cached_blocks(BlockDescriptor::SizeCompare)
      , live_blocks(BlockDescriptor::PtrCompare)
  {}

  /**
   * @brief Default constructor.
   *
   * Configured with:
   * @par
   * - @p bin_growth          = 8
   * - @p min_bin             = 3
   * - @p max_bin             = 7
   * - @p max_cached_bytes    = ( @p bin_growth ^ @p max_bin) * 3 ) - 1 = 6,291,455 bytes
   *
   * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
   * sets a maximum of 6,291,455 cached bytes per device
   */
  CachingDeviceAllocator(bool skip_cleanup = false, bool debug = false)
      : bin_growth(8)
      , min_bin(3)
      , max_bin(7)
      , min_bin_bytes(IntPow(bin_growth, min_bin))
      , max_bin_bytes(IntPow(bin_growth, max_bin))
      , max_cached_bytes((max_bin_bytes * 3) - 1)
      , skip_cleanup(skip_cleanup)
      , debug(debug)
      , cached_blocks(BlockDescriptor::SizeCompare)
      , live_blocks(BlockDescriptor::PtrCompare)
  {}

  /**
   * @brief Sets the limit on the number bytes this allocator is allowed to cache per device.
   *
   * Changing the ceiling of cached bytes does not cause any allocations (in-use or
   * cached-in-reserve) to be freed.  See \p FreeAllCached().
   */
  cudaError_t SetMaxCachedBytes(size_t max_cached_bytes_)
  {
    // Lock
    mutex.lock();

#ifdef CUB_DEBUG_LOG
    _CubLog(
      "Changing max_cached_bytes (%lld -> %lld)\n", (long long) this->max_cached_bytes, (long long) max_cached_bytes_);
#endif

    this->max_cached_bytes = max_cached_bytes_;

    // Unlock
    mutex.unlock();

    return cudaSuccess;
  }

  /**
   * @brief Provides a suitable allocation of device memory for the given size on the specified
   *        device.
   *
   * Once freed, the allocation becomes available immediately for reuse within the @p
   * active_stream with which it was associated with during allocation, and it becomes available
   * for reuse within other streams when all prior work submitted to @p active_stream has
   * completed.
   *
   * @param[in] device
   *   Device on which to place the allocation
   *
   * @param[out] d_ptr
   *   Reference to pointer to the allocation
   *
   * @param[in] bytes
   *   Minimum number of bytes for the allocation
   *
   * @param[in] active_stream
   *   The stream to be associated with this allocation
   */
  cudaError_t DeviceAllocate(int device, void** d_ptr, size_t bytes, cudaStream_t active_stream = 0)
  {
    *d_ptr                = nullptr;
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    cudaError_t error     = cudaSuccess;

    if (device == INVALID_DEVICE_ORDINAL)
    {
      error = CubDebug(cudaGetDevice(&entrypoint_device));
      if (cudaSuccess != error)
      {
        return error;
      }

      device = entrypoint_device;
    }

    // Create a block descriptor for the requested allocation
    bool found = false;
    BlockDescriptor search_key(device);
    search_key.associated_stream = active_stream;
    NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

    if (search_key.bin > max_bin)
    {
      // Bin is greater than our maximum bin: allocate the request
      // exactly and give out-of-bounds bin.  It will not be cached
      // for reuse when returned.
      search_key.bin   = INVALID_BIN;
      search_key.bytes = bytes;
    }
    else
    {
      // Search for a suitable cached allocation: lock
      mutex.lock();

      if (search_key.bin < min_bin)
      {
        // Bin is less than minimum bin: round up
        search_key.bin   = min_bin;
        search_key.bytes = min_bin_bytes;
      }

      // Iterate through the range of cached blocks on the same device in the same bin
      CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
      while ((block_itr != cached_blocks.end()) && (block_itr->device == device) && (block_itr->bin == search_key.bin))
      {
        // To prevent races with reusing blocks returned by the host but still
        // in use by the device, only consider cached blocks that are
        // either (from the active stream) or (from an idle stream)
        bool is_reusable = false;
        if (active_stream == block_itr->associated_stream)
        {
          is_reusable = true;
        }
        else
        {
          const cudaError_t event_status = cudaEventQuery(block_itr->ready_event);
          if (event_status != cudaErrorNotReady)
          {
            CubDebug(event_status);
            is_reusable = true;
          }
        }

        if (is_reusable)
        {
          // Reuse existing cache block.  Insert into live blocks.
          found                        = true;
          search_key                   = *block_itr;
          search_key.associated_stream = active_stream;
          live_blocks.insert(search_key);

          // Remove from free blocks
          cached_bytes[device].free -= search_key.bytes;
          cached_bytes[device].live += search_key.bytes;

#ifdef CUB_DEBUG_LOG
          _CubLog("\tDevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with "
                  "stream %lld).\n",
                  device,
                  search_key.d_ptr,
                  (long long) search_key.bytes,
                  (long long) search_key.associated_stream,
                  (long long) block_itr->associated_stream);
#endif

          cached_blocks.erase(block_itr);

          break;
        }
        block_itr++;
      }

      // Done searching: unlock
      mutex.unlock();
    }

    // Allocate the block if necessary
    if (!found)
    {
      // Set runtime's current device to specified device (entrypoint may not be set)
      if (device != entrypoint_device)
      {
        error = CubDebug(cudaGetDevice(&entrypoint_device));
        if (cudaSuccess != error)
        {
          return error;
        }

        error = CubDebug(cudaSetDevice(device));
        if (cudaSuccess != error)
        {
          return error;
        }
      }

      // Attempt to allocate
      error = CubDebug(cudaMalloc(&search_key.d_ptr, search_key.bytes));
      if (error == cudaErrorMemoryAllocation)
      {
        // The allocation attempt failed: free all cached blocks on device and retry
#ifdef CUB_DEBUG_LOG
        _CubLog("\tDevice %d failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                device,
                (long long) search_key.bytes,
                (long long) search_key.associated_stream);
#endif

        error = cudaSuccess; // Reset the error we will return
        cudaGetLastError(); // Reset CUDART's error

        // Lock
        mutex.lock();

        // Iterate the range of free blocks on the same device
        BlockDescriptor free_key(device);
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

        while ((block_itr != cached_blocks.end()) && (block_itr->device == device))
        {
          // No need to worry about synchronization with the device: cudaFree is
          // blocking and will synchronize across all kernels executing
          // on the current device

          // Free device memory and destroy stream event.
          error = CubDebug(cudaFree(block_itr->d_ptr));
          if (cudaSuccess != error)
          {
            break;
          }

          error = CubDebug(cudaEventDestroy(block_itr->ready_event));
          if (cudaSuccess != error)
          {
            break;
          }

          // Reduce balance and erase entry
          cached_bytes[device].free -= block_itr->bytes;

#ifdef CUB_DEBUG_LOG
          _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks "
                  "(%lld bytes) outstanding.\n",
                  device,
                  (long long) block_itr->bytes,
                  (long long) cached_blocks.size(),
                  (long long) cached_bytes[device].free,
                  (long long) live_blocks.size(),
                  (long long) cached_bytes[device].live);
#endif

          block_itr = cached_blocks.erase(block_itr);
        }

        // Unlock
        mutex.unlock();

        // Return under error
        if (error)
        {
          return error;
        }

        // Try to allocate again
        error = CubDebug(cudaMalloc(&search_key.d_ptr, search_key.bytes));
        if (cudaSuccess != error)
        {
          return error;
        }
      }

      // Create ready event
      error = CubDebug(cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming));

      if (cudaSuccess != error)
      {
        return error;
      }

      // Insert into live blocks
      mutex.lock();
      live_blocks.insert(search_key);
      cached_bytes[device].live += search_key.bytes;
      mutex.unlock();

#ifdef CUB_DEBUG_LOG
      _CubLog("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
              device,
              search_key.d_ptr,
              (long long) search_key.bytes,
              (long long) search_key.associated_stream);
#endif

      // Attempt to revert back to previous device if necessary
      if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
      {
        error = CubDebug(cudaSetDevice(entrypoint_device));
        if (cudaSuccess != error)
        {
          return error;
        }
      }
    }

    // Copy device pointer to output parameter
    *d_ptr = search_key.d_ptr;

#ifdef CUB_DEBUG_LOG
    if (debug)
    {
      _CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
              (long long) cached_blocks.size(),
              (long long) cached_bytes[device].free,
              (long long) live_blocks.size(),
              (long long) cached_bytes[device].live);
    }
#endif

    return error;
  }

  /**
   * @brief Provides a suitable allocation of device memory for the given size on the current
   *        device.
   *
   * Once freed, the allocation becomes available immediately for reuse within the @p
   * active_stream with which it was associated with during allocation, and it becomes available
   * for reuse within other streams when all prior work submitted to @p active_stream has
   * completed.
   *
   * @param[out] d_ptr
   *   Reference to pointer to the allocation
   *
   * @param[in] bytes
   *   Minimum number of bytes for the allocation
   *
   * @param[in] active_stream
   *   The stream to be associated with this allocation
   */
  cudaError_t DeviceAllocate(void** d_ptr, size_t bytes, cudaStream_t active_stream = 0)
  {
    return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
  }

  /**
   * @brief Frees a live allocation of device memory on the specified device, returning it to the
   *        allocator.
   *
   * Once freed, the allocation becomes available immediately for reuse within the
   * @p active_stream with which it was associated with during allocation, and it becomes
   * available for reuse within other streams when all prior work submitted to @p active_stream
   * has completed.
   */
  cudaError_t DeviceFree(int device, void* d_ptr)
  {
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    cudaError_t error     = cudaSuccess;

    if (device == INVALID_DEVICE_ORDINAL)
    {
      error = CubDebug(cudaGetDevice(&entrypoint_device));
      if (cudaSuccess != error)
      {
        return error;
      }
      device = entrypoint_device;
    }

    // Lock
    mutex.lock();

    // Find corresponding block descriptor
    bool recached = false;
    BlockDescriptor search_key(d_ptr, device);
    BusyBlocks::iterator block_itr = live_blocks.find(search_key);
    if (block_itr != live_blocks.end())
    {
      // Remove from live blocks
      search_key = *block_itr;
      live_blocks.erase(block_itr);
      cached_bytes[device].live -= search_key.bytes;

      // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
      if ((search_key.bin != INVALID_BIN) && (cached_bytes[device].free + search_key.bytes <= max_cached_bytes))
      {
        // Insert returned allocation into free blocks
        recached = true;
        cached_blocks.insert(search_key);
        cached_bytes[device].free += search_key.bytes;

#ifdef CUB_DEBUG_LOG
        _CubLog("\tDevice %d returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld "
                "bytes), %lld live blocks outstanding. (%lld bytes)\n",
                device,
                (long long) search_key.bytes,
                (long long) search_key.associated_stream,
                (long long) cached_blocks.size(),
                (long long) cached_bytes[device].free,
                (long long) live_blocks.size(),
                (long long) cached_bytes[device].live);
#endif
      }
    }

    // Unlock
    mutex.unlock();

    // First set to specified device (entrypoint may not be set)
    if (device != entrypoint_device)
    {
      error = CubDebug(cudaGetDevice(&entrypoint_device));
      if (cudaSuccess != error)
      {
        return error;
      }

      error = CubDebug(cudaSetDevice(device));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    if (recached)
    {
      // Insert the ready event in the associated stream (must have current device set properly)
      error = CubDebug(cudaEventRecord(search_key.ready_event, search_key.associated_stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    if (!recached)
    {
      // Free the allocation from the runtime and cleanup the event.
      error = CubDebug(cudaFree(d_ptr));
      if (cudaSuccess != error)
      {
        return error;
      }

      error = CubDebug(cudaEventDestroy(search_key.ready_event));
      if (cudaSuccess != error)
      {
        return error;
      }

#ifdef CUB_DEBUG_LOG
      _CubLog("\tDevice %d freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld "
              "bytes), %lld live blocks (%lld bytes) outstanding.\n",
              device,
              (long long) search_key.bytes,
              (long long) search_key.associated_stream,
              (long long) cached_blocks.size(),
              (long long) cached_bytes[device].free,
              (long long) live_blocks.size(),
              (long long) cached_bytes[device].live);
#endif
    }

    // Reset device
    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
    {
      error = CubDebug(cudaSetDevice(entrypoint_device));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return error;
  }

  /**
   * @brief Frees a live allocation of device memory on the current device, returning it to the
   *        allocator.
   *
   * Once freed, the allocation becomes available immediately for reuse within the @p
   * active_stream with which it was associated with during allocation, and it becomes available
   * for reuse within other streams when all prior work submitted to @p active_stream has
   * completed.
   */
  cudaError_t DeviceFree(void* d_ptr)
  {
    return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
  }

  /**
   * @brief Frees all cached device allocations on all devices
   */
  cudaError_t FreeAllCached()
  {
    cudaError_t error     = cudaSuccess;
    int entrypoint_device = INVALID_DEVICE_ORDINAL;
    int current_device    = INVALID_DEVICE_ORDINAL;

    mutex.lock();

    while (!cached_blocks.empty())
    {
      // Get first block
      CachedBlocks::iterator begin = cached_blocks.begin();

      // Get entry-point device ordinal if necessary
      if (entrypoint_device == INVALID_DEVICE_ORDINAL)
      {
        error = CubDebug(cudaGetDevice(&entrypoint_device));
        if (cudaSuccess != error)
        {
          break;
        }
      }

      // Set current device ordinal if necessary
      if (begin->device != current_device)
      {
        error = CubDebug(cudaSetDevice(begin->device));
        if (cudaSuccess != error)
        {
          break;
        }
        current_device = begin->device;
      }

      // Free device memory
      error = CubDebug(cudaFree(begin->d_ptr));
      if (cudaSuccess != error)
      {
        break;
      }

      error = CubDebug(cudaEventDestroy(begin->ready_event));
      if (cudaSuccess != error)
      {
        break;
      }

      // Reduce balance and erase entry
      const size_t block_bytes = begin->bytes;
      cached_bytes[current_device].free -= block_bytes;
      cached_blocks.erase(begin);

#ifdef CUB_DEBUG_LOG
      _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld "
              "bytes) outstanding.\n",
              current_device,
              (long long) block_bytes,
              (long long) cached_blocks.size(),
              (long long) cached_bytes[current_device].free,
              (long long) live_blocks.size(),
              (long long) cached_bytes[current_device].live);
#endif
    }

    mutex.unlock();

    // Attempt to revert back to entry-point device if necessary
    if (entrypoint_device != INVALID_DEVICE_ORDINAL)
    {
      error = CubDebug(cudaSetDevice(entrypoint_device));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return error;
  }

  /**
   * @brief Destructor
   */
  virtual ~CachingDeviceAllocator()
  {
    if (!skip_cleanup)
    {
      FreeAllCached();
    }
  }
};

CUB_NAMESPACE_END
