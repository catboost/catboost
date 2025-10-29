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
 * @file
 * Thread utilities for writing memory using PTX cache modifiers.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * @brief Enumeration of cache modifiers for memory store operations.
 */
enum CacheStoreModifier
{
  STORE_DEFAULT, ///< Default (no modifier)
  STORE_WB, ///< Cache write-back all coherent levels
  STORE_CG, ///< Cache at global level
  STORE_CS, ///< Cache streaming (likely to be accessed once)
  STORE_WT, ///< Cache write-through (to system memory)
  STORE_VOLATILE, ///< Volatile shared (any memory space)
};

/**
 * @name Thread I/O (cache modified)
 * @{
 */

/**
 * @brief Thread utility for writing memory using cub::CacheStoreModifier cache modifiers.
 *        Can be used to store any data type.
 *
 * @par Example
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/thread/thread_store.cuh>
 *
 * // 32-bit store using cache-global modifier:
 * int *d_out;
 * int val;
 * cub::ThreadStore<cub::STORE_CG>(d_out + threadIdx.x, val);
 *
 * // 16-bit store using default modifier
 * short *d_out;
 * short val;
 * cub::ThreadStore<cub::STORE_DEFAULT>(d_out + threadIdx.x, val);
 *
 * // 128-bit store using write-through modifier
 * float4 *d_out;
 * float4 val;
 * cub::ThreadStore<cub::STORE_WT>(d_out + threadIdx.x, val);
 *
 * // 96-bit store using cache-streaming cache modifier
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val;
 * cub::ThreadStore<cub::STORE_CS>(d_out + threadIdx.x, val);
 * @endcode
 *
 * @tparam MODIFIER
 *   <b>[inferred]</b> CacheStoreModifier enumeration
 *
 * @tparam InputIteratorT
 *   <b>[inferred]</b> Output iterator type \iterator
 *
 * @tparam T
 *   <b>[inferred]</b> Data type of output value
 */
template <CacheStoreModifier MODIFIER, typename OutputIteratorT, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore(OutputIteratorT itr, T val);

//@}  end member group

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
/// Helper structure for templated store iteration (inductive case)
template <int COUNT, int MAX>
struct iterate_thread_store
{
  template <CacheStoreModifier MODIFIER, typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void Store(T* ptr, T* vals)
  {
    ThreadStore<MODIFIER>(ptr + COUNT, vals[COUNT]);
    iterate_thread_store<COUNT + 1, MAX>::template Store<MODIFIER>(ptr, vals);
  }

  template <typename OutputIteratorT, typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void Dereference(OutputIteratorT ptr, T* vals)
  {
    ptr[COUNT] = vals[COUNT];
    iterate_thread_store<COUNT + 1, MAX>::Dereference(ptr, vals);
  }
};

/// Helper structure for templated store iteration (termination case)
template <int MAX>
struct iterate_thread_store<MAX, MAX>
{
  template <CacheStoreModifier MODIFIER, typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void Store(T* /*ptr*/, T* /*vals*/)
  {}

  template <typename OutputIteratorT, typename T>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void Dereference(OutputIteratorT /*ptr*/, T* /*vals*/)
  {}
};
} // namespace detail

/**
 * Define a uint4 (16B) ThreadStore specialization for the given Cache load modifier
 */
#  define _CUB_STORE_16(cub_modifier, ptx_modifier)                                                      \
    template <>                                                                                          \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, uint4*, uint4>(uint4 * ptr, uint4 val) \
    {                                                                                                    \
      asm volatile("st." #ptx_modifier ".v4.u32 [%0], {%1, %2, %3, %4};"                                 \
                   :                                                                                     \
                   : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));                          \
    }                                                                                                    \
    template <>                                                                                          \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, ulonglong2*, ulonglong2>(              \
      ulonglong2 * ptr, ulonglong2 val)                                                                  \
    {                                                                                                    \
      asm volatile("st." #ptx_modifier ".v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y));  \
    }

/**
 * Define a uint2 (8B) ThreadStore specialization for the given Cache load modifier
 */
#  define _CUB_STORE_8(cub_modifier, ptx_modifier)                                                               \
    template <>                                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, ushort4*, ushort4>(ushort4 * ptr, ushort4 val) \
    {                                                                                                            \
      asm volatile("st." #ptx_modifier ".v4.u16 [%0], {%1, %2, %3, %4};"                                         \
                   :                                                                                             \
                   : "l"(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w));                                  \
    }                                                                                                            \
    template <>                                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, uint2*, uint2>(uint2 * ptr, uint2 val)         \
    {                                                                                                            \
      asm volatile("st." #ptx_modifier ".v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y));          \
    }                                                                                                            \
    template <>                                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, unsigned long long*, unsigned long long>(      \
      unsigned long long* ptr, unsigned long long val)                                                           \
    {                                                                                                            \
      asm volatile("st." #ptx_modifier ".u64 [%0], %1;" : : "l"(ptr), "l"(val));                                 \
    }

/**
 * Define a unsigned int (4B) ThreadStore specialization for the given Cache load modifier
 */
#  define _CUB_STORE_4(cub_modifier, ptx_modifier)                                              \
    template <>                                                                                 \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, unsigned int*, unsigned int>( \
      unsigned int* ptr, unsigned int val)                                                      \
    {                                                                                           \
      asm volatile("st." #ptx_modifier ".u32 [%0], %1;" : : "l"(ptr), "r"(val));                \
    }

/**
 * Define a unsigned short (2B) ThreadStore specialization for the given Cache load modifier
 */
#  define _CUB_STORE_2(cub_modifier, ptx_modifier)                                                  \
    template <>                                                                                     \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, unsigned short*, unsigned short>( \
      unsigned short* ptr, unsigned short val)                                                      \
    {                                                                                               \
      asm volatile("st." #ptx_modifier ".u16 [%0], %1;" : : "l"(ptr), "h"(val));                    \
    }

/**
 * Define a unsigned char (1B) ThreadStore specialization for the given Cache load modifier
 */
#  define _CUB_STORE_1(cub_modifier, ptx_modifier)                                                \
    template <>                                                                                   \
    _CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore<cub_modifier, unsigned char*, unsigned char>( \
      unsigned char* ptr, unsigned char val)                                                      \
    {                                                                                             \
      asm volatile(                                                                               \
        "{"                                                                                       \
        "   .reg .u8 datum;"                                                                      \
        "   cvt.u8.u16 datum, %1;"                                                                \
        "   st." #ptx_modifier ".u8 [%0], datum;"                                                 \
        "}"                                                                                       \
        :                                                                                         \
        : "l"(ptr), "h"((unsigned short) val));                                                   \
    }

/**
 * Define powers-of-two ThreadStore specializations for the given Cache load modifier
 */
#  define _CUB_STORE_ALL(cub_modifier, ptx_modifier) \
    _CUB_STORE_16(cub_modifier, ptx_modifier)        \
    _CUB_STORE_8(cub_modifier, ptx_modifier)         \
    _CUB_STORE_4(cub_modifier, ptx_modifier)         \
    _CUB_STORE_2(cub_modifier, ptx_modifier)         \
    _CUB_STORE_1(cub_modifier, ptx_modifier)

/**
 * Define ThreadStore specializations for the various Cache load modifiers
 */
_CUB_STORE_ALL(STORE_WB, wb)
_CUB_STORE_ALL(STORE_CG, cg)
_CUB_STORE_ALL(STORE_CS, cs)
_CUB_STORE_ALL(STORE_WT, wt)

// Macro cleanup
#  undef _CUB_STORE_ALL
#  undef _CUB_STORE_1
#  undef _CUB_STORE_2
#  undef _CUB_STORE_4
#  undef _CUB_STORE_8
#  undef _CUB_STORE_16

/**
 * ThreadStore definition for STORE_DEFAULT modifier on iterator types
 */
template <typename OutputIteratorT, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore(
  OutputIteratorT itr, T val, detail::constant_t<STORE_DEFAULT> /*modifier*/, ::cuda::std::false_type /*is_pointer*/)
{
  *itr = val;
}

/**
 * ThreadStore definition for STORE_DEFAULT modifier on pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void
ThreadStore(T* ptr, T val, detail::constant_t<STORE_DEFAULT> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  *ptr = val;
}

/**
 * ThreadStore definition for STORE_VOLATILE modifier on primitive pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStoreVolatilePtr(T* ptr, T val, ::cuda::std::true_type /*is_primitive*/)
{
  *reinterpret_cast<volatile T*>(ptr) = val;
}

/**
 * ThreadStore definition for STORE_VOLATILE modifier on non-primitive pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStoreVolatilePtr(T* ptr, T val, ::cuda::std::false_type /*is_primitive*/)
{
  // Create a temporary using shuffle-words, then store using volatile-words
  using VolatileWord = typename UnitWord<T>::VolatileWord;
  using ShuffleWord  = typename UnitWord<T>::ShuffleWord;

  constexpr int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);
  constexpr int SHUFFLE_MULTIPLE  = sizeof(T) / sizeof(ShuffleWord);

  VolatileWord words[VOLATILE_MULTIPLE];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < SHUFFLE_MULTIPLE; ++i)
  {
    reinterpret_cast<ShuffleWord*>(words)[i] = reinterpret_cast<ShuffleWord*>(&val)[i];
  }

  detail::iterate_thread_store<0, VOLATILE_MULTIPLE>::Dereference(reinterpret_cast<volatile VolatileWord*>(ptr), words);
}

/**
 * ThreadStore definition for STORE_VOLATILE modifier on pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void
ThreadStore(T* ptr, T val, detail::constant_t<STORE_VOLATILE> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  ThreadStoreVolatilePtr(ptr, val, detail::bool_constant_v<detail::is_primitive<T>::value>);
}

/**
 * ThreadStore definition for generic modifiers on pointer types
 */
template <typename T, CacheStoreModifier MODIFIER>
_CCCL_DEVICE _CCCL_FORCEINLINE void
ThreadStore(T* ptr, T val, detail::constant_t<MODIFIER> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  // Create a temporary using shuffle-words, then store using device-words
  using DeviceWord  = typename UnitWord<T>::DeviceWord;
  using ShuffleWord = typename UnitWord<T>::ShuffleWord;

  constexpr int DEVICE_MULTIPLE  = sizeof(T) / sizeof(DeviceWord);
  constexpr int SHUFFLE_MULTIPLE = sizeof(T) / sizeof(ShuffleWord);

  DeviceWord words[DEVICE_MULTIPLE];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < SHUFFLE_MULTIPLE; ++i)
  {
    reinterpret_cast<ShuffleWord*>(words)[i] = reinterpret_cast<ShuffleWord*>(&val)[i];
  }

  detail::iterate_thread_store<0, DEVICE_MULTIPLE>::template Store<CacheStoreModifier(MODIFIER)>(
    reinterpret_cast<DeviceWord*>(ptr), words);
}

/**
 * ThreadStore definition for generic modifiers
 */
template <CacheStoreModifier MODIFIER, typename OutputIteratorT, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadStore(OutputIteratorT itr, T val)
{
  ThreadStore(
    itr, val, detail::constant_v<MODIFIER>, detail::bool_constant_v<::cuda::std::is_pointer_v<OutputIteratorT>>);
}

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
