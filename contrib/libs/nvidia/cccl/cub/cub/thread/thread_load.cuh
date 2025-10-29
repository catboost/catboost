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
 * Thread utilities for reading memory using PTX cache modifiers.
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
 * @brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier
{
  LOAD_DEFAULT, ///< Default (no modifier)
  LOAD_CA, ///< Cache at all levels
  LOAD_CG, ///< Cache at global level
  LOAD_CS, ///< Cache streaming (likely to be accessed once)
  LOAD_CV, ///< Cache as volatile (including cached system lines)
  LOAD_LDG, ///< Cache as texture
  LOAD_VOLATILE, ///< Volatile (any memory space)
};

/**
 * @name Thread I/O (cache modified)
 * @{
 */

/**
 * @brief Thread utility for reading memory using cub::CacheLoadModifier cache modifiers.
 *        Can be used to load any data type.
 *
 * @par Example
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/thread/thread_load.cuh>
 *
 * // 32-bit load using cache-global modifier:
 * int *d_in;
 * int val = cub::ThreadLoad<cub::LOAD_CA>(d_in + threadIdx.x);
 *
 * // 16-bit load using default modifier
 * short *d_in;
 * short val = cub::ThreadLoad<cub::LOAD_DEFAULT>(d_in + threadIdx.x);
 *
 * // 128-bit load using cache-volatile modifier
 * float4 *d_in;
 * float4 val = cub::ThreadLoad<cub::LOAD_CV>(d_in + threadIdx.x);
 *
 * // 96-bit load using cache-streaming modifier
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val = cub::ThreadLoad<cub::LOAD_CS>(d_in + threadIdx.x);
 * \endcode
 *
 * @tparam MODIFIER
 *   <b>[inferred]</b> CacheLoadModifier enumeration
 *
 * @tparam RandomAccessIterator
 *   <b>[inferred]</b> The input's iterator type \iterator
 */
template <CacheLoadModifier MODIFIER, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE cub::detail::it_value_t<RandomAccessIterator> ThreadLoad(RandomAccessIterator itr);

//@}  end member group

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
template <CacheLoadModifier MODIFIER, typename T, int... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE void
UnrolledThreadLoadImpl(T const* src, T* dst, ::cuda::std::integer_sequence<int, Is...>)
{
  ((dst[Is] = ThreadLoad<MODIFIER>(src + Is)), ...);
}

template <typename RandomAccessIterator, typename T, int... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE void
UnrolledCopyImpl(RandomAccessIterator src, T* dst, ::cuda::std::integer_sequence<int, Is...>)
{
  ((dst[Is] = src[Is]), ...);
}

} // namespace detail

template <int Count, CacheLoadModifier MODIFIER, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void UnrolledThreadLoad(T const* src, T* dst)
{
  detail::UnrolledThreadLoadImpl<MODIFIER>(src, dst, ::cuda::std::make_integer_sequence<int, Count>{});
}

template <int Count, typename RandomAccessIterator, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void UnrolledCopy(RandomAccessIterator src, T* dst)
{
  detail::UnrolledCopyImpl(src, dst, ::cuda::std::make_integer_sequence<int, Count>{});
}

/**
 * Define a uint4 (16B) ThreadLoad specialization for the given Cache load modifier
 */
#  define _CUB_LOAD_16(cub_modifier, ptx_modifier)                                                               \
    template <>                                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE uint4 ThreadLoad<cub_modifier, uint4 const*>(uint4 const* ptr)                \
    {                                                                                                            \
      uint4 retval;                                                                                              \
      asm volatile("ld." #ptx_modifier ".v4.u32 {%0, %1, %2, %3}, [%4];"                                         \
                   : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w)                              \
                   : "l"(ptr));                                                                                  \
      return retval;                                                                                             \
    }                                                                                                            \
    template <>                                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE ulonglong2 ThreadLoad<cub_modifier, ulonglong2 const*>(ulonglong2 const* ptr) \
    {                                                                                                            \
      ulonglong2 retval;                                                                                         \
      asm volatile("ld." #ptx_modifier ".v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));   \
      return retval;                                                                                             \
    }

/**
 * Define a uint2 (8B) ThreadLoad specialization for the given Cache load modifier
 */
#  define _CUB_LOAD_8(cub_modifier, ptx_modifier)                                                              \
    template <>                                                                                                \
    _CCCL_DEVICE _CCCL_FORCEINLINE ushort4 ThreadLoad<cub_modifier, ushort4 const*>(ushort4 const* ptr)        \
    {                                                                                                          \
      ushort4 retval;                                                                                          \
      asm volatile("ld." #ptx_modifier ".v4.u16 {%0, %1, %2, %3}, [%4];"                                       \
                   : "=h"(retval.x), "=h"(retval.y), "=h"(retval.z), "=h"(retval.w)                            \
                   : "l"(ptr));                                                                                \
      return retval;                                                                                           \
    }                                                                                                          \
    template <>                                                                                                \
    _CCCL_DEVICE _CCCL_FORCEINLINE uint2 ThreadLoad<cub_modifier, uint2 const*>(uint2 const* ptr)              \
    {                                                                                                          \
      uint2 retval;                                                                                            \
      asm volatile("ld." #ptx_modifier ".v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr)); \
      return retval;                                                                                           \
    }                                                                                                          \
    template <>                                                                                                \
    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long ThreadLoad<cub_modifier, unsigned long long const*>(     \
      unsigned long long const* ptr)                                                                           \
    {                                                                                                          \
      unsigned long long retval;                                                                               \
      asm volatile("ld." #ptx_modifier ".u64 %0, [%1];" : "=l"(retval) : "l"(ptr));                            \
      return retval;                                                                                           \
    }

/**
 * Define a uint (4B) ThreadLoad specialization for the given Cache load modifier
 */
#  define _CUB_LOAD_4(cub_modifier, ptx_modifier)                                                                      \
    template <>                                                                                                        \
    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int ThreadLoad<cub_modifier, unsigned int const*>(unsigned int const* ptr) \
    {                                                                                                                  \
      unsigned int retval;                                                                                             \
      asm volatile("ld." #ptx_modifier ".u32 %0, [%1];" : "=r"(retval) : "l"(ptr));                                    \
      return retval;                                                                                                   \
    }

/**
 * Define a unsigned short (2B) ThreadLoad specialization for the given Cache load modifier
 */
#  define _CUB_LOAD_2(cub_modifier, ptx_modifier)                                                  \
    template <>                                                                                    \
    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned short ThreadLoad<cub_modifier, unsigned short const*>( \
      unsigned short const* ptr)                                                                   \
    {                                                                                              \
      unsigned short retval;                                                                       \
      asm volatile("ld." #ptx_modifier ".u16 %0, [%1];" : "=h"(retval) : "l"(ptr));                \
      return retval;                                                                               \
    }

/**
 * Define an unsigned char (1B) ThreadLoad specialization for the given Cache load modifier
 */
#  define _CUB_LOAD_1(cub_modifier, ptx_modifier)                                                \
    template <>                                                                                  \
    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned char ThreadLoad<cub_modifier, unsigned char const*>( \
      unsigned char const* ptr)                                                                  \
    {                                                                                            \
      unsigned short retval;                                                                     \
      asm volatile(                                                                              \
        "{"                                                                                      \
        "   .reg .u8 datum;"                                                                     \
        "    ld." #ptx_modifier ".u8 datum, [%1];"                                               \
        "    cvt.u16.u8 %0, datum;"                                                              \
        "}"                                                                                      \
        : "=h"(retval)                                                                           \
        : "l"(ptr));                                                                             \
      return (unsigned char) retval;                                                             \
    }

/**
 * Define powers-of-two ThreadLoad specializations for the given Cache load modifier
 */
#  define _CUB_LOAD_ALL(cub_modifier, ptx_modifier) \
    _CUB_LOAD_16(cub_modifier, ptx_modifier)        \
    _CUB_LOAD_8(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_4(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_2(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_1(cub_modifier, ptx_modifier)

/**
 * Define powers-of-two ThreadLoad specializations for the various Cache load modifiers
 */
_CUB_LOAD_ALL(LOAD_CA, ca)
_CUB_LOAD_ALL(LOAD_CG, cg)
_CUB_LOAD_ALL(LOAD_CS, cs)
_CUB_LOAD_ALL(LOAD_CV, cv)
_CUB_LOAD_ALL(LOAD_LDG, global.nc)

// Macro cleanup
#  undef _CUB_LOAD_ALL
#  undef _CUB_LOAD_1
#  undef _CUB_LOAD_2
#  undef _CUB_LOAD_4
#  undef _CUB_LOAD_8
#  undef _CUB_LOAD_16

/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on iterator types
 */
template <typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE cub::detail::it_value_t<RandomAccessIterator> ThreadLoad(
  RandomAccessIterator itr, detail::constant_t<LOAD_DEFAULT> /*modifier*/, ::cuda::std::false_type /*is_pointer*/)
{
  return *itr;
}

/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadLoad(const T* ptr, detail::constant_t<LOAD_DEFAULT> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  return *ptr;
}

/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on primitive pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadLoadVolatilePointer(const T* ptr, ::cuda::std::true_type /*is_primitive*/)
{
  T retval = *reinterpret_cast<const volatile T*>(ptr);
  return retval;
}

/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on non-primitive pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadLoadVolatilePointer(const T* ptr, ::cuda::std::false_type /*is_primitive*/)
{
  // Word type for memcpying
  using VolatileWord              = typename UnitWord<T>::VolatileWord;
  constexpr int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);

  T retval;
  VolatileWord* words = reinterpret_cast<VolatileWord*>(&retval);
  UnrolledCopy<VOLATILE_MULTIPLE>(reinterpret_cast<const volatile VolatileWord*>(ptr), words);
  return retval;
}

/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on pointer types
 */
template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadLoad(const T* ptr, detail::constant_t<LOAD_VOLATILE> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  return ThreadLoadVolatilePointer(ptr, detail::bool_constant_v<detail::is_primitive_v<T>>);
}

/**
 * ThreadLoad definition for generic modifiers on pointer types
 */
template <typename T, CacheLoadModifier MODIFIER>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadLoad(T const* ptr, detail::constant_t<MODIFIER> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  using DeviceWord              = typename UnitWord<T>::DeviceWord;
  constexpr int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);

  DeviceWord words[DEVICE_MULTIPLE];
  UnrolledThreadLoad<DEVICE_MULTIPLE, CacheLoadModifier(MODIFIER)>(reinterpret_cast<const DeviceWord*>(ptr), words);
  return *reinterpret_cast<T*>(words);
}

template <CacheLoadModifier MODIFIER, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE cub::detail::it_value_t<RandomAccessIterator> ThreadLoad(RandomAccessIterator itr)
{
  return ThreadLoad(
    itr, detail::constant_v<MODIFIER>, detail::bool_constant_v<::cuda::std::is_pointer_v<RandomAccessIterator>>);
}

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
