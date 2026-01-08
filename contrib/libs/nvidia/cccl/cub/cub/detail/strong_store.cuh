/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * @file Utilities for strong memory operations.
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

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(uint4* ptr, uint4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.relaxed.gpu.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "r"(val.x),
                  "r"(val.y),
                  "r"(val.z),
                  "r"(val.w) : "memory");),
    (asm volatile(
       "st.cg.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(ulonglong2* ptr, ulonglong2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");),
               (asm volatile("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(ushort4* ptr, ushort4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.relaxed.gpu.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "h"(val.x),
                  "h"(val.y),
                  "h"(val.z),
                  "h"(val.w) : "memory");),
    (asm volatile(
       "st.cg.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(uint2* ptr, uint2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");),
               (asm volatile("st.cg.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned long long* ptr, unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");),
               (asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned int* ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");),
               (asm volatile("st.cg.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned short* ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");),
               (asm volatile("st.cg.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned char* ptr, unsigned char val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.relaxed.gpu.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");),
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.cg.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(uint4* ptr, uint4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "r"(val.x),
                  "r"(val.y),
                  "r"(val.z),
                  "r"(val.w) : "memory");),
    (__threadfence(); asm volatile(
       "st.cg.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(ulonglong2* ptr, ulonglong2 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");),
    (__threadfence(); asm volatile("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(ushort4* ptr, ushort4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "h"(val.x),
                  "h"(val.y),
                  "h"(val.z),
                  "h"(val.w) : "memory");),
    (__threadfence(); asm volatile(
       "st.cg.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(uint2* ptr, uint2 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");),
    (__threadfence(); asm volatile("st.cg.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned long long* ptr, unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned int* ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned short* ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned char* ptr, unsigned char val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.release.gpu.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");),
    (__threadfence(); asm volatile(
       "{"
       "  .reg .u8 datum;"
       "  cvt.u8.u16 datum, %1;"
       "  st.cg.u8 [%0], datum;"
       "}" : : "l"(ptr),
       "h"((unsigned short) val) : "memory");));
}

} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
