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
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/detail/cpp_compatibility.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace detail
{

static __device__ __forceinline__ void store_relaxed(uint4 *ptr, uint4 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v4.u32 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                             : "memory");),
               (asm volatile("st.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(ulonglong2 *ptr, ulonglong2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u64 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val.x), "l"(val.y)
                             : "memory");),
               (asm volatile("st.cg.v2.u64 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val.x), "l"(val.y)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(ushort4 *ptr, ushort4 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v4.u16 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w)
                             : "memory");),
               (asm volatile("st.cg.v4.u16 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(uint2 *ptr, uint2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y)
                             : "memory");),
               (asm volatile("st.cg.v2.u32 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(unsigned long long *ptr,
                                                     unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u64 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val)
                             : "memory");),
               (asm volatile("st.cg.u64 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(unsigned int *ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u32 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val)
                             : "memory");),
               (asm volatile("st.cg.u32 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(unsigned short *ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u16 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val)
                             : "memory");),
               (asm volatile("st.cg.u16 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val)
                             : "memory");));
}

static __device__ __forceinline__ void store_relaxed(unsigned char *ptr, unsigned char val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("{"
                             "  .reg .u8 datum;"
                             "  cvt.u8.u16 datum, %1;"
                             "  st.relaxed.gpu.u8 [%0], datum;"
                             "}"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"((unsigned short)val)
                             : "memory");),
               (asm volatile("{"
                             "  .reg .u8 datum;"
                             "  cvt.u8.u16 datum, %1;"
                             "  st.cg.u8 [%0], datum;"
                             "}"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"((unsigned short)val)
                             : "memory");));
}

__device__ __forceinline__ void store_release(uint4 *ptr, uint4 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.v4.u32 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                             : "memory");),
               (__threadfence();
                asm volatile("st.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                             : "memory");));
}

__device__ __forceinline__ void store_release(ulonglong2 *ptr, ulonglong2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.v2.u64 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val.x), "l"(val.y)
                             : "memory");),
               (__threadfence(); asm volatile("st.cg.v2.u64 [%0], {%1, %2};"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "l"(val.x), "l"(val.y)
                                              : "memory");));
}

__device__ __forceinline__ void store_release(ushort4 *ptr, ushort4 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.v4.u16 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w)
                             : "memory");),
               (__threadfence();
                asm volatile("st.cg.v4.u16 [%0], {%1, %2, %3, %4};"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w)
                             : "memory");));
}

__device__ __forceinline__ void store_release(uint2 *ptr, uint2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.v2.u32 [%0], {%1, %2};"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y)
                             : "memory");),
               (__threadfence(); asm volatile("st.cg.v2.u32 [%0], {%1, %2};"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "r"(val.x), "r"(val.y)
                                              : "memory");));
}

__device__ __forceinline__ void store_release(unsigned long long *ptr, unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u64 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "l"(val)
                             : "memory");),
               (__threadfence(); asm volatile("st.cg.u64 [%0], %1;"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "l"(val)
                                              : "memory");));
}

__device__ __forceinline__ void store_release(unsigned int *ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u32 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "r"(val)
                             : "memory");),
               (__threadfence(); asm volatile("st.cg.u32 [%0], %1;"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "r"(val)
                                              : "memory");));
}

__device__ __forceinline__ void store_release(unsigned short *ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u16 [%0], %1;"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"(val)
                             : "memory");),
               (__threadfence(); asm volatile("st.cg.u16 [%0], %1;"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "h"(val)
                                              : "memory");));
}

__device__ __forceinline__ void store_release(unsigned char *ptr, unsigned char val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("{"
                             "  .reg .u8 datum;"
                             "  cvt.u8.u16 datum, %1;"
                             "  st.release.gpu.u8 [%0], datum;"
                             "}"
                             :
                             : _CUB_ASM_PTR_(ptr), "h"((unsigned short)val)
                             : "memory");),
               (__threadfence(); asm volatile("{"
                                              "  .reg .u8 datum;"
                                              "  cvt.u8.u16 datum, %1;"
                                              "  st.cg.u8 [%0], datum;"
                                              "}"
                                              :
                                              : _CUB_ASM_PTR_(ptr), "h"((unsigned short)val)
                                              : "memory");));
}

} // namespace detail

#endif // DOXYGEN_SHOULD_SKIP_THIS

CUB_NAMESPACE_END
