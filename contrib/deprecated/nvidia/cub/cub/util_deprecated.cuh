/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Define CUB_DEPRECATED macro.
 */

#pragma once
#pragma clang system_header



#include <cub/detail/type_traits.cuh>
#include <cub/util_compiler.cuh>
#include <cub/util_cpp_dialect.cuh>
#include <cub/util_debug.cuh>


#if defined(THRUST_IGNORE_DEPRECATED_API) && !defined(CUB_IGNORE_DEPRECATED_API)
#  define CUB_IGNORE_DEPRECATED_API
#endif

#ifdef CUB_IGNORE_DEPRECATED_API
#  define CUB_DEPRECATED
#  define CUB_DEPRECATED_BECAUSE(MSG)
#elif CUB_CPP_DIALECT >= 2014
#  define CUB_DEPRECATED [[deprecated]]
#  define CUB_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC
#  define CUB_DEPRECATED __declspec(deprecated)
#  define CUB_DEPRECATED_BECAUSE(MSG) __declspec(deprecated(MSG))
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_CLANG
#  define CUB_DEPRECATED __attribute__((deprecated))
#  define CUB_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#elif CUB_HOST_COMPILER == CUB_HOST_COMPILER_GCC
#  define CUB_DEPRECATED __attribute__((deprecated))
#  define CUB_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#else
#  define CUB_DEPRECATED
#  define CUB_DEPRECATED_BECAUSE(MSG)
#endif

#define CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED                         \
  CUB_DEPRECATED_BECAUSE(                                                      \
    "CUB no longer accepts `debug_synchronous` parameter. "                    \
    "Define CUB_DEBUG_SYNC instead, or silence this message with "             \
    "CUB_IGNORE_DEPRECATED_API.")

#define CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG                                \
  if (debug_synchronous)                                                       \
  {                                                                            \
    _CubLog("%s\n",                                                            \
            "CUB no longer accepts `debug_synchronous` parameter. "            \
            "Define CUB_DEBUG_SYNC instead.");                                 \
  }
