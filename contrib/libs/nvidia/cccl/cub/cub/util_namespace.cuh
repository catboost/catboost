/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

/**
 * \file util_namespace.cuh
 * \brief Utilities that allow `cub::` to be placed inside an
 * application-specific namespace.
 */

#pragma once

// This is not used by this file; this is a hack so that we can detect the
// CUB version from Thrust on older versions of CUB that did not have
// version.cuh.
#include <cub/version.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/detect_cuda_runtime.cuh>

// Prior to 1.13.1, only the PREFIX/POSTFIX macros were used. Notify users
// that they must now define the qualifier macro, too.
#if (defined(CUB_NS_PREFIX) || defined(CUB_NS_POSTFIX)) && !defined(CUB_NS_QUALIFIER)
#  error CUB requires a definition of CUB_NS_QUALIFIER when CUB_NS_PREFIX/POSTFIX are defined.
#endif

/**
 * \def THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` and `cub::` namespaces.
 * This macro should not be used with any other CUB namespace macros.
 */
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#  define CUB_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `cub::` namespace.
 * If THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other CUB namespace macros.
 */
#ifdef CUB_WRAPPED_NAMESPACE
#  define CUB_NS_PREFIX             \
    namespace CUB_WRAPPED_NAMESPACE \
    {

#  define CUB_NS_POSTFIX }

#  define CUB_NS_QUALIFIER ::CUB_WRAPPED_NAMESPACE::cub
#endif

/**
 * \def CUB_NS_PREFIX
 * This macro is inserted prior to all `namespace cub { ... }` blocks. It is
 * derived from CUB_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case CUB_NS_PREFIX,
 * CUB_NS_POSTFIX, and CUB_NS_QUALIFIER must all be set consistently.
 */
#ifndef CUB_NS_PREFIX
#  define CUB_NS_PREFIX
#endif

/**
 * \def CUB_NS_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace cub { ... }` block. It is defined appropriately when
 * CUB_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case CUB_NS_PREFIX, CUB_NS_POSTFIX, and
 * CUB_NS_QUALIFIER must all be set consistently.
 */
#ifndef CUB_NS_POSTFIX
#  define CUB_NS_POSTFIX
#endif

/**
 * \def CUB_NS_QUALIFIER
 * This macro is used to qualify members of cub:: when accessing them from
 * outside of their namespace. By default, this is just `::cub`, and will be
 * set appropriately when CUB_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case CUB_NS_PREFIX, CUB_NS_POSTFIX, and
 * CUB_NS_QUALIFIER must all be set consistently.
 */
#ifndef CUB_NS_QUALIFIER
#  define CUB_NS_QUALIFIER ::cub
#endif

#if defined(CUB_DISABLE_NAMESPACE_MAGIC) || defined(CUB_WRAPPED_NAMESPACE)
#  if !defined(CUB_WRAPPED_NAMESPACE)
#    if !defined(CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
#      error "Disabling namespace magic is unsafe without wrapping namespace"
#    endif // !defined(CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
#  endif // !defined(CUB_WRAPPED_NAMESPACE)
#  define CUB_DETAIL_MAGIC_NS_BEGIN
#  define CUB_DETAIL_MAGIC_NS_END
#else // not defined(CUB_DISABLE_NAMESPACE_MAGIC)
#  if defined(_NVHPC_CUDA)
#    define CUB_DETAIL_MAGIC_NS_BEGIN                                                                  \
      inline namespace _CCCL_PP_SPLICE_WITH(_, CUB, CUB_VERSION, SM, NV_TARGET_SM_INTEGER_LIST, NVHPC) \
      {
#    define CUB_DETAIL_MAGIC_NS_END }
#  else // not defined(_NVHPC_CUDA)
#    define CUB_DETAIL_MAGIC_NS_BEGIN                                                    \
      inline namespace _CCCL_PP_SPLICE_WITH(_, CUB, CUB_VERSION, SM, __CUDA_ARCH_LIST__) \
      {
#    define CUB_DETAIL_MAGIC_NS_END }
#  endif // not defined(_NVHPC_CUDA)
#endif // not defined(CUB_DISABLE_NAMESPACE_MAGIC)

/**
 * \def CUB_NAMESPACE_BEGIN
 * This macro is used to open a `cub::` namespace block, along with any
 * enclosing namespaces requested by CUB_WRAPPED_NAMESPACE, etc.
 * This macro is defined by CUB and may not be overridden.
 */
#define CUB_NAMESPACE_BEGIN \
  CUB_NS_PREFIX             \
  namespace cub             \
  {                         \
  CUB_DETAIL_MAGIC_NS_BEGIN

/**
 * \def CUB_NAMESPACE_END
 * This macro is used to close a `cub::` namespace block, along with any
 * enclosing namespaces requested by CUB_WRAPPED_NAMESPACE, etc.
 * This macro is defined by CUB and may not be overridden.
 */
#define CUB_NAMESPACE_END   \
  CUB_DETAIL_MAGIC_NS_END   \
  } /* end namespace cub */ \
  CUB_NS_POSTFIX

// Declare these namespaces here for the purpose of Doxygenating them
CUB_NS_PREFIX

/*! \namespace cub
 *  \brief \p cub is the top-level namespace which contains all CUB
 *         functions and types.
 */
namespace cub
{
}

CUB_NS_POSTFIX
