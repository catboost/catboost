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
#pragma clang system_header


// This is not used by this file; this is a hack so that we can detect the
// CUB version from Thrust on older versions of CUB that did not have
// version.cuh.
#include "version.cuh"

// Prior to 1.13.1, only the PREFIX/POSTFIX macros were used. Notify users
// that they must now define the qualifier macro, too.
#if (defined(CUB_NS_PREFIX) || defined(CUB_NS_POSTFIX)) && !defined(CUB_NS_QUALIFIER)
#error CUB requires a definition of CUB_NS_QUALIFIER when CUB_NS_PREFIX/POSTFIX are defined.
#endif

/**
 * \def THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` and `cub::` namespaces.
 * This macro should not be used with any other CUB namespace macros.
 */
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `cub::` namespace.
 * If THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other CUB namespace macros.
 */
#ifdef CUB_WRAPPED_NAMESPACE
#define CUB_NS_PREFIX                                                       \
  namespace CUB_WRAPPED_NAMESPACE                                           \
  {

#define CUB_NS_POSTFIX }

#define CUB_NS_QUALIFIER ::CUB_WRAPPED_NAMESPACE::cub
#endif

/**
 * \def CUB_NS_PREFIX
 * This macro is inserted prior to all `namespace cub { ... }` blocks. It is
 * derived from CUB_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case CUB_NS_PREFIX,
 * CUB_NS_POSTFIX, and CUB_NS_QUALIFIER must all be set consistently.
 */
#ifndef CUB_NS_PREFIX
#define CUB_NS_PREFIX
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
#define CUB_NS_POSTFIX
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
#define CUB_NS_QUALIFIER ::cub
#endif

#if !defined(CUB_DETAIL_MAGIC_NS_NAME)
#define CUB_DETAIL_COUNT_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                           _14, _15, _16, _17, _18, _19, _20, N, ...)              \
                           N
#define CUB_DETAIL_COUNT(...)                                                      \
  CUB_DETAIL_IDENTITY(CUB_DETAIL_COUNT_N(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, \
                                         11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define CUB_DETAIL_IDENTITY(N) N
#define CUB_DETAIL_APPLY(MACRO, ...) CUB_DETAIL_IDENTITY(MACRO(__VA_ARGS__))
#define CUB_DETAIL_MAGIC_NS_NAME1(P1) \
    CUB_##P1##_NS
#define CUB_DETAIL_MAGIC_NS_NAME2(P1, P2) \
    CUB_##P1##_##P2##_NS
#define CUB_DETAIL_MAGIC_NS_NAME3(P1, P2, P3) \
    CUB_##P1##_##P2##_##P3##_NS
#define CUB_DETAIL_MAGIC_NS_NAME4(P1, P2, P3, P4) \
    CUB_##P1##_##P2##_##P3##_##P4##_NS
#define CUB_DETAIL_MAGIC_NS_NAME5(P1, P2, P3, P4, P5) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_NS
#define CUB_DETAIL_MAGIC_NS_NAME6(P1, P2, P3, P4, P5, P6) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_NS
#define CUB_DETAIL_MAGIC_NS_NAME7(P1, P2, P3, P4, P5, P6, P7) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_NS
#define CUB_DETAIL_MAGIC_NS_NAME8(P1, P2, P3, P4, P5, P6, P7, P8) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_NS
#define CUB_DETAIL_MAGIC_NS_NAME9(P1, P2, P3, P4, P5, P6, P7, P8, P9) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_NS
#define CUB_DETAIL_MAGIC_NS_NAME10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_NS
#define CUB_DETAIL_MAGIC_NS_NAME11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_NS
#define CUB_DETAIL_MAGIC_NS_NAME12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_NS
#define CUB_DETAIL_MAGIC_NS_NAME13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_NS
#define CUB_DETAIL_MAGIC_NS_NAME14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_NS
#define CUB_DETAIL_MAGIC_NS_NAME15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_NS
#define CUB_DETAIL_MAGIC_NS_NAME16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_NS
#define CUB_DETAIL_MAGIC_NS_NAME17(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_NS
#define CUB_DETAIL_MAGIC_NS_NAME18(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_NS
#define CUB_DETAIL_MAGIC_NS_NAME19(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_##P19##_NS
#define CUB_DETAIL_MAGIC_NS_NAME20(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20) \
    CUB_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_##P19##_##P20##_NS
#define CUB_DETAIL_DISPATCH(N) CUB_DETAIL_MAGIC_NS_NAME ## N
#define CUB_DETAIL_MAGIC_NS_NAME(...) CUB_DETAIL_IDENTITY(CUB_DETAIL_APPLY(CUB_DETAIL_DISPATCH, CUB_DETAIL_COUNT(__VA_ARGS__))(__VA_ARGS__))
#endif // !defined(CUB_DETAIL_MAGIC_NS_NAME)

#if defined(CUB_DISABLE_NAMESPACE_MAGIC)
#if !defined(CUB_WRAPPED_NAMESPACE)
#if !defined(CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
#error "Disabling namespace magic is unsafe without wrapping namespace"
#endif // !defined(CUB_IGNORE_NAMESPACE_MAGIC_ERROR)
#endif // !defined(CUB_WRAPPED_NAMESPACE)
#define CUB_DETAIL_MAGIC_NS_BEGIN
#define CUB_DETAIL_MAGIC_NS_END
#else // not defined(CUB_DISABLE_NAMESPACE_MAGIC)
#if defined(_NVHPC_CUDA)
#define CUB_DETAIL_MAGIC_NS_BEGIN inline namespace CUB_DETAIL_MAGIC_NS_NAME(CUB_VERSION, NV_TARGET_SM_INTEGER_LIST) {
#define CUB_DETAIL_MAGIC_NS_END }
#else // not defined(_NVHPC_CUDA)
#define CUB_DETAIL_MAGIC_NS_BEGIN inline namespace CUB_DETAIL_MAGIC_NS_NAME(CUB_VERSION, __CUDA_ARCH_LIST__) {
#define CUB_DETAIL_MAGIC_NS_END }
#endif // not defined(_NVHPC_CUDA)
#endif // not defined(CUB_DISABLE_NAMESPACE_MAGIC)

/**
 * \def CUB_NAMESPACE_BEGIN
 * This macro is used to open a `cub::` namespace block, along with any
 * enclosing namespaces requested by CUB_WRAPPED_NAMESPACE, etc.
 * This macro is defined by CUB and may not be overridden.
 */
#define CUB_NAMESPACE_BEGIN                                                 \
  CUB_NS_PREFIX                                                             \
  namespace cub                                                             \
  {                                                                         \
  CUB_DETAIL_MAGIC_NS_BEGIN                                                        

/**
 * \def CUB_NAMESPACE_END
 * This macro is used to close a `cub::` namespace block, along with any
 * enclosing namespaces requested by CUB_WRAPPED_NAMESPACE, etc.
 * This macro is defined by CUB and may not be overridden.
 */
#define CUB_NAMESPACE_END                                                   \
  CUB_DETAIL_MAGIC_NS_END                                                   \
  } /* end namespace cub */                                                 \
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
