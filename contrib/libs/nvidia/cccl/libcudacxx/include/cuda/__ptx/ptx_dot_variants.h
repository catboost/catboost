// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// WARNING: The source of truth for this file is libcuda-ptx. Do not modify without syncing with libcuda-ptx.

#ifndef _CUDA_PTX_DOT_VARIANTS_H_
#define _CUDA_PTX_DOT_VARIANTS_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

/*
 * Public integral constant types and values for ".variant"s:
 *
 * - .sem:     acquire, release, ..
 * - .space:   global, shared, constant, ..
 * - .scope:   cta, cluster, gpu, ..
 * - .op:      add, min, cas, ..
 *
 * For each .variant, the code below defines:
 * - An enum `dot_variant` with each possible value
 * - A type template `variant_t<dot_variant>`
 * - Types `variant_A_t`, ..., `variant_Z_t`
 * - Constexpr values `variant_A` of type `variant_A_t`
 *
 * These types enable specifying fine-grained overloads of a PTX binding. If a
 * binding can handle multiple variants, then it is defined as:
 *
 * template <dot_variant var>
 * [...] void ptx_binding(variant_t<var> __v) { ... }
 *
 * If it only handles a single variant, then it is defined as:
 *
 * [...] void ptx_binding(variant_A __v) { ... }
 *
 * If two variants have different behaviors or return types (see .space
 * overloads of mbarrier.arrive.expect_tx for an example), then these can be
 * provided as separate overloads of the same function:
 *
 * [...] void ptx_binding(variant_A __v) { ... }
 * [...] int ptx_binding(variant_B __v) { ... }
 *
 */

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operation-types
enum class dot_sem
{
  acq_rel,
  acquire,
  relaxed,
  release,
  sc,
  weak
};

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces
enum class dot_space
{
  global,
  cluster, // The PTX spelling is shared::cluster
  shared, // The PTX spelling is shared::cta

  // The following state spaces are unlikely to be used in cuda::ptx in the near
  // future, so they are not exposed:

  // reg,
  // sreg,
  // const_mem, // Using const_mem as `const` is reserved in C++.
  // local,
  // param,
  // tex // deprecated
};

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope
enum class dot_scope
{
  cta,
  cluster,
  gpu,
  sys
};

enum class dot_op
{
  add,
  dec,
  inc,
  max,
  min,
  and_op, // Using and_op, as `and, or, xor` are reserved in C++.
  or_op,
  xor_op,
  cas,
  exch
};

enum class dot_cta_group
{
  cta_group_1,
  cta_group_2
};

enum class dot_kind
{
  f16,
  f8f6f4,
  i8,
  mxf4,
  mxf4nvf4,
  mxf8f6f4,
  tf32
};

template <dot_sem __sem>
using sem_t         = _CUDA_VSTD::integral_constant<dot_sem, __sem>;
using sem_acq_rel_t = sem_t<dot_sem::acq_rel>;
using sem_acquire_t = sem_t<dot_sem::acquire>;
using sem_relaxed_t = sem_t<dot_sem::relaxed>;
using sem_release_t = sem_t<dot_sem::release>;
using sem_sc_t      = sem_t<dot_sem::sc>;
using sem_weak_t    = sem_t<dot_sem::weak>;

static constexpr sem_acq_rel_t sem_acq_rel{};
static constexpr sem_acquire_t sem_acquire{};
static constexpr sem_relaxed_t sem_relaxed{};
static constexpr sem_release_t sem_release{};
static constexpr sem_sc_t sem_sc{};
static constexpr sem_weak_t sem_weak{};

template <dot_space __spc>
using space_t         = _CUDA_VSTD::integral_constant<dot_space, __spc>;
using space_global_t  = space_t<dot_space::global>;
using space_shared_t  = space_t<dot_space::shared>;
using space_cluster_t = space_t<dot_space::cluster>;

static constexpr space_global_t space_global{};
static constexpr space_shared_t space_shared{};
static constexpr space_cluster_t space_cluster{};

template <dot_scope __scope>
using scope_t         = _CUDA_VSTD::integral_constant<dot_scope, __scope>;
using scope_cluster_t = scope_t<dot_scope::cluster>;
using scope_cta_t     = scope_t<dot_scope::cta>;
using scope_gpu_t     = scope_t<dot_scope::gpu>;
using scope_sys_t     = scope_t<dot_scope::sys>;

static constexpr scope_cluster_t scope_cluster{};
static constexpr scope_cta_t scope_cta{};
static constexpr scope_gpu_t scope_gpu{};
static constexpr scope_sys_t scope_sys{};

template <dot_op __op>
using op_t        = _CUDA_VSTD::integral_constant<dot_op, __op>;
using op_add_t    = op_t<dot_op::add>;
using op_dec_t    = op_t<dot_op::dec>;
using op_inc_t    = op_t<dot_op::inc>;
using op_max_t    = op_t<dot_op::max>;
using op_min_t    = op_t<dot_op::min>;
using op_and_op_t = op_t<dot_op::and_op>;
using op_or_op_t  = op_t<dot_op::or_op>;
using op_xor_op_t = op_t<dot_op::xor_op>;
using op_cas_t    = op_t<dot_op::cas>;
using op_exch_t   = op_t<dot_op::exch>;

static constexpr op_add_t op_add{};
static constexpr op_dec_t op_dec{};
static constexpr op_inc_t op_inc{};
static constexpr op_max_t op_max{};
static constexpr op_min_t op_min{};
static constexpr op_and_op_t op_and_op{};
static constexpr op_or_op_t op_or_op{};
static constexpr op_xor_op_t op_xor_op{};
static constexpr op_cas_t op_cas{};
static constexpr op_exch_t op_exch{};

template <dot_cta_group __cta_group>
using cta_group_t   = _CUDA_VSTD::integral_constant<dot_cta_group, __cta_group>;
using cta_group_1_t = cta_group_t<dot_cta_group::cta_group_1>;
using cta_group_2_t = cta_group_t<dot_cta_group::cta_group_2>;

static constexpr cta_group_1_t cta_group_1{};
static constexpr cta_group_2_t cta_group_2{};

template <dot_kind __kind>
using kind_t          = _CUDA_VSTD::integral_constant<dot_kind, __kind>;
using kind_f16_t      = kind_t<dot_kind::f16>;
using kind_f8f6f4_t   = kind_t<dot_kind::f8f6f4>;
using kind_i8_t       = kind_t<dot_kind::i8>;
using kind_mxf4_t     = kind_t<dot_kind::mxf4>;
using kind_mxf4nvf4_t = kind_t<dot_kind::mxf4nvf4>;
using kind_mxf8f6f4_t = kind_t<dot_kind::mxf8f6f4>;
using kind_tf32_t     = kind_t<dot_kind::tf32>;

static constexpr kind_f16_t kind_f16{};
static constexpr kind_f8f6f4_t kind_f8f6f4{};
static constexpr kind_i8_t kind_i8{};
static constexpr kind_mxf4_t kind_mxf4{};
static constexpr kind_mxf4nvf4_t kind_mxf4nvf4{};
static constexpr kind_mxf8f6f4_t kind_mxf8f6f4{};
static constexpr kind_tf32_t kind_tf32{};

template <int n>
using n32_t = _CUDA_VSTD::integral_constant<int, n>;

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_DOT_VARIANTS_H_
