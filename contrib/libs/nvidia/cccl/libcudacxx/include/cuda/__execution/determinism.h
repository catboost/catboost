//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_DETERMINISM_H
#define __CUDA___EXECUTION_DETERMINISM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__execution/require.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_one_of.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_EXECUTION

namespace determinism
{

struct __get_determinism_t;

enum class __determinism_t
{
  __not_guaranteed,
  __run_to_run,
  __gpu_to_gpu
};

template <__determinism_t _Guarantee>
struct __determinism_holder_t : __requirement
{
  static constexpr __determinism_t value = _Guarantee;

  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(const __get_determinism_t&) const noexcept
    -> __determinism_holder_t<_Guarantee>
  {
    return *this;
  }
};

using gpu_to_gpu_t     = __determinism_holder_t<__determinism_t::__gpu_to_gpu>;
using run_to_run_t     = __determinism_holder_t<__determinism_t::__run_to_run>;
using not_guaranteed_t = __determinism_holder_t<__determinism_t::__not_guaranteed>;

_CCCL_GLOBAL_CONSTANT gpu_to_gpu_t gpu_to_gpu{};
_CCCL_GLOBAL_CONSTANT run_to_run_t run_to_run{};
_CCCL_GLOBAL_CONSTANT not_guaranteed_t not_guaranteed{};

struct __get_determinism_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(_CUDA_STD_EXEC::__queryable_with<_Env, __get_determinism_t>)
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(_CUDA_STD_EXEC::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_determinism = __get_determinism_t{};

} // namespace determinism

_LIBCUDACXX_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_DETERMINISM_H
