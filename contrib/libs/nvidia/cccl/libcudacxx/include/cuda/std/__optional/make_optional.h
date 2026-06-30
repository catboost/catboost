//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_MAKE_OPTIONAL_H
#define _LIBCUDACXX___OPTIONAL_MAKE_OPTIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__optional/optional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_TEMPLATE(class _Tp = nullopt_t::__secret_tag, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(is_same, _Tp, nullopt_t::__secret_tag))
_CCCL_API constexpr optional<decay_t<_Up>> make_optional(_Up&& __v)
{
  return optional<decay_t<_Up>>(_CUDA_VSTD::forward<_Up>(__v));
}

_CCCL_TEMPLATE(class _Tp, class... _Args)
_CCCL_REQUIRES((!_CCCL_TRAIT(is_reference, _Tp)))
_CCCL_API constexpr optional<_Tp> make_optional(_Args&&... __args)
{
  return optional<_Tp>(in_place, _CUDA_VSTD::forward<_Args>(__args)...);
}

_CCCL_TEMPLATE(class _Tp, class _Up, class... _Args)
_CCCL_REQUIRES((!_CCCL_TRAIT(is_reference, _Tp)))
_CCCL_API constexpr optional<_Tp> make_optional(initializer_list<_Up> __il, _Args&&... __args)
{
  return optional<_Tp>(in_place, __il, _CUDA_VSTD::forward<_Args>(__args)...);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___OPTIONAL_MAKE_OPTIONAL_H
