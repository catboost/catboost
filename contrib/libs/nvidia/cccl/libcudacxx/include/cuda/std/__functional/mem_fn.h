// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_MEM_FN_H
#define _LIBCUDACXX___FUNCTIONAL_MEM_FN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/weak_result_type.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
class __mem_fn : public __weak_result_type<_Tp>
{
public:
  // types
  using type = _Tp;

private:
  type __f_;

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __mem_fn(type __f) noexcept
      : __f_(__f)
  {}

  // invoke
  template <class... _ArgTypes>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __invoke_return<type, _ArgTypes...>::type
  operator()(_ArgTypes&&... __args) const
  {
    return _CUDA_VSTD::__invoke(__f_, _CUDA_VSTD::forward<_ArgTypes>(__args)...);
  }
};

template <class _Rp, class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 __mem_fn<_Rp _Tp::*> mem_fn(_Rp _Tp::* __pm) noexcept
{
  return __mem_fn<_Rp _Tp::*>(__pm);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_MEM_FN_H
