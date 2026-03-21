// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/unary_function.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class _Arg, class _Result>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED
pointer_to_unary_function : public __unary_function<_Arg, _Result>
{
  _Result (*__f_)(_Arg);

public:
  _CCCL_API inline explicit pointer_to_unary_function(_Result (*__f)(_Arg))
      : __f_(__f)
  {}
  _CCCL_API inline _Result operator()(_Arg __x) const
  {
    return __f_(__x);
  }
};

template <class _Arg, class _Result>
_LIBCUDACXX_DEPRECATED _CCCL_API inline pointer_to_unary_function<_Arg, _Result> ptr_fun(_Result (*__f)(_Arg))
{
  return pointer_to_unary_function<_Arg, _Result>(__f);
}

_CCCL_SUPPRESS_DEPRECATED_POP

#endif // defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H
