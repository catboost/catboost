// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H
#define _LIBCUDACXX___NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp, class _UnaryOp>
_CCCL_API constexpr _OutputIterator transform_exclusive_scan(
  _InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init, _BinaryOp __b, _UnaryOp __u)
{
  if (__first != __last)
  {
    _Tp __saved = __init;
    do
    {
      __init    = __b(__init, __u(*__first));
      *__result = __saved;
      __saved   = __init;
      ++__result;
    } while (++__first != __last);
  }
  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H
