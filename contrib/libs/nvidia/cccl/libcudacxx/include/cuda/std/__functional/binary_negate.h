// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H
#define _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/binary_function.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS)

template <class _Predicate>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED binary_negate
    : public __binary_function<typename _Predicate::first_argument_type, typename _Predicate::second_argument_type, bool>
{
  _Predicate __pred_;

public:
  _CCCL_API explicit constexpr binary_negate(const _Predicate& __pred)
      : __pred_(__pred)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  constexpr _CCCL_API inline bool operator()(const typename _Predicate::first_argument_type& __x,
                                             const typename _Predicate::second_argument_type& __y) const
  {
    return !__pred_(__x, __y);
  }
};

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Predicate>
_LIBCUDACXX_DEPRECATED _CCCL_API constexpr binary_negate<_Predicate> not2(const _Predicate& __pred)
{
  return binary_negate<_Predicate>(__pred);
}
_CCCL_SUPPRESS_DEPRECATED_POP

#endif // _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H
